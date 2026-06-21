# Course 19 · vLLM PD 分离（Prefill / Decode Disaggregation）

本目录是一份**自洽的教学材料**，目标：掌握 vLLM **PD 分离**部署的**核心原理** —— 为什么要把
Prefill 和 Decode 拆开、KV cache 如何在 P/D 节点之间交接、远端块与本地块如何对号、
Proxy 如何编排 xPyD —— 并能回答下面的 10 个核心问题（Q1–Q10）。

> 设计哲学同 course16：**四个零依赖（纯 Python 标准库）可运行 demo** 把原理讲透、可验证；
> 真实 GPU / vLLM 源码级细节放在 `DOC.md`（1400+ 行长文，按需查）。

## 目录与学习路径

| 文件 | 覆盖 | 一句话 |
|---|---|---|
| `pd_why_demo.py` | **Q1–Q3** | roofline 玩具模型算出 prefill 算力受限、decode 带宽受限，以及混部为何互相干扰 |
| `kv_handoff_demo.py` | **Q4–Q6** | 手写玩具 Transformer（真实 KV cache），证明 PD 分离的 decode 输出与单机**逐 token 一致** |
| `block_remap_demo.py` | **Q7–Q8** | 分页 KV：P/D 物理块号不同，靠**逻辑序**对号（extract/inject），并解释 `len(prompt)-1` |
| `proxy_xpyd_demo.py` | **Q9–Q10** | 复刻官方 proxy 的 6 步流程 + request_id 编址 + xPyD 轮询路由 |
| `DOC.md` | 源材料 | 源码级长文：架构演进、KVConnector 接口、端到端调用链、KV Offload、P2pNcclConnector |
| `README.md` | 本文 | 学习指南 + 核心架构图（ASCII）+ Q1–Q10 蒸馏答案 |

建议顺序：先按上表从上到下跑 4 个 demo（每个文件顶部都有逐课说明），再对照下面的 Q&A，
最后用 `DOC.md` 深挖任意一个细节。

```bash
python pd_why_demo.py        # 为什么要分
python kv_handoff_demo.py    # KV 怎么交接（核心）
python block_remap_demo.py   # 远端/本地块怎么对号
python proxy_xpyd_demo.py    # 谁来指挥、怎么路由
```

全部零依赖（只用标准库 `math`/`random`/`itertools`），用玩具模型把原理讲透、可运行可验证。

---

## 一图看懂 PD 分离整体架构

```
                        ┌──────────────────────────────┐
        客户端  ───────► │   API Proxy (纯 CPU, 无 GPU)   │  对外是一个普通 OpenAI 接口
        (HTTP)          │  · 路由/选路(轮询)             │
                        │  · 生成内嵌 P/D 地址的 req_id  │
                        └───────┬───────────────┬────────┘
            ① max_tokens=1      │               │  ④ 原始请求(完整 max_tokens)
               发给 P           ▼               ▼
                   ┌─────────────────┐   ┌─────────────────┐
                   │  P 节点 (Prefill)│   │  D 节点 (Decode) │
                   │  高算力 GPU      │   │  大显存/高带宽 GPU│
                   │  (H100/H200)    │   │  (H20/A100)      │
                   │  ② 算 prefill    │   │  ⑤ 注入 KV,      │
                   │     生成 KV     │   │     跳过 prefill,│
                   └────────┬────────┘   │     继续 decode  │
                            │            └────────┬────────┘
                            │  ③ KV cache 直发 D    │  ⑥ token 流式返回
                            └───── NCCL (GPU→GPU) ──┘     给 Proxy → 客户端
                                  点对点, 零拷贝

  控制面 = ZMQ（握手 / 元数据 / 服务发现）   数据面 = NCCL（真正搬 KV cache）
```

**6 步流程**（`proxy_xpyd_demo.py` 完整复刻）：
1. Proxy 把请求复制一份、`max_tokens` 强制改成 **1**，发给 P（只触发 prefill）；
2. P 完成 prefill，生成 KV cache，**主动**把 KV 发给 D；
3. Proxy **丢弃** P 的响应（不要它的 token）；
4. Proxy 把**原始**请求（完整 `max_tokens`）转发给 D；
5. D **直接读取**收到的 KV，跳过 prefill，继续 decode；
6. D 把后续所有 token 返回 Proxy → 客户端。

---

## 三层职责：vLLM 怎么把 KV「拦」下来搬走

PD 分离不改模型、不改注意力，而是通过 **KVConnector** 接口在两个时机「拦截」：

```
  Scheduler 侧 (调度进程)                    Worker 侧 (GPU 进程)
  ─────────────────────────                ─────────────────────────
  get_num_new_matched_tokens  ← 问：有多少    start_load_kv      ← D: forward 前注入 KV
       token 的 KV 来自外部                    save_kv_layer      ← P: 每层算完后发 KV
  update_state_after_alloc    ← 记录块分配    wait_for_save      ← P: 等所有层发完
  build_connector_meta        ← 打包元数据    get_finished       ← 轮询异步传输完成
  request_finished            ← 收尾释放
           │                                          │
           └────────── KVConnectorMetadata ───────────┘
                      (block_ids / req_id 等)
```

- **Scheduler 侧**只碰元数据（block_ids、token 数），决定「搬什么、搬到哪」；
- **Worker 侧**真正搬数据（`extract`/`inject` + NCCL）。
- 同一套接口（`KVConnectorBase_V1`）既支撑 **PD 分离**（跨网络，`P2pNcclConnector`），
  也支撑 **KV Offload**（跨 PCIe 到 CPU，`OffloadingConnector`）—— 底层同源（见 DOC 第四章）。

---

## Q&A 蒸馏（10 问）

### Q1. 为什么要做 PD 分离？（→ `pd_why_demo.py`）
同一个模型，**prefill 算力密集（compute-bound）**、**decode 显存带宽密集（memory-bound）**，
对硬件的最优形态完全相反。一张既算力强又显存大的卡很贵；拆开后 P 用算力卡、D 用大显存卡、
Proxy 零 GPU，各自按负载独立扩容，**资源利用率↑、成本↓、更易扩展**。

### Q2. prefill 和 decode 的瓶颈差异具体体现在哪？
- Prefill：一次吃整段 prompt，算术强度高，瓶颈在 **TFLOPS**。
- Decode：每步只算 1 个 token，但要把整套权重 + 全部历史 KV 读一遍，算术强度极低，
  瓶颈在 **显存容量 + 带宽**。decode 的访存量随上下文长度**线性增长**。

### Q3. 什么时候**不**该用 PD 分离？
模型小、prompt 短、并发低，单卡算力显存都富余时。KV 传输本身有成本（NCCL/RDMA、握手），
只有「省下的重复 prefill + 解耦收益」明显大于「搬运成本」才划算。主场是长 prompt、高并发的在线服务。

### Q4. PD 分离凭什么能成立？（→ `kv_handoff_demo.py`，核心）
因为 **prefill 阶段唯一需要交接给 decode 的中间产物就是 KV cache**（外加最后一个 token）。
只要把 KV 原样搬到 D，decode 对「KV 来自本机还是远端」**完全透明**，输出与单机**逐 token 相同**。
demo 用手写 Transformer 跑「单机」vs「PD 分离」两条路径，断言结果一字不差。

### Q5. KV cache 在 P/D 之间到底传的是什么？
传的是 prefill 算出的、每层每个 token 的 **K 和 V 张量**（不是 attention 权重，也不是隐藏态）。
P 侧 `save_kv_layer → extract_kv_from_layer` 按 block_ids 取出 → NCCL `send_tensor`；
D 侧后台线程 `recv` → `start_load_kv → inject_kv_into_layer` 写进本地 KV pool。

### Q6. 为什么 D 还要重算最后 1 个 token？（`get_num_new_matched_tokens` 的 `-1`）
D 从外部能拿到的是 `len(prompt) - 1` 个 token 的 KV。最后 1 个 prompt token 要在 D 上
**补算一次前向**，才能得到用于预测「第一个生成 token」的隐藏态。代价仅 1 个 token，几乎零浪费。

### Q7. P 和 D 的物理块号不同，KV 怎么对上号？（→ `block_remap_demo.py`）
靠 **逻辑块顺序**，不靠物理块号。P 给请求分配的物理块可能是 `[2,7,5]`，D 是 `[5,9,11]`；
P 的 `extract` 按逻辑序 **gather** 成一段连续 KV（不含任何物理块号），D 的 `inject` 把
「第 i 个逻辑块」**scatter** 到自己的第 i 个物理块。两边逻辑 token 序完全对齐。

### Q8. chunked prefill 下为什么有「部分注入」分支？
P 可能分多个 step 才 prefill 完，每个 step 只产出部分块的 KV，D 收到的块数可能少于已分配的
block_ids 数，所以 `inject` 里有 `block_ids[:num_block]` 只写已到达部分的分支。

### Q9. Proxy 干了什么？为什么 request_id 里要编进地址？（→ `proxy_xpyd_demo.py`）
Proxy 是纯 CPU 总调度，执行上面的 6 步流程。它生成的 `request_id` 内嵌了 P 和 D 的 zmq 地址
（`___prefill_addr_..._decode_addr_..._uuid`）。P 算完 KV 后**自己解析 request_id** 就知道
该把 KV 发给哪个 D —— **去中心化路由**，无需 Proxy 二次通知，天然支持动态扩缩容。

### Q10. xPyD 怎么路由和扩展？
Proxy 维护 P 列表和 D 列表，按 `count % len` **轮询**各选一个 P、一个 D。扩缩容只是改变列表长度；
P↔D 是点对点 NCCL（每对只需 world_size=2 的 NCCL 组），**增删实例无需重启系统**。
KV 传输模式优先 `PUT_ASYNC`（计算/传输重叠），性能 `PUT_ASYNC > GET > PUT`。

---

## 与真实 vLLM 源码的对应关系

| demo 里的简化 | 真实 vLLM | 文件 |
|---|---|---|
| `cache.serialize()` | `save_kv_layer()` → `extract_kv_from_layer()` → `send_tensor()` | `.../v1/p2p/p2p_nccl_connector.py` |
| `cache.deserialize()` | `start_load_kv()` → 后台 `recv` → `inject_kv_into_layer()` | 同上 |
| `extract_kv` / `inject_kv` (gather/scatter) | `layer[:, block_ids, ...]` 读写 | 同上 |
| `get_num_new_matched_tokens` 的 `-1` | `len(prompt_token_ids) - 1 - num_computed_tokens` | 同上 |
| `ApiProxy.handle` 6 步 | `handle_request()` | `examples/.../disagg_proxy_p2p_nccl_xpyd.py` |
| `request_id` 编址 | `___prefill_addr_..._decode_addr_..._{uuid}` | 同上 |
| Scheduler/Worker 双侧接口 | `KVConnectorBase_V1` | `.../v1/base.py` |

部署实战命令（1P1D / 1P3D / 3P1D 启动参数、`kv_buffer_size` 调参、benchmark）见 `DOC.md` 第五章。
