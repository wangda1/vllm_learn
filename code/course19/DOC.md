一 vLLM PD 分离部署概述
1.1 为什么要做 PD（Prefill/Decode）分离部署 LLM 应用
大模型推理本质上由两个串行但资源特征完全不同的阶段组成：
1. Prefill（上下文处理）
  - 输入：整段 prompt（上下文 token 数量大）
  - 主要成本：大量矩阵计算（算力密集），同时会生成并写入 KV Cache
  - 瓶颈更偏向：算力/带宽/并行度
2. Decode（增量生成）
  - 输入：每步（step） 1 个（或少量）新 token
  - 主要成本：每步都要读取历史 KV Cache、追加写入新 KV
  - 瓶颈更偏向：显存容量（KV 越长所需显存越大）+ 显存带宽（频繁读写）
  - 算力需求相对更“细碎”（batch 小、计算不够饱和），但对内存系统要求高
核心矛盾：同一块设备同时兼顾“极强算力”和“超大显存”成本很高。如果把 Prefill + Decode 都放在一台设备上：
  - 你需要既有很强的计算能力（支撑长 prompt 的 prefill 吞吐）
  - 又要有足够大的显存（支撑长上下文、多并发的 KV cache 常驻）
→ 这往往意味着选择高算力且大显存 GPU，芯片的单位成本急剧上升。
从芯片/系统设计角度来看：在固定的芯片面积、功耗、成本约束下（厂商受良率、工艺、功耗、供应链等限制），芯片设计常面临取舍：
  - 更多计算单元（更强算力）通常会挤占芯片面积与功耗预算
  - 留给 内存接口/封装/显存容量 的空间与预算就更紧
因此 Prefill（算力为主）与 Decode（显存/带宽为主）对硬件的最优形态并不一致。既然单设备“又要算力又要显存”短期难以同时做到成本最优，那为什么不把这两个阶段拆开，分别在不同特性的计算设备上分别完成 Prefill 和 Decode 阶段。由此，诞生了 PD 分离部署的思路。
  - Prefill 节点：选择算力强的设备（高 TFLOPS、强并行）
  - Decode 节点：选择显存更大/更适合存 KV 的设备（大显存、高带宽）
PD 分离的技术方案有以下优势：
1. 资源利用率提升：Prefill（算力密集）与 Decode（显存密集）彻底解耦，可独立扩容
2. 成本优化：P 用高端卡，D 用中低端卡，Proxy 零 GPU。（高算力的 GPU 更贵）
3. 更稳定也更好拓展：prefill 与 decode 可以分别按负载独立扩容；decode 的 KV 压力和 prefill 抢同一块显存
什么时候不需要 PD 分离呢：
- 模型较小、prompt 较短、并发不高。
- 单卡的算力与显存都足够，且通信/架构复杂度不值得引入，这类情况一体化部署更简单。
1.2 PD 分离架构概述

```
                  ┌────────────────────────────┐
   客户端 ───────►│  API Proxy (纯 CPU, 无 GPU)  │  对外是普通 OpenAI 接口
   (HTTP)         │  路由 / 选路 / 状态聚合       │
                  └──────┬───────────────┬───────┘
       ① max_tokens=1    │               │  ④ 原始请求(完整 max_tokens)
                         ▼               ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ P 节点 (Prefill)  │  │ D 节点 (Decode)   │
              │ 高算力 GPU        │  │ 大显存/高带宽 GPU  │
              │ ② 算 prefill→KV   │  │ ⑤ 注入 KV,跳过    │
              └────────┬─────────┘  │    prefill,续 decode│
                       │            └────────┬─────────┘
                       │ ③ KV 直发 D          │ ⑥ token 流式返回
                       └── NCCL (GPU→GPU) ────┘    Proxy → 客户端
        控制面 = ZMQ(握手/元数据)     数据面 = NCCL(搬 KV cache)
```
（同款带 6 步编号的大图见 README.md「一图看懂 PD 分离整体架构」）

PD 分离应用的核心组成（三个服务）：
1. API Proxy 服务
  - 流量入口，纯 CPU 部署（无需 GPU）
  - 负责请求路由、状态管理、响应聚合
  - 只需与 P/D 节点保持低延迟网络即可
2. P 节点（Prefill 节点）
  - 专注 Prefill 阶段：处理完整 prompt，生成 KV Cache
  - 推荐高算力 GPU（如 H200、H100）
3. D 节点（Decode 节点）
  - 专注 Decode 阶段：读取 KV Cache，持续生成 token
  - 可使用性价比更高 GPU（如 H20、A100）
部署灵活性：P 节点与 D 节点的推理服务所用代码完全相同（同一镜像、同一代码），仅计算设备不同。API Proxy 独立部署，也是流量的出入口，实现计算资源解耦。PD 分离应用的处理流程可简单总结为 6 步：
1. API Proxy 将请求发给 P 节点（强制 max_tokens=1）
2. P 节点完成 Prefill，生成 KV Cache 并返回第一个 token
3. Proxy 主动丢弃 P 节点的响应
4. Proxy 将原始请求转发给 D 节点
5. D 节点直接读取共享 KV Cache，继续 Decode
6. D 节点将后续所有 token 返回给 Proxy，完成响应
1.3 vLLM 如何部署 PD 分离应用
vLLM PD 分离功能依靠 KV transfer 模块来完成，也能够支持简单 1P1D 场景的运行。其工作关键流程:
1. 顺序执行P和D计算；
2. 用一个 kv transfer 线程交换 kv cache 信息;
3. 通过 api proxy 控制交互过程。
1P1D 场景的 PD 分离应用的软件流程图如下所示:
> _（1P1D 软件流程图：见 README.md「整体架构」时序，及 `proxy_xpyd_demo.py` 复刻的 6 步流程）_
PD 分离离线推理实例
examples/offline_inference/disaggregated-prefill-v1/run.sh 脚本展示了 vLLM 离线模式下的 pd 分离的预填充功能。运行 run.sh之前，请确保你终端当前位于examples/offline_inference/disaggregated-prefill-v1 目录下！！！并修改 prefill_example.py 和 decode_example.py 代码中的 model 为你本地权重路径。
- run.sh 会依次运行 prefill_example.py 和 decode_example.py。
- prefill_example.py - 仅执行预填充操作的脚本，将 KV 状态保存至 local_storage 目录，并将提示词保存至 output.txt。
- decode_example.py - 仅执行解码操作的脚本，从 local_storage 目录加载 KV 状态，并从 output.txt 加载提示词。
代码运行成功后当前目录下会有 out.txt 文件。
> _（运行结果截图：当前目录生成 out.txt，prefill/decode 两阶段吞吐量见下文日志）_
脚本运行成功后的示意图
> _（脚本成功运行示意图，此处略）_
run.sh 脚本的作用，简单来说是它不是普通“跑一次模型推理”的脚本，而是一个两阶段（prefill/decode）以及离线 KV cache 复用 demo。
它的核心目标是：
1. 第一阶段：先对一批 prompt 做推理，并把 prefill 产生的 KV cache 通过 ExampleConnector 保存到外部共享存储
2. 第二阶段：再重新加载这些 prompt，验证能否从外部存储命中 KV cache，而不是重新完整 prefill
  - 对比第二阶段是否出现：
  - External Cache Hit!
  - Inject KV cache ...
  - 更高吞吐 / 更短推理时间
结合运行后日志信息，可知上述目标都已经验证成功。
# prefill 阶段吞吐量
WARNING 03-20 21:40:13 [example_connector.py:167] In connector.start_load_kv, but the attn_metadata is None
Processed prompts: 100%|████████████████████| 4/4 [00:01<00:00,  3.49it/s, est. speed input: 2636.49 toks/s, output: 3.49 toks/s]
# decode 阶段吞吐量
WARNING 03-20 21:40:25 [example_connector.py:167] In connector.start_load_kv, but the attn_metadata is None
Processed prompts: 100%|█████████████████| 4/4 [00:00<00:00, 18.89it/s, est. speed input: 14276.27 toks/s, output: 189.02 toks/s]
PD分离在线服务实例
disaggregated_prefill.sh PD分离脚本步骤解析
vLLM 官方示例（位于 examples/online_serving/disaggregated_prefill.sh），是用于 1P1D 最小在线 PD 分离部署。它通过 两个独立的 vLLM 实例 + 一个轻量 Proxy 实现 Prefill 和 Decode 彻底解耦，KV Cache 通过 P2pNcclConnector进行高速点对点传输。
脚本对应的整体架构拓扑如下（1P + 1D 最小示例）:

```
   client ──► Proxy(:8000) ──①max_tokens=1──► P(:8100, GPU0, kv_producer)
                  │                                    ║ P2pNcclConnector
                  └──④原始请求──► D(:8200, GPU1, ◄═════╝ NCCL 传 KV
                                    kv_consumer) ──⑥token──► client
```

1，设置环境
VLLM_HOST_IP=127.0.0.1（支持多机分布式）
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
2, 安装依赖自动安装 quart（API Proxy 服务必须）
3，启动 P 节点（Prefill Producer）
# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"30001","http_ip":"'"$VLLM_HOST_IP"'","http_port":"8100","send_type":"PUT_ASYNC"}}' &
4，启动 D 节点（Decode Consumer）
# decoding instance, which is the KV consumer  
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"30001","http_ip":"'"$VLLM_HOST_IP"'","http_port":"8200","send_type":"PUT_ASYNC"}}' &
5，等待服务就绪(最多等 20 分钟）)
# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200
6, 启动 API Proxy, 两个 curl 到 http://localhost:8000/v1/completions（max_tokens=10）
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 1
7，测试请求
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')
7，清理。Ctrl+C 或脚本结束时杀掉所有 python 进程
# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
一键部署使用方法（推荐）
我的 A10 机器上跑 vllm0.17.1+ 版本的官方脚本没成功，后面按照自己理解修复了一些 bug 才运行成功，需要替换 p2p_nccl_connector.py文件，然后运行我提供的 disaggregated_prefill.sh
disaggregated_prefill.sh
p2p_nccl_connector.py
# 1. 克隆最新 vLLM（推荐 v0.17.0+）
git clone https://github.com/vllm-project/vllm.git && cd vllm && uv pip install vllm

# 2. 运行官方示例脚本
cd examples/online_serving/disaggregated_prefill/
bash disaggregated_prefill.sh
验证成功标志：
- 看到 🎉🎉 Successfully finished 2 test requests! 🎉🎉
- 输出中两个请求均正常返回完整文本
> _（两个 curl 请求均返回完整文本的输出截图，此处略）_
1.4 disaggregated_prefill.sh 脚本的已知 Bug 与修复
截至 vLLM v0.17.x，官方 disaggregated_prefill.sh 存在两个会导致脚本无法正常跑通的 Bug。以下是详细的根因分析与修复方案。
Bug 1：Producer 端AssertionError —— 非 Chunked Prefill 请求触发断言失败
现象：Producer 引擎在处理第二轮调度时崩溃，抛出：
AssertionError: assert req_id in self.chunked_prefill
根因分析：
在 P2pNcclConnector.build_connector_meta() 的 cached_reqs 循环中，原始代码假设所有出现在 cached_reqs 中的 Producer 端请求都是 Chunked Prefill 的续传分片。但实际上，当一个短 Prompt 在单步内完成整个 Prefill（非 Chunked）后，该请求会作为 cached_req 重新出现（例如进入 Decode 阶段或被 Preemption 恢复），此时它从未被记录到 self.chunked_prefill 字典中，直接访问就会触发断言失败。
# p2p_nccl_connector.py — build_connector_meta() 的 cached_reqs 循环
# 原始代码：
if self.is_producer:
    assert req_id in self.chunked_prefill  # ← 这里崩溃
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    ...
修复：在断言前增加条件检查，跳过非 Chunked Prefill 的请求：
# 修复后：
if self.is_producer:
    if req_id not in self.chunked_prefill:
        # 该请求在单步内完成了全部 Prefill，已经通过 new_req 分支
        # 的 meta.add_request() 发送过了。现在它以 cached_req 身份
        # 重新出现（如进入 decode 或 preemption 恢复），无需再处理。
        continue
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    ...
Bug 2：Consumer 端 KV 传输死锁 ——request_id 随机后缀导致 tensor_id 不匹配
现象：修复 Bug 1 后，脚本不再崩溃，但 Consumer（Decode）引擎永远无法完成 KV 加载，请求挂起不返回，脚本永远看不到成功输出。
原因分析：
这是一个跨越 4 层代码的 request_id 变换链路问题。链路比较长，需要追踪 request_id 从 Proxy 到 P2P Connector 的完整变换过程：
第一步：Proxy 构造原始 request_id，嵌入 P/D 地址信息：
___prefill_addr_localhost:14579___decode_addr_localhost:14580_{UUID}
第二步：OpenAI Completion 端点添加 cmpl- 前缀和循环索引后缀 -{i}：
# completion/serving.py
request_id = f"cmpl-{self._base_request_id(raw_request, ...)}"  # 从 X-Request-Id 读取
request_id_item = f"{request_id}-{i}"  # i=0 → 追加 "-0"
此时 request_id 变为：cmpl-___prefill_addr_..._UUID-0
第三步（关键）：InputProcessor.assign_request_id() 追加 8 位随机十六进制后缀：
# input_processor.py
request.external_req_id = request.request_id  # 保存外部 ID
request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
此时 Producer 的内部 request_id 变为：
cmpl-___prefill_addr_..._UUID-0-a1b2c3d4   （Producer 随机后缀）
Consumer 的内部 request_id 变为：
cmpl-___prefill_addr_..._UUID-0-e5f6g7h8   （Consumer 随机后缀，与 Producer 不同！）
第四步：P2P Connector 使用内部 request_id 构造 tensor_id 进行 KV 传输：
# p2p_nccl_connector.py
# Producer 发送时：
self.p2p_nccl_engine.send_tensor(
    request_id + "#" + layer_name,  # tensor_id = "cmpl-...-0-a1b2c3d4#model.layers.0..."
    kv_cache, remote_address
)

# Consumer 接收时：
self.p2p_nccl_engine.recv_tensor(
    request.request_id + "#" + layer_name,  # tensor_id = "cmpl-...-0-e5f6g7h8#model.layers.0..."
    remote_address
)
由于 Producer 和 Consumer 各自独立生成了不同的随机后缀，tensor_id 永远无法匹配。Consumer 的 recv_tensor() 内部在 self.recv_store_cv.wait() 上无限等待一个永远不会到来的 Tensor，造成死锁。
整个 request_id 变换链路如下所示：
Proxy 原始 ID     ___prefill_addr_...:14579___decode_addr_...:14580_UUID
      │
      ▼  Completion Endpoint（加前缀 + 循环索引）
cmpl-___prefill_addr_...:14579___decode_addr_...:14580_UUID-0
      │
      ├──── Producer InputProcessor（加随机后缀）
      │     cmpl-..._UUID-0-a1b2c3d4
      │            │
      │            ▼  send_tensor("cmpl-..._UUID-0-a1b2c3d4#layer_name")
      │
      └──── Consumer InputProcessor（加不同随机后缀）
            cmpl-..._UUID-0-e5f6g7h8
                   │
                   ▼  recv_tensor("cmpl-..._UUID-0-e5f6g7h8#layer_name")  ← 永远匹配不上！
修复：在 P2pNcclConnector 中新增 _get_kv_request_id() 方法，在构造 tensor_id 前剥离随机后缀，确保 P/D 两端使用一致的稳定标识：
# p2p_nccl_connector.py — 新增方法
@staticmethod
def _get_kv_request_id(request_id: str) -> str:
    """剥离 InputProcessor 追加的随机后缀，获取稳定的 KV 传输标识。"""
    if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
        return request_id
    idx = request_id.rfind("-")
    if idx > 0:
        return request_id[:idx]
    return request_id
然后在 send_tensor 和 recv_tensor 的调用处替换为稳定 ID：
# save_kv_layer() — Producer 发送
kv_request_id = self._get_kv_request_id(request_id)
self.p2p_nccl_engine.send_tensor(kv_request_id + "#" + layer_name, ...)

# start_load_kv() — Consumer 接收
kv_request_id = self._get_kv_request_id(request.request_id)
kv_cache = self.p2p_nccl_engine.recv_tensor(kv_request_id + "#" + layer_name, ...)
补充说明：vLLM 在 envs.py 中提供了 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 环境变量作为临时 workaround（注释写道："Temporary: skip adding random suffix to internal request IDs. May be needed for KV connectors that match request IDs across instances."），但该变量已被标记为 deprecated，未来版本会移除，为了跑下 disaggregated_prefill.sh 开启上述环境变量也可以。我估计后续 vllm 版本从代码层面的修复是更可靠的长期方案，而不是用环境变量让 VLLM 禁用请求 ID 随机化。
Bug 3：脚本清理阶段set -e 导致提前退出
现象：即使 P/D 引擎正确完成了全部请求，脚本仍以非零退出码结束，且不打印 `🎉🎉 Successfully finished 2 test requests! 🎉🎉`。
根因：脚本开头 set -xe 开启了"任何命令失败即退出"模式。清理阶段的 pgrep python | xargs kill -9 会尝试杀死系统中所有 Python 进程（包括其他用户的），对无权操作的进程返回 Operation not permitted（退出码 123），触发 set -e 使脚本立即终止，跳过了后续的结果打印。
修复：对清理命令添加错误抑制：
# 修复前：
pgrep python | xargs kill -9
pkill -f python

# 修复后：
pgrep python | xargs kill -9 2>/dev/null || true
pkill -f python 2>/dev/null || true
代码层面修复总结
Bug
层级
现象
根因
修复文件
Bug 1
Scheduler → Connector
Producer AssertionError 崩溃
非 Chunked Prefill 请求误触 chunked_prefill 断言
p2p_nccl_connector.py
Bug 2
InputProcessor → Connector
Consumer KV 传输死锁
request_id 随机后缀导致 P/D 两端 tensor_id 不匹配
p2p_nccl_connector.py
Bug 3
Shell 脚本
无成功输出、非零退出码
set -e + kill -9 权限错误
disaggregated_prefill.sh
二 vLLM PD 分离的设计方案
vLLM 的 PD 分离功能主要通过 KV Transfer 模块实现。在早期版本中，该功能仅支持简单的 1P1D（一个 Prefill 节点 + 一个 Decode 节点）场景。其核心运行机制为：Prefill（P）和 Decode（D）节点顺序执行计算，由专属的 KV Transfer 线程负责异步传输 KV Cache，并通过 API Proxy 节点协调整体的交互流程。
本质上，vLLM 的 PD 分离方案本质是一个生产者-消费者模型：
1. Prefill 作为生产者 (Producer)：在完成 Prompt 计算后，以非阻塞 (Non-blocking) 方式将生成的 KV Cache 写入缓冲区 (Buffer)。
2. Decode 作为消费者 (Consumer)：以阻塞 (Blocking) 或半阻塞方式从缓冲区获取所需的 KV Cache 以继续生成 Token。
3. KV Cache 数据承载与传输机制：
- 本地场景下，通过双端队列（如 LookupBuffer）进行高效数据承载与传递；
- 跨节点分布式场景下，则通过 Pipe（管道） 进行传输，底层可依托 P2pNcclConnector（P2P 直连）或 MooncakeConnector等分布式存储/传输机制。
vLLM 对分布式的 KV Transfer 进行了极具扩展性的三层抽象设计，目前最成熟且性能最佳的实现是基于 P2P 直连（ZMQ + NCCL） 方案：
1. 连接器层 (Connector Layer)：直接挂载在 vLLM 的模型执行引擎（Worker）上，典型实现为 P2pNcclConnector。它负责拦截 Attention 层的前向传播 (Forward)，剥离逻辑层面的 Block ID，并进行物理显存 Tensor 的精准切片与注入。
2. 通信引擎层 (Engine Layer)：实现了控制面与数据面的彻底分离。
  - 控制面 (Control Plane)：使用 ZeroMQ (ZMQ) 发送轻量级的握手信号与元数据（例如："准备发送 Request X 的 KV cache，tensor 形状举例是 [2, 32, 128]"）。
  - 数据面 (Data Plane)：使用底层 NCCL 的 ncclSend/ncclRecv 接口建立高速数据通道，实现跨节点 GPU-to-GPU 的高速显存零拷贝 (Zero-Copy) 直连传输。
3. 弹性缓冲池层 (Memory Pool Layer)：当网络传输速度不匹配，或者 Decode 端显存不足/尚未准备好接收时，P 端多余的 Tensor 会被暂时卸载到由 TensorMemoryPool 管理的主机锁页内存 (Pinned Host Memory) 中。这有效防止了发送端 GPU 因积压请求而导致显存溢出 (OOM)。
2.1 KV Cache 传输架构演进与 vLLM 现状
2.1.1 业界主流 KV Cache 传输方案对比
KV Cache 传输模块是现代大模型分布式推理框架的核心组件。纵观目前的业界方案，KV Cache 的传输架构主要分为中心化存储和分布式 P2P 传输两种流派（或二者的混合）。
1. 中心化存储 (Centralized Storage) 架构：
指构建一个跨设备、跨节点共享的 KV Store 存储集群（如 Redis 内存网格或专属的分布式显存池）。它统一负责 KV 数据的存储、路由、生命周期管理与垃圾回收。推理实例（无论是 Prefill 还是 Decode）退化为无状态的客户端，只需向中心存储发起 Read/Write 请求即可。
  - 优势：状态集中，易于实现全局调度、容错和多路复用（Prefix Caching）。
  - 代表方案：Mooncake, Dynamo 等。
2. 分布式 P2P 传输 (Decentralized P2P) 架构：
采用点对点 (Peer-to-Peer) 方式在推理实例之间直接交换 KV 数据。各个实例拥有独立的显存并自行管理本地的 Block Table。例如，某个 Prefill 实例完成计算后，由 Proxy 调度器指定目标，直接与 Decode 实例建立 NCCL 通信通道并推送 KV 数据。
  - 优势：没有中心节点的网络瓶颈，数据传输路径最短，延迟最低。
  - 代表方案：vLLM 的 P2pNcclConnector。
> _（KV 传输方案演进图：P2P 直连 → 池化/外部缓存；类层次见下文「三层抽象架构」ASCII 图）_
图：中心化存储 vs P2P 直连架构对比
目前，像 Mooncake 等面向超大规模集群的解决方案更倾向于中心存储方式，将数据流转的复杂度交给底层存储层解决；而 vLLM 考虑到极低延迟的追求，目前主推 P2P NCCL 直连方案。
> _（超大规模 xPyD 趋势示意，此处略；当前 vLLM 主线为 P2pNcclConnector）_
图：Mooncake 中心化 KV Cache 存储架构示意
2.1.2 vLLM V1 架构中的 KV Transfer 模块
在 vLLM V1 架构演进中，P (Prefill) 和 D (Decode) 的角色界限逐渐变得灵活。一个实例既可以是生产者也可以是消费者；特别是在开启 Prefix Caching（前缀缓存）特性后，Prefill 实例甚至可以从其他 Decode 实例中反向拉取已计算好的公共 Prompt KV 块。
vLLM V1 的 KV Transfer 架构如下图所示。其最大的设计亮点在于：Connector 组件被精巧地拆分为两个执行角色 (Role)，分别侵入到 Scheduler（调度器线程）和 Worker（GPU 执行线程）中。两者之间通过强类型的元数据 (KVConnectorMetadata) 进行桥接通信。

```
   一个 vLLM 实例
   ┌──────────────────────────┐        ┌──────────────────────────┐
   │ Scheduler 进程            │        │ Worker 进程 (GPU)         │
   │  Connector(role=SCHEDULER)│        │  Connector(role=WORKER)   │
   │   get_num_new_matched_... │        │   start_load_kv (D 注入)  │
   │   update_state_after_alloc│        │   save_kv_layer (P 发送)  │
   │   build_connector_meta ───┼──┐     │   wait_for_save/get_finished│
   └──────────────────────────┘  │     └──────────────▲───────────┘
                                  │ KVConnectorMetadata │
                                  └─────(随 SchedulerOutput 下发)┘
   设计亮点：同一 Connector 拆成 SCHEDULER/WORKER 两个 role，
            前者只管元数据(搬什么/搬哪)，后者真正搬数据(extract/inject+NCCL)
```

图：vLLM V1 KV Transfer 组件架构设计
为什么调度器 (Scheduler) 需要感知 KV Transfer？
在 PD 分离场景下，"远端的 KV Cache" 在逻辑上等价于 "本地已经计算好的 Block"。因此，调度器必须具备远端感知能力：
1. 它需要决定：本地是否还需要重新计算某个 KV Block，还是只需让 Worker 从远端拉取？
2. 如果远端 KV 尚未传输完毕，它需要将当前 Request 置于等待状态，避免触发本地计算。
3. 反之，Worker 在后台异步拉取完远端 KV  Block 后，必须通知调度器更新本地的逻辑 Block Table，以便将其纳入后续的调度池。
为此，基类 KVConnectorBase_V1 定义了两套核心接口。
1. Scheduler 端接口（用于指挥调度）
在 scheduler.schedule() 核心循环中被调用，主要用于影响 Token 的计算规划与显存块分配：
  - get_num_new_matched_tokens()：查询远端已计算好的、可复用的 KV Cache Token 数量（用于跳过本地前向传播）。
  - update_state_after_alloc()：在调度器为请求分配物理 Block 后调用，同步更新 Connector 的状态机。
  - build_connector_meta()：构建本轮 Step 的元数据包，精准告诉 Worker："这几个请求的这几块 KV，你去远端拉取 / 推送到远端"。
2. Worker 端接口（用于物理执行）
主要在 GPU 模型前向传播 (execute_model) 期间被调用，直接操作底层的显存 Tensor，支持按 Transformer 层 (Layer-by-Layer) 进行细粒度的异步传输：
  - bind_connector_metadata()：接收并解析来自 Scheduler 的元数据指令。
  - 消费者 (Decode) 端接口：
    - start_load_kv()：触发非阻塞的异步 KV 拉取（从网络读取到 GPU 显存）。
    - wait_for_layer_load()：同步原语。阻塞执行流，直到当前 Transformer 层的 KV 数据确实到达并落盘到 Paged Buffer 中。
  - 生产者 (Prefill) 端接口：
    - save_kv_layer()：将当前计算完的 Transformer 层的 KV 数据切片，并推入异步发送队列。
    - wait_for_save()：同步原语。在整个 Step 结束时调用，确保所有后台发送任务均已清空。
2.2 PD 分离架构框图
以下是 vLLM PD 分离在节点之间的物理传输简单架构图：

```
  P 节点 GPU 显存                              D 节点 GPU 显存
  ┌───────────────┐    ① ZMQ 控制消息          ┌───────────────┐
  │ paged KV pool │ ───(req_id/shape/dtype)──► │ paged KV pool │
  │  [blk2][blk7] │                            │  [blk5][blk9] │
  │  [blk5] ...   │ ═══② NCCL 数据(GPU→GPU)═══► │  [blk11]...   │
  └───────────────┘    零拷贝 DMA, 绕过 CPU     └───────────────┘
       ▲                                              │ 缓冲区满时
       └ extract_kv (gather by block_ids)             ▼ 溢出
                                              TensorMemoryPool (CPU Pinned)
```

从API Proxy + P节点 + D节点的视角来构建 vLLM v0.17.0+ 的PD 分离完整架构如下图所示：

```
                    ┌─────────── API Proxy ───────────┐
   client ────────► │  LB_P(轮询选P)   LB_D(轮询选D)   │
                    └──────┬─────────────────┬─────────┘
            max_tokens=1   │                 │  原始请求
                           ▼                 ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │ P: Scheduler         │     │ D: Scheduler         │
        │   └ KVConnector(sched)│     │   └ KVConnector(sched)│
        │ P: Worker            │     │ D: Worker            │
        │   └ save_kv_layer ───┼──┐  │   └ start_load_kv    │
        └─────────────────────┘  │  └─────────▲───────────┘
            控制面 ZMQ ───────────┼────────────┘
            数据面 NCCL/NIXL ═════┴═══► (GPU 显存直传 KV)
```
架构框图简单解析：
1. API Proxy 层（流量调度与状态管理）：
Proxy 是整个 PD 分离架构的大脑，它并不参与实际的矩阵运算和显存管理，而是负责路由分发。
  - 请求拆分：当收到用户的完整请求时，Proxy 会首先将请求派发给 P Node 负载均衡器（LB_P）。此时设定 max_tokens=1（或由 P 节点默认行为决定），让 P 节点只负责跑完 Prompt 的 Prefill 阶段。
  - 状态交接：P 节点计算完毕后，生成了 KV Cache 并开始向目标 D 节点异步传输。Proxy 在收到 P 节点返回的首个 Token（TTFT 达成）后，将该请求的上下文状态和目标分配给 D Node 负载均衡器（LB_D）。
  - 无缝流式响应：对 Client 而言，Proxy 隐藏了底层的 P/D 切换，Client 看到的是一个持续不断的 SSE（Server-Sent Events）数据流。
2. KV 传输层（解耦与高速通道）：
  - 控制面（Control Plane）：基于 ZMQ（ZeroMQ）实现，主要传输轻量级的元数据（Metadata），包括 request_id、layer_id、src_block_ids 以及 Tensor 的 shape 和 dtype。ZMQ 保证了消息的可靠送达和握手。
  - 数据面（Data Plane）：基于 NCCL（针对单机多卡/多机直连）或 NIXL（GPU-Direct RDMA 技术）实现。数据面绕过 CPU（Zero-Copy），直接在 P 节点的 GPU 显存和 D 节点的 GPU 显存之间进行 DMA 搬运。
核心代码分析（基于 vLLM 0.17.0）：
在 Proxy 层，调度逻辑通常实现在类似 pd_router.py 或特定的 FastAPI Router 中。P 节点和 D 节点的连接建立依赖于 vLLM Engine 初始化时传入的 --kv-transfer-config。
# Proxy 层的简化逻辑抽象：
async def generate_response(request: Request):
    # 1. 路由给 P 节点 (Prefill)
    p_node = lb_p.get_best_node()
    prefill_response = await p_node.generate(request, max_tokens=1)
    
    # 获取 request_id 和分配给该请求的 D 节点
    req_id = prefill_response.request_id
    d_node = lb_d.get_best_node()
    
    # 2. 路由给 D 节点 (Decode)
    # 告知 D 节点该请求的 KV Cache 即将通过底层网络到达
    async for token in d_node.generate_stream(request_id=req_id):
        yield token
在 vLLM 的 AsyncLLMEngine 中，角色的区分是通过 engine_config.kv_transfer_config 中的 kv_role 参数决定的（kv_producer 或 kv_consumer），引擎据此初始化不同的 Pipeline。
2.3 开启 pd 分离优化时的 P/D 节点内部架构
开启 pd 分离优化时的 P 节点内部架构

```
  P 节点 (kv_role=kv_producer)
  ┌──────────────────────────────────────────────────┐
  │ Prefill Scheduler  (prefill-only, Chunked Prefill) │
  ├──────────────────────────────────────────────────┤
  │ Model Forward:  layer L 算完 K,V                    │
  │     │  写入本地 KV Pool(仅中转站)                    │
  │     ▼                                              │
  │ KVSender (Connector.save_kv_layer)                 │
  │   extract: kv_layer[:, block_ids, ...]            │
  │   → engine.send_tensor(req_id#layer)  ══ NCCL ══►  │ 发往 D
  │   逐层发送: 算 L+1 层时网卡正发 L 层 → 计算/通信重叠  │
  └──────────────────────────────────────────────────┘
```

架构简单解析：
P 节点（Prefill Node）的核心使命是最快速度吞吐海量 Prompt Token 并生成 KV Cache。
1. 调度层（Prefill Scheduler）：P 节点的调度器通常配置为 prefill-only（在 v0.17.0 中通过策略参数或角色判定）。它采用 Chunked Prefill 机制，将超长的 Prompt 切分为多个 Chunk，以最大化 GPU 计算单元的利用率。
2. 执行层与 KV 生成（Attention & KV Pool）：
在模型前向传播（Forward Pass）时，QKV Projection 会生成当前层的 Key 和 Value。在正常的 LLM 中，这些数据被写入本地 KV Pool 供后续 Decode 使用。但在 P 节点中，本地的 KV Pool 只是一个中转站。
3. KV 传输层（KVSender）：
  - 拦截与提取：连接器（如 NixlConnector 或 P2pNcclConnector）在每一层 Attention 计算完成后，拦截执行流，利用高级索引将属于当前 Request 的 KV Cache 从全局显存池中 extract（提取）出来。
  - 异步非阻塞发送：提取后，数据被推入发送队列，底层的 RDMA Write / NCCL Send 被触发。由于是逐层发送，GPU 计算第 L+1 层时，网卡正在发送第 L 层的 KV，实现了计算与通信的完美重叠。
核心代码分析（基于 vLLM 0.17.0）：
在 vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py 等连接器实现中，P 节点的行为主要体现在 save_kv_layer 方法上。
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py (简化抽象)
def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: AttentionMetadata):
    if not self.is_producer:
        return # 仅 P 节点执行
        
    for request in attn_metadata.requests:
        # 1. 解析目标 D 节点的地址
        remote_ip, remote_port = self.parse_target(request.request_id)
        
        # 2. 从本地 KV Pool 提取当前请求刚刚计算出的 KV Cache 切片
        # kv_layer 形状如 [2, num_blocks, block_size, num_heads, head_size]
        kv_cache_slice = kv_layer[:, request.block_ids, ...]
        
        # 3. 交给底层 Engine 异步发送
        # tensor_id 绑定了 req_id 和 layer_name，确保接收端逐层对齐
        tensor_id = f"{request.request_id}#{layer_name}"
        self.engine.send_tensor(tensor_id, kv_cache_slice, dest=f"{remote_ip}:{remote_port}")
通过上述代码，P 节点在计算每一层时都悄无声息地将 KV Cache 分流到了网络侧，计算流程本身毫无感知。
开启 pd 分离优化时的 D 节点内部架构

```
  D 节点 (kv_role=kv_consumer)
  ┌──────────────────────────────────────────────────┐
  │ 后台接收线程: NCCL recv → recv_store(GPU)           │
  │              满则溢出 → TensorMemoryPool(CPU Pinned)│
  ├──────────────────────────────────────────────────┤
  │ Decode Scheduler                                   │
  │   get_num_new_matched_tokens = len(prompt)-1       │
  │   BlockManager 分配本地 dst_block_ids              │
  ├──────────────────────────────────────────────────┤
  │ KVReceiver (Connector.start_load_kv) ← forward 前   │
  │   inject: layer[:, dst_block_ids, ...] = 收到的 KV  │
  │     ▼                                              │
  │ Model Forward → 自回归 decode（对 KV 来源无感知）    │
  └──────────────────────────────────────────────────┘
```

架构简单解析：
D 节点（Decode Node）的使命是极致的显存管理与高并发自回归生成。
1. 入口与接收（KVReceiver）：
当 D 节点收到 Proxy 派发的 Decode 请求时，它首先要在本地的 BlockManager 中为该请求申请足够的空闲物理 Block（物理槽位）。此时本地是空的，它必须等待 P 节点传来的数据。
2. 重映射与注入（Block Remapping & Injection）：
P 节点发来的 KV 块携带的是 P 节点的物理/逻辑 ID（src_block_id）。D 节点接收到底层 DMA 传来的 Tensor 后，根据事先协商好的映射表，将其原封不动地切片注入到刚才申请的本地 dst_block_id 对应的显存槽位中。这个过程（KV Injection）发生在前向传播开始之前。
3. 执行与输出（PagedAttention & Streaming）：
一旦第一层（或全部层）的 KV 数据准备就绪，请求就会被放行进入 Running Queue。ModelRunner 执行 Decode 阶段的 Forward，此时底层的 FlashInfer / PagedAttention 算子会像往常一样读取本地 KV Pool 中的历史数据，丝毫不知道这些数据是从网络另一端"瞬移"过来的。
核心代码分析（基于 vLLM 0.17.0）：
D 节点的关键在于 start_load_kv 函数，它在模型开始执行前被调用，负责阻塞等待并注入 KV。
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py (简化抽象)
def start_load_kv(self, forward_context: ForwardContext):
    if self.is_producer:
        return # 仅 D 节点执行
        
    for request in forward_context.requests:
        remote_address = self.parse_source(request.request_id)
        
        # 遍历所有层，接收并注入
        for layer_name, layer_obj in forward_context.layers.items():
            # 获取当前层的全局大 KV Cache 池的引用
            global_layer_tensor = layer_obj.kv_cache
            
            # 1. 阻塞接收：等待底层引擎收齐对应层级和 req_id 的 Tensor
            tensor_id = f"{request.request_id}#{layer_name}"
            received_kv_cache = self.engine.recv_tensor(tensor_id, remote_address)
            
            # 2. 黑魔法：显存注入 (KV Injection)
            # request.block_ids 是 D 节点本地刚分配的空闲块的 ID 列表
            # 利用 PyTorch 的高级索引，直接将连续的收到的 KV Tensor 塞入离散的物理槽位
            global_layer_tensor[:, request.block_ids, ...] = received_kv_cache
通过预先准备 RDMA MR（Memory Region），这步操作甚至可以将网络数据直接 DMA 写到 global_layer_tensor 所属的 GPU 显存地址上，实现极致的 Zero-Copy 性能。
2.4 KV Transfer 架构及类解析
KV Transfer 模块是 PD 分离架构的核心桥梁，解决以下关键问题：
核心问题：
  P Node 完成 prefill 后，产生的 KV Cache 存储在 P Node 的 GPU 显存中
  D Node 需要这份 KV Cache 才能开始 decode
  
  挑战1: 数据量大   → Llama3-70B 单请求 KV 可达 5-10GB
  挑战2: 延迟敏感   → KV 传输延迟直接叠加到 TTFT
  挑战3: 跨机传输   → P/D Node 通常在不同物理机器
  挑战4: 块映射     → P/D Node 各自独立管理 KV 块编号
  挑战5: 并发管理   → 多请求同时传输，需要流水线化

解决方案：
  NIXL (NVIDIA Inference Xfer Library)
    → GPU-Direct RDMA 零拷贝跨机传输
    → 绕过 CPU 和系统内存，直接 GPU→GPU
    → 配合 InfiniBand 网络，带宽可达 400Gbps
KV Transfer 数据量简单估算：
KV Cache 大小公式：
  size = 2 × num_layers × num_heads × head_dim × seq_len × batch × dtype_bytes

以 Llama3-70B 为例：
  num_layers  = 80
  num_kv_heads = 8   (GQA)
  head_dim    = 128
  seq_len     = 4096
  batch       = 1
  dtype       = bf16 (2 bytes)

  size = 2 × 80 × 8 × 128 × 4096 × 1 × 2
       = 2 × 80 × 8 × 128 × 4096 × 2
       ≈ 5.37 GB / request

传输时间估算 (400Gbps IB + RDMA)：
  5.37GB × 8 = 42.96 Gbits
  42.96 / 400 ≈ 107ms  ← 这是 TTFT 的额外开销
  
优化手段：
  1. 流水线传输：prefill 每层计算完立即传输该层 KV
  2. 压缩传输：KV 量化到 fp8/int8
  3. 前缀复用：prefix cache 命中则跳过传输
vLLM 对分布式的 KV Transfer 模块进行了极具扩展性的三层抽象设计。无论是点对点直连（P2P NCCL），还是借助外部缓存组件（如 LMCache），都统一在这一套架构下:

```
  ┌──────────────────────────────────────────────────────┐
  │ Connector 层  KVConnectorBase_V1                       │  ← 业务逻辑:
  │   P2pNcclConnector / NixlConnector / OffloadingConnector│    切 KV、管 block_id
  ├──────────────────────────────────────────────────────┤
  │ Engine 层     P2pNcclEngine                            │  ← 分布式通信:
  │   控制面 ZMQ(元数据握手)  +  数据面 NCCL(Tensor 直传)   │    建链、收发张量
  ├──────────────────────────────────────────────────────┤
  │ 内存层        TensorMemoryPool (CPU Pinned, 伙伴分配)   │  ← 兜底防 OOM:
  │   GPU 缓冲区满时把待传/已收的 KV 溢出到主机锁页内存      │    溢出暂存
  └──────────────────────────────────────────────────────┘
```

KV Transfer 的三层抽象架构简单解读：
1. 控制面与数据面分离（Engine 层）：P2pNcclEngine 使用 ZMQ 发送微小的控制消息（"我要传一个 Tensor，ID 是 XXX"），然后利用 NCCL 建立高速数据通道进行真正的 Tensor 搬运。
2. 连接器（Connector 层）：P2pNcclConnector 挂载在 vLLM 的执行引擎上。它负责拦截 Attention 层的输入输出，把逻辑上的 Block ID 和真实的物理 Tensor 关联起来。
3. 后备存储（Memory Pool）：当网络传输速度不匹配，或者 Decode 节点还没准备好接收时，多余的 Tensor 会被暂存到由 TensorMemoryPool 管理的主机钉扎内存（Pinned Host Memory）中，防止 OOM。
vLLM 将连接器的基类定义为 KVConnectorBase_V1，它同时承担了 Scheduler 端（分配 Token 和 Block） 和 Worker 端（拦截计算图） 的双重职责。 KVConnectorBase_V1 家族及其在 P2P 直连模式下的核心类 UML 类图如下：

```
        ┌─────────────────────────────┐
        │   KVConnectorBase_V1 (ABC)   │
        │  Scheduler 侧:               │
        │   get_num_new_matched_tokens │
        │   update_state_after_alloc   │
        │   build_connector_meta       │
        │   request_finished           │
        │  Worker 侧:                  │
        │   start_load_kv / save_kv_layer│
        │   wait_for_save / get_finished│
        └──────────────▲──────────────┘
                       │ 继承
        ┌──────────────┴──────────────┐
        │      P2pNcclConnector        │  持有 ↓
        │  + extract_kv_from_layer     │   ┌────────────────┐
        │  + inject_kv_into_layer      │──►│ P2pNcclEngine  │
        │  + parse_request_id          │   │ ZMQ + NCCL     │──► TensorMemoryPool
        └──────────────────────────────┘   └────────────────┘
        元数据: P2pNcclConnectorMetadata ─┬─ ReqMeta(request_id, block_ids)
```

KV Block 传输状态机

```
  [P] KVGenerated ──► MemoryRegistered ──► TransferInFlight ═══(NCCL)═══►
                                              │ D 拥塞/缓冲满
                                              ▼
                                       溢出到 TensorMemoryPool(CPU)
  [D] Received ──► BlockRemapped(逻辑序→本地 block_id) ──► Decoding ──► Finished
                                                          (EOS 后清理本地块+残留元数据)
```

状态流转详细解析：
KV Block 的生命周期跨越了三个主要物理节点（P节点、传输层、D节点），其状态流转必须精密控制以避免死锁或显存泄漏（OOM）。
1. 生成与锁定（KVGenerated -> MemoryRegistered）：
   KV Cache 在 P 节点刚生成时位于 GPU 显存。如果要使用 NIXL (RDMA) 传输，显存页必须被锁定并注册为 Memory Region (MR)。由于每次动态注册开销大，vLLM 通常在启动时预分配注册好全局显存池，这里只需传递偏移量（Offset）。
2. 主机显存池兜底（TensorMemoryPool 介入）：
   图中隐藏了一个关键错误处理状态：如果 TransferInFlight 因为 D 节点拥塞而长期阻塞，P 节点的 GPU 显存会被迅速耗尽。此时 TensorMemoryPool 会介入，将积压的 Tensor 从 GPU 下载到 CPU Pinned Memory 中，释放 GPU 显存。
3. 映射与销毁（BlockRemapped -> Decoding -> Finished）：
   D 节点维护了 req_id 到本地 block_ids 的映射。当 Decode 完毕（遇到 EOS）时，不仅要清理 D 节点本地的物理 Blocks，Proxy 还会发送清理信号，确保任何滞留在传输队列中的残留 Metadata 或 CPU Buffer 均被彻底销毁。
三 PD 分离端到端流程解析
3.1 PD 分离执行链路概述
PD 分离的一个技术难点在于层级流水线（Layer-by-layer Pipeline）。数据发送不能等整个模型跑完再做，而是计算与通信深度重叠。下面我们从一个 HTTP 请求进入 API Proxy 算起，梳理一条完整的 PD 分离执行链路。
阶段 1：Proxy 网关层面的请求派发
API Proxy 的核心作用是“欺骗”集群。它把用户发来的一个完整请求强行拆开，让 P 节点只算 Prompt，让 D 节点负责后续所有 Token 的生成。
下面是官方最小化 xPyD 代理服务 disagg_proxy_p2p_nccl_xpyd.py 中的核心路由代码：
# vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py

@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    original_request_data = await request.get_json()

    # 1. 克隆请求，并偷偷篡改 max_tokens 强制设为 1
    # 这确保 P 节点只跑一轮完整的 Prefill 算完 Prompt，生成首个 Token 就结束
    prefill_request = original_request_data.copy()
    prefill_request["max_tokens"] = 1
    
    # ... 省略负载均衡逻辑，获取 P/D 节点的 zmq_addr (P2P通信地址) ...
    
    # 2. 魔改 request_id，将通信地址硬编码在 ID 中！
    # 这样底层网络引擎只需解析 Request ID 就能知道把 KV Cache 发给谁 / 找谁要
    request_id = (
        f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
        f"{decode_zmq_addr}_{random_uuid()}"
    )

    # 3. 将被篡改过的请求发给 P 节点 (Prefill)
    # 并且故意丢弃 (continue) 它的任何返回，纯粹只为了让它产生并发送 KV Cache
    async for _ in forward_request(
        f"http://{prefill_addr}{request.path}", prefill_request, request_id
    ):
        continue

    # 4. P 节点跑完后，将【原始包含完整 max_tokens 的请求】原封不动发给 D 节点
    # D 节点收到时，P 节点已经把 KV Cache 发过去了，D 节点直接从 Decode 开始
    generator = forward_request(
        f"http://{decode_addr}{request.path}", original_request_data, request_id
    )
    return await make_response(generator)
阶段 2：P 节点 (Producer) 的预填充与发车
当调度器 (Scheduler) 开始执行这个 Request 时：
1. ModelRunner 启动：进行 Prompt 的前向计算（Forward Pass）。
2. 逐层 Attention 计算：
  - 计算到第 0 层时，生成了第 0 层的 KV Cache。此时，计算流程被拦截！
3. Kv Connector 提取并发送 kv cache 数据 (对应save_kv_layer)：
  - save_kv_layer 被调用。它根据当前请求分配的本地物理块 ID (block_ids)，从海量的 KV Pool 中把仅仅属于这个请求的散落的数据，提取（Extract）成一块连续的 Tensor。
  - 这块连续的 Tensor 被加上了一个快递单号，形如 req_42#layer_0，然后通过 send_tensor 函数推入异步发送队列。
  - 此时 GPU 甚至已经开始做第 1 层的推理！ 这正是层级通信-计算流水线的魅力：当第 1 层在做计算密集的 GEMM 乘法时，第 0 层的 KV 数据已经通过 ZMQ/NCCL 在光缆中飞向 D 节点了（PUT_ASYNC 异步模式）。
4. P 节点收尾：跑完全部层，产出首个 Token 返回给 Proxy，Proxy 收到后，立刻丢给对应的 D 节点。
阶段 3：D 节点 (Consumer) 的接收与解码
D 节点的 AsyncLLMEngine 在收到请求时，Prompt Token 和第一个生成的 Token 都在里面。
1. 调度器分配物理槽位：
  - Scheduler 执行时调用 Connector 的 get_num_new_matched_tokens。Connector 告诉它：“这个 Prompt 已经在远端算过了，别再算了！”
  - Scheduler 听懂了，于是直接为这些远端 Token 在 D 节点本地申请了一批全新的、空闲的物理块（准备用来接客）。
2. 模型 forward 执行前夕完成显存注入 (start_load_kv)：
  - D 节点的 Worker 准备做 Decode 的 Forward 之前，必须先拿到 KV Cache。所以它调用 start_load_kv，在这里阻塞等待。
3. 自回归生成：
  - 在显存注入完毕后！模型继续向下执行标准的前向推理（Decode）。
  - 底层的 PagedAttention 算子会去读本地显存池，此时它会惊奇地发现：“咦？怎么里面已经有以前历史上下文的数据了？”虽然它根本不知道这是网络凭空送过来的。但是，它还是愉快地生成了后续的 Token，最终流式返回给用户。
3.2 PD 分离端到端流程解析
以 P2P NCCL Connector（最典型的实现）为主线，结合代码逐层分析。
第一层：LLM 请求入口
文件：vllm/v1/engine/async_llm.py
用户调用 AsyncLLM.generate() 或 add_request()，触发：
# vllm/v1/engine/async_llm.py — generate() / add_request()
# 对 prompt tokenize，构建 EngineCoreRequest（含 token_ids、采样参数、request_id）
request = self.input_processor.process_inputs(
    request_id, prompt, params,
    arrival_time=arrival_time,
    ...   # lora_request, priority, data_parallel_rank 等可选参数
)
process_inputs 对 prompt 进行 tokenize，构建出一个 EngineCoreRequest 对象（包含 token ids、采样参数、request_id 等）。然后通过 _add_request 完成双路分发：
# vllm/v1/engine/async_llm.py
# ① 当前进程：注册到 OutputProcessor，负责 token→文本流还原
self.output_processor.add_request(request, prompt, parent_req, index, queue)
# ② 跨进程（ZMQ）：发送到独立 EngineCore 进程执行推理
await self.engine_core.add_request_async(request)
- output_processor：负责把 EngineCore 的 token 输出还原成文本流，运行在当前进程。
- engine_core.add_request_async：把 request 通过 ZMQ socket 发送到独立的 EngineCore 进程。
第二层：EngineCore 的 step() 循环
文件：vllm/v1/engine/core.py
EngineCore 是一个独立进程，在不断执行 step() 循环：
# vllm/v1/engine/core.py — EngineCore.step()（主推理循环）
def step(self):
    # ① 调度：决定本轮 request，构建 SchedulerOutput（含 KV block 分配 + connector_metadata）
    scheduler_output = self.scheduler.schedule()
    # ② 异步提交 GPU 执行（non_block=True：提交后立即返回，不阻塞等待结果）
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    # ③ 等待 GPU 完成并采样 token
    model_output = future.result()
    # ④ 更新调度状态：处理 KV 传输完成通知、生成 token 输出
    return self.scheduler.update_from_output(scheduler_output, model_output)
这里有三个关键步骤：
1. scheduler.schedule() → 决定本轮跑哪些 request，构建 SchedulerOutput（含 KV 分配信息）
2. model_executor.execute_model(scheduler_output) → 下发给 Worker 执行
3. scheduler.update_from_output(scheduler_output, model_output) → 用模型结果更新调度状态（含 KV 传输完成通知）
第三层：Scheduler — 调度层的 KV Connector
文件：vllm/v1/core/sched/scheduler.py
Scheduler 在初始化时，如果配置了 kv_transfer_config，就创建调度侧 Connector：
# vllm/v1/core/sched/scheduler.py — __init__()
if self.vllm_config.kv_transfer_config is not None:
    # Scheduler 角色：仅做调度决策（判断能否从远端拉 KV、分配 block）
    # Worker 角色：由 ensure_kv_transfer_initialized() 独立创建，负责实际 GPU→GPU 传输
    self.connector = KVConnectorFactory.create_connector(
        config=self.vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=self.kv_cache_config,
    )
注意：这是 Scheduler 角色的 Connector，专门用于调度决策（比如判断这个 request 能不能从远端拉 KV）。而 Worker 侧会独立创建 Worker 角色的 Connector，专门执行实际的 KV 数据传输。
schedule() 调用结束后，SchedulerOutput 对象里会携带 kv_connector_metadata（包含每个 request 的 block_ids、remote_address 等传输元数据），这个 metadata 是后续 Worker 执行 KV 传输的"快递单"。
第四层：Worker — GPU 工作进程初始化 KV Transfer
文件：vllm/v1/worker/gpu_worker.py
Worker 在启动时会调用 ensure_kv_transfer_initialized：
# vllm/v1/worker/gpu_worker.py
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,  # 创建 Role=WORKER 的 Connector，绑定全局单例
    get_kv_transfer_group,            # 后续各处通过此函数取 Connector 实例
    has_kv_transfer_group,
)
这个函数创建 Worker 角色的 Connector（如 P2pNcclConnector），并绑定到全局单例 kv_transfer_state._KV_CONNECTOR_AGENT，所有后续 get_kv_transfer_group() 调用都会取到这个实例。
Worker 收到 execute_model(scheduler_output) 指令后，把调用转发给 model_runner.execute_model(scheduler_output)。
第五层：GPUModelRunner — Forward Pass 的三阶段
文件：vllm/v1/worker/gpu/model_runner.py
这是最核心的执行层，KV 传输被包裹在 forward pass 的三个阶段里：
# vllm/v1/worker/gpu/model_runner.py — execute_model() 三阶段
with set_forward_context(attn_metadata, self.vllm_config, ...):
    # 阶段①  pre_forward：绑定 connector_metadata；D 节点同步接收所有层 KV 并注入
    self.kv_connector.pre_forward(scheduler_output)
    # 阶段②  model forward：逐层执行 attention；
    #         每层由 @maybe_transfer_kv_layer 拦截 → P 节点提取 KV 推入发送队列
    model_output = self.model(**model_inputs)

# 阶段③  post_forward：P 节点等待所有层 KV 发完（确保 block 可安全复用）；收集传输状态
kv_connector_output = self.kv_connector.post_forward(scheduler_output)
第六层：ActiveKVConnector — pre/post_forward 的具体实现
文件：vllm/v1/worker/gpu/kv_connector.py
ActiveKVConnector 是 ModelRunner 和底层 Connector 之间的桥接层：
pre_forward 做了什么：
# vllm/v1/worker/gpu/kv_connector.py — ActiveKVConnector.pre_forward()
def pre_forward(self, scheduler_output):
    # ① 处理被抢占的请求，释放其 KV 传输资源
    if scheduler_output.preempted_req_ids:
        self.kv_connector.handle_preemptions(scheduler_output.preempted_req_ids)
    # ② 绑定"快递单"：block_ids + remote_address → Connector 当前帧元数据
    self.kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
    # ③ 启动 KV 加载：
    #    D 节点 → recv_tensor + inject（P2P 同步拉取全部层）
    #    P 节点 → no-op（发送在 @maybe_transfer_kv_layer 里按层逐一触发）
    self.kv_connector.start_load_kv(get_forward_context())
1. bind_connector_metadata：把 Scheduler 下发的"快递单"（block_ids、remote_address）绑定到当前 Connector 实例
2. start_load_kv：对于 D 节点，这里会启动异步 KV 拉取（对于 P2P 模式，实际是在 start_load_kv 里同步拉取并注入；对于 Mooncake，则是发起 RDMA 拉取）
post_forward 做了什么：
# vllm/v1/worker/gpu/kv_connector.py — ActiveKVConnector.post_forward()
def post_forward(self, scheduler_output, wait_for_save=True):
    output = KVConnectorOutput()
    # P 节点：阻塞等待 send_queue 清空，确保所有层 KV 已发出才允许释放 paged block
    if wait_for_save:
        self.kv_connector.wait_for_save()
    # 查询本轮哪些 request 的 KV 传输全部完成，Scheduler 据此决定何时释放 block
    output.finished_sending, output.finished_recving = (
        self.kv_connector.get_finished(scheduler_output.finished_req_ids)
    )
    self.kv_connector.clear_connector_metadata()  # 清理本帧元数据
    return output
1. wait_for_save()：阻塞等待所有异步 KV 发送完成，防止 P 节点的 paged KV buffer 在发送完成前被复用
2. get_finished：查询哪些 request 的 KV 传输已全部完成（Scheduler 据此决定何时释放 P 节点的 blocks）
第七层：@maybe_transfer_kv_layer— 注意力层的 KV 拦截装饰器
文件：vllm/model_executor/layers/attention/kv_transfer_utils.py
这是整个 PD 分离机制中最精妙的设计——在 注意力层 上打了一个装饰器钩子：
# vllm/model_executor/layers/attention/kv_transfer_utils.py
def maybe_transfer_kv_layer(func):
    # 在每个 attention 层前后无侵入地插入 KV 收发逻辑；未启用时零额外开销
    @wraps(func)
    def wrapper(*args, **kwargs):
        connector = get_kv_transfer_group()
        layer_name = args[layer_name_index]
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        # 未启用 KV 传输 / 无元数据 → 直接透传，不影响原有计算路径
        if not has_kv_transfer_group() or attn_metadata is None \
                or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # ① D 节点：等待该层 KV 到达（P2P 同步模式已在 start_load_kv 完成，此处 no-op）
        connector.wait_for_layer_load(layer_name)
        # ② 执行 attention forward（读写 kv_cache）
        result = func(*args, **kwargs)
        # ③ P 节点：提取本层 KV 推入发送队列（D 节点 no-op）
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
        return result
    return wrapper
这个装饰器被加在 unified_attention 函数上：
# vllm/model_executor/layers/attention/attention.py
@maybe_transfer_kv_layer          # ← 在每个 attention 层挂上 KV 收发钩子
def unified_attention(query, key, value, layer_name):
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    return self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
调用顺序（以第 k 层为例）：
顺序
调用
角色
说明
1
wait_for_layer_load("layer_k")
D 节点
等待 P 节点的第 k 层 KV 已写入本地 paged buffer
2
self.impl.forward(...)
双方
执行 attention 计算，读写 kv_cache
3
save_kv_layer("layer_k", kv_cache, attn_metadata)
P 节点
把第 k 层的 KV 提取出来发给 D 节点
第八层：save_kv_layer — P 节点的 KV 提取与发送
文件：vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py
# p2p_nccl_connector.py — save_kv_layer()（每个 attention 层执行后，P 节点调用）
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    if not self.is_producer:
        return   # D 节点跳过，KV 发送只在 P 节点执行

    def extract_kv_from_layer(layer, block_ids):
        # Fancy Index：把 paged KV pool 中散落的块聚合成连续 Tensor（物理 ID 脱敏）
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            return layer[block_ids, ...]        # MLA/FlashInfer: block_id 在 dim=0
        if layer.shape[0] == 2:
            return layer[:, block_ids, ...]     # FlashAttention: block_id 在 dim=1

    for request in self._get_connector_metadata().requests:
        ip, port = self.parse_request_id(request.request_id, True)
        remote_address = f"{ip}:{port + self._rank}"
        kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
        # tensor_id = "req_id#layer_name"，P/D 两端通过相同 ID 对齐数据
        self.p2p_nccl_engine.send_tensor(
            request.request_id + "#" + layer_name, kv_cache, remote_address
        )
关键点解析：
- 只有 is_producer=True（P 节点）才执行，D 节点直接 return
- kv_layer 是整个 pool 的大 Tensor（形如 [num_blocks, 2, block_size, num_heads, head_dim]）
- extract_kv_from_layer 用 request.block_ids（物理块 ID 列表）进行索引切片，把散落在 paged buffer 里的 KV 拼成一块连续 Tensor
- Tensor ID 构造为 "req_42#layer_model.layers.0.self_attn" 这样的格式，形成全局唯一标识
- send_tensor 把该 Tensor 推入发送队列
第九层：send_tensor — 异步发送队列（PUT_ASYNC 模式）
文件：vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
# p2p_nccl_engine.py — send_tensor()
def send_tensor(self, tensor_id, tensor, remote_address=None):
    if remote_address is None:
        # GET 模式：P 节点缓存到 recv_store，等待 D 节点主动来拉
        with self.recv_store_cv:
            self.recv_store[tensor_id] = tensor
            self.recv_store_cv.notify()
        return True

    item = SendQueueItem(tensor_id=tensor_id, remote_address=remote_address, tensor=tensor)

    if self.send_type == "PUT":
        return self.send_sync(item)    # 同步阻塞：等对端 ACK 确认后才返回

    if self.send_type == "PUT_ASYNC":
        # 推入队列立即返回，后台 _sender_loop 异步执行 ZMQ 握手 + NCCL 传输
        # 关键：第 k 层 KV 入队后，第 k+1 层 GEMM 立即开始 → 计算与传输完全并行
        with self.send_queue_cv:
            self.send_queue.append(item)
            self.send_queue_cv.notify()
        return True
三种模式：
- PUT_ASYNC（默认推荐）：把 SendQueueItem 放入发送队列，立即返回。后台线程（_sender_loop）消费队列，通过 ZMQ + NCCL 异步传输。这正是层级流水线的关键：第 0 层计算完，KV 入队立即返回，第 1 层的 GEMM 和第 0 层的 KV 传输同时在跑。
- PUT：同步发送，阻塞直到对端收到。
- GET：D 节点主动拉取模式，P 节点把 Tensor 存入 send_store，D 节点通过 ZMQ 请求 + NCCL 拉取。
第十层：D 节点接收 KV ——start_load_kv 与 recv_tensor
文件：vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py（start_load_kv方法）
在 pre_forward 中 D 节点调用 start_load_kv：
        # vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py
        # Load the KV for each request each layer
        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue

                layer = kv_cache[forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name, remote_address
                )
                ...
                inject_kv_into_layer(
                    layer, kv_cache, request.block_ids, request.request_id
                )
- D 节点用同样的 tensor_id（"req_42#layer_model.layers.0.self_attn"）向 P 节点请求数据
- recv_tensor 会阻塞等待，直到 P 节点把该 Tensor 传输过来（通过 ZMQ 信令 + NCCL 传输）
- inject_kv_into_layer 把收到的连续 Tensor 写回 D 节点的 paged KV buffer 对应的 block 位置
对应地，wait_for_layer_load 在 P2P NCCL Connector 里是空操作（因为 start_load_kv 已经同步完成了所有层的加载），而对于支持真正层级流水线的 Connector（如 Mooncake），wait_for_layer_load 才是真正的等待点。
补充：D 节点调度器是如何分配物理槽位的？
上面的 Worker 代码中，request.block_ids 是从哪来的？Worker 在调用 inject_kv_into_layer 时，直接拿着 D 节点分配好的物理块 ID 就能注入，但这些 block_ids 早在 Scheduler 阶段就已经确定了。以下是从请求到达 D 节点直到物理块 ID 被传递给 Worker 的完整链路。
阶段一：get_num_new_matched_tokens() — Connector 告诉 Scheduler 有多少 token 需要从远端加载
文件：vllm/v1/core/sched/scheduler.py（_schedule_new_requests 方法）
# scheduler.py — 调度新请求时，询问 Connector 有多少 token 的 KV 来自外部
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(
            request, num_new_local_computed_tokens
        )
    )
    request.num_external_computed_tokens = ext_tokens
    num_external_computed_tokens = ext_tokens
对应地，P2P NCCL Connector 的实现如下：
# p2p_nccl_connector.py — get_num_new_matched_tokens()（D 节点侧）
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    if self.is_producer:
        return 0, False  # P 节点不加载 KV，直接返回 0

    prompt_token_ids = request.prompt_token_ids or []
    # 所有 prompt token（除最后一个）都来自 P 节点
    # 减去已经本地命中的 prefix cache token 数
    num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens
    if num_external_tokens < 0:
        num_external_tokens = 0

    return num_external_tokens, False
    # 返回值: (需要从 P 加载的 token 数, 是否异步加载)
    # load_kv_async=False 表示同步加载（P2P NCCL 是 start_load_kv 里同步等待的）
> 为什么是 len(prompt) - 1 而不是 len(prompt)？
vLLM 规定：最后一个 prompt token 必须由 D 节点本地计算（需要它的 logit 来确定第一个 decode token），P 节点只需要传输前 N-1 个 token 的 KV。因此需要减去 1。
函数返回后，Scheduler 得到了两个关键数字：
num_external_computed_tokens = 47   （来自 P 节点的 KV，无需重算）
num_new_tokens               =  1   （最后 1 个 prompt token 需要本地计算）
阶段二：kv_cache_manager.allocate_slots() — 为整个请求分配物理块
文件：vllm/v1/core/sched/scheduler.py → vllm/v1/core/kv_cache_manager.py
Scheduler 随即调用 allocate_slots()，一次性为整个请求（包括 P 节点传来的 47 个 token + 本地 1 个 token）申请所需的全部物理块：
# scheduler.py
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens=1,                            # 需要本地计算的新 token 数
    num_new_computed_tokens=0,                   # 本地 prefix cache 命中的块数
    new_computed_blocks=new_computed_blocks,
    num_external_computed_tokens=47,             # 来自 P 节点的 token 数（用于计算需几个块）
    delay_cache_blocks=False,                    # P2P 同步传输，不需要延迟缓存
)
allocate_slots() 内部的块数计算逻辑：
# kv_cache_manager.py — allocate_slots() 内部
# 总共需要容纳的 token 数 = 本地已算 + 外部 token + 新 token
total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
#                     = 0 + 47 = 47
num_tokens_main_model = total_computed_tokens + num_new_tokens
#                     = 47 + 1 = 48
# 每块 block_size=16 个 token，48 个 token 需要 ceil(48/16)=3 块
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, max_model_len)
#                    = 48

# 检查空闲块是否足够
num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)
if num_blocks_to_allocate > block_pool.get_num_free_blocks():
    return None  # 显存不足，本次调度跳过该请求
阶段三：BlockPool.get_new_blocks() — 从 FreeKVCacheBlockQueue 中取物理槽
文件 vllm/v1/core/block_pool.py
allocate_slots() 内部最终调用 block_pool.get_new_blocks(num_blocks)，从空闲链表的头部弹出所需数量的物理块：
# block_pool.py — get_new_blocks()
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

    # 从 FreeKVCacheBlockQueue（双向链表）头部弹出 num_blocks 个块（LRU 顺序）
    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

    for block in ret:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)  # 如果该块是旧的 prefix cache，先驱逐
        block.ref_cnt += 1   # 引用计数 +1，标记为"已分配"
    return ret
FreeKVCacheBlockQueue 的数据结构和分配顺序如下：
```
  FreeKVCacheBlockQueue（双向链表，LRU 顺序）
   head ⇄ [blk] ⇄ [blk] ⇄ ... ⇄ [blk] ⇄ tail
   分配从 head 取（最久空闲优先）；释放挂回 tail
```
KVCacheBlock 的数据结构
@dataclass
class KVCacheBlock:
    block_id: int          # 物理槽位 ID（= GPU 显存中 KV Pool 的第几个 block）
                           # 取值范围 [0, num_gpu_blocks)，就是张量索引
    ref_cnt: int = 0       # 引用计数。0=在空闲链表中，≥1=已被分配
    _block_hash: ...       # prefix cache 哈希值，prefix cache 命中时使用
    prev_free_block: ...   # 双向链表前驱（仅 ref_cnt=0 时有效）
    next_free_block: ...   # 双向链表后继（仅 ref_cnt=0 时有效）
    is_null: bool = False  # 是否是占位 null block（block_id=0）
关键：block_id 就是物理 GPU 槽位号。block_pool.blocks[i]对应 KV Pool 张量的第 i 个 block 切片，block_id=i 意味着数据在 layer[:, i, :, :, :]（FlashAttention）。这是 inject_kv_into_layer 能直接用 block_ids 索引 KV Pool 的根本原因。
阶段四：update_state_after_alloc() — Connector 记录分配结果供后续使用
文件：vllm/v1/core/sched/scheduler.py → p2p_nccl_connector.py
块分配成功后，Scheduler 立即通知 Connector 保存这个请求对应的物理块列表：
# scheduler.py
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request,
        self.kv_cache_manager.get_blocks(request_id),  # KVCacheBlocks 对象
        num_external_computed_tokens,                   # 47
    )
# p2p_nccl_connector.py — update_state_after_alloc()
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    if not self.is_producer and num_external_tokens > 0:
        # 将 KVCacheBlocks 对象（包含 KVCacheBlock 对象列表）转为纯整数 ID 列表
        # blocks.get_block_ids()[0] 返回第 0 个 KV cache group 的 block_id 列表
        # 例如: [5, 9, 11]
        self._requests_need_load[request.request_id] = (
            request,
            blocks.get_block_ids()[0],  # ← 这就是最终注入时用的 block_ids！
        )
blocks.get_block_ids() 的实现就是把 KVCacheBlock 对象列表展开成整数列表：
# kv_cache_manager.py — KVCacheBlocks.get_block_ids()
def get_block_ids(self) -> tuple[list[int], ...]:
    return tuple([blk.block_id for blk in group] for group in self.blocks)
    # 例如 blocks[0] = [KVCacheBlock(5), KVCacheBlock(9), KVCacheBlock(11)]
    # 返回 ([5, 9, 11],)
阶段五：build_connector_meta() — 把 block_ids 打包进 SchedulerOutput 发给 Worker
# p2p_nccl_connector.py — build_connector_meta()
for new_req in scheduler_output.scheduled_new_reqs:
    if new_req.req_id in self._requests_need_load:
        # 从暂存字典里取回之前保存的 block_ids
        meta.add_request(
            request_id=new_req.req_id,
            token_ids=new_req.prompt_token_ids or [],
            block_ids=new_req.block_ids[0],  # ← D 节点分配的物理块 ID [5, 9, 11]
            block_size=self._block_size,
        )
        self._requests_need_load.pop(new_req.req_id)
最终，这个 block_ids=[5, 9, 11] 被封装进 ReqMeta → P2pNcclConnectorMetadata → SchedulerOutput.kv_connector_metadata，随着调度结果一起下发给 Worker。Worker 的 start_load_kv() 拿到后直接用于 inject_kv_into_layer(layer[:, [5,9,11], ...] = kv_cache)。
完整调度侧链路图
> _（完整调度侧链路：见 3.3 时序图 ASCII；核心是 get_num_new_matched_tokens → allocate_slots → build_connector_meta → start_load_kv）_
关于 delay_cache_blocks 的特殊处理
当 load_kv_async=True（如 Mooncake Connector 的异步传输场景）时，Scheduler 会把 delay_cache_blocks=True 传入 allocate_slots()。这会使 vLLM 跳过 prefix cache 的哈希登记，因为 KV 数据还没到，不能把一个空块登记为"已缓存"：
# kv_cache_manager.py — allocate_slots() 末尾
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)  # 直接返回，不做 cache_blocks()

# P/D 传输完成后（update_from_kv_xfer_finished），才会补充 cache 这些块
同时，P2P NCCL Connector 在 load_kv_async=False 的情况下，请求会直接进入 RUNNING 状态，Scheduler 在同一个 step 里就把它发给 Worker 执行；而异步场景下，请求会先进入 WAITING_FOR_REMOTE_KVS 状态，等 KV 传输完毕后再被重新调度。
3.3 完整数据流时序图

```
 Client   Proxy        P(Prefill)              D(Decode)
   │  请求  │                                      
   ├──────► │  max_tokens=1                        
   │        ├──────────► P.prefill                 
   │        │            ├ save_kv_layer(每层)     
   │        │            │   extract→send_tensor ══════(NCCL)═════► recv 后台线程
   │        │            └ 返回首token             │                存入 recv_store
   │        │◄───丢弃─────┘                         
   │        │  原始请求(完整 max_tokens)            
   │        ├────────────────────────────────────► D.schedule          
   │        │            get_num_new_matched_tokens=len(prompt)-1       
   │        │            allocate_slots→本地 block_ids                  
   │        │            start_load_kv: inject_kv_into_layer(本地槽位)  
   │        │◄───────────────── token 流 ──────────┤ decode 循环      
   │◄───────┤  SSE 流式返回                          
```

关键设计总结

> **三句话记住全流程**：① Proxy 用 `max_tokens=1` 把 P 变成纯 prefill，丢弃其响应；
> ② P 每算完一层就 `extract` 出 KV 经 NCCL 推给 D，D 后台线程收进 `recv_store`；
> ③ D 调度时算出 `len(prompt)-1` 个外部 token，分配本地块并 `inject` 注入，跳过 prefill 直接 decode。

3.4 远端的 Block 和本地的 Block 是怎么对应的？
远端的 Block 和本地的 Block 是怎么对应的？
在阅读源码或了解 PD 分离时，最让人困惑的问题往往是：P 节点的物理显存和 D 节点的物理显存是独立的。P 节点说"我算完了 block_id=5 和 12"，D 节点根本不认这两个 ID。那数据是怎么精准对应的？
答案是：物理 ID 脱敏，逻辑顺序对齐，显存切片注入。
下面我结合 p2p_nccl_connector.py 的真实源码，逐层拆解整个流程。
第零步：KV Pool 的真实张量布局（基础知识）
理解这套机制，首先要搞清楚 KV Cache 在 GPU 显存里的实际形状。vLLM 支持两种主流 Attention Backend，它们的 KV Pool 张量布局维度顺序完全不同：

```
  FlashAttention 布局:  [2, num_blocks, block_size, num_kv_heads, head_size]
                         ↑                                                  
                         维度0 = K/V    → extract: layer[:, block_ids, ...]

  MLA / FlashInfer 布局: [num_blocks, 2, block_size, num_kv_heads, head_size]
                                     ↑                                       
                                     维度1 = K/V → extract: layer[block_ids, ...]
```

其中维度为 2 代表 K 和 V 两条缓存。num_blocks 是整个系统的物理总块数，block_size 是每块包含的 token 数（通常为 16）。

```
  按 block_ids=[2,7,5] 做 gather（extract，P 侧）：
    pool[...]: [blk0][blk1][blk2][blk3]...[blk7]...     物理散落
                            └──┐      └──┐
    连续 KV :              [ 第0逻辑块 ][ 第1逻辑块 ][ 第2逻辑块 ]   ← 只剩逻辑序
  按 block_ids=[5,9,11] 做 scatter（inject，D 侧）：
    连续 KV : [ 第0逻辑块 ][ 第1逻辑块 ][ 第2逻辑块 ]
                  └────► pool[5]   └────► pool[9]   └────► pool[11]
```
（运行 `block_remap_demo.py` 可亲眼看到这套 gather→传输→scatter 的对号过程）
第一步：block_ids 是怎么从 Scheduler 传到 Worker 的？
在 P 节点，Scheduler 在每次调度时都会为每个请求建立一份 ReqMeta，记录了它在 P 节点上的物理块地址列表，并打包进 P2pNcclConnectorMetadata 随 SchedulerOutput 一起发往 Worker：
# p2p_nccl_connector.py — Scheduler 侧：build_connector_meta()
# 把 P 节点本地分配的 block_ids 记录进元数据
meta.add_request(
    request_id=new_req.req_id,           # 请求 ID（用于和 D 节点对齐）
    token_ids=new_req.prompt_token_ids,  # prompt token 列表（计算 num_tokens 用）
    block_ids=new_req.block_ids[0],      # ← P 节点的物理块 ID 列表，如 [3, 18, 42]
    block_size=self._block_size,
)
ReqMeta 的内部结构如下，block_ids 直接转为 Tensor 方便后续 GPU 索引：
@dataclass
class ReqMeta:
    request_id: str          # 请求唯一标识（P/D 两端完全相同）
    block_ids: torch.Tensor  # P 节点的物理块 ID，转为 Tensor
    num_tokens: int          # 该请求的 prompt token 总数
D 节点的 Scheduler 同样在 update_state_after_alloc() 里记录了 D 节点自己刚分配的 block_ids（如 [5, 9, 11]）。两端 request_id 完全相同，但 block_ids 互不相关：
> _（D 侧把本地 dst_block_ids 记入连接器状态的示意，详见下方源码片段）_
第二步：P 节点 Worker — 物理块提取
P 节点的 save_kv_layer() 在每个 Attention 层计算完毕后被调用。它用 PyTorch 高级索引（Fancy Indexing） 一次性把散落在各个物理槽位的 KV 数据聚合成连续张量：
# p2p_nccl_connector.py — save_kv_layer() 内嵌函数（真实源码）
def extract_kv_from_layer(
    layer: torch.Tensor,      # P 节点整个 KV Pool，shape=[2,50,16,8,128]
    block_ids: torch.Tensor,  # P 节点的物理块 ID，tensor([3, 18, 42])
) -> torch.Tensor:

    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        # MLA / FlashInfer：block_id 在 dim=0
        return layer[block_ids, ...]       # 输出 shape: [3, 2, 16, 8, 128]

    if layer.shape[0] == 2:
        # FlashAttention：block_id 在 dim=1
        return layer[:, block_ids, ...]    # 输出 shape: [2, 3, 16, 8, 128]
可视化：分散 → 聚合
> _（分散→聚合可视化：P 按 block_ids 把散落的 KV gather 成连续张量；可运行 `block_remap_demo.py` 实测）_
提取完后立刻调用 send_tensor()，把这个连续 Tensor 连同唯一标识推入发送队列：
# save_kv_layer() 中
kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
self.p2p_nccl_engine.send_tensor(
    tensor_id      = request_id + "#" + layer_name,
    # e.g. "abc-123#model.layers.0.self_attn"  ← 全局唯一，D 端按此 key 取数据
    tensor         = kv_cache,                  # 上面提取出的连续 Tensor
    remote_address = ip + ":" + str(port + rank) # D 节点的 ZMQ 地址
)
第三步：传输层 — ZMQ 控制平面 + NCCL 数据平面
P2pNcclEngine 把一次 Tensor 传输拆成两个独立信道：
```
  控制面 ZMQ:  P ──(req_id, shape, dtype)──► D, D 准备好缓冲区并 ACK
  数据面 NCCL: P ══(ncclSend)══► D(ncclRecv 写入预备好的显存)
```
> 为什么要先发 ZMQ 控制消息？ NCCL 的 ncclSend / ncclRecv 是配对原语：接收方必须提前 torch.empty() 准备好正确形状的显存缓冲区，NCCL 才能把数据写进去。ZMQ 控制消息就是让 D 节点提前知晓 shape 和 dtype，准备缓冲区并 ACK 后，双方才开启 NCCL 传输。
三种发送模式（send_type 配置项）：

```
  模式        发起方   是否阻塞主进程   机制                          性能
  ─────────  ──────  ─────────────  ───────────────────────────  ────
  PUT         P 主动   阻塞           同步 send，等发完才继续          低
  PUT_ASYNC   P 主动   不阻塞(专用线程) 异步发送队列, 计算/传输重叠     最高 ★
  GET         D 主动   —             P 先存 buffer, D 分配后来拉取    中
  官方实测性能：PUT_ASYNC > GET > PUT，生产环境优先 PUT_ASYNC
```
TensorMemoryPool — 接收缓冲区的溢出保护：
D 节点的后台监听线程把收到的 Tensor 先存在 GPU 显存的 recv_store 字典中。如果累积大小超过 buffer_size_threshold，则把数据卸载到 Pinned Host Memory（锁页 CPU 内存） 中，通过 Buddy Allocation 管理：
# listen_for_requests() — 接收侧溢出保护（真实逻辑）
if self.buffer_size + tensor_size > self.buffer_size_threshold:
    # GPU 缓冲区满，把 tensor 卸载到锁页内存（GPU → CPU DMA）
    addr = self.pool.store_tensor(tensor)
    tensor = (addr, tensor.dtype, tensor.shape)  # 仅保存元数据引用
else:
    self.buffer_size += tensor_size

# 后续 recv_tensor() 取用时：
if isinstance(tensor, tuple):
    addr, dtype, shape = tensor
    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)  # CPU → GPU
> _（接收侧溢出：GPU recv_store 满 → 卸载到 TensorMemoryPool(CPU Pinned)，取用时再 load 回 GPU）_
第四步：D 节点 Worker — 显存切片注入
D 节点的 start_load_kv() 在 forward pass 开始前被调用，它从 recv_store 取出已收到的 Tensor，用高级索引赋值注入到 D 节点本地的 KV Pool：
# p2p_nccl_connector.py — start_load_kv() 内嵌函数（真实源码）
def inject_kv_into_layer(
    layer: torch.Tensor,      # D 节点的 KV Pool，shape=[2,60,16,8,128]（60个槽）
    kv_cache: torch.Tensor,   # 从网络收到的连续 Tensor，shape=[2,3,16,8,128]
    block_ids: torch.Tensor,  # D 节点自己分配的物理块 ID，tensor([5, 9, 11])
    request_id: str,
) -> None:

    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        # MLA / FlashInfer
        num_block = kv_cache.shape[0]
        if len(block_ids) == num_block:
            layer[block_ids, ...] = kv_cache          # 完整注入
        else:
            layer[block_ids[:num_block], ...] = kv_cache  # 部分注入（chunked prefill）

    elif layer.shape[0] == 2:
        # FlashAttention
        num_block = kv_cache.shape[1]
        if len(block_ids) == num_block:
            layer[:, block_ids, ...] = kv_cache       # 完整注入
        else:
            layer[:, block_ids[:num_block], ...] = kv_cache  # 部分注入
为什么有 block_ids[:num_block] 的部分注入分支？ 这是 Chunked Prefill 的场景：P 节点可能分多个 step 才能完成全部 prompt 的 prefill，每个 step 只产生部分 block 的 KV 数据，因此 D 节点收到的 Tensor block 数可能少于分配的 block_ids 数量，需要只写入已有的部分。
可视化：聚合 → 再分散
> _（聚合→再分散可视化：D 把收到的连续张量按本地 block_ids scatter 注入；见 `block_remap_demo.py`）_
注入完成后，D 节点的 Attention 层在 decode 阶段就直接用 [5, 9, 11] 这三个本地物理槽位做注意力计算，完全感知不到这些 KV 数据是从远端传来的。
完整端到端数据流（带调用链）
> _（完整端到端调用链：见 3.3 时序图 ASCII 与 README.md「三层职责」图）_
关键设计汇总
> _（关键设计：物理 ID 脱敏 + 逻辑序对齐 + 显存切片注入；逐层发送实现计算/通信重叠）_
通过这套"分散提取 (P) → ZMQ 握手 → NCCL 传输 → 分散注入 (D)"的机制，vLLM 完美解决了异构物理显存之间的 KV 对应关系，实现了对上层 Attention 计算的完全透明。
四 KV Offloading Connector
KV cache offloading 功能是从 vLLM 0.11.0 版本开始引入的。
4.1  KV Offload 模块概述
1. 背景与痛点：显存瓶颈
在大语言模型推理中，显存（HBM）是最紧缺的物理资源。随着批处理量（Batch Size）的增加和上下文长度（如长文档、数十兆 PDF）的激增，GPU 显存极易被庞大的中间状态（KV Cache）耗尽。显存一旦触顶，将导致：
- 新请求无法被调度。
- 正在运行的请求被迫抢占（Preemption）。
2. 核心思想：显存的“多级缓存”架构
KV Offload 的本质是用极其廉价的 CPU 内存资源，换取逻辑上数倍放大的 KV Cache 容量。
它引入了分级缓存（Tiered Caching）的概念：
- 一级缓存（L1）：昂贵、高速的 GPU 显存。
- 二级缓存（L2）：容量巨大、廉价的主机 CPU 钉扎内存（或外部存储如 lmcache）。
3. 核心运行机制
模块通过 PCIe 总线在 CPU 与 GPU 之间进行高效的异步数据搬运：
- 换出（Offload / Store）：当 GPU 显存告急时，系统将暂时不活跃的请求（或预测短期内用不到的长 Prompt）的 KV 块，异步复制到 CPU 内存中，并释放对应的 GPU 物理块。
- 换入（Fetch / Load）：当请求被重新调度，或新请求恰好命中已被卸载的长前缀时，系统提前将对应的 KV 块从 CPU 异步拉回 GPU 继续生成（Decode）。免去了成百上千 Token 的重复计算（Prefill）过程。
4. 核心收益
- 突破容量天花板：彻底打破单机物理显存的限制。
- 降低延迟与抢占率：大幅降低由于显存不足导致的请求抢占率，显著优化 Time-To-First-Token (TTFT)。
- 长文本高效复用：极大提升了前缀缓存（Prefix Cache）的命中率，跨请求复用长文本变得成本极低。
5. 进阶架构：与 PD 分离的高度协同
在 vLLM V1 架构中，KV Offload 与 PD 分离（Prefill-Decode Disaggregation） 底层同源，且在应用上形成完美的闭环。
- 底层同源：两者均建立在 vLLM V1 的扩展接口 KVConnectorBase_V1 之上。它们拦截 Scheduler 的时机与 Worker 计算的时机完全一致。
  - PD 分离 (P2pNcclConnector)：跨越网络（IP/NCCL）搬运 KV。解决计算与显存的异构瓶颈。
  - KV Offload (OffloadingConnector)：跨越总线（PCIe）搬运 KV。解决单机显存的容量瓶颈。
- 业务链条协同（极端长文本场景下的流水线）：
  1. Producer 节点（计算型）：完成长文本 Prefill，产生海量 KV Cache，通过 PD 分离发送给 Consumer。
  2. Consumer 节点（显存型）：接收 KV 并进行 Decode。在极高并发下，即使是纯显存节点也会被塞满。
  3. KV Offload 兜底：Consumer 节点开启该模块，将处于“休眠”等待状态的对话 KV 淘汰至 CPU 内存。需要生成时再无缝载入。
总结：PD 分离解决了计算与显存的异构瓶颈；KV Offload 则彻底打破了单机物理显存的容量天花板。
开启该模块只需在启动时附加参数：--kv_offloading_backend native --kv_offloading_size <size_in_GB>。
借助底层的异步 DMA（直接内存访问）技术，数据在 GPU 与 CPU 之间的搬移开销被降至极低。在官方（v0.12.0版本）的 Llama-3.1-8B 测试中，该技术使 TTFT（首字延迟）降低高达 4 倍，吞吐量提升 5 倍。
更改 vLLM 的内存布局：
默认的 KV Cache 布局高度碎片化（以 16 tokens/块 为单位细分到每一层的 K 和 V），这对重度依赖连续大块数据传输的 PCIe 带宽是毁灭性的打击。
- vLLM 的解法：引入 register_cross_layers_kv_cache，将所有层的 KV 数据合并成一个连续的物理大块。
- 收益：物理块大小从原先的几 KB 暴增至 0.5 - 2 MB，使得 Offloading 传输吞吐量实现了数量级的飞跃。
4.2 KV Offload全局架构：各司其职的三层体系
vLLM 将整个 KV Offload 拆解成了三个完全解耦的层级：
1. 规则层 (KV Cache Interface)：负责“量体裁衣”。通过 KVCacheSpec 和 KVCacheConfig 计算每个缓存块的物理字节大小，为内存分配提供精确依据。
2. 大脑层 (Scheduler 侧)：负责“运筹帷幄”。包含 ConnectorScheduler（安排任务）和 OffloadingManager（记录哈希状态、执行 LRU 淘汰）。它们只处理元数据，绝不触碰真实数据。
3. 苦力层 (Worker 侧)：负责“物理搬运”。ConnectorWorker 接收指令后，交由底层的 DMA Handler 通过异步 CUDA 流执行实际的显存/内存拷贝。
> _（KV Offload 三层体系：规则层(KVCacheSpec) / 大脑层(Scheduler+OffloadingManager) / 苦力层(Worker+DMA)，此处略图）_
4.3 KV Offload 设计方案
vLLM 对 KV Offload 进行了清晰的控制面与数据面分离设计：
控制面：智能淘汰策略 (Eviction Policies)：
OffloadingManager 负责在 Scheduler 侧进行全局视野的块生命周期管理：
- LRUOffloadingManager：最经典的最近最少使用策略，按 touch() 时间戳排序，优先淘汰最久未访问的块。
- ARCOffloadingManager：自适应替换缓存（Adaptive Replacement Cache），结合了 LRU 和 LFU（最不经常使用），对抵抗大规模并发长序列带来的 Cache Thrashing（缓存抖动）具有极好效果。
- FilterReusedOffloadingManager：可选的包装器，针对具有系统级 Prompt 共享的情况，设置过滤阈值（store_threshold），高频复用的 Block 将一直锁定在 GPU，绝不 Offload。
数据面：异步双向流水线 (Async Pipeline)：
数据搬移绝不能阻塞 GPU 的矩阵计算（Compute-Bound 任务）。
- 模块实例化了两个 SingleDirectionOffloadingHandler：一个专职 GPU->CPU，一个专职 CPU->GPU。
- 独立 CUDA Stream：为每个 Transfer 请求从池中分配独立的 torch.cuda.Stream。
- 调用底层 C++ 算子 _custom_ops.copy_blocks，在硬件级实现 GPU 与 CPU Pinned Memory 的零拷贝 Direct Memory Access (DMA)。
为了让 Offload 过程“无感”且高效，vLLM 做了三个极其巧妙的设计：
设计一：内容寻址（BlockHash）而非位置寻址（Block ID）
- 常规做法：调度器记录 "把 GPU 的第 5 号块搬到 CPU 的第 10 号块"。这会导致无法跨请求复用，因为不同请求即使内容相同，分配的 GPU 块 ID 也不同。
- vLLM 方案：调度器只认 BlockHash（你可以把它想象成这个块内所有 token 内容的 MD5 值）。
- 效果：请求 A 计算了 "Hello World"，存入 CPU；请求 B 也包含了 "Hello World"，计算其 Hash 一看，发现 CPU 里已经有了！直接加载复用。
设计二：双队列异步 DMA 传输
- 问题：把几个 GB 的数据从 GPU 拷贝到 CPU，会阻塞显卡计算，导致模型生成卡顿。
- vLLM 方案：开启独立的 CUDA Stream（旁路通道）。GPU 的主流继续计算（Forward），旁路流悄悄在后台通过 PCIe 总线做内存拷贝（DMA）！！！
- 效果：计算和传输完全重叠，用户感知不到延迟。
设计三：Store 任务的“拖延症”
- 问题：如果推理完立刻把数据写回 CPU，此时系统正忙着返回生成的 token（采样过程），抢占 PCIe 带宽会导致单个 token 的延迟飙升。
- vLLM 方案：故意拖延。这一步要存的数据，先记在账上，等当前步骤彻底结束，下一个计算步骤开始的瞬间，再去执行写操作。错峰出行，保证丝滑。
4.4 模块代码解析
本文会剥离冗长源码，只用伪代码展示关键逻辑。
4.4.1 规格定义（KV Cache Interface）
职责：算出“一个缓存块到底占多少字节”，为后续内存分配提供依据。
# [精简伪代码] KVCacheSpec：计算缓存块物理大小
class FullAttentionSpec:
    block_size: int   # 每个块装几个 token（比如 16）
    head_size: int    # 维度大小
    
    def real_page_size_bytes(self):
        # 核心公式：K和V(2) * token数 * 头数 * 维度 * 数据类型字节
        # 就像算一个箱子的体积：长 × 宽 × 高
        return 2 * self.block_size * self.num_heads * self.head_size * dtype_bytes
4.4.2 调度大脑（Offloading Manager）
职责：在不碰真实数据的前提下，用字典记录哪些 Hash 存在 CPU 里。
# [精简伪代码] OffloadingManager：调度器侧的状态机
class LRUOffloadingManager:
    def __init__(self):
        # 记录 CPU 里的数据： Hash -> 物理块状态
        self.blocks = OrderedDict() 
        
    def prepare_store(self, block_hashes):
        """准备把数据写到 CPU"""
        过滤掉已经在 CPU 里的 Hash (避免重复写)
        
        if CPU 没地方了:
            从 blocks 最左边 (最老的数据) 淘汰掉几个旧块
            
        分配新的 CPU 物理块，标记状态为 "正在写入(ref_cnt=-1)"
        return 目标 CPU 块的 ID 列表
        
    def prepare_load(self, block_hashes):
        """准备把数据读回 GPU"""
        把命中块的引用计数 +1 (上锁，防止读的过程中被淘汰)
        return 源 CPU 块的 ID 列表
4.4.3 物理搬运（Offloading Handler）
职责：拿到调度器给的 ID 列表，调用底层 C++ 算子搬运数据。
# [精简伪代码] SingleDirectionHandler：异步 DMA 传输
class SingleDirectionHandler:
    def transfer_async(self, src_spec, dst_spec):
        # 1. 把大块 ID 展开成底层的小块 ID
        src_to_dst = 展开并映射(src_spec.ids, dst_spec.ids)
        
        # 2. 从池子里拿一个独立的旁路通道
        stream = 获取旁路_CUDA_Stream()
        
        # 3. 异步执行底层拷贝 (非阻塞)
        with torch.cuda.stream(stream):
            ops.swap_blocks(GPU_Tensor, CPU_Tensor, src_to_dst)
            
        # 4. 记录任务，供稍后轮询是否完成
        self.transfers.append((job_id, stream))
4.4.4 Scheduler 端的淘汰决断
在 vllm/v1/kv_offload/lru_manager.py 的 prepare_store 中，展现了经典的 LRU 淘汰流程：
def prepare_store(self, block_hashes: Iterable[BlockHash]) -> PrepareStoreOutput | None:
    # 1. 计算要往 CPU 塞入的新块与空闲槽位的差值
    num_blocks_to_evict = len(block_hashes_to_store) - self.backend.get_num_free_blocks()
    
    to_evict = []
    # 2. 如果 CPU 空间也不够了，根据 LRU 顺序踢掉最旧且无人引用 (ref_cnt == 0) 的块
    for block_hash, block in self.blocks.items():
        if block.ref_cnt == 0 and block_hash not in protected:
            to_evict.append(block_hash)
            num_blocks_to_evict -= 1
            if num_blocks_to_evict == 0: break
    
    # 3. 在 Backend (CPU Memory) 分配物理块
    blocks = self.backend.allocate_blocks(block_hashes_to_store)
    
    # 4. 生成 Spec 供 Worker 读取物理地址
    store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)
    return PrepareStoreOutput(...)
4.4.5 Worker 端的高效数据面执行
在 vllm/v1/kv_offload/worker/cpu_gpu.py 中，实际发生的数据搬移是被极其仔细地异步化的。它巧妙利用了 PyTorch 的 Stream 和 Event，实现了零阻塞的调度：
def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
    # ... 省略张量索引展平逻辑 ...
    
    # 1. 从池中取出 CUDA 独立流和事件
    stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
    start_event = self._event_pool.pop() if self._event_pool else torch.Event(...)
    end_event = self._event_pool.pop() if self._event_pool else torch.Event(...)

    # 2. 阻塞当前 stream 直到上一个任务流完成（保证依赖顺序，但不阻塞 CPU 主线程）
    if self._transfers:
        stream.wait_event(self._transfers[-1].end_event)

    with torch.cuda.stream(stream):
        start_event.record(stream)
        # 3. 核心大招：调用底层的 C++ 算子直接进行离散显存块的 DMA 拷贝
        # 该算子知道如何处理 Paged Attention 不连续的 Block
        ops.copy_blocks(
            self.dst_tensors,
            self.src_tensors,
            src_to_dst_tensor, # 源块到目标块的映射表
        )
        end_event.record(stream)

    # 4. 把记录放进双端队列，由主循环后续轮询检查是否完成
    self._transfers.append(Transfer(job_id, stream, start_event, end_event, num_bytes))
    return True
在这里，copy_blocks 是关键。传统的 tensor.copy_() 是连续内存操作，而由于 Paged KV Cache 的块是极其碎片化的，必须在底层 C++ 内核中一次性派发大量的离散 Copy 任务，以榨干 PCIe 的带宽。
4.5 实例化理解：内存怎么分？数据怎么搬？
为了更直观，我们以 Llama-3-8B (张量并行 TP=2) 为例。
内存分配账本
- GPU 块大小：16 tokens / 块
- CPU 块大小：64 tokens / 块（CPU 块通常更大，为了大块传输提升带宽）
- 单块物理体积：GPU 上一个 16 tokens 的块，算下来约 32 KB。
- 配置总量：假设允许 CPU 使用 4GB 内存。
初始化时发生了什么？
1. GPU 侧：分到了 1000 个块，占用约 1 GB 显存。
2. CPU 侧：系统分配了一块锁页内存 (Pin Memory)，大小为 4GB。因为 CPU 块是 GPU 块的 4 倍大，所以 CPU 共有 4GB / (32KB * 4) = 16,384 个大块。锁页内存能让 DMA 传输不经过 CPU 直接存取，速度极快。
形状对齐的魔术（expand_block_ids）
因为 CPU 块包含 64 个 token，GPU 块只有 16 个 token。调度器说：“把 GPU 的 42, 43, 44, 45 块，搬到 CPU 的第 200 块”。
底层是怎么做的？
- Worker 会将指令“展开”：
  - CPU 的大块 200，其实对应底层的一段连续空间：[800, 801, 802, 803]
  - 最终映射：GPU 42 -> CPU底层 800，GPU 43 -> CPU底层 801...以此类推，调用 C++ 算子精准填入。
端到端场景：一次长文档对话的奇幻漂流
用一个生动的场景串起所有代码流：系统里有两个用户，用户 A 上传了一份长文档，用户 B 稍后也对这份文档提问。
> _（长文档对话端到端场景图：用户 A 上传→KV 落 CPU，用户 B 命中同前缀→直接 load 复用，此处略）_

价值体现：用户 B 完全跳过了 320 Tokens 的前缀计算（Prefill 阶段），只花了一点点内存搬运的带宽，省下了几秒钟的算力和几百毫秒的延迟。
4.6 淘汰策略：LRU、ARC与频次门控的较量
当 CPU 的 4GB 内存也塞满时，淘汰谁？vllm 系统提供了三种“断舍离”策略：
1，常规武器：LRU (最近最少使用)
- 原理：把所有数据排成一队。谁被用到了，就插队到队尾。CPU 满了，就把队头（最久没被理睬的）一脚踢出去。
- 缺点：如果有个人突然上传了一本 10MB 的小说（只问一次），这本小说会把队列里原本经常被使用的“系统提示词”全挤掉（缓存污染）。
2，高级武器：ARC (自适应替换缓存)
- 原理：不仅看“最近用没用”（时间局部性），还看“用的频次多不多”（频率局部性）。
  - 它把内存分成两半：T1（只用过一次的冷数据）和 T2（用过多次的热数据）。
  - 还保留了两个“幽灵记事本” B1/B2，记录被踢出去的 Hash。
- 聪明之处：如果系统发现，刚从 T1 踢出去的数据又被访问了（B1 命中），它会反思：“最近新来的数据很重要”，自动把 T1 的空间调大。反之亦然。完全动态自适应。
3，外挂装甲：FilterReused (频次门控)
- 原理：一个装饰器外挂。相当于在内存门口站个保安。
- 策略：一个 Hash 第一次来，保安只做登记，不让存（跳过 Store，省下带宽）。只有当它第二次、第三次来（达到 store_threshold），保安才放行让它写进 CPU。
- 效果：完美解决“一次性超长文档”的污染问题，极大地节省了 PCIe 带宽。
总结：vLLM 的 KV Offload 模块完美地向我们展示了如何在兼顾计算性能的情况下，使用一层抽象同时兼容网络传输 (P2P PD 分离) 和总线传输 (CPU Offload)。这套基于 Manager 的智能驱逐算法配合底层的异步 CUDA 流水线，构成了 V1 引擎在海量并发下的显存护城河。
五 P2pNcclConnector 组件
P2pNcclConnector(P2P NCCL Connector) 是 vLLM v1 架构下为 xPyD 部署设计的 KV 传输组件，具有基于点对点通信的动态扩展能力，部分灵感来源于 Dynamo。它的核心作用：
1. 张量极速传输：利用 NCCL P2P (Peer-to-Peer) 实现跨节点/跨进程的 GPU 直通传输，绕过低效的 CPU 拷贝。
2. 异步与背压控制：支持异步发送队列（PUT_ASYNC）和按需拉取（GET）模式，防止网络拥塞阻塞推理主干。
3. 显存溢出保护：内置 TensorMemoryPool (伙伴分配算法)，当接收端 GPU 显存缓冲池满时，自动溢出到 CPU Pinned Memory。
4. 引擎状态解耦：将 vLLM 复杂的调度器逻辑（SchedulerOutput、Block ID 映射）翻译为底层引擎能理解的线性张量传输指令。
P2P NCCL Connector 的本质是“Proxy 路由 + ZMQ 控制 + NCCL 直连” 的动态 xPyD 架构：Prefill 只负责生成 KV（max_tokens=1），通过 NCCL 异步推送给 Decode，Decode 直接加载 KV 继续生成，从而实现 Prefill/Decode 彻底解耦 + 计算传输完美重叠。
组件核心代码实现在 distributed/kv_transfer/kv_connector/v1/p2p 目录。
5.1 设计方案
总体流程
P2pNcclConnector 工作的总体流程如下图所示，该 PD 分解解决方案的整个过程通过请求流程来描述：
1. 客户端向代理/路由器的 /v1/completions 接口发送 HTTP 请求。
2. 代理/路由器通过轮询或随机选择的方式选择一个 1P1D（1 个预填充实例 + 1 个解码实例），生成一个 request_id（规则稍后介绍），将 HTTP 请求消息中的 max_tokens 修改为 1，然后将请求转发到 P 实例。
3. 随后，代理/路由器立即将原始 HTTP 请求转发到 D 实例。
4. P 实例执行预填充，然后主动将生成的 KV 缓存发送到 D 实例（使用 PUT_ASYNC 模式）。D 实例的 zmq_addr 可以通过 request_id 解析。
5. D 实例有一个专用线程用于接收键值缓存（以避免阻塞主进程）。接收到的键值缓存会保存到 GPU 内存缓冲区中，缓冲区大小由 vLLM 启动参数 kv_buffer_size 决定。当 GPU 缓冲区满时，键值缓存会存储到本地 Tensor 内存池中。
6. 在解码过程中，D 实例的主进程从 GPU 缓冲区或内存池中检索 KV 缓存（由 P 实例传输），从而跳过预填充。
7. 解码完成后，D 实例将结果返回给代理/路由器，然后代理/路由器将其转发给客户端。
> _（xPyD 请求全流程图：见 README.md 整体架构与本节 6/7 步文字描述）_
实例规模：2 个 Prefill 实例（P） + 3 个 Decode 实例（D），每个实例内部 TP=2（2 个 GPU/Rank）。对应的
P2pNcclConnector 整体架构概述：
1. Proxy/Router：单点入口，使用 hash=robin（轮询） 路由 + ZMQ Server 做服务发现。
2. 控制面：ZMQ（ZMQ client/server）负责元数据交换、连接建立、Service Discovery。
3. 数据面：NCCL（红/蓝粗箭头）负责 KV Cache 的 GPU-to-GPU 直连传输（零拷贝、异步）。
4. KV 存储：每个 Rank 都有 KVCache + send/recv store（Tensor Memory Pool）。
上述工作流程的核心环节是 NCCL 异步传输：
1. Prefill 的 ZMQ Client 通知 Decoder 的 ZMQ Server。
2. 通过 NCCL All-Reduce / P2P（红/蓝箭头）将 KV Cache 直接从 Prefill GPU 推送到 Decoder GPU。
3. 支持 PUT_ASYNC 模式，实现计算与传输完全重叠。
KV 缓存传输方法
KVCache 传输有三种方法：PUT、GET 和 PUT_ASYNC。这些方法可以通过 --kv-transfer-config 和 kv_connector_extra_config 参数指定，具体来说，可以通过 send_type 字段来指定。
PUT 和 PUT_ASYNC 都涉及 P 实例主动向 D 实例发送 KVCache。区别在于，PUT 是同步传输方法，会阻塞主进程，而 PUT_ASYNC 是异步传输方法。PUT_ASYNC 使用专用线程来发送 KVCache，这意味着它不会阻塞主进程。相比之下，GET 方法涉及 P 实例在计算预填充后将 KVCache 保存到内存缓冲区。D 实例在为 KVCache 分配空间后，会主动从 P 实例检索计算出的 KVCache。
vLLM 官方实验结果表明，这些方法的性能从高到低依次为：PUT_ASYNC → GET → PUT。
通过 ZMQ 和 NCCL 进行 P2P 通信
只要知道对应 P/D 实例的地址，就可以执行点对点键值缓存传输（使用 NCCL），而不受 rank 和 world 大小的限制。为了支持采用 PD 解耦的实例的动态扩展（扩展和收缩），这意味着添加或删除 P/D 实例不需要完全重启系统。
每个 P/D 实例只需创建一个 P2pNcclEngine 实例。该实例维护一个 ZMQ 服务器，该服务器运行一个专用线程来监听 zmq_addr 地址，并接收来自其他实例的控制流请求。这些请求包括建立 NCCL 连接的请求和发送 KVCache 元数据（例如张量形状和数据类型）的请求。但是，它实际上并不传输 KVCache 数据本身。
当 P 实例和 D 实例首次传输 KVCache 数据时，它们需要建立 ZMQ 连接和 NCCL 组。后续的 KVCache 传输将复用此 ZMQ 连接和 NCCL 组。NCCL 组仅包含两个 rank，这意味着 world 大小为 2。此设计旨在支持动态扩展，即添加或移除 P/D 实例无需重启整个系统。只要知道对方的地址，即可执行点对点 KVCache 传输，而不受 rank 或 world 大小的限制。
NCCL 组拓扑结构
目前，KVCache 传输仅支持对称张量并行 (TP) 方法。未来将支持非对称 TP 和流水线并行 (PP) 方法。图 2 展示了 1P2D 设置，其中每个实例的张量并行度 (TP) 为 2。共有 7 个 NCCL 组：三个 vLLM 实例各自拥有一个 TP=2 的 NCCL 组。此外，P 实例的第 0 个 GPU 卡与每个 D 实例的第 0 个 GPU 卡建立一个 NCCL 组。类似地，P 实例的第 1 个 GPU 卡与每个 D 实例的第 1 个 GPU 卡建立一个 NCCL 组。

```
  1P2D, 每实例 TP=2  →  共 7 个 NCCL 组
                P 实例              D1 实例            D2 实例
              ┌────────┐         ┌────────┐         ┌────────┐
   rank0 行   │ GPU0 ──┼────①────┼─ GPU0  │         │ GPU0   │
              │  │TP组  │ ╲──────②──────────────────┼─ ↑     │
              │  │(组A) │  ╲      │ TP组   │         │ TP组   │
   rank1 行   │ GPU1 ──┼───╲┼─③──┼─ GPU1  │ (组B)   │ (组C)  │
              └────────┘    ╲────┼④───────┼─────────┼─ GPU1  │
                                 └────────┘         └────────┘
   组A/B/C = 3 个实例各自 TP=2 内部组
   ① P.gpu0↔D1.gpu0  ② P.gpu0↔D2.gpu0  ③ P.gpu1↔D1.gpu1  ④ P.gpu1↔D2.gpu1
   = 4 个跨实例 P↔D 同号卡组 ；  3 + 4 = 7 个 NCCL 组
   每组 world_size=2（仅两个 rank）→ 增删实例无需重启，天然支持动态扩缩容
```
每个 NCCL 组都会占用一定量的 GPU 内存缓冲区用于通信，其大小主要受环境变量 NCCL_MAX_NCHANNELS 影响。当 NCCL_MAX_NCHANNELS=16 ，一个 NCCL 组通常占用 100MB；而当 NCCL_MAX_NCHANNELS=8 时，通常占用 52MB。对于大规模 xPyD 配置（例如 DeepSeek 的 96P144D），这种实现方式目前尚不可行。未来，我们正在考虑使用 RDMA 进行点对点通信，同时也在关注 UCCL。
GPU 内存缓冲区和张量内存池
如果将 P 实例的 --max-num-seqs 参数设置得过大，由于批次大小较大，P 实例会同时生成大量的 KVCache，这可能会超出 D 实例的内存缓冲区容量，从而导致 KVCache 丢失。对应代码则是在 P2pNcclEngine.listen_for_requests() 函数中，若接收到的张量导致 buffer_size > buffer_size_threshold，引擎会将这个 GPU 张量立刻卸载。而一旦 KVCache 丢失，D 实例就需要重新计算 Prefill，这相当于执行两次 Prefill。因此，首次令牌获取时间 (TTFT) 将显著增加，导致性能下降。
为了解决上述问题，vLLM 在组件内手写了一个 TensorMemoryPool 本地 Tensor 内存池来存储 KVCache，在初始化时直接分配一块巨大的 Pinned Host Memory (锁页内存)，并采用 Buddy Allocation (伙伴分配算法) 将这块大内存按 2 的幂次切分与合并，实现微秒级的内存块分配 (store_tensor 返回一个指针地址，存入 recv_store)。
TensorMemoryPool 其灵感来源于 Linux 内存模块中使用的伙伴系统。由于内存容量足够大（通常在服务器上达到 TB 级），因此无需考虑前缀缓存或使用基于块的设计来重用内存，从而节省空间。当内存缓冲区不足时，KVCache 可以直接存储在 Tensor 内存池中，D 实例随后可以从中检索 KVCache。读写速度与 PCIe 相当，PCIe 4.0 的速度约为 21 GB/s，通常比预填充速度更快。否则，像 Mooncake 和 lmcache 这样的解决方案就没有必要了。Tensor 内存池充当流量分流区，通常只在突发流量高峰时才会启用。在最坏的情况下，我的解决方案的性能不会比使用缓存存储的正常情况更差。
总结
P2pNcclConnector 及底层机制是典型的通信与计算重叠（Overlap Computation and Communication） 范式设计。通过异步队列隐藏 NCCL 延迟、利用零拷贝 ZMQ 完成控制流，配合 TensorMemoryPool 防 OOM。
5.2 整体模块架构设计
整体模块架构设计图
下述模块架构设计图展示了 P2pNcclConnector 及其底层引擎在 PD 分离架构下的物理与逻辑位置。
> _（P2pNcclConnector 模块架构：ModelRunner → Connector → Engine(ZMQ+NCCL)，见上文「三层抽象架构」ASCII 图）_
上述架构图基本对应了 vLLM v0.17.0+ 中 P2pNcclConnector 的真实代码框架：ModelRunner → Connector → Engine（ZMQ + NCCL） 的分层设计，Producer 负责主动推送，Consumer 负责接收注入，实现了 PD 分离场景下最高效的 GPU-to-GPU KV Cache 传输。
核心类 UML 设计
模块严格遵循了职责分离原则。上层 Connector 负责 vLLM 业务逻辑（如何切割 KV Cache、处理 Block ID），下层 Engine 负责分布式通信，旁路组件负责内存管理。
> _（核心类 UML：见 2.4 节「KVConnectorBase_V1 家族」ASCII 类图）_
端到端流程解析 (时序图)
下图展示了一次完整的 PUT_ASYNC 异步传输端到端流转（涵盖了 ZMQ 握手与 NCCL 传输）：
> _（PUT_ASYNC 端到端时序：见 3.3 时序图与本节「三种发送模式」说明）_
5.3 P2pNcclConnector 运行示例 (xPyD 架构)
在生产环境中，vLLM 支持运行任意数量的 Prefill 节点和 Decode 节点（即 xPyD 架构，x 个 Prefill，y 个 Decode），并通过一个轻量级的 Python 代理服务 (disagg_proxy_p2p_nccl_xpyd.py) 进行流量分发与实例发现。
5.3.1 xPyD 模块架构与数据流示意图
下面是 xPyD 部署模式的全局架构图。图中展示了请求如何通过 Proxy 节点路由，以及 KV Cache 如何在 P 节点和 D 节点之间通过 P2P NCCL 高效流转。
> _（xPyD 全局架构：见 README.md 整体架构图；路由逻辑可运行 `proxy_xpyd_demo.py` 复现）_
5.3.2 核心配置参数解析
部署前，需要重点关注 --kv-transfer-config 中的参数设置，特别是 kv_buffer_size：
1. kv_buffer_size 的重要性：它是用于暂存接收到的 KV Cache 的显存缓冲区大小（单位为 Bytes）。经验值为 GPU 显存容量的 10%。
  - 设置过小：显存 Buffer 容易溢出，导致接收到的 KV Cache 被迫转存到 CPU 内存池（TensorMemoryPool），这会显著增加延迟（需经过 CPU-GPU 拷贝）。
  - 设置过大：会挤占可用于正常推理的 KV Cache 空间，导致系统最大 Batch Size 降低，从而降低整体吞吐量。
2. P 实例与 D 实例的差异化配置：
  - Prefill 节点 (生产者)：如果使用的是 PUT 或 PUT_ASYNC 模式，Prefill 节点只负责发送，不接收 KV Cache，因此 kv_buffer_size 可以设置得非常小（例如 1e1 即 10 bytes）。*注：如果使用 GET 模式，则需分配较大 Buffer 用于暂存待发送的数据。*
  - Decode 节点 (消费者)：作为接收端，需要配置充足的 kv_buffer_size（例如 8e9 即 8GB）以避免接收缓冲区溢出。
3. 通信模式 (send_type)：
  - 强烈建议优先使用 PUT_ASYNC 模式，它能提供最佳的性能和吞吐量。
4. 端口映射：
  - vllm 启动参数中的 --port 必须与 kv_connector_extra_config 中的 http_port 保持完全一致。
  - 确保 kv_port (用于 ZMQ/NCCL 通信) 各节点不冲突。
5.3.3 部署实战一：1P3D（1 个 Prefill，3 个 Decode）
以下示例基于一台具有多张 A800 80GB 显卡的机器，代理服务器 IP 设为 10.0.1.1，模型为 Llama-3.1-8B-Instruct。
1. 启动 Proxy 代理服务
Proxy 节点负责统一接收请求并调度到下游节点。
# 启动代理服务（需提前 pip install quart）
cd vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
2. 启动 Prefill 实例 (KV 生产者)
将显卡 0 用作 Prefill。配置 kv_role: kv_producer，缓冲区给 10 字节。
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 20001 \
    --served-model-name base_model \
    --max-model-len 10000 --max-num-batched-tokens 10000 --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' &
3. 启动 3 个 Decode 实例 (KV 消费者)
使用显卡 1、2、3 分别启动 Decode 节点。配置 kv_role: kv_consumer，缓冲区给 8GB (8e9)。
# Decode 1 (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 20002 \
    --served-model-name base_model \
    --max-model-len 10000 --max-num-batched-tokens 10000 --max-num-seqs 256 \
    --gpu-memory-utilization 0.7 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' &

# Decode 2 (GPU 2)
# 与上方命令一致，仅修改：
# CUDA_VISIBLE_DEVICES=2, --port 20003, "kv_port":"23001", "http_port":"20003"

# Decode 3 (GPU 3)
# 与上方命令一致，仅修改：
# CUDA_VISIBLE_DEVICES=3, --port 20004, "kv_port":"24001", "http_port":"20004"
5.3.4 部署实战二：3P1D（3 个 Prefill，1 个 Decode）
架构变为 3 个节点专职计算长文本 Prompt，1 个节点专职 Decode。
配置调整要点：
- 代理服务 (Proxy) 不变。
- Prefill 1、2、3 (GPU 0, 1, 2)：均配置为 --port 20001~20003，"kv_role":"kv_producer", "kv_buffer_size":"1e1"。
- Decode 1 (GPU 3)：配置为 --port 20004，"kv_role":"kv_consumer", "kv_buffer_size":"8e9"。
具体启动命令与 1P3D 类似，只需对换节点角色的参数即可。
5.3.5 请求测试与基准性能 (Benchmark)
单次请求测试：
向 Proxy 节点 (10001 端口) 发起请求：
curl -X POST -s http://10.0.1.1:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "base_model",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}'
Benchmark 压力测试：
使用 vLLM 内置的 benchmark 脚本向 Proxy 施压：
vllm bench serve \
    --backend vllm \
    --model base_model \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name "random" \
    --host 10.0.1.1 \
    --port 10001 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --ignore-eos \
    --burstiness 100 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "90,95,99" \
    --request-rate 3 \
    --num-prompts 1000
典型测试数据参考：
在 1000 个 Input Tokens，200 个 Output Tokens 的场景下，端到端 (E2E) P99 延迟表现极佳，通常在 ~2秒 左右完成（具体取决于硬件通信带宽）。
清理环境：
测试完成后，可以通过以下命令停止所有服务：
pgrep python | xargs kill -9 && pkill -f python
参考资料
- vLLM 部署 PD 分离应用
- Inference without Interference:Disaggregate LLM Inference for Mixed Downstream Workloads
- P2P NCCL Connector
- Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput
- vLLM PD分离KV cache传递机制详解与演进分析
