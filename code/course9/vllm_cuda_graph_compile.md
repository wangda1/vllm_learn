# vLLM 里的 CUDA Graph 与 torch.compile：prefill / decode / attention 怎么落地

> 承接 [`TUTORIAL.md`](./TUTORIAL.md) 第 5、8、9 章、[`warmup.md`](./warmup.md)，配套可跑 demo [`vllm_transformer.py`](./vllm_transformer.py)。
>
> 本文按「先分清概念、再看工程取舍、最后落到实践技巧」组织，回答这些连环问题：
> 1. torch.compile 和 CUDA Graph 到底各自解决什么？为什么要一起用？
> 2. 还有个第三者「FX 图切片」，它和 CUDA Graph 是一回事吗？
> 3. prefill 为什么基本不用「分桶 + padding」？
> 4. attention 的 `Q@Kᵀ` 随 context 变长、shape 在变，那 attention 还能上 CUDA Graph / torch.compile 吗？
> 5. vLLM 把 attention 包成 custom op、再 FX 切片、再 CUDA Graph —— 是否多此一举？
> 6. 实践中结合 custom_op 和 CUDA Graph 有哪些必须遵守的技巧？

---

## 0. 先分清三个概念：作用各不相同，别混为一谈

初学最容易把下面三件事搅在一起。它们处在**不同层、解决不同问题**：

| 概念 | 在哪一层 | 解决什么问题 | 一句话 |
|---|---|---|---|
| **torch.compile** | 编译层 | kernel **太多太碎 + 中间结果反复读写 HBM** | 把多个小算子**融合**成更少、更高效的 kernel（减少 kernel 数量与访存） |
| **FX 图切片**（split_graph） | 编译层 | 图里有 Inductor **编译不了的算子**（如 attention），不能整张一起编 | 在这些算子处把一张 FX 图**切成多段**，每段各自交 Inductor 编译 |
| **CUDA Graph** | 运行层 | 每步 **CPU 逐个 launch kernel 太慢**（launch-bound） | 把一串 kernel **录成一张图**，一次提交重放整段（减少 CPU launch 次数） |

记住三句话，后面都不会乱：
- **torch.compile 砍的是「kernel 数量与访存」**（靠融合）。
- **CUDA Graph 砍的是「CPU launch 次数」**（靠录制重放），**不加速 GPU 计算本身**。
- **FX 切片不是性能手段，是「编译可行性」手段** —— 因为 attention 这种手写 kernel 编不了，必须把它从可编译部分里隔开（详见第 5 节）。

> 为什么 torch.compile 和 CUDA Graph 要**一起用**？因为它们正交：compile 先把图变小变快（少 kernel、少访存），CUDA Graph 再把剩下那些 kernel 的 launch 开销也吃掉。vLLM 默认两者叠加。

---

## 1. 分桶 + padding 是给谁用的？

「分桶 + padding」**只是 CUDA Graph「静态 shape」限制的补丁**，torch.compile 不需要它：

| 工具 | 对动态 shape 的态度 | 需要分桶 + padding 吗 |
|---|---|---|
| **torch.compile** | 原生支持 dynamic shape（把 token 维标记 symbolic，guard 守卫） | **不需要**：一次编译产物服务所有长度，Inductor 生成的 Triton kernel 直接吃符号化 `num_tokens` |
| **CUDA Graph** | 只支持静态 shape（铁律三） | 原则上需要；但只有 **decode** 划算，prefill 基本不干 |

所以「prefill 为什么不用分桶 + padding」对两者答案不同：
- **compile**：根本不需要（它能直接吃动态 shape）；
- **CUDA Graph**：代价收益倒挂，不划算（下一节）。

---

## 2. prefill 的成本收益与 decode 完全相反

decode 用「分桶 + padding + FULL 图」是绝佳交易，但搬到 prefill，三项全反：

### ① 变化的维度：小而有界 vs 大而近连续
- **decode**：变的是 batch size，范围小（≈ [1,256]），log 间隔几档（1,2,4,…,256）即可覆盖，padding 至多多几行。
- **prefill**：变的是**总 token 数**（chunked prefill 把多条变长 prompt 摊平成一条 varlen），范围 [1, `max_num_batched_tokens`]（几千~上万），近乎连续。要分桶要么档位极多、要么间隔极大。

### ② padding 的代价：几乎免费 vs 真金白银（最关键）
- **decode 是 launch-bound / 访存受限**：GPU 单步算得极快，瓶颈是 CPU 发 kernel。pad 几行 ≈ 几个 token 的额外计算，GPU 上近乎免费；CUDA Graph 砍掉的 ~1ms launch 才是大头 → 净赚。
- **prefill 是 compute-bound**：GPU 本就被大 GEMM 喂满。padding 出来的 token 要走**完整 transformer 的真实 FLOPs**，且算的是垃圾数据。

  > 数值直觉：`num_tokens=5000` pad 到 `8192` → 多算 3192 token ≈ **浪费 ~39% 算力**，直接砍 ~39% 吞吐；而 prefill 里 kernel 巨大、launch 开销早被摊薄，CUDA Graph 能省的也许就几个百分点。**pad 成本 ≫ 图收益，交易亏本。**

### ③ CUDA Graph 的收益：decode 大 / prefill 小
CUDA Graph 只在 **launch-bound** 时收益明显。decode 正是；prefill 不是 → 给 prefill 整张 FULL 图本就没多少可省，却要背上 ② 的 padding 成本与多张大图的显存。

### ④ prefill 的 attention 天生数据相关，装不进静态全图
变长序列、各请求不同 seqlen、block table、prefix caching、causal mask —— attention 的工作量**运行前不可知**，无法冻进一张 FULL 静态图。这正是 PIECEWISE 的根因（第 5 节）。

---

## 3. attention 的 shape 在变，为什么 decode 还能上 CUDA Graph？

### 核心区分：CUDA Graph 限制「静态 shape + 静态 launch 配置」，不限制「算的量」

> CUDA Graph 要求每个 kernel 的**实参 shape、grid/block 维度、指针**在 capture 与 replay 间不变；但**不要求 kernel 内部干的活一样多**。kernel 的工作量完全可以**数据相关**，只要那个「数据」是从**固定 shape 张量里读出来的「值」**，而不是某个张量的「shape」。

vLLM 的 PagedAttention 就是照这个约束**反向设计**的：把「KV 变长」从 shape 里挤出去，变成固定缓冲里的值。

### decode 的两个变量，各自被中和

decode 每步有**两个**会变的东西，用**不同机制**分别压平：

| 变的东西 | 朴素表现为 | vLLM 怎么变成「固定 shape」 |
|---|---|---|
| **per-request 的 KV 长度** | `Q@Kᵀ` 的 KV 维 / score shape 变 | KV cache 是**预分配的分页定形缓冲**（num_blocks×block_size，shape 不随对话增长）；kernel 多吃 `block_tables`、`seq_lens`（定 shape，**值**变）；按 `max_seq` 起**固定 grid**，再用 `seq_lens` 的**值**做 mask / early-exit（超出真实长度的位置算了也丢弃） |
| **batch size**（同时跑几条） | Q 的行数变 → grid 变 | **分桶 + padding**（第 2 节的 decode 策略）：pad 到捕获档，Q buffer 定形 |

两个变量一个靠「分页 + metadata 把变长藏进**值**」，一个靠「分桶 + padding 把 batch 钉死」，**capture 看到的全是固定 shape** → decode attention 能整体进 FULL 图。每步只是 `copy_` 进新的 `seq_lens` / `block_tables` 值再 replay —— 正是 TUTORIAL 第 3 章「铁律二：地址固定、内容 in-place 更新」的体现。

> ⚠️ 关键实现细节：这个「按 seq_len 做 mask/early-exit」必须在 kernel 内部用**设备端比较**（`pos < seq_lens`）完成，**绝不能**在 host 上 `seq_lens[i].item()` 取值来决定循环 —— 后者是 D2H 同步，capture 时直接报 `operation not permitted when stream is capturing`（违反铁律一）。详见第 6 节实践技巧与 `vllm_transformer.py`。

### prefill attention 为什么仍难进全图

prefill 多了一个**查询 token 数本身大范围变化**（1~数千）：这直接改 Q 的前导维 → 改 kernel **grid** → 改 capture 看到的 shape。这个变量不像 KV 长度那样能藏进值里（它决定要启动多少并行单位），只能 pad，而 prefill 是 compute-bound、pad 代价高 → 所以 prefill attention 走 **varlen kernel（`cu_seqlens` 累积长度）+ 不进 FULL 图**。

---

## 4. torch.compile 对 attention：根本不「编译」注意力本身

- vLLM 里 attention 是**手写自定义算子**（FlashAttention / FlashInfer / PagedAttention），用 `direct_register_custom_op` 注册成 custom op（如 `vllm::unified_attention_with_output`），并**配一个 `register_fake`（假实现，只描述输出 shape/dtype）**。
- 有了 fake 实现，Dynamo 能把它当成**一个不透明的单节点**收进图里 —— **不会 graph break**，整层 trace 成一张完整 FX 图（attention 只是其中一个黑盒节点）。
- Inductor **无法跨这个不透明节点做融合**，于是它天然成为「可编译片段」之间的**边界**。torch.compile 只**融合 attention 前后的逐元素 / Linear / Norm**，attention 内核本身留给手写 kernel。
- 变长 KV 的处理因此**完全在那个手写 kernel 内部**（分页 / cu_seqlens），不归 torch.compile 管。

> 🔑 常见误解纠正：很多人说「attention 是 graph break」。**不准确**。带 fake 实现的 custom op **不触发 Dynamo break**；真正的「切」是 vLLM 在编译后端**主动做的 FX 切片**（下一节）。只有当 custom op **没**写 fake，或在 Python 里出现 `.item()` / 依赖张量值的 `if` 时，才会触发真正的 Dynamo break。

---

## 5. 图切片（FX split）≠ CUDA Graph：两根正交的轴

这是最容易混的地方，也是「vLLM 为什么不只用一张 FULL 图、是否多此一举」的答案。

### 两件事处在不同层

| | FX 图切片（`split_graph`） | CUDA Graph 模式（FULL / PIECEWISE） |
|---|---|---|
| 属于 | **编译层** | **运行层** |
| 决定 | 把整层切成几个**编译单元**（在 attention 处切） | 用几次 **capture** 把这些编译单元包起来重放 |
| 为什么存在 | Inductor 编不了 attention，必须把它隔开 | 减少 CPU launch 次数 |
| 关掉 CUDA Graph 还在吗 | **在**（编译本身就需要切） | —— |
| 依赖关系 | 独立 | **叠在「已编译 + 已切片」的产物之上** |

一句话：**切片决定「编译成几段」，FULL/PIECEWISE 决定「这几段用几次 capture 包」。** 同一份切好的产物，decode 用一次 FULL capture 全包住，prefill 用多次 PIECEWISE capture 分段包（attention 段留在图外 eager 跑）。

### 那为什么不干脆只用 FULL 图？（是否多此一举）

分情况看，就知道切片**不可省**：

| 场景 | 能否一张 FULL 图搞定 | 切片是否必要 |
|---|---|---|
| **纯 decode、规整 batch** | ✅ 能（paged-decode attention 定 shape、可捕获），vLLM 也确实用 FULL | 仍必要——见下「②③」 |
| **prefill / mixed** | ❌ 不能（varlen attention 不可捕获） | 必要，且是 **PIECEWISE 的前提** |

切片必要的三个理由：
1. **前提修正**：fake 实现对 custom op **不可省**。不写 `register_fake`，compile 推不出输出 meta → **会在该算子破图**。demo 里 `graph=1 / break=0` 正是因为写了 fake。
2. **attention 必须是不透明手写 kernel**：分页 gather + `block_tables` 间接寻址、online-softmax 不物化 `[seq,seq]` 分数矩阵、varlen、读写 KV cache + 经 `get_forward_context()` 取 metadata —— 这些 Inductor **生不出来**，只能作为预写 kernel「被调用」。所以 attention 处**必然**是编译边界，与用不用 CUDA Graph 无关。
3. **prefill 只能 PIECEWISE**：varlen attention 不可捕获，只能绕开它 —— 把 attention 之间的静态计算片段各自捕成小图、attention 段图外 eager 跑。这**依赖切片给出的边界**才知道「哪段捕、哪段放 eager」。

> **结论：不是多此一举。** 切片本质是**编译层的硬需求**（嵌入手写 kernel + 给 Inductor 干净的编译单元），CUDA Graph 只是**顺势复用**这些边界。纯 decode 下 FULL 一张图够用、切片不增加额外捕获开销；但这套基础设施支撑着「调用强制手写的 attention kernel」和「prefill 的 PIECEWISE」，缺了它 prefill 根本上不了 CUDA Graph，attention 甚至没法在 compile 下运行。

---

## 6. 实践技巧：结合 custom_op 和 CUDA Graph 的「六条军规」

把 attention（或任何手写 kernel）塞进「compile + CUDA Graph」体系时，必须遵守：

1. **手写 kernel 包成 custom op + 必配 `register_fake`**。fake 让 Dynamo 不破图、收成单节点，vLLM 才能确定性切片；不写 fake = 破图。
2. **把「会变的量」藏进 metadata 的「值」，不要藏进 shape**。`seq_lens` / `block_tables` 用**固定 shape**张量承载，长度变化体现在它们的**取值**上 → 满足 CUDA Graph 静态 shape（铁律三）。
3. **kernel 内部禁止任何 host 同步**。不要 `seq_lens[i].item()` / `.cpu()` 去决定控制流；改用**设备端 mask**（`pos < seq_lens[:,None]` + `masked_fill(-inf)`）或在 CUDA kernel 里按 `context_lens` 值 early-exit。否则 capture 直接失败（铁律一）。
4. **固定地址 + in-place 更新**（铁律二）。所有输入/输出/metadata 缓冲在 capture 前一次性分配好；每步用 `copy_` 把新值写进**同一块地址**再 `replay`，绝不重新赋值换对象。
5. **如实声明副作用 `mutates_args`**。custom op 若原地写输出（如 `output`），注册时要标出来，让 compile 正确处理依赖边界、不做错误的重排/消除。
6. **按阶段选 CUDA Graph 模式**：decode 走 FULL、prefill 走 PIECEWISE（vLLM 默认 `FULL_AND_PIECEWISE`），两者共用同一份编译 + 切片产物；dispatcher 按 batch 描述符（num_tokens / 是否 uniform decode 等）在运行时选用已捕获的图。

> 这六条在 [`vllm_transformer.py`](./vllm_transformer.py) 里都有可跑的最小实现：第 1 条见 `@torch.library.custom_op` + `register_fake`；第 2、3 条见 mask 版 `paged_attention`（无 `.item()`，故能被 FULL 图捕获）；第 4 条见 `__main__` 里 `seq_lens[0] += 1` 后 replay 输出随之改变。

---

## 7. 汇总

| 阶段 / 算子 | CUDA Graph | torch.compile |
|---|---|---|
| **decode（整体）** | ✅ FULL 图：batch 分桶 + padding；KV 变长藏进 `block_tables`/`seq_lens` 的值 | ✅ 融合非 attention 部分 |
| **prefill（整体）** | ⚠️ 一般不 FULL：token 数改 grid、compute-bound pad 太贵 → PIECEWISE / eager | ✅ dynamic shape，一次编译服务所有长度，**无需 padding** |
| **attention 算子** | decode 能进图（变长做成值 + 设备端 mask）；prefill 走 varlen kernel 不进全图 | ❌ 不编译注意力内核本身（custom op 黑盒），只融合其前后 |

> **三句话收尾**：
> - **作用不同**：torch.compile 砍「kernel 数 + 访存」，CUDA Graph 砍「CPU launch 次数」，FX 切片是「编译可行性」手段 —— 三者正交。
> - **分桶 + padding** 是 CUDA Graph 静态 shape 的补丁，只在「launch-bound + pad 几乎免费 + 形状可枚举」时划算（= decode，≠ prefill）。
> - **CUDA Graph 怕「shape / launch 维度变」，不怕「算的量变」**；vLLM 的 PagedAttention 把「KV 变长」从前者搬到后者（藏进 metadata 的值 + 设备端 mask），才让 decode attention 得以整体图化。
