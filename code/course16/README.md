# Course 16 · 采样（Sampling）与投机解码（Speculative Decoding）

本目录是一份**自洽的教学材料**，目标：掌握"大模型每步吐出 logits 之后怎么变成 token（采样）"
与"怎么少调几次大模型把 token 更快吐出来（投机解码）"这两件事的**核心原理**，并能回答
面试常见的 10 个问题（Q1–Q10）。

## 目录与学习路径

| 文件 | 覆盖 | 一句话 |
|---|---|---|
| `sampling_demo.py` | **Q1–Q4** | 从 logits 到 token 的完整采样链路（greedy/温度/top-k/top-p/min-p/penalty/批量异构/确定性/logprobs/约束解码），纯标准库可运行 |
| `spec_decode_demo.py` | **Q5–Q7** | 投机解码主循环 + 拒绝采样无偏性的蒙特卡洛证明 + 加速比，复刻 vLLM 扁平索引 |
| `tree_attention_demo.py` | **Q8** | 链→树升级、Tree Attention mask、一次前向验证整棵树、EAGLE 演进线 |
| `batching_spec_demo.py` | **Q9** | roofline 模型解释"大 batch 下投机解码收益塌/变慢"，KV cache 回滚与 PagedAttention 配合 |
| `DOC.md` | 源材料 | 1900+ 行长文：背景瓶颈、投机框架、Medusa、EAGLE、vLLM v1 源码走读（最详细，按需查） |
| `README.md` | 本文 | 学习指南 + Q1–Q10 蒸馏答案 + Q10(多模态)的纯文字解答 |

建议顺序：先跑 `sampling_demo.py` → `spec_decode_demo.py`，再 `tree_attention_demo.py`
→ `batching_spec_demo.py`，每个文件顶部都有逐课说明，边读边对照下面的 Q&A。

```bash
python sampling_demo.py
python spec_decode_demo.py
python tree_attention_demo.py
python batching_spec_demo.py
```

全部零依赖（只用 Python 标准库 `math`/`random`），用玩具 bigram / roofline 模型把原理讲透、
可运行可验证；真实 GPU/vLLM 细节在 `DOC.md`。

---

# 一、采样（Sampling）

## Q1 从 logits 到 next token 的采样链路

完整流水线（与 vLLM 顺序一致）：
```
logits → [惩罚/约束 mask] → ÷temperature → top-k → top-p/min-p → softmax → 采样
```
- **greedy**：直接取 argmax（等价 T→0）。确定性输出。
- **temperature**：`p_i = softmax(z_i / T)`。给 logits 整体除以 T 再 softmax。
- **top-k**：只保留 logit 最大的 k 个 token，**按排名**截断（固定数量）。
- **top-p (nucleus)**：按概率从大到小累加，保留累计概率刚达到 p 的最小集合，**按概率质量**截断（数量自适应：分布越尖保留越少）。
- **min-p**：保留 `prob ≥ min_p · max_prob` 的 token，**相对最强候选的比例门槛**，对温度更鲁棒。

被淘汰的 token 在 logit 空间置 `-inf`，最后统一 softmax + 多项式采样。

**追问 · temperature 数学**：T 缩放 logits 的"对比度"。
- T→0：最大 logit 概率→1，退化成 **argmax/greedy**。
- T→∞：`z_i/T→0`，所有 token 概率→`1/|V|`，退化成 **均匀分布**（最大随机）。
- T<1 更尖锐(更确定)，T>1 更平坦(更随机)。

**追问 · 顺序**：vLLM 顺序是 `temperature → top-k → top-p/min-p`。
- **top-k 对温度顺序不敏感**：温度是单调变换，不改变 logits 排名，所以选哪 k 个不变。
- **top-p/min-p 对温度顺序敏感**：它们比较的是**概率值**，温度会改概率，换序则 nucleus 集合会变。
（见 `sampling_demo.py` 第 3 课的实验对拍。）

**追问 · 三种 penalty**（改的量不同）：
- **presence**：只看"出没出现过"(0/1)，出现过就扣固定值 → 鼓励引入新词。
- **frequency**：正比"出现次数"，出现越多扣越多 → 抑制高频复读。
- **repetition**（乘性，HF 风格）：出现过就把 logit 往 0 拉（z>0 变小、z<0 变大）。

## Q2 vLLM serving 里 sampler 在哪、怎么批量异构采样

- **位置**：在模型 forward **之后**，拿到 `logits[batch, vocab]` 再采样；**在 GPU 上向量化执行**（避免 D2H 往返）。
- **批量异构**：一个 batch 里不同请求的 `temperature/top_p/top_k` 被打包成 per-row 张量（形状 `[batch]`），对 `logits[batch, vocab]` 逐行广播；不同 `seed` 各自维护独立 RNG 状态。所以**一次 forward 出 logits，再向量化采样**就能把异构请求一起处理，不需要拆循环（见 `sampling_demo.py` 第 5 课）。

**追问 · sort 瓶颈**：词表 128k+、batch 几百时，top-k/top-p 的全排序确实可能成为瓶颈。优化：
- top-k 用 **partial sort / radix-select**（只找前 k，不全排）；
- top-p 在已排序基础上做累加，或用**阈值二分**避免完整排序；
- 算子融合（temperature+penalty+mask 在一个 kernel）、把不需要 top-k/p 的 greedy 请求走快路径。

**追问 · logprobs**：`logprob(token) = log(softmax(logits)[token]) = z/T − logsumexp(z/T)`。算 logprob 本身几乎免费（softmax 已有）；**代价在取 top-n + gather/sort + D2H 拷贝**。返回 top-n 要在 `[batch,128k]` 上做 top-n 选择，n、batch 越大越贵；`prompt_logprobs`（对整段 prompt 每个位置都要分布）代价更大。

## Q3 确定性：固定 seed + T=0 为何 batch 推理仍可能不一致

**根因不在采样的随机数，而在浮点加法不满足结合律**：`(a+b)+c ≠ a+(b+c)`。
GPU 上 matmul/reduction 的累加顺序依赖 batch 大小、kernel 的 tiling/split-K 策略。
同一请求在 batch=1 和 batch=32 里归约顺序不同 → logits 在低位 bit 上有微小差异。
平时无所谓，但当两个 token 的 logit 几乎并列时，**T=0 的 argmax 会被这点噪声"翻盘"**，于是同 prompt、同 seed、T=0 两次结果不同（见 `sampling_demo.py` 第 6 课的可运行演示）。

**batch-invariant 可复现做法（2025，如 Thinking Machines 的工作）**：用"batch 不变"的 kernel——固定归约顺序/split 策略，使每行 logits 与 batch 无关（RMSNorm/matmul/attention 都要 batch-invariant 实现）；固定一切非确定源（cuBLAS workspace、原子加、flash-attn split）。代价：通常牺牲一些吞吐（放弃对当前 batch 最优的 kernel 配置）。

## Q4 约束解码 / JSON 模式怎么和采样结合

做法：**在采样前，把"当前语法状态下不合法的 token"的 logit 置 `-inf`（mask）**，softmax 后它们概率为 0，绝不会被采样。约束解码 = 一个特殊的 **logits processor**，只改"候选集"，之后 temperature/top-k/p/采样流程完全不变（见 `sampling_demo.py` 第 8 课）。

谁决定哪些 token 合法？一个跟随已生成 token 前进的状态机：JSON/正则→有限状态自动机(FSA)，上下文无关文法→下推自动机(PDA)。每步根据当前状态预计算一个"允许 token 的 bitmask"。

**和 xgrammar 的关系**：SGLang/vLLM 用 **xgrammar** 把 grammar 编译成自动机，高效地为每步生成 token bitmask（缓存状态、字节级 trie 加速），再作为 logits processor 应用。

---

# 二、投机解码（Speculative Decoding）

## Q5 为什么能加速，免费午餐来自哪

一句话：**decode 是 memory-bound 的串行瓶颈，瓶颈在反复搬权重而非算力**。batch=1 时每步只算 1 个 token，算力大量闲置。投机解码用便宜的 draft 先猜 γ 个 token，target **一次宽前向并行验证 γ+1 个位置**——而一次宽前向(验证 γ+1 个)和一次窄前向(1 个)耗时相近（瓶颈在搬权重）。于是**把多次串行窄前向合并成一次宽前向**，减少 target 调用次数。免费午餐 = 把 decode 阶段**闲置的算力**拿来并行验证（见 `spec_decode_demo.py` 第 0 课、`batching_spec_demo.py` 第 0 课）。

## Q6 正确性：怎么保证输出分布和直接用 target 采样完全一致

靠**拒绝采样（speculative/rejection sampling）**。对每个草稿 token y（draft 分布 q、target 分布 p）：
- 以概率 **`min(1, p(y)/q(y))` 接受**；
- 否则**拒绝**，从**修正分布 `recovered = normalize(max(p − q, 0))`** 里重采一个 token 顶替，并丢弃其后所有草稿（首次拒绝即停）。
- 若 γ 个草稿**全部接受**，末尾再从 target 第 γ+1 个位置的分布**免费采 1 个 bonus token**。

可以证明这样每个位置最终 token 的边际分布**严格等于 p**（`spec_decode_demo.py` 第 2 课用 20 万次蒙特卡洛验证：经验分布贴合 target M_p 而非 draft M_q，最大偏差仅 ~0.001）。

**追问 · 拒绝后从什么分布重采**：`normalize(max(p−q, 0))`——从 target 里扣掉 draft 已高估的部分再归一化，恰好补偿拒绝带来的偏差。

**追问 · greedy 退化**：T=0 时 p、q 都是 one-hot。验证退化成**简单字符串匹配**：草稿 token == target argmax 就接受，第一个不匹配处停下并采 target 的 argmax。无需概率运算。

## Q7 性价比：加速比由什么决定

定义 **mean acceptance length τ = 每次 target forward 期望产出的 token 数**（链式下 `τ = 1 + E[接受的草稿数]`，范围 `[1, γ+1]`）。
- **理想加速比 ≈ τ**（一次 target 调用产出 τ 个 token），实际还要减去 draft 自身开销与验证开销。
- 更完整：`speedup ≈ τ / (1 + γ·c)`，c = draft 单步成本 / target 单步成本。

**追问 · draft 大小权衡**：draft 越大 → 草稿越准 → 接受率↑ → τ↑，但 draft 自身越贵(c↑)、每步等草稿越久。draft 越小 → 便宜但接受率低。要在"接受率"和"草稿开销"之间找平衡——EAGLE 的巧妙在于 drafter 极小却接受率高（见 Q8）。

**追问 · 实测指标**：在线上报 **acceptance rate / mean acceptance length τ**、**draft 命中分布**、**端到端 TPOT（每 token 延迟）与吞吐**、**verify 占比**。判断赚没赚：看实际 TPOT 是否下降、τ 是否显著 >1、大 batch 下有没有反而变慢（见 Q9）。

## Q8 演进：独立 draft → Medusa → EAGLE/2/3 → Lookahead

每一步解决前一代的痛点（详见 `tree_attention_demo.py` 第 5 课）：
- **独立 draft 小模型**(Leviathan'22)：要单独训练/部署、分布不一定对齐 → 接受率有限。
- **Medusa**：不要独立小模型，在 target 最后一层 hidden 上接 K 个轻量"头"，每个头直接预测未来第 i 个 token。单卡可微调、部署简单。痛点：各头**无自回归依赖**，候选质量受限。
- **EAGLE/2/3**：关键洞察——自回归不确定性主要在**特征(hidden state)空间**。drafter 在**特征空间自回归**：输入 = 上一步 token embedding ⊕ target 上一步 hidden state，小 transformer 预测下一步 hidden 再映射成 token。复用了 target 高层特征，所以 drafter 极小却接受率高。EAGLE-2 用**动态树**(按置信度扩展)，EAGLE-3 去掉"特征要能重建 logits"的约束、多层特征融合。
- **Lookahead/Jacobi**：完全不要 drafter，用 n-gram 池 + Jacobi 迭代并行猜。

**追问 · 树状验证 & tree attention 省了什么**：草稿不是一条链而是一棵树（每个位置多个候选分支），把整棵树拍扁成序列，用 **tree attention mask 让每个节点只 attend 自己的祖先**，于是**一次前向并行验证所有路径**，且**共享前缀只算一次**（省重复计算 + 省多次 kernel/搬权重）。某分支被拒还能换兄弟分支 → 接受长度更高（见 `tree_attention_demo.py`，含可打印的 mask 矩阵）。

**追问 · EAGLE 特征级自回归 vs 再训小模型的关键差异**：EAGLE drafter **吃 target 的 hidden state**，站在 target 肩膀上——(1) 分布天然对齐 → 接受率高；(2) 自身可极小 → 草稿便宜。同时优化了 Q7 加速比的两个因子（接受率↑ 且 草稿开销↓）。

## Q9 和 continuous batching 共存（题眼）

**为什么大 batch 下收益塌甚至变慢**：投机解码的免费午餐来自 decode 的**闲置算力**。continuous batching 把几百请求拼成大 batch 后，本身就把算力喂饱了——**没有闲置算力可白嫖**。此时 verify 要处理 `B×(γ+1)` 个 token 位置，进入 compute-bound 区间，耗时∝token 数线性上涨；而它只多产出 `τ` 倍 token（τ<γ+1）。结果：单位 token 反而更贵，**加速比跌破 1，净变慢**（`batching_spec_demo.py` 用 roofline 模型把 2.5x → 0.62x 的塌缩算给你看）。

**什么场景该开 / 要不要动态开关**：
- **该开**：低并发/小 batch（交互聊天、本地单用户）、延迟敏感、高接受率负载（代码/结构化文本、EAGLE drafter）。
- **该关**：高并发/大 batch 的吞吐优先服务、低接受率负载。
- **引擎应动态开关**：按 batch 大小/GPU 利用率/在线 accept rate 自适应开关与调 γ（vLLM 有 `speculative_disable_by_batch_size` 之类阈值，batch 超阈值停用）。

**追问 · 被拒草稿的 KV cache 回滚 & PagedAttention 配合**：verify 时 target 前向会为草稿位置 d1,d2,d3 都算 K/V 写进 KV cache，若 d2 被拒，**回滚 = 把 `num_computed_tokens` 截断到"已接受长度"**——O(1) 指针操作，不用逐元素清 KV；脏 KV 下一轮被新 token 覆盖，且 attention 只读到 `num_computed_tokens` 以内。与 PagedAttention 配合：写了一半的尾块不急于释放(下步接着写)，接受长度可变(0~γ+1) 带来的变长靠 `block_table` + 扁平索引(见 `spec_decode_demo.py` 第 4 课)统一管理（`batching_spec_demo.py` 第 4 课有可运行的回滚演示）。

## Q10 多模态：投机解码在 VLM 和 Diffusion 上还适用吗（加分项）

**先回到第一性原理**：投机解码的前提是 ——(1) 生成是**自回归、串行、逐 token** 的；(2) 该过程 **memory-bound、算力闲置**；(3) 存在一个**便宜且分布对齐的 draft** 能猜下一步。按这三条逐类判断：

**VLM（图文理解，如 LLaVA/Qwen-VL）——适用，几乎"免费迁移"**：
- VLM 的生成端就是一个 LLM 解码器，图像只是被编码成一段 prefix token（vision encoder + projector），**decode 阶段和纯文本 LLM 完全一样是自回归 + memory-bound**。所以投机解码（含 EAGLE）**直接适用**，且已有 EAGLE for VLM 的工作。
- 注意点：(1) draft 最好也能"看到"图像特征，否则在强依赖图像的 token 上接受率会掉（EAGLE 吃 target hidden 天然带上了图像信息，优势明显）；(2) **prefill 很重**（一张图几百~上千 vision token），但投机解码加速的是 decode，不改善 prefill 首字延迟；(3) 图像 token 长 prefix → KV cache 大，和 PagedAttention/前缀缓存的配合更关键。

**Diffusion（图像/视频生成）——基本不适用，至少不能照搬**：
- Diffusion 生成**不是自回归逐 token**，而是从噪声开始**迭代去噪**（几十步 denoising），每步是对**整张图/整段 latent** 的一次大卷积/Transformer 前向——这是**compute-bound 的稠密计算**，不存在"逐 token 串行 + 算力闲置"的结构，拒绝采样那套（min(1,p/q) 验证下一个 token）**根本无从套用**。
- Diffusion 的"加速"走的是另一条技术线：**减少去噪步数**（DDIM、DPM-Solver 等高阶 ODE 求解器）、**蒸馏**（Consistency Models、LCM、一步生成）、**caching**（相邻去噪步的特征复用，如 DeepCache）。
- 一个**形式上类比**：有研究把"小模型/少步先出草稿、大模型/多步验证修正"的思想迁移到 diffusion（speculative/cascaded sampling、draft-then-verify denoising），但本质是**步数维度**的投机，不是 token 维度的拒绝采样，机制和正确性保证都不同。
- **自回归式图像/视频生成**（如 VAR、把图像 patch 当 token 的 AR 模型、或视频按帧/patch 自回归）则**重新具备**逐 token 自回归结构 → 投机解码又适用了。所以关键不在"图像 vs 文本"，而在**生成范式是不是自回归 + memory-bound**。

**一句话总结 Q10**：投机解码绑定的是"**自回归 + memory-bound**"这个结构，不是模态。VLM 的解码端就是 LLM → 适用；标准 Diffusion 是迭代去噪的稠密计算 → 不适用，加速另走减步/蒸馏/缓存；而 AR 式视觉生成又把投机解码请了回来。

---

## 核心要点速记

**采样**：链路 = `[惩罚/约束 mask] → ÷T → top-k → top-p/min-p → softmax → 采样`；
T 控尖/平(T→0 贪心、T→∞ 均匀)；top-k 按排名、top-p 按概率质量、min-p 按相对门槛；
top-k 对温度顺序不敏感，top-p/min-p 敏感；sampler 在 GPU 上向量化按 per-row 参数采样；
batch-invariance 难在上游 logits 逐 bit 一致；约束解码 = logit mask(-inf)，xgrammar 编译 grammar 出 bitmask。

**投机解码**：decode 是 memory-bound 串行瓶颈 → draft 猜 γ 个、target 一次宽前向并行验证；
拒绝采样 `min(1,p/q)` + `recovered=normalize(max(p−q,0))` 保证无偏；加速比 ≈ mean acceptance length；
演进 独立draft→Medusa→EAGLE(特征级自回归+动态树)→Lookahead；tree attention 一次验证整棵树；
大 batch 下免费午餐消失、收益塌甚至变慢，引擎应动态开关；KV 回滚 = 截 `num_computed_tokens`；
绑定"自回归+memory-bound"结构：VLM 适用、标准 Diffusion 不适用。
