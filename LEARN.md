## [2026-06-03] FFN 与 MoE:Transformer 前馈子层的两种形态
**标签**: transformer, ffn, moe, swiglu, llm-arch
FFN 是 Transformer block 里逐 token(position-wise)的非线性变换子层;MoE 把单个 FFN 换成"N 个 FFN(专家)+ router",实现"参数大、每 token 计算小"。
- 经典 FFN = 带瓶颈的 2 层 MLP:`Linear(d→d_ff) → 激活 → Linear(d_ff→d)`,通常 `d_ff=4d`(先升维做非线性,再降回)。
- 现代 LLM(LLaMA/Qwen/Mistral)用 SwiGLU:gate/up/down 三个 Linear,`down(SiLU(gate(x)) * up(x))`;多一路 gate,故 `d_ff` 常取 ≈8/3·d 以对齐参数量。角色仍是 FFN。
- 一个专家 = 一个独立 FFN。MoE 层 = N 个并列 FFN + Router;Router 是个小 Linear(d→N),对 token 打分后取 top-k 并 softmax 得组合权重。
- Dense vs MoE(同一 d):MoE 总参数 ≈ N 倍(容量大),但每 token 只激活 top-k 个专家(计算省);对外都是 `[..,d]→[..,d]`,可互换。
- 代码示例:`vllm_learn/code/course6/ffn.py`(ClassicFFN/SwiGLUFFN)、`moe.py`(Router/MoELayer,复用 SwiGLU 当专家)。

## [2026-06-04] 用 VSCode 调试 vLLM 在线推理(OpenAI API server)
**标签**: vllm, debug, vscode, api-server, online-inference
断点只在"真正执行的那份代码"里命中——本机跑的是 site-packages 里的已编译 vLLM,不是未编译的本地源码树。
- 区分两份 vLLM:已安装的(site-packages,带 `*.abi3.so` 编译扩展,`import vllm` 解析到这里,能跑)vs 本地源码 `./vllm`(纯源码、无 `.so`/无 `_version.py`,且版本更新,无 CUDA build 跑不起来,只能静态阅读)。要调试/打印,必须改"能跑的那份"。
- 在线推理入口 = `vllm/entrypoints/openai/api_server.py` 的 `create_chat_completion`(`/v1/chat/completions`);最小 demo 是 `entrypoints/api_server.py` 的 `/generate`。请求路径:handler → `OpenAIServingChat.create_chat_completion` → engine 调度 → 流式返回。
- launch.json 关键点:用 `"module": "vllm.entrypoints.openai.api_server"` 启动;`"justMyCode": false` 才能单步进 site-packages 的 vllm 库;`--enforce-eager` 跳过 CUDA graph 便于单步;env 设 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 让 EngineCore 同进程运行,否则引擎在子进程里断点打不中。
- 小模型调试用 `model/Qwen3-0.6B` 足够快。

## [2026-06-04] vLLM 安装时编译哪些 .so(CMake 扩展)
**标签**: vllm, build, setup, cuda, so, flash-attention
vLLM 的 `.so` 来自 setup.py 的 `ext_modules`(一组 CMakeExtension),编哪些取决于后端(CUDA/ROCm/CPU)和 CUDA 版本;`optional=True` 的在硬件不支持时静默跳过,所以不同机器数量不同。
- CUDA 机器典型 7 个:`_C`(核心算子:PagedAttention/RMSNorm/RoPE/量化/激活)、`_moe_C`(MoE fused 算子)、`cumem_allocator`(KV cache 显存管理)、`_vllm_fa2_C`(FA2)、`_vllm_fa3_C`(FA3,CUDA≥12.3)、`_flashmla_C`/`_flashmla_extension_C`(DeepSeek MLA,CUDA≥12.9+Hopper)。
- 后端分支:CUDA/ROCm 共有 `_moe_C`+`cumem_allocator`;仅 CUDA 才有 FA2/FA3/FlashMLA/DeepGEMM;仅 ROCm 有 `_rocm_C`;仅 CPU 有 `_C`+x86 的 `_C_AVX512`/`_C_AVX2`;所有平台都编 `spinloop`。
- `VLLM_USE_PRECOMPILED=1` 不本地编译,直接用官方预编译 wheel 的 .so(纯 Python 改动时推荐,省去 CUDA 编译)。
- 判断源码树有没有 build:看 `vllm/vllm/*.so` 是否存在;没有 .so = 没编译 = 跑不起来。出处 `vllm/setup.py:991-1049`。

## [2026-06-05] vLLM 数据并行(DP)的 wave 与 Coordinator 协调机制
**标签**: vllm, data-parallel, wave, coordinator, async-llm, engine-core
DP(`--data-parallel-size N`)= 启动 N 个 DP Rank(各一个 EngineCore 进程/一份完整权重+独立 KV cache),三大组件 AsyncLLM(前端)/Coordinator(中枢)/Engine(后端) 经 ZMQ 协作。
- 两条数据通路要分清:**请求推理走直连**(AsyncLLM→Engine,不过 Coordinator);**唤醒控制信号走中转**(AsyncLLM→Coordinator 发 `FIRST_REQ`→Coordinator 广播 `START_DP_WAVE`→Engine)。
- **wave = 全局轮次编号**,不是把请求打成 batch,而是让所有 Engine 对"现在第几轮协同"保持一致;wave 数 = 所有 Engine 从 running→paused 的次数。后端暂停时前端发请求会先 `FIRST_REQ` 唤醒全员。
- **步调同步**:每个 Engine 即使本轮无真实请求也要执行 dummy batch,以对齐 `torch.distributed.all_reduce`(聚合"是否还有任意 rank 有未完成请求");全局确认空闲后全员同时 paused 且 wave++,rank0 上报 `wave_complete`。Coordinator 还把各 Engine 的 `[waiting,running]` 负载快照(counts)发布给前端做内置负载均衡。
- **MoE vs Dense 区别**:仅 MoE 走 `DPEngineCoreProc` 的跨 rank wave 同步(EP 通信需要所有 rank 协同 forward);Dense 模型各 DP rank 用普通 `EngineCoreProc` 独立处理,DP 字段被重置为 1,不走 wave 同步。
- **对称布局(关键,别误解)**:vLLM 主线大 MoE 部署是 **DP-attention + EP-moe,同一组 N 张卡身兼两职**——每张卡既持完整 attention 权重各算各的请求(DP),又托管全模型 1/N 的专家切片(EP)。**不是**"attention 卡 + 专家卡"的角色分离/点对点 send-recv(那是 attn-FFN disaggregation,vLLM 不这么做)。MoE 跨 rank 通信是 **all-to-all 集合通信,横跨整个 EP 组(=所有 DP rank)**:dispatch(按 token 选中的专家所在 rank 分发)→本地专家算→combine 收回。因 MoE 路由数据相关(每 token 的 top-k 运行前不可知,任意 rank 可能发往任意 rank),天然是 all-to-all 而非固定 send/recv。
- **死锁根因(为什么 rank0 有请求 rank1 没请求)**:attention 段是 DP→各 rank 独立 scheduler/独立队列→负载天然不均(请求异步到达、长短不一),某刻 rank0 有 token、rank1 空很正常;但 MoE 段 all-to-all 要求全员一起进 forward→rank0 跑到 MoE 等大家、rank1 不进这步 forward→rank0 永久挂死。矛盾=「attention DP(允许不均) + MoE all-to-all(要求全员)」叠在同一组卡上,wave+dummy batch 即消解此矛盾的同步机制。
- **wave 到底解决什么(为什么需要)**:MoE+EP 下专家切在不同 rank,每步 forward 内嵌跨 rank 的 all-to-all/all-reduce——集合通信"缺一个就整体阻塞"。这把本应独立的 N 个 rank 强绑成必须步调一致的整体。**没有 wave 的后果**:① 死锁——rank0 有请求跑到 all-reduce 阻塞等待,rank1 无请求不执行该步 forward、永不调用对应 all-reduce,rank0 永久挂死、整个 DP 组连带垮掉(这正是 dummy batch:无请求的 rank 也空跑一遍只为陪调 all-reduce);② 无法安全整体休眠——单个 rank 不能擅自退出循环(一退别人 all-reduce 缺人挂死),也无从判断全局是否真空闲。wave 用 all-reduce 聚合"全局是否还有未完成请求",确认空闲才全员同时 paused 且 wave++,下次 FIRST_REQ→START_DP_WAVE 全员一起唤醒。本质=给被 EP 通信强绑定的 rank 一个"全员同进同出"的全局节拍器。
- 教学 demo:`vllm_learn/code/course12/dp_wave_coordinator_demo.py`(线程模拟三组件,无 GPU/vLLM 依赖)。

## [2026-06-07] vLLM 采样与投机解码(Speculative Decoding)原理
**标签**: vllm, speculative-decoding, rejection-sampling, medusa, eagle, ngram
投机解码=「先草稿后验证(Draft-then-Verify)」:小模型/草稿器先猜 γ 个 token,大模型**一次宽前向并行验证**多个 token,把"多次窄前向"合并成"一次宽前向",减少大模型串行调用次数。动机:decode 阶段每步只生成 1 token,GEMM 退化为 GEMV,瓶颈是反复读写权重(访存密集 memory-bound)而非算力;一次宽前向(验证 γ+1 位置)与窄前向(1 位置)耗时相近。
- **单步流程**:① Drafting 小模型自回归生成 γ 个草稿+记录 q 分布;② Verification 大模型在 context+草稿 上一次 forward 拿 γ+1 个位置 logits;③ Accept/Reject 从左到右拒绝采样,**首次拒绝即停**。全接受→末尾免费多采 1 个 **bonus token**(位置 γ+1 分布已算好);第 i 个被拒→保留前 i 个、第 i 位换 recovered token、丢弃其后草稿。每步至少推进 1、至多 γ+1 个 token。
- **拒绝采样数学(保证无偏)**:接受概率 `min(1, p/q)`(p=target 概率,q=draft 概率);拒绝时从修正分布 `recovered = normalize(max(p-q, 0))` 重采。可证首个产出 token 边际分布恰为 p:`min(p,q)+max(p-q,0)=p`。故输出分布**严格等于从大模型逐 token 采样**,只是更快。greedy 是特例(比 argmax)。
- **加速边界**:取决于 draft 与 target 分布相似度,草稿越准接受率越高加速越大;最坏退化为接近普通自回归(每轮仍≥1 token,不更慢);上界≈γ+1 倍。vLLM 口径 `mean acceptance length = 1 + 累计接受草稿数/草稿轮数`。
- **三种 vLLM 草稿器(只改"怎么造草稿",验证流程通用不变)**:**N-gram**(零权重,后缀匹配历史复用后续 token,prompt_lookup_min/max 控匹配长度);**Medusa**(单模型多头,在 target hidden state 上加多个头 argmax 出多步候选,ResBlock 权重 0 初始化≈恒等);**Eagle/Eagle3**(轻量 drafter 复用 target 的 hidden states 做"特征外推"预测下一隐状态,共享 LM Head 映射回 token,Eagle3 融合多层 hidden state)。树状注意力(tree attention)用特殊 mask 让每节点只看父路径,一次并行验证多条候选路径提升接受率。
- **工程关键 SpecDecodeMetadata**(`_calc_spec_decode_metadata`):batch 内各请求草稿数不同([3,0,2,0,1]),压成扁平索引一次前向+一次 gather 搞定:`num_sampled=draft+1`;`logits_indices`(hidden_states 空间要算 logits 的位置);`target_logits_indices`(压缩 logits 里验证草稿的行);`bonus_logits_indices`(每段最后一行,全接受时采 bonus)。RejectionSampler(`vllm/v1/sample/rejection_sampler.py`)据此判定接受/拒绝。
- **三个 for 循环的可并行性判据(易混点)**:demo 三阶段都写成 for,但能否并行只看「迭代间有没有数据依赖」,与写没写成 for 无关。① **Drafting** for:第 i+1 次迭代输入依赖第 i 次采样出的 token(`sample`后`ctx.append(tok)`)→**真串行**,草稿器必须串行调 γ 次(故须便宜)。② **Verification** for:`target_dist(ctx)` 的输入是 `context+draft_tokens[0..i-1]`,而草稿在上阶段**已全部生成**→迭代间**无数据依赖**(后一个不吃前一个输出,只吃已知草稿)→**可并行**;CPU 上只能逐个填数组,GPU 上是**一次宽前向**:把 `[context,draft_0..draft_{γ-1}]` 整条喂入,causal attention 保证位置 i 只看 `context+draft[0..i-1]`,一次 forward 同时出 γ+1 个分布。**投机解码省时间的本质=把这种"看着串行实则无依赖"的验证从 γ 次窄前向压成 1 次宽前向**。③ **Accept/Reject** for:逻辑串行(首次拒绝即停),但**不调模型**,p/q 都是已算好的概率查表+一次除法+掷骰子,几乎零成本,不在关键路径;vLLM 里 RejectionSampler 仍把它向量化(布尔比较+前缀操作求截断点)。
- 教学 demo:`vllm_learn/code/course16/spec_decode_demo.py`(纯 stdlib 玩具 bigram 模型,蒙特卡洛实测输出分布==target 偏差仅 ~1e-3,并复刻 DOC 8.3 扁平索引对拍)。原理文档 `course16/DOC.md`。

## [2026-06-13] 采样链路:从 logits 到 next token 的完整流水线
**标签**: llm-inference, sampling, temperature, top-k, top-p, min-p, penalty, constrained-decoding, batch-invariance
完整链路(与 vLLM 顺序一致):`logits → [惩罚/约束 mask] → ÷temperature → top-k → top-p/min-p → softmax → 多项式采样`。被淘汰 token 在 logit 空间统一置 `-inf`,最后一次 softmax+采样。
- **temperature**:`p_i=softmax(z_i/T)`,缩放 logits 对比度。**T→0** 退化 argmax/greedy;**T→∞** 退化均匀分布 `1/|V|`。T<1 更尖(确定),T>1 更平(随机)。
- **三种候选集截断**:**top-k** 按排名留最大 k 个(固定数量);**top-p(nucleus)** 按概率从大到小累加到 ≥p 的最小集合(数量自适应,分布越尖留越少);**min-p** 留 `prob≥min_p·max_prob`(相对最强候选的比例门槛,对温度更鲁棒)。
- **为什么要多层过滤(非冗余)**:四个管**正交的轴**且各有失效场景,叠加=取**交集**互相补盲。① temperature 只缩放不截断,高 T 会把**长尾垃圾 token 概率一并抬高**→必须靠 top-k/p 砍尾,才能"既多样又不胡说"。② top-k 固定数量**不自适应置信度**:分布尖时仍放进垃圾,分布平时又切掉合理候选。③ top-p 自适应形状,但极平分布下可能纳入上千 token 失控、且对温度敏感→用 **top-k 兜一个数量硬上限**。④ min-p 锚定 max_prob,高温下比 top-p 更稳。组成漏斗:`÷T 重塑形状 → top-k 硬数量上限 → top-p/min-p 按质量/相对门槛精修`。截断类先砍掉模型最不可靠的长尾,temperature 再在幸存者里调随机性。投机解码里 target 分布也要先过这条链(temp+top-k/p)得到 `target_logits` 当"标准答案 p",再 `min(1,p/q)` 校验草稿→保证输出分布严格等于大模型按同参数逐 token 采样(见 `rejection_sampler.py`、DOC.md 1575 行)。
- **流水线顺序敏感性(易混点)**:**top-k 对温度顺序不敏感**——温度是单调变换不改排名;**top-p/min-p 对温度顺序敏感**——它们比概率值,温度改概率则 nucleus 集合变。
- **为什么 penalty/mask 放最前(架构级判据:argmax-invariant 与否)**:整条链里**只有 penalty/约束 mask 这一层不是 argmax-invariant**——它会把最大 token 压下去或直接置 `-inf`,**能改 greedy 结果**,故必须在"要不要走 greedy"决策**之前**执行(`sampler.py forward()` 里 `apply_logits_processors` 在 `sample()` 之前无条件跑;greedy 分支取的是过了 penalty/mask 但没过 temperature 的 argmax)。而 **temperature/top-k/top-p/min-p 全是 argmax-invariant**(温度单调不改排名、top-k/p/min-p 永远保留最大 token),绝不改单个最大值是谁→对 greedy 无意义,下沉到 `sample()` 随机分支、greedy 跳过。其余三个放最前的理由:① 它定义"分布内容"而非"怎么采样";② 硬 mask 须早于截断,`-inf` 后概率=0 才**不占 top-k 名额、不计入 top-p 累计质量**(否则禁词污染截断);③ 惩罚须作用在原始 logits 上供下游 top-p/softmax 读取。**惩罚/mask 内容**:硬约束(置 -inf)=allowed_token_ids 白名单/bad_words/grammar mask/min_tokens 禁 EOS;软惩罚(加减 logit)=repetition/frequency/presence + logit_bias。
- **三种 penalty 改的量不同**:**presence**(出没出现过 0/1,扣固定值→鼓励新词);**frequency**(正比出现次数,越多扣越狠→抑制复读);**repetition**(乘性,出现过把 logit 往 0 拉,z>0 变小/z<0 变大,HF 风格)。
- **批量异构采样**:sampler 在模型 forward **之后、GPU 上向量化**执行;不同请求的 temp/top_p/top_k 打包成 per-row 张量对 `logits[batch,vocab]` 逐行广播,不同 seed 各维护独立 RNG→一次 forward 出 logits 再向量化采样,异构请求一起处理无需拆循环。词表 128k+ 时 top-k/p 全排序是瓶颈→用 partial sort/radix-select、阈值二分避免全排。
- **logprobs**:`logprob=z/T−logsumexp(z/T)`,算本身几乎免费;代价在取 top-n+gather/sort+D2H 拷贝,prompt_logprobs(逐位置)更贵。
- **确定性 Q3(2025 热点)**:同 prompt+固定 seed+T=0 在 batch 推理下两次仍可能不一致,根因**不是采样随机数,而是浮点加法不满足结合律** `(a+b)+c≠a+(b+c)`——GPU matmul/reduction 累加顺序依赖 batch 大小/tiling/split-K,batch=1 vs batch=32 归约顺序不同→logits 低位 bit 差异→两 token logit 近并列时 T=0 的 argmax 被噪声翻盘。**batch-invariant 复现**:用固定归约顺序/split 的 batch-invariant kernel(RMSNorm/matmul/attention 全要),代价是牺牲吞吐。
- **约束解码/JSON(Q4)**:= 一个 logits processor,采样前把当前语法状态下不合法 token 的 logit 置 `-inf`,softmax 后概率为 0 绝不被采样;之后 temperature/top-k/p 流程不变。状态机:JSON/正则→FSA,CFG→PDA,每步算"允许 token 的 bitmask"。**xgrammar**(SGLang/vLLM)把 grammar 编译成自动机、高效生成每步 bitmask(缓存状态+字节级 trie 加速)再作为 logits processor 应用。
  - **约束 ≠ 纠错(易混点)**:它是**事前约束生成空间**(非法 token 不可达,错误从源头不产生),不是**事后修复**已生成的坏 JSON。状态机的职责是逐步回答"从当前状态下一个 token 哪些合法",不是识别并改正错误。**模型仍在合法 token 中按自身概率/temperature/top-k/p 选谁胜出**,mask 只删非法候选不改谁赢。**只保证句法合法(合 schema 结构),不保证语义/字段值正确**。采样产出 token 后状态机吃掉它前进、再算下一步 mask;需处理 token 边界/字节对齐。
- 教学 demo:`vllm_learn/code/course16/sampling_demo.py`(纯 stdlib,8 课覆盖温度两极/三种截断/顺序对拍/三种 penalty/批量异构/浮点非结合性导致 argmax 翻盘/logprobs/约束 mask 实测非法 token=0)。

## [2026-06-13] 投机解码进阶:Tree Attention、大 batch 收益塌、KV 回滚、多模态
**标签**: speculative-decoding, tree-attention, eagle, medusa, continuous-batching, kv-cache, paged-attention, vlm, diffusion
承接 [2026-06-07] 投机解码框架,补 Q8–Q10 三个进阶点。
- **链→树 + Tree Attention(Q8)**:链式投机每步草稿是一条 γ 链、首次拒绝即停→接受长度被"最弱一环"卡死。Medusa/EAGLE-2 改成**草稿树**(每位置多候选分支),某分支被拒可换兄弟分支→每步期望产出更高。要在**一次**前向验证整棵树:把树拍扁成序列,用 **tree attention mask 让每个节点只 attend 自己的祖先**(不是 causal 的下三角,否则不同分支会串味),于是一次前向并行验证所有根→叶路径,且**共享前缀只算一次**(省重复计算+省多次 kernel/搬权重)。验证规则同链式(min(1,p/q)),取最长被接受路径,仍无偏。
- **演进线痛点(Q8)**:独立 draft 小模型(要单训/分布不齐)→Medusa(多头无自回归依赖,质量受限)→EAGLE/2/3(**特征级自回归**:drafter 吃 target 的 hidden state,在特征空间自回归再映射 token;分布天然对齐+自身极小→接受率↑且草稿↓,同时优化加速比两因子;EAGLE-2 动态树,EAGLE-3 去掉特征重建约束+多层融合)→Lookahead/Jacobi(不要 drafter,n-gram 池+Jacobi 迭代)。
- **和 continuous batching 共存(Q9,题眼)**:投机解码免费午餐=decode 的**闲置算力**。大 batch 把算力喂饱后(compute-bound)午餐消失:verify 要算 `B×(γ+1)` 个位置耗时∝token 线性涨,却只多产出 τ(<γ+1)倍 token→单位 token 更贵→**加速比跌破 1 净变慢**。roofline:`前向耗时≈max(搬权重时间, token数×单token算时)`,小 batch 卡地板(免费),大 batch 进斜坡(线性贵)。**该开**:低并发/小 batch/延迟敏感/高接受率;**该关**:高并发吞吐优先/低接受率。引擎应**动态开关**(按 batch 大小/GPU 利用率/在线 accept rate 自适应,vLLM 有 `speculative_disable_by_batch_size` 阈值)。
- **被拒草稿的 KV 回滚 + PagedAttention(Q9)**:verify 前向会为草稿位置 d1,d2,d3 都写 K/V 进 KV cache,某个被拒后**回滚=把 `num_computed_tokens` 截断到已接受长度**——O(1) 指针操作,不用逐元素清 KV;脏 KV 下轮被新 token 覆盖,attention 只读到 num_computed_tokens 以内。配合 PagedAttention:写一半的尾块不急释放(下步接着写),接受长度可变(0~γ+1)的变长靠 block_table+扁平索引统一管理。
- **多模态适用性(Q10)**:投机解码绑定的是"**自回归+memory-bound**"这个**结构**而非模态。**VLM(图文理解)适用**——解码端就是 LLM,图像只是 prefix token,decode 仍自回归+memory-bound(EAGLE 吃 target hidden 天然带图像信息,优势大);注意 prefill 重但投机只加速 decode。**标准 Diffusion 不适用**——迭代去噪是对整张图的稠密 compute-bound 计算,无逐 token 串行结构,拒绝采样无从套用;其加速另走减步(DDIM/DPM-Solver)/蒸馏(Consistency/LCM)/缓存(DeepCache)。**AR 式视觉生成(VAR/图像 patch 当 token/视频按帧自回归)又重新适用**。
- 教学 demo:`course16/tree_attention_demo.py`(可打印 tree mask 矩阵+蒙特卡洛验无偏)、`course16/batching_spec_demo.py`(roofline 模型把加速比 2.5x→0.62x 的塌缩算出来+KV 回滚演示)。学习指南 `course16/README.md` 含 Q1–Q10 完整蒸馏答案。

## [2026-06-08] vLLM bench latency 参数:长度 vs 数量别混淆
**标签**: vllm, benchmark, bench-latency, max-num-seqs, input-len
`vllm bench latency` 的参数分两个量纲——「长度」(一条序列多少 token)和「数量」(同时跑几条),易混的是 `--max-num-seqs`(数量,不是长度)。
- **一条 seq 的最终长度 = `--input-len` + `--output-len`**(prompt 长度 + 生成 token 数)。例:`--input-len 512 --output-len 8` → 单条 seq 最终 520 tokens(prefill 进 512,decode 逐步 +1 到 520)。`--max-num-seqs` 完全不进长度计算。
- **`--max-num-seqs N`(数量,引擎调度配置)**:一个 batch 里**最多并发多少条序列**(batch 宽度上限),与"一条多长"正交。
- **`--batch-size M`**:本次 benchmark 实际发多少条请求。`M ≤ max-num-seqs` 时一个 batch 装下全部并发;`M > max-num-seqs` 则拆成多轮调度,但**每条 seq 长度不变**(仍 input+output)。
- 一句话:`max-num-seqs`/`batch-size` 决定"几条并排跑"(宽度),`input-len`/`output-len` 决定"每条多长"(长度);求长度用不到 max-num-seqs。出处 `vllm_learn/code/course18/nsys.sh`(配 nsys profile 抓 vLLM 时间线)。

## [2026-06-05] vLLM 中 TP 与 DP 的正交关系及数据流动
**标签**: vllm, tensor-parallel, data-parallel, orthogonal, all-reduce, all-gather
TP(切模型)与 DP(复制模型)是**正交**的两个维度,可同时叠加,总卡数 = DP × TP;全局编号 `gpu_id = dp_rank * tp_size + tp_rank`。把 GPU 排成二维网格:一行(同 dp_rank)是一个 TP 组=一个模型副本,一列(同 tp_rank)是一个 DP 组(持相同权重切片、处理不同请求)。
- **TP(纵向,组内协同算一次 forward)**:同一份权重切到组内多卡。MLP 经典切法——第一层 `W1[hidden,ffn]` 列并行(按 ffn 维切,各卡算不重叠的隐藏维切片,需要完整维度时 **all-gather** 拼接);第二层 `W2[ffn,hidden]` 行并行(各卡吃自己那段 h 算出 partial sum,组内 **all-reduce(SUM)** 累加得最终输出)。列并行输出恰好对齐行并行输入,故 MLP 常省掉中间 all-gather,只在 W2 后做一次 all-reduce。
- **为什么是 column→row 而非反过来(关键,常被误归因为"维度大小")**:W1 列(`axis=1`)和 W2 行(`axis=0`)切的其实是**同一个 ffn 维**,与"列大切列/行大切行"无关。选这个顺序是让**中间激活 `h=[tokens,ffn]` 沿 ffn 维切片**,换来三点:① **GELU 本地正确**——逐元素非线性 `GELU(a+b)≠GELU(a)+GELU(b)`,列并行每卡得到的是**完整神经元切片**(非部分和),可直接本卡做 GELU 零通信;若 W1 用行并行则每卡只有部分和,必须在非线性**之前**插一次 all-reduce(坏)。② **层间无缝**——列并行输出 `[tokens,ffn/tp]` 正是行并行 W2 所需的 ffn 切片,两层间不需 all-gather。③ **全程仅一次 all-reduce**(在 W2 之后)。反过来 row→column 则 GELU 前被迫通信、通信次数≥2。
- **数学等价**:TP 多卡协同结果与单卡不切分结果完全一致(误差仅浮点级 ~1e-17),代价是显存/算力摊到组内多卡。
- **DP(横向,副本间独立)**:不同副本吃不同请求子集、并行计算、互不干扰,提升并发吞吐;推理时 DP 组的 all-reduce 用于调度协同/wave 同步,不是把不同请求的激活相加。
- 一句话:TP=把一个模型摊开给几张卡一起算(降单卡显存/延迟);DP=把整模型复制几份各算不同请求(提吞吐)。
- 教学 demo:`vllm_learn/code/course12/dp_tp_orthogonal_demo.py`(numpy 真切 MLP 并对拍单卡结果)。

## [2026-06-07] 激活函数 vs 采样:位置不同、阶段不同,几乎无关
**标签**: transformer, activation, sampling, dropout, llm-inference
常见混淆:把 dropout 当激活、把采样和 FFN 内部组件混为一谈。本质区别是**位置**(网络中间 vs 末端)和**阶段**(训练随机性 vs 推理随机性)。
- FFN 四类组件按功能分清:**Linear/MLP**=线性变换(可学习权重);**ReLU/GELU/SiLU**=激活函数(引入非线性,确定性,推理生效);**Dropout**=正则化(随机失活,推理时关闭,**不是激活**);**LayerNorm**=归一化。"激活层"特指激活函数,不是 dropout。
- 术语歧义:"activation/激活值"两义——① 激活**函数**(ReLU 这类);② 任意层的**输出张量**(intermediate activations)。
- 数据流:`token→embedding→[Transformer×N: Attn+FFN(Linear→激活→Linear)]→最终 LayerNorm→LM Head(投影到词表得 logits)→★采样★(logits→token)`。
- **激活函数**在网络中间、每层都有、重复 N 次、确定性,作用=非线性拟合("算"的一部分);**采样**在末端、只发生一次、(除 greedy 外)随机,作用=从词表分布挑 token("选"的一步)。方法:greedy/temperature/top-k/top-p。
- 随机性辨析(易混点):Dropout 随机性只在**训练**(推理关闭);采样随机性只在**推理**。两者不同阶段、不叠加。

## [2026-06-07] LLM 推理采样随机性:greedy/top-k/top-p 谁有随机性
**标签**: llm-inference, sampling, greedy, top-k, top-p, temperature
"采样随机性"=前向算出 logits→softmax 得词表分布后,**最后选 token 那一步是否抽签**(前向本身确定,dropout 已关)。top-k/top-p 不是"随机开关",只是先把候选集裁小,真正随机与否取决于裁完后那步是 argmax 还是抽样。
- **greedy**:取 argmax(概率最高 token),**无随机性**,等价 `temperature=0`。
- **top-k**:只留概率最高 k 个,重归一化后**从中抽样**→**有随机**(在 k 个里按概率抽)。
- **top-p(nucleus)**:按概率降序累加凑够累计 p 的最小集合,**从中抽样**→**有随机**(核内按概率抽)。它们的作用是既保多样性、又过滤长尾垃圾 token。
- **temperature 才是随机总闸**:logits 先 `/T` 再 softmax。`T=0`→one-hot→等价 greedy;`T<1` 分布更尖锐(保守);`T>1` 更平坦(发散)。top-k/top-p 在 T 缩放**之后**裁剪再抽样。
- **退化为确定的边界**:`top-k=1`、`top-p→0`、`T=0` 都退化成 greedy。
- **伪随机可复现**:固定 seed 后即使 top-k/top-p 输出也可复现,但这是"伪随机可复现",分布层面仍是随机采样,≠greedy 的数学确定。

## [2026-06-08] nsys / ncu 性能分析:常用命令与指标避坑
**标签**: profiling, nsys, ncu, nsight, gpu
nsys 看系统级"有没有并行、谁拖时间";ncu 看单 kernel"瓶颈在算还是访存"。常见命令坑:
- **ncu 正则过滤是 `-k regex:gemm`**,不是 `--kernel-regex`(旧写法已在现代 ncu/2022.4 移除)。配 `-s N` 跳预热、`-c 1` 限采样次数。
- **ncu 读硬件计数器要 root**,否则报 `ERR_NVGPUCTRPERM`;`--set full` 会 replay 同一 kernel 采全 section、很慢,快速定位用 `--set basic` 或 `--section SpeedOfLight`。
- **nsys 的 GPU Utilization 采样行需 `--gpu-metrics-device=all`,且 nsys ≥ 2021.2**;旧版(如 2020.4)没有此选项,时间线"利用率"只是按 kernel 覆盖时间估算,非硬件计数器。
- **`-t nvtx` 只有代码里有 `torch.cuda.nvtx.range_push/pop` 才采得到东西**;否则该 trace 为空。
- **SOL 的 Memory% 是各存储子系统(DRAM/L1/L2/共享内存)取最大值**,不一定等于 DRAM 带宽,具体看 Memory Workload Analysis。
- 判瓶颈三态:DRAM 高/SM 低=Memory Bound;SM 高/DRAM 低=Compute Bound;两者都低=Latency Bound(调 Occupancy/减依赖)。
- bash 坑:`\` 续行块中间**不能插 `#` 注释**,会提前终止命令(注释吃掉后续续行,下一行被当成新命令执行)。

## [2026-06-14] Roofline 模型:用算术强度判 compute/memory-bound
**标签**: roofline, arithmetic-intensity, compute-bound, memory-bound, performance-model, llm-inference
Roofline 是性能**上界**模型:把"快不快"归结为卡算力还是卡带宽。三量——计算量 `W`(FLOPs)、访存量 `Q`(Bytes)、**算术强度 `I=W/Q`**(每搬 1 字节换几次计算,kernel 固有属性、与硬件无关)。
- **核心公式**:可达性能 `P = min(π, β·I)`,π=峰值算力(FLOP/s)、β=峰值带宽(Byte/s)。等价时间式 `T = max(W/π, Q/β)`(算与搬可重叠,取较大者)——这就是笔记里 `前向耗时≈max(搬权重时间, token数×单token算时)` 的出处。
- **脊点(拐点)** `I* = π/β`:硬件固有平衡点(A100 FP16: 312TFLOPS/2TBps ≈ **156 FLOP/Byte**)。`I<I*` → 在斜坡 → **memory-bound**(优化方向=减访存/提复用);`I>I*` → 在屋顶 → **compute-bound**(优化方向=提算力利用率/上 Tensor Core)。利用率 ≈ `I/I*`。
- **解释 LLM 推理三连**:① **decode memory-bound**——每步 1 token,GEMV 把整套权重+全部 KV 读一遍,`I≈1`≪156(算例:A100 decode 利用率仅 ~0.6%),算力大量闲置→投机解码"免费午餐"、W4A16 提速、PD 分离的共同根因;② **prefill/大 batch compute-bound**——权重读一次服务大量 token,`I` 高爬上屋顶→W4A16 无算力红利、要 W8A8/FP8 吃 Tensor Core;③ 大 batch 推高 `I` 从斜坡爬到屋顶→投机解码午餐被吃没。详见 [[2026-06-07]] 投机解码、量化条、PD 分离条。
- **边界**:它是上界不是预测值,假设访存/计算完全重叠,忽略 latency(小 kernel 启动)、cache、同步、带宽打不满→真实点恒在屋顶下。价值=指出"瓶颈在哪个维度+天花板在哪",非精确测时。进阶版加多重屋顶(L2 带宽斜坡、无 FMA 算力屋顶)。

## [2026-06-08] 用 nsys/ncu 抓 vLLM 真实 kernel:CUDA Graph 是头号坑
**标签**: profiling, vllm, nsys, ncu, cuda-graph
vLLM 默认开 CUDA Graph,kernel 被整图重放→profiler 抓不到图内 kernel 名。实测对 `vllm bench latency` 的 nsys 报告做 `gpukernsum`,99.8% 时间落在 `[Unknown]`,只有个别没进图的 GEMM(实测 `cutlass_80_wmma_tensorop_bf16_...gemm`)漏出名字。
- **两种解法**:① ncu 分析必须 `--enforce-eager` 关 Graph(ncu 无法 profile 图重放的 kernel);② 想留 Graph 真实性能则给 nsys 加 `--cuda-graph-trace=node` 展开图内节点还原名字。
- **流程**:nsys `--cuda-graph-trace=node` + `nsys stats --report gpukernsum xxx.qdrep` 找 Top1 kernel 名 → 截稳定子串填 ncu `-k regex:...` 精分析。`-s` 给够大以跳过 vLLM 大量预热/图捕获 launch;`sudo -E` 保留 `CUDA_VISIBLE_DEVICES`(裸 sudo 丢环境跑错卡)。
- **nsys stats report 名随版本变**:2020.4 是 `gpukernsum`/`cudaapisum`/`gpumemtimesum`;≥2021.2 才是 `cuda_gpu_kern_sum`/`cuda_api_sum`/`cuda_gpu_mem_time_sum`。
- **读结果**:decode 阶段 attention 多为 Memory/Latency Bound(算术强度低、SM 低);prefill/大 batch 下 GEMM 偏 Compute Bound(看 TensorCore 利用率)。这正是 vLLM 把 prefill/decode 分开调度的根因。

## [2026-06-09] nvcc 查 kernel 寄存器/smem 用量:-Xptxas -v(别带 -abi=no)
**标签**: cuda, nvcc, ptxas, occupancy, registers
编译期查每个 kernel 的寄存器/共享内存占用(用于预判 Occupancy),正确写法是 `nvcc -Xptxas -v xxx.cu -o out`;老资料里的 `-abi=no` 已被新版 ptxas 移除。
- 报错 `ptxas error : Invalid value 'no' for option -abi` = 你传了 `-Xptxas -v,-abi=no`,而该 CUDA 版本(实测 10.1)的 ptxas 根本没有 `-abi` 这个可传 yes/no 的开关(`ptxas --help` 里只剩"ABI 最小寄存器数"说明)。去掉 `-abi=no` 即可。
- `-Xptxas -v` 输出形如 `ptxas info : Used 32 registers, 2048 bytes smem, 360 bytes cmem[0]`,这是算 occupancy 要的核心数据;`--ptxas-options=-v` 是等价写法。
- 配套:`-Xptxas -v,-warn-spills` 额外提示寄存器溢出到 local memory(occupancy 杀手);`-maxrregcount=N` 或 kernel 上加 `__launch_bounds__()` 可直接控寄存器上限做实验。
- `-abi=no` 是老技巧(关 ABI 看"裸"寄存器数),新版改用内联(`__forceinline__`/`static`)达到类似目的。

## [2026-06-09] 验证 GPU Occupancy:ncu 实测 + Occupancy API 兜底,及两大静默坑
**标签**: cuda, occupancy, ncu, nsight-compute, ampere, shared-memory
验 occupancy 两条路:ncu(需 root,给 Theoretical+Achieved+限制因子)和 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`(无需权限,只给理论值,适合无 root/CI)。两个静默坑会让你以为"复现成功"实则 kernel 根本没跑。
- **坑1:Ampere 不是一档,RTX 3090 ≠ A100**。compute 8.0(A100)= 2048 线程/SM = **64 warps**;compute 8.6(RTX 3090/30 系)= 1536 线程/SM = **48 warps**。同一个"1 block(8 warp)/SM"的 demo,A100 occupancy=8/64=12.5%,3090=8/48=**16.7%**。很多教程默认写 64 warps/12.5%,在 3090 上是错的。smem opt-in 上限两者都是 100KB。
- **坑2:>48KB 动态共享内存必须 opt-in,否则 launch 静默失败**。Volta/Ampere 单 block 动态 smem 默认上限仅 48KB,要用 64KB 必须先 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`;否则 kernel 启动返回 `invalid argument` 根本不运行。若代码没查返回值会照常打印"运行结束",而 ncu 报 `No kernels were profiled`。务必加 `CUDA_CHECK` + `cudaGetLastError()`。
- **ncu 命令**:`ncu --section Occupancy --section LaunchStats -c 1 ./prog`,读 `Theoretical Occupancy`/`Achieved Occupancy`,限制因子看 `Block Limit {Shared Mem,Registers,Warps,Blocks}` **取最小的那项**(本例 Shared Mem=1 → 被共享内存卡死)。
- **ncu 要 root**:非 root 报 `ERR_NVGPUCTRPERM` + `No kernels were profiled`;用 `sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" ncu ...` 保留环境,或改驱动参数 `NVreg_RestrictProfilingToAdminUsers=0`。
- **无 root 兜底**:`cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, blockSize, dynSmem)` 拿每 SM 活跃 block 数,× warps/block ÷ (maxThreadsPerMultiProcessor/warpSize) = 理论 occupancy,与 ncu 的 Theoretical 对得上。
- 出处 demo:`vllm_learn/code/course18/ncu_use/low_occupancy_smem.cu` + `warp_stall.md`。

## [2026-06-12] LLM 推理量化:weight-only vs weight+act,及"量化≠必然变快"
**标签**: quantization, vllm, gptq, awq, smoothquant, fp8, memory-bound, decode
量化省的首先是**显存/访存带宽**,不一定省算力;"变快"是派生收益,只在「瓶颈正好是被量化省掉的资源」时兑现——用 memory-bound/compute-bound 这把尺子先判断阶段。
- **两大流派**:① weight-only(W8A16/W4A16,GPTQ/AWQ)只压权重、激活保 fp16,算时反量化回 fp16 → 省显存带宽、**无算力红利**、**decode(memory-bound)受益**。② weight+act(W8A8/FP8,SmoothQuant)权重+激活都压 → 能用 INT8/FP8 Tensor Core 算力、**prefill(compute-bound)受益**,但激活难量化。
- **第③问题眼**:W4A16 在 decode **会变快**,快在"权重少搬到 1/4"(decode 算力本就闲置,瓶颈在搬权重),**不是算得快**;prefill 是 compute-bound,W4A16 不给低精度算力 → **基本不快只省显存**。模型太小/batch 太大/硬件不匹配时,dequant 开销会吃掉收益。
- **权重 vs 激活的数值分布(纠误解)**:权重是**零均值近高斯**、std 仅 ~0.01–0.05、绝大多数落 [-0.1,0.1],但**不是都<1**(尾部少量到 0.5+/偶有>1);各通道尺度均匀 → 对称+per-channel 量化贴合好。激活/embed 是 **fp16 浮点数,不是 int8**(int8 是量化*目标*格式,需 `scale=max|x|/127` 才映射进去);embedding 那一刻较温和近高斯,但残差流逐层累加→**越深层离群通道越明显**,过 SiLU/ReLU 后还偏单边(故激活用非对称量化)。
- **激活为何难量化**:有**系统性离群通道**(固定维度大几十倍)、动态变化、不能裁;权重乖且静态。解法 **SmoothQuant** 按通道恒等迁移 `x/s @ (W*s)ᵀ==x@Wᵀ`,`s_j=max|x_j|^α/max|W_j|^(1-α)`,把难度从激活搬到权重;FP8 因浮点格点非均匀、动态范围大,天生更耐离群。
- **层间混合精度**:大 Linear(attn+ffn,占参数 ~92%)是量化主战场;embedding/lm_head/RMSNorm/router 小而敏感,保 fp16(llmcompressor recipe 里 `ignore=["lm_head"]`)。层内还有 per-tensor/per-channel/per-group(128) 粒度旋钮,常比层间更影响误差。
- **vLLM 实操**:在线 `LLM(quantization="fp8")`(只支持无需校准的 fp8);预量化 GPTQ/AWQ/compressed-tensors checkpoint 自动识别;`kv_cache_dtype="fp8"` 正交可叠加。**RTX 3090(sm_86)无原生 FP8**,vLLM 退化成 weight-only FP8 走 Marlin(日志有警告)→ 省显存但 prefill 不快;实测 Qwen3-0.6B fp8 权重 1.12→0.72 GiB、KV 109k→113k tokens,但 decode 吞吐几乎持平(模型太小+无 FP8 算力)。
- 出处 demo:`vllm_learn/code/quant/`(DOC.md + 01_quant_numerics / 02_layer_by_layer / 03_weight_only_awq / 04_activation_smoothquant / 05_vllm_quant)。

## [2026-06-13] AWQ 如何"发现"显著通道:看激活,不挑 top-k,连续缩放
**标签**: quantization, awq, salient-channel, activation-aware, weight-only
AWQ 找显著通道 = 在一小批校准数据上算每个输入通道的平均激活幅度 `act_scale = x.abs().mean(dim=0)`(shape `[in_f]`),幅度大的就是显著通道;但它【不】离散挑 top-k,而是把幅度映射成连续 per-channel 缩放因子,显著性靠 `s_j` 大小自动生效。
- **判据是激活不是权重**:输出 `y=x@Wᵀ`,某列权重量化误差 `ΔW[:,j]` 对输出的贡献被 `x[:,j]` 放大;故激活大的列必须保护,权重幅度 `|W|` 几乎无关。AWQ 论文实测按 `|W|` 选≈没选。
- **不挑 top-k,是连续曲线**:代码无 `topk`/`if j in salient`。`s = act_scale^α; s = s/s.mean()`(α=0.5,真实 AWQ 用 grid search 搜 α)。激活越大 `s_j` 越大、保护越多;普通通道 `s_j`≈1 甚至<1(让出精度)。"挑出显著通道"通过 `s_j` 连续大小体现。
- **恒等变换**:`(x/s)@(W*s)ᵀ == x@Wᵀ`,输出数学不变,但放大后的显著列 int4 量化相对误差变小,误差被"挪"到不重要通道。
- **demo 里的 `salient` 列表是造测试数据用的**(手工往 8 个通道塞大激活 ~40),不是算法的一部分;真实场景靠校准激活统计发现。实测:RTN 误差 11.7% → AWQ 1.9%;可视化显示预埋的 8 个通道被准确发现,`s_j`≈18–21x,中位通道 `s_j`≈0.96x。
- 出处 `vllm_learn/code/quant/03_weight_only_awq.py`(含 Top-k 显著通道 + `s_j` 放大倍数可视化打印)。

## [2026-06-13] 量化里的静态 vs 动态:迁移因子 s 永远静态,动态的是激活量化 scale
**标签**: quantization, awq, smoothquant, static-dynamic, per-token, calibration
常见误解:以为 AWQ/SmoothQuant 的迁移/缩放因子 s 随输入序列变化(换个 prompt 重新选显著通道)。实际上 s 是【离线一次性算好并冻结】的,与请求无关;量化里真正"动态"的是 W8A8 激活的量化 scale,不是 s。
- **s 是静态、且 fold 进权重**:用 ~128 条校准样本统计 `act_scale` 算出 s 后冻结。`W*s` 预先算好存进 checkpoint;激活侧 `x/s` 不在 forward 现除,而是 fuse 进上一层算子(LayerNorm scale / 上一个 Linear 权重)→ 推理时零额外计算,"算 s/除 s"已消失在权重里。demo 里 `s=...` 那几行对应离线量化阶段,不是 forward 的一部分。
- **为什么能静态(根因)**:显著/离群通道是【结构性】的——持续出现在固定特征维度上,跨 token、跨序列稳定(不是这句话第5个token大、那句第8个大)。故小校准集能泛化到所有输入;若显著通道随序列乱跳,离线校准就失效、整套方法不成立。demo 写死 `outlier_ch=[...]` 且所有 token 都 +30 正是还原此"通道级、跨 token 稳定"性质。
- **真正动态的是激活量化 scale `=max|x|/127`**:① AWQ 是 weight-only(W4A16),激活保 fp16,根本没有运行时激活量化,连动态 scale 都没有,唯一数据相关量 s 也冻结。② SmoothQuant 是 W8A8,激活要运行时压 int8,这个 scale 才有 static(离线校准、快但怕漂移)vs dynamic per-token(每 token forward 现算 max|x|、更准、多一次 reduce)之分;vLLM 的 W8A8/FP8 默认常用 per-token 动态激活量化。
- 一句话:随输入变的是 W8A8 激活量化 scale(且仅动态模式);识别显著通道 + 迁移因子 s 始终静态/离线/冻结,两件事。
- 出处 `vllm_learn/code/quant/03_weight_only_awq.py`、`04_activation_smoothquant.py`。

## [2026-06-13] CUDA Graph 原理与正确用法（含 demo 计时陷阱）
**标签**: cuda-graph, vllm, inference, kernel-launch, benchmark, decode
CUDA Graph 省的是【CPU 端 kernel launch / 调度开销】，不是 GPU 算力；把一串固定的设备侧操作 capture 成静态 DAG，之后一次 `replay()` 重放，免去逐个 launch。收益只在 launch 开销占比大时显现。
- **何时有收益（Amdahl）**：launch-bound 才快。实测同一 SimpleGPT2 decode：batch=1 → **2.26x**，batch=64 → 1.02x，batch=128 → 1.01x（GPU 算力一旦占主导，graph 几乎无用）。单个大 matmul（1 个 kernel）也几乎无收益；要看到加速得用「小张量+几十个串行小 kernel」。这正是 vLLM 只对 decode（FULL graph）、小 batch 最受益的根因。
- **计时陷阱（最常见 demo bug）**：CUDA 异步，`time.time()` 不夹 `torch.cuda.synchronize()` 只测到 launch 派发耗时——`replay()` 会在 GPU 没算完就返回，于是得出 0.0004s vs 0.0206s（≈50x）这种**假**加速比。正确：warmup + sync + 多次迭代取平均；真实加速 1.5~3x。
- **capture 五条铁律**：① 输入/输出/中间缓冲区**捕获前预分配**，replay 复用同一批显存地址；② 喂新数据只能**in-place 写回固定缓冲区**（`copy_`/`fill_`/`normal_`），`x = new_tensor` 换了地址 → graph 仍读旧地址 → 拿到旧值（实测：fill_(2) 后 replay 得 4✓；再 `x=full(5)` replay 仍是 4✗）；③ capture 期间**禁止任何主机同步**：同步 `copy_`(non_blocking=False)、`.item()`、`.cpu()`、`synchronize()` 都报 `operation not permitted when stream is capturing`；④ shape/stride/dtype/执行路径必须静态，动态 shape 靠分桶；⑤ 多流要同步关系清晰。
- **要喂主机数据的正确写法**：host 张量 `pin_memory()` + `copy_(..., non_blocking=True)`（可捕获的 `cudaMemcpyAsync`）；要结果回主机则**等 replay 结束后再统一 D2H**，不要放进图。注意 **capture 失败会污染本进程 CUDA 状态**，后续 capture 连锁报 `captures_underway INTERNAL ASSERT`——对照「错误 vs 正确」要放到独立子进程跑。
- **capture 不提交结果**：`with torch.cuda.graph(g): z=...` 只记录操作，**首次 replay 前 z 内容是未定义的**（实测读到 0），要 replay 一次 z 才有意义的值。
- **动态 batch → 分桶+padding**（vLLM `bs_to_padded_graph_size`）：只为少数档位 [1,2,4,8,16,32] capture；查表把任意 bs 向上取整到最近档（5→8, 9→16），真实行填数据、padding 行填占位，只取前 bs 行结果。capture 顺序**从大到小**并共享 `g.pool()` 内存池省显存。
- **warmup 要在 side stream**：capture 前在 `torch.cuda.Stream()` 上预热若干次（`s.wait_stream(current)` → 跑 → `current.wait_stream(s)`），让 cuBLAS/cuDNN 懒加载和 autotune 在捕获外完成，否则捕获不稳或把首次开销固化进图。
- **profiler 联动**见 [2026-06-08 nsys/ncu] 条：graph 内 kernel 名被吞，ncu 需 `--enforce-eager`，nsys 需 `--cuda-graph-trace=node`。
- 出处 demo（已修计时/预热/正确性 bug）：`vllm_learn/code/course9/` 的 cuda_graph.py(eager vs graph 基线)、cuda_graph_gpt.py(decode 各 batch 加速比)、cuda_graph_input.py(地址 vs 值)、cuda_graph_blocking.py(失败+修法)、cuda_graph_padding.py(分桶 padding)。

## [2026-06-14] vLLM 调度器:num_computed_tokens vs get_num_new_matched_tokens
**标签**: vllm, scheduler, prefix-cache, kv-connector, chunked-prefill, num-computed-tokens
两者分属不同层次:`num_computed_tokens` 是**请求自身的状态计数器(int)**,`get_num_new_matched_tokens` 是 **KVConnector 的回调钩子**;调度器(`v1/core/sched/scheduler.py`)里前者由后者(等)喂数,再决定"还剩几个 token 要真前向"。承接 [[2026-06-13]] PD 分离。
- **num_computed_tokens(Request 上的游标)**=该请求前缀里**已有多少 token 的 KV 算好/复用好**。驱动 `num_new_tokens = request.num_tokens - num_computed_tokens`(决定 chunked prefill 切块、decode 推格)。**调度即推进**:`_update_after_schedule` 里 `+= num_scheduled_token`(:964),好让下步立刻接着切;**抢占清零** `=0`(:941)整段重算;**投机草稿被拒按接受数回调**(KV 回滚=截断它)。`is_prefill_chunk = num_computed_tokens < num_tokens(+placeholders)` 用来分 prefill/decode 阶段。
- **get_num_new_matched_tokens(KVConnectorBase_V1 抽象方法,base.py:454)**=在本地已算的基础上,**外部 KV 缓存(远端 P 节点/CPU-SSD 卸载/分布式 KV 池)还能再加载多少 token 的 KV 免本地重算**。签名 `(request, num_computed_tokens) -> (ext_tokens|None, load_async)`:返回 `None`=connector 还没判断好→调度器跳过该请求稍后再问(:605);`load_async=True`=跨步异步搬 KV,本步 `num_new_tokens=0` 不分配新计算;第一元素为 0 时 async 必须 False。仅**首次调度**(`num_computed_tokens==0`)查询一次。
- **三层前缀复用(关键链路,:591-624)**:① 本地前缀缓存 `get_computed_blocks` → `num_new_local_computed_tokens`;② 外部 connector `get_num_new_matched_tokens(req, 本地命中数)` → `ext_tokens`;③ 合并 `num_computed_tokens = 本地 + 外部`;④ 真要算 `num_new_tokens = num_tokens - num_computed_tokens`。**即 connector 返回值与本地命中数相加构成 num_computed_tokens**。注意传给 connector 的入参只是本地命中数,connector 应只匹配"本地没覆盖到的更长前缀"。
- **配套钩子**:命中后 `update_state_after_alloc`(分配块给外部 KV 落地,async 时可能调两次);async 路径请求先停 WAITING、KV 收完 `num_computed_tokens>0` 再正式调度(:634-639)。
- 对照表:num_computed_tokens(Scheduler 全程读写/会推进回退清零/定 num_new_tokens) vs get_num_new_matched_tokens(各 connector 实现/仅首次查一次/报外部可复用数喂给前者)。demo:`course19/{kv_handoff_demo,block_remap_demo}.py`。

## [2026-06-13] vLLM PD（Prefill/Decode）分离原理
**标签**: vllm, pd-disaggregation, kv-cache, kv-connector, p2p-nccl, xpyd, inference
PD 分离把推理的两个阶段拆到不同硬件上：**prefill 算力受限(compute-bound)、decode 显存带宽受限(memory-bound)**，对硬件最优形态相反；唯一需要交接的中间产物是 **KV cache**，交接好后 decode 对「KV 来自本机还是远端」完全透明，输出与单机逐 token 一致。
- **为什么分（roofline）**：prefill 一次吃整段 prompt，算术强度高→吃 TFLOPS；decode 每步只算 1 token 却要把整套权重+全部历史 KV 读一遍，算术强度极低→吃显存容量/带宽，且访存量随上下文长度线性增长。混部时 prefill 长计算拖慢 decode 的 TPOT、decode 的 KV 常驻又压缩 prefill 的 batch，两个维度同时抢资源。小模型/短 prompt/低并发则不值得（KV 传输有成本）。
- **三服务架构**：API Proxy(纯 CPU,路由) + P 节点(高算力卡,kv_producer) + D 节点(大显存卡,kv_consumer)。控制面=ZMQ(握手/元数据/服务发现)，数据面=NCCL(GPU→GPU 零拷贝直传 KV)。
- **6 步流程**：Proxy 把请求复制一份 `max_tokens=1` 发给 P(只触发 prefill)→P 算完 KV 主动发 D→Proxy **丢弃** P 响应→Proxy 把**原始**请求(完整 max_tokens)发 D→D 注入 KV 跳过 prefill 续 decode→D 流式返回。
- **request_id 编址(去中心化路由)**：Proxy 生成 `___prefill_addr_{P_zmq}___decode_addr_{D_zmq}_{uuid}`，P 算完 KV 自己 `parse_request_id` 就知道发给哪个 D，无需 Proxy 二次通知；xPyD 轮询 `count % len`，P↔D 每对只需 world_size=2 的 NCCL 组→增删实例无需重启。
- **远端块↔本地块怎么对号**：靠**逻辑块顺序**，不靠物理块号。P 的 block_ids(如 [2,7,5]) 与 D 的(如 [5,9,11]) 完全不同；P 侧 `extract_kv_from_layer` 按 block_ids **gather** 成连续张量(不含任何物理块号)→NCCL→D 侧 `inject_kv_into_layer` 按自己的 block_ids **scatter** 写回。布局 FlashAttn=`layer[:,block_ids,...]`、MLA/FlashInfer=`layer[block_ids,...]`。
- **get_num_new_matched_tokens 的 -1**：D 从外部能拿 `len(prompt)-1` 个 token 的 KV，最后 1 个 prompt token 在 D 上补算一次前向以得到预测首个生成 token 的隐藏态，代价仅 1 token。
- **KVConnectorBase_V1 双侧接口**：同一 Connector 拆 SCHEDULER role(get_num_new_matched_tokens/update_state_after_alloc/build_connector_meta/request_finished，只碰元数据) 与 WORKER role(start_load_kv 注入/save_kv_layer 发送/wait_for_save/get_finished，真搬数据)，靠 KVConnectorMetadata 桥接。**同接口既支撑 PD 分离(跨网络 P2pNcclConnector)也支撑 KV Offload(跨 PCIe 到 CPU 的 OffloadingConnector)，底层同源**。
- **send_type**：PUT_ASYNC(专用线程异步,计算/通信重叠,最快) > GET(D 主动拉) > PUT(同步阻塞)。kv_buffer_size 经验值≈GPU 显存 10%；P(生产者,PUT 模式)可设极小(1e1)，D(消费者)要充足(如 8e9)；溢出由 TensorMemoryPool(CPU Pinned,伙伴分配)兜底防 OOM。
- 源码：`vllm/distributed/kv_transfer/kv_connector/v1/{base.py, p2p/p2p_nccl_connector.py, p2p/p2p_nccl_engine.py, p2p/tensor_memory_pool.py}`、proxy `examples/disaggregated/p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`。
- 教学材料(零依赖可运行 demo)：`vllm_learn/code/course19/` 的 pd_why_demo.py(roofline)、kv_handoff_demo.py(玩具 Transformer 证明 PD 与单机逐 token 一致)、block_remap_demo.py(gather/scatter 对号)、proxy_xpyd_demo.py(6 步流程+xPyD 路由)，长文 DOC.md，导读 README.md。

## [2026-06-14] PagedAttention 的非连续物理块如何参与计算(不拼连续、不走 cuBLAS)
**标签**: vllm, paged-attention, kv-cache, block-table, gemm, decode, nccl, tensor-parallel
常见误解:以为分页 KV(物理块不连续)要先 gather 拼成一块大连续 tensor 再送 cuBLAS GEMM,decode 时还以为靠 NCCL 收集全部 num_computed_tokens 的 KV 再组装。**两者都反了**——PagedAttention 的全部意义就是【不物化连续 KV】,且 KV 在实例内从不走 NCCL 收集。承接 [[2026-06-13]] PD 分离、[[2026-06-04]] vLLM 安装编译的 _C/FA 算子。
- **只有 attention 碰 paged KV,其余 GEMM 全在连续激活上跑**:一个 layer 里 QKV proj / o_proj / MLP(gate/up/down) 吃的都是当前这拍 token 的隐藏态 `[num_tokens,d]`(连续),普通 cuBLAS GEMM,与分页无关。唯一碰分页、物理不连续 KV 的是 attention 一处。时序:QKV proj 先算出当前 token 的连续 K/V → `reshape_and_cache` kernel 按 `slot_mapping` **scatter 写进** paged cache 物理槽位 → attention 再读整段历史 KV。
- **非连续怎么算 = block_table 在 kernel 内部逐块 gather,不拼连续、不调 cuBLAS**:attention 是【自定义融合 CUDA 核】(vLLM `paged_attention`,或 FlashAttention/FlashInfer 的 paged 变体,出自 `_C`/`_vllm_fa*_C`),多吃一个 `block_table`(逻辑块号→物理块号映射)入参。kernel 内 `for 逻辑块 i: phys=block_table[i]; load cache[phys]; QK^T→online-softmax→累乘 V`。"非连续"被 block_table 一层间接寻址吸收,gather 发生在寄存器/共享内存层,**全程不在显存物化大连续 KV 张量**;Q@K^T 与 softmax@V 是 kernel 内小块矩阵乘,不是一次 cuBLAS 大 GEMM。
- **与 PD 分离"gather 成连续"不矛盾**:PD 那个 gather 是【跨节点传输前】为走 NCCL 才临时拼连续,到 D 节点又 scatter 回各自物理块;**真正算 attention 时仍是 paged 不连续**。传输 ≠ 计算。
- **decode 没有"NCCL 收 KV 再组装"**:实例内 KV 从不跨卡 NCCL 收集。两种 NCCL 角色别混:① **TP**:把 attention 的 head 切到不同卡,每卡只持自己 head 的 KV 在本地 paged cache,decode 各算各 head,**KV 全程不动**,NCCL 只在 o_proj/MLP 后 all-reduce【激活】`[num_tokens,d]`;② **PD 分离**:prefill→decode 节点【一次性】交接 KV,只在请求进 decode 前,不是 decode 每步。
- **decode 真实流**:`hidden[1,d]`(连续)→QKV proj(GEMM)→`reshape_and_cache` 把新 k/v 写本地尾块→paged attention kernel(q 读本地分页不连续的全部历史 KV,block_table 间接寻址 kernel 内 gather)→o_proj(GEMM)→[TP]all-reduce 激活→MLP(GEMM)→all-reduce。**无任何一步把 num_computed_tokens 的 KV 收集组装成大连续 tensor 再送 GEMM**,设计初衷恰是不做此事。

## [2026-06-14] Qwen3 dense:transformer blocks 之后到 logits 只有两层(head 结构)
**标签**: qwen3, llm-arch, rmsnorm, lm-head, tie-embeddings, logits, sampling
承接 [[2026-06-13]] 采样链路(那条讲 logits→token,本条补它的上游:hidden→logits)。N 层 decoder block 之后,到 logits 只剩**两层**:`hidden[B,S,H] → model.norm(最终 RMSNorm) → lm_head(Linear H→vocab,无 bias) → logits[B,S,vocab]`,再接采样流水线。
- **block 内构成**(Qwen3):RMSNorm → Attention(带 **QK-Norm**,对每个 head 的 Q/K 各做一次 RMSNorm) → 残差 → RMSNorm → SwiGLU FFN → 残差;pre-norm 架构。
- **第一层 model.norm = 最终 RMSNorm**:pre-norm 架构残差一路累加,尺度不受控,所有 block 之后必须收尾归一化再投影。Qwen3 用 RMSNorm(无 bias、不减均值),非 LayerNorm。
- **第二层 lm_head = Linear(H→vocab),无 bias**:把每个位置隐状态投影成整个词表打分=logits。
- **权重绑定(tie_word_embeddings)**:Qwen3 小模型(0.6B/1.7B/4B)`lm_head.weight` 与输入 `embed_tokens.weight` 共享同一张表;大模型(8B/14B/32B)不绑定,lm_head 独立权重。
- **推理关键优化「只取最后位置」**:自回归 decode 下一个 token 只依赖最后位置的 logits,故只对 `hidden[:,-1,:]` 过 lm_head,避免对整条序列做 `[B,S,H]@[H,V]`(V≈150k)的大矩阵乘;prefill 不要 prompt_logprobs 时同理只算最后位置。
- 一句话:**…→第 N 层 block→RMSNorm→lm_head→logits→采样**,中间仅「一个 RMSNorm + 一个 Linear」两层 + 推理时只取最后位置这个工程优化。

## [2026-06-15] AR耗时 = 自回归 decode 阶段耗时(及 TTFT/TPOT 指标口径)
**标签**: llm-inference, ar, decode, prefill, ttft, tpot, latency-metrics
"AR耗时"= AutoRegressive(自回归)解码阶段的累计耗时,指模型逐 token 串行生成、从第 2 个 token 直到 EOS/max_tokens 的所有 decode step 时间总和;与 prefill(预填充)分属推理两阶段。承接 [[2026-06-14]] Roofline、[[2026-06-07]] 投机解码。
- **两阶段对照**:prefill 一次 forward 并行处理整个 prompt(N token)→compute-bound、产出第 1 个 token;decode/AR 每次 forward 只生成 1 token、要把整套权重+全部历史 KV 读一遍→memory-bound、算术强度 `I≈1`≪脊点。
- **为什么单拎出来**:生成 M 个 token 要做 M−1 次**串行** forward 无法并行→AR 耗时通常是长输出场景端到端延迟的大头,且 memory-bound 导致 GPU 算力大量闲置——这正是 continuous batching / 投机解码 / W4A16 / PD 分离共同攻击的目标。
- **指标口径**:`TTFT`(Time To First Token)≈ prefill 耗时;`TPOT/ITL`(每输出 token 时间)≈ AR耗时 / 生成 token 数;`总延迟 ≈ TTFT + AR耗时`。优化 AR 耗时 = 压 TPOT 或压 decode 步数(投机解码一次出多 token)。

## [2026-06-16] 网卡改 MTU/上 Jumbo Frame:验证命令的硬伤与 SRE 踩坑
**标签**: networking, mtu, jumbo-frame, pmtud, sre, rollback
改 MTU(如 1500→8500)的核心风险是 **PMTUD 黑洞**(握手成功、大传输 stall),但更隐蔽的是"上线前验证"本身存在物理上做不到的死结,以及改网络配置的纪律缺失。
- **先有鸡蛋问题(最易踩)**:`ping -M do -s <payload>` / `tracepath` 在**本机网卡还是 1500** 时,内核见包>本地 MTU+DF 直接本地返回 `Message too long`,**包根本不出网卡**;tracepath 也被本地接口 MTU 封顶。→ **1500 网卡探不出 8500 路径是否支持**,"只读预证路径"是伪命题。解法:从已是 jumbo 的对端反探(探测方 MTU 必须 ≥ 待测值),或临时升 MTU 探完即退。
- **探测三盲区**:① 单向——`ping`/`tracepath` 只测去程,回程小 MTU 链路一样黑洞,**两端必须互探**;② ICMP≠TCP——中间盒对大 ICMP 和大 TCP 区别对待,ping 通不代表大 TCP 段能过,真实验证要 `iperf3 -c X -M 8460` 逼出大段;③ offload 掩盖——TSO/GSO/GRO 开着时 tcpdump 看到聚合段非真实线上帧,探测期临时 `ethtool -K eth0 tso off gso off gro off`。
- **关键观测命令**:`ss -tin dst <IP>` 看 advmss/pmtu/retrans(别 `grep` 会丢连接归属);`ip route get <IP>` 看缓存 PMTU;`nstat -az | grep -iE 'frag|reasm.*fail'` 看分片/重组失败(飙升=分片在毒害);NAT 机加 `cat /proc/sys/net/netfilter/nf_conntrack_count`、`iptables -t mangle -S | grep TCPMSS` 确认 clamping 已就位。
- **回滚头号纪律(最大缺口)**:你很可能正通过这块网卡 SSH 上去改它,flap/黑洞会让你失联且无法回滚。改之前就挂 **dead-man 自动回滚**:`( sleep 600 && ip link set eth0 mtu 1500 )&` 记 PID,确认 OK 再 `kill`;或确保**带外通道**(IPMI/串口/console)。回滚还要 `ip route flush cache` + drain 旧 PMTU 长连接。
- **SRE 环境放大器(决定爆炸半径)**:① **健康检查盲区**——LB 探活/心跳全是小包,黑洞期间全绿,必须加大包合成探针 canary;② **容器/K8s**——cni0/docker0/flannel/veth 的 MTU 不随物理网卡联动,pod 内失配更隐蔽(NAT 机常就是做 SNAT 的 k8s node);③ **云 VM**——虚拟网络有硬上限(AWS VPC 9001/GCP 8896,跨 VPN/peering/IGW 退 1500)且 hypervisor 强制;④ **IPv6 更严**——路由器永不分片,过滤 ICMPv6 PTB 必黑洞,最小 MTU 1280;⑤ **bond/LACP**——LACP 不校验 MTU,两端失配静默丢包;⑥ 持久化可能被 **DHCP option 26/cloud-init** 重新覆盖。
- **一句话**:MTU 是 L2 域属性,"改一台"实为"改一个 L2 域";公网侧维持 1500、NAT 机默认否决,只在统一规划的内网域灰度,全程按"有损操作+可回滚+可观测"三原则,且先确认机器是物理机/云 VM/k8s node(三者失配点完全不同)。出处:本仓 `jumbo.md`。

## [2026-06-17] Triton 写大模型算子：核心入门 + 算子融合 + FlashAttention
**标签**: triton, 算子融合, flashattention, rmsnorm, memory-bound, course8
手写算子加速的本质是**省 HBM 访存**：先判断算子 compute-bound 还是 memory-bound（算术强度=FLOPs/Bytes），LLM 里归一化/激活/逐元素/decode-attention 都是 memory-bound，融合即数倍加速。
- **Triton 心智模型**：粒度是 program/Tile 不是 thread，只写「第 pid 块怎么算」。五件套 `pid / block_start / offsets / mask / load-store`；多维寻址=指针+stride+`[:,None]`(行)/`[None,:]`(列)广播；`tl.constexpr` 是编译期常量。性能旋钮 `BLOCK_*`/`num_warps`/`num_stages`，用 `@triton.autotune(configs, key=[...])` 自动搜（key 形状变才重搜）。
- **算子融合=让中间结果不落显存**。memory-bound 算子加速比≈访存压缩比：naive softmax 5 个 kernel 约 8MN 访存，融合后 2MN→理论 ~4x（softmax.py 实测 4x）。三形态：逐元素链式(bias→GELU)、归约融合(RMSNorm/LayerNorm/softmax)、生产者-消费者(residual-add→RMSNorm、matmul→activation)。
- **Fused RMSNorm**(`course8/fused_rmsnorm.py`，实测 8192×4096 fp16 **3.17x**)：一个 program 处理一整行，只读一次只写一次，square/mean/rsqrt/mul 全在寄存器；**归约必须先 `.to(tl.float32)` 累加**否则 fp16 丢精度。vLLM 的 RMSNorm 带 `residual` 入参=add+rmsnorm 生产者-消费者融合，相加结果留寄存器直接喂归一化并把 residual 写回。
- **在线 softmax = FlashAttention 基石**：分块读入维护运行最大值 m 和分母 l，新块来时旧结果乘**重缩放因子 `exp(m_old-m_new)`**(≤1)对齐到新基准，单遍扫描得正确结果，无需物化整行。
- **FlashAttention**(`course8/attention.py`)=分块+在线 softmax+**永不物化 S=QKᵀ 矩阵**（否则 seq×seq 访存/显存爆炸）。Q 沿 seq 切 BLOCK_M 用 grid-0 并行、K/V 切 BLOCK_N 用 kernel 内 for 循环、batch×head 走 grid-1；内层循环把 acc 按重缩放打折再加新块 P·V。prefill(seq>1,compute-bound) vs decode(seq=1 但 KV 长,memory-bound)。
- **融合边界**：爆寄存器/共享内存降 occupancy 反而慢；compute-bound 算子收益有限；分块策略冲突时别强融。
- 教学主线见 `course8/TUTORIAL.md`（0 为什么手写→1 Triton 入门→2 融合→3 在线 softmax→4 FlashAttention→5 vLLM 落地），参考手册见 `course8/DOC.md`。

## [2026-06-17] torch.compile 三段栈 + 与 CUDA Graph 的协同
**标签**: torch-compile, dynamo, inductor, cuda-graph, graph-break, reduce-overhead, vllm, course9
torch.compile 与 CUDA Graph 解决**不同且正交**的问题：compile 靠**算子融合**砍 kernel 数量与访存，CUDA Graph 砍 **launch 次数**；最优是叠加用（compile 把图变小变快，CUDA Graph 再吃掉剩余 launch 开销）。
- **三段式编译栈**：Python 字节码 →①**TorchDynamo**(抓字节码→FX Graph，加 guard 守卫形状/类型，遇不认识的就 graph break) →②**AOTAutograd**(拆前/反向 + 算子分解为 aten) →③**TorchInductor**(默认后端，GPU 生成融合 Triton kernel、CPU 生成 C++)。融合收益同 Course8：中间结果留寄存器、省 HBM 往返。实测逐元素链 `sin*cos+tanh*2-exp.clamp` 在 8192² 上 **eager 6.67ms → compile 0.64ms = 10.4x**。
- **Graph Break**：`.item()`/`.cpu()`/依赖张量值的 `if`/不支持的库调用会断图(图1→python→图2)。`torch._dynamo.explain(fn)(x)` 数断点：带 `.item()` 的 if → graph=2/break=1，无依赖 → graph=1/break=0。断点降低融合 + CUDA Graph 收益（CUDA Graph 不能跨断点连续捕获）→ 这正是 vLLM **PIECEWISE** 的由来：在不可避免断点(如 attention)处把前向切多段、每段各自编译+各自 CUDA Graph。
- **`mode="reduce-overhead"` = compile 自带 CUDA Graph**：在 Inductor 之上自动套一层 CUDA Graph。launch-bound 小模型(12×[Linear+GELU],bs=8)实测 eager 0.406ms / compile-default 0.332ms / **reduce-overhead 0.109ms (3.71x)**——default 只融合仍逐个 launch，大头加速来自那层 CUDA Graph。
- **动态 shape 触发重编译**：guard 失败就重编。默认开**自动动态**：首形状按静态编、遇第二个形状把该维升级为动态图、之后全复用→喂 [128,256,512,256,128] 实测**编 2 张图**；`dynamic=True` 一上来就编形状无关图→**1 张**。故形状恒定的 **decode(seq_len=1) 最适合 compile+CUDA Graph**，prefill 形状多变只能走 piecewise / padding 分桶。
- **vLLM 5 种 cudagraph_mode**：NONE / PIECEWISE(prefill,兼容强) / FULL(整图,规整负载) / FULL_DECODE_ONLY / **FULL_AND_PIECEWISE(默认:decode 用 FULL、prefill 用 PIECEWISE)**；dispatcher 按 **FULL>PIECEWISE>NONE** 优先级、用 BatchDescriptor(num_tokens/uniform_decode/has_lora) 作 key 匹配已捕获图。承接 [2026-06-13 CUDA Graph] 条。
- 出处 demo：`vllm_learn/code/course9/torch_compile_demo.py`(4 实验:基础融合/graph-break/reduce-overhead/动态 shape)；教学主线 `course9/TUTORIAL.md`(0 为什么→1-5 CUDA Graph→6-8 torch.compile→9 vLLM 落地)，源码级细节见 `course9/DOC.md`。

## [2026-06-18] Medusa 多头并行 vs 独立 Draft 模型(投机解码 draft 方案对比)
**标签**: speculative-decoding, medusa, draft-model, tree-attention, eagle
Medusa 用"去自回归 + 共享 backbone 的多头"造草稿:同一隐状态 `h_t` 一次并行预测 `t+1…t+K+1`,头间无依赖;独立 draft 是个小 LLM,自回归逐个出草稿。承接 [2026-06-13 投机解码演进线] 条。
- **优点**:①draft 无自回归→一次前向出全部候选(O(1) 而非 O(K) 次小模型前向);②单模型部署,只多几个线性层(`W1∈ℝ^{d×d}`,`W2∈ℝ^{|V|×d}`),省显存/省第二套模型的调度与 TP-PP 配置;③分布天然对齐——共用 backbone 表征,且 `W1=0`、`W2` 拷贝原 LM Head,起点≈原模型→接受率高;④训练便宜,Medusa-1 只训头冻结骨干,单卡可做。
- **缺点**:①根本代价是远距离头精度衰减——第 k 头从 `h_t` 盲猜 `t+k+1`,看不到已定下的 `t+1…t+k`,故 K 一般只能 4~5;②单条候选短→必须用 **tree attention** 取各头 top-k 组合成草稿树、一次验证多分支,靠"广度"补"深度",验证更复杂;③头与具体 backbone 的 `h_t` 绑死,换基座要重训,可移植性差;④各头独立 softmax 只建模边缘分布,无 token 间条件/联合建模能力。
- EAGLE 正是补 Medusa 这点:在**特征空间做轻量自回归**(drafter 吃 target hidden state 外推下一隐状态再共享 LM Head 映射回 token),分布更齐、草稿更长。出处 `vllm_learn/code/course16/DOC.md`(L144-245)、`tree_attention_demo.py`。

## [2026-06-19] CUDA C++ 手写算子 vs Triton:thread-centric vs tile-centric
**标签**: triton, cuda, warp-reduce, block-reduce, grid-stride, course8
两者核心分野:**CUDA 让你管 thread,Triton 让你管 tile**。Triton 把「block 内如何切到 thread、如何同步、如何用 shared memory」交给编译器;CUDA 全手动,可控性与性能上限更高。
- **向量加法(add.py)**:Triton 写「一个 program 处理 BLOCK_SIZE 个元素」——`offsets=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)`+`mask`+`tl.load/store`,无 threadIdx、越界用 mask 向量、直接吃 torch.Tensor 零绑定;CUDA 写「一个线程处理一个元素」——手算 `blockIdx*blockDim+threadIdx`、`if(idx<N)` 标量分支、要写 host launch+pybind 绑定。element-wise 算子两者性能一致(都 memory-bound),差异只是开发成本。
- **warp 规约**:CUDA 显式 `__shfl_down_sync(0xffffffff,val,offset)` 蝶式折叠(offset=16→1),你得知道 warp=32、自己管 mask;Triton **完全没有 warp 概念**,一行 `tl.sum(x,axis=0)`,编译器降级时自动生成 shuffle。
- **block 规约**:CUDA 三层编排——warp 规约→`if(lane==0) shared[wid]=val`→`__syncthreads()`→第 0 warp 二次规约,漏写同步=数据竞争;Triton **还是 `tl.sum`**,warp 规约和 block 规约写法完全一样,无 shared/无 syncthreads,跨多少 warp 由编译器按 BLOCK_SIZE/num_warps 决定。softmax/rmsnorm 的整行求和本质都是 block 规约。
- **grid-stride**:CUDA `stride=gridDim.x*blockDim.x` 的 `for(i=idx;i<N;i+=stride)`,grid 与 N 解耦(persistent kernel);Triton 默认让 grid 维度承担跨步(program 数随 N 增长,program 内不循环),需 persistent 时才写 `stride=tl.num_programs(0)*BLOCK_SIZE` 的 tile 级跨步循环。
- **选型**:标准 element-wise/规约/softmax/norm/GEMM → Triton(快、不易错、autotune 自动调参);需极限压硬件、非标准并行、tensor core 细粒度排布/warp specialization/双缓冲 → CUDA C++ 上限更高。出处 `course8/cuda_triton_compare.md`,承接 [2026-06-17 Triton 写大模型算子] 条。

## [2026-06-20] 预热(Warmup)的本质:消化一次性开销 + 进入稳态
**标签**: warmup, cuda-graph, benchmark, lazy-init, cold-start, course9
预热本质=把「只在第一次执行才发生、稳态不再重复」的一次性开销,与「需跑一会儿才稳定」的瞬态,提前在「不计时/不捕获/不服务」阶段消化掉,让后续测量/capture/真实请求面对 warm、确定、可复现的状态。不是玄学空跑。
- **消化的一次性开销(lazy init)**:CUDA context 创建、cuBLAS/cuDNN handle 加载与建立、kernel 的 JIT/PTX→SASS module load、首次 `cudaMalloc`(慢且**同步**;之后 PyTorch caching allocator 从内存池复用)、cuBLAS/cuDNN autotune 试跑选最快算法、torch.compile 的 Dynamo trace+Inductor 编译。这些第一次奇慢、之后不再发生。
- **进入稳态(steady state)**:GPU 时钟 boost(冷态低频,持续负载才升频/或热降频)、L2 与指令缓存预热。
- **对 CUDA Graph 是铁律非可选**:capture 要求图内只有纯异步设备侧操作;而 cuBLAS 懒加载/cudaMalloc/autotune 多是**同步**的,不预热直接 capture 会①把非计算初始化录进图污染重放,或②含同步操作直接令 capture 失败(`operation not permitted when stream is capturing`)。故 `cuda_graph.py` 在 **side stream** 预热(`s.wait_stream(current)`→跑→`current.wait_stream(s)`):既触发懒加载又不污染默认流。承接 [2026-06-13 CUDA Graph] 的「warmup 要在 side stream」。
- **对 benchmark**:不预热则首次迭代把初始化/cudaMalloc/频率爬升算进平均,测到的是「冷启动+算子」混合值而非稳态吞吐。
- **推理服务的 warmup 请求(vLLM/TGI)同理**:启动时打假请求,把 kernel 编译/cuBLAS 初始化/allocator 预分配/CUDA Graph capture/padding 分桶图预捕获全在启动阶段做完,使第一个真实请求不吃冷启动延迟、避免首 token 延迟尖刺。出处 `course9/warmup.md`、`cuda_graph.py`。

## [2026-06-20] CUDA Graph 分桶 padding 的三个工程细节
**标签**: cuda-graph, padding, bucketing, memory-pool, inference-mode, course9
动态 batch 用「分桶+padding」复用静态图(只为少数档 size 各捕一图,运行时 bs 向上取整到档位、真实行填数据/padding 行填占位、replay 后只取真实行)。三个易忽略的工程点:
- **多图共享内存池要从大到小捕获**:`pool=g.pool()` 让各图捕获期的**临时分配**(forward 中间激活,capture 内分配又释放)复用同一池。大图先捕获 → 一次把池撑到峰值工作集,后续小图临时块都装进已预留大块、池不再增长,也不留「小到装不下后续大请求」的碎块;反向(小→大)则要为大图额外申请、抬高峰值并制造碎片。结果总预留显存 ≈ 最大图而非各图之和。这就是 vLLM 按 size 逆序 capture 的原因。
- **PyTorch 写 CUDA Graph 全程无 H2D/D2H 显式搬运**:`inp`/`out`/权重都建在 `device="cuda"`,喂数据 `buf[:bs].copy_(x)`(x 也是 cuda 张量)是**设备内 D2D**,取结果和 allclose 校验也都在 GPU。手写 CUDA C++ demo 常见 `cudaMemcpy` 只因那种例子从 CPU 备数据、CPU 打印。更关键:**capture 期禁止同步 H2D/D2H(铁律一)**,本就只能纯设备计算。
- **`torch.inference_mode()` = 更激进的 no_grad**:关闭 autograd,不建反向图/不存梯度元数据,还关掉 version counter/view 追踪 → 张量无法再参与 autograd,换更低运行时开销(适合推理/benchmark)。与 `model.eval()` 分工不同:eval() 管 dropout/BN 行为切换,inference_mode() 管 autograd 开关,推理两者配合。
- 出处 `course9/cuda_graph_padding.py`(逐行注释含 Q/A),承接 [2026-06-13 CUDA Graph] 与 [2026-06-20 预热] 条。

## [2026-06-20] vLLM 里 CUDA Graph/torch.compile 落地:prefill vs decode vs attention
**标签**: vllm, cuda-graph, torch-compile, prefill, decode, paged-attention, padding, course9
分桶+padding 是 CUDA Graph 静态 shape 的补丁,只在「launch-bound + pad 几乎免费 + 形状可枚举」时划算——这恰好是 decode、恰好不是 prefill;torch.compile 原生支持 dynamic shape,根本不需要分桶+padding。
- **prefill 为何不用分桶+padding(三项与 decode 全反)**:① 变化维度——decode 变 batch size(小而有界 [1,256],几档覆盖,pad 多几行);prefill 变总 token 数(chunked prefill 摊平成 varlen,[1,max_num_batched_tokens] 近连续)。② padding 代价——decode 是 launch-bound,pad 几行近乎免费,CUDA Graph 砍 ~1ms launch 是大头净赚;prefill 是 compute-bound,pad 出来的 token 要走完整 transformer 真实 FLOPs(5000→8192≈浪费 39% 算力直接砍吞吐),pad 成本≫图收益。③ 图收益——CUDA Graph 只在 launch-bound 收益大,decode 是 prefill 不是。
- **torch.compile vs CUDA Graph 对动态 shape 的态度**:compile 把 token 维标记 symbolic,一次编译服务所有长度,Inductor 生成的 Triton kernel 直接吃符号化 num_tokens → 无需 padding;CUDA Graph 只支持静态 shape,才需要分桶+padding(decode 才划算)。
- **CUDA Graph 真正的限制是「静态 shape + 静态 launch 配置」,不是「静态工作量」**:kernel 可以有数据相关内部循环 `for i in range(seq_len)`,只要 seq_len 是从**固定 shape 张量里读出的值**(而非张量 shape)。这是 decode attention 能进 FULL 图的关键。
- **decode attention 的两个变量各自被中和**:① per-request KV 长度(score 的 KV 维变)→ KV cache 是预分配分页定形缓冲(shape 不随对话增长),kernel 多吃 block_tables/seq_lens(定 shape、值变),内部按读到的 seq_len 循环、超出 mask/early-exit,grid 按 max_seq_len 分区数定大小;② batch size → 分桶+padding 钉死 Q 行数。两者都压平 → capture 只见固定 shape,每步 copy_ 新值再 replay(即铁律二)。prefill attention 因 query token 数改 grid,走 varlen kernel(cu_seqlens)不进全图。
- **torch.compile 不编译 attention 内核本身**:attention 是手写 custom op(FlashAttn/FlashInfer/PagedAttention,如 torch.ops.vllm.unified_attention),Dynamo 当黑盒不 trace,Inductor 不替它生成 kernel;compile 只融合 attention 前后的逐元素/Linear/Norm,attention 调用是天然 graph-break 边界 → 这就是 PIECEWISE 的由来。变长 KV 完全在手写 kernel 内部(分页/cu_seqlens)消化。
- **一句话**:CUDA Graph 怕「shape/launch 维度变」,不怕「算的量变」;PagedAttention 把「KV 变长」从前者搬到后者,才让 decode attention 整体图化。出处 `course9/vllm_cuda_graph_compile.md`,承接 [2026-06-17 torch.compile 三段栈]、[2026-06-20 CUDA Graph 分桶 padding]、[2026-06-13 CUDA Graph] 与 paged attention 各条。

## [2026-06-21] vLLM transformer block 如何配合 torch.compile/CUDA Graph(含两个易错点)
**标签**: vllm, torch-compile, cuda-graph, custom-op, paged-attention, graph-break, course9
精炼复刻 vLLM v1 的 DecoderLayer(RMSNorm→Attn→RMSNorm→MLP,pre-norm+residual,对齐 LlamaDecoderLayer);attention 走自定义算子、KV 走分页定形缓冲+metadata。两个常被讲错的点:
- **「attention 是 graph break」是误解**:vLLM 把 attention 注册成**带 fake/meta 实现的 custom op**(`direct_register_custom_op`,如 `vllm::unified_attention_with_output`)。带 fake 实现 → Dynamo **不 break**,把整层 trace 成**一张图**,attention 是其中一个**不透明节点**(实测 `torch._dynamo.explain` graph=1/break=0)。真正的「切」发生在 vLLM **自己的编译后端**:`compilation/backends.py` 的 `split_graph` 按 `config/compilation.py` 的 `splitting_ops=['vllm::unified_attention_with_output', ...]` 把这张 FX 图切成 **piecewise** 子图,每段(norm+proj+MLP)各自 compile/CUDA Graph,attention 段单独跑 varlen kernel。对照:custom op **没** fake 实现、或在 Python 里 `seq_lens.max().item()`/依赖张量值分支,才会触发真正的 **Dynamo break**。
- **可捕获的 attention 参考 kernel 必须无 host 同步**:玩具实现若用 Python 循环 + `seq_lens[i].item()` 驱动循环边界,capture 时报 `operation not permitted when stream is capturing`(`.item()` 是 D2H 同步,违反铁律一)。正解=**mask 版**:按固定 max_ctx gather KV、用设备端比较 `pos < seq_lens[:,None]` 把超出真实长度的位置 masked_fill(-inf)。无 `.item()`/无 D2H → 可捕获;变长体现为 **mask 的值** 而非 shape。这正是真实 GPU kernel「固定 grid + 按 context_lens 做 early-exit/mask」的思路 → 「CUDA Graph 怕 shape/launch 变,不怕算的量变」的活例。
- **实测三结论**:① decode 整层能捕获为 FULL CUDA Graph 并 replay;in-place 把 `seq_lens[0]` 5→6 后 replay,该序列输出确实改变 → 固定 shape 图里靠改 metadata 的「值」就能让变长 KV 参与。② graph=1/break=0(custom op 不破图)。③ KV cache 形状恒定(num_blocks×block_size×H×D),4 条序列 ctx 长度 [5,17,33,40] 各异,变化只在 seq_lens/block_tables 的值。
- **为什么要 fake+FX 切片,而不是只用一张 FULL 图(回答「是否多此一举」)**:① 前提修正——fake 实现对 custom op **不可省**,不写 `register_fake` 则 compile 推不出输出 meta、**会在该算子破图**(demo 的 graph=0 break 正是因为写了 fake)。② attention **必须**是不透明手写 kernel(FlashAttention/PagedAttention):分页 gather+block_tables 间接寻址、online-softmax 不物化 `[seq,seq]` 分数、varlen、读写 KV cache+`get_forward_context()` 取 metadata——Inductor **生不出**,只能作为 kernel「调用」,fake 让 Dynamo 不破图、收成干净单节点。③ **关键正交性**:「FX 切片」是**编译层**的事(Inductor 本就无法跨不透明 attention 融合,边界是内在的,关掉 CUDA Graph 也存在),「FULL/PIECEWISE」是 **CUDA Graph 捕获粒度**的事,两者叠在**同一份已编译+已切片的产物**上。④ 故 decode(paged-decode attention 定 shape 可捕获)用 **FULL 一张图**即可,vLLM 也确实这么干、切片不增额外捕获开销;prefill 的 varlen attention **不可捕获**,**只能**绕开它→只把静态计算片段各自捕成小图、attention 图外 eager 跑=**PIECEWISE**,而这**依赖切片给的边界**。结论:切片是编译硬需求(嵌入手写 kernel+给 Inductor 干净单元)、又是 prefill 上 CUDA Graph 的前提,**非多此一举**;纯 decode 下 FULL 够用但基础设施不可省。
- 出处 `course9/vllm_transformer.py`(纯 PyTorch 可 CPU/GPU 直跑),承接 [2026-06-20 vLLM CUDA Graph/compile 落地]、[2026-06-13 CUDA Graph]。
