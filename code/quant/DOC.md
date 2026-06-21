# LLM 推理量化:从原理到 vLLM 实操

> 这份材料为"给 LLM 推理服务上量化降本"这道题准备,配套 `quant/` 下 5 个可运行的 `.py`。
> 学习顺序:先读本文建立框架 → 跑 01~05 验证每个结论 → 回头再看本文的三问总结。

---

## 0. 一句话框架:量化省的到底是什么

量化 = 用「低位宽整数/浮点 + 缩放因子 scale」近似表示「高位宽浮点」。

```
反量化:  x ≈ (q - zero_point) * scale
```

它直接省的是 **存储/显存** 和 **访存带宽**,**不一定** 省算力。
能不能"变快",取决于你这个阶段是 **memory-bound** 还是 **compute-bound**——
这跟你 `NCU.md` 里那把尺子是同一把。先记住这句,第③问全靠它。

| 旋钮 | 含义 | 典型取值 |
|------|------|----------|
| 位宽 | 整数用几个 bit | int8 / int4 / fp8 |
| 对称性 | 有无 zero_point | 权重对称、激活非对称 |
| 粒度(层内) | 几个元素共用一个 scale | per-tensor / per-channel / per-group(128) |
| 范围(层间) | 哪些层量化、量化到几 bit | 大 Linear 量化,norm/embedding/lm_head 保 fp16 |

> 你之前理解的"对不同层用不同精度",是上表最后一行的"层间"旋钮;
> 但真正决定误差的,往往是倒数第二行的"层内粒度"。两个维度都要会。
> → 跑 `01_quant_numerics.py` 看粒度对误差的影响。

---

## 1. 两大流派:weight-only vs weight+activation

这是第①问的核心。区别只有一个:**激活(activation)要不要也量化**。

### Weight-only(W8A16 / W4A16:GPTQ、AWQ)
- 只量化权重,激活保持 fp16/bf16。
- 计算时把权重 **反量化回 fp16** 再算(或用 Marlin 这类 kernel 边解边算)。
- **省的是显存和访存**:权重从显存搬到计算单元的字节数减半(int8)/减到 1/4(int4)。
- **算力没变**:矩阵乘还是 fp16 的乘加。
- **适合 decode 阶段**(memory-bound):decode 每步只算 1 个 token,算力闲置、瓶颈在搬权重 → 少搬字节 = 直接变快。

### Weight+Activation(W8A8:INT8 SmoothQuant、FP8)
- 权重和激活 **都** 量化成 int8/fp8。
- 这是唯一能真正用上 **INT8/FP8 Tensor Core** 的方案——矩阵乘本身用低精度算力,吞吐可翻倍。
- **省显存 + 省算力**,但代价是 **激活很难量化**(见第②问),需要 SmoothQuant 这类预处理。
- **适合 prefill 阶段**(compute-bound):prefill 一次算几百上千 token,瓶颈在算力 → 低精度算力翻倍才有用。

| | weight-only (W4A16/W8A16) | weight+act (W8A8/FP8) |
|---|---|---|
| 省什么 | 显存、访存带宽 | 显存、带宽、**算力** |
| 算力红利 | ❌ 无(还是 fp16 算) | ✅ 有(INT8/FP8 Tensor Core) |
| 激活处理 | 不动,简单 | 要量化,难,需 SmoothQuant |
| 最受益阶段 | **decode**(memory-bound) | **prefill**(compute-bound) |
| 代表算法 | GPTQ、AWQ | SmoothQuant、FP8 |

> → 跑 `02_layer_by_layer.py`,在同一个 Linear 上看 W8A16/W4A16/W8A8 的误差差异。

---

## 2. 为什么激活比权重难量化?(第②问)

### 难在哪
1. **权重分布"乖"**:训练好的权重近似钟形、各通道尺度接近,per-channel 量化贴合得很好。
2. **激活分布"野"**:LLM 激活存在 **系统性离群通道(outlier channels)**——
   极少数固定的特征维度,值比其他维度大几十上百倍。
   量化时 scale 被这几个离群通道撑大,99% 的正常值被挤进很少的格点 → 信息丢失。
3. **离群值不能裁**:它们恰恰携带关键信息,clip 掉会直接掉点。
4. **激活是动态的**:每次输入都不同,scale 必须运行时现算(动态量化),
   而权重是常量可离线静态量化。现算本身有开销。

### 业界怎么解
**SmoothQuant:把"难"从激活搬到权重。** 对 `y = x @ Wᵀ`,按输入通道选迁移因子 `s`:

```
x_smooth = x / s        (激活变平滑,离群被压下去)
W_smooth = W * s        (权重吸收尺度,权重"乖"放大点也好量化)
恒等:x_smooth @ W_smoothᵀ == x @ Wᵀ

s_j = max|x_j|^α / max|W_j|^(1-α)     α 控制迁移多少
```

> → 跑 `04_activation_smoothquant.py`:实测离群通道幅度 32→1.4,
> W8A8 输出误差 6.6%→1.6%。

补充:**FP8 比 INT8 更耐激活离群**,因为浮点格点是非均匀的(指数位给了大动态范围),
天然对离群值更友好。这是 FP8 在新硬件(Hopper/Ada)上越来越主流的原因之一。

---

## 3. 题眼:W4A16 在 decode 阶段真的会变快吗?(第③问)

**会,但快的不是"计算",是"少搬了权重"。而且有前提。**

### 为什么会快
decode 阶段每步只生成 1 个 token,矩阵乘是"瘦高矩阵 × 权重",
计算量极小、**算力大量闲置**,瓶颈在 **把权重从显存搬到 SM**(memory-bound)。
W4A16 把权重压到 1/4 → 搬运字节减到 1/4 → 这一步直接快近 4 倍(理想情况)。
**计算本身没变快(还是 fp16 乘加),但计算本来就不是瓶颈,所以无所谓。**

### 什么时候"不快",瓶颈在哪
1. **反量化(dequant)开销**:int4 要先解回 fp16 才能算。
   batch 很大时 dequant 本身可能成为新瓶颈;kernel 不好(没融合)也会吃掉收益。
2. **prefill 阶段**:prefill 是 compute-bound,W4A16 不提供低精度算力 → **基本不会变快**,
   省的只是显存。想让 prefill 快得上 W8A8/FP8(真低精度算力)。
3. **模型太小 / 硬件不匹配**:模型小到权重搬运不是瓶颈,或 GPU 没有对应低精度算力时,
   收益被 dequant 开销抵消。
   → `05_vllm_quant.py` 实测:Qwen3-0.6B 在 3090 上 fp8 的 decode 吞吐几乎持平
   (4096 vs 4122 tok/s),因为模型太小 + 3090 无原生 FP8 算力。

### 一张判断表(把"量化=必然变快"这个误区彻底拆掉)

| 阶段 | bound | weight-only(W4A16) | weight+act(W8A8/FP8) |
|------|-------|--------------------|----------------------|
| **decode** | memory-bound | ✅ 变快(少搬权重) | ✅ 变快(少搬 + 算力) |
| **prefill** | compute-bound | ❌ 基本不快(只省显存) | ✅ 变快(低精度算力) |

> **结论金句**:量化首先是省显存/带宽的技术;"变快"是 **派生收益**,
> 只在「瓶颈正好是被量化省掉的那种资源」时才兑现。
> 用 memory-bound / compute-bound 这把尺子,先判断阶段,再判断方案。

---

## 4. KV Cache 量化(常被忽略的第三块)

长上下文时,KV Cache 往往比权重还吃显存。它本质是一种"激活量化",和权重量化正交、可叠加。
- vLLM:`kv_cache_dtype="fp8"`,KV 显存直接减半 → 能开更长上下文 / 更大并发。
- 对精度影响通常很小(FP8 对 KV 的离群也较耐受)。

---

## 5. vLLM 实操速查

```python
from vllm import LLM
# 1) 在线量化:普通 fp16 模型,启动时实时压(只支持 fp8 这类无需校准的)
LLM(model="...", quantization="fp8")
# 2) 加载预量化 checkpoint(GPTQ/AWQ/compressed-tensors 自动识别,生产推荐)
LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ")
# 3) KV Cache 量化(可与上面叠加)
LLM(model="...", kv_cache_dtype="fp8")
```

命令行:
```bash
vllm serve <model> --quantization fp8
vllm serve <model>-AWQ                 # 自动识别
vllm serve <model> --kv-cache-dtype fp8
```

离线产出需校准的量化(W4A16-GPTQ / W8A8-SmoothQuant)用 `llmcompressor`,
recipe 见 `05_vllm_quant.py` 文末。注意 recipe 里 `ignore=["lm_head"]` ——
对应第 1 节"层间"原则:敏感层保 fp16。

### ⚠ 本机(RTX 3090 / Ampere sm_86)注意
3090 **无原生 FP8 计算单元**(要 Ada sm_89 / Hopper)。vLLM 会退化成 weight-only FP8
(走 Marlin kernel),日志里能看到明确警告。所以本机:
- fp8 ✅ 省显存(decode 受益),❌ 拿不到 FP8 算力(prefill 不快)。
- 想体验 W8A8 真·算力红利,需要 Ada/Hopper 卡。

---

## 6. 配套代码索引

| 文件 | 讲什么 | 对应问题 |
|------|--------|----------|
| `01_quant_numerics.py` | 量化数值基础:对称/非对称、per-tensor/channel/group、显存账 | 框架 |
| `02_layer_by_layer.py` | 不同层怎么量化、哪些层不量化、W8A16/W4A16/W8A8 对比 | ①+层间 |
| `03_weight_only_awq.py` | weight-only 为什么要算法:RTN vs AWQ(11.7%→1.9%) | ① |
| `04_activation_smoothquant.py` | 激活为什么难 + SmoothQuant(6.6%→1.6%) | ② |
| `05_vllm_quant.py` | vLLM 加载量化模型 + 显存/吞吐实测 + 3090 FP8 caveat | ③+实操 |

---

## 7. 三问的"电梯答案"(背这个)

- **①weight-only vs weight+act**:前者只压权重、省显存带宽、decode(memory-bound)受益、代表 GPTQ/AWQ;后者连激活一起压、能用低精度算力、prefill(compute-bound)受益、代表 SmoothQuant/FP8。
- **②激活为什么难**:激活有系统性离群通道、动态变化、不能裁;权重乖且静态。解法是 SmoothQuant 把难度按通道恒等迁移到权重,或用动态范围更大的 FP8。
- **③W4A16 decode 会变快吗**:会——但快在"少搬权重"(memory-bound 阶段),不是算得快;prefill(compute-bound)基本不快,且 batch 大 / 模型小 / 硬件不匹配时 dequant 开销会吃掉收益。一句话:量化省的是显存带宽,变快是派生收益,看阶段的 bound。
