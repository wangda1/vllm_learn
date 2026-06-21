1. 背景：Transformer 解码与 GPU 性能瓶颈
1.1 Transformer 结构
[图片]
以典型 Decoder-only LLM（如 LLaMA/Qwen 系列）为例，单个 Decoder Block 主要包含：
1. 自注意力（Self-Attention）
  - QKV 预投影：qkv_proj（通常含 3 个线性层，或合并为一个大矩阵）
  - 注意力计算：self attention
  - 输出投影：o_proj
2. 前馈网络（FFN）
  - 第一线性层：gate_up_proj（对于 SwiGLU 结构，通常包含 gate 和 up 两个投影）
  - 第二线性层：down_proj
3. 其他：LayerNorm/RMSNorm、残差连接、激活函数等
在 prefill 阶段，输入张量形状为：
- $$X ∈ ℝ^{B × L × H}$$
  - B: batch size
  - L：序列长度（上下文 token 数）
  - H：隐藏维度（如 LLaMA‑13B 为 5120）
每层的算子张量形状示意如下表所示，核心结论是：矩阵乘规模与 B、L、H 成正比；prefill 阶段 L 一般较大，因此可以很好占满 GPU。表 1 展示了 decoder block 中主要算子的输入、输出和权重张量形状。
[图片]
1.2 算子瓶颈：为什么 decode 阶段反而更慢
prefill 阶段：
- 输入通常是数百 token 长度的上下文
- 自注意力和 FFN 均在较大 L 上运算
- GEMM（矩阵乘）规模大，GPU 算力利用率高（compute bound 或接近）
decode 阶段（带 KV Cache）：
- 每次只生成 1 个新 token
- 输入序列长度逻辑上在增长，但历史 token 的 key/value 已缓存在 KV Cache 中
- 当前 step 的计算，主干矩阵乘基本退化为 矩阵 × 单个向量 / 小批量向量
简单来说，就是在大部分场景中 LLM 的 decode 阶段的大部分算子都为访存密集型。而 decode 阶段处于访存密集型的算子，当我们扩大 batch size 时，所有算子的运行时间并非会随着计算量线性增长的。
以 A100 40G 单卡上执行 LLM-7B 模型的定量分析为例，测试时使用 Hidden dim=4096， Layer=32， Context Seqlen=128。从 batch size=1 增长到 batch size=512，我们的运算量增长了 500 倍，但执行时间仅仅延长了 3 倍。具体时间对比如下表所示：
暂时无法在飞书文档外展示此内容
[图片]
1.3 自回归采样的结构性问题
标准 Decoder-only LLM 采用 自回归采样（autoregressive decoding）：
1. Prefill 阶段：
  - 输入完整上下文（前缀）
  - 得到最后一个位置的 logits，经 softmax + top‑k/top‑p 采样生成下一个 token
2. Decode 阶段：
  - 从第一个生成 token 开始，一次只生成一个 token：
    - 当前输出 token 与历史 tokens 拼接为新输入
    - 再运行一次 forward，采样下一个 token
  - 重复直到生成 EOS 或达最大长度
能不能有别的办法在decode阶段，一次生成2个或者更多个token
[图片]
LLM 自回归采样过程，在 decode 阶段是逐 token 的串行解码，解码 K 个 token 需要模型进行 K 次串行运算。而在使用 kv cache 优化后，自回归采样的 decoding 阶段输入输出都是单个 token，在 batch size 为 1 时，Transformer block 中的矩阵乘都退化为矩阵乘向量操作，对于 GPU 推理来说，这是非常明显的 IO bound（内存带宽瓶颈），导致 GPU 计算资源利用率低下，并且序列的生成时间随着序列长度的增加而线性增加。
而矩阵乘法是大语言模型推理中最为耗时的部分，因此，我们可以认为基于 kv cache 推理加速中模型 decoding 阶段主要还是访存瓶颈（memory bound），即 GPU 读取数据的花费的时间比 GPU 计算花费的时间更长。
简单理解就是在 decode 阶段：
- 解码 K 个 token，需要调用 K 次 LLM forward，严格串行
- 使用 KV Cache 后：
  - 每一步只有 1 个新 token 参与 Attention/FFN
  - 大部分 GEMM 退化为 GEMV
  - 当在batch = 1又会加重这种情况，前向推理计算会变得更加的 IO bound
[图片]
因此，在 decode 阶段，主要瓶颈是权重参数的反复读写，而不是算力。想加速，其中重要的一个方法就是需要减少对大模型的串行调用次数。
2. 推测解码：并行解码的基本框架
推测解码的本质：用一个 更便宜的近似模型（Draft Model） 先生成一串候选 tokens，然后用大模型 一次并行验证多个 token，以减少大模型 forward 的总调用次数。或者说其本质上是并行解码，通过增加每个解码步目标 LLM 计算的 tokens 数目，减少总的解码步数（即减少了 LLM 参数的反复读写），从而实现推理加速。
[图片]
2.1 投机解码原理
推测解码使用两个模型：一个是原始目标模型，另一个是比原始模型小得多的近似模型（Draft Model）。小型近似模型可以采用与原始模型相同的结构，但参数更少，或者干脆使用 n-gram 模型。小型模型不仅计算量较小，同样内存访问也更少。
暂时无法在飞书文档外展示此内容
在每个解码步骤中，推测解码首先通过近似模型高效地推测 target LLM（待加速的 LLM）未来多个解码步可能生成的 tokens；然后再用 target LLM 同时验证这些 tokens，通过验证的 tokens 才能作为当前解码步的解码结果， 从而保证生成质量。
- 目标模型（大模型）：$$M_p$$，token 条件分布为 $$p(x_t | x_{<t})$$
- 近似（草稿）模型：$$M_q$$，token 条件分布为 $$q(x_t | x_{<t})$$
给定当前上下文 $$x_{<t}$$，我们要生成后续 token 序列 $$(x_t, x_{t+1}, ...)$$，它的核心思想就是用更快的$$M_q$$提议候选序列，由$$M_p$$进行并行验证和修正，在统计上仍等价于从$$M_p$$逐 token 采样。推测解码的核心思想是通过$$M_q$$生成草稿 token，并由$$M_p$$进行并行验证和修正，以减少$$M_p$$的串行调用次数。
[图片]
2.2 算法流程
设一次推测长度为$$γ$$（gamma），即草稿一次生成$$γ$$个 token：
1. 草稿生成（Drafting）
  - 用小模型$$M_q$$在当前上下文上生成$$γ$$个连续 token：
$$y_1, y_2, ..., y_γ$$
  - 这些 token 来自$$q$$分布的采样或贪心解码
2. 并行验证（Verification）
  - 构造拼接序列：$$context + [y_1, ..., y_γ]$$
  - 用大模型 $$M_p$$ 在该序列上做一次 forward：
    - 得到 $$γ$$ 个位置的 logit
    - 以及一个额外位置$$γ+1$$的 logits（免费，多出来的一个 token 分布）
3. 接受 / 拒绝（Accept / Reject）
  - 对 $$y_1..y_γ$$ 从左到右依次检查：
    - 若 $$M_p$$ 对 $$y_i$$ 的概率 ≥ $$M_q$$ 对 $$y_i$$ 的概率：直接接受
    - 否则，以概率 $$p_{large}(y_i) / p_{draft}(y_i)$$ 随机接受
  - 在第一次拒绝处停止，或全部通过，可以分为以下的两种情况：
  - 情况 A：$$k$$个草稿 token 被接受，第$$k+1$$个被拒：
    - 如果出现拒绝，则在该位置创建一个新的重新平衡分布$$p_{large}-p_{draft}$$，将最小值限制为 0，归一化使其总和为 1，并从中采样最后一个 token 。
    - 实际本轮推进 token 数为$$k+1$$至少 1
  - 情况 B：$$γ$$个草稿 token 全部通过：
    - 直接从大模型在位置 $$γ+1$$ 的 logits 采样下一个 token，因为大模型在一次前向里，已经顺带算出了第 $$k+1$$个位置的概率分布，当所有$$k$$个草稿 token 都被接受时，就可以直接在这个现成的分布上再抽一个 token，不用再多跑一次大模型。也就是说，如果这$$k$$个草稿全部被验证正确，目标模型在验证最后一个草稿 token 时，其实已经计算出了下一个位置的概率分布，自然也就可以免费再采样一个 Token，这就是 bonus token。
    - 本轮推进 token 数为$$γ+1$$

暂时无法在飞书文档外展示此内容
Bonus token 的前因后果：
输入位置 （Input）
模型看到的内容 （Context）
模型的输出 （Output Logits）
也就是在预测...
作用
Input （"今天"）
"今天"
Logits_1
预测 "今天" 后面是什么
用来验证 A （"天气"）
A （"天气"）
"今天" + "天气"
Logits_2
预测 "天气" 后面是什么
用来验证 B （"真"）
B （"真"）
"今天" + "天气" + "真"
Logits_3
预测 "真" 后面是什么
用来验证 C （"好"）
C （"好"）
"今天" + "天气" + "真" + "好"
Logits_4
预测 "好" 后面是什么
生成 Bonus D ！
推测解码的详细步骤如下：
1. 草稿：在当前上下文中运行小型模型并生成$$k$$个 token 
2. 验证：在上下文$$k$$个草稿 token 上运行一次大模型。这将为这些$$k$$个位置生成概率，外加一个额外位置（因此我们获得$$k+1$$个候选 token ）
3. 接受/拒绝：从左到右遍历$$k$$个草稿 token 。
  - 若大模型对草稿 token 的概率 ≥ 草稿自身的概率，则接受该 token
  - 否则，以$$p_{large}/p_{draft}$$的概率接受该 token
  - 在首次拒绝时停止，或接受所有$$k$$个草稿 token 。
    - 如果所有$$k$$个草稿 token 都被接受，则从大模型中免费额外采样第$$(k+1)$$个 token （我们已经计算过该分布）
    - 如果出现拒绝，则在该位置创建一个新的重新平衡分布（$$p_{large}-p_{draft}$$，将最小值限制为 0，归一化使其总和为 1），并从中采样最后一个 token 。
这一策略保证：
- 每轮至少前进 1 个 token（退化为纯大模型采样）
- 最好情况下，每轮前进$$γ+1$$个 token → 近似$$γ+1$$倍加速上界
- 严格意义上：生成序列分布与按 $$M_p$$逐 token 采样完全一致
投机解码原理说明：虽然我们使用小模型来提议候选 tokens，但接受/拒绝规则保证了在期望值上，序列的分布完全等同于从大模型中逐个 token 采样。这意味着推测性解码在统计上等同于标准自回归解码——但可能更快，因为单次大模型前向传播最多可生成$$k+1$$个 token。推测解码的可视化过程如下所示：
[图片]
[图片]
2.3 性能分析与边界
- 最坏情况：草稿 token 接受率很低
  - 相当于$$M_p$$串行运行，也就是说相当于直接调用$$M_p$$
  - 推理时间不比原始自回归差多少（每轮仍推进 ≥1 token）
- 最好情况：草稿 token 几乎全部通过
  - 每轮生成$$γ+1$$个 token，1就是奖励token
  - 大模型调用次数 ≈ 原来自回归的$$1 / (γ+1)$$
  - 解码加速比接近$$γ+1$$
- 实际情况：
  - 取决于大模型$$M_p$$与草稿模型$$M_q$$的分布相似度
  - 草稿越接近大模型，接受率越高，加速比越大
  - 实测多数方案可达 2–3× 解码加速
3. Medusa：无 Draft 小模型的多头并行解码
Speculative Decoding 需要额外部署小模型$$M_q$$，在工程上有明显问题：
- 多模型管理与部署复杂度增加
- 内存占用与调度开销更大
- 分布式环境中 pipeline/TP 配置更复杂
Medusa 提出了一种 单模型（One-Model）并行解码方案：在主干 LLM 顶部增加多个解码头（Medusa Heads），直接做 Next-Next-Token 预测，代替额外 Draft 小模型。
3.1 Medusa 整体流程
Medusa 仍遵循草稿‑验证框架：
[图片]
1. 生成候选序列
  - 利用 backbone LLM 的最后隐藏状态$$h_t$$
  - 通过原始 LM 头 + 多个 Medusa 头，一次预测：
    - $$p_t^{(0)}$$：位置 $$t+1$$ 的分布（原始 LM head）
    - $$p_t^{(1)}$$：位置 $$t+2$$ 的分布（Medusa head 1）
    - ...
    - $$p_t^{(K)}$$：位置 $$t+K+1$$ 的分布（Medusa head K）
2. 处理候选序列：Medusa 在原始大模型上叠加了多头结构，配合一种树形注意力（tree attention）来向前扩展多个候选 token。这样做的好处是：可以直接复用上一步已经算好的 logits 在下一步的子步骤里继续用起来，避免重复计算，大幅加速解码。
3. 验证候选序列：对这些候选序列，用拒绝采样（rejection sampling）或其他更简单的策略进行验证。目标不是一个个按常规生成，而是在保证最终分布无偏、质量不掉的前提下，一次性尽可能多地接受候选 token，从而在时间轴上并行推进多个步长，加快整体推理速度。
和 Speculative Decoding 一样：
- 本质仍是：用更高效的预测结构一次给出多步候选并统一验证
- 差别在于：Medusa 的 Draft 来自 同一个大模型的扩展头，而不是独立小模型
3.2 模型架构
常规的 decoding 过程称为 Next-Token 预测，Medusa 这种多 token 并行解码称为 Next-Next-Tokens 预测，其通过增加多个 Medusa Head，与原模型上的LM Head一同做预测。具体来说，给定原始模型在位置 $$t$$ 的最后隐藏状态张量$$h_
t$$，为其添加$$K$$个解码头。其中第$$k$$个头用于预测后续第$$(t+k+1)$$个位置的词元（原始语言模型头仍负责预测第$$(t+1)$$个位置）。第 k 个头的预测结果记为$$p_
t(k)$$，表示词表上的概率分布，而原始模型的预测结果记为$$p_
t(0)$$。
设在位置 t 的预训练 LLM 输出隐藏状态为：$$h_t ∈ ℝ^d$$，标准 LM Head 预测：$$p_t^{(0)} = softmax(W_{LM} h_t)$$，$$W_{LM} ∈ ℝ^{|V| × d}$$，而 Medusa 为每个未来 step 额外增加$$k$$个解码头，第$$k$$个解码头预测第$$t+k+1$$个位置的分布，形式为一个带残差连接的单层前馈：$$p_{t}^{(k)} = \mathrm{softmax}\left(W_2^{(k)} \cdot \left( \mathrm{SiLU}\left(W_1^{(k)} \cdot h_t\right) + h_t\right)\right) $$
- $$W_1^{(k)} ∈ ℝ^{d × d}$$
- $$W_2^{(k)} ∈ ℝ^{|V| × d}$$
- 初始化策略：
  - $$W_2^{(k)}$$ 初始化为与原 LM Head 相同的参数（拷贝）
  - $$W_1^{(k)}$$初始化为 0（即初始时输出等价于原 LM Head）
含义：
- 初始时，每个 Medusa 头基本与原 LM Head 等价（几乎不改变分布）
- 训练时通过微调$$W_1^{(k)}$$（和可选的 $$W_2^{(k)}$$）学习预测“更远位置”的分布
- 因为结构是浅层 FFN + 残差，参数量远小于 backbone，所以对大模型的主体架构几乎无侵入
3.3 训练策略与分布一致性
Medusa 头的训练有两种主要方案，这两种方法都能充分利用强大基础模型已习得的表征能力。
- Medusa‑1：仅训练 Medusa 头部，冻结骨干模型参数
- Medusa‑2：骨干模型与 Medusa 头共同训练（开销更大，效果更佳）
优势：
1. 单卡可行：新增参数只有若干线性层 + 残差块，即使 7B/13B 规模，仍可在单张高端 GPU 上微调。
2. 分布对齐：
  - 初始 $$W_2^{(k)}$$ 与 LM Head 相同
  - $$W_1^{(k)}$$ 初始为 0 ，输出分布起点与原模型一致
  - 在训练过程中，偏移相对可控，有利于减小 Draft / Target 分布差异
3.4 Medusa 模型结构
Medusa 模型结构代码实现如下（简略版本），从代码中可以看出：
- 权重初始化为 0：
  - 刚开始 linear(x) 输出 ≈ 0
  - act(0) ≈ 0 → forward(x) ≈ x
  - 即整个 ResBlock 初始为 近似恒等映射
- 训练过程中，linear.weight 从 0 逐步学到有意义的变换，形成轻量 decoder layer
class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """
    def __init__(self, config):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """super().__init__(config)    
        # 创建Medusa头模块列表# 每个Medusa头是一个序列：多个ResBlock + 最后的线性输出层
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),  # 多个残差块
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),  # 输出层，预测词表分布
                )
                for _ in range(medusa_num_heads)  # 创建指定数量的头
            ]
        )
在 self.medusa_head 是
- 是一个 ModuleList，每个元素是：
  - medusa_num_layers 个 ResBlock(hidden_size)（共享结构）
  - 外加一个输出层 Linear(hidden_size, vocab_size)
- 等价于论文中的多个$$p_t^{(k)}$$头
3.5 Medusa 模型输出测试
测试代码运行后，输出结果示例：
 Loading model...
 Running Normal Forward...
 Normal output logits shape: torch.Size([1, 4, 151936])
 Running Medusa Forward...
 Medusa logits shape: torch.Size([4, 1, 4, 151936])
从输出结果可以看出，测试代码执行了两种前向传播：
- 正常前向（Normal forward）：调用 Qwen3ForCausalLM 基类的 forward 方法，输出标准的 logits，形状为 [batch_size, seq_len, vocab_size]，即 [1, 4, 151936]。
- Medusa 前向（Medusa forward）：设置 medusa_forward=True 后，模型会先提取隐藏状态，再经过多个 Medusa 头处理。输出的 logits 形状为 [num_heads, batch_size, seq_len, vocab_size]，即 [8, 1, 4, 151936]。其中第一个维度 $$8$$ 表示 Medusa 头的数量，每个头分别对隐藏状态进行变换，并输出形状为 [1, 4, vocab_size] 的 logits。
完整的测试代码如下所示:
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig, Qwen2Config, Qwen2ForCausalLM # 假设是Qwen2，如果是Qwen3请保持原样
# 如果您的环境确实有 Qwen3，请使用您原来的导入：
# from transformers import AutoTokenizer, AutoConfig, PretrainedConfig, Qwen3Config, Qwen3ForCausalLM
from huggingface_hub import hf_hub_download
import os

# 为了代码能跑通，这里做一个别名处理，如果您确实有Qwen3Config请忽略这几行
try:
    Qwen3Config
except NameError:
    # 假设使用 Qwen2 作为替代（因为目前 HuggingFace 主要支持 Qwen2）
    print("Warning: Qwen3Config not found, aliasing Qwen2Config for demonstration.")
    from transformers import Qwen2Config as Qwen3Config
    from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM

class MedusaConfig(PretrainedConfig):
    """Medusa模型的配置类，用于定义模型结构参数"""
    model_type = "medusa"

    def __init__(self, base_model_name_or_path="Qwen/Qwen2-1.5B", medusa_num_heads=2, medusa_num_layers=1, **kwargs):
        # 基础模型路径
        self.base_model_name_or_path = base_model_name_or_path
        # Medusa头数量（预测头数量）
        self.medusa_num_heads = medusa_num_heads
        # Medusa层数（每个预测头的层数）
        self.medusa_num_layers = medusa_num_layers
        super().__init__(**kwargs)

class ResBlock(nn.Module):
    """残差块模块，包含线性层和SiLU激活函数"""
    def __init__(self, hidden_size):
        super().__init__()
        # 线性变换层，权重初始化为0
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        # SiLU激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        # 残差连接：输入 + 激活(线性变换(输入))
        return x + self.act(self.linear(x))

class MedusaModelQwen3(Qwen3ForCausalLM):
    """基于Qwen3的Medusa模型，用于加速推理的多头预测模型"""
    def __init__(self, config):
        # 1. 修复：必须显式调用父类初始化，且不能写在注释行里
        super().__init__(config)
        
        # 从配置中获取Medusa参数
        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        
        # 基础模型路径
        base_model_name_or_path = getattr(config, "base_model_name_or_path", "Qwen/Qwen2-1.5B")
        
        # 模型维度参数
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Medusa配置
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        
        # 加载tokenizer (通常建议在模型外部加载，但保持您的逻辑)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        
        # 构建Medusa多头预测层
        # 每个头包含：多个残差块 + 线性输出层
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),  # 多层残差块
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),  # 词汇表输出层
                )
                for _ in range(medusa_num_heads)  # 创建多个预测头
            ]
        )

    @property
    def base_model(self):
        """返回基础模型（自身）"""
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """从预训练模型加载，支持加载Medusa头权重"""
        try:
            # 尝试直接加载配置
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs, config=config
            )
        except:
            # 如果失败，使用Medusa配置加载
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            # 加载基础模型配置
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = config.medusa_num_heads
            base_model_config.medusa_num_layers = config.medusa_num_layers
            
            # 加载基础模型
            model = super().from_pretrained(
                config.base_model_name_or_path, *args, **kwargs, config=base_model_config
            )
            
            # 加载Medusa头权重
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            
            # 2. 修复：else 语句和逻辑分行
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path  # 本地文件
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")  # 从Hub下载
            
            # 加载权重到Medusa头
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model

    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """前向传播"""
        if not medusa_forward:
            # 普通模式：使用基础模型的前向传播
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        
        # Medusa模式：使用推理模式加速
        with torch.inference_mode():
            # 获取基础模型的隐藏状态
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            
            # 如果需要输出原始logits
            orig = None
            if output_orig:
                orig = self.lm_head(outputs[0])  # 基础模型的输出层
            
            # 复制隐藏状态用于Medusa头计算
            hidden_states = outputs[0].clone()
            
            # 计算所有Medusa头的logits
            medusa_logits = []
            for i in range(self.medusa):
                medusa_logits.append(self.medusa_head[i](hidden_states))
            
            # 3. 修复：return 语句分行
            if output_orig:
                # 返回：Medusa logits, 模型输出, 原始logits
                return torch.stack(medusa_logits, dim=0), outputs, orig
            
            # 返回：Medusa logits
            return torch.stack(medusa_logits, dim=0)

# 测试用例
if __name__ == "__main__":
    # 自定义模型权重名称或者路径
    # 注意：这里使用 Qwen2-1.5B 作为示例，因为 Qwen3 可能不存在
    base_model_name_or_path = "Qwen/Qwen2-1.5B-Instruct" 
    
    try:
        # 加载基础配置并设置Medusa参数
        base_config = Qwen3Config.from_pretrained(base_model_name_or_path)
        base_config.medusa_num_heads = 8  # 设置4个预测头
        base_config.medusa_num_layers = 1  # 每个头1层
        
        # 初始化模型
        model = MedusaModelQwen3(base_config)
        model.eval()  # 设置为评估模式
        
        # 获取tokenizer
        tokenizer = model.get_tokenizer()
        
        # 测试输入
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # 普通前向传播测试
        outputs_normal = model(**inputs)
        print("Normal output logits shape:", outputs_normal.logits.shape)
        
        # Medusa前向传播测试
        medusa_logits = model(**inputs, medusa_forward=True)
        print("Medusa logits shape:", medusa_logits.shape)
    except Exception as e:
        print(f"Run failed: {e}")
        print("Tip: Ensure you have access to the model on HuggingFace Hub and transformers installed.")
4. Eagle 投机解码
Eagle (Extrapolation Algorithm for Greater Language-model Efficiency) 及其后续版本 Eagle-3 是目前投机解码（Speculative Decoding）领域中最先进（SOTA）的技术之一。
核心思想：与直接预测 token 不同，EAGLE 预测的是目标模型在下一时刻会产生的隐藏状态（Hidden State），然后通过共享的 LM Head 将其映射回 token 空间。可以理解为是一种“插件式”的投机解码方法，它需要训练一个小型的“草稿网络”（Eagle Head），挂载在冻结的大模型（Base Model）之上。
Eagle 的结构非常简单：通常只是一层 Transformer Decoder Layer（加上一些 Projection 层）。它的参数量极小（通常 < 1B，甚至几百 MB）。
推理流程：特征外推 (Feature Extrapolation)
完整流程示例
假设目标模型刚刚输出了 Token $$x_t = \text{"sat"}$$ 及其对应的最后一层特征向量 $$H_t \in \mathbb{R}^{4096}$$（假设 hidden_size=4096）。
Target model ，draft model，用draft model去生成token，draft model可以接收target model的hidden states和已经生成的token id去生成新的draft token。
4.1 Step 1: 组合输入
EAGLE 需要两种输入：
1. 当前 Token 的 Embedding：$$E_t = \text{Embedding}(x_t) \in \mathbb{R}^{4096}$$
2. 目标模型的隐藏状态：$$H_t \in \mathbb{R}^{4096}$$（目标模型最后一层的输出）
EAGLE3（EAGLE 的增强版本），使用多层特征融合机制：EAGLE3 通过提取目标模型三个不同深度的隐藏状态来实现多层特征融合，覆盖低、中、高三个层次：
  - 第 2 层（低层）：捕捉基础的词汇级（token-level）特征。
  - num_layers//2 层（中层）：捕捉中间的语义级（semantic）特征。
  - num_layers-3 层（高层）：捕捉更深层的上下文感知（context-aware）特征。
输入组合方式：
- 标准 EAGLE：$$Input_{eagle} = Concat(E_t, H_t)$$ 或直接使用 $$H_t$$作为输入
- EAGLE3：$$Input_{eagle} = \text{Concat}(E_t, H_t^{(2)}, H_t^{(L//2)}, H_t^{(L-3)})$$ 或使用多层隐藏状态的加权组合
# 伪代码示例：EAGLE 输入准备
def prepare_eagle_input(target_token_id, target_hidden_state, method="eagle"):
    # 获取当前 token 的 embedding
    token_embedding = target_model.embed_tokens(target_token_id)
    
    if method == "eagle3":
        # EAGLE3: 组合多层隐藏状态
        token_embedding = target_model.embed_tokens(target_token_id)   # Et ∈ R^4096
        # 官方选取的三层（以 Llama-3-8B 为例，num_layers=32）
        low_layer_h  = target_hidden_states[2]           # low-level (early)
        mid_layer_h  = target_hidden_states[num_layers // 2]   # middle
        high_layer_h = target_hidden_states[num_layers - 3]    # high-level (near last)
        # 融合
        fused = torch.cat([low_layer_h, mid_layer_h, high_layer_h], dim=-1)  # 3×dim
        g = fc_layer(fused)   # Linear(3*dim → dim)，得到融合特征
        eagle_input = torch.cat([token_embedding, g], dim=-1)
    else:
        # 标准 EAGLE: 使用最后一层隐藏状态
        eagle_input = torch.cat([token_embedding, target_hidden_state], dim=-1)
        # 或者直接使用隐藏状态（如果 EAGLE 层设计为直接接受隐藏状态）
        # eagle_input = target_hidden_state
    
    return eagle_input
4.2 Step 2: 自回归生成特征
EAGLE 网络（通常是一个轻量级的 1-4 层 的 Transformer Decoder Layer 或 MLP）接收组合后的输入，预测下一时刻的特征向量, 即目标模型在位置$$t+1$$会产生的隐藏状态，而不是直接预测 token, 这个过程是自回归的：每一步的预测都依赖于前一步的结果：
$$\hat{H}_{t+1} = \text{EagleLayer}(Input_{eagle})$$
EAGLE3 放弃了特征预测，改为直接 token 预测 + Training-Time Test，进一步提升接受率。
EAGLE vs EAGLE3 的区别
暂时无法在飞书文档外展示此内容
def eagle_forward(eagle_input, eagle_model):
    """
    Eagle 网络前向传播
    
    Args:
        eagle_input: shape: [batch_size, input_dim] - 组合后的输入
        eagle_model: Eagle 网络（轻量级 Transformer 或 MLP）
    
    Returns:
        predicted_hidden_state: [batch_size, hidden_size] - 预测的隐藏状态
    """# Eagle 网络处理（可能是 Transformer Decoder Layer 或 MLP）
    predicted_hidden_state = eagle_model(eagle_input)
    
    return predicted_hidden_state
4.3 Step 3: 映射为 Token
直接利用目标模型共享的 LM Head（语言模型头），这样不需要为 EAGLE 单独训练一个 LM Head，直接复用目标模型的 LM Head。将预测出的特征向量映射到词汇表空间:
$$\hat{x}_{t+1} = \text{argmax}(\text{LM\_Head}(\hat{H}_{t+1}))$$
def hidden_to_token(predicted_hidden_state, target_model):
    """
    将预测的隐藏状态映射为 token
    
    Args:
        predicted_hidden_state: shape: [batch_size, hidden_size] - EAGLE 预测的隐藏状态
        target_model: 目标模型（共享 LM Head）
    
    Returns:
        draft_token_id: [batch_size] - 预测的 token ID
    """# 使用目标模型的 LM Head（共享）
    logits = target_model.lm_head(predicted_hidden_state)  # [batch_size, vocab_size]# 采样（通常使用 argmax 或 top-k）
    draft_token_id = logits.argmax(dim=-1)  # [batch_size]return draft_token_id
4.4 Step 4: 自回归循环生成
生成第一个草稿 token 后，EAGLE 继续自回归地生成后续 token：
初始状态: $$x_t$$ = "sat", $$H_t$$ = [目标模型的隐藏状态]
1. 第1轮:
  输入:$$E_t$$= Embedding("sat"), $$H_t$$
  预测: $$H_{t+1}$$ → $$x_{t+1}$$ = "on"
2. 第2轮:
  输入: $$E_{t+1}$$ = Embedding("on"), $$H_{t+1}$$
  预测: $$H_{t+2}$$ → $$x_{t+2}$$ = "the"
3. 第3轮:
  输入: $$E_{t+2}$$ = Embedding("the"), $$H_{t+2}$$
  预测: $$H_{t+3}$$ → $$x_{t+3}$$ = "mat"
4. 最终草稿序列: ["on", "the", "mat"]
自回归代码流程：
def eagle_propose(target_token_ids, target_hidden_states, num_draft_tokens=3):
    """
    EAGLE 自回归生成草稿 token
    
    Args:
        target_token_ids: [batch_size] - 目标模型当前输出的 token
        target_hidden_states: [batch_size, hidden_size] - 目标模型的隐藏状态
        num_draft_tokens: int - 要生成的草稿 token 数量
    
    Returns:
        draft_token_ids: [batch_size, num_draft_tokens] - 生成的草稿 token
    """
    batch_size = target_token_ids.shape[0]
    draft_token_ids = []
    
    # 当前状态
    current_token_ids = target_token_ids
    current_hidden_states = target_hidden_states
    
    for i in range(num_draft_tokens):
        # Step 1: 准备输入
        token_embeddings = target_model.embed_tokens(current_token_ids)
        eagle_input = torch.cat([token_embeddings, current_hidden_states], dim=-1)
        
        # Step 2: EAGLE 网络预测下一时刻的隐藏状态
        predicted_hidden_states = eagle_model(eagle_input)
        
        # Step 3: 通过共享的 LM Head 映射为 token
        logits = target_model.lm_head(predicted_hidden_states)
        next_token_ids = logits.argmax(dim=-1)
        
        draft_token_ids.append(next_token_ids)
        
        # Step 4: 更新状态，准备下一轮
        current_token_ids = next_token_ids
        current_hidden_states = predicted_hidden_states
    
    return torch.stack(draft_token_ids, dim=1)  # [batch_size, num_draft_tokens]
验证：树状注意力 (Tree Attention)
EAGLE 支持两种验证模式：线性验证（Linear Verification）和树状验证（Tree Verification）。树状验证是 EAGLE 的高级特性，能够进一步提升吞吐量。
线性验证 vs 树状验证
线性验证（标准模式）：
草稿序列: ["on", "the", "mat"]
验证方式: 顺序验证每个 token
  - 验证 "on" → 接受/拒绝
  - 如果接受，验证 "the" → 接受/拒绝
  - 如果接受，验证 "mat" → 接受/拒绝

特点: 简单直接，但只能生成一条路径
树状验证（Tree Attention）：
树状结构:
        "on"
       /    \
    "the"  "cat"    ← 第1层：多个候选
     / \    / \
  "mat" "was" "sat" "on"  ← 第2层：每个节点多个候选

验证方式: 并行验证整棵树
  - Base Model 一次性处理所有节点
  - 使用特殊的 Attention Mask，每个节点只能看到其父节点
  - 并行计算所有路径的 logits

特点: 可以探索多条路径，提高接受率
树状注意力的工作原理
1. 树结构构建
EAGLE 在生成草稿时，不是只生成一条线性序列，而是为每个位置生成多个候选 token，形成一棵树
import torch

def propose_tree(eagle_model, target_model, root_token_id, root_hidden_state, tree_config):
    """
    生成树状草稿结构 (逻辑修正版 - 非并行)
    
    Args:
        eagle_model: EAGLE 预测网络
        target_model: 目标大模型 (提供 embed_tokens 和 lm_head)
        root_token_id: [1] 当前确定的最后一个 token
        root_hidden_state: [1, hidden_size] 对应的 hidden state
        tree_config: 字典, e.g. {"branching_factor": [2, 3]}
    """
    tree_tokens = {}
    
    # Level 0: 根节点
    tree_tokens["level_0"] = [root_token_id]
    
    # Level 1: 从根节点预测下一层
    root_embedding = target_model.embed_tokens(root_token_id)
    eagle_input_l1 = torch.cat([root_embedding, root_hidden_state], dim=-1)
    
    hidden_l1 = eagle_model(eagle_input_l1)
    logits_l1 = target_model.lm_head(hidden_l1)
    
    topk_l1 = torch.topk(logits_l1, k=tree_config["branching_factor"][0], dim=-1)
    level_1_tokens = topk_l1.indices[0].tolist()
    tree_tokens["level_1"] = level_1_tokens

    # Level 2: 基于 Level 1 的结果继续分支
    level_2_tokens = []
    
    for parent_token_id in level_1_tokens:
        parent_token_tensor = torch.tensor([parent_token_id], device=root_hidden_state.device)
        parent_embedding = target_model.embed_tokens(parent_token_tensor)
        
        eagle_input_l2 = torch.cat([parent_embedding, hidden_l1], dim=-1)
        hidden_l2 = eagle_model(eagle_input_l2)
        
        logits_l2 = target_model.lm_head(hidden_l2)
        topk_l2 = torch.topk(logits_l2, k=tree_config["branching_factor"][1], dim=-1)
        
        children_tokens = topk_l2.indices[0].tolist()
        level_2_tokens.extend(children_tokens)
    
    tree_tokens["level_2"] = level_2_tokens
    
    return tree_tokens
2. Attention Mask 设计
树状验证的关键在于设计特殊的 Attention Mask，使得每个节点只能看到其父节点，从而实现并行验证：
树结构:
    0: "on" (根节点)
    /         \
  1: "the"   2: "cat"
  /    \      /    \
3:"mat" 4:"was" 5:"sat" 6:"on"

Attention Mask (下三角矩阵，1表示可见，0表示不可见):
位置:  0  1  2  3  4  5  6
  0  [1  0  0  0  0  0  0]  ← 根节点只能看到自己
  1  [1  1  0  0  0  0  0]  ← 节点1能看到根节点0
  2  [1  0  1  0  0  0  0]  ← 节点2能看到根节点0
  3  [1  1  0  1  0  0  0]  ← 节点3能看到路径 0→1→3
  4  [1  1  0  0  1  0  0]  ← 节点4能看到路径 0→1→4
  5  [1  0  1  0  0  1  0]  ← 节点5能看到路径 0→2→5
  6  [1  0  1  0  0  0  1]  ← 节点6能看到路径 0→2→6
什么能这么验证？ 这利用了 Transformer 的并行计算能力 和 Self-Attention 机制，把多个假设路径的逐个验证转换为了一次大规模矩阵运算。
1. 正常的自回归（串行验证） 如果你想验证路径 A (Root -> Cat -> Sat) 和路径 B (Root -> The -> Mat)，通常做法是： Step 1: 输入 Root，算 Cat 和 The 的概率。 Step 2 (Path A): 输入 Root, Cat，算 Sat 的概率。 Step 3 (Path B): 输入 Root, The。这样算各自的概率，速度上会很慢。
2. 树状掩码（并行验证） 现在我们将所有节点拍扁放在一起： [Root, Cat, The, Sat, Mat] (共5个token) 如果我们不做任何限制，Transformer 的 Self-Attention 会让每个词都看到所有其他的词（全连接），这显然不对（比如 Sat 不应该看到另一条分支的 The）。
  Tree Attention Mask 的作用就是切断这种非法的视线： 对于 Sat (属于 Cat 分支)： Mask 允许它看：Root, Cat (它的祖先) Mask 禁止它看：The, Mat (另一条分支) 这意味着，当模型计算 Sat 位置的输出时，它所利用的上下文仅仅是 Root -> Cat。这等价于单独把 Root -> Cat 输入给模型。
3. 并行验证
Base Model 使用树状 Attention Mask 一次性处理整棵树：
def verify_tree(target_model, tree_tokens, attention_mask):
    """
    并行验证树状草稿
    
    Args:
        tree_tokens: [num_tree_nodes] - 扁平化的树节点 token
        attention_mask: [num_tree_nodes, num_tree_nodes] - 树状注意力掩码
    
    Returns:
        logits: [num_tree_nodes, vocab_size] - 每个节点的 logits
    """# 将所有树节点拼接成一个序列
    input_ids = flatten_tree(tree_tokens)  # [num_tree_nodes]# Base Model 前向传播（使用树状 Attention Mask）# 关键：每个节点只能看到其父节点路径
    logits = target_model(
        input_ids=input_ids,
        attention_mask=attention_mask  # 树状掩码
    )
    
    return logits  # [num_tree_nodes, vocab_size]
4. 路径选择
验证完成后，需要从树中选择一条最优路径：
def select_best_path(tree_structure, logits, draft_probs):
    """
    从树中选择最优路径
    
    策略：
    1. 从根节点开始，逐层选择概率最高的分支
    2. 使用拒绝采样算法验证每个节点
    3. 一旦某个节点被拒绝，选择该节点的恢复 token
    """
    best_path = []
    current_node = root_node
    
    for level in range(tree_depth):
        # 获取当前节点的所有子节点
        children = get_children(current_node, level)
        
        # 对每个子节点进行拒绝采样
        accepted_children = []
        for child in children:
            if rejection_sample(child, logits[child], draft_probs[child]):
                accepted_children.append(child)
        
        if accepted_children:
            # 选择概率最高的子节点
            best_child = max(accepted_children, key=lambda x: logits[x])
            best_path.append(best_child)
            current_node = best_child
        else:
            # 所有子节点都被拒绝，使用恢复 token
            recovery_token = sample_from_logits(logits[current_node])
            best_path.append(recovery_token)
            breakreturn best_path
树状验证的优势
1. 更高的接受率：通过探索多条路径，即使某条路径被拒绝，仍可能在其他路径找到接受的 token
2. 更好的并行性：Base Model 一次性验证整棵树，充分利用 GPU 并行计算能力
3. 灵活的探索策略：可以根据不同策略（贪心、采样等）选择最优路径
vLLM 中的实现
在 vLLM 中，树状注意力通过 TreeAttentionMetadata 实现：
if isinstance(attn_metadata, TreeAttentionMetadata):
    # 使用树状注意力生成草稿
    draft_token_ids_list = self.propose_tree(
        batch_size=batch_size,
        logits=logits,
        positions=positions,
        hidden_states=hidden_states,
        common_attn_metadata=common_attn_metadata,
    )
    return torch.cat(draft_token_ids_list, dim=1)
总结对比
特性
线性验证
树状验证
草稿结构
单一路径
多路径树
验证方式
顺序验证
并行验证整棵树
接受率
中等
更高（多路径探索）
内存占用
较低
较高（需要存储整棵树）
实现复杂度
简单
复杂（需要特殊 Attention Mask）
适用场景
一般场景
对吞吐量要求极高的场景
总结
Eagle 是目前 最聪明 的投机解码方式。
- 它不像 N-Gram 那样只懂复制粘贴。
- 它不像 Medusa 那样盲目预测未来。
- 它真正模拟了 Base Model 的思维过程（Feature Level），用极小的代价实现了高质量的“预测未来”。在 vLLM 中，它是提升推理吞吐量的首选方案之一。
5. 推测解码的分类&设计
总结下推测解码的定义：推测解码是一种先推测后验证（Draft-then-Verify）的解码算法：在每个解码步，该算法首先高效地推测 target LLM 未来多个解码步的结果，然后用 target LLM 同时进行验证，以加速推理。
换句话说，所有符合在每个解码步高效推测->并行验证模式的推理算法，都可以称为是推测解码（或其变体）。推测解码实现加速的关键要素，主要在于如下三点：
- 相比 Decode 阶段的串行解码，推测解码的 LLM 并行计算额外引入的 latency 很小，甚至可以忽略；
- 推测的高效性&准确性：如何又快又准地推测 LLM 未来多个解码步的生成结果；
- 验证策略的选择：如何在确保质量的同时，让尽可能多的推测 token 通过验证，提高解码并行性。
LLM decode 阶段推理的主要 latency 瓶颈在于推理过程中权重参数的反复读写。如果在只考虑一个解码步的情况下，decoder-only LLM 的 forward latency 主要和 decoder 层数有关—层数越深，推理时间越长。相比于这两者，LLM 并行计算带来的额外latency很小。
根据推测（Drafting）的高效性和准确性，以及验证策略（Verification）的选择的算法研究分类，推测解码相关研究的归纳分类如下图所示：
[图片]
1. vllm 中的投机采样
暂时无法在飞书文档外展示此内容
6.1 什么是投机解码
之前的内容中我们说过，投机解码（Speculative Decoding 是近两年 LLM 推理加速的核心技术之一，本质理念是：用一个便宜的草稿生成过程先预测多步 token，再用昂贵的目标模型一次性验证，尽量减少“目标模型 decode 调用次数”。
对比：
- 标准自回归解码：每次只能生成 1 个 token，都要跑一次目标模型前向：$$[ \text{step } t: x_{1:t} \xrightarrow{\text{target model}} x_{t+1} ]$$
- 投机解码：
在 step t，先用草稿过程生成 k 个候选：$$[ \text{drafter: } x_{1:t} \to \hat{x}{t+1:t+k} ]$$然后目标模型一次性在序列$$[ x{1:t}, \hat{x}{t+1}, \dots, \hat{x}{t+k} ]$$上前向，验证这些 token，接受尽量长的前缀。
优势：
- 把多次 decode 合并为一次更宽的 decode 前向，在 GPU 上一般更高效。
- 合理设计草稿过程，可以在保持质量接近的同时显著提高 throughput。
6.2 vLLM v1 支持的投机解码方法
vLLM V1 框架支持多种投机解码方法。下面重点介绍三类常见方法：N-Gram、Medusa、EAGLE/EAGLE3。
1. N-Gram 投机解码
N-Gram 投机解码在 vLLM 配置中也和 prompt_lookup_min、prompt_lookup_max 相关。它是一种基于“上下文复用”的加速技术，不需要训练或加载额外的小模型（Draft Model），而是利用当前请求已有上下文中出现过的重复 token 模式来预测后续 token。
核心原理：N-Gram 投机解码基于一个观察：文本，尤其是代码、结构化文档、法律文本中，常常存在重复短语。比如，如果模型刚刚输出了 "Artificial Intelligence"，而这个词组在之前的上下文中出现过，并且后面紧跟着 " is a field"，那么这次后面也可能继续出现类似片段。
在 vLLM V1 的实现中，N-Gram proposer 会在当前 token 序列中查找与末尾后缀匹配的最长 n-gram，匹配长度受 prompt_lookup_min 和 prompt_lookup_max 控制。找到匹配后，它会从历史匹配位置之后取最多 num_speculative_tokens 个 token 作为 draft tokens。如果没有满足条件的 n-gram，就不生成草稿 token。
2. medusa 投机解码
Medusa 和传统大模型（Target Model）+ 小模型（Draft Model）的方案类似，它依赖已经训练好的 Medusa head/checkpoint，在目标模型的隐藏状态之上增加多个预测头，用多个 head 同时给出未来 token 的候选。
核心原理：Medusa 的直觉是让不同 head 分别负责不同步长的预测：
- Head 1：预测 t+1 位置的 token。
- Head 2：预测 t+2 位置的 token。
- ...
- Head K：预测 t+K 位置的 token。
在 vLLM V1 中，Medusa proposer 会加载 Medusa head，对 target model 输出的 hidden states 计算多个 head 的 logits，并对每个 head 取 top-1/argmax 作为 draft token。
3. eagle/eagle3 投机解码
EAGLE/EAGLE3 的核心思想是：不直接只靠一个小语言模型从 token 序列中逐步猜测，而是利用目标模型的 hidden states 作为草稿网络的输入，让一个专门训练的 EAGLE draft model/head 生成候选 token。
可以把它理解为一种插件式的投机解码方法：目标大模型仍然负责最终验证，EAGLE draft model/head 负责根据目标模型已经产生的隐藏状态快速提出候选 token。vLLM V1 中的 EagleProposer 会把 target hidden states 传给 drafter；EAGLE3 还可以使用多层辅助 hidden states。
方法
核心逻辑
缺点
标准投机 （Draft Model）
Token-level 自回归：独立 draft model 根据 token 序列逐步生成候选 token。
不直接使用 target model 的 hidden states；draft model 与 target model 的能力差距会影响接受率，并且需要额外 draft model 权重。
Medusa
非自回归：基于 target hidden state，通过多个 Medusa heads 同时预测多个未来 token。
后面位置的预测主要依赖同一个 hidden state，而不是逐步消费前一个 draft token，因此越靠后的候选通常更难保持准确。
Eagle
drafter 接收 target hidden states，并生成后续候选 token；EAGLE3 还可以使用多层辅助 hidden states。
需要专门的 EAGLE/EAGLE3 drafter 权重；实现比 ngram 以及其他投机采样方法更复杂，并可能带来额外 hidden-state 传递和计算开销。
这些方法的本质都是先提出候选 token，最终仍需要 target model 验证，并通过 rejection sampling 决定哪些候选可以接受。
不同的投机解码方法适合不同的部署业务场景，以下是一些选择建议总结：
方法
核心思想
优点
缺点
典型场景
ngram
基于当前历史 token 序列做 n-gram 后缀匹配，命中后直接复用历史片段中的后续 token。
无需额外模型权重，实现和部署成本低。

依赖上下文中是否存在可复用片段，泛化能力有限；需要配置 prompt_lookup_min / prompt_lookup_max 控制匹配窗口。
资源紧张、希望快速验证投机解码收益，或输入中重复片段较多的场景。
medusa
使用已训练好的 Medusa heads，基于 target hidden states 一次提出多个候选 token。
多个 head 可并行给出候选，推理路径相对直接。
需要专门 Medusa head 权重；模型结构和权重会增加额外显存占用。
已有 Medusa 权重，希望在较少业务改动下加速解码的场景。
eagle
使用 EAGLE drafter/head，并把目标模型 hidden states 传给 drafter 来生成候选 token。
drafter 能利用目标模型的 hidden-state 信息，通常比只看 token 的小模型更有信息量。
需要单独 EAGLE drafter/head 权重，工程和模型适配复杂度更高。
已有 EAGLE 权重，并愿意通过压测验证吞吐收益的在线推理场景。
eagle3
在 EAGLE 基础上进一步使用 auxiliary hidden states，为 drafter 提供更多中间层信息。
候选 token 能利用更多目标模型中间表示；在合适权重和模型支持下，有机会提升接受率。
默认会引入额外 hidden-state 输出、传递和合并计算；目标模型需要支持 EAGLE3 接口，部分 EAGLE3 head 也可能通过配置关闭 auxiliary hidden states。
有 EAGLE3 权重、目标模型支持相应接口，并能接受额外工程复杂度和压测验证成本的在线推理场景。
Hugging Face hub 上提供了多种 EAGLE 草稿模型：
[图片]
6.3 快速开始：使用示例
6.3.1 N-gram
vLLM v1 使用 --speculative_config 来设置所有与推测解码相关的配置。之前通过 --speculative_model 指定模型并单独添加相关参数（例如 --num_speculative_tokens ）的方法现已被弃用！！！ speculative_config 核心配置参数说明如下：
1. N-gram 投机解码方法离线推理示例，对应代码文件 test_ngram.py。
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config = {
    "method": "ngram",
    "prompt_lookup_min": 3,      # 在历史上下文中查找匹配时，N-gram 的最小长度
    "prompt_lookup_max": 5,      # 在历史上下文中查找匹配时，N-gram 的最大长度
    "num_speculative_tokens": 3, # 每个解码步骤最多提出 3 个候选 token
}

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    speculative_config=speculative_config,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
speculative_config 在内部会被封装为 SpeculativeConfig。其中最重要的配置包括：
- method: 指定投机解码方法，例如 ngram、eagle、eagle3、medusa 等。对于 N-gram，需要设置为 "ngram"。
- model: 指定草稿模型、EAGLE head、Medusa head 或其他额外权重。N-gram 不需要额外模型权重，因此通常不需要设置 model。
- num_speculative_tokens: 每次最多提出的候选 token 数量。候选机制越准确，可以尝试调大；如果候选经常被目标模型拒绝，调小它可以减少浪费的计算。
- prompt_lookup_min / prompt_lookup_max: 仅用于 N-gram 方法，控制在历史上下文中做后缀匹配时允许的最小和最大 n-gram 长度。
@config
class SpeculativeConfig:
    # 通用控制
    enforce_eager: bool | None = None
    num_speculative_tokens: int | None = None
    model: str | None = None
    method: SpeculativeMethod | None = None
    draft_tensor_parallel_size: int | None = None

    # drafter 模型配置
    quantization: me_quant.QuantizationMethods | str | None = None
    moe_backend: MoEBackend | None = None
    attention_backend: AttentionBackendEnum | None = None
    max_model_len: int | None = None
    revision: str | None = None
    code_revision: str | None = None

    # 高级控制
    disable_padded_drafter_batch: bool = False
    use_local_argmax_reduction: bool = False

    # ngram 专用
    prompt_lookup_max: int | None = None
    prompt_lookup_min: int | None = None

    # 并行 drafting
    parallel_drafting: bool = False

    # Engine 注入字段
    target_model_config: SkipValidation[ModelConfig] = None
    target_parallel_config: SkipValidation[ParallelConfig] = None

    # 初始化后生成
    draft_model_config: SkipValidation[ModelConfig] = None
    draft_parallel_config: SkipValidation[ParallelConfig] = None

    # 其他方法相关配置
    suffix_decoding_max_tree_depth: int = 24
    suffix_decoding_max_cached_requests: int = 10000
    draft_load_config: LoadConfig | None = None
    rejection_sample_method: RejectionSampleMethod = "standard"
    draft_sample_method: DraftSampleMethod = "greedy"
要点：
- 用户配置通常只需要关心前半部分，例如 method、model、num_speculative_tokens、prompt_lookup_min、prompt_lookup_max 等。
- 比如对于 ngram，通常只需要指定 method="ngram"、num_speculative_tokens，以及可选的 prompt_lookup_min / prompt_lookup_max。
- 对于 eagle、eagle3、medusa 等依赖额外权重的方法，通常还需要通过 model 指定 drafter、EAGLE head 或 Medusa head。
- 同时，Engine 创建 SpeculativeConfig 时会补入目标模型配置和并行配置，用于后续构造 draft model config、校验并行设置和初始化内部 proposer。
而上述代码code/course14/test_ngram.py直接运行即可，运行结果日志如下所示：
[图片]
虽然日志显示 "Loading drafter model"，但由于模式是 ngram，这里实际上是在初始化 N-Gram 匹配器所需的内部数据结构，而非加载实际的 drafter model 权重。
6.3.2   EAGLE3
投机解码方法离线推理示例，对应代码文件 test_eagle3.py。先下载 eagle3 和主模型权重，下载命令示例：
pip install modelscope
mkdir EAGLE3-LLaMA3.1-Instruct-8B
mkdir Meta-Llama-3.1-8B-Instruct
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir ./Meta-Llama-3.1-8B-Instruct
# 如果是下面的例子，请以yuhuili/EAGLE-LLaMA3.1-Instruct-8B作为draft模型
modelscope download --model vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B  --local_dir ./EAGLE3-LLaMA3.1-Instruct-8B

test_eagle3.py 代码如下，我本地运行了一下，又如下几个关键参数：
1. mean acceptance length: 2.02：表示每次 draft/verify 周期平均推进的 token 数。vLLM 中该指标的计算方式是：
mean acceptance length = 1 + accepted_draft_tokens / num_drafts
  其中：
  - num_drafts 表示发生了多少次 draft/verify 周期。
  - accepted_draft_tokens 表示这些周期中被 target model 接受的草稿 token 总数。
  - 公式中的 +1 表示每轮验证除了被接受的 draft token 外，还会推进 1 个由 target model 确定的 token。
因此，2.02 可以理解为：主模型每完成一次验证周期，平均推进约 2.02 个 token。
2. acceptance at token 0: 0.69：表示第 1 个 draft 位置的候选 token 有约 69% 的比例被 target model 接受。
3. acceptance at token 1: 0.33：表示第 2 个 draft 位置的候选 token 接受率约为 33%。通常越靠后的 draft token 越难预测，因此接受率下降是常见现象。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/somefish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/starfish.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/snail.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/thistle.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/husky.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/orangetabbycat.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/guineapig.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/rabbit.jpg",
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/horsepony.jpg",
]


def get_custom_mm_prompts(num_prompts):
    prompts = []
    for url in IMAGE_URLS:
        prompts.append(
            [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": QUESTION},
            ]
        )
    if num_prompts > len(IMAGE_URLS):
        prompts = prompts * (num_prompts // len(IMAGE_URLS) + 1)

    return [[{"role": "user", "content": prompt}] for prompt in prompts[:num_prompts]]


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="eagle",
        choices=["ngram", "eagle", "eagle3", "mtp"],
    )
    parser.add_argument("--num-spec-tokens", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--eagle-dir", type=str, default=None)
    parser.add_argument("--custom-mm-prompts", action="store_true")
    return parser.parse_args()


def main(args):
    args.endpoint_type = "openai-chat"

    model_dir = args.model_dir
    if args.model_dir is None:
        if args.custom_mm_prompts:
            raise ValueError(
                "custom_mm_prompts requires mm based models"
                "default llama3.1-8b-instruct is not mm based"
                "please specify model_dir to give a mm based model"
            )
        model_dir = "/root/vllm_learn/Meta-Llama-3.1-8B-Instruct"
    print(f"model_dir: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    args.custom_skip_chat_template = True

    if not args.custom_mm_prompts:
        prompts = get_samples(args, tokenizer)
        # add_special_tokens is False to avoid adding bos twice
        # when using chat templates
        prompt_ids = [
            tokenizer.encode(prompt.prompt, add_special_tokens=False)
            for prompt in prompts
        ]
    else:
        prompts = get_custom_mm_prompts(args.num_prompts)

    if args.method == "eagle" or args.method == "eagle3":
        eagle_dir = args.eagle_dir
        if args.method == "eagle" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

        elif args.method == "eagle3" and eagle_dir is None:
            eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:
        raise ValueError(f"unknown method: {args.method}")

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.9,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 5},
        disable_chunked_mm_input=True,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
    if not args.custom_mm_prompts:
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=x) for x in prompt_ids],
            sampling_params=sampling_params,
        )
    else:
        outputs = llm.chat(prompts, sampling_params=sampling_params)

    # print the generated text
    if args.print_output:
        for output in outputs:
            print("-" * 50)
            print(f"prompt: {output.prompt}")
            print(f"generated text: {output.outputs[0].text}")
            print("-" * 50)

    metrics = llm.get_metrics()

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")

    return acceptance_length


if __name__ == "__main__":
    args = parse_args()
    acceptance_length = main(args)

    if args.test:
        # takes ~30s to run on 1xH100
        assert args.method in ["eagle", "eagle3"]
        assert args.tp == 1
        assert args.num_spec_tokens == 3
        assert args.dataset_name == "hf"
        assert args.dataset_path == "philschmid/mt-bench"
        assert args.num_prompts == 80
        assert args.temp == 0
        assert args.top_p == 1.0
        assert args.top_k == -1
        assert args.enable_chunked_prefill

        # check acceptance length is within 2% of expected value
        rtol = 0.02
        expected_acceptance_length = 2.296 if args.method == "eagle" else 2.811

        assert (
            acceptance_length <= (1 + rtol) * expected_acceptance_length
            and acceptance_length >= (1 - rtol) * expected_acceptance_length
        ), (
            f"acceptance_length {acceptance_length} is not "
            f"within {rtol * 100}% of {expected_acceptance_length}"
        )

        print(
            f"Test passed! Expected AL: "
            f"{expected_acceptance_length}, got {acceptance_length}"
        )
[图片]
2. 投机解码核心组件与目录结构
7.1 核心组件概览
vLLM v1 的投机解码实现位于 vllm/v1/spec_decode/ 目录，包含以下核心组件：
文件名
功能描述
核心类
metadata.py
定义投机解码元数据结构
SpecDecodeMetadata
metrics.py
投机解码性能指标统计
SpecDecodingStats, SpecDecodingLogging
ngram_proposer.py
N-gram 提议器实现
NgramProposer
medusa.py
Medusa 提议器实现
MedusaProposer
eagle.py
EAGLE/EAGLE3 提议器实现
EagleProposer
utils.py
工具函数（采样参数检查等）
is_spec_decode_unsupported
核心数据结构：SpecDecodeMetadata
SpecDecodeMetadata 是 vLLM V1 投机解码中的核心元数据结构。它把不同请求的 draft token、验证位置和额外采样位置整理成扁平化索引，方便 GPU 在统一 batch 上执行 target model logits 计算和 rejection sampling。
为什么需要它？
在同一个 batch 中，不同请求当前 step 的 draft token 数量可能不同：
- 请求 1：2 个 draft token
- 请求 2：0 个 draft token
- 请求 3：3 个 draft token
但 target model 验证时需要统一处理这些位置。SpecDecodeMetadata 负责记录每个请求的 draft token 边界，以及这些 token 在扁平 logits 中对应的位置。
它主要覆盖两个环节：
1. 准备 logits 计算位置：logits_indices 指定 target model 前向后，哪些 hidden states 需要取出来计算 logits。这些位置包括 draft token 的验证位置，以及每个请求额外的 1 个采样位置。
2. rejection sampling，该阶段会使用：
  - draft_token_ids：本轮 drafter 提出的候选 token。
  - cu_num_draft_tokens：每个请求的 draft token 边界。
  - target_logits_indices：用于验证 draft token 的 target logits 位置。
  - bonus_logits_indices：每个请求额外采样位置的 logits。
这些字段共同描述了：候选 token 是什么、每个请求要验证哪些候选、target logits 中哪些位置用于验证、哪些位置用于额外采样。rejection sampler 据此判断 draft token 是否接受，并组装每个请求本轮最终输出的 token。
这里给出的是 logits 索引，而不是具体 token ID，是因为 SpecDecodeMetadata 处理的是 target model 前向后的扁平 logits 张量。具体输出哪个 token，还需要 sampler 根据这些 logits 做 argmax 或采样决定。因此，target_logits_indices 和 bonus_logits_indices 记录的是去 logits 张量的哪些位置取分布，不是已经确定的 token ID。
例如，当前 batch 有 3 个请求，它们的 draft token 数量分别是 [2, 0, 3]。vLLM 会为每个请求额外保留 1 个采样位置，因此每个请求需要处理的位置数是：
num_sampled_tokens = num_draft_tokens + 1
                  = [3, 1, 4]
扁平化后的 logits 槽位可以理解为：
请求 1: [0, 1, 2]     其中 0、1 用于验证 2 个 draft token，2 是额外采样位置
请求 2: [3]           没有 draft token，3 是常规采样位置
请求 3: [4, 5, 6, 7]  其中 4、5、6 用于验证 3 个 draft token，7 是额外采样位置
因此：
target_logits_indices = [0, 1, 4, 5, 6]
bonus_logits_indices  = [2, 3, 7]
target_logits_indices 指向用于验证 draft token 的 logits 位置；bonus_logits_indices 指向每个请求额外采样(bonus token)位置的 logits。最终 sampler 会根据这些 logits 分布决定哪些 draft token 被接受，以及每个请求本轮输出哪些 token。
数据结构示例：
暂时无法在飞书文档外展示此内容
假设有 3 个请求，draft token 数量分别为 [2, 0, 3]。 每个请求除了 draft token 的验证位置外，还会额外保留 1 个采样位置，因此：
num_sampled_tokens = num_draft_tokens + 1
                   = [3, 1, 4]
可以把扁平 logits 槽位理解为，此处的额外采样即是bonus token。
请求 1: [0, 1, 2]     0、1 用于验证 draft token，2 是额外采样位置
请求 2: [3]           没有 draft token，3 是常规采样位置
请求 3: [4, 5, 6, 7]  4、5、6 用于验证 draft token，7 是额外采样位置
对应的数据结构可以简化理解为：
@dataclass
class SpecDecodeMetadata:
    """投机解码元数据，包含验证和采样所需的扁平化索引。"""

    # 扁平化后的 draft token ID。
    # 请求 1 有 2 个，请求 2 有 0 个，请求 3 有 3 个。
    draft_token_ids: torch.Tensor

    # 每个请求的 draft token 数量。
    # 例如：[2, 0, 3]
    num_draft_tokens: list[int]

    # num_draft_tokens 的前缀和。
    # 例如：[2, 2, 5]
    # 请求 1 的 draft token 在 [0, 2)
    # 请求 2 没有 draft token
    # 请求 3 的 draft token 在 [2, 5)
    cu_num_draft_tokens: torch.Tensor

    # num_sampled_tokens 的前缀和。
    # num_sampled_tokens = num_draft_tokens + 1
    # 例如：[3, 4, 8]
    cu_num_sampled_tokens: torch.Tensor

    # 用于验证 draft token 的 logits 位置，对应上面的扁平 logits 槽位：
    # 请求 1 验证位置是 0、1
    # 请求 3 验证位置是 4、5、6
    # 因此：[0, 1, 4, 5, 6]
    target_logits_indices: torch.Tensor

    # 每个请求额外采样位置的 logits 位置。
    # 请求 1 是 2，请求 2 是 3，请求 3 是 7。
    # 因此：[2, 3, 7]
    bonus_logits_indices: torch.Tensor

    # 从 target model 输出的 hidden states 中选择哪些位置计算 logits。
    # 长度等于 sum(num_draft_tokens + 1)。
    logits_indices: torch.Tensor
7.2 三种提议器的核心实现
1. NgramProposer( ngram_proposer.py) 
  NgramProposer 以 CPU 上维护的 token_ids_cpu 序列为主要输入，基于当前上下文末尾的 n-gram 在历史 token 中查找最长匹配；如果找到匹配，就把该历史匹配位置之后的最多 k 个 token 作为草稿 token。整个提案过程使用 Numba JIT 做批量加速，不需要加载任何模型权重，本质上是纯 token 序列匹配逻辑。
class NgramProposer:
    def propose(self, sampled_token_ids, num_tokens_no_spec,
                token_ids_cpu, slot_mappings=None):
        """
        为每个请求生成草稿 token。

        核心流程：
        1. 过滤有效请求：
           - 跳过本轮没有 sampled token 的请求
           - 跳过已经达到 max_model_len 的请求
        2. 对有效请求批量执行 n-gram 提案
        3. batch_propose 内部调用 Numba JIT 函数加速匹配
        """
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if len(sampled_ids) == 0:
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                continue

            valid_ngram_requests.append(i)

        return self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )
  核心匹配逻辑：
context_token_ids = token_ids_cpu[idx, :num_tokens]

drafter_output = _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens=context_token_ids,
    min_ngram=self.min_n,
    max_ngram=self.max_n,
    max_model_len=self.max_model_len,
    k=self.k,
)
  _find_longest_matched_ngram_and_propose_tokens 会查找当前上下文后缀中长度位于 [prompt_lookup_min, prompt_lookup_max] 的最长 n-gram 匹配；如果找到，就从历史匹配位置之后截取最多 num_speculative_tokens 个 token 作为草稿 token。
  例如当前上下文 token 是：[10, 20, 30, 40, 50, 30, 40]，假设：
prompt_lookup_min = 2
prompt_lookup_max = 3
num_speculative_tokens = 2
  当前上下文的末尾是：[30, 40]，而NgramProposer会在更早的历史 token 中查找这个后缀 n-gram。这里 [30, 40] 曾经出现在位置 2-3：
[10, 20, 30, 40, 50, 70, 30, 40]
         ^^^^^^          ^^^^^^
         历史匹配         当前后缀
  历史匹配 [30, 40] 后面跟着的 token 是 [50, 70]，如果 num_speculative_tokens = 1，它会返回 [50] 作为草稿 token；如果 num_speculative_tokens = 2，则会返回 [50, 70]
2. MedusaProposer - 基于多个预测头
  - MedusaProposer 依赖目标模型输出的 sample_hidden_states。在包含 draft tokens 的 speculative decoding 场景中，vLLM 会先按每个请求当前采样 token 的位置选出对应 hidden states，再传给 Medusa proposer。
  - Medusa proposer 会加载一个 draft model，也就是 Medusa head 模型；它接收目标模型 hidden states，并通过多个 Medusa head 并行预测后续 token。
  - 每个 Medusa head 输出一组 logits，对每个 head 的 logits 执行 argmax(dim=-1)，也就是沿着最后一维 vocab_size 找最大值的位置，返回对应的 token id，然后用 torch.stack(..., dim=1) 组合成 [batch_size, num_heads] 的草稿 token 张量。
class MedusaProposer:
    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        slot_mappings=None,
    ) -> torch.Tensor:
        """
        基于目标模型 hidden states 生成草稿 token。

        核心流程：
        1. 将目标模型 hidden states 输入 Medusa head 模型
        2. 多个 Medusa head 分别输出 logits
        3. 对每个 head 的 logits 取 argmax，得到该 head 的预测 token
        4. 将结果堆叠成 [batch_size, num_heads] 的 draft token 张量
        """
        # Medusa head 前向传播
        blocks = self.model(target_hidden_states)

        # 每个 head 输出一组 logits
        logits = self.model.compute_logits(blocks)

        # 对每个 head 的 logits 取 argmax，并堆叠为 [batch_size, num_heads]
        draft_tokens = torch.stack(
            [logit.argmax(dim=-1) for logit in logits],
            dim=1,
        )

        return draft_tokens
  也就是说，MedusaProposer 不是再跑一个完整的自回归 draft model，而是利用目标模型已经算出的 hidden states，通过外接的多个 Medusa head 一次性给每个 batch 样本预测一串草稿 token。
3. EagleProposer - 使用轻量级草稿模型
  EAGLE及其后续版本EAGLE3 是投机解码（Speculative Decoding）中的代表性高性能方法。它的核心思想不是直接用一个独立小模型从 token 序列预测下一个 token，而是复用目标模型已经产生的 hidden states，让轻量 drafter 基于这些 hidden states 和 token ids 继续生成草稿 token。
  简单理解：EAGLE 使用一个单独的轻量 drafter 模型，接收目标模型最近一次 forward 产生的 hidden states 和对应 token ids 作为输入；drafter forward 会先产生 draft hidden states，再通过 LM head / logits processor 映射到 token 空间，最终得到草稿 token。
  EAGLE3 还会额外使用目标模型的辅助 hidden states，vLLM 会先将多层辅助 hidden states 拼接起来，再通过 combine_hidden_states 中的线性层等结构融合后送入 EAGLE3 drafter。需要注意的是，draft 模型的结构和参数量取决于具体 checkpoint 与配置，不能简单概括为固定一层 Transformer Decoder Layer。
暂时无法在飞书文档外展示此内容
def propose(...):
    """
    使用 EAGLE draft model 自回归生成 K 个草稿 token。
    输出形状: [batch_size, K]
    """
    
    # 1. 构造第一次 draft forward 的输入：
    # 将 target_token_ids 左移一位，并用 next_token_ids
    # 补齐每个 request 的最后位置，使 input_ids 与 target_hidden_states 对齐。
    if token_indices_to_sample is None:
        token_indices_to_sample = cad.query_start_loc[1:] - 1

    self.input_ids[:num_tokens - 1] = target_token_ids[1:]
    self.input_ids[token_indices_to_sample] = next_token_ids
    self.hidden_states[:num_tokens] = target_hidden_states

    # 2. 第一次 draft forward
    ret_hidden_states = self.model(
        input_ids=self.input_ids[:num_input_tokens],
        positions=self._get_positions(num_input_tokens),
        hidden_states=self.hidden_states[:num_input_tokens],
    )

    last_hidden_states, hidden_states = ret_hidden_states

    # 3. 只在 token_indices_to_sample 对应的位置生成第 1 个 draft token
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    draft_token_ids, draft_probs = self._sample_draft_tokens(
        sample_hidden_states,
        sampling_metadata,
    )

    draft_token_ids_list = [draft_token_ids]

    # 4. 自回归生成剩余 K-1 个 draft token
    hidden_states = hidden_states[token_indices_to_sample]

    for _ in range(K - 1):
        input_ids = draft_token_ids_list[-1].int()

        self.input_ids[:batch_size] = input_ids
        self.hidden_states[:batch_size] = hidden_states

        ret_hidden_states = self.model(
            input_ids=self.input_ids[:input_batch_size],
            positions=self._get_positions(input_batch_size),
            hidden_states=self.hidden_states[:input_batch_size],
        )

        last_hidden_states, hidden_states = ret_hidden_states
        hidden_states = hidden_states[:batch_size]

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            last_hidden_states[:batch_size],
            sampling_metadata,
        )
        draft_token_ids_list.append(draft_token_ids)

    return torch.stack(draft_token_ids_list, dim=1)
整体流程分为三步：
1. 构造初始输入（对齐到起草位置）
EAGLE 路径下，vLLM 会先把 target_token_ids 左移一位，写入 self.input_ids[:num_tokens - 1]，再用每个 request 最新确定的 next_token_ids 补齐各自的采样位置 token_indices_to_sample。同时，target model 最近一次 forward 得到的 target_hidden_states 会写入 EAGLE drafter 的 hidden state buffer。这样做的目的，是让 drafter 在每个位置看到当前位置的 target hidden state + 下一个 token id，从而完成初始输入的构造。
  例如 batch 中有 3 个 request，target model 本轮处理过的 token 展平后是：
target_token_ids = [a1, b1, b2, c1, c2, c3]
  每个 request 的最后采样位置是：
token_indices_to_sample = [0, 2, 5]
  先左移一位：
self.input_ids[:num_tokens - 1] = target_token_ids[1:]

self.input_ids = [b1, b2, c1, c2, c3, ?]
  再用每个 request 最新确定的 token 补齐边界位置：
next_token_ids = [a2, b3, c4]

self.input_ids[token_indices_to_sample] = next_token_ids

self.input_ids = [a2, b2, b3, c2, c3, c4]
  按 request 看就是：
request A: [a2]
request B: [b2, b3]
request C: [c2, c3, c4]
  这表示 drafter 会用左移后的 token + 最新确认 token作为输入，与对应的 target_hidden_states 对齐后生成下一批 draft tokens。
2. 第一次前向生成第 1 个草稿 token
EAGLE draft model 先执行一次 forward，得到 draft model 的 last_hidden_states 和 hidden_states，vLLM随后从 last_hidden_states[token_indices_to_sample] 中取出每个 request 需要采样的位置，通过 _sample_draft_tokens 得到第 1 个 draft token。
  默认 greedy 情况下，这一步内部会执行 compute_logits(hidden_states).argmax(dim=-1)；如果启用了 probabilistic draft sampling，则会按概率采样。
  例如一个 batch 里有 3 个 request，第一次 draft forward 后：
last_hidden_states = [
  h_a1,
  h_b1, h_b2,
  h_c1, h_c2, h_c3,
]
  展平后每个 request 需要采样的位置是：token_indices_to_sample = [0, 2, 5]，那么：sample_hidden_states = last_hidden_states[token_indices_to_sample]，这样的方式取出的就是：[h_a1, h_b2, h_c3]
  也就是每个 request 的最后采样位置。然后 vLLM 对这些 hidden states 调用：
_sample_draft_tokens(sample_hidden_states, sampling_metadata)
  得到第 1 个 draft token：
request A -> draft token a2
request B -> draft token b3
request C -> draft token c4
3. 自回归循环生成剩余草稿 token
后续每一步都会把上一步生成的 draft token 作为新的 input_ids，并把上一步 draft model 返回的 hidden_states 作为下一步输入的一部分。每次 forward 只为每个 request 推进一个 draft 位置，再通过 logits / sampling 得到下一个 draft token。循环结束后，vLLM 用 torch.stack(draft_token_ids_list, dim=1) 得到形状为 [batch_size, K] 的草稿 token 张量。
7.3 元数据结构 SpecDecodeMetadata
@dataclass
class SpecDecodeMetadata:
    """投机解码元数据，用于对齐 draft token、target logits 和 bonus logits。"""

    # 扁平化后的所有 draft token id。
    draft_token_ids: torch.Tensor

    # 每个 request 的 draft token 数量。
    num_draft_tokens: list[int]

    # num_draft_tokens 的前缀和，
    # 用于定位每个 request 在 draft_token_ids 中的起止区间。
    cu_num_draft_tokens: torch.Tensor

    # 每个 request 需要计算的 logits 数量的前缀和。
    # 每个 request 的 logits 数量 = num_draft_tokens + 1，
    # 其中 +1 是 bonus / 修正采样位置。
    cu_num_sampled_tokens: torch.Tensor

    # 从压缩后的 logits 矩阵中，取哪些行用于校验 draft token。
    target_logits_indices: torch.Tensor

    # 从压缩后的 logits 矩阵中，取哪些行作为 bonus token 的采样位置。
    bonus_logits_indices: torch.Tensor

    # 从 target model 本轮 hidden_states 中取哪些位置来计算 logits。
    logits_indices: torch.Tensor
简单整理下：
- draft_token_ids：扁平化后的所有 draft token id，顺序与 target_logits_indices 对应。
- num_draft_tokens / cu_num_draft_tokens：记录每个 request 有多少个 draft token，并用前缀和定位每个 request 在扁平 draft token ids 数组中的起止区间。
- cu_num_sampled_tokens：记录每个 request 的 num_draft_tokens + 1 的前缀和，用于定位本轮实际计算出的 logits 矩阵中，每个 request 对应的采样区间。
- logits_indices：从 target model 本轮 hidden_states 中选择哪些位置来计算 logits，长度为 sum(num_draft_tokens) + batch_size。
- target_logits_indices：在本轮实际计算出的 logits 矩阵中，哪些行用于校验 draft tokens。
- bonus_logits_indices：在本轮实际计算出的 logits 矩阵中，哪些行是各 request 的 bonus 采样位。
假设一个 batch 有 3 个 request：num_draft_tokens = [2, 0, 3]，表示：
1. request 1 有 2 个 draft token
2. request 2 没有 draft token
3. request 3 有 3 个 draft token
那么所有 draft token 会被扁平化成draft_token_ids = [r1_d1, r1_d2, r3_d1, r3_d2, r3_d3]，那么对应的 draft token 数量前缀和是：cu_num_draft_tokens = [2, 2, 5]，所以每个 request 的 draft token 区间是：
request 1: draft_token_ids[0:2] = [r1_d1, r1_d2]
request 2: draft_token_ids[2:2] = []
request 3: draft_token_ids[2:5] = [r3_d1, r3_d2, r3_d3]
投机解码校验时，每个 request 除了校验 draft tokens，还需要多算 1 个 bonus token：
num_sampled_tokens = num_draft_tokens + 1 = [3, 1, 4]
cu_num_sampled_tokens = [3, 4, 8]
因此使用logits_indices索引提取hidden_states后的结果矩阵一共有 8 行：
logits =
[
  r1_verify_d1,   # row 0
  r1_verify_d2,   # row 1
  r1_bonus,       # row 2

  r2_bonus,       # row 3

  r3_verify_d1,   # row 4
  r3_verify_d2,   # row 5
  r3_verify_d3,   # row 6
  r3_bonus,       # row 7
]
于是：
target_logits_indices = [0, 1, 4, 5, 6]
bonus_logits_indices  = [2, 3, 7]
含义是：
1. target_logits_indices 取出的 logits 用来校验 draft_token_ids
2. bonus_logits_indices 取出的 logits 用来在 draft 全接受时采样额外 token (bonus token)
对应关系是：
draft_token_ids[0] = r1_d1  <-> logits[0]
draft_token_ids[1] = r1_d2  <-> logits[1]
draft_token_ids[2] = r3_d1  <-> logits[4]
draft_token_ids[3] = r3_d2  <-> logits[5]
draft_token_ids[4] = r3_d3  <-> logits[6]
每个 draft_token_id 都会对应一行 target model 计算出的 logits，vLLM 会用这行 logits 来判断该 draft token 是否应该被接受。
这里的 8 行 logits 由 logits_indices 决定：它指定从 target model 本轮 hidden_states 中取哪些位置来计算 logits。在此基础上，target_logits_indices表示本轮实际计算出的 logits 中，哪些行用于校验 draft token；bonus token 的采样行则由 bonus_logits_indices 指定。
7.4 拒绝采样器 RejectionSampler
位于 vllm/v1/sample/rejection_sampler.py 的 RejectionSampler 负责用 SpecDecodeMetadata 和 target model 本轮计算出的 logits 完成投机解码的接受 / 拒绝流程：
1. 它先根据 bonus_logits_indices 取出 bonus 位置的 logits，并预先采样 bonus token；然后根据 target_logits_indices 取出草稿验证位置的 logits，经过 logits processor、temperature、top-k / top-p 等采样约束处理后得到 target_logits，而target_logits的作用是作为 target model 的标准答案分布，用来校验 draft token 是否可以接受。
2. 最后把 draft_token_ids、draft_probs、target_logits、bonus_token_ids 以及相关前缀和元数据传入 rejection_sample()。rejection_sample() 会把 target model 的分布当作校验标准，逐个检查 draft token 是否可接受；greedy 时直接比较 argmax，非 greedy 时用概率分布做接受 / 拒绝采样。具体的流程我们会在之后的章节中展开。
  我们举个例子非 greedy 情况下的例子，不只是比较 argmax，而是看 target model 分布中 draft token 的概率。比如第一个 draft token 是 A：
draft_probs[A] = 0.40
target_probs[A] = 0.32
  接受概率大致由 target / draft 的比例决定：
accept_prob = min(1, target_probs[A] / draft_probs[A])
            = min(1, 0.32 / 0.40)
            = 0.8
  如果随机数小于 0.8，就接受 A；否则拒绝，并从 target 分布重采样一个 token。
8. 推理主流程总览
暂时无法在飞书文档外展示此内容
对于某个Batch的六个核心阶段：
1. 草稿生成（Draft Generation）：scheduler 取出上一轮生成的 draft tokens，放入本轮 batch 进行验证。
2. 批次构建与元数据计算（Batch + Metadata）：vLLM 拼接 draft tokens，并计算 SpecDecodeMetadata，记录 logits 计算和校验所需的索引。
3. 模型前向（Model Forward）：target model 执行 forward，vLLM 按 logits_indices 取 hidden states，并通过 compute_logits() 得到验证用 logits。
4. 奖励采样 + 拒绝采样（Bonus + Rejection Sampling）：
  rejection sampler 先采样 bonus token，再用 target_logits 校验 draft tokens：greedy 比较 argmax，非 greedy 按概率接受或拒绝。
5. 账务同步（Bookkeeping）：_bookkeeping_sync() 将 GPU 采样结果整理成 CPU 可见输出，scheduler 再更新 request 状态、输出 token、num_computed_tokens 和 stop 条件。
6. 下一轮草稿准备（Next Draft）：vLLM 生成下一轮 draft 并写回 request。部分投机采样方法可用 GPU sampled_token_ids 在 bookkeeping 前生成，部分投机采样方法依赖 CPU token 序列，通常在 bookkeeping 后生成。
8.1 草稿生成阶段
草稿生成是投机解码的第一个阶段，发生在每次推理循环的开始，为当前批次的所有请求快速生成候选 token，供后续阶段验证和采样。草稿生成在两种情况下被触发：
1. 首次生成（Prefill 阶段）：在第一次解码之前，基于 prompt 生成初始草稿 token
2. 后续生成（Decode 阶段）：在每次采样完成后，基于已接受的 token 生成下一轮草稿 token
调用位置：
# 位置：vllm/v1/worker/gpu_model_runner.py 的 propose_draft_token_ids 方法
# 该方法在每次采样完成后被调用，为下一轮推理生成草稿token
draft_token_ids = self.propose_draft_token_ids(...)
我们以 n-gram 方法为例，N-gram 草稿生成为例：
核心原理： 基于历史序列中的 n-gram 模式匹配，无需模型权重，速度最快。
输入：
- sampled_token_ids： 已接受的 token 序列（每个请求）
- token_ids_cpu： 完整的历史 token 序列（prompt + 已生成）
- num_tokens_no_spec： 每个请求的 token 数量（不含草稿）
输出： list[list[int]] - 每个请求的草稿 token ID 列表
它的核心逻辑如下：
1. 对每个请求，取最近 prompt_lookup_max 个 token 作为 n-gram 上下文；
2. 在较早的序列部分搜索与当前后缀匹配的子串；
3. 若匹配成功，从匹配位置之后截取 num_speculative_tokens 个 token 作为草稿；
4. 否则草稿数为 0（该请求暂不参加投机解码）。
核心代码（精简版）：
根据各自算法的不同，生成draft tokens
# 位置：vllm/v1/spec_decode/ngram_proposer.py
class NgramProposer:
    def propose(self, sampled_token_ids, req_ids, num_tokens_no_spec,
                token_ids_cpu, spec_decode_unsupported_reqs):
        # 过滤有效请求
        valid_requests = [
            i for i, sampled_ids in enumerate(sampled_token_ids)
            if len(sampled_ids) > 0 and
               req_ids[i] not in spec_decode_unsupported_reqs and
               num_tokens_no_spec[i] < self.max_model_len
        ]
        
        # 批量n-gram匹配（Numba加速）
        return self.batch_propose(
            len(sampled_token_ids), valid_requests,
            num_tokens_no_spec, token_ids_cpu
        )
草稿生成完成后，生成的 draft_token_ids 会被传递给调度器，调度器将其存储在 scheduler_output.scheduled_spec_decode_tokens 中。在下一轮推理的准备阶段（8.3 节），这些草稿 token 会被：
1. 拼接到原始序列末尾（批次构建）
2. 用于计算 SpecDecodeMetadata（元数据计算）
3. 最终在模型执行阶段（8.4 节）被目标模型验证
8.2 是否启用投机解码的判定
上一轮有没有draft tokens生成
是否启用投机解码的判定在 GPUModelRunner._prepare_inputs 中，并准备模型前向传播所需的关键元数据。
# 位置：vllm/v1/worker/gpu_model_runner.py 的 _prepare_inputs 方法

# 1. 判断是否启用投机解码
# scheduler_output.scheduled_spec_decode_tokens 是一个字典 {request_id: [draft_token_ids]}
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0

if use_spec_decode:
    # 2. 初始化：num_draft_tokens 存储每个请求的草稿 token 数量
    num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
    
    # 3. 遍历调度器提供的草稿 tokens，填充每个请求的草稿数量
    for (req_id, draft_token_ids) in scheduler_output.scheduled_spec_decode_tokens.items():
        req_idx = self.input_batch.req_id_to_index[req_id]
        num_draft_tokens[req_idx] = len(draft_token_ids)
    
    # 4. 计算投机解码专用的元数据（SpecDecodeMetadata）
    spec_decode_metadata = self._calc_spec_decode_metadata(
        num_draft_tokens, cu_num_tokens
    )
    
    # 5. 获取 logits 索引
    logits_indices = spec_decode_metadata.logits_indices
else:
    # 标准解码：只需为每个序列的最后一个 token 计算 logits
    logits_indices = query_start_loc[1:] - 1
    spec_decode_metadata = None
这段代码实现了推测解码（Speculative Decoding）的逻辑，这是一种通过之前说过的“草稿-验证”机制来加速文本生成的技术。
1. 首先，代码检查调度器是否在上一轮为请求生成了草稿 token 序列（scheduled_spec_decode_tokens）。如果没有草稿 token，则采用标准解码方式，直接对每个序列的最后一个 token 计算 logits。
2. 如果存在草稿 token，说明至少有一个请求启用了推测解码，此时需要为每个请求统计草稿 token 的数量
另外，我们知道scheduled_spec_decode_tokens 是 scheduler 在上一轮执行后分派好的草稿 token。
- 若为 0，直接走普通 sampling。
- 若 > 0，说明至少有一个请求在本轮启用了 speculative decoding，需要计算元数据。
假设有 5 个请求，其在当前 batch 中的累计 token 数（包括 prompt+已 decode+草稿位）为：
cu_num_scheduled_tokens = [4, 104, 107, 207, 209]
这表示：
- req0：累计到 index 4（长度 4）
- req1：累计到 index 104（长度 100）
- req2：累计到 index 107（长度 3）
- req3：累计到 index 207（长度 100）
- req4：累计到 index 209（长度 2）
草稿 token 数：
num_draft_tokens = [3, 0, 2, 0, 1]
接下来 _calc_spec_decode_metadata 要干的事情就是：把这两个数组，变成一套扁平索引，驱动 GPU 只对真正需要的位置做 logits 计算和概率对比。
8.3 计算投机解码元数据（SpecDecodeMetadata）
在 7.2 当中我们使用了_calc_spec_decode_metadata，_calc_spec_decode_metadata 这个函数是投机解码的核心函数之一，在草稿模型（draft model）生成草稿 token 后调用。它的作用是将上层调度器提供的离散信息（"每个请求有多少个草稿 token"）转换成底层 GPU Kernel 执行采样和验证时所需的、精确的、扁平化的索引数组。
# vllm/vllm/v1/worker/gpu_model_runner.py: GPUModelRunner 类的实例方法
def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # ==================== 步骤1: 计算每个请求的采样 token 数量 ===================
        num_sampled_tokens = num_draft_tokens + 1

        # ================== 步骤2: 计算累积采样token数和arange ======================
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)

        # ===== 步骤3: 计算每个请求的 logits 起始位置（在 scheduled tokens 中的位置）======
        # Step 3.1: 计算每个请求的起始位置
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3.2: 加上相对索引得到最终位置
        logits_indices += arange

        # ========== 步骤4: 计算bonus logits索引 =============================
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # ============= 步骤5: 计算目标模型的logits索引（只针对draft tokens）============
        # Step 5.1: 计算累积draft token数
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)

        # Step 5.2: 计算目标logits的起始位置
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)

        # Step 5.3: 加上相对索引得到最终位置
        target_logits_indices += arange

        # ============== 步骤6: 将numpy数组转换为GPU张量 ============
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # ========== 步骤7: 从input_ids中提取对应的draft token IDs ==========
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        # ========== 步骤8: 创建元数据对象 ==========
        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata
为什么需要这个转换？
在 GPU 上执行时，我们需要处理一个扁平的批次，但不同请求可能有不同数量的草稿 token。例如：
- 请求 1：3 个草稿 token
- 请求 2：0 个草稿 token（未启用投机解码）
- 请求 3：2 个草稿 token
- 请求 4：0 个草稿 token
- 请求 5：1 个草稿 token
我们需要计算出：
- 哪些位置需要计算 logits（用于验证草稿）
- 哪些位置的 logits 用于验证草稿 token
- 哪些位置的 logits 用于采样奖励 token
示例
- 输入：cu_num_scheduled_tokens = [4， 104， 107， 207， 209]（5 个请求的累积 token 数）
- 输入：num_draft_tokens = [3， 0， 2， 0， 1]（5 个请求的草稿 token 数），每个请求的原始草稿 token 数量。
- 输出：logits_indices = [0， 1， 2， 3， 103， 104， 105， 106， 206， 207， 208]（需要计算 logits 的位置），它包含了批次中所有需要目标模型计算 logits 的 token 位置。这是 GPU 执行前向计算的直接依据。
- 输出：target_logits_indices = [0， 1， 2， 5， 6， 9]（用于验证草稿 token 的 logits 索引），它是一个索引数组，用于从计算出的所有 logits 张量中获取对应于草稿 token 的那些 logits。这是获取 p_target(draft_token) 以进行接受/拒绝判断的关键。
- 输出：bonus_logits_indices = [3， 4， 7， 8， 10]（用于采样奖励 token 的 logits 索引），它用于获取每个请求最后一个提议 token 的 logits。如果某个请求的所有草稿都被接受，就用这个 logit 来免费采样一个额外的"奖励"token。
按示例一步步算（数组长度=5 个请求）：
1. 先算每请求的采样段长度 = 草稿数 + 1（奖励位） ，也就是总共采样的长度，包括草稿+bonus。 num_sampled_tokens = [3, 0, 2, 0, 1] + 1 = [4, 1, 3, 1, 2]; 
2. 累积草稿数（前缀和），cu_num_draft_tokens = [3， 3， 5， 5， 6]，也就是 num_sampled_tokens 求前缀和；
3. 累积采样段长度（前缀和），cu_num_sampled_tokens = [4， 5， 8， 9， 11]；
4. logits_indices，它就是把每段长度展开，加上局部 0..len-1
  - 请求 1：起点 0，长度 4 → 0，1，2，3
  - 请求 2：起点 103，长度 1 → 103 。请求 2 只有一个 bonus
  - 请求 3：起点 104，长度 3 → 104,105,106 请求 3，2 个草稿+一个 bonus
  - 请求 4：起点 206，长度 1 → 206
  - 请求 5：起点 207，长度 2 → 207,208
合并即 logits_indices = [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
   现在到了确定 draft 和 bonus 的相对偏移的时候
  一共采样的个数 -1 = 最后一位的索引 减去 1 = [3， 4， 7， 8， 10]
  1. bonus_logits_indices（每段最后一个位置）
bonus_logits_indices = cu_num_sampled_tokens - 1
= [4-1, 5-1, 8-1, 9-1, 11-1] = [3,4,7,8,10]
  2. target_logits_indices（每段去掉最后一个奖励位，只对草稿位展开）
先算草稿段起点 = cu_num_sampled_tokens - num_sampled_tokens
= [4-4, 5-1, 8-3, 9-1, 11-2] = [0,4,5,8,9] 采样的相对起点 然后按草稿数重复并加局部索引：
    - 请求 1：起点 0，草稿 3 → 0，1，2
    - 请求 2：草稿 0 → 无
    - 请求 3：起点 5，草稿 2 → 5，6
    - 请求 4：草稿 0 → 无
    - 请求 5：起点 9，草稿 1 → 9
合并 target_logits_indices = [0,1,2, 5,6, 9]
因为要把“不同请求、不同草稿数”的位置统一成一个扁平批次，便于 GPU 一次前向，避免为每个请求单独跑或做大量 padding。展平 indices 带来的好处：
- 单次前向：logits_indices 把原始尾部 + 草稿验证位 + 奖励位揉成一串索引，模型只算这些必要位置的 logits，减少无关计算。
- 规整内存访问：各请求的段在扁平数组中连续，cu_num_draft_tokens 等前缀和能快速定位每个请求的草稿起止，GPU kernel 按段遍历效率高。
- 统一 gather：target_logits_indices / bonus_logits_indices 直接在扁平 logits 上切片，无需按请求再拼/拆。
- 避免 padding 膨胀：不同请求草稿数不一样，展平后只保留真实需要的位，不用为齐长做大 padding。
可能还是有点混乱，所以我们用上面同一个例子走一遍（5 个请求）：
- 已知：
  - cu_num_scheduled_tokens = [4, 104, 107, 207, 209]（定位每个请求在扁平批次的区间终点）
  - num_draft_tokens = [3, 0, 2, 0, 1]（5个请求，每个请求它各自生成的草稿数）
- 计算：
  - num_sampled_tokens = num_draft_tokens + 1 = [4, 1, 3, 1, 2]（草稿 + bonus）
  - cu_num_sampled_tokens = 前缀和 = [4, 5, 8, 9, 11]
  - logits_indices（本轮要算 logits 的所有位置）：
    - req0: 从 0 开始，取 4 个 → 0,1,2,3
    - req1: 从 103 开始，取 1 个 → 103
    - req2: 从 104 开始, 取 3 个 → 104,105,106
    - req3: 从 206 开始, 取 1 个 → 206
    - req4: 从 207 开始, 取 2 个 → 207,208
    - 合并：[0,1,2,3, 103, 104,105,106, 206, 207,208] 这里记录了5个请求各自所属的logits 索引。
暂时无法在飞书文档外展示此内容


  - bonus_logits_indices = cu_num_sampled_tokens - 1 = [3, 4, 7, 8, 10]（各请求 bonus 位的索引）
  - target_logits_indices（草稿验证位）：
    - 先算各请求采样段起点：cu_num_sampled_tokens - num_sampled_tokens = [0,4,5,8,9]
    - 按草稿数展开：
      - req0 起点 0，草稿 3 → 0,1,2
      - req1 草稿 0 → 无
      - req2 起点 5，草稿 2 → 5,6
      - req3 草稿 0 → 无
      - req4 起点 9，草稿 1 → 9
      - 合并：[0,1,2, 5,6, 9]
8.4  执行阶段：模型前向传播
在标准解码过程中，模型仅需计算序列最后一个 token 对应的 logits。而在投机解码中，则需为目标模型在多个位置（包括原始输入及所有草稿 token）计算 logits，具体位置由 logits_indices 指定。
# 位置：vllm/v1/worker/gpu_model_runner.py 的 execute_model 方法中

# 核心逻辑：从隐藏状态中提取需要计算logits的位置
sample_hidden_states = hidden_states[logits_indices]  # 提取指定位置的隐藏状态
logits = self.model.compute_logits(sample_hidden_states)  # 计算logits

# Pipeline Parallelism模式：在最后rank计算并广播logits
if self.broadcast_pp_output:
    if not get_pp_group().is_last_rank:
        get_pp_group().send_tensor_dict(hidden_states.tensors, ...)
        logits = None
    else:
        logits = self.model.compute_logits(hidden_states[logits_indices])
    
    # 广播logits到所有PP rank
    logits = get_pp_group().broadcast_tensor_dict({"logits": logits}, ...)["logits"]
模型前向传播完成后，我们得到了所有需要验证位置的 logits。接下来的关键步骤是：使用拒绝采样算法验证草稿 token，决定哪些草稿可以被接受。
8.5 验证阶段：拒绝采样算法
数学原理：为何 ratio = p_target / p_draft
以一维情形为例：
- 草稿模型给出分布 $$q(\cdot)$$，从中采样到草稿 token $$y\_1, \dots, y\_k$$；
- 目标模型真实分布为 $$p(\cdot)$$；
我们希望在只调用一次目标模型的情况下，让最终输出的序列分布尽量接近直接从 p 上自回归采样。
典型的 speculative sampling 构造（简化版）：
1. 对草稿 token 序列按顺序处理，对于 token $$y_i$$，计算：$$r_i = \frac{p(y_i)}{q(y_i)}$$
1. 采样一个均匀随机数$$u_i \sim U(0,1)$$：
  - 若$$r_i \ge u_i$$：接受$$y_i$$；
  - 否则：拒绝，并从$$p(\cdot)$$重新采样该步及之后所有 token。
直观解释：
- 若草稿在该 token 上低估了目标模型概率$$（p/q \gt 1）$$，比例更容易$$\ge u$$，更容易被接受；
- 若草稿在该 token 上高估了目标模型概率$$（p/q \lt 1）$$，被拒绝概率更大，回退到目标模型原生采样。
这个机制保证草稿序列被接受的概率与直接用目标模型逐步采样在统计意义上保持一致，具体的证明此处不展开。
if not rejected:
    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
    if NO_DRAFT_PROBS:
        draft_prob = 1.0
    else:
        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)

    target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)

    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
        # 接受草稿
        token_id = draft_token_id
    else:
        # 拒绝，回退到 recovered token
        rejected = True
        token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)

    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
1. 准备阶段：加载数据
if not rejected:
    # 只有当前面的 token 都没有被拒绝时，才继续验证当前 token。
    # 一旦有一个 token 被拒绝，后续所有的 draft token 都会自动无效（因果链断裂）。
    
    # 1. 加载草稿模型生成的 token ID
    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
    
    # 2. 获取该 token 在草稿模型中的概率 (q(x))，从显存加载 draft_prob
    draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)

    # 3. 获取该 token 在目标模型（大模型）中的概率 (p(x))
    target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
    
    # 4. 加载一个随机数 (u)，用于随机性采样
    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
2. 核心判断：拒绝采样公式
这里的 target_prob / draft_prob >= uniform_prob 其实是 Rejection Sampling 的变体。
- 如果 target_prob >= draft_prob，比值大于等于1，而 uniform_prob 最大是1，所以条件恒成立 -> 必定接受。这符合直觉：大模型觉得这个词比小模型觉得的更好。
- 如果 target_prob < draft_prob，我们以 target_prob / draft_prob 的概率接受它。这保证了最终输出的分布严格服从目标模型的分布。
# 标准的拒绝采样条件：p(x) / q(x) >= u
# 其中 u 是 [0, 1] 之间的均匀分布随机数
if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
    # 【接受情况】
    # 如果目标概率(target)比草稿概率(draft)大，或者虽然小一点但在容忍范围内（由 uniform_prob 决定），
    # 则接受这个草稿 token。
    token_id = draft_token_id
else:
    # 【拒绝情况】
    # 标记 rejected 为 True。注意：这个变量通常是在寄存器或共享内存中维护的状态。
    # 一旦变为 True，后续的 pos 循环（如果有）将直接跳过验证逻辑。
    rejected = True

    # 回退：使用从目标模型分布中重新采样得到的 token (recovered token)。
    # 这通常是修正后的正确 token。
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
对每个请求的处理模式：
1. 依次检查每个草稿：
  - 若全部接受：拼接 [所有草稿 token, 奖励 token]。
  - 若在第 i 个被拒绝：保留前 i-1 个草稿，将第 i 个位置改为恢复 token（从目标模型原生分布采样），第 i+1 及之后草稿全部忽略。
2. 得到的输出序列再进入下一轮 decode
每请求的输出序列长度为（接受草稿前缀长度 + 1），1 要么是奖励 token，要么是恢复 token。
8.6 异步调度与跨轮次优化
异步传输是一种避免在每一轮生成后将 Token ID 从 GPU 传回 CPU，再从 CPU 传回 GPU 的方法。也就是说，尽量减少 sampled token 在 GPU/CPU 之间的来回拷贝，把 token 写回下一轮输入的动作留在 GPU 上完成。
1. 第一阶段：_bookkeeping_sync 将上一轮生成的 Token ID 暂存在 GPU 中（当开启 async scheduling 时）
  1. 同步调度：对应第一条分支 if not self.use_async_scheduling
    - GPU 计算生成新的 Token。
    - CPU 等待 GPU 完成（同步点）。
    - 将新生成的 Token ID 从 GPU 显存拷贝回 CPU 内存（使得调度器/CPU 侧能直接拿到本轮 token 结果）。
    - 调度器在 CPU 上决定下一轮要跑哪些请求，并把这些 Token ID 拼接到下一轮的输入 input_ids 中，随后再将新的 input_ids从 CPU 拷贝回 GPU。
  2.  异步调度：对应第二条分支（else）
    - 本轮 decode 将 GPU 上的 sampled_token_ids 暂存在 prev_sampled_token_ids；
    - 不立即拷回 CPU，而是在下一轮 _prepare_input_ids 中，直接在 GPU 上把这些 token 写入到下一轮 input_ids 对应位置（通过 index/scatter 等方式），从而绕过一次 GPU→CPU→GPU 往返。
    - 也就是说，当开启 use_async_scheduling=True 时，我们希望 CPU 尽可能少地干预 token 数据传输（但 CPU 仍然会做调度决策与 bookkeeping，只是尽量不需要“拿到 token 本身”）。

暂时无法在飞书文档外展示此内容
class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    def _bookkeeping_sync(...):
        if not self.use_async_scheduling:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(...)
        else:
            # 不同步回 CPU，仅缓存到 input_batch（GPU tensor）
            self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {...}
2. 第二阶段：_prepare_input_ids 中的数据填入
到了下一轮推理（Step N+1），我们需要准备输入数据 input_ids。如果开启了异步优化，我们直接利用上一轮暂存在 GPU 上的 prev_sampled_token_ids，使用 scatter_ 将这些 Token ID 写入到 input_ids.gpu 的正确位置。
class GPUModelRunner:
    def _prepare_input_ids():
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )
而如果没有开启异步优化（use_async_scheduling=False），那么本轮生成的 token 已经被同步回 CPU，CPU 侧会更新/拼接下一轮所需的输入与状态；随后在准备下一轮计算时，再把下一轮需要的 input_ids拷贝到 GPU。
class GPUModelRunner:
    def _prepare_input_ids():
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        ...
        ...
参考资料
https://arxiv.org/abs/2503.01840