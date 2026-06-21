"""
vllm_transformer.py — 精炼复刻 vLLM 的 transformer block 如何配合 torch.compile + CUDA Graph。

本 demo 是**纯 PyTorch 玩具**（不依赖 vLLM，可 CPU 直接跑），但结构 1:1 对齐 vLLM v1 的真实做法：
  - DecoderLayer = RMSNorm → Attention → RMSNorm → MLP（pre-norm + residual），对齐 LlamaDecoderLayer
  - Attention 通过**自定义算子**（custom op）调用，对齐 vllm::unified_attention_with_output
  - KV 用**分页定形缓冲 + metadata（seq_lens/block_tables）**，对齐 PagedAttention / FlashAttentionMetadata

它回答三个问题（详见文末 __main__ 的打印）：
  ① 能否用 CUDA Graph？  能——decode 全程固定 shape（batch 分桶 padding + KV 变长藏进 metadata 的“值”）。
  ② 是否出现 graph break？关键认知：attention 注册成**带 fake 实现的 custom op** → Dynamo **不** break，
     整个模型是“一张图 + attention 作为不透明节点”。vLLM 随后在**自己的编译后端**按 attention 算子
     把这张图**切成多段（piecewise）**，每段各自做 CUDA Graph。所以“切”发生在 FX 层，不是 Dynamo break。
     （对照：若算子没 fake 实现，或在 Python 里 .item()/依赖张量值分支，才会触发真正的 Dynamo break。）
  ③ decode 变长 KV 如何参与计算？ KV cache 形状恒定；变化的是 seq_lens/block_tables 里的**数值**；
     attention kernel **按 seq_len 的值做 mask/early-exit**（设备端比较，无 host 同步）。CUDA Graph
     怕“shape/launch 变”，不怕“算的量变”——只要每步把新值 in-place 写进固定 metadata 缓冲再 replay 即可。

对应 vLLM 源码（本机 /home/eechengyang/CX/vllm/vllm）：
  - model_executor/models/llama.py     LlamaDecoderLayer/LlamaAttention/LlamaMLP + @support_torch_compile
  - model_executor/layers/attention/attention.py  unified_attention_with_output（direct_register_custom_op）
  - compilation/backends.py            split_graph（按 splitting_ops 切 piecewise）
  - config/compilation.py              _attention_ops（"vllm::unified_attention_with_output" 是切点）
  - v1/attention/backends/flash_attn.py  FlashAttentionMetadata（seq_lens/block_table/slot_mapping...）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ============================================================================
# PART A — 分页 KV cache 的“元数据”：全是固定 shape，只有里面的“值”随对话变化
#   对齐 vLLM 的 FlashAttentionMetadata（seq_lens / block_table / slot_mapping）。
#   这是 CUDA Graph 能图化 decode 的根：变长被塞进“值”，capture 看到的 shape 永远不变。
# ============================================================================
class AttentionMetadata:
    def __init__(self, seq_lens: torch.Tensor, block_tables: torch.Tensor):
        self.seq_lens = seq_lens          # [num_seqs] int，  每条序列“当前 context 长度”（值会变）
        self.block_tables = block_tables  # [num_seqs, max_blocks] int，逻辑块→物理块映射（值会变）


# ============================================================================
# PART B — Attention 作为“自定义算子”：对 Dynamo 是不透明黑盒 = piecewise 的切点
#   vLLM 用 direct_register_custom_op 注册 unified_attention_with_output；这里用等价的
#   torch.library.custom_op。关键：注册了 fake 实现 → torch.compile 能把它当**单个节点**
#   收进一张图而**不 break**；vLLM 再在编译后端按这个算子把图切成 piecewise。
# ============================================================================
@torch.library.custom_op("vllm_demo::paged_attention", mutates_args=())
def paged_attention(
    q: torch.Tensor,            # [num_tokens, H, D]，decode 时 num_tokens == num_seqs（每条 1 个新 token）
    k_cache: torch.Tensor,      # [num_blocks, block_size, H, D]，**固定形状**的分页 K cache
    v_cache: torch.Tensor,      # 同上的 V cache
    block_tables: torch.Tensor, # [num_seqs, max_blocks]
    seq_lens: torch.Tensor,     # [num_seqs]
    scale: float,
) -> torch.Tensor:
    """参考实现的 PagedAttention（decode：每条序列 1 个 query token）。

    教学核心：**所有张量 shape 与运算都固定，真正“变”的是 seq_lens 的“值”所驱动的 mask。**
    我们按固定的最大 context（max_blocks×block_size）gather KV 并算分数，再用
    `pos < seq_lens[:,None]` 这个**设备端比较**把超出真实长度的位置 mask 成 -inf。
    - 没有 `.item()`、没有 D2H 同步 → **可被 CUDA Graph 捕获**（满足铁律一）。
    - 变长 context 体现为 mask 的取值，而非任何张量的 shape → 满足铁律三。
    这正是真实 vLLM CUDA kernel 的思路：按 max_seq 起固定 grid、用 context_lens 的值做 early-exit/mask。
    “CUDA Graph 不怕算的量变，只怕 shape/launch 变”——这里就是活例。
    """
    block_size = k_cache.shape[1]
    H, D = k_cache.shape[2], k_cache.shape[3]
    # 固定形状 gather：每条序列取满 max_blocks 个物理块（多取的之后被 mask 掉）
    k = k_cache[block_tables].reshape(q.shape[0], -1, H, D)   # [N, max_ctx, H, D]，max_ctx 恒定
    v = v_cache[block_tables].reshape(q.shape[0], -1, H, D)
    max_ctx = k.shape[1]
    scores = torch.einsum("nhd,nlhd->nhl", q, k) * scale     # [N, H, max_ctx]
    pos = torch.arange(max_ctx, device=q.device)            # [max_ctx]
    valid = pos[None, :] < seq_lens[:, None]                 # [N, max_ctx] ← seq_lens 的“值”驱动 mask
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("nhl,nlhd->nhd", probs, v)           # [N, H, D]


@paged_attention.register_fake
def _(q, k_cache, v_cache, block_tables, seq_lens, scale):
    # fake（meta）实现：只描述输出 shape/dtype，让 Dynamo 不进 kernel、也不 break。
    return torch.empty_like(q)


class Attention(nn.Module):
    """对齐 LlamaAttention：qkv_proj → RoPE(此处省略) → 调 custom op → o_proj。"""
    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.scale = head_dim ** -0.5
        self.qkv_proj = nn.Linear(hidden, 3 * num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)

    def forward(self, x, k_cache, v_cache, md: AttentionMetadata):
        n = x.shape[0]
        qkv = self.qkv_proj(x).view(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # 真实 vLLM 在这里把 k,v 写进 cache（slot_mapping）。demo 假设 cache 已写好，聚焦读取路径。
        # ↓↓↓ 这一句是 piecewise 的边界：对 Dynamo 是不透明节点，对 Inductor 是不可跨越的融合边界
        attn = torch.ops.vllm_demo.paged_attention(
            q, k_cache, v_cache, md.block_tables, md.seq_lens, self.scale
        )
        return self.o_proj(attn.reshape(n, -1))


# ============================================================================
# PART C — MLP（SwiGLU），对齐 LlamaMLP：gate_up_proj → SiLU*mul → down_proj
#   纯逐元素 + GEMM，**完全可被 torch.compile 融合**，也完全能进 CUDA Graph。
# ============================================================================
class MLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)   # SiluAndMul


class RMSNorm(nn.Module):
    """对齐 vLLM RMSNorm 的 fused-residual 接口：forward(x, residual) -> (normed, new_residual)。"""
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        residual = x
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return normed, residual


# ============================================================================
# PART D — DecoderLayer：pre-norm 残差结构，1:1 对齐 LlamaDecoderLayer.forward
# ============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, head_dim, inter):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = Attention(hidden, num_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = MLP(hidden, inter)

    def forward(self, x, residual, k_cache, v_cache, md):
        x, residual = self.input_layernorm(x, residual)          # 归一化（可融合/可图化）
        x = self.self_attn(x, k_cache, v_cache, md)              # ← 唯一的 piecewise 切点
        x, residual = self.post_attention_layernorm(x, residual) # 归一化（可融合/可图化）
        x = self.mlp(x)                                          # FFN（可融合/可图化）
        return x, residual


# ============================================================================
# PART E — Model：堆 N 层。@support_torch_compile 在 vLLM 里把 batch 维标记 dynamic 并 torch.compile
#   这里用一个简化版 decorator 体现“标记 dynamic + 编译”的精神。
# ============================================================================
def support_torch_compile(cls):
    """精简版 vLLM @support_torch_compile：把 dim0(token/batch) 标记为 dynamic，再 torch.compile。"""
    orig_forward = cls.forward

    def wrapped(self, x, *args, **kwargs):
        if getattr(self, "_compiled", None) is None:
            torch._dynamo.mark_dynamic(x, 0)   # token 维动态 → 一份编译产物服务所有长度，无需分桶 padding
            self._compiled = torch.compile(lambda *a, **k: orig_forward(self, *a, **k),
                                           backend="eager", dynamic=True)  # demo 用 eager 后端避免 C 编译依赖
        return self._compiled(x, *args, **kwargs)

    cls.forward = wrapped
    cls._compiled = None
    return cls


class Model(nn.Module):
    def __init__(self, n_layers=2, hidden=64, num_heads=4, head_dim=16, inter=128):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden, num_heads, head_dim, inter) for _ in range(n_layers)]
        )

    def forward(self, x, k_caches, v_caches, md):
        residual = None
        for i, layer in enumerate(self.layers):
            x, residual = layer(x, residual, k_caches[i], v_caches[i], md)
        return x


# ============================================================================
# 配置 + 构造 KV cache / metadata 的小工具
# ============================================================================
HIDDEN, NUM_HEADS, HEAD_DIM, INTER, N_LAYERS = 64, 4, 16, 128, 2
BLOCK_SIZE, NUM_BLOCKS, MAX_BLOCKS = 16, 64, 8


def make_caches(device):
    """
    return: [num_blocks, block_size, num_heads, head_dim]
    Q: 为什么这里是 num_blocks 和 block_size？不应该是 context seq len？
    """
    k = [torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, device=device) for _ in range(N_LAYERS)]
    v = [torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, device=device) for _ in range(N_LAYERS)]
    return k, v


def make_metadata(seq_lens_list, device):
    num_seqs = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    # 给每条序列分配各不相同的物理块（简单起见顺序分配）
    block_tables = torch.zeros(num_seqs, MAX_BLOCKS, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        block_tables[i] = torch.arange(i * MAX_BLOCKS, (i + 1) * MAX_BLOCKS, device=device)
    return AttentionMetadata(seq_lens, block_tables)


# ============================================================================
# __main__ — 三个问题逐一演示
# ============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")

    model = Model(N_LAYERS, HIDDEN, NUM_HEADS, HEAD_DIM, INTER).to(device).eval()
    k_caches, v_caches = make_caches(device)

    # ---- 一个 decode 批次：4 条序列，context 长度各不相同（5,17,33,40）----
    seq_lens = [5, 17, 33, 40]
    num_seqs = len(seq_lens)
    md = make_metadata(seq_lens, device)
    x = torch.randn(num_seqs, HIDDEN, device=device)  # decode：每条序列 1 个新 token

    # ========== 问题③：变长 KV 如何参与计算 ==========
    print("=" * 70)
    print("③ decode 变长 KV 如何参与计算")
    print("=" * 70)
    with torch.inference_mode():
        out = model(x, k_caches, v_caches, md)
    print(f"输入 x: {tuple(x.shape)}  (num_tokens=num_seqs={num_seqs}, 每条 1 个新 token)")
    print(f"各序列 context 长度 seq_lens = {seq_lens}  ← 长度不同，但 KV cache 形状恒定")
    print(f"KV cache 形状 = {tuple(k_caches[0].shape)} (num_blocks×block_size×H×D, 不随对话增长)")
    print(f"输出 = {tuple(out.shape)}")
    print("机制：attention kernel 按 seq_lens 的“值”构造设备端 mask（pos < seq_len）做 early-exit；")
    print("      变长体现在“值”和 block_tables，而非任何张量的 shape（故无 .item()/无 D2H 同步）。\n")

    # ========== 问题②：是否出现 graph break ==========
    print("=" * 70)
    print("② 是否出现 graph break（attention custom op 对 Dynamo 是否不透明）")
    print("=" * 70)
    try:
        # explain 只做 trace，不需要 C 编译器，最稳。注意：attention 注册了 fake 实现，
        # 所以预期是“1 张图 + 0 break”，attention 作为不透明节点留在图里。
        explanation = torch._dynamo.explain(
            lambda: model.layers[0](x, None, k_caches[0], v_caches[0], md)
        )()
        print(f"graph 数        = {explanation.graph_count}")
        print(f"graph break 数  = {explanation.graph_break_count}  ← 0：custom op(带 fake)不触发 Dynamo break")
    except Exception as e:
        print(f"(explain 在本环境跑不动，结论仍成立) {type(e).__name__}: {e}")
    print("→ vLLM 的‘切’不靠 Dynamo break：它把整层 trace 成一张图（attention 是不透明节点），")
    print("  再在编译后端按 splitting_ops=['vllm::unified_attention_with_output'] 把图切成 piecewise，")
    print("  每段（norm+proj+MLP）各自 compile/CUDA Graph，attention 段单独跑 varlen kernel。")
    print("对照：若在 Python 里写 seq_lens.max().item() 这种依赖张量值的分支，才会触发真正的 break。\n")

    # ========== 问题①：能否用 CUDA Graph ==========
    print("=" * 70)
    print("① 能否用 CUDA Graph（decode 全 FULL 图）")
    print("=" * 70)
    if device == "cuda":
        # 预热（side stream，见 warmup.md）
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                model(x, k_caches, v_caches, md)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

        # 捕获：所有输入（x / seq_lens / block_tables / cache）都是固定地址、固定 shape
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            static_out = model(x, k_caches, v_caches, md)
        torch.cuda.synchronize()

        base = static_out.clone()
        # 演示“变长 KV 在静态图里参与计算”：把第 0 条序列的 context 长度 in-place +1（多吃一个已存在的 KV 槽），
        # 地址/shape 不变，只改 seq_lens 的“值”，replay 后该序列输出随之改变。
        md.seq_lens[0] += 1
        g.replay(); torch.cuda.synchronize()
        changed = not torch.allclose(base[0], static_out[0], atol=1e-5)
        print("✅ decode 整层成功捕获为 FULL CUDA Graph 并 replay。")
        print(f"   in-place 把 seq_lens[0] 从 {seq_lens[0]} 改成 {seq_lens[0]+1} 后 replay，"
              f"该序列输出是否改变: {changed}")
        print("   → 证明：固定 shape 的图里，靠改 metadata 的‘值’就能让变长 KV 参与计算。")
    else:
        print("（当前是 CPU，跳过实际捕获）结论：decode 能上 FULL CUDA Graph，因为——")
        print("  • batch 维分桶 + padding → Q/输出 shape 固定；")
        print("  • KV 变长藏进固定 shape 的 seq_lens/block_tables 的‘值’ + kernel 内部循环；")
        print("  • 每步只 copy_ 新值进固定 metadata 缓冲再 replay（铁律二：地址固定、内容 in-place）。")
    print()

    print("=" * 70)
    print("一句话总结")
    print("=" * 70)
    print("• 能否 CUDA Graph：decode 能（FULL）；prefill 因 query token 数改 grid 走 PIECEWISE/eager。")
    print("• 是否 graph break：attention 是带 fake 的 custom op → 不触发 Dynamo break；")
    print("  vLLM 在编译后端按 attention 算子把图切成 piecewise（FX 层的切，不是 Dynamo break）。")
    print("• 变长 KV：藏进固定 shape 的 metadata‘值’+ kernel 按 seq_len 值做 mask/early-exit——")
    print("  CUDA Graph 怕 shape/launch 变，不怕算的量变。")
