"""
不同的 LLM 层是怎么量化的:逐层拆解
=================================================================

这个文件回答你最关心的问题:"不同的层怎么量化、哪些层不量化"。

先建立全局图景。一个 Transformer block 里的层,按"该不该量化 / 怎么量化"分三类:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ A. 大矩阵乘 Linear —— 量化的主战场                                     │
  │    attention 的 q_proj/k_proj/v_proj/o_proj                          │
  │    FFN 的 gate_proj/up_proj/down_proj(或 MoE 的 expert)             │
  │    特点:参数量占全模型 ~90%,decode 时显存搬运的大头 → 必须量化       │
  ├─────────────────────────────────────────────────────────────────────┤
  │ B. 敏感 / 小开销层 —— 通常保留 fp16,不量化                            │
  │    embedding、最后的 lm_head、RMSNorm/LayerNorm、router(MoE 门控)   │
  │    原因:要么对精度极敏感(norm/router 一错全错),                    │
  │          要么参数量小(省不了多少),量化收益与风险不成正比           │
  ├─────────────────────────────────────────────────────────────────────┤
  │ C. KV Cache —— 单独的一类"激活量化"                                   │
  │    长上下文时 KV Cache 比权重还吃显存,可单独量化成 int8/fp8           │
  │    (vLLM 用 kv_cache_dtype 控制,见 05 文件)                         │
  └─────────────────────────────────────────────────────────────────────┘

这就是你说的"对不同层用不同精度"的工程化版本:不是凭感觉,而是按
"参数占比 + 精度敏感度 + 它在 decode 里是不是访存瓶颈"来决策。

本文件用一个 Linear 层,把三种主流方案在同一层上跑一遍,看清区别:
    W8A16 / W4A16(weight-only) vs  W8A8(weight+activation)

    python quant/02_layer_by_layer.py
"""

import torch
import torch.nn.functional as F


# 复用 01 文件的量化原语(这里为自包含重写精简版)
def per_channel_sym_quant(w, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(w / scale).clamp(-qmax - 1, qmax)
    return q, scale


def per_group_sym_quant(w, num_bits=4, group_size=128):
    out_f, in_f = w.shape
    qmax = 2 ** (num_bits - 1) - 1
    wg = w.reshape(out_f, in_f // group_size, group_size)
    scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(wg / scale).clamp(-qmax - 1, qmax)
    return q.reshape(out_f, in_f), scale


def per_group_dequant(q, scale, group_size=128):
    out_f, in_f = q.shape
    qg = q.reshape(out_f, in_f // group_size, group_size)
    return (qg * scale).reshape(out_f, in_f)


def per_token_sym_quant(x, num_bits=8):
    """激活按 per-token 动态量化:每一行(每个 token)一个 scale,推理时实时算。
    这是 W8A8 里激活那一半的标准做法——为什么是动态的?见下方说明。"""
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q, scale


# ---------------------------------------------------------------------------
# 三种方案,同一个 Linear: y = x @ W^T
# ---------------------------------------------------------------------------
class W8A16Linear:
    """Weight-only int8。权重存 int8,激活保持 fp16。
    计算时把权重反量化回 fp16 再做 matmul(或用专门 kernel 边解边算)。
    → 省的是显存和访存,算还是 fp16 的算。decode(memory-bound)最爱这个。"""

    def __init__(self, w):
        self.q, self.scale = per_channel_sym_quant(w, 8)

    def __call__(self, x):
        w_deq = self.q * self.scale          # [out, in],反量化回 fp16
        return F.linear(x, w_deq)


class W4A16Linear:
    """Weight-only int4 + group量化。权重压到 ~1/4,激活仍 fp16。
    int4 太糙,必须靠 group(128)和 GPTQ/AWQ 这类算法压误差(见 03 文件)。"""

    def __init__(self, w, group_size=128):
        self.group_size = group_size
        self.q, self.scale = per_group_sym_quant(w, 4, group_size)

    def __call__(self, x):
        w_deq = per_group_dequant(self.q, self.scale, self.group_size)
        return F.linear(x, w_deq)


class W8A8Linear:
    """Weight + activation 都 int8。这是唯一能真正用上 INT8 Tensor Core 的方案。
    权重 per-channel 静态量化;激活 per-token 动态量化(运行时实时算 scale)。
    → prefill(compute-bound)能靠 INT8 算力翻倍获益;但激活难量化(见 04 文件)。"""

    def __init__(self, w):
        self.wq, self.wscale = per_channel_sym_quant(w, 8)  # [out,in], [out,1]

    def __call__(self, x):
        xq, xscale = per_token_sym_quant(x, 8)               # [tokens,in], [tokens,1]
        # 真实硬件上这一步是 INT8 矩阵乘,累加到 int32;这里用 float 模拟数值
        acc = F.linear(xq.float(), self.wq.float())          # [tokens, out]
        # 反量化:乘回两边的 scale。x 的 scale 在行上,w 的 scale 在列上
        y = acc * xscale * self.wscale.squeeze(-1).unsqueeze(0)
        return y


def rel_err(ref, out):
    return ((ref - out).abs().norm() / ref.norm() * 100).item()


def main():
    torch.manual_seed(0)
    print("=" * 72)
    print("在同一个 Linear 上对比 W8A16 / W4A16 / W8A8")
    print("=" * 72)

    in_f, out_f, tokens = 4096, 4096, 32
    # 模拟一个真实 FFN 的 down_proj 权重 + 输入激活
    W = torch.randn(out_f, in_f) * (1.0 / in_f ** 0.5)
    x = torch.randn(tokens, in_f)
    # 给激活制造几个"离群通道"——这是激活难量化的根源(04 文件展开)
    x[:, 17] *= 20.0
    x[:, 1234] *= 15.0

    y_ref = F.linear(x, W)  # fp16/fp32 参考答案

    for name, layer in [
        ("W8A16 (weight-only int8)", W8A16Linear(W)),
        ("W4A16 (weight-only int4,g=128)", W4A16Linear(W)),
        ("W8A8  (weight+act int8)", W8A8Linear(W)),
    ]:
        y = layer(x)
        print(f"  {name:<34} 输出相对误差 = {rel_err(y_ref, y):.3f}%")

    print()
    print("观察与结论:")
    print("  - W8A16 误差很小:只动权重,激活无损,decode 阶段性价比最高。")
    print("  - W4A16 误差变大:int4 格点少,是'省显存换一点精度',靠 GPTQ/AWQ 补回。")
    print("  - W8A8 误差最大:因为激活里的离群通道把 per-token scale 撑大,")
    print("    多数正常值被压进很少的格点 → 这正是 SmoothQuant 要解决的(04 文件)。")

    print()
    print("=" * 72)
    print("为什么激活要'动态'量化,权重可以'静态'量化?")
    print("=" * 72)
    print("  权重:推理时是常量,可以离线一次性算好 scale 存下来(静态)。")
    print("  激活:每次前向输入都不同,scale 必须运行时现算(动态);")
    print("        现算本身有开销,这也是 W8A8 在 decode 阶段未必划算的原因之一。")

    print()
    print("=" * 72)
    print("哪些层'不'量化:用数字说话")
    print("=" * 72)
    # 用一个小配置估算各类层的参数占比,解释为什么只压 Linear
    d, d_ff, vocab, n_layer = 4096, 14336, 152064, 32
    attn = 4 * d * d                      # q,k,v,o
    ffn = 3 * d * d_ff                    # gate,up,down
    per_layer = attn + ffn
    linear_total = per_layer * n_layer
    embed = vocab * d                     # embedding 与 lm_head 通常共享或各一份
    norm = 2 * d * n_layer + d            # 每层两个 RMSNorm + 最后一个
    total = linear_total + embed + norm
    print(f"  大 Linear(attn+ffn) 占比 : {linear_total/total*100:5.1f}%  <- 量化主战场")
    print(f"  embedding / lm_head 占比 : {embed/total*100:5.1f}%  <- 大但敏感,常保 fp16")
    print(f"  RMSNorm 占比             : {norm/total*100:5.2f}%  <- 小到忽略,且极敏感")
    print("  → 把 90%+ 的参数(Linear)量化,就拿到了几乎全部的显存收益;")
    print("    剩下的小而敏感的层保留 fp16,避免精度雪崩。这就是'层间混合精度'。")


if __name__ == "__main__":
    main()
