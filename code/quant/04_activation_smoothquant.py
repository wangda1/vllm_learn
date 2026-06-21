"""
为什么"激活比权重难量化",以及 SmoothQuant 怎么解决
=================================================================

这是面试第②问的核心,也是整个量化里最反直觉的一点。

------------------------------------------------------------------------
为什么激活难,权重不难?
------------------------------------------------------------------------
1) 权重分布"乖":训练好的权重大致是钟形分布、各通道尺度接近,
   per-channel 量化就能贴合得很好(01/02 文件已验证)。

2) 激活分布"野":LLM 的激活里存在系统性的"离群通道"(outlier channels)——
   极少数特征维度上的值比其他维度大几十上百倍,而且持续出现在固定通道上。
   per-tensor / per-token 量化时,scale 被这几个离群通道撑大,
   导致 99% 的正常值被挤进很少的几个整数格点 → 信息大量丢失。

3) 更糟的是:离群值"不能简单裁掉"。它们恰恰携带重要信息(常和高频词/关键
   语义相关),裁掉会直接掉点。所以不能用"clip 一刀切"了事。

------------------------------------------------------------------------
业界思路:把"难"从激活搬到权重(SmoothQuant)
------------------------------------------------------------------------
既然激活难、权重易,那就做一次"难度迁移"。对 y = x @ Wᵀ:
   按输入通道 j 选一个迁移因子 s_j,令
        x_smooth[:,j] = x[:,j] / s_j        (激活变平滑,离群被压下去)
        W_smooth[:,j] = W[:,j] * s_j        (权重吸收了这部分尺度)
   数学恒等:x_smooth @ W_smoothᵀ == x @ Wᵀ

   s_j 的取法(SmoothQuant 论文):
        s_j = max|x[:,j]|^α / max|W[:,j]|^(1-α)      α∈[0,1] 控制迁移多少
   含义:激活越离群的通道,s_j 越大,被压得越多;代价是权重那一列被放大,
        但权重"乖",放大一点也还好量化。两边都变得容易量化 → W8A8 可行。

这就是为什么 W8A8(INT8/FP8)方案几乎都要配 SmoothQuant 这类预处理:
不处理离群值,激活量化误差会大到不可用(02 文件 W8A8 的 5% 误差就是例子)。

跑:  python quant/04_activation_smoothquant.py
"""

import torch
import torch.nn.functional as F


def per_token_quant_dequant(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q * scale


def per_channel_quant_dequant(w, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(w / scale).clamp(-qmax - 1, qmax)
    return q * scale


def rel_err(ref, out):
    return ((ref - out).abs().norm() / ref.norm() * 100).item()


def main():
    torch.manual_seed(0)
    in_f, out_f, tokens = 4096, 4096, 64

    W = torch.randn(out_f, in_f) * (1.0 / in_f ** 0.5)
    x = torch.randn(tokens, in_f) * 0.5
    # 制造系统性离群通道:固定的几个输入通道,所有 token 上都很大
    outlier_ch = [17, 256, 1234, 3000]
    for j in outlier_ch:
        x[:, j] += 30.0  # 加偏置,模拟"持续出现在固定通道"的离群

    y_ref = F.linear(x, W)

    print("=" * 70)
    print("第一步:看清激活的离群有多夸张")
    print("=" * 70)
    chan_max = x.abs().amax(dim=0)  # 每个输入通道在所有 token 上的最大幅度
    print(f"  正常通道幅度中位数 ≈ {chan_max.median():.2f}")
    print(f"  离群通道幅度       ≈ {chan_max.amax():.2f}   <- 差了 ~{(chan_max.amax()/chan_max.median()):.0f} 倍")
    print("  per-token 量化时,scale 被这 4 个通道撑大,其余 4092 个通道被压扁。")

    # -------- 不做处理的 W8A8 --------
    xq = per_token_quant_dequant(x, 8)
    wq = per_channel_quant_dequant(W, 8)
    y_plain = F.linear(xq, wq)

    # -------- SmoothQuant:先迁移难度,再做 W8A8 --------
    alpha = 0.5
    act_max = x.abs().amax(dim=0).clamp(min=1e-5)        # [in_f]
    wgt_max = W.abs().amax(dim=0).clamp(min=1e-5)        # [in_f] 按输入通道
    s = (act_max ** alpha) / (wgt_max ** (1 - alpha))
    s = s.clamp(min=1e-4)
    x_smooth = x / s.unsqueeze(0)
    W_smooth = W * s.unsqueeze(0)
    # 看迁移后激活离群是否被压下去
    smooth_max = x_smooth.abs().amax(dim=0)
    print()
    print("=" * 70)
    print("第二步:SmoothQuant 迁移后,激活离群被压平")
    print("=" * 70)
    print(f"  迁移前 激活最大通道幅度 = {chan_max.amax():.2f}")
    print(f"  迁移后 激活最大通道幅度 = {smooth_max.amax():.2f}   <- 被压下来了")

    xq_s = per_token_quant_dequant(x_smooth, 8)
    wq_s = per_channel_quant_dequant(W_smooth, 8)
    y_smooth = F.linear(xq_s, wq_s)

    print()
    print("=" * 70)
    print("第三步:W8A8 输出误差对比")
    print("=" * 70)
    print(f"  朴素 W8A8(不处理离群)      相对误差 = {rel_err(y_ref, y_plain):.3f}%")
    print(f"  SmoothQuant + W8A8           相对误差 = {rel_err(y_ref, y_smooth):.3f}%")
    print()
    print("  结论:激活难量化的根因是'固定通道的系统性离群值',不能裁、不能忽略;")
    print("  SmoothQuant 用一次 per-channel 恒等缩放,把难度从激活搬到(更乖的)权重,")
    print("  让 W8A8 重新变得可用。FP8 方案同理,只是格点是浮点排布、动态范围更大。")


if __name__ == "__main__":
    main()
