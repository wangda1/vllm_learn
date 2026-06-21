"""
Weight-only 量化为什么需要算法:从 RTN 到 AWQ 的核心思想
=================================================================

02 文件里 W4A16 的误差有 11% 之多,大到会显著掉点。那为什么工业界
(Llama/Qwen 的 AWQ/GPTQ 权重)能把 int4 做到几乎无损?

因为它们不是"傻量化"(RTN, Round-To-Nearest),而是"看激活下菜"(activation-aware)。

本文件用最小代码复现 AWQ(Activation-aware Weight Quantization)的核心直觉。
GPTQ 思路不同(用二阶 Hessian 信息逐列纠错),但出发点一致:
    "不是所有权重一样重要,要把量化误差花在不重要的地方。"

------------------------------------------------------------------------
AWQ 的一句话原理
------------------------------------------------------------------------
对 y = x @ Wᵀ,某些"输入通道 j"上的激活 x[:,j] 特别大(salient,显著通道)。
这些通道对应的权重列 W[:,j] 一旦量化出错,会被大激活放大 → 输出错得多。

技巧:对显著通道,先把权重"放大 s 倍"再量化(放大后相对量化误差变小),
      同时把对应激活"缩小 s 倍",数学上输出完全不变:

          (x / s) @ (W * s)ᵀ  ==  x @ Wᵀ     ← 恒等变换,不改变结果

      关键在于:W*s 之后,显著列的有效量化精度提高了。
      s 怎么选?按激活幅度:  s_j ∝ (mean|x[:,j]|) ^ α   (α 是平滑强度)

跑:  python quant/03_weight_only_awq.py
"""

import torch
import torch.nn.functional as F


def per_group_sym_quant_dequant(w, num_bits=4, group_size=128):
    """RTN 的 group 量化 + 立即反量化,返回反量化后的权重(用于看误差)。"""
    out_f, in_f = w.shape
    qmax = 2 ** (num_bits - 1) - 1
    wg = w.reshape(out_f, in_f // group_size, group_size)
    scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(wg / scale).clamp(-qmax - 1, qmax)
    return (q * scale).reshape(out_f, in_f)


def rel_err(ref, out):
    return ((ref - out).abs().norm() / ref.norm() * 100).item()


def main():
    torch.manual_seed(0)
    in_f, out_f, tokens = 4096, 4096, 64
    group_size = 128

    # 权重 + 校准激活(AWQ 需要一小批校准数据来"看"激活分布)
    W = torch.randn(out_f, in_f) * (1.0 / in_f ** 0.5)

    # ⚠️ 下面这段只是"构造测试数据",不是 AWQ 算法的一部分 ⚠️
    # 真实 LLM 的关键事实:输出信号主要由少数"显著输入通道"驱动,普通通道幅度很小。
    # 这里手工预埋 8 个显著通道(幅度 ~40)来还原这一结构,好让 demo 有戏看。
    # 真实场景里你【并不知道】哪些通道显著——那正是下面第 1) 步要"发现"的事。
    x = torch.randn(tokens, in_f) * 0.1
    salient = list(range(0, in_f, in_f // 8))[:8]   # (仅造数据用)假装这 8 个是显著通道
    for j in salient:
        x[:, j] = torch.randn(tokens) * 40.0

    y_ref = F.linear(x, W)

    # -------- 方案 A:RTN,直接 int4 group 量化 --------
    W_rtn = per_group_sym_quant_dequant(W, 4, group_size)
    y_rtn = F.linear(x, W_rtn)

    # -------- 方案 B:AWQ,先按激活幅度做 per-channel 缩放,再量化 --------
    # 1) ★这才是 AWQ "发现显著通道"的地方★
    #    拿校准激活,统计【每个输入通道】的平均幅度——值大的通道就是显著通道。
    #    关键:判据是【激活】幅度,不是权重幅度。因为 y=x@Wᵀ,显著通道的大激活会
    #    把该列权重的量化误差【放大】到输出,所以要看激活。AWQ 论文实测:按激活选
    #    远好于按权重 |W| 选,后者几乎等同没选。这里对全部 in_f 个通道统一统计,
    #    无需 if j in salient——显著性会通过下面连续的 per-channel scale 自动体现。
    act_scale = x.abs().mean(dim=0)                  # [in_f],每个输入通道的平均幅度
    # 2) 由激活幅度生成缩放因子 s_j:激活越大 → s_j 越大 → 该列被保护得越多。
    #    注意 AWQ 不是"离散挑出 top-k 个显著通道单独处理",而是这条连续缩放曲线;
    #    α 控制保护强度(α 越大越偏向显著通道),AWQ 论文用 grid search 搜最优 α。
    alpha = 0.5
    s = act_scale.clamp(min=1e-4) ** alpha
    s = s / s.mean()                                 # 归一化,避免整体尺度漂移

    # 2.5) ★把"发现显著通道"这一步可视化★
    #      纯粹为了看清楚:取激活幅度最大的几个通道,看它们的 s_j 被放大了多少倍。
    #      注意算法本身【不需要】这一步——它对全部 in_f 个通道统一缩放,这里只是打印。
    topk = 8
    top_val, top_idx = act_scale.topk(topk)          # 激活幅度最大的 topk 个通道
    print("=" * 70)
    print(f"AWQ 发现的显著通道(按校准激活幅度 act_scale 排序,Top-{topk}):")
    print(f"  {'通道 j':>8} | {'平均激活|x_j|':>14} | {'缩放 s_j':>10} | {'相对普通通道放大':>16}")
    print("  " + "-" * 60)
    for j, v in zip(top_idx.tolist(), top_val.tolist()):
        print(f"  {j:>8} | {v:>14.3f} | {s[j].item():>10.3f} | {s[j].item():>15.2f}x")
    # 普通(非显著)通道作对照:s_j 接近甚至小于 1,几乎不被保护
    median_ch = act_scale.argsort()[in_f // 2].item()
    print(f"  {'(中位通道)':>8} | {act_scale[median_ch].item():>14.3f} | "
          f"{s[median_ch].item():>10.3f} | {s[median_ch].item():>15.2f}x")
    print(f"  → 显著通道 s_j 明显 >1(被放大保护),普通通道 s_j≈1 甚至<1(让出精度)。")
    print()

    # 3) 恒等变换:权重列乘 s,激活列除 s
    W_scaled = W * s.unsqueeze(0)                     # [out, in] * [1, in]
    x_scaled = x / s.unsqueeze(0)
    # 4) 对"放大后的权重"做同样的 int4 量化
    W_scaled_q = per_group_sym_quant_dequant(W_scaled, 4, group_size)
    y_awq = F.linear(x_scaled, W_scaled_q)            # 注意用 x_scaled

    print("=" * 70)
    print("int4 group 量化:RTN(傻量化) vs AWQ(看激活下菜)")
    print("=" * 70)
    print(f"  RTN  (直接量化)        输出相对误差 = {rel_err(y_ref, y_rtn):.3f}%")
    print(f"  AWQ  (激活感知缩放)    输出相对误差 = {rel_err(y_ref, y_awq):.3f}%")
    print()
    print("  原理回顾:")
    print("  - 显著通道(大激活)对应的权重列被放大后量化,相对误差变小;")
    print("  - 激活同步缩小,数学上输出不变 → 误差被'挪'到了不重要的通道上。")
    print("  - 真实 AWQ 还会用 grid search 找最优 α,并保护极少数通道,效果更好。")
    print("  - 注意:AWQ 的收益在'输出由少数显著通道主导'时最大(真实 LLM 正是如此);")
    print("    若误差均匀分布在所有通道,缩放能改善的空间就有限。")
    print()
    print("  和 GPTQ 的区别:GPTQ 用 Hessian(二阶信息)逐列量化并把误差补偿给")
    print("  后面未量化的列;AWQ 用一次缩放+RTN,更快、无需反传。两者都是 weight-only,")
    print("  目标都是让 W4A16 在 decode(memory-bound)阶段又省显存又不掉点。")


if __name__ == "__main__":
    main()
