"""
量化的数值基础:一个浮点张量是怎么变成"整数 + scale"的
=================================================================

你说你只了解"量化=降精度"。这个文件把"降精度"这件事拆开,讲清楚到底发生了什么。

核心一句话:
    量化 = 用"低位宽整数 q"+"一个缩放因子 scale"(可能再加一个 zero_point)
           去近似表示"高位宽浮点 x"。
           反量化时:  x ≈ (q - zero_point) * scale

所以量化省的从来不是"算力本身",而是:
    1) 存储/显存:  权重从 fp16(2字节) 变成 int8(1字节)/int4(0.5字节)
    2) 访存带宽:   从显存搬到计算单元的字节数变少

记住这两点,后面"W4A16 在 decode 阶段到底快不快"才讲得清(见 DOC.md 第③问)。

本文件只用 CPU + torch 就能跑:
    python quant/01_quant_numerics.py
"""

import torch


# ---------------------------------------------------------------------------
# 0. 位宽与取值范围:int8 / int4 能表示多少个数
# ---------------------------------------------------------------------------
# 量化的本质是把连续的浮点数,映射到一个"格点很少"的整数网格上。
#   int8  (有符号):  [-128, 127]      共 256 个格点
#   uint8 (无符号):  [0, 255]         共 256 个格点
#   int4  (有符号):  [-8, 7]          共 16  个格点   <- 注意:只有 16 个!
# 格点越少,两个相邻格点之间的"台阶"越大,量化误差越大。
# 这就是为什么 int4 比 int8 难做,需要更聪明的算法(GPTQ/AWQ,见 03/04 文件)。


def qrange(num_bits: int, signed: bool = True):
    """返回某个位宽下整数的 [qmin, qmax]。"""
    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
    return qmin, qmax


# ---------------------------------------------------------------------------
# 1. 对称量化(symmetric):zero_point = 0,最常用于"权重"
# ---------------------------------------------------------------------------
# 思路:假设数据大致以 0 为中心(权重通常如此),只需要一个 scale。
#   scale = max(|x|) / qmax
#   q     = round(x / scale)           然后 clamp 到 [qmin, qmax]
#   x'    = q * scale                  (反量化)
# 优点:推理时矩阵乘法里没有 zero_point 的偏置项,kernel 更简单更快。
def symmetric_quantize(x: torch.Tensor, num_bits: int = 8):
    qmin, qmax = qrange(num_bits, signed=True)
    # scale 由数据里绝对值最大的那个数决定(它要正好落在 qmax 上)
    amax = x.abs().max().clamp(min=1e-8)
    scale = amax / qmax
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q, scale


def symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q * scale


# ---------------------------------------------------------------------------
# 2. 非对称量化(asymmetric):带 zero_point,常用于"激活"
# ---------------------------------------------------------------------------
# 思路:激活经过 ReLU/SiLU 后往往是单边分布(比如全是正数),
#       用 [min, max] 整个区间去映射 [qmin, qmax] 更省格点。
#   scale      = (max - min) / (qmax - qmin)
#   zero_point = round(qmin - min / scale)   # 让浮点 0 能被精确表示
#   q          = round(x / scale) + zero_point
#   x'         = (q - zero_point) * scale
def asymmetric_quantize(x: torch.Tensor, num_bits: int = 8):
    qmin, qmax = qrange(num_bits, signed=False)  # 非对称常配 uint
    xmin = x.min()
    xmax = x.max()
    scale = (xmax - xmin).clamp(min=1e-8) / (qmax - qmin)
    zero_point = torch.round(qmin - xmin / scale)
    q = (torch.round(x / scale) + zero_point).clamp(qmin, qmax)
    return q, scale, zero_point


def asymmetric_dequantize(q, scale, zero_point):
    return (q - zero_point) * scale


# ---------------------------------------------------------------------------
# 3. 量化粒度(granularity):同一份权重,用几个 scale?
# ---------------------------------------------------------------------------
# 这是量化里最重要、最容易被忽略的工程点。scale 越"细",越能贴合局部数据,
# 误差越小,但存的 scale 也越多。三种常见粒度:
#
#   per-tensor   整个张量 1 个 scale            最省、最糙
#   per-channel  每个输出通道 1 个 scale         权重量化的主流(W8A8/W8A16)
#   per-group    每 group_size 个元素 1 个 scale  W4 的标配(group=128)
#
# 你之前说"对不同的层用不同精度"——那是"层间"的混合精度;
# 这里讲的是"层内"的粒度,两者是不同维度的旋钮,都要会。
def per_channel_symmetric_quantize(w: torch.Tensor, num_bits: int = 8, axis: int = 0):
    """权重形状 [out_features, in_features],按 axis=0(每个输出通道)各算一个 scale。"""
    qmin, qmax = qrange(num_bits, signed=True)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [out, 1]
    scale = amax / qmax
    q = torch.round(w / scale).clamp(qmin, qmax)
    return q, scale


def per_group_symmetric_quantize(w: torch.Tensor, num_bits: int = 4, group_size: int = 128):
    """把每一行(输出通道)再切成若干 group,每个 group 一个 scale。W4 量化几乎都这么干。"""
    out_features, in_features = w.shape
    assert in_features % group_size == 0, "in_features 要能被 group_size 整除(教学起见)"
    qmin, qmax = qrange(num_bits, signed=True)
    # reshape 成 [out, n_group, group_size],在最后一维上求 scale
    w_g = w.reshape(out_features, in_features // group_size, group_size)
    amax = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    q = torch.round(w_g / scale).clamp(qmin, qmax)
    return q.reshape(out_features, in_features), scale  # scale: [out, n_group, 1]


def per_group_dequantize(q, scale, group_size: int = 128):
    out_features, in_features = q.shape
    q_g = q.reshape(out_features, in_features // group_size, group_size)
    return (q_g * scale).reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# 误差度量:量化好不好,看反量化后和原值差多少
# ---------------------------------------------------------------------------
def report(name: str, x: torch.Tensor, x_hat: torch.Tensor):
    err = (x - x_hat).abs()
    # 相对误差用 Frobenius 范数比值,直观反映"量化掉了百分之几"
    rel = err.norm() / x.norm()
    print(f"  {name:<34} 最大绝对误差={err.max():.5f}  相对误差={rel * 100:.3f}%")


def main():
    torch.manual_seed(0)

    print("=" * 70)
    print("1) 对称 vs 非对称:对一份'单边分布'(模拟 SiLU 激活)的数据")
    print("=" * 70)
    # 模拟激活:大部分在 0 附近,但有正的离群值(后面 04 文件会专门讲离群值)
    act = torch.randn(4096).abs() * 0.5
    act[0] = 6.0  # 一个离群值,故意制造麻烦

    q, s = symmetric_quantize(act, 8)
    report("对称 int8", act, symmetric_dequantize(q, s))
    q, s, z = asymmetric_quantize(act, 8)
    report("非对称 uint8", act, asymmetric_dequantize(q, s, z))
    print("  结论:单边分布下,非对称能用上全部 256 个格点,误差更小。")

    print()
    print("=" * 70)
    print("2) 量化粒度:同一份权重 [512, 1024],per-tensor vs per-channel vs per-group")
    print("=" * 70)
    w = torch.randn(512, 1024) * 0.1
    # 制造"通道间尺度差异":让某些输出通道整体偏大,模拟真实权重
    w[100] *= 10.0
    w[200] *= 0.01

    # per-tensor int8
    q, s = symmetric_quantize(w, 8)
    report("per-tensor  int8 (1 个 scale)", w, symmetric_dequantize(q, s))
    # per-channel int8
    q, s = per_channel_symmetric_quantize(w, 8)
    report("per-channel int8 (512 个 scale)", w, (q * s))
    # per-group int4
    q, s = per_group_symmetric_quantize(w, 4, group_size=128)
    report("per-group   int4 (g=128)", w, per_group_dequantize(q, s, 128))
    print("  结论:通道尺度差异大时,per-tensor 被离群通道'拖累';")
    print("        per-channel 显著改善;int4 即使用 group 也明显更糙(格点只有 16 个)。")

    print()
    print("=" * 70)
    print("3) 显存账:量化到底省了多少字节")
    print("=" * 70)
    numel = w.numel()
    for name, bits in [("fp16  权重", 16), ("int8  权重", 8), ("int4  权重", 4)]:
        # int4 还要算上 group scale 的开销(每 group 一个 fp16 scale)
        extra = ""
        bytes_main = numel * bits / 8
        if bits == 4:
            n_scale = numel // 128
            bytes_scale = n_scale * 2  # fp16 scale
            extra = f"  (+{bytes_scale/1024:.1f} KB 的 group scale)"
            print(f"  {name}: {bytes_main/1024:.1f} KB{extra}")
        else:
            print(f"  {name}: {bytes_main/1024:.1f} KB")
    print("  结论:int8≈半个 fp16,int4≈四分之一。模型越大,这笔账越关键——")
    print("        因为 decode 阶段是 memory-bound,省字节≈省时间(详见 DOC.md 第③问)。")


if __name__ == "__main__":
    main()
