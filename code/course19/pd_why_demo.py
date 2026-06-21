#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
course19 · Demo 1 —— 为什么要做 PD（Prefill / Decode）分离？
================================================================

对应 DOC.md：第一章 1.1「为什么要做 PD 分离部署」

核心要回答的问题：
    Prefill 和 Decode 明明是同一个模型、同一段代码，为什么要拆到不同的机器上跑？

一句话结论：
    Prefill 是 **算力密集（compute-bound）**，Decode 是 **显存带宽密集（memory-bound）**，
    两者对硬件的最优形态完全不同；放在同一张卡上还会互相干扰（interference）。

本 demo 用一个 **roofline（屋顶线）玩具模型** 把这件事算出来，纯标准库可运行：
    python pd_why_demo.py

我们不依赖任何 GPU，只用「峰值算力」和「显存带宽」两个数字，
按照 roofline 模型 time = max(compute_time, memory_time) 估算两个阶段的瓶颈。
"""

# --------------------------------------------------------------------------
# 0. 一张「典型高端卡」的两个关键参数（量级接近 H100 SXM，便于教学）
# --------------------------------------------------------------------------
PEAK_FLOPS = 1.0e15      # 1 PFLOP/s   (BF16 峰值算力)
MEM_BW     = 3.35e12     # 3.35 TB/s   (HBM3 显存带宽)

# 模型：一个 8B 稠密模型（Llama-3-8B 量级）
N_PARAMS   = 8.0e9       # 80 亿参数
BYTES_PARAM = 2          # 权重 BF16，每参数 2 字节
BYTES_KV    = 2          # KV cache 也用 BF16

# KV cache 每个 token 占多少字节（Llama-3-8B：32 层 × 8 KV head × 128 dim × 2(K,V) × 2B）
KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * BYTES_KV   # ≈ 1 MB / token


def fmt(x):
    """把大数字格式化成人类可读单位。"""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(x) < 1000:
            return f"{x:6.2f}{unit}"
        x /= 1000
    return f"{x:6.2f}E"


# --------------------------------------------------------------------------
# 1. roofline：一次前向的「算力需求」与「访存需求」
# --------------------------------------------------------------------------
def forward_cost(num_tokens, context_len):
    """
    估算一次前向（处理 num_tokens 个 token）的两类成本。

    返回 (flops, bytes_moved)：
      - flops      ：矩阵乘法浮点运算量。经验公式：每 token 约 2 * N_params 次乘加。
      - bytes_moved：必须从 HBM 搬到计算单元的字节数：
          * 权重：无论处理几个 token，整套权重都要读一遍 → N_params * 2B
          * KV cache：每个 query token 都要读一遍历史 KV → num_tokens * context_len * (每token的KV)
            （注意这里 KV 的访存量随 context_len 线性增长，是 decode 的命门）
    """
    flops = 2 * N_PARAMS * num_tokens

    weight_bytes = N_PARAMS * BYTES_PARAM
    # 每个 query token 都要扫一遍当前已存在的 KV（注意力的本质）
    kv_read_bytes = num_tokens * context_len * KV_BYTES_PER_TOKEN
    bytes_moved = weight_bytes + kv_read_bytes

    return flops, bytes_moved


def analyze(stage, num_tokens, context_len):
    flops, bytes_moved = forward_cost(num_tokens, context_len)

    t_compute = flops / PEAK_FLOPS        # 受算力限制需要的时间
    t_memory  = bytes_moved / MEM_BW      # 受带宽限制需要的时间
    t_real    = max(t_compute, t_memory)  # roofline：实际时间取两者较大值

    # 算术强度 = 每搬 1 字节能做多少次浮点运算。越高越「算力密集」。
    intensity = flops / bytes_moved
    ridge = PEAK_FLOPS / MEM_BW           # 屋顶线拐点：高于它算力受限，低于它带宽受限
    bound = "算力受限 (compute-bound)" if t_compute > t_memory else "带宽受限 (memory-bound)"

    print(f"\n【{stage}】 处理 {num_tokens} token，上下文长度 {context_len}")
    print(f"    算力需求 : {fmt(flops)}FLOP   → 需 {t_compute*1e3:8.3f} ms")
    print(f"    访存需求 : {fmt(bytes_moved)}B     → 需 {t_memory*1e3:8.3f} ms")
    print(f"    算术强度 : {intensity:8.1f} FLOP/Byte   (屋顶线拐点 = {ridge:.1f})")
    print(f"    >>> 瓶颈 : {bound}，单次前向约 {t_real*1e3:.3f} ms")
    return t_real, bound


# --------------------------------------------------------------------------
# 2. 实验：同一个模型，prefill 与 decode 的瓶颈截然相反
# --------------------------------------------------------------------------
def experiment_bottleneck():
    print("=" * 70)
    print("实验一：同一个 8B 模型，Prefill 与 Decode 的瓶颈完全相反")
    print("=" * 70)

    PROMPT_LEN = 2048   # 一段较长的 prompt

    # Prefill：一次性处理 2048 个 prompt token（context = 这 2048 个 token 本身）
    analyze("Prefill", num_tokens=PROMPT_LEN, context_len=PROMPT_LEN)

    # Decode：每步只处理 1 个新 token，但要读 2048 长度的历史 KV
    analyze("Decode (单步)", num_tokens=1, context_len=PROMPT_LEN)

    print("""
解读：
  · Prefill 一把吃 2048 个 token，算力需求极高 → 算力受限。它喜欢 **高 TFLOPS** 的卡。
  · Decode 每步只算 1 个 token，FLOPs 极小，但仍要把整套权重 + 历史 KV 全读一遍，
    几乎纯访存 → 带宽受限。它喜欢 **大显存 / 高带宽** 的卡，对算力反而不敏感。
  · 结论：两个阶段对硬件的「最优形态」不一致。一张既算力强、显存又大的卡非常贵，
    不如把 P 放在算力卡、D 放在大显存卡上，分别按需扩容 —— 这就是 PD 分离的动机。""")


# --------------------------------------------------------------------------
# 3. 实验：放在同一张卡上时的「相互干扰」
# --------------------------------------------------------------------------
def experiment_interference():
    print("\n" + "=" * 70)
    print("实验二：P 与 D 混在一张卡上时的相互干扰（为什么混部体验差）")
    print("=" * 70)

    PROMPT_LEN = 4096

    # 一个正在 decode 的请求，期望的「每 token 时延」TPOT
    t_decode, _ = forward_cost(1, PROMPT_LEN)[0], None
    t_decode_step = max(forward_cost(1, PROMPT_LEN)[0] / PEAK_FLOPS,
                        forward_cost(1, PROMPT_LEN)[1] / MEM_BW)

    # 此时来了一个新请求的 prefill（4096 token），它会霸占计算单元一段时间
    t_prefill = max(forward_cost(PROMPT_LEN, PROMPT_LEN)[0] / PEAK_FLOPS,
                    forward_cost(PROMPT_LEN, PROMPT_LEN)[1] / MEM_BW)

    print(f"\n  正常 decode 一步 TPOT ≈ {t_decode_step*1e3:.3f} ms")
    print(f"  插入一个 {PROMPT_LEN} token 的 prefill 需要 ≈ {t_prefill*1e3:.3f} ms")
    print(f"  → 若不抢占，decode 请求这一步要多等 {t_prefill*1e3:.1f} ms，")
    print(f"    TPOT 被放大约 {t_prefill/t_decode_step:.0f} 倍，用户能明显感到「卡顿 / 吐字结巴」。")

    # 显存维度的干扰
    kv_per_seq_gb = PROMPT_LEN * KV_BYTES_PER_TOKEN / 1e9
    print(f"\n  显存维度：每个长上下文请求常驻 KV ≈ {kv_per_seq_gb:.2f} GB。")
    print(f"  decode 请求越多、上下文越长，常驻 KV 越大，能留给 prefill 大 batch 的显存越少，")
    print(f"  → prefill 吞吐被迫下降。两个阶段在 **算力** 和 **显存** 两个维度上同时抢资源。")

    print("""
解读：
  混部 = 让「算力大胃王（prefill）」和「带宽细水长流（decode）」共用一张卡，
  prefill 的长计算拖慢 decode 的 TPOT，decode 的 KV 常驻又压缩 prefill 的 batch。
  PD 分离把它们物理隔开：各自独占资源、各自按负载扩容，互不干扰。""")


# --------------------------------------------------------------------------
# 4. 什么时候【不】需要 PD 分离
# --------------------------------------------------------------------------
def when_not_to():
    print("\n" + "=" * 70)
    print("反面：什么时候不值得做 PD 分离")
    print("=" * 70)
    print("""
  · 模型小、prompt 短、并发低：单卡算力与显存都富余，prefill 占用时间很短，
    decode 也不会把显存撑爆 —— 此时一体化部署最简单，没必要引入跨节点 KV 传输的复杂度。
  · KV 传输本身有成本（NCCL/RDMA、序列化、握手）。只有当「省下的重复 prefill / 解耦收益」
    明显大于「KV 搬运成本」时，PD 分离才划算。
  → 经验法则：长 prompt、高并发、对 TTFT/TPOT 都有要求的在线服务，才是 PD 分离的主场。""")


if __name__ == "__main__":
    print(__doc__)
    experiment_bottleneck()
    experiment_interference()
    when_not_to()
    print("\n下一课 → kv_handoff_demo.py：KV cache 到底是怎么从 P 交接到 D 的。\n")
