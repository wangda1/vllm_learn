"""
sampling_demo.py
================
教学示例：从 logits 到 next token 的【采样链路】核心原理（对应 Q1~Q4）。

和 spec_decode_demo.py 一样：纯 Python 标准库（math/random），零依赖，可运行可验证。
spec_decode_demo.py 讲"怎么减少大模型调用次数"，本文件讲"每一步大模型吐出 logits 之后，
到底怎么变成下一个 token"——这是 spec decode 里 RejectionSampler 也要复用的同一套采样语义。

  第 0 课  logits -> 概率 -> token：整条采样流水线鸟瞰
  第 1 课  greedy / temperature：温度在数学上对 softmax 做了什么（T->0, T->inf 两极）
  第 2 课  top-k / top-p(nucleus) / min-p：三种"截断候选集"的策略各自在裁什么
  第 3 课  流水线顺序：temperature / top-k / top-p 谁先谁后，换序会不会变结果
  第 4 课  三种 penalty：repetition / frequency / presence 分别改的是什么量
  第 5 课  批量异构采样：一个 batch 里每个请求带不同 temp/top_p/seed，一次性采出来
  第 6 课  确定性与 batch-invariance：固定 seed + T=0 为什么 batch 推理仍可能不一致
  第 7 课  logprobs：怎么算、返回 top-n 的代价在哪
  第 8 课  约束解码 / JSON：往 logits 上做什么操作，和 xgrammar 是什么关系

运行：
    python sampling_demo.py
"""

import math
import random
from collections import Counter

NEG_INF = float("-inf")

# 一个玩具词表与一组 logits（大模型某一步对每个 token 打的"原始分数"）
VOCAB = ["猫", "狗", "鱼", "鸟", "树", "石头", "。", "<eos>"]
#         强       次强         中            弱             很弱
LOGITS = [3.0, 2.0, 1.5, 0.5, 0.0, -1.0, 0.2, -2.0]


# ============================================================
# 基础算子：softmax(带温度) / top-k / top-p / min-p
# 约定：过滤类算子都在 "logits 空间" 把被淘汰的 token 置为 -inf，
#       最后统一 softmax + 采样。这与 vLLM sampler 的做法一致。
# ============================================================

def softmax(logits, temperature=1.0):
    """带温度的 softmax：p_i = exp(z_i / T) / sum_j exp(z_j / T)。

    温度在数学上就是"给 logits 整体除以 T"再做 softmax：
      - T = 1 : 原始分布
      - T -> 0 : 最大 logit 的概率 -> 1，其余 -> 0，退化成 argmax(greedy)
      - T -> inf : z_i/T -> 0，所有 token 概率趋于均匀 1/V（最大随机）
    T<1 让分布更尖锐(更确定)，T>1 让分布更平坦(更随机)。
    """
    if temperature <= 0:                       # greedy：一热分布
        m = max(logits)
        # 处理并列最大值时取第一个，保持确定性
        out = [0.0] * len(logits)
        out[logits.index(m)] = 1.0
        return out
    scaled = [z / temperature for z in logits]
    m = max(v for v in scaled if v != NEG_INF)  # 数值稳定：减去最大值
    exps = [math.exp(v - m) if v != NEG_INF else 0.0 for v in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def apply_top_k(logits, k):
    """只保留 logit 最大的 k 个 token，其余置 -inf。
    top-k 是【按排名】截断：固定保留 k 个候选，不管它们概率多大。"""
    if k is None or k <= 0 or k >= len(logits):
        return list(logits)
    kth = sorted(logits, reverse=True)[k - 1]   # 第 k 大的 logit 作为阈值
    return [z if z >= kth else NEG_INF for z in logits]


def apply_top_p(logits, p, temperature=1.0):
    """nucleus 采样：按概率从大到小累加，保留"累计概率刚达到 p"的最小集合，其余置 -inf。
    top-p 是【按累计概率质量】截断：候选集大小随分布形状自适应——
      分布越尖锐保留越少，越平坦保留越多。
    注意：要用"概率"比较，所以这里需要先按当前 temperature 算出概率。"""
    if p is None or p >= 1.0:
        return list(logits)
    probs = softmax(logits, temperature)
    order = sorted(range(len(logits)), key=lambda i: probs[i], reverse=True)
    keep, cum = set(), 0.0
    for i in order:
        keep.add(i)
        cum += probs[i]
        if cum >= p:                            # 一旦累计达到 p 就停（含跨过 p 的那个）
            break
    return [logits[i] if i in keep else NEG_INF for i in range(len(logits))]


def apply_min_p(logits, min_p, temperature=1.0):
    """min-p 采样：保留 prob >= min_p * max_prob 的 token，其余置 -inf。
    阈值是"相对最大概率"的比例——比 top-p 更看重"和最强候选差多少"。
    最强候选越确定(max_prob 越大)，绝对门槛越高，保留越少。"""
    if min_p is None or min_p <= 0:
        return list(logits)
    probs = softmax(logits, temperature)
    thresh = min_p * max(probs)
    return [logits[i] if probs[i] >= thresh else NEG_INF
            for i in range(len(logits))]


def multinomial_sample(probs, rng):
    """按概率分布采样一个下标。"""
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]


def show(logits, temperature=1.0, tag=""):
    """打印某组 logits 经 softmax 后的分布（只显示非零项）。"""
    probs = softmax(logits, temperature)
    items = [(VOCAB[i], probs[i]) for i in range(len(probs)) if probs[i] > 1e-9]
    items.sort(key=lambda kv: kv[1], reverse=True)
    s = "  ".join(f"{t}:{p:.3f}" for t, p in items)
    print(f"    {tag:<22} {s}")


# ============================================================
# 第 0 课：流水线鸟瞰
# ============================================================

def lesson0():
    print("=" * 72)
    print("第 0 课：从 logits 到 next token 的整条流水线")
    print("=" * 72)
    print("""  大模型每一步对词表里每个 token 输出一个 logit(原始分数)。把它变成 token：

    logits(|V|维)
      │  ① 惩罚类 logits processor（repetition/frequency/presence、bad-words、约束解码 mask）
      │     —— 直接在 logits 上加减/置 -inf
      ▼
      │  ② 除以 temperature（缩放）
      ▼
      │  ③ top-k（按排名截断候选集）
      ▼
      │  ④ top-p / min-p（按概率质量截断候选集）
      ▼
      │  ⑤ softmax -> 概率分布
      ▼
      │  ⑥ 多项式采样（greedy 则直接 argmax）
      ▼
     next token

  本组玩具 logits（已按强弱排好）：""")
    for t, z in zip(VOCAB, LOGITS):
        print(f"      {t:<6} logit={z:+.1f}")
    show(LOGITS, 1.0, "原始分布 (T=1)：")


# ============================================================
# 第 1 课：greedy 与 temperature
# ============================================================

def lesson1():
    print("\n" + "=" * 72)
    print("第 1 课：greedy 与 temperature（温度对 softmax 做了什么）")
    print("=" * 72)
    print("  temperature 把 logits 整体除以 T 再 softmax。看两极与中间：")
    show(LOGITS, 0.01, "T->0  (≈greedy):")     # 退化为 argmax，'猫' 概率->1
    show(LOGITS, 0.5,  "T=0.5 (更尖锐):")
    show(LOGITS, 1.0,  "T=1   (原始):")
    show(LOGITS, 2.0,  "T=2   (更平坦):")
    show(LOGITS, 100.0, "T->inf(≈均匀):")       # 趋于 1/V
    print("  结论：T<1 更确定(尖)，T>1 更随机(平)；T->0 退化 argmax，T->inf 退化均匀分布。")
    print(f"  均匀分布每项 = 1/|V| = {1/len(VOCAB):.3f}，与 T=100 行基本一致。")


# ============================================================
# 第 2 课：top-k / top-p / min-p
# ============================================================

def lesson2():
    print("\n" + "=" * 72)
    print("第 2 课：top-k / top-p(nucleus) / min-p —— 三种候选集截断")
    print("=" * 72)
    show(LOGITS, 1.0, "不截断：")
    show(apply_top_k(LOGITS, 3), 1.0, "top-k=3 (留排名前3)：")
    show(apply_top_p(LOGITS, 0.9, 1.0), 1.0, "top-p=0.9 (留累计概率0.9)：")
    show(apply_min_p(LOGITS, 0.1, 1.0), 1.0, "min-p=0.1 (留>=0.1*max)：")
    print("""  区别一句话：
    - top-k ：固定数量。无论分布怎样，留排名前 k 个。简单但不自适应。
    - top-p ：固定"累计概率质量"。候选集大小随分布尖/平自适应（尖->少，平->多）。
    - min-p ：相对最强候选的比例门槛。最强越确定，门槛越高，保留越少；对温度更鲁棒。""")


# ============================================================
# 第 3 课：流水线顺序会不会影响结果
# ============================================================

def lesson3():
    print("\n" + "=" * 72)
    print("第 3 课：temperature / top-k / top-p 的先后顺序")
    print("=" * 72)
    print("  vLLM 顺序：penalties -> temperature -> top-k -> top-p/min-p -> softmax -> sample\n")

    # (a) top-k 只看排名，温度单调不改变排名 -> top-k 的候选集与温度先后无关
    a1 = apply_top_k([z / 0.5 for z in LOGITS], 3)        # 先温度后 top-k
    a2 = [z / 0.5 if math.isfinite(z) else z              # 先 top-k 后温度
          for z in apply_top_k(LOGITS, 3)]
    keep1 = {i for i, z in enumerate(a1) if math.isfinite(z)}
    keep2 = {i for i, z in enumerate(a2) if math.isfinite(z)}
    print(f"  (a) top-k vs temperature：候选集 {'相同' if keep1 == keep2 else '不同'}")
    print("      原因：温度是单调变换，不改变 logits 排名，所以不影响 top-k 选谁。")

    # (b) top-p 看的是"概率值"，温度会改概率 -> 先后顺序影响 nucleus 集合
    keep_before = {VOCAB[i] for i, z in enumerate(apply_top_p(LOGITS, 0.9, 2.0))
                   if math.isfinite(z)}                    # 温度=2 先作用再 top-p
    keep_after = {VOCAB[i] for i, z in enumerate(apply_top_p(LOGITS, 0.9, 1.0))
                  if math.isfinite(z)}                     # 不加温度直接 top-p
    print(f"\n  (b) top-p vs temperature：")
    print(f"      先用 T=2 平滑再 top-p=0.9 -> 候选 {sorted(keep_before)}")
    print(f"      直接 top-p=0.9(T=1)       -> 候选 {sorted(keep_after)}")
    print(f"      候选集 {'相同' if keep_before == keep_after else '不同'}：top-p 比较的是概率，")
    print("      温度改变概率值 => 温度与 top-p 换序，nucleus 集合会变。")
    print("\n  小结：top-k 对温度顺序不敏感(只看排名)；top-p / min-p 对温度顺序敏感(看概率)。")


# ============================================================
# 第 4 课：repetition / frequency / presence penalty
# ============================================================

def apply_penalties(logits, counts, *, repetition=1.0, frequency=0.0, presence=0.0):
    """在 logits 上施加三种惩罚（counts: token下标 -> 已出现次数）。
      - repetition penalty(乘性, HF/CTRL 风格)：出现过就 z/=r (z>0) 或 z*=r (z<0)，
        把已出现 token 往 0 拉。改的是"是否出现过"(布尔)，幅度由 r 控制。
      - frequency penalty(加性, 正比次数)：z -= frequency * count。出现越多扣越多。
      - presence penalty(加性, 一次性)：z -= presence * 1{count>0}。出现过就扣固定值，
        与次数无关。
    OpenAI 风格用 frequency/presence；HF 风格用 repetition。"""
    out = list(logits)
    for i in range(len(out)):
        c = counts.get(i, 0)
        if c > 0 and repetition != 1.0:
            out[i] = out[i] / repetition if out[i] > 0 else out[i] * repetition
        if frequency:
            out[i] -= frequency * c
        if presence:
            out[i] -= presence * (1 if c > 0 else 0)
    return out


def lesson4():
    print("\n" + "=" * 72)
    print("第 4 课：repetition / frequency / presence penalty")
    print("=" * 72)
    # 假设 '猫' 已出现 3 次、'狗' 出现 1 次
    counts = {0: 3, 1: 1}
    print("  已生成历史里：'猫'×3, '狗'×1。看三种惩罚如何压低它们的概率：\n")
    show(LOGITS, 1.0, "无惩罚：")
    show(apply_penalties(LOGITS, counts, repetition=1.3), 1.0, "repetition=1.3：")
    show(apply_penalties(LOGITS, counts, frequency=0.7), 1.0, "frequency=0.7：")
    show(apply_penalties(LOGITS, counts, presence=0.7), 1.0, "presence=0.7：")
    print("""  改的量不同：
    - presence  ：只看"出没出现过"(0/1)，'猫'和'狗'扣一样多 -> 鼓励引入新词。
    - frequency ：正比"出现次数"，'猫'(×3)被扣得比'狗'(×1)狠 -> 抑制高频复读。
    - repetition：乘性把已出现 token 往 0 拉(>0 变小, <0 变大)，与正负号有关。""")


# ============================================================
# 第 5 课：批量异构采样（一次 forward，逐请求不同 temp/top_p/seed）
# ============================================================

def sample_one(logits, *, temperature, top_k, top_p, min_p, rng):
    """单请求完整采样：按 vLLM 顺序 temp -> top_k -> top_p -> min_p -> softmax -> sample。"""
    z = apply_top_k(logits, top_k)
    z = apply_top_p(z, top_p, temperature)
    z = apply_min_p(z, min_p, temperature)
    probs = softmax(z, temperature)
    return multinomial_sample(probs, rng)


def lesson5():
    print("\n" + "=" * 72)
    print("第 5 课：批量异构采样（同一次 forward，逐请求不同参数/seed）")
    print("=" * 72)
    print("""  关键：sampler 在 GPU 上向量化执行。不同请求的 temperature/top_p 等被打包成
  per-row 的张量(形状 [batch])，对 logits[batch, vocab] 逐行广播；seed 不同则各自
  维护独立的随机发生器状态。所以"一次 forward 出 [batch,vocab] logits，再向量化采样"
  就能把异构请求一起处理，不需要拆开循环。下面用 4 个请求模拟：\n""")
    batch = [
        dict(name="贪心",      temperature=0.0, top_k=0,  top_p=1.0, min_p=0.0, seed=1),
        dict(name="保守",      temperature=0.7, top_k=0,  top_p=0.9, min_p=0.0, seed=1),
        dict(name="发散",      temperature=1.3, top_k=0,  top_p=1.0, min_p=0.0, seed=1),
        dict(name="top_k=2",   temperature=1.0, top_k=2,  top_p=1.0, min_p=0.0, seed=42),
    ]
    print(f"  {'请求':<8}{'T':>5}{'top_k':>7}{'top_p':>7}{'seed':>6}   采样3次")
    for cfg in batch:
        outs = []
        for trial in range(3):
            rng = random.Random(cfg["seed"] + trial * 1000)
            idx = sample_one(LOGITS, temperature=cfg["temperature"], top_k=cfg["top_k"],
                             top_p=cfg["top_p"], min_p=cfg["min_p"], rng=rng)
            outs.append(VOCAB[idx])
        print(f"  {cfg['name']:<8}{cfg['temperature']:>5.1f}{cfg['top_k']:>7}"
              f"{cfg['top_p']:>7.1f}{cfg['seed']:>6}   {outs}")
    print("\n  贪心请求 3 次恒为 '猫'(argmax)；发散请求明显更跳。各请求互不干扰。")


# ============================================================
# 第 6 课：确定性 与 batch-invariance（Q3 热点）
# ============================================================

def lesson6():
    print("\n" + "=" * 72)
    print("第 6 课：确定性与 batch-invariance —— 固定 seed+T=0 为何仍可能不一致")
    print("=" * 72)
    print("""  根因不在采样的随机数，而在【浮点加法不满足结合律】：(a+b)+c != a+(b+c)。
  GPU 上 matmul/reduction 的累加顺序依赖 batch 大小、kernel 的 tiling/split-K 策略；
  同一个请求在 batch=1 和 batch=32 里，归约顺序不同 -> logits 在低位 bit 上有微小差异。
  平时无所谓，但当两个 token 的 logit 几乎并列时，T=0 的 argmax 会被这点噪声"翻盘"，
  于是"同 prompt、同 seed、T=0"两次请求结果不同。\n""")

    # (1) 演示浮点非结合性：同一组数，不同累加顺序，结果不同
    vals = [1e16, 1.0, -1e16, 1.0, 1.0]
    fwd = 0.0
    for v in vals:
        fwd += v                                # 顺序累加
    rev = 0.0
    for v in reversed(vals):
        rev += v                                # 逆序累加
    print(f"  (1) 同一组数不同累加顺序：正序={fwd}  逆序={rev}  -> 相等? {fwd == rev}")

    # (2) 演示近并列 argmax 因归约顺序翻盘
    # token A 的 logit 由三项归约而来：一个大正、一个 +1、一个大负，本应=1.0。
    # 不同 batch 的 kernel 用不同的归约顺序：
    #   顺序1 先 (大正 + 1) 再 - 大正 -> 那个 +1 被舍入吃掉 -> 得 0.0
    #   顺序2 先 (大正 - 大正)=0 再 + 1 -> 精确保留 -> 得 1.0
    big = 1e16
    a_terms_sum_order1 = (big + 1.0) - big      # = 0.0（+1 在大数面前被舍入丢失）
    a_terms_sum_order2 = 1.0 + (big - big)      # = 1.0（先精确抵消再加）
    logitB = 0.5                                # token B 的 logit 恰好夹在 0 和 1 之间
    am1 = "A" if a_terms_sum_order1 > logitB else "B"
    am2 = "A" if a_terms_sum_order2 > logitB else "B"
    print(f"\n  (2) 近并列 argmax（logitA 本应=1.0，logitB={logitB}）：")
    print(f"      归约顺序1: logitA = (1e16+1)-1e16 = {a_terms_sum_order1}  -> argmax={am1}")
    print(f"      归约顺序2: logitA = 1+(1e16-1e16) = {a_terms_sum_order2}  -> argmax={am2}")
    print(f"      两种顺序 argmax {'翻转了!' if am1 != am2 else '一致'} "
          f"<- batch 改变归约顺序，T=0 的结果就可能不同")

    print("""
  怎么做到 batch-invariant 可复现（2025 业界做法，如 Thinking Machines 的工作）：
    - 用"batch 不变"的 kernel：固定归约顺序/固定 split 策略，使每行 logits 与 batch 无关；
      RMSNorm/matmul/attention 都要选 batch-invariant 实现。
    - 固定一切非确定源：cuBLAS workspace、原子加(用确定性归约替代)、flash-attn 的 split。
    - 采样侧本就可控：同 seed + 同算法 + 同顺序即可复现；难的是上游 logits 逐 bit 一致。
  代价：通常牺牲一些吞吐（放弃了对当前 batch 最优的 kernel 配置）。""")


# ============================================================
# 第 7 课：logprobs
# ============================================================

def lesson7():
    print("\n" + "=" * 72)
    print("第 7 课：logprobs 怎么算、返回 top-n 的代价")
    print("=" * 72)
    print("""  logprob(token) = log(softmax(logits)[token]) = logit/T - logsumexp(logits/T)。
  注意 vLLM 默认返回的是"采样后处理过的"分布的 logprob（受 temperature/top-k/p 影响）。\n""")
    T = 1.0
    probs = softmax(LOGITS, T)
    logps = [math.log(p) if p > 0 else NEG_INF for p in probs]
    order = sorted(range(len(LOGITS)), key=lambda i: logps[i], reverse=True)
    print(f"  {'token':<8}{'prob':>8}{'logprob':>10}")
    for i in order[:5]:
        print(f"  {VOCAB[i]:<8}{probs[i]:>8.3f}{logps[i]:>10.3f}")
    # 验证 logsumexp 关系
    lse = math.log(sum(math.exp(z / T) for z in LOGITS))
    chk = LOGITS[0] / T - lse
    print(f"\n  验证：'猫' 的 logprob = logit/T - logsumexp = {LOGITS[0]/T:.3f} - {lse:.3f}"
          f" = {chk:.3f}  (与上表一致)")
    print("""
  返回 top-n logprobs 的代价：
    - 算 logprob 本身几乎免费（softmax 已有），主要代价是"取 top-n + 传回 CPU"：
      词表 128k+ 时要在 [batch, 128k] 上做 top-n 选择 + 排序，再 D2H 拷贝 n 个值/请求。
    - n 越大、batch 越大，gather/sort 和 PCIe 传输越贵；prompt logprobs(对整段 prompt
      逐位置都要 logprob)代价更大，因为要保留并计算每个位置的分布。""")


# ============================================================
# 第 8 课：约束解码 / JSON（logit mask）与 xgrammar
# ============================================================

def lesson8():
    print("\n" + "=" * 72)
    print("第 8 课：约束解码 / JSON 模式 —— 往 logits 上做什么")
    print("=" * 72)
    print("""  做法：在采样前，把"当前语法状态下不合法的 token"的 logit 置 -inf（mask 掉），
  这样 softmax 后它们概率为 0，绝不会被采样到。约束解码 = 一个特殊的 logits processor。

  谁来决定哪些 token 合法？一个跟随已生成 token 前进的状态机：
    - JSON/正则 -> 有限状态自动机(FSA)；上下文无关文法 -> 下推自动机(PDA)。
    - 每一步根据当前状态，预计算一个"允许 token 的 bitmask"，套到 logits 上。

  这正是 SGLang/vLLM 用 xgrammar 做的事：xgrammar 把 grammar 编译成自动机，
  高效地为每一步生成 token bitmask（还会缓存状态、用字节级 trie 加速），
  再作为 logits processor 应用。和采样的关系：它只改"候选集"(mask)，
  之后 temperature/top-k/top-p/采样 的流程完全不变。\n""")

    # 玩具：强制只能从 {"鱼","。","<eos>"} 里选（模拟"当前语法只允许这几个 token"）
    allowed = {"鱼", "。", "<eos>"}
    mask = [0.0 if VOCAB[i] in allowed else NEG_INF for i in range(len(VOCAB))]
    masked = [LOGITS[i] + mask[i] for i in range(len(VOCAB))]
    print(f"  约束：当前状态只允许 {sorted(allowed)}")
    show(LOGITS, 1.0, "约束前分布：")
    show(masked, 1.0, "约束后分布：")
    rng = random.Random(0)
    cnt = Counter(VOCAB[multinomial_sample(softmax(masked, 1.0), rng)] for _ in range(5000))
    illegal = sum(v for t, v in cnt.items() if t not in allowed)
    print(f"  采样 5000 次落在非法 token 上的次数 = {illegal} (应为 0)")
    print("  -> 被 mask 的 token 概率严格为 0，保证输出 100% 符合语法/JSON schema。")


def main():
    lesson0(); lesson1(); lesson2(); lesson3(); lesson4()
    lesson5(); lesson6(); lesson7(); lesson8()
    print("\n" + "=" * 72)
    print("总结：采样链路 = [惩罚/约束 mask] -> /T -> top-k -> top-p/min-p -> softmax -> 采样")
    print("  · temperature: 缩放 logits，控制尖/平；T->0 贪心，T->inf 均匀")
    print("  · top-k(排名截断) / top-p(概率质量截断) / min-p(相对最强的比例门槛)")
    print("  · 顺序：top-k 对温度顺序不敏感；top-p/min-p 敏感(看概率)")
    print("  · penalty: presence(出现过) / frequency(出现次数) / repetition(乘性拉向0)")
    print("  · 工程：GPU 上向量化按 per-row 参数采样；batch-invariance 难在上游 logits 逐 bit 一致")
    print("  · 约束解码 = logit mask(-inf)，xgrammar 负责把 grammar 编译成每步 bitmask")
    print("=" * 72)


if __name__ == "__main__":
    main()
