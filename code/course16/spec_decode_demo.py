"""
spec_decode_demo.py
===================
教学示例：vLLM 采样与投机解码（Speculative Decoding）核心原理

本示例不依赖 GPU / vLLM / numpy，只用 Python 标准库，用"玩具概率模型"
把 DOC.md 里最核心的几件事讲清楚、并且可运行可验证：

  第 0 课  为什么 decode 慢 —— 投机解码到底在省什么
  第 1 课  投机解码主循环：草稿(draft) → 并行验证(verify) → 接受/拒绝 → bonus
  第 2 课  正确性的"实验证明"：投机解码输出分布 == 目标模型分布（蒙特卡洛）
  第 3 课  drafter 是可插拔的：用 N-gram proposer 当草稿器（vLLM 同款语义）
  第 4 课  vLLM 批处理的扁平索引：复刻 SpecDecodeMetadata 的 _calc_spec_decode_metadata
  第 5 课  性能指标：mean acceptance length 与加速比

一句话核心思想（对应 DOC 第 2 章）：
    decode 阶段是访存密集(memory-bound)的串行瓶颈——解码 K 个 token 要串行调 K 次大模型。
    投机解码用"便宜的草稿模型先猜 γ 个 token，大模型一次前向并行验证"，
    把"多次窄前向"合并成"一次宽前向"，从而减少大模型被调用的次数。
    而拒绝采样(rejection sampling)保证：最终输出在统计上与"逐 token 从大模型采样"完全一致。

运行：
    python spec_decode_demo.py
"""

import random
from collections import Counter


# ============================================================
# 玩具"语言模型"：一个 bigram（只看上一个 token）的概率模型
#   - target model  M_p：我们要加速的大模型，代表"标准答案分布"
#   - draft  model  M_q：便宜的近似模型，分布接近 M_p 但不完全相同
# 用 bigram 是为了让"条件分布"可以直接算出来，便于把验证逻辑讲透。
# ============================================================

VOCAB = ["今天", "天气", "真", "好", "不错", "坏", "。", "<eos>"]

# 目标模型 M_p 的转移表：上一个 token -> {下一个 token: 概率}
TARGET_TABLE = {
    "今天": {"天气": 0.7, "真": 0.2, "不错": 0.1},
    "天气": {"真": 0.6, "不错": 0.3, "好": 0.1},
    "真":   {"好": 0.7, "不错": 0.2, "坏": 0.1},
    "好":   {"。": 0.8, "<eos>": 0.2},
    "不错": {"。": 0.6, "<eos>": 0.4},
    "坏":   {"。": 0.7, "<eos>": 0.3},
    "。":   {"<eos>": 1.0},
    "<eos>": {"<eos>": 1.0},
}


def target_dist(context):
    """目标模型 M_p：给定上下文，返回覆盖整个词表的概率分布(dict)。"""
    last = context[-1]
    row = TARGET_TABLE.get(last, {"<eos>": 1.0})
    return {t: row.get(t, 0.0) for t in VOCAB}


def draft_dist(context, alpha):
    """
    草稿模型 M_q：把目标分布和均匀分布混合得到。
        q = (1 - alpha) * p  +  alpha * uniform
    - alpha 越小 → q 越接近 p → 草稿越准 → 接受率越高（DOC 2.3 的核心结论）。
    - 混入 uniform 保证 q(t) > 0（拒绝采样里要做 p/q，分母不能为 0）。
    这只是为了教学方便地"造一个接近但不相同的近似模型"，
    真实场景里 M_q 是一个独立小模型 / Medusa 头 / Eagle drafter。
    """
    p = target_dist(context)
    u = 1.0 / len(VOCAB)
    return {t: (1 - alpha) * p[t] + alpha * u for t in VOCAB}


def sample(dist, rng):
    """从一个 {token: prob} 分布里采样一个 token。"""
    toks = list(dist.keys())
    weights = [dist[t] for t in toks]
    return rng.choices(toks, weights=weights, k=1)[0]


def recovered_dist(p, q):
    """
    拒绝发生时的"修正分布"（DOC 2.2 情况 A）：
        p' = normalize(max(p - q, 0))
    含义：从目标分布里"扣掉草稿已经高估的部分"，再归一化后重采样。
    数学上它恰好补偿了被拒绝带来的偏差，使最终分布严格等于 p（见第 2 课）。
    """
    raw = {t: max(p[t] - q[t], 0.0) for t in VOCAB}
    z = sum(raw.values())
    if z <= 0:  # 理论上 p、q 都是分布时不会发生，兜底回退到 p
        return dict(p)
    return {t: raw[t] / z for t in VOCAB}


# ============================================================
# 第 1 课：投机解码的一个解码步（核心！）
#   draft γ 个 token  →  大模型并行验证  →  逐个接受/拒绝  →  bonus / recovered
# ============================================================

def speculative_step(context, gamma, alpha, rng, verbose=False):
    """
    执行一次投机解码步，返回 (本步产出的 token 列表, 被接受的草稿 token 数)。

    对应 DOC 2.2 算法流程：
      1) Drafting：小模型自回归生成 γ 个草稿 token，并记录每步的 q 分布。
      2) Verification：大模型在 context+草稿 上"一次前向"，得到 γ+1 个位置的分布。
         （这里 bigram 模型直接算条件分布；真实 vLLM 是一次并行 forward 拿 logits。）
      3) Accept/Reject：从左到右用拒绝采样判定，首次拒绝即停。
         - 全部接受 → 末尾再免费采样 1 个 bonus token（位置 γ+1 的分布已经算好了）。
         - 第 i 个被拒 → 保留前 i 个，把第 i 个换成 recovered token，丢弃其后所有草稿。
    每步至少产出 1 个 token，最多产出 γ+1 个 token。
    """
    # ---- 1) Drafting：小模型连续生成 γ 个草稿 token ----
    draft_tokens, draft_dists = [], []
    ctx = list(context)
    for _ in range(gamma):
        q = draft_dist(tuple(ctx), alpha)
        tok = sample(q, rng)
        draft_tokens.append(tok)
        draft_dists.append(q)
        ctx.append(tok)
    if verbose:
        print(f"    [draft ] 草稿器提议 {gamma} 个 token: {draft_tokens}")

    # ---- 2) Verification：大模型并行算出 γ+1 个位置的目标分布 ----
    #     位置 i 的分布 = 在 context + 前 i 个草稿 上的 M_p 条件分布
    target_dists = []
    ctx = list(context)
    for i in range(gamma):
        target_dists.append(target_dist(tuple(ctx)))
        ctx.append(draft_tokens[i])
    target_dists.append(target_dist(tuple(ctx)))  # 第 γ+1 个位置（bonus 用）

    # ---- 3) Accept / Reject：逐个验证，首次拒绝即停 ----
    output = []
    for i in range(gamma):
        y = draft_tokens[i]
        p = target_dists[i][y]
        q = draft_dists[i][y]
        accept_prob = min(1.0, p / q)         # 拒绝采样核心：min(1, p/q)
        if rng.random() < accept_prob:
            output.append(y)                  # 接受这个草稿 token
            if verbose:
                print(f"    [verify] 位置{i} '{y}': p={p:.2f} q={q:.2f} "
                      f"接受率={accept_prob:.2f} -> ✔ 接受")
        else:
            rec = sample(recovered_dist(target_dists[i], draft_dists[i]), rng)
            output.append(rec)                # 拒绝：用 recovered token 顶替
            if verbose:
                print(f"    [verify] 位置{i} '{y}': p={p:.2f} q={q:.2f} "
                      f"接受率={accept_prob:.2f} -> ✗ 拒绝, 改采 '{rec}'(后续草稿丢弃)")
            return output, i                  # 首次拒绝即停，已接受 i 个草稿

    # ---- 全部草稿被接受 → 免费多采一个 bonus token ----
    bonus = sample(target_dists[gamma], rng)
    output.append(bonus)
    if verbose:
        print(f"    [bonus ] {gamma} 个草稿全部通过 -> 免费 bonus token '{bonus}'")
    return output, gamma


def generate_speculative(start, gamma, alpha, rng, max_tokens=40):
    """用投机解码自回归生成一段序列，统计调用次数等指标。"""
    context = list(start)
    produced = []
    num_steps = 0           # 大模型前向次数（每步只调一次"宽前向"）
    accepted_draft = 0      # 累计被接受的草稿 token 数
    while len(produced) < max_tokens:
        toks, n_acc = speculative_step(tuple(context), gamma, alpha, rng)
        num_steps += 1
        accepted_draft += n_acc
        done = False
        for t in toks:
            produced.append(t)
            context.append(t)
            if t == "<eos>":
                done = True
                break
        if done:
            break
    return produced, num_steps, accepted_draft


def generate_autoregressive(start, rng, max_tokens=40):
    """标准自回归：每个 token 都要调一次大模型，作为对照基线。"""
    context = list(start)
    produced = []
    num_calls = 0
    while len(produced) < max_tokens:
        tok = sample(target_dist(tuple(context)), rng)
        num_calls += 1
        produced.append(tok)
        context.append(tok)
        if tok == "<eos>":
            break
    return produced, num_calls


# ============================================================
# 第 3 课：N-gram proposer —— 一种"零权重"草稿器（vLLM 同款语义）
#   不需要任何模型，靠"当前后缀在历史里出现过"来复用后续 token。
# ============================================================

def ngram_propose(context_ids, min_n, max_n, k):
    """
    在 context_ids 中，找与"末尾后缀"匹配的最长 n-gram（长度 ∈ [min_n, max_n]），
    命中后返回该历史匹配位置之后的最多 k 个 token 作为草稿。找不到返回 []。
    对应 DOC 6.2 / 7.2 的 NgramProposer。
    """
    n = len(context_ids)
    upper = min(max_n, n - 1)
    for ng in range(upper, min_n - 1, -1):          # 优先匹配更长的后缀
        suffix = context_ids[n - ng:]
        for start in range(n - ng - 1, -1, -1):     # 从最近的历史往前找
            if context_ids[start:start + ng] == suffix:
                draft = context_ids[start + ng:start + ng + k]
                if draft:
                    return draft
    return []


# ============================================================
# 第 4 课：复刻 vLLM 的 _calc_spec_decode_metadata
#   把"每个请求草稿数不同"的 batch，压成一套扁平索引，
#   让 GPU 一次前向 + 一次 gather 就能完成验证 + bonus 采样。
#   完全对应 DOC 8.3 的例子。
# ============================================================

def calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens):
    """
    输入:
      num_draft_tokens          每个请求的草稿 token 数,        如 [3, 0, 2, 0, 1]
      cu_num_scheduled_tokens   每个请求在扁平 batch 中的累计终点, 如 [4,104,107,207,209]
    输出: dict, 含 logits_indices / target_logits_indices / bonus_logits_indices
    """
    nreq = len(num_draft_tokens)
    # 每个请求要采样的位置数 = 草稿数 + 1（那个 +1 是 bonus / 修正位）
    num_sampled = [d + 1 for d in num_draft_tokens]

    # 前缀和：cu_num_sampled[i] = sum(num_sampled[:i+1])
    cu_num_sampled, acc = [], 0
    for s in num_sampled:
        acc += s
        cu_num_sampled.append(acc)

    # (a) logits_indices —— 在"扁平 hidden_states 空间"里要算 logits 的位置
    #     每个请求的起点 = 该请求终点 - 它的采样段长度，然后展开该段
    logits_indices = []
    for i in range(nreq):
        start = cu_num_scheduled_tokens[i] - num_sampled[i]
        logits_indices += list(range(start, start + num_sampled[i]))

    # (b) bonus_logits_indices —— 在"压缩后的 logits 空间"里，每段的最后一行
    bonus_logits_indices = [c - 1 for c in cu_num_sampled]

    # (c) target_logits_indices —— 压缩空间里，每段去掉 bonus 位、只留草稿验证位
    target_logits_indices = []
    for i in range(nreq):
        seg_start = cu_num_sampled[i] - num_sampled[i]   # 该请求在压缩空间的起点
        target_logits_indices += list(range(seg_start, seg_start + num_draft_tokens[i]))

    return {
        "num_sampled": num_sampled,
        "cu_num_sampled": cu_num_sampled,
        "logits_indices": logits_indices,
        "target_logits_indices": target_logits_indices,
        "bonus_logits_indices": bonus_logits_indices,
    }


# ============================================================
# 主流程：依次跑 6 节课
# ============================================================

def main():
    rng = random.Random(2026)
    sep = "=" * 72

    # ---------- 第 0 课：动机 ----------
    print(sep)
    print("第 0 课：为什么 decode 慢，投机解码省的是什么")
    print(sep)
    print("""  - decode 阶段每步只生成 1 个 token，主干矩阵乘退化为"矩阵×向量"(GEMV)，
    瓶颈是反复读写权重(访存密集 memory-bound)，不是算力。
  - 解码 K 个 token = 串行调用大模型 K 次。
  - 投机解码：小模型先猜 γ 个，大模型"一次宽前向"并行验证多个 token，
    把"多次窄前向"合并成"一次宽前向" -> 减少大模型调用次数。
  - 关键：一次宽前向(验证 γ+1 个位置)和一次窄前向(1 个位置)耗时相近，
    因为瓶颈在搬权重，不在算这点 token。""")

    # ---------- 第 1 课：单步细节 ----------
    print("\n" + sep)
    print("第 1 课：投机解码的一个解码步（draft -> verify -> accept/reject -> bonus）")
    print(sep)
    print("  上下文 = ['今天']，γ=3，draft 质量 alpha=0.3\n")
    speculative_step(("今天",), gamma=3, alpha=0.3, rng=rng, verbose=True)

    # ---------- 第 2 课：正确性（蒙特卡洛） ----------
    print("\n" + sep)
    print("第 2 课：实验证明 —— 投机解码的输出分布 == 目标模型分布")
    print(sep)
    print("  固定上下文 ['天气']，重复 N 次取每步'第一个产出 token'，统计经验分布。")
    print("  理论保证：它应等于 target M_p，而不是 draft M_q。\n")
    ctx = ("天气",)
    N = 200_000
    alpha_test = 0.5  # 故意让草稿明显偏离，凸显"拒绝采样在做修正"
    cnt = Counter()
    for _ in range(N):
        toks, _ = speculative_step(ctx, gamma=3, alpha=alpha_test, rng=rng)
        cnt[toks[0]] += 1
    emp = {t: cnt[t] / N for t in VOCAB}
    tgt = target_dist(ctx)
    dft = draft_dist(ctx, alpha_test)
    print(f"  {'token':<8}{'目标p(M_p)':>12}{'草稿q(M_q)':>12}{'投机经验分布':>14}")
    for t in VOCAB:
        if tgt[t] > 0 or emp[t] > 1e-4:
            print(f"  {t:<8}{tgt[t]:>12.3f}{dft[t]:>12.3f}{emp[t]:>14.3f}")
    max_err = max(abs(emp[t] - tgt[t]) for t in VOCAB)
    print(f"\n  与目标分布最大偏差 = {max_err:.4f}（蒙特卡洛噪声级别）"
          f" -> 经验分布贴合 M_p 而非 M_q，✔ 无偏。")

    # ---------- 第 3 课：N-gram 草稿器 ----------
    print("\n" + sep)
    print("第 3 课：drafter 可插拔 —— N-gram proposer（零权重草稿器）")
    print(sep)
    history = [10, 20, 30, 40, 50, 70, 30, 40]
    draft = ngram_propose(history, min_n=2, max_n=3, k=2)
    print(f"  历史 token: {history}")
    print(f"  末尾后缀 [30,40] 曾出现在 index 2-3，其后是 [50,70]")
    print(f"  N-gram 草稿(min_n=2,max_n=3,k=2) = {draft}   (期望 [50, 70])")
    print("  说明：drafter 怎么造草稿可以换(独立小模型/Medusa 头/Eagle/N-gram)，")
    print("        但'大模型并行验证 + 拒绝采样'这套流程是通用、不变的。")

    # ---------- 第 4 课：SpecDecodeMetadata 扁平索引 ----------
    print("\n" + sep)
    print("第 4 课：vLLM 批处理扁平索引（复刻 _calc_spec_decode_metadata，DOC 8.3）")
    print(sep)
    num_draft = [3, 0, 2, 0, 1]
    cu_sched = [4, 104, 107, 207, 209]
    md = calc_spec_decode_metadata(num_draft, cu_sched)
    print(f"  num_draft_tokens        = {num_draft}")
    print(f"  cu_num_scheduled_tokens = {cu_sched}")
    print(f"  -> num_sampled(=draft+1)= {md['num_sampled']}")
    print(f"  -> logits_indices       = {md['logits_indices']}")
    print(f"  -> target_logits_indices= {md['target_logits_indices']}  (验证草稿用)")
    print(f"  -> bonus_logits_indices = {md['bonus_logits_indices']}  (全接受时采 bonus)")
    # 与 DOC 中给出的期望值对拍
    assert md["logits_indices"] == [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
    assert md["target_logits_indices"] == [0, 1, 2, 5, 6, 9]
    assert md["bonus_logits_indices"] == [3, 4, 7, 8, 10]
    print("  ✔ 与 DOC 8.3 给出的期望索引完全一致。")
    print("  意义：不同请求草稿数不同(2/0/3...)，扁平化后一次前向 + 一次 gather 全搞定，")
    print("        避免逐请求单独跑或大量 padding。")

    # ---------- 第 5 课：性能指标 ----------
    print("\n" + sep)
    print("第 5 课：性能指标 —— mean acceptance length 与加速比")
    print(sep)
    print("  从 ['今天'] 生成多段，统计'大模型前向次数'。draft 越准(alpha 越小)，加速越大。\n")
    print(f"  {'alpha(草稿误差)':<16}{'产出tokens':>12}{'大模型前向':>12}"
          f"{'平均接受长度':>14}{'加速比':>10}")
    for alpha in (0.1, 0.3, 0.6, 0.9):
        rng2 = random.Random(7)
        tot_tokens = tot_steps = tot_acc = 0
        for _ in range(2000):                      # 跑多段取平均
            prod, steps, acc = generate_speculative(("今天",), gamma=3,
                                                    alpha=alpha, rng=rng2)
            tot_tokens += len(prod)
            tot_steps += steps
            tot_acc += acc
        # vLLM 口径：mean acceptance length = 1 + 累计接受草稿数 / 草稿轮数
        mean_al = 1 + tot_acc / tot_steps
        speedup = tot_tokens / tot_steps           # 标准AR需 tot_tokens 次, 投机只需 tot_steps 次
        print(f"  {alpha:<16}{tot_tokens:>12}{tot_steps:>12}"
              f"{mean_al:>14.2f}{speedup:>9.2f}x")
    print("\n  结论(对应 DOC 2.3)：")
    print("   - 草稿越接近大模型(alpha→0)，接受率越高，每轮推进越多 token，加速比越大；")
    print("   - 最坏情况(草稿很差)退化为接近普通自回归，每轮仍≥1 token，不会更慢；")
    print("   - 上界 ≈ γ+1 倍（全部接受 + 1 个 bonus）。")

    print("\n" + sep)
    print("小结：")
    print("  1) 动机：decode 是访存密集的串行瓶颈，要减少大模型串行调用次数。")
    print("  2) 框架：draft(便宜地猜 γ 个) -> verify(大模型一次宽前向并行验证) ->")
    print("           accept/reject(拒绝采样, 首次拒绝即停) -> 全接受再 +1 bonus。")
    print("  3) 正确性：min(1,p/q) 接受 + recovered=normalize(max(p-q,0)) 重采,")
    print("           保证输出分布严格 == 从大模型逐 token 采样。")
    print("  4) 变体：N-gram / Medusa(多头) / Eagle(预测hidden state) 只改'怎么造草稿',")
    print("           验证流程通用不变。")
    print("  5) 工程：SpecDecodeMetadata 把变长草稿压成扁平索引, GPU 一次前向批量验证。")
    print(sep)


if __name__ == "__main__":
    main()
