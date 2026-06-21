"""
batching_spec_demo.py
=====================
教学示例：投机解码与 Continuous Batching 共存的工程坑（对应 Q9，题眼）。

spec_decode_demo 证明了"草稿越准加速越大"，但那是 batch=1 的故事。真实 serving 用
continuous batching：几百个请求拼一个大 batch。这时投机解码的收益会随 batch 增大而塌，
大 batch 下甚至可能【变慢】。本示例用一个 roofline(屋顶线) 性能模型把这件事算清楚，
并讲清两个工程细节：(1) 引擎要不要动态开关投机解码；(2) 被拒草稿写进 KV cache 怎么回滚。

  第 0 课  免费午餐的前提：decode 在 memory-bound 区间，前向时间与 token 数几乎无关
  第 1 课  roofline 模型：一次前向的耗时 = max(搬权重时间, 算 FLOP 时间)
  第 2 课  扫描 batch：投机解码加速比如何从 >1 塌到 <1
  第 3 课  什么场景该开投机解码 / 引擎要不要动态开关
  第 4 课  被拒投机 token 的 KV cache 回滚，与 PagedAttention 的配合

运行：
    python batching_spec_demo.py
"""

# ============================================================
# 一个极简但抓住本质的 roofline 性能模型
#   一次 decode 前向要处理 N 个 token 位置（continuous batching 把所有请求的
#   待算 token 拼在一起）。两个开销：
#     - 访存：把模型权重从 HBM 读进来，时间 ~ 常数(与 N 无关)。decode 主因。
#     - 计算：FLOP ~ 正比于 N。N 小时算力闲置，N 大时算力打满。
#   一次前向耗时 ≈ max(weight_read_time, N * flop_time_per_token)。
#   memory-bound（N 小）：耗时≈常数 → 多塞 token 几乎免费 ← 投机解码的"免费午餐"
#   compute-bound（N 大）：耗时∝N    → 多塞 token 线性变贵 ← 免费午餐消失
# ============================================================

WEIGHT_READ_TIME = 100.0      # 读一遍权重的固定耗时(任意单位)，代表 memory-bound 地板
FLOP_TIME_PER_TOKEN = 1.0     # 每多算一个 token 位置的计算耗时


def forward_time(num_tokens):
    """一次前向处理 num_tokens 个 token 位置的耗时（roofline）。"""
    return max(WEIGHT_READ_TIME, num_tokens * FLOP_TIME_PER_TOKEN)


def regime(num_tokens):
    return "memory-bound(算力闲置)" if num_tokens * FLOP_TIME_PER_TOKEN <= WEIGHT_READ_TIME \
        else "compute-bound(算力打满)"


# ============================================================
# 对比：标准自回归 vs 投机解码，在 batch=B 下解码一段序列
# ============================================================

def throughput_baseline(batch):
    """标准自回归：每步每个请求产出 1 个 token，一次前向处理 B 个 token 位置。
    返回 (每步前向耗时, 每步产出token数, 单位token耗时)。"""
    N = batch * 1
    t = forward_time(N)
    produced = batch * 1
    return t, produced, t / produced


def throughput_spec(batch, gamma, accept_len):
    """投机解码：每步每个请求提议 gamma 个草稿，一次前向验证 gamma+1 个位置 ×B。
      accept_len = mean acceptance length ∈ [1, gamma+1]：每步每请求实际产出的 token 数。
    返回 (每步前向耗时, 每步产出token数, 单位token耗时)。"""
    N = batch * (gamma + 1)                 # 验证要把 gamma+1 个位置都喂进去
    t = forward_time(N)
    produced = batch * accept_len           # 实际被接受(+bonus)的才算产出
    return t, produced, t / produced


def lesson0():
    print("=" * 72)
    print("第 0 课：免费午餐的前提 —— decode 在 memory-bound 区间")
    print("=" * 72)
    print(f"""  roofline：一次前向耗时 ≈ max(搬权重={WEIGHT_READ_TIME}, token数×{FLOP_TIME_PER_TOKEN})
  decode 每步每请求只算 1 个 token，N 很小 → 卡在"搬权重"那条地板上，
  算力大量闲置。投机解码正是把这块【闲置算力】拿来并行验证 γ+1 个 token：\n""")
    for N in (1, 10, 50, 100, 150, 300):
        print(f"    N={N:>4} token位置 -> 前向耗时={forward_time(N):>6.0f}  [{regime(N)}]")
    print("""
  看 N≤100：耗时恒为 100(被权重读取卡住)，多算的 token 完全免费。
  看 N≥150：耗时开始随 N 线性涨 —— 这块就不再免费了。投机解码的命运全看 batch 把
  总 token 数推到了哪个区间。""")


def lesson1():
    print("\n" + "=" * 72)
    print("第 1 课：单请求(batch=1) —— 免费午餐最香")
    print("=" * 72)
    gamma, accept = 3, 2.5
    tb, pb, ub = throughput_baseline(1)
    ts, ps, us = throughput_spec(1, gamma, accept)
    print(f"  标准AR  : 前向处理 N={1} -> 耗时{tb:.0f}, 产出{pb}token, {ub:.1f}/token")
    print(f"  投机(γ=3): 前向处理 N={1*(gamma+1)} -> 耗时{ts:.0f}, 产出{ps}token, {us:.1f}/token")
    print(f"  加速比 = {ub/us:.2f}x  （N={1*(gamma+1)} 仍在 memory-bound 区，验证几乎白送）")
    print("  → batch=1 时验证 4 个 token 和验证 1 个 token 一样快，接受多少就赚多少。")


def lesson2():
    print("\n" + "=" * 72)
    print("第 2 课：扫描 batch —— 加速比如何从 >1 塌到 <1")
    print("=" * 72)
    gamma, accept = 3, 2.5     # 草稿质量固定：每步平均接受 2.5 个 token
    print(f"  固定 γ={gamma}, 平均接受长度={accept}。看加速比随 batch 变化：\n")
    print(f"  {'batch':>6}{'AR-N':>7}{'spec-N':>8}{'AR耗时':>8}{'spec耗时':>9}"
          f"{'加速比':>8}   区间")
    for B in (1, 4, 16, 25, 33, 50, 100):
        tb, pb, ub = throughput_baseline(B)
        ts, ps, us = throughput_spec(B, gamma, accept)
        spec_N = B * (gamma + 1)
        flag = "" if ub / us >= 1.0 else "  <-- 变慢!"
        print(f"  {B:>6}{B:>7}{spec_N:>8}{tb:>8.0f}{ts:>9.0f}"
              f"{ub/us:>7.2f}x   {regime(spec_N).split('(')[0]}{flag}")
    print(f"""
  读这张表：
    - 小 batch：spec-N 还在 memory-bound，spec 耗时 == AR 耗时(都是地板 100)，
      但 spec 一次产出 {accept} token，所以加速比 ≈ 接受长度 {accept}x。这是赚的。
    - batch 变大：spec-N = B×{gamma+1} 先冲进 compute-bound，spec 耗时开始∝N 上涨，
      而此时 AR-N=B 可能还在 memory-bound(便宜)。于是 spec 多花的算力换不回足够 token。
    - 大 batch：两者都 compute-bound 时，spec 处理 {gamma+1}×token 却只多产出 {accept}×，
      单位 token 反而更贵 -> 加速比跌破 1，投机解码【净变慢】。
  根因：大 batch 本身已经把算力喂饱了(没有闲置算力可白嫖)，投机解码的免费午餐就没了，
  反而要为"验证了但被拒"的草稿 token 付出真金白银的 FLOP。""")


def lesson3():
    print("\n" + "=" * 72)
    print("第 3 课：什么场景该开投机解码 / 引擎要不要动态开关")
    print("=" * 72)
    print("""  该开（收益大）：
    - 低并发 / 小 batch：交互式聊天、本地单用户、低 QPS 服务 —— 算力大量闲置。
    - 接受率高的负载：代码/结构化文本(可预测)、配 EAGLE 这类高接受率 drafter。
    - 延迟敏感(TPOT/首字后吐字速度)优先于吞吐 的场景。

  别开（可能亏）：
    - 高并发 / 大 batch 的吞吐优先服务：算力已打满，验证开销纯亏。
    - 接受率很低的负载：草稿老被拒，白算一堆 FLOP。

  所以现代引擎倾向【动态开关 / 自适应】：
    - 按当前 batch 大小或 GPU 利用率决定这一步开不开投机(小 batch 开、大 batch 关)。
    - 在线测 accept rate，太低就调小 γ 甚至关掉(SpecDecoding 的 'disable by batch size'：
      vLLM 有 speculative_disable_by_batch_size 之类阈值，batch 超阈值就停用)。
    - γ 也可以自适应：接受率高就多猜几个，低就少猜。""")


def lesson4():
    print("\n" + "=" * 72)
    print("第 4 课：被拒投机 token 的 KV cache 回滚，与 PagedAttention 配合")
    print("=" * 72)
    print("""  问题：验证时 target 对 context + [d1,d2,d3] 做了一次前向，这一步会为 d1,d2,d3
  这几个【草稿位置】都算出 K/V 并写进 KV cache。但若 d2 被拒，d2(及其后)其实不该留下。
  怎么"回滚"这些已经写进去的 KV？\n""")

    # 用 PagedAttention 的 block 表模拟回滚
    BLOCK = 4
    print(f"  PagedAttention：KV 按 block 存(本例每 block={BLOCK} 个槽)。一个请求维护：")
    print("    - block_table：逻辑块 -> 物理块映射")
    print("    - num_computed_tokens：已落实的有效 KV 长度(序列逻辑长度)\n")

    seq_len = 5                                   # 已落实 5 个 token
    draft = ["d1", "d2", "d3"]
    print(f"  起点：seq_len(num_computed_tokens) = {seq_len}")
    print(f"  本步验证写入 3 个草稿 token 的 KV：{draft}")
    # 前向把草稿位置的 KV 都写进 slot 5,6,7（可能跨 block）
    written_slots = [seq_len + i for i in range(len(draft))]
    print(f"    -> 它们的 KV 临时写到 slot {written_slots} "
          f"(逻辑块 {[s // BLOCK for s in written_slots]})")

    accepted = 1                                  # 假设只接受 d1，d2/d3 被拒
    print(f"\n  验证结果：接受 {accepted} 个草稿(d1) + 1 个修正/bonus token，d2、d3 被拒。")
    new_len = seq_len + accepted + 1              # +1 是 bonus / recovered token
    print(f"  回滚做法：根本不用搬数据/清零，只把 num_computed_tokens 改成 {new_len}：")
    print(f"    seq_len: {seq_len} -> {new_len}")
    print(f"    d2、d3 写在 slot {written_slots[1:]} 的'脏 KV'被直接忽略，")
    print("    下一步新 token 会覆盖这些 slot。block 没被 commit 的部分等于不存在。")
    print(f"""
  和 PagedAttention 的配合要点：
    - 回滚 = 把逻辑序列长度(num_computed_tokens)截断到"已接受长度"，O(1) 指针操作，
      不需要逐元素清 KV —— 脏数据下轮被覆盖，且 attention 只读到 num_computed_tokens 以内。
    - 物理块不立刻释放：刚写了一半的尾块留着，下一步接着往里写，避免频繁分配/回收。
    - 接受长度可变(0~γ+1) → 每个请求下一轮的起始 KV 长度不同，这正是 continuous batching
      要处理的变长，靠 block_table + cu_num_tokens 扁平索引(见 spec_decode_demo 第4课)统一管理。""")

    assert new_len == 7
    print("\n  ✔ 回滚后 seq_len=7（5 旧 + 1 接受 d1 + 1 bonus），d2/d3 的脏 KV 被无害忽略。")


def main():
    lesson0()
    lesson1()
    lesson2()
    lesson3()
    lesson4()
    print("\n" + "=" * 72)
    print("小结（Q9）：")
    print("  · 投机解码的免费午餐来自 decode 的【闲置算力】；大 batch 把算力喂饱后午餐消失。")
    print("  · 加速比随 batch 增大而塌，大 batch + 低接受率可能净变慢(多算被拒草稿的 FLOP)。")
    print("  · 该开：低并发/小 batch/延迟敏感/高接受率；该关：高并发吞吐优先/低接受率。")
    print("  · 引擎应动态开关：按 batch 大小/GPU 利用率/在线 accept rate 自适应开关与调 γ。")
    print("  · KV 回滚：把 num_computed_tokens 截到已接受长度即可，O(1)，脏 KV 下轮覆盖；")
    print("    与 PagedAttention 的 block 管理天然契合(尾块不急于释放、变长靠扁平索引)。")
    print("=" * 72)


if __name__ == "__main__":
    main()
