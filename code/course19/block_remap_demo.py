#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
course19 · Demo 3 —— 远端 Block 和本地 Block 是怎么对上号的（分页 KV 的搬运）
==============================================================================

对应 DOC.md：第三章 3.4「远端的 Block 和本地的 Block 是怎么对应的？」
            以及 inject_kv_into_layer / extract_kv_from_layer 两个真实源码函数

上一课 (kv_handoff_demo) 把 KV 当成一个整体搬过去。但真实 vLLM 的 KV cache 是
**分页（PagedAttention）** 的：每个请求的 KV 被切成固定大小的「块（block）」，
散落在一个全局 KV pool 的不同物理槽位里。

于是出现一个关键问题：
    P 节点给这个请求分配的物理块号是 [2, 7, 5]，
    D 节点给同一个请求分配的物理块号可能是 [5, 9, 11]，
    物理块号根本对不上，KV 怎么搬才不会错位？

答案（也是本 demo 要演示的）：
    ★ 搬运按 **逻辑顺序（logical order）** 进行，与物理块号无关。
      P 侧 extract：按自己的 block_ids **聚合（gather）** 成一段连续 KV，按 token 逻辑序排好；
      D 侧 inject ：把这段连续 KV 按自己的 block_ids **分散（scatter）** 写回本地 pool。
      两边的「第 i 个逻辑块」对应同一段 token，物理槽位各管各的。

纯标准库运行：
    python block_remap_demo.py
"""

BLOCK_SIZE = 4     # 每个块装 4 个 token（真实 vLLM 默认 16，这里改小便于观察）
POOL_SLOTS = 16    # KV pool 一共有 16 个物理块槽位


# --------------------------------------------------------------------------
# 1. 一个极简的「分页 KV pool」：模拟 GPU 上的 paged KV cache
#    pool[slot] 是一个物理块，能装 BLOCK_SIZE 个 token 的 KV（这里用 token 标记代替真实张量）
# --------------------------------------------------------------------------
class PagedKVPool:
    def __init__(self, name, slots=POOL_SLOTS):
        self.name = name
        # 每个槽位是一个长度 BLOCK_SIZE 的列表，None 表示空位
        self.pool = [[None] * BLOCK_SIZE for _ in range(slots)]
        self.free = list(range(slots))

    def allocate(self, num_blocks, prefer=None):
        """分配 num_blocks 个物理块，返回它们的 block_ids（物理槽位号）。
        prefer 用来人为制造「P 和 D 分到不同物理块号」的效果，纯为教学演示。"""
        ids = []
        if prefer:
            for p in prefer:
                if p in self.free:
                    self.free.remove(p)
                    ids.append(p)
        while len(ids) < num_blocks:
            ids.append(self.free.pop(0))
        return ids

    def write_block(self, slot, kv_block):
        self.pool[slot] = list(kv_block)

    def read_block(self, slot):
        return self.pool[slot]


# --------------------------------------------------------------------------
# 2. P 侧：extract_kv_from_layer —— 按自己的 block_ids 把 KV「聚合」成连续序列
#    真实源码（FlashAttention 分支）：  return layer[:, block_ids, ...]
#    本质就是：按 block_ids 的顺序，把这些物理块依次取出来拼接。
# --------------------------------------------------------------------------
def extract_kv(pool, block_ids):
    """gather：按逻辑顺序（block_ids 的排列顺序）取出每个块，拼成一段连续 KV。"""
    contiguous = []
    for slot in block_ids:                 # 注意：遍历顺序 = 逻辑顺序，不是物理槽位大小顺序
        contiguous.append(pool.read_block(slot))
    return contiguous                      # contiguous[i] 是「第 i 个逻辑块」


# --------------------------------------------------------------------------
# 3. D 侧：inject_kv_into_layer —— 把收到的连续 KV 按自己的 block_ids「分散」写回
#    真实源码（FlashAttention 分支）：  layer[:, block_ids, ...] = kv_cache
#    本质就是：第 i 个逻辑块，写到 D 自己分配的第 i 个物理块里。
# --------------------------------------------------------------------------
def inject_kv(pool, block_ids, contiguous):
    """scatter：第 i 个逻辑块 → 写入 D 自己的第 i 个物理块。"""
    for i, slot in enumerate(block_ids):
        if i < len(contiguous):            # 部分注入分支：chunked prefill 时收到的块可能更少
            pool.write_block(slot, contiguous[i])


def visualize(pool, block_ids, title):
    print(f"  {title}")
    print(f"    逻辑块顺序(本请求): {block_ids}")
    for logical_i, slot in enumerate(block_ids):
        toks = [t for t in pool.read_block(slot) if t is not None]
        print(f"      逻辑块#{logical_i}  → 物理槽位[{slot:2d}]  内容(token): {toks}")


# ==========================================================================
# 4. 端到端：P 把一个 47-token 请求的 KV 搬给 D，物理块号故意错开
# ==========================================================================
def main():
    print(__doc__)

    # 一个有 11 个 prompt token 的请求（每块 4 个 token → ceil(11/4)=3 块）
    prompt_tokens = [f"t{i}" for i in range(11)]
    num_blocks = (len(prompt_tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("=" * 70)
    print(f"请求 prompt 有 {len(prompt_tokens)} 个 token，block_size={BLOCK_SIZE} "
          f"→ 需要 {num_blocks} 个 KV 块")
    print("=" * 70)

    # ---- P 节点：prefill 后，KV 落在 P 自己分配的物理块上 ----
    p_pool = PagedKVPool("P-pool")
    p_blocks = p_pool.allocate(num_blocks, prefer=[2, 7, 5])   # P 拿到物理块 [2,7,5]
    for i, slot in enumerate(p_blocks):
        chunk = prompt_tokens[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        chunk += [None] * (BLOCK_SIZE - len(chunk))
        p_pool.write_block(slot, chunk)

    print("\n[P 节点] prefill 完成，KV 分布在 P 自己的物理槽位上：")
    visualize(p_pool, p_blocks, "P 的物理布局")

    # ---- 传输：extract（P）→ 网络上的一段连续 KV ----
    payload = extract_kv(p_pool, p_blocks)
    print("\n[传输] P 调用 extract_kv 按逻辑顺序聚合，得到一段【连续 KV】(与物理槽位无关)：")
    for i, blk in enumerate(payload):
        print(f"      连续KV[{i}] = {[t for t in blk if t is not None]}")
    print("      ↑ 这段连续 KV 通过 NCCL 发给 D。注意：里面【没有】任何物理块号信息，")
    print("        只有 token 的逻辑先后顺序。物理块号是各节点的私事。")

    # ---- D 节点：为同一请求分配【不同的】物理块号，再 inject ----
    d_pool = PagedKVPool("D-pool")
    # 先占掉一些槽位，制造 D 的 block_ids 与 P 完全不同的效果
    d_pool.allocate(5)                                  # 占位，模拟 D 上已有别的请求
    d_blocks = d_pool.allocate(num_blocks, prefer=[5, 9, 11])  # D 拿到物理块 [5,9,11]
    print(f"\n[D 节点] 为同一请求分配的物理块号 = {d_blocks}（与 P 的 {p_blocks} 完全不同！）")
    inject_kv(d_pool, d_blocks, payload)
    print("[D 节点] 调用 inject_kv 把连续 KV 按【自己的】block_ids 分散写回本地 pool：")
    visualize(d_pool, d_blocks, "D 的物理布局")

    # ---- 验证：D 按逻辑顺序读出来的 token 序列，与 P 的完全一致 ----
    def logical_tokens(pool, blocks):
        seq = []
        for slot in blocks:
            seq += [t for t in pool.read_block(slot) if t is not None]
        return seq

    p_seq = logical_tokens(p_pool, p_blocks)
    d_seq = logical_tokens(d_pool, d_blocks)

    print("\n" + "=" * 70)
    print(f"P 逻辑序列: {p_seq}")
    print(f"D 逻辑序列: {d_seq}")
    if p_seq == d_seq:
        print("""✅ 一致！物理块号不同，但逻辑 token 序列完全对齐。

关键结论：
  · KV 搬运的「对应关系」建立在 **逻辑块顺序** 上，而不是物理块号。
  · P 用 extract（gather）把散落的块按逻辑序聚合成连续 KV；
    D 用 inject（scatter）把连续 KV 按自己的物理块号分散写回。
  · 这就是真实 vLLM 里 extract_kv_from_layer / inject_kv_into_layer 干的事，
    也是「远端 block 和本地 block 怎么对上号」的答案：靠逻辑序，不靠物理号。""")
    else:
        print("❌ 不一致，demo 有 bug")

    # ---- 附：get_num_new_matched_tokens 的「-1」是怎么回事 ----
    print("=" * 70)
    print("附加知识点：D 侧 get_num_new_matched_tokens 为什么返回 len(prompt) - 1？")
    num_prompt = len(prompt_tokens)
    num_external = num_prompt - 1                # 真实源码：len(prompt_token_ids) - 1 - num_computed
    print(f"""
  prompt 共 {num_prompt} 个 token。D 从外部（P）能直接拿到 KV 的 token 数 = {num_prompt} - 1 = {num_external}。
  为什么要减 1？因为最后 1 个 prompt token 需要在 D 上「重算一次」前向，
  才能得到用于预测【第一个生成 token】的隐藏态（上一课 DecodeWorker 里就是这么做的）。
  → D 跳过了 {num_external}/{num_prompt} 的 prefill 计算，只补算最后 1 个 token，几乎零浪费。""")

    print("\n下一课 → proxy_xpyd_demo.py：Proxy 怎么编排 P/D，xPyD 怎么路由。\n")


if __name__ == "__main__":
    main()
