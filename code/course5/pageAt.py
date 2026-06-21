"""
pageAt.py — vLLM PagedAttention 核心机制教学 Demo

覆盖 DOC.md 中的六大核心设计：
  1. 物理块池预分配 (BlockPool)          — 启动时一次划分，消除运行时碎片
  2. KVCacheBlock 元数据                 — block_id / ref_cnt / block_hash
  3. 链式 Hash（只对满块）               — hash(parent_hash, token_ids)，断一处则全链断
  4. Prefix Cache 匹配                   — 逐块检查 hash，命中则 touch()
  5. 懒惰驱逐                            — ref_cnt=0 时保留 hash 映射，窗口期内仍可复用
  6. 引用计数 + LRU 逆序释放             — 前缀块排队尾，存活最久

运行方式：python3 pageAt.py  （无需 GPU / MPI）
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════
# 0. 全局参数（对应 vLLM 中 KVCacheSpec 的字段）
# ══════════════════════════════════════════════════════════════════
BLOCK_SIZE = 4    # 每块存 4 个 token（vLLM 默认 16，教学用小值）
NUM_BLOCKS = 12   # 物理块总数（vLLM 由 GPU profiling 动态确定）
NONE_HASH  = 0    # 首块无父 hash 时使用的种子（vLLM 中为随机值）


def _sha(*args) -> int:
    """SHA256 哈希，返回前 8 字节的整数，用于模拟 vLLM 的 hash_fn。"""
    raw = str(args).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _sep(title: str):
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print('═' * 62)


# ══════════════════════════════════════════════════════════════════
# 1. KVCacheBlock — 物理块元数据
#    vLLM 对应：vllm/v1/core/kv_cache_utils.py :: KVCacheBlock
# ══════════════════════════════════════════════════════════════════
@dataclass
class KVCacheBlock:
    """
    代表一个物理 KV Cache 块的元数据，不含实际 GPU 显存。
    真实显存是一个预分配的大 Tensor；block_id 是这个 Tensor 的物理偏移索引。
    BlockPool 只管理这些"元数据"，不直接操作 GPU 显存。
    """
    block_id:   int
    ref_cnt:    int = 0                # 引用计数：被多少请求共享
    block_hash: Optional[int] = None  # 填满后计算的链式 hash（未满则为 None）
    token_ids:  tuple = ()            # 教学用：记录块内 token（真实存在 GPU Tensor 里）

    def reset(self):
        """被重新分配给新请求前，清除旧的 hash 元数据。GPU 数据在写入时自然覆盖。"""
        self.block_hash = None
        self.token_ids  = ()

    @property
    def h(self) -> str:
        return f"{self.block_hash & 0xFFFF:04X}" if self.block_hash else "----"

    def __repr__(self):
        return f"B{self.block_id:02d}(ref={self.ref_cnt}, hash={self.h}, toks={list(self.token_ids)})"


# ══════════════════════════════════════════════════════════════════
# 2. BlockPool — 核心块管理器
#    vLLM 对应：vllm/v1/core/block_pool.py :: BlockPool
# ══════════════════════════════════════════════════════════════════
class BlockPool:
    """
    两个核心数据结构：
      free_queue:    OrderedDict 模拟双向链表 LRU 队列
                     popitem(last=False) = 从队头弹出（LRU 最旧，优先驱逐）
      hash_to_block: block_hash → KVCacheBlock
                     Prefix Cache 查找表；ref_cnt=0 时映射仍保留（懒惰驱逐）
    """

    def __init__(self, num_blocks: int):
        # 系统启动时一次性预创建所有物理块
        # 对应 vLLM：_allocate_kv_cache_tensors() 申请 GPU Tensor，BlockPool 创建元数据
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue: OrderedDict[int, KVCacheBlock] = OrderedDict(
            (b.block_id, b) for b in self.blocks
        )
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        print(f"[BlockPool] 预分配 {num_blocks} 个物理块，全部空闲")
        print(f"            ↳ 对应 GPU 侧：一次 torch.zeros(size, dtype=int8) 申请大 Tensor")

    # ── 分配新块 ─────────────────────────────────────────────────
    def get_new_blocks(self, num: int) -> list[KVCacheBlock]:
        """
        从 LRU 队头弹出 num 个块（最旧/最不常用的优先被重用）。

        [设计要点] 弹出时若块有旧 hash，立即清除哈希映射（懒惰驱逐的"触发点"）：
          - 之前该块的 KV 数据即将被覆盖，旧 hash → block 的映射必须失效
          - 但在此之前的窗口期内，hash 映射一直保留，供同前缀请求复用
        """
        if num > len(self.free_queue):
            raise RuntimeError(f"OOM：需要 {num} 块，仅剩 {len(self.free_queue)} 块")
        allocated = []
        for _ in range(num):
            _, block = self.free_queue.popitem(last=False)  # 队头弹出（LRU）
            self._maybe_evict_cached_block(block)           # 清除旧 hash（懒惰驱逐触发）
            assert block.ref_cnt == 0
            block.ref_cnt = 1
            allocated.append(block)
        return allocated

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        [设计要点] 懒惰驱逐：只有在块真正被重新分配给新内容时才清除旧 hash 映射，
        而非在 ref_cnt=0 归还时立即清除。这样在两次分配之间的"窗口期"内，
        相同前缀的新请求仍能通过 hash 找到这个块，实现零开销复用。
        """
        if block.block_hash is None:
            return False  # 从未被缓存，无需清理
        old_h = f"{block.block_hash & 0xFFFF:04X}"
        self.hash_to_block.pop(block.block_hash, None)  # 从全局 hash 表移除
        block.reset()                                    # 清除块自身的 hash 元数据
        print(f"      [懒惰驱逐] B{block.block_id:02d} 旧 hash={old_h} 清除，准备写入新内容")
        return True

    # ── 释放块 ───────────────────────────────────────────────────
    def free_blocks(self, blocks: list[KVCacheBlock]):
        """
        [设计要点 1] ref_cnt -= 1；只有降为 0 才归还空闲队列。
        [设计要点 2] 逆序归还（reversed）：后缀块先入队（排队头，优先被驱逐），
                    前缀块后入队（排队尾，存活最久）→ 最大化 prefix cache 复用机会。
        [设计要点 3] hash_to_block 映射不清除 → 懒惰驱逐保留窗口期。
        """
        freed = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                freed.append(block)

        # 逆序追加到 OrderedDict 尾部（尾 = LRU 最新 = 最后被驱逐）
        for block in reversed(freed):
            self.free_queue[block.block_id] = block

    # ── Prefix Cache 命中时复用 ──────────────────────────────────
    def touch(self, block: KVCacheBlock):
        """
        prefix cache 命中：将块从空闲队列摘除（若 ref_cnt=0 说明在队列里），
        然后 ref_cnt += 1，标记为正在使用。
        """
        if block.ref_cnt == 0:
            self.free_queue.pop(block.block_id, None)  # 从空闲队列摘除
        block.ref_cnt += 1

    def get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
        return self.hash_to_block.get(block_hash)

    # ── 注册满块到 Prefix Cache ──────────────────────────────────
    def cache_full_block(self, block: KVCacheBlock, block_hash: int, token_ids: tuple):
        """块被填满时注册到 hash_to_block，供后续请求匹配前缀。"""
        block.block_hash = block_hash
        block.token_ids  = token_ids
        self.hash_to_block[block_hash] = block

    def num_free(self) -> int:
        return len(self.free_queue)

    def print_status(self):
        used = [b for b in self.blocks if b.ref_cnt > 0]
        print(f"\n  [BlockPool] 空闲 {self.num_free()}/{len(self.blocks)} 块"
              f"  |  Prefix Cache 条目：{len(self.hash_to_block)} 个")
        print(f"  LRU 空闲队列（队头→队尾，队头优先驱逐）: {list(self.free_queue.keys())}")
        if used:
            print(f"  使用中的块：")
            for b in used:
                print(f"    {b}")
        if self.hash_to_block:
            print(f"  Prefix Cache (hash → block)：")
            for h, b in self.hash_to_block.items():
                print(f"    hash={h & 0xFFFF:04X} → B{b.block_id:02d}"
                      f"  ref={b.ref_cnt}  toks={list(b.token_ids)}")


# ══════════════════════════════════════════════════════════════════
# 3. Hash 计算（链式，只对满块）
#    vLLM 对应：kv_cache_utils.py :: hash_block_tokens()
#              kv_cache_utils.py :: request_block_hasher()
# ══════════════════════════════════════════════════════════════════
def hash_block_tokens(parent_hash: Optional[int], token_ids: tuple) -> int:
    """
    [设计要点] 链式 hash：hash_i = sha256(hash_{i-1}, token_ids_i)
      - 首块：parent_hash = NONE_HASH（固定种子）
      - 即使两块 token_ids 完全相同，若 parent_hash 不同 → hash 不同
      - 保证"断一处则全链断裂"，不同上下文的相同内容不会被错误匹配
    """
    p = parent_hash if parent_hash is not None else NONE_HASH
    return _sha(p, token_ids)


def compute_block_hashes(token_ids: list) -> list[dict]:
    """
    对 token_ids 按 BLOCK_SIZE 切分，为每个满块计算链式 hash。
    尾部不满的块不计算（内容未稳定，还可能追加 token）。
    对应 vLLM request_block_hasher() 的增量计算逻辑。
    """
    entries, prev_hash = [], None
    for i in range(len(token_ids) // BLOCK_SIZE):
        toks = tuple(token_ids[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE])
        h = hash_block_tokens(prev_hash, toks)
        entries.append({'hash': h, 'tokens': toks, 'parent_hash': prev_hash})
        prev_hash = h
    return entries


# ══════════════════════════════════════════════════════════════════
# 4. Request — 请求抽象（含预计算的 block_hashes）
# ══════════════════════════════════════════════════════════════════
@dataclass
class Request:
    """
    vLLM 在 Request 构造时即预计算 block_hashes，
    调度阶段只用 hash 去匹配，无需再遍历 token_ids。
    """
    req_id:    str
    token_ids: list[int]
    blocks:    list[KVCacheBlock] = field(default_factory=list)

    def __post_init__(self):
        self._entries    = compute_block_hashes(self.token_ids)
        self.block_hashes = [e['hash'] for e in self._entries]

    @property
    def num_full_blocks(self) -> int:
        return len(self.token_ids) // BLOCK_SIZE

    @property
    def has_partial(self) -> bool:
        return len(self.token_ids) % BLOCK_SIZE > 0

    @property
    def num_blocks_needed(self) -> int:
        return self.num_full_blocks + (1 if self.has_partial else 0)

    def show_block_table(self):
        print(f"  Block Table（逻辑块索引 → 物理 block_id）：")
        for i, blk in enumerate(self.blocks):
            suffix = " ← prefix cache 命中" if blk.block_hash is not None and i < len(self.block_hashes) else ""
            print(f"    逻辑块[{i}] → B{blk.block_id:02d}{suffix}")


# ══════════════════════════════════════════════════════════════════
# 5. KVCacheManager — 调度器调用的高层接口
#    vLLM 对应：KVCacheCoordinator / FullAttentionManager
# ══════════════════════════════════════════════════════════════════
class KVCacheManager:

    def __init__(self, pool: BlockPool):
        self.pool = pool

    def allocate_slots(self, req: Request) -> int:
        """
        分配 block，优先复用 prefix cache。返回命中的块数（无需重算 KV）。

        流程：
          1. find_longest_cache_hit — 逐块检查 hash 链
          2. get_new_blocks         — 为未命中的块申请新物理块
          3. 建立 block table
          4. cache_full_blocks      — 注册新满块到 Prefix Cache
        """
        print(f"\n  [{req.req_id}] tokens={req.token_ids}")
        print(f"  需要 {req.num_blocks_needed} 块"
              f"（{req.num_full_blocks} 满块"
              f"{'+ 1 尾部块' if req.has_partial else ''}）")

        # ── Step 1：find_longest_cache_hit ──────────────────────
        # 逐块匹配 hash，遇到 miss 立即 break（链式依赖，后续必然也 miss）
        cached: list[KVCacheBlock] = []
        for bh in req.block_hashes:
            blk = self.pool.get_cached_block(bh)
            if blk is None:
                break                # 链断裂，不再继续
            self.pool.touch(blk)     # ref_cnt += 1；从空闲队列摘除（若在其中）
            cached.append(blk)

        num_hit = len(cached)
        if num_hit:
            print(f"  ✓ Prefix Cache 命中 {num_hit} 块 → "
                  f"B{[b.block_id for b in cached]}  (省去重算这部分 KV Cache)")
        else:
            print(f"  ✗ Prefix Cache 未命中")

        # ── Step 2：为剩余块申请新物理块 ────────────────────────
        num_new  = req.num_blocks_needed - num_hit
        new_blks = self.pool.get_new_blocks(num_new) if num_new else []
        if new_blks:
            print(f"  + 新分配 {num_new} 块 → B{[b.block_id for b in new_blks]}")

        # ── Step 3：建立 Block Table（逻辑块 → 物理块映射） ─────
        req.blocks = cached + new_blks

        # ── Step 4：cache_full_blocks（注册新满块到 Prefix Cache）
        # 只有满块才注册；尾部不满块内容未稳定，不注册
        for i, blk in enumerate(new_blks):
            logical_idx = num_hit + i
            if logical_idx < req.num_full_blocks:
                e = req._entries[logical_idx]
                self.pool.cache_full_block(blk, e['hash'], e['tokens'])
                print(f"    注册 Prefix Cache: B{blk.block_id:02d} "
                      f"hash={e['hash'] & 0xFFFF:04X}  toks={list(e['tokens'])}")

        req.show_block_table()
        return num_hit

    def free_request(self, req: Request):
        ids = [b.block_id for b in req.blocks]
        print(f"\n  [{req.req_id}] 释放  block_ids={ids}")
        print(f"              （逆序归还：后缀块先入 LRU 队头，前缀块后入队尾，存活最久）")
        self.pool.free_blocks(req.blocks)
        req.blocks = []


# ══════════════════════════════════════════════════════════════════
# 6. 演示主程序
# ══════════════════════════════════════════════════════════════════
def main():
    pool    = BlockPool(NUM_BLOCKS)
    manager = KVCacheManager(pool)

    # ━━ 场景 1：首个请求，全新分配，展示链式 Hash 计算 ━━━━━━━━━
    _sep("场景 1：首个请求 — 全部新分配，展示链式 Hash 计算")
    print("  [关键] 只有满块才计算 hash；当前 hash 依赖父块 hash（链式）")
    #
    #  tokens: [0,1,2,3 | 4,5,6,7 | 8]
    #           满块 0     满块 1    尾部不满块（不计算 hash）
    #
    req1 = Request("req-1", list(range(9)))
    print(f"\n  token_ids: {req1.token_ids}  (BLOCK_SIZE={BLOCK_SIZE})")
    print(f"  满块 Hash 链：")
    for i, e in enumerate(req1._entries):
        ph = f"{e['parent_hash'] & 0xFFFF:04X}" if e['parent_hash'] else "NONE_HASH"
        print(f"    Block[{i}]: toks={list(e['tokens'])}"
              f"  parent={ph} → hash={e['hash'] & 0xFFFF:04X}")
    print(f"    Block[2]: toks={req1.token_ids[8:]}  (尾部不满，不计算 hash，不入 Prefix Cache)")

    manager.allocate_slots(req1)
    pool.print_status()

    # ━━ 场景 2：相同前缀请求，Prefix Cache 命中 ━━━━━━━━━━━━━━━━
    _sep("场景 2：相同前缀请求 — Prefix Cache 命中，ref_cnt 增加")
    print("  [关键] 命中的块直接复用（touch → ref_cnt+1），无需重算 KV")
    print("  [关键] 前 8 个 token 相同 → 前 2 块 hash 相同 → 全部命中")
    #
    #  req2: [0,1,2,3 | 4,5,6,7 | 100,101,102,103]
    #          命中B0    命中B1      新块
    #
    req2 = Request("req-2", list(range(8)) + [100, 101, 102, 103])
    hit = manager.allocate_slots(req2)
    print(f"\n  B0 和 B1 现在 ref_cnt=2（被 req-1 和 req-2 共享）")
    pool.print_status()

    # ━━ 场景 3：头部插入新 token，hash 链从第一块断裂 ━━━━━━━━━━
    _sep("场景 3：前缀不同 — hash 链从第一块完全断裂")
    print("  [关键] 链式设计保证：断一处则全链断裂")
    print("  [关键] token 999 导致 Block[0] hash 变 → parent_hash 变 → 后续全变")
    #
    #  req3: [999,0,1,2 | 3,4,5,6 | 50,51,52,53]
    #         999 在首块 → hash_0 变 → hash_1 也变 → 全部 miss
    #
    req3 = Request("req-3", [999] + list(range(7)) + [50, 51, 52, 53])
    print(f"\n  req-3 的 hash 与 Prefix Cache 的匹配情况：")
    for i, e in enumerate(req3._entries):
        in_cache = e['hash'] in pool.hash_to_block
        mark = "✓ 命中" if in_cache else "✗ 未命中"
        print(f"    Block[{i}]: hash={e['hash'] & 0xFFFF:04X}  {mark}")

    manager.allocate_slots(req3)
    pool.print_status()

    # ━━ 场景 4：释放 req-1 和 req-2，观察 ref_cnt 与 LRU 归还 ━━
    _sep("场景 4：释放 req-1 和 req-2 — ref_cnt 联动，LRU 逆序归还")
    print("  [关键] B0/B1 被 req-1 和 req-2 共享，req-1 释放后 ref 降为 1（不回收）")
    print("  [关键] 逆序归还：B1 先入队尾，B0 最后入队尾（B0 为前缀块，最久存活）")
    print("  [关键] hash 映射不清除（懒惰驱逐），保留窗口期")

    manager.free_request(req1)   # B0.ref: 2→1, B1.ref: 2→1, B2.ref: 1→0 → B2 回队
    print(f"  req-1 释放后：B2 归还队尾，B0/B1 仍被 req-2 持有（ref=1）")

    manager.free_request(req2)   # B0.ref: 1→0, B1.ref: 1→0, B3.ref: 1→0 → 逆序入队
    print(f"  req-2 释放后：freed=[B0,B1,B3]，逆序追加 → 队尾顺序 ...B3, B1, B0")
    print(f"  B0 在队尾（前缀块，最后被驱逐）✓")
    pool.print_status()

    # ━━ 场景 5：新请求验证懒惰驱逐（ref_cnt=0 的块仍可命中） ━━━
    _sep("场景 5：懒惰驱逐验证 — ref_cnt=0 的块仍能通过 hash 命中")
    print("  [关键] B0 和 B1 虽已 ref_cnt=0 归还空闲队列，")
    print("          但 hash_to_block 映射仍保留 → 新请求仍可命中复用！")
    print("          只有当块被 get_new_blocks 重新分配时，旧 hash 才被清除。")

    req4 = Request("req-4", list(range(8)) + [200, 201, 202, 203])
    hit = manager.allocate_slots(req4)
    print(f"\n  结果：命中 {hit} 块（B0, B1 从空闲队列摘出，ref_cnt: 0 → 1）")
    pool.print_status()

    # ━━ 场景 6：释放全部，观察最终 LRU 状态 ━━━━━━━━━━━━━━━━━━━
    _sep("场景 6：释放所有请求 — 观察 LRU 队列最终顺序")
    print("  [关键] 前缀块（被多个请求复用的块）应排在 LRU 队列最尾端")

    manager.free_request(req3)   # B4/B5/B6 全部 ref→0，逆序：B6先，B4最后
    manager.free_request(req4)   # B0/B1/B7 全部 ref→0，逆序：B7先，B0最后

    pool.print_status()
    print(f"\n  LRU 队尾（最后被驱逐）= B0 = 被 req-1/2/4 三次复用的前缀块")
    print(f"  LRU 队头（最先被驱逐）= 从未被共享的新块（无复用价值）")

    # ━━ 总结 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sep("vLLM PagedAttention 核心设计总结")
    print("""
  1. 预分配块池
       启动时一次 torch.zeros(size, int8) 申请全部显存，
       运行时零 cudaMalloc/Free → 无外部碎片，无分配延迟。

  2. Block Table（逻辑 → 物理映射）
       请求不要求连续物理块；调度器维护 [block_id, ...] 列表，
       Attention Kernel 通过 block_table[逻辑索引] 定位物理 KV Cache。

  3. 链式 Hash（只对满块）
       hash(block_i) = sha256( hash(block_{i-1}),  token_ids_i )
       → 尾部不满块不参与 Prefix Cache（内容未稳定）。
       → 两条路径即使 token 相同，parent_hash 不同则 hash 不同，避免误匹配。
       → 断一处则全链断裂（遇到 miss 立即停止查找）。

  4. Prefix Cache（find_longest_cache_hit）
       新请求按 hash 链逐块匹配，命中则 touch()（ref_cnt+1，从空闲队列摘除），
       直到第一个 miss → 后续块全不查，节省开销。

  5. 懒惰驱逐（Lazy Eviction）
       free_blocks：ref_cnt→0 归还空闲队列，但 hash_to_block 不清除。
       get_new_blocks：弹出块时才调用 _maybe_evict_cached_block 清除旧 hash。
       窗口期内相同前缀的新请求可"零开销"命中并从空闲队列摘回。

  6. LRU 逆序释放
       free_blocks 对 freed 列表做 reversed() 后追加到队尾，
       后缀块（无复用价值）排队头优先被驱逐，
       前缀块（高复用价值）排队尾最久存活。
""")


if __name__ == "__main__":
    main()
