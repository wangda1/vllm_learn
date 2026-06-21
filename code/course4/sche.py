"""
vLLM 调度器核心机制教学 Demo
==============================
本文件用纯 Python 模拟 vLLM v1 调度器（vllm/v1/core/sched/scheduler.py）的核心逻辑，
不依赖任何 GPU / CUDA 环境，可直接运行。

覆盖的核心主题：
  1. 请求状态机：WAITING → RUNNING ↔ PREEMPTED → FINISHED
  2. KV Cache Block 池：物理显存的最小管理单元
  3. 调度核心循环（schedule 方法）：
       a. 优先处理 running 队列（保持 decode 连续性）
       b. token_budget 限制单步 batch 大小
       c. KV Cache 不足时触发抢占（FCFS / Priority 两种策略）
       d. 无抢占时再从 waiting 队列接纳新请求（chunked prefill 也在此处处理）
  4. SchedulerOutput：调度结果的结构化表达，是调度器与执行引擎的桥梁
  5. _update_after_schedule：乐观推进每个请求的 num_computed_tokens
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. 请求状态机
# 对应源码：vllm/v1/request.py :: RequestStatus
# ─────────────────────────────────────────────────────────────────────────────

class RequestStatus(enum.IntEnum):
    """
    请求在调度器生命周期中的状态流转：

        add_request()  ──→  WAITING
        schedule()     ──→  RUNNING      （waiting 队列 → running 队列）
        KV Cache 不足  ──→  PREEMPTED    （释放 KV Cache，回 waiting 队首）
        生成完成       ──→  FINISHED
    """
    WAITING   = 1   # 在 waiting 队列中排队，尚未开始执行
    RUNNING   = 2   # 已被调度，正在执行（prefill 或 decode 阶段）
    PREEMPTED = 3   # 被抢占，KV Cache 已释放，重回 waiting 队首等待恢复
    FINISHED  = 4   # 生成完成（达到 max_tokens 或 EOS）


# ─────────────────────────────────────────────────────────────────────────────
# 2. 请求对象
# 对应源码：vllm/v1/request.py :: Request
# ─────────────────────────────────────────────────────────────────────────────

class Request:
    """
    调度器视角下的单个推理请求。

    关键字段与计算逻辑：
      num_tokens          = len(prompt_token_ids) + len(output_token_ids)
      num_computed_tokens : 调度器已安排计算的 token 数（"乐观推进"，GPU 尚未真正完成）

    本轮需计算的 token 数：
      num_new_tokens = num_tokens - num_computed_tokens

    is_prefill 判断：
      num_computed_tokens < num_prompt_tokens → 仍在处理 prompt（prefill 阶段）
      num_computed_tokens >= num_prompt_tokens → 在生成 output（decode 阶段）
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_tokens: int = 20,
        priority: int = 0,
    ) -> None:
        self.request_id       = request_id
        self.prompt_token_ids = prompt_token_ids
        self.max_tokens       = max_tokens
        self.priority         = priority
        self.arrival_time     = time.monotonic()

        self.status                = RequestStatus.WAITING
        self.num_computed_tokens   = 0    # 调度器已安排计算的 token 数（prefill+decode 累计）
        self.output_token_ids: list[int] = []  # 已生成的 output token ID
        self.num_preemptions       = 0    # 被抢占次数（统计用）

        # 当前持有的 KV Cache block 编号列表
        self.block_ids: list[int] = []

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """当前序列总长度 = prompt + 已生成 output。"""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_prefill(self) -> bool:
        """num_computed_tokens < num_prompt_tokens → 仍在 prefill 阶段。"""
        return self.num_computed_tokens < self.num_prompt_tokens

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, "
            f"prompt={self.num_prompt_tokens}, "
            f"computed={self.num_computed_tokens}, "
            f"output={self.num_output_tokens}, "
            f"status={self.status.name})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. 调度策略与等待队列
# 对应源码：vllm/v1/core/sched/request_queue.py
# ─────────────────────────────────────────────────────────────────────────────

class SchedulingPolicy(enum.Enum):
    FCFS     = "fcfs"      # 先来先服务（默认）
    PRIORITY = "priority"  # priority 越小越优先；相同 priority 则 arrival_time 越早越优先


class WaitingQueue:
    """
    对应源码：vllm/v1/core/sched/request_queue.py :: RequestQueue

    封装 waiting 队列，同时支持 FCFS 和 Priority 策略。
    """

    def __init__(self, policy: SchedulingPolicy) -> None:
        self.policy = policy
        self._queue: deque[Request] = deque()

    def add(self, request: Request) -> None:
        """新请求进入队尾。"""
        self._queue.append(request)

    def prepend(self, request: Request) -> None:
        """被抢占的请求放回队首，下一步优先被恢复调度。"""
        self._queue.appendleft(request)

    def peek(self) -> Optional[Request]:
        """查看下一个应被调度的请求，不弹出。"""
        if not self._queue:
            return None
        if self.policy == SchedulingPolicy.FCFS:
            return self._queue[0]
        # Priority 策略：(priority, arrival_time) 越小越优先
        return min(self._queue, key=lambda r: (r.priority, r.arrival_time))

    def pop(self) -> Optional[Request]:
        """弹出下一个应被调度的请求。"""
        req = self.peek()
        if req is None:
            return None
        self._queue.remove(req)
        return req

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 简化版 KV Cache 管理器
# 对应源码：vllm/v1/core/kv_cache_manager.py :: KVCacheManager
#
# 真实实现涉及 block 复用、prefix caching hash、引用计数、
# 跨层 KV 压缩等复杂机制，这里只模拟"分配/释放 block"的核心行为。
# ─────────────────────────────────────────────────────────────────────────────

class SimplifiedKVCacheManager:
    """
    教学用简化 KV Cache 管理器。

    核心概念：
      block_size  : 每个 block 可存放多少个 token 的 KV Cache
      total_blocks: 系统中物理 block 的总数（模拟显存容量）
      free_blocks : 当前空闲的 block 编号集合

    核心接口：
      allocate_slots(request, num_new_tokens) → list[int] | None
          为请求本轮新增的 num_new_tokens 分配 block。
          返回新分配的 block ID 列表（空列表表示无需新 block）；
          若显存不足返回 None → 调度器需触发抢占。

      free(request)
          释放请求占用的全部 block，归还到空闲池（抢占时调用）。
    """

    def __init__(self, total_blocks: int = 20, block_size: int = 4) -> None:
        self.block_size   = block_size
        self.total_blocks = total_blocks
        self._free_blocks: set[int] = set(range(total_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def _extra_blocks_needed(self, current_tokens: int, additional_tokens: int) -> int:
        """
        计算增加 additional_tokens 后，相比当前还需要额外分配多少个 block。

        示例（block_size=4）：
          current=5 → ceil(5/4)=2 blocks 已在用
          additional=3 → total=8 → ceil(8/4)=2 blocks
          需新增 = max(0, 2-2) = 0  （当前 block 还够）

          current=4 → ceil(4/4)=1 block
          additional=1 → total=5 → ceil(5/4)=2 blocks
          需新增 = max(0, 2-1) = 1
        """
        total_needed   = (current_tokens + additional_tokens + self.block_size - 1) // self.block_size
        current_blocks = (current_tokens + self.block_size - 1) // self.block_size if current_tokens > 0 else 0
        return max(0, total_needed - current_blocks)

    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> Optional[list[int]]:
        """
        对应源码：KVCacheManager.allocate_slots()

        为 request 本轮新增 num_new_tokens 分配 KV Cache block。
        分配成功返回新 block ID 列表（可为空）；显存不足返回 None。
        """
        need = self._extra_blocks_needed(request.num_computed_tokens, num_new_tokens)
        if need > len(self._free_blocks):
            return None  # 显存不足

        new_block_ids: list[int] = []
        for _ in range(need):
            bid = self._free_blocks.pop()
            new_block_ids.append(bid)
            request.block_ids.append(bid)
        return new_block_ids

    def free(self, request: Request) -> None:
        """
        对应源码：KVCacheManager.free()

        释放请求占用的全部 KV Cache block。
        抢占时调用：block 归还后，请求的 KV 数据丢失，
        下次调度必须从 num_computed_tokens=0 重新开始计算。
        """
        self._free_blocks.update(request.block_ids)
        request.block_ids = []

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(total={self.total_blocks}, "
            f"free={self.num_free_blocks}, block_size={self.block_size})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. 调度器输出
# 对应源码：vllm/v1/core/sched/output.py :: SchedulerOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerOutput:
    """
    调度结果，是调度器与执行引擎（ModelRunner/Worker）的桥梁。

    执行引擎依据此结构知道：
      1. 本轮执行哪些请求（scheduled_new_reqs + scheduled_running_reqs）
      2. 每个请求计算多少 token（num_scheduled_tokens）
      3. KV Cache 写入哪些 block（req_to_new_block_ids）

    区分 new_reqs 和 running_reqs 的原因：
      - new_reqs     : 首次调度，worker 还没有请求元数据，需完整传输
      - running_reqs : 已在 worker 侧缓存过元数据，只需增量同步新 block
    """
    scheduled_new_reqs:     list[Request]        # 首次被调度（来自 waiting）
    scheduled_running_reqs: list[Request]        # 继续执行（已在 running 中）
    num_scheduled_tokens:   dict[str, int]       # req_id → 本轮调度的 token 数
    req_to_new_block_ids:   dict[str, list[int]] # req_id → 本轮新分配的 block IDs
    preempted_req_ids:      list[str] = field(default_factory=list)

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())

    def summary(self) -> str:
        lines = [
            "─" * 64,
            f"[SchedulerOutput] total_tokens={self.total_num_scheduled_tokens}",
        ]
        if self.scheduled_new_reqs:
            lines.append(f"  首次调度 (prefill/restore): "
                         f"{[r.request_id for r in self.scheduled_new_reqs]}")
        if self.scheduled_running_reqs:
            lines.append(f"  继续执行 (decode/chunk)  : "
                         f"{[r.request_id for r in self.scheduled_running_reqs]}")
        for rid, n in self.num_scheduled_tokens.items():
            new_blks = self.req_to_new_block_ids.get(rid, [])
            lines.append(f"    {rid}: tokens={n}, new_blocks={new_blks}")
        if self.preempted_req_ids:
            lines.append(f"  ⚡ 被抢占: {self.preempted_req_ids}")
        lines.append("─" * 64)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 核心调度器
# 对应源码：vllm/v1/core/sched/scheduler.py :: Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:
    """
    vLLM v1 调度器核心类（教学简化版）。

    内部维护两个核心队列：
      self.waiting : WaitingQueue  —— 等待执行的请求（WAITING / PREEMPTED）
      self.running : list[Request] —— 当前执行中的请求（RUNNING，按进入顺序排列）

    每调度步调用：
      output = scheduler.schedule()     # 产生本轮任务清单
      scheduler.update_from_output(...) # 模拟 GPU 执行后的状态回写
    """

    def __init__(
        self,
        max_num_running_reqs:     int = 4,
        max_num_scheduled_tokens: int = 16,
        max_model_len:            int = 128,
        kv_cache_manager: Optional[SimplifiedKVCacheManager] = None,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
    ) -> None:
        # ── 调度约束 ──────────────────────────────────────────────────────────
        self.max_num_running_reqs     = max_num_running_reqs       # 最大并发请求数
        self.max_num_scheduled_tokens = max_num_scheduled_tokens   # 单步最大 token 数
        self.max_model_len            = max_model_len
        self.policy                   = policy

        # ── 核心队列 ──────────────────────────────────────────────────────────
        self.waiting = WaitingQueue(policy)
        self.running: list[Request] = []     # 越靠前 = 越早进入 running

        # ── 请求全局索引（request_id → Request） ──────────────────────────────
        self.requests: dict[str, Request] = {}

        # ── KV Cache 管理器 ───────────────────────────────────────────────────
        self.kv_cache_manager = kv_cache_manager or SimplifiedKVCacheManager()

        self._step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 6.1 增加请求
    # 对应源码：Scheduler.add_request()
    # ─────────────────────────────────────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """
        将新请求放入 waiting 队列，初始状态为 WAITING。

        真实请求接入路径：
          客户端 ─ZMQ→ EngineCoreProc.input_queue
            → _handle_client_request() → EngineCore.add_request()
              → Scheduler.add_request()  ← 此处
        """
        request.status = RequestStatus.WAITING
        self.waiting.add(request)
        self.requests[request.request_id] = request
        print(f"  [add_request] {request.request_id} "
              f"(prompt_len={request.num_prompt_tokens}, "
              f"max_tokens={request.max_tokens})")

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    # ─────────────────────────────────────────────────────────────────────────
    # 6.2 核心调度方法
    # 对应源码：Scheduler.schedule() → SchedulerOutput
    # ─────────────────────────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """
        执行一次 scheduler step，返回本轮 SchedulerOutput。

        ┌───────────────────────────────────────────────────────────────┐
        │ Phase 1：优先处理 running 队列                                  │
        │   for request in running:                                     │
        │     num_new_tokens = num_tokens - num_computed_tokens         │
        │     num_new_tokens = min(num_new_tokens, token_budget, ...)   │
        │     new_blocks = kv_cache_manager.allocate_slots(...)         │
        │     if new_blocks is None:  → 触发抢占                         │
        │       FCFS    : 抢 running 队列末尾的请求                       │
        │       Priority: 抢优先级最低的请求                               │
        │                                                               │
        │ Phase 2：若无抢占且有 token_budget 余量，处理 waiting 队列        │
        │   while waiting and token_budget > 0:                        │
        │     request = waiting.peek()                                  │
        │     num_new_tokens = num_tokens - num_computed_tokens         │
        │     new_blocks = kv_cache_manager.allocate_slots(...)         │
        │     if new_blocks is None:  → 停止接纳新请求（不抢占）           │
        │     else: waiting → running                                   │
        └───────────────────────────────────────────────────────────────┘
        """
        self._step += 1
        print(f"\n{'='*64}")
        print(f"[Step {self._step}] 开始调度 "
              f"| running={len(self.running)} | waiting={len(self.waiting)} "
              f"| free_blocks={self.kv_cache_manager.num_free_blocks}")

        # ── 本轮调度结果容器 ───────────────────────────────────────────────────
        scheduled_new_reqs:     list[Request]        = []
        scheduled_running_reqs: list[Request]        = []
        preempted_reqs:         list[Request]        = []
        num_scheduled_tokens:   dict[str, int]       = {}
        req_to_new_block_ids:   dict[str, list[int]] = {}

        # token_budget：本 step 最多调度多少 token（由 max_num_batched_tokens 决定）
        token_budget = self.max_num_scheduled_tokens

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1：调度 RUNNING 队列
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n[Phase 1] 调度 running 队列 (token_budget={token_budget})")

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # ── 计算本轮应为该请求处理多少 token ───────────────────────────────
            # 对应源码：
            #   num_new_tokens = (request.num_tokens_with_spec
            #                     + request.num_output_placeholders
            #                     - request.num_computed_tokens)
            # 教学简化（无 spec decode）：
            num_new_tokens = request.num_tokens - request.num_computed_tokens

            # 受 token_budget 约束
            num_new_tokens = min(num_new_tokens, token_budget)

            # 受模型最大长度约束（避免 input_pos 越界）
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

            if num_new_tokens == 0:
                req_index += 1
                continue

            phase_label = "prefill/chunk" if request.is_prefill else "decode"
            print(f"  尝试调度 {request.request_id} [{phase_label}]: "
                  f"need={num_new_tokens}, budget={token_budget}, "
                  f"free_blks={self.kv_cache_manager.num_free_blocks}")

            # ── 尝试分配 KV Cache block（可能触发多次抢占） ──────────────────────
            # 对应源码：
            #   while True:
            #       new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
            #       if new_blocks is not None: break
            #       preempted_req = self.running.pop()  # FCFS
            #       self._preempt_request(preempted_req)
            #       if preempted_req == request: break   # 自身被抢，无法调度
            while True:
                new_block_ids = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_block_ids is not None:
                    break  # 分配成功

                # 分配失败 → 触发抢占
                print(f"  ⚠  KV Cache 不足！触发抢占 "
                      f"(free={self.kv_cache_manager.num_free_blocks})")

                if self.policy == SchedulingPolicy.PRIORITY:
                    # Priority 策略：抢优先级最低（priority 最大）的请求；
                    # 相同 priority 则抢到达最晚的（arrival_time 最大）
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                    self.running.remove(preempted_req)
                else:
                    # FCFS 默认策略：抢 running 队列末尾的请求
                    # 越靠后 = 越晚进入 running = 对整体连续性影响最小
                    preempted_req = self.running.pop()

                self._preempt_request(preempted_req)
                preempted_reqs.append(preempted_req)

                if preempted_req == request:
                    # 边界情况：自身被抢占 → 说明已无其他请求可腾空间
                    # 对应源码：if preempted_req == request: break
                    print(f"  ✗ {request.request_id} 自身被抢，停止 Phase 1")
                    new_block_ids = None
                    break

            if new_block_ids is None:
                # 自身被抢或无法分配，退出 Phase 1
                break

            # ── 调度成功 ────────────────────────────────────────────────────
            scheduled_running_reqs.append(request)
            req_to_new_block_ids[request.request_id] = new_block_ids
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            print(f"  ✓ {request.request_id}: "
                  f"tokens={num_new_tokens}, new_blks={new_block_ids}, "
                  f"budget_left={token_budget}")

        # ══════════════════════════════════════════════════════════════════════
        # Phase 2：调度 WAITING 队列
        #
        # 前提条件：
        #   (1) Phase 1 中没有发生抢占（系统资源足够，值得接纳新请求）
        #   (2) 还有 token_budget 余量
        #   (3) running 队列未达到上限 max_num_running_reqs
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n[Phase 2] 调度 waiting 队列 "
              f"(preempted={len(preempted_reqs)}, token_budget={token_budget})")

        # 对应源码：if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        if not preempted_reqs:
            while self.waiting and token_budget > 0:

                # 对应源码：if len(self.running) == self.max_num_running_reqs: break
                if len(self.running) >= self.max_num_running_reqs:
                    print(f"  running 满 ({self.max_num_running_reqs})，停止接纳")
                    break

                request = self.waiting.peek()
                if request is None:
                    break

                # ── 计算本轮 num_new_tokens ──────────────────────────────────
                # 对应源码：
                #   num_computed_tokens = num_new_local_computed_tokens
                #                       + num_external_computed_tokens
                #   （本地 Prefix Cache 命中 + 外部 KV Cache 命中）
                #   num_new_tokens = request.num_tokens - num_computed_tokens
                #   num_new_tokens = min(num_new_tokens, token_budget)
                #
                # 简化：无 Prefix Cache，num_computed_tokens 直接用字段值
                # （PREEMPTED 请求已被重置为 0；新请求也是 0）
                num_computed_tokens = request.num_computed_tokens
                num_new_tokens = request.num_tokens - num_computed_tokens
                num_new_tokens = min(num_new_tokens, token_budget)

                if num_new_tokens <= 0:
                    self.waiting.pop()
                    continue

                print(f"  尝试接纳 {request.request_id} "
                      f"[{'恢复' if request.status == RequestStatus.PREEMPTED else '首次'}]: "
                      f"need={num_new_tokens}, budget={token_budget}, "
                      f"free_blks={self.kv_cache_manager.num_free_blocks}")

                # ── 尝试分配 KV Cache block ─────────────────────────────────
                new_block_ids = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )

                if new_block_ids is None:
                    # 显存不足：停止接纳 waiting 请求（不在此处触发抢占）
                    print(f"  ✗ {request.request_id}: 显存不足，停止接纳 waiting 请求")
                    break

                # ── 从 waiting 弹出并加入 running ──────────────────────────
                self.waiting.pop()
                self.running.append(request)

                # 区分首次调度 vs 被抢占后恢复
                # 对应源码：
                #   if request.status == RequestStatus.WAITING:
                #       scheduled_new_reqs.append(request)
                #   elif request.status == RequestStatus.PREEMPTED:
                #       scheduled_resumed_reqs.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    label = "首次调度"
                else:  # PREEMPTED
                    scheduled_running_reqs.append(request)  # 简化：与 running_reqs 合并
                    label = "恢复执行"

                # new_block_ids 即本轮为该请求新分配的 block（新请求 = 全部 blocks）
                req_to_new_block_ids[request.request_id] = new_block_ids
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens

                # 更新请求状态（num_computed_tokens 在 _update_after_schedule 中统一累加）
                request.status = RequestStatus.RUNNING

                print(f"  ✓ {request.request_id} [{label}]: "
                      f"tokens={num_new_tokens}, all_blks={request.block_ids}, "
                      f"budget_left={token_budget}")
        else:
            print("  跳过（本轮发生了抢占，不接纳新请求）")

        # ── 构造 SchedulerOutput ────────────────────────────────────────────
        output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            req_to_new_block_ids=req_to_new_block_ids,
            preempted_req_ids=[r.request_id for r in preempted_reqs],
        )

        # ── 调度后状态更新 ──────────────────────────────────────────────────
        self._update_after_schedule(output)
        print(output.summary())
        return output

    # ─────────────────────────────────────────────────────────────────────────
    # 6.3 抢占请求
    # 对应源码：Scheduler._preempt_request()
    # ─────────────────────────────────────────────────────────────────────────

    def _preempt_request(self, request: Request) -> None:
        """
        对应源码：Scheduler._preempt_request()

        抢占流程（3 步）：
          1. kv_cache_manager.free(request) → 释放 KV Cache block，归还空闲池
          2. request.status = PREEMPTED，num_computed_tokens = 0
             （KV 数据已丢失，下次必须从头计算）
          3. waiting.prepend(request) → 放回 waiting 队首，优先被恢复
        """
        freed = list(request.block_ids)
        self.kv_cache_manager.free(request)        # 1. 释放 block
        request.status = RequestStatus.PREEMPTED   # 2a. 修改状态
        request.num_computed_tokens = 0            # 2b. 重置计算进度
        request.num_preemptions += 1
        self.waiting.prepend(request)              # 3. 回 waiting 队首
        print(f"  ⚡ 抢占 {request.request_id}: "
              f"释放 blocks={freed}，放回 waiting 队首")

    # ─────────────────────────────────────────────────────────────────────────
    # 6.4 调度后状态更新
    # 对应源码：Scheduler._update_after_schedule()
    # ─────────────────────────────────────────────────────────────────────────

    def _update_after_schedule(self, output: SchedulerOutput) -> None:
        """
        对应源码：Scheduler._update_after_schedule()

        将本轮已调度的 token 数累加到每个请求的 num_computed_tokens：
          request.num_computed_tokens += num_scheduled_tokens[req_id]

        为什么在 GPU 计算完成前就更新？（乐观推进）
          调度器需在下一步开始时知道每个请求推进到哪里，
          以正确计算 num_new_tokens，避免重复调度相同 token。
          假设 GPU 一定会成功完成本轮计算（失败时由上层处理）。
        """
        for req_id, n in output.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n

    # ─────────────────────────────────────────────────────────────────────────
    # 6.5 处理 GPU 输出（模拟）
    # 对应源码：Scheduler.update_from_output()
    # ─────────────────────────────────────────────────────────────────────────

    def update_from_output(self, new_token_per_req: dict[str, int]) -> list[str]:
        """
        模拟 GPU 执行完毕后，将新生成的 output token 写回请求，并判断是否完成。

        真实实现：ModelRunner 执行 → EngineCoreOutput → Scheduler.update_from_output()
        返回：本轮完成的请求 ID 列表。

        【重要】请求完成后必须释放 KV Cache block，否则显存永远无法归还，
        后续等待中的请求（包括被抢占后恢复的请求）将因显存不足而无法调度。
        对应真实源码：Scheduler.free_request() → kv_cache_manager.free(request)
        """
        finished_ids = []
        for req_id, new_token in new_token_per_req.items():
            request = self.requests.get(req_id)
            if request is None or request.status != RequestStatus.RUNNING:
                continue
            request.output_token_ids.append(new_token)
            if request.num_output_tokens >= request.max_tokens:
                request.status = RequestStatus.FINISHED
                self.running.remove(request)
                # 释放 KV Cache block，将显存归还给空闲池
                freed = list(request.block_ids)
                self.kv_cache_manager.free(request)
                finished_ids.append(req_id)
                print(f"  [完成] {req_id}: "
                      f"prompt={request.num_prompt_tokens}, "
                      f"output={request.num_output_tokens}, "
                      f"preemptions={request.num_preemptions}, "
                      f"freed_blocks={freed}")
        return finished_ids


# ─────────────────────────────────────────────────────────────────────────────
# 7. 演示场景
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'█'*64}")
    print(f"  {title}")
    print(f"{'█'*64}")


def _run_loop(scheduler: Scheduler, max_steps: int = 12) -> None:
    """共用推理主循环：schedule → GPU 模拟（非 prefill 请求产生 output token）→ update。"""
    for _ in range(max_steps):
        if not scheduler.has_requests():
            break
        scheduler.schedule()
        # 模拟 GPU：decode 阶段请求各产生 1 个 output token（固定 token ID=1）
        new_tokens = {
            req.request_id: 1
            for req in scheduler.running
            if not req.is_prefill
        }
        scheduler.update_from_output(new_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# 场景一：基础调度流程（FCFS + chunked prefill）
# ──────────────────────────────────────────────────────────────────────────────

def demo_basic():
    """
    配置：token_budget=6，block_size=4，total_blocks=20

    演示要点：
      - Step 1: token_budget=6，req-A(3tok) + req-B(3tok) 恰好填满，req-C 留 waiting
      - Step 2: running 队列优先处理（req-A/B decode 各 1tok），
                随后接纳 req-C 进 prefill
      - req-B 比 req-A 先完成（max_tokens=1）
      - 全程无抢占，展示正常 prefill→decode 生命周期
    """
    _section("场景一：基础调度流程 (FCFS + token_budget 限制)")
    kv = SimplifiedKVCacheManager(total_blocks=20, block_size=4)
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=6,  # 每步最多 6 个 token
        kv_cache_manager=kv,
        policy=SchedulingPolicy.FCFS,
    )
    print()
    sched.add_request(Request("req-A", list(range(3)), max_tokens=3))
    sched.add_request(Request("req-B", list(range(3)), max_tokens=1))
    sched.add_request(Request("req-C", list(range(5)), max_tokens=2))  # 长 prompt，展示 chunked prefill
    _run_loop(sched)


# ──────────────────────────────────────────────────────────────────────────────
# 场景二：KV Cache 不足触发抢占（FCFS）
# ──────────────────────────────────────────────────────────────────────────────

def demo_preemption_fcfs():
    """
    配置：token_budget=16，block_size=3，total_blocks=4（总容量 12 个 token）

    抢占触发逻辑：
      Step 1: req-X(3tok) + req-Y(3tok) + req-Z(3tok) 各占 1 block，共用 3 blocks
              还剩 1 个 free block。
      Step 2 (decode):
        - req-X 已有 3 tok（满 1 block），decode 变 4 tok → 需 ceil(4/3)=2 blocks，
          当前 1 block，需新增 1 → 消耗最后 1 free block → free=0
        - req-Y 同样要新增 1 block → free=0，allocate 失败！
          → FCFS 抢占 running 末尾：req-Z 被踢出，释放 1 block
          → req-Y 重新分配成功
        - req-Z 被放回 waiting 队首
      Step 3: req-Z 恢复执行（PREEMPTED → RUNNING）

    关键结论：
      - FCFS 策略保护越早进入 running 的请求，牺牲越晚的请求
      - 被抢占请求 num_computed_tokens 归零，下次重头计算
    """
    _section("场景二：KV Cache 不足触发抢占 (FCFS)")
    kv = SimplifiedKVCacheManager(total_blocks=4, block_size=3)
    sched = Scheduler(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=16,
        kv_cache_manager=kv,
        policy=SchedulingPolicy.FCFS,
    )
    print()
    sched.add_request(Request("req-X", list(range(3)), max_tokens=3))
    sched.add_request(Request("req-Y", list(range(3)), max_tokens=3))
    sched.add_request(Request("req-Z", list(range(3)), max_tokens=3))
    _run_loop(sched)


# ──────────────────────────────────────────────────────────────────────────────
# 场景三：Priority 调度策略（接纳顺序 + 抢占顺序由 priority 决定）
# ──────────────────────────────────────────────────────────────────────────────

def demo_priority():
    """
    配置：token_budget=5，block_size=2，total_blocks=5

    演示要点：
      - 三个请求按 low → high → med 顺序加入 waiting
      - Priority 策略：waiting 队列按 priority 排序，high(1) 最先调度
      - 若显存不足，Priority 策略抢占 priority 最大（最低优先级）的请求
        而不是 running 队列末尾的请求（与 FCFS 的区别）
    """
    _section("场景三：Priority 调度策略")
    kv = SimplifiedKVCacheManager(total_blocks=5, block_size=2)
    sched = Scheduler(
        max_num_running_reqs=3,
        max_num_scheduled_tokens=5,
        kv_cache_manager=kv,
        policy=SchedulingPolicy.PRIORITY,
    )
    print()
    print("  加入顺序：low(priority=10) → high(priority=1) → med(priority=5)")
    sched.add_request(Request("req-low",  list(range(2)), max_tokens=2, priority=10))
    sched.add_request(Request("req-high", list(range(2)), max_tokens=2, priority=1))
    sched.add_request(Request("req-med",  list(range(2)), max_tokens=2, priority=5))
    _run_loop(sched)


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_basic()
    demo_preemption_fcfs()
    demo_priority()
