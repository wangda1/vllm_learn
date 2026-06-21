"""
dp_wave_coordinator_demo.py
============================
教学示例 1：vLLM 数据并行（DP）中的核心概念
                —— 多卡 wave 与 Coordinator 之间的协调

本示例不依赖 GPU、不依赖 vLLM，用线程 + 队列模拟真实的三大组件：

    AsyncLLM (前端)  <--ZMQ-->  Coordinator (中枢)  <--ZMQ-->  Engine x N (后端)

想讲清楚的 4 件事（对应 DOC.md 第一章 1.2 节）：

  1. 请求走直连，控制信号走中转：
       请求推理：     AsyncLLM ---------------------> Engine        (不经过 Coordinator)
       Start Wave：   AsyncLLM --> Coordinator --> Engine           (Coordinator 中转广播)

  2. wave（波次）是什么：
       wave 不是把一批请求打包成 batch，而是一个“全局轮次编号”，
       让所有 Engine 对“现在处于第几轮协同执行”保持一致。
       wave 数 = 所有 Engine 从 running 切换到 paused 的次数。

  3. 唤醒流程（FIRST_REQ -> START_DP_WAVE）：
       前端发现后端 engines_running==False（已暂停），
       先给 Coordinator 发 FIRST_REQ，Coordinator 置 engines_running=True，
       并广播 START_DP_WAVE，唤醒所有 Engine 进入新一轮 wave。

  4. all-reduce 步调同步 + dummy batch + wave_complete：
       即使某个 Engine 本轮没有真实请求，也要执行 dummy pass，
       以保证所有 Engine 都参与 all-reduce、步调一致；
       当 all-reduce 确认“全局没有任何未完成请求”时，
       各 Engine 同时进入 paused，wave += 1，等待下一次 FIRST_REQ。

运行：
    python dp_wave_coordinator_demo.py
"""

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional

# 全局打印锁，避免多线程日志交错
_PRINT_LOCK = threading.Lock()


def log(who: str, msg: str):
    with _PRINT_LOCK:
        # 用前缀对齐，方便阅读不同组件的日志
        print(f"  [{who:<14}] {msg}", flush=True)


# ============================================================
# 消息类型：模拟 vLLM 中通过 ZMQ 传递的几种关键消息
# ============================================================
@dataclass
class Msg:
    kind: str                  # FIRST_REQ / START_DP_WAVE / REQUEST / WAVE_COMPLETE / STATS
    wave: int = 0              # 该消息关联的 wave 号
    payload: object = None     # 附带数据：请求体、被排除的 engine、负载快照等


# ============================================================
# Engine：后端执行组件（每个 DP rank 一个进程，这里用一个线程模拟）
# 对应 DOC.md 中的 DPEngineCoreProc
# ============================================================
class Engine(threading.Thread):
    def __init__(self, index: int, num_engines: int, coord: "Coordinator"):
        super().__init__(daemon=True)
        self.index = index
        self.num_engines = num_engines
        self.coord = coord

        # 该 Engine 的输入队列：AsyncLLM 直连发请求 / Coordinator 发 START_DP_WAVE
        self.inbox: "queue.Queue[Msg]" = queue.Queue()

        # —— DP 关键状态 ——
        self.current_wave = 0          # 本 Engine 认为的当前 wave
        self.engines_running = False   # 是否处于 running（False = paused/idle）

        # 本地调度队列：模拟 scheduler 的 waiting / running
        self.waiting: List[str] = []   # 待处理请求
        self.running: List[List[str]] = []  # [ [req, 剩余step数], ... ]

        self._stop = False

    # 前端直连：把请求放入本 Engine 的输入队列
    def submit_request(self, msg: Msg):
        self.inbox.put(msg)

    # ---- 处理来自队列的控制/请求消息 ----
    def _drain_inbox(self):
        try:
            while True:
                msg = self.inbox.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

    def _handle_msg(self, msg: Msg):
        if msg.kind == "START_DP_WAVE":
            # 对应 DOC：收到 START_DP_WAVE 后的处理逻辑
            new_wave, exclude_idx = msg.wave, msg.payload
            # 如果自己不是被排除的那个 engine，且 new_wave 不小于当前 wave，则唤醒
            if exclude_idx != self.index and new_wave >= self.current_wave:
                self.current_wave = new_wave
                if not self.engines_running:
                    log(f"Engine-{self.index}",
                        f"EngineCore starting idle loop for wave {new_wave}  (被 Coordinator 唤醒)")
                    self.engines_running = True

        elif msg.kind == "REQUEST":
            # 前端直连发来的真实请求；附带前端打上的 wave 号
            req = msg.payload
            self.waiting.append(req)
            log(f"Engine-{self.index}", f"收到请求 {req!r} (req.wave={msg.wave})，加入 waiting 队列")
            # 收到真实请求就应进入 running（若此前是 paused）
            if not self.engines_running:
                self.engines_running = True

    # ---- 执行一个 step：把 waiting 调度进 running，再推进 running ----
    def _process_engine_step(self) -> bool:
        # 简化的连续批处理：每个 step 把 waiting 全部纳入 running
        while self.waiting:
            req = self.waiting.pop(0)
            self.running.append([req, 5])  # 假设每个请求需要 5 个 step（decode）才完成

        if not self.running:
            return False  # 本地没有任何请求在跑

        still = []
        for req, remain in self.running:
            remain -= 1
            if remain > 0:
                still.append([req, remain])
            else:
                log(f"Engine-{self.index}", f"请求 {req!r} 完成 ✔")
        self.running = still
        return True

    def _has_local_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ---- run_busy_loop：DP Engine 的核心循环（对应 DOC 1.2.2 / 第二章）----
    def run(self):
        while not self._stop:
            self._drain_inbox()

            # paused 状态：没被唤醒就空转等待，跳过 dummy pass 和 all-reduce
            if not self.engines_running:
                time.sleep(0.02)
                continue

            # 1) 执行本地请求；即使没有请求也要执行 dummy pass，保证 all-reduce 对齐
            executed = self._process_engine_step()
            if not executed:
                log(f"Engine-{self.index}",
                    "本地无请求 -> 执行 dummy batch（仅为对齐 all-reduce 通信）")

            # 2) all-reduce 同步全局是否仍有未完成请求
            #    （真实 vLLM 是 torch.distributed.all_reduce；这里委托 Coordinator 做聚合）
            local_unfinished = self._has_local_unfinished()
            global_unfinished = self.coord.all_reduce_unfinished(self.index, local_unfinished)

            # 3) 上报负载快照 [waiting, running] 给 Coordinator（供前端做负载均衡）
            self.coord.report_stats(self.index, len(self.waiting), len(self.running))

            if not global_unfinished:
                # 全局都没有未完成请求 -> 本轮 wave 结束，进入 paused
                if self.index == 0:
                    # 约定：只有 rank 0 向 Coordinator 报告 wave_complete
                    log("Engine-0", f"全局已无未完成请求 -> 上报 WAVE_COMPLETE(wave={self.current_wave})")
                    self.coord.on_wave_complete(self.current_wave)
                self.current_wave += 1        # 推进到下一个 wave
                self.engines_running = False  # 暂停
                log(f"Engine-{self.index}",
                    f"进入 paused，等待下一次唤醒；下一轮 current_wave={self.current_wave}")

            time.sleep(0.03)  # 放慢节奏，便于观察

    def stop(self):
        self._stop = True


# ============================================================
# Coordinator：中枢协调组件（对应 DOC 中的 DPCoordinatorProc）
#   - 接收 FIRST_REQ，广播 START_DP_WAVE
#   - 聚合各 Engine 的 all-reduce（全局是否有未完成请求）
#   - 汇总负载快照 counts，发布给前端
# ============================================================
class Coordinator:
    def __init__(self, num_engines: int):
        self.num_engines = num_engines
        self.engines: List[Engine] = []

        self.current_wave = 0
        self.engines_running = False

        # 负载快照：每个 engine 一个 [waiting, running]
        self.counts = [[0, 0] for _ in range(num_engines)]

        # all-reduce 用：收集本轮各 engine 上报的 local_unfinished
        self._reduce_lock = threading.Lock()
        self._reduce_votes = {}

    def bind_engines(self, engines: List[Engine]):
        self.engines = engines

    # ---- 前端 -> Coordinator：收到 FIRST_REQ，广播 START_DP_WAVE ----
    def handle_first_req(self, wave: int, chosen_engine: int):
        log("Coordinator", f"收到 FIRST_REQ (chosen_engine={chosen_engine}, wave={wave})")
        self.engines_running = True
        # 排除掉前端已经直连发过请求的那个 engine（它会自己进入 running）
        self._send_start_wave(wave, exclude_engine_index=chosen_engine)

    def _send_start_wave(self, wave: int, exclude_engine_index: Optional[int]):
        log("Coordinator",
            f"广播 START_DP_WAVE(wave={wave}, exclude={exclude_engine_index}) 给所有 Engine")
        for eng in self.engines:
            eng.submit_request(Msg(kind="START_DP_WAVE", wave=wave, payload=exclude_engine_index))

    # ---- all-reduce：聚合所有 engine 的 local_unfinished -> 全局 OR ----
    # 真实 vLLM 用 torch.distributed.all_reduce；这里用“收齐所有 rank 的投票”模拟同步屏障。
    def all_reduce_unfinished(self, engine_index: int, local_unfinished: bool) -> bool:
        with self._reduce_lock:
            self._reduce_votes[engine_index] = local_unfinished
        # 自旋等待，直到本轮所有 engine 都投了票（模拟 all-reduce 的同步语义）
        while True:
            with self._reduce_lock:
                if len(self._reduce_votes) >= self.num_engines:
                    result = any(self._reduce_votes.values())
                    # 最后一个进来的把票箱清空，开始下一轮
                    # 用 engine_index 标记简单处理：所有人读到同一 result
                    return result
            time.sleep(0.005)

    # 每个 wave 收尾时清空投票箱（由 on_wave_complete 触发）
    def _reset_votes(self):
        with self._reduce_lock:
            self._reduce_votes.clear()

    # ---- Engine -> Coordinator：上报负载快照 ----
    def report_stats(self, engine_index: int, waiting: int, running: int):
        self.counts[engine_index] = [waiting, running]

    # ---- Engine-0 -> Coordinator：wave 完成 ----
    def on_wave_complete(self, wave: int):
        self.engines_running = False
        self.current_wave = wave + 1
        self._reset_votes()
        log("Coordinator",
            f"收到 WAVE_COMPLETE -> engines_running=False, current_wave 推进到 {self.current_wave}")

    # ---- Coordinator -> 前端：发布负载快照 ----
    def publish_to_frontend(self):
        return (list(self.counts), self.current_wave, self.engines_running)


# ============================================================
# AsyncLLM：前端请求入口（对应 DOC 中的 DPAsyncMPClient / AsyncLLM）
#   - 维护本地缓存的 current_wave / engines_running
#   - 内置负载均衡：按 counts 选最空闲的 engine
#   - 若后端已暂停，先发 FIRST_REQ 唤醒
#   - 请求直连发给目标 Engine
# ============================================================
class AsyncLLM:
    def __init__(self, coord: Coordinator, engines: List[Engine]):
        self.coord = coord
        self.engines = engines
        # 前端本地缓存（通过订阅 Coordinator 的发布信息更新）
        self.current_wave = 0
        self.engines_running = False
        self.lb_counts = [[0, 0] for _ in engines]

    # 模拟“订阅 Coordinator 的发布信息并刷新本地缓存”
    def refresh_from_coordinator(self):
        counts, wave, running = self.coord.publish_to_frontend()
        self.lb_counts = counts
        self.current_wave = wave
        self.engines_running = running
        log("AsyncLLM",
            f"订阅刷新: counts={counts}, wave={wave}, running={running}")

    # 内置负载均衡：选 (waiting+running) 最小的 engine
    def _select_engine(self) -> int:
        loads = [w + r for (w, r) in self.lb_counts]
        chosen = loads.index(min(loads))
        return chosen

    def send_request(self, req: str):
        self.refresh_from_coordinator()
        chosen = self._select_engine()
        log("AsyncLLM",
            f"为请求 {req!r} 选中 Engine-{chosen} (依据负载 counts={self.lb_counts})")

        # 关键：若后端处于 paused，先发 FIRST_REQ 唤醒，再直连发请求
        if not self.engines_running:
            log("AsyncLLM", "发现后端 engines_running=False -> 先发送 FIRST_REQ 唤醒")
            self.coord.handle_first_req(self.current_wave, chosen)
            self.engines_running = True

        # 请求直连发给目标 Engine（不经过 Coordinator），并打上当前 wave 号
        self.engines[chosen].submit_request(
            Msg(kind="REQUEST", wave=self.current_wave, payload=req))


# ============================================================
# 主流程：搭起 1 个前端 + 1 个 Coordinator + N 个 Engine，发几条请求观察 wave 协调
# ============================================================
def main():
    NUM_ENGINES = 3
    print("=" * 70)
    print(f"DP 核心概念演示：{NUM_ENGINES} 个 Engine（DP rank）+ Coordinator + AsyncLLM")
    print("=" * 70)

    coord = Coordinator(NUM_ENGINES)
    engines = [Engine(i, NUM_ENGINES, coord) for i in range(NUM_ENGINES)]
    coord.bind_engines(engines)
    front = AsyncLLM(coord, engines)

    for e in engines:
        e.start()

    print("\n--- 阶段 1：初始状态，所有 Engine 都是 paused（engines_running=False）---\n")
    time.sleep(0.2)

    print("\n--- 阶段 2：发送第 1 条请求，触发 FIRST_REQ -> START_DP_WAVE 唤醒全员 ---\n")
    front.send_request("写一首关于秋天的诗")
    time.sleep(0.6)

    print("\n--- 阶段 3：再连发 2 条请求，负载均衡会分发到不同 Engine ---\n")
    front.send_request("用一句话介绍 vLLM")
    time.sleep(0.05)
    front.send_request("解释什么是 PagedAttention")
    time.sleep(0.8)

    print("\n--- 阶段 4：请求全部处理完 -> all-reduce 确认全局空闲 -> wave++ -> 全员 paused ---\n")
    time.sleep(0.8)

    print("\n--- 阶段 5：稍后再发 1 条请求，会开启新的一轮 wave（wave 号已递增）---\n")
    front.send_request("第二轮的新请求")
    time.sleep(0.8)

    for e in engines:
        e.stop()
    time.sleep(0.2)

    print("\n" + "=" * 70)
    print("小结：")
    print("  1) 请求直连 Engine；唤醒信号 START_DP_WAVE 由 Coordinator 中转广播。")
    print("  2) wave 是全局轮次编号，所有 Engine 靠它保持步调一致。")
    print("  3) 即使某 Engine 本轮无请求，也要做 dummy batch 以对齐 all-reduce。")
    print("  4) all-reduce 确认全局无未完成请求后，全员同时 paused 且 wave += 1。")
    print(f"  最终 Coordinator.current_wave = {coord.current_wave}")
    print("=" * 70)


if __name__ == "__main__":
    main()
