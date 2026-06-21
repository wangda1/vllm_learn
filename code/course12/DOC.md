一、DP 原理
1.1 DP 概述
当用户通过 --data-parallel-size N 启动 vLLM 服务时，系统会启动 N 个 DP Rank。对于 Dense 模型，可以近似将每个 DP Rank 理解为一个独立的模型副本实例：它们各自拥有一份模型权重、独立的 KV cache，以及各自的调度队列。API 服务器接收到推理请求后（除去external LB，也就是外部负载均衡模式），会根据不同的负载均衡模式，将请求路由到某个 DP Rank 处理。在内置负载均衡模式下，vLLM 会根据各 DP Rank 对应 Engine 的 waiting / running 队列状态分发请求；如果采用外部负载均衡模式，请求分发则由外部负载均衡器负责。但是需要注意的是，只有MOE模型才能采用外部负载均衡模式，具体原因下文中会讲。
所谓数据并行（DP, Data Parallel），通常是指将同一份模型权重复制到多个 DP Rank 上，由各个 Rank 分别处理一部分并发请求或一部分批次。在推理场景中，这种方式通常用于提升系统的并发处理能力。  
- 优点：实现相对简单，吞吐通常可以随着副本数增加而提升；  
- 缺点：每个副本都需要占用完整的模型权重和 KV cache（但是如果组内还有 TP/PP，这份副本会在组内切分），因此显存开销会随着 DP Rank 数量增加而上升。  
无论是在大模型训练还是推理中，数据并行都是一种常见的扩展手段；相较于训练阶段，推理阶段的数据并行逻辑通常更直接一些。在 vLLM 中，数据并行既可用于 Dense 模型，也可用于 MoE 模型；既支持单机部署，也支持 V1 架构下的多机分布式部署。需要注意的是，单机单卡本身并不属于真正的数据并行，只有当 data_parallel_size > 1 时，才真正启用了 DP。  
对于 DeepSeek V3 这类采用 MLA（多头潜在注意力）的模型，实践中通常会考虑这样的并行组合：注意力部分使用数据并行（DP），专家部分使用专家并行或张量并行（EP 或 TP）；在不少场景下，这样的搭配往往更有优势。
1.2 DP 逻辑框架图
[图片]
目前 vLLM 的数据并行（DP）方案逻辑框架如下图所示。整体结构上，系统主要由一个或多个 类似AsyncLLM 的前端，Coordinator以及多个 Engine Core 后端组成，三者之间通过 ZMQ 通信。
在典型的多节点部署中，主节点负责启动前端服务，并运行部分或全部 Engine Core；其他节点通常只需启动本地的 Engine Core。主节点前端根据不同节点上 Engine 的负载信息，将请求路由到某个合适的 Engine Core 处理；这些负载信息由 Coordinator 汇总并同步给 AsyncLLM。请求处理完成后，前端再将结果返回给用户。
Engine Core 由进程管理器CoreEngineProcManager负责拉起和管理。每个 DP Rank 对应一个EngineCore进程，它们通过 ZMQ 与前端及 Coordinator 建立连接，并接收前端分发的请求。在这种架构下，可以重点关注以下三个核心组件：
- AsyncLLM：前端请求入口，负责维护当前的 wave 编号，向目标 Engine 发送请求，并监听Coordinator发布的统计信息和 wave 状态更新。
- Coordinator：中枢协调组件，负责接收 Engine 上报的状态信息，汇总各个 DP Rank 的负载统计并同步给前端；在需要 wave 协调的场景下，还会广播 START_DP_WAVE 等控制消息，驱动相关 Engine 进入新一轮执行。
- Engine：后端执行组件，负责具体的调度与推理执行；它既处理前端发来的请求，也会上报运行状态，并接收 Coordinator 下发的 wave 控制和同步指令。
从下图中也可以看出这三个组件的关系：
[图片]
wave 可以理解为 vLLM 在数据并行场景下维护的一个编号。它的作用并不是单纯将一批请求打包成 batch，而是让多个 Engine Core 对“当前处于第几轮协同执行”保持一致，从而避免分布式场景下由于请求到达时序不同造成的状态错位。  
在需要 wave 协调的场景下，多个Engine Core 虽然不一定处理完全相同的请求集合，但需要在同一个 wave 上协同推进。可以将一个 wave 理解为一轮需要各个相关 Engine 保持同步节奏的执行周期：当一个 engine 收到新请求或 stale wave 请求时，Coordinator 通过 START_DP_WAVE 唤醒其他 engines；wave 完成后，rank 0 向 coordinator 报告 wave_complete，各 engine 增加 current_wave。
1.2.1 Coordinator 和 AsyncLLM 之间的通信
暂时无法在飞书文档外展示此内容
AsyncLLM → Coordinator
当新请求到达时，AsyncLLM 客户端首先检查从 Coordinator 通过 XSUB socket 订阅获得的全局状态信息engines_running。若该标志为False，表明后端引擎当前处于暂停状态，无法处理请求，此时 AsyncLLM 会主动通过内部通信通道发送一个特殊的FIRST_REQ控制信号给 Coordinator，用以触发引擎 Engine 的唤醒流程。
Coordinator 接收到FIRST_REQ后，将engines_running置为True，并广播START_DP_WAVE命令至所有Engine节点，通知它们启动并开始处理新一轮的请求波次（wave）。下面是AsyncLLM发送FIRST_REQ的流程：
这里的触发条件是客户端收到了一个新的请求，但是客户端发现现在的Engine并没有启动，也就是not self.engines_running。
class DPAsyncMPClient(AsyncMPClient):
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()
    
        request.current_wave = self.current_wave
        request.client_index = self.client_index
        # 选择当前状态最好的一个engine
        chosen_engine = self.get_core_engine_for_request(request)
        # 先把请求发给选中的 engine
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        # 如果全局 DP engines 处于 paused 状态，则通知唤醒
        if not self.engines_running:
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)
    
            await to_await

            self._ensure_output_queue_task()
如果在此时向推理框架发送一个请求，通常会观察到类似下面的输出。需要特别关注 EngineCore starting idle loop 这条日志：它表明 EngineCore 已收到来自 Coordinator 的 START_DP_WAVE 指令，并从 paused 状态切换回 running 状态，开始当前 wave 的 idle loop。
(APIServer pid=27732) dp_engines_running False
(EngineCore_DP0 pid=27907) DEBUG 11-18 15:50:28 [v1/engine/core.py:743] EngineCore loop active.
(EngineCore_DP0 pid=27907) DEBUG 11-18 15:50:28 [v1/engine/core.py:1016] EngineCore starting idle loop for wave 0.
(EngineCore_DP1 pid=27908) DEBUG 11-18 15:50:28 [v1/engine/core.py:1016] EngineCore starting idle loop for wave 0.
(EngineCore_DP1 pid=27908) DEBUG 11-18 15:50:28 [v1/engine/core.py:743] EngineCore loop active.
EngineCore starting idle loop 这条日志表明 EngineCore 接收到了来自 Coordinator 的启动指令，即将进入运行状态。具体来说，该行为由 DPEngineCoreProc._handle_client_request 方法处理。当请求类型为EngineCoreRequestType.START_DP_WAVE 时，EngineCore(DPEngineCoreProc)会接收一个新的wave并将自身的运行状态engines_running置为true。这说明该 EngineCore 已经被 Coordinator 唤醒，重新进入当前 wave 的运行状态。
暂时无法在飞书文档外展示此内容
现在数据链路是已经很清晰了，只是此时我们尚未涉及Coordinator如何向Engine发送START_DP_WAVE指令，也未展开其如何接收来自前端的 FIRST_REQ 信号。
在这里，我们再来回顾一下 Wave 的概念，Wave 数就是所有 Engine 从运行状态切换到暂停状态的次数，它用于所有Engine的暂停 → 启动 → 运行 → 完成 → 暂停。以下在 Engine 中，接收到来自 Coordinator 中的 START_DP_WAVE 信号后，将 engines_running 状态置为 true。
下面这段 Engine 侧代码展示了接收到 START_DP_WAVE 后的处理逻辑。
- Engine 收到来自 Coordinator 的 START_DP_WAVE 后，首先检查自身是否就是被排除的那个 engine；如果不是，则判断 new_wave 是否不小于当前 current_wave。满足这些条件后，它会更新自己的 current_wave；
- 若当前仍处于 paused 状态（即 self.engines_running == False），则会打印 EngineCore starting idle loop 日志，并将 engines_running 置为 True，表示该 Engine 已被唤醒，进入当前 wave 的运行状态。
class DPEngineCoreProc(EngineCoreProc):
    def _handle_client_request(self, request_type: EngineCoreRequestType,
                              request: Any) -> None:
        # 接收到来自Coodinator的指令
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            # 通过编号检查自身是否需要启动。
            if exclude_eng_index != self.engine_index and (new_wave >= self.current_wave):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
Coordinator→AsyncLLM 
状态的上报
暂时无法在飞书文档外展示此内容
这条通路的主要作用就是 Coordinator 向 AsyncLLM 发送后端负载，让前端通过订阅这个地址以获取负载信息和 request wave 变化，同时也及时让 AsyncLLM 自己内部更新对 wave 号的缓存，以便它在收到这些消息后，会及时更新本地缓存的 current_wave 和 engines_running，这样下次向 Engine 发送请求时，就能携带正确的 wave 号,也就是说Coordinator向前端发布的是以下的三个信息：to_publish = (engine_req_counts_list, current_wave, engines_running) ，其中：
- current_wave：当前全局 DP request wave 编号。每当所有 engines 从 running 转为 paused 完成一轮后，该编号就会推进，用于标记系统当前所处的 wave 轮次。  
- engine_req_counts_list：各个 EngineCore 当前的请求负载信息。每个元素是一个 [waiting, running] 对，分别表示等待中的请求数和运行中的请求数。  
- engines_running：全局 engines 的运行状态。若为 True，表示当前 DP engines 整体处于 running 阶段；若为 False，表示系统已进入 paused 状态。
我们看看在Coordinator是怎么发送到AsyncLLM中，同时也看看AsyncLLM接收到订阅信息之后是怎么更新自身状态的。从下方的代码可以看出Coordinator设置了一个超时时间，每间隔一定间隔或者有状态改变stats_changed就向前端发送信息，也就是to_publish = (engine_req_counts_list, current_wave, engines_running)。
class DPCoordinatorProc:
    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
    ):
       while True:
            elapsed = int(time.time() * 1000) - last_publish_time
            # Send at stats_update_interval_ms interval if the stats have
            # changed, or otherwise every 5 seconds.
            wait_for = self.stats_update_interval_ms if stats_changed else 5000

            # Wait at least 50ms to ensure we've received all stats for
            # the current step.
            min_timeout = 50 if last_step_counts is None else 0

            events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
            if not events:
               # Poller timeout - publish current stats to front-ends.
               if last_step_counts is not None:
                  engine_req_counts_list = last_step_counts
                  last_step_counts = None
               else:
                  engine_req_counts_list = self._get_engine_counts()
                  stats_changed = False
               # 向前端coordinate发布的信息
               to_publish = (engine_req_counts_list, current_wave, engines_running)
               publish_front.send(msgspec.msgpack.encode(to_publish))
               last_publish_time = int(time.time() * 1000)
               continue

现在看看 AsyncLLM 在接收到上报的负载信息后是怎么更新的，我们这里打印了这counts信息目的就是为了看看每个engine上的负载信息，在这里加了断点之后我们又向推理框架发送了一条请求。
class DPAsyncMPClient(AsyncMPClient):
    async def run_engine_stats_update_task():
          ...
          ...
          counts, wave, running = msgspec.msgpack.decode(buf)
          print(f"counts: {counts}, wave: {wave}, running: {running}")
          self.current_wave = wave
          self.engines_running = running
在这里，每次从 Coordinator 订阅到一条 stats 信息，就刷新前端客户端的 current_wave、engines_running 和负载均衡用的 lb_engines，从下方打印的counts: [[0, 1], [0, 0]], wave: 1, running: True就可以看出，在当前这个快照时刻下，只有Engine-0有一个请求正在被执行，而且该请求是running状态的。
注意这里的lb_engines为后续请求的负载均衡至关重要，具体见2.2节。
 curl -s --noproxy '*' http://127.0.0.1:13333/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' 
  
counts: [[0, 1], [0, 0]], wave: 1, running: True
这里的 counts 是前端从 Coordinator 收到的一份 Engine 负载快照。每个元素都是一个 [waiting, running] 二元组，因此 [[0, 1], [0, 0]] 表示：
- Engine-0 当前有 0 个 waiting 请求、1 个 running 请求
- Engine-1 当前有 0 个 waiting 请求、0 个 running 请求
也就是说，在这个快照时刻下，只有 Engine-0 正在执行 1 个请求。日志中也会输出：
(APIServer pid=12366) DEBUG 01-04 15:30:24 [v1/engine/core_client.py:1145] Received counts: [[0, 0], [0, 0]] (slice(0, 2, None))
(APIServer pid=12366) DEBUG 01-04 15:30:29 [v1/engine/core_client.py:1145] Received counts: [[0, 0], [0, 0]] (slice(0, 2, None))
1.2.2 Engine 和 Coordinator 之间的通信
暂时无法在飞书文档外展示此内容
之前分析过，虽然 AsyncLLM 会直接将推理请求（Request）发送给 Engine，但启动 wave 的控制信号并不是直接发给 Engine 的。AsyncLLM 会先通知 Coordinator（发送 FIRST_REQ），再由 Coordinator 通过 _send_start_wave 函数向所有 Engine 广播 START_DP_WAVE 信号来统一启动。下文中我们要重点探究的，就是 Coordinator 发送这个启动信号的具体时机。
这样就非常清晰了，这里有一点很容易弄错，那就是请求的推理是不经过Coordinator的。
请求：AsyncLLM -> Engine (直连)
控制信号 (Start Wave)：AsyncLLM -> Coordinator -> Engine (中转)
class DPCoordinatorProc:
    @staticmethod
    def _send_start_wave(socket: zmq.Socket, wave: int, exclude_engine_index: int | None):
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))
_send_start_wave发送的时机有两个：
1. 第一个就是AsyncLLM 发送的是 ("FIRST_REQ", chosen_engine)；Coordinator 收到 "FIRST_REQ" 后，直接广播向Engine发送控制消息START_DP_WAVE。
2. 第二个就是Engine 在暂停状态下收到“过期 wave 的请求，就会反过来跟Coordinator说，我现在发现了一个过期的请求，它属于上一个wave，但是我们现在的引擎关了，你看看情况是不是要重启所有的Engine来处理可能迟到的请求。
  因为在Engine处理新请求的时候，同样还会给Coordinator发送现在该请求对应的wave号，如果Coordinator发送过来的wave号不对，也就是虽然是当前wave但是engine已经停了，或者就是发送过来一个未来的wave，见如下的条件。如果符合这两种情况，就需要向所有的Engine发送这一轮的启动信号。
注意这边的两个时机的区别：
第一个时机是从AsyncLLM中通过publish_front获取的，更像一个从上到下的指令，或者说这是一种标准的唤醒流程；
第二个时机则是来自Engine的返回，更多是对Engine上报的异常情况的处理，通过output_back获取，也就是说Engine在休眠时意外收到了请求，或者收到的请求 Wave 号不对。
- wave > current_wave：表示某个 Engine 已收到属于下一轮波次的请求，而 Coordinator 仍停留在当前波次，因此需要由 Coordinator 跟进更新 wave，并向其他 Engine 广播启动信号。  
- wave == current_wave：表示某个 Engine 已收到当前波次的请求，但 Coordinator 尚未唤醒其余 Engine；此时需由 Coordinator 补发启动信号，使所有 Engine 进入同一轮 wave 的运行状态。
class DPCoordinatorProc:
    def process_input_socket(...):
        while True:
            ...
            ...
            # 客户端发现engine没启动，委托Coordinate进行广播，启动所有的engine
            if not engines_running:
                if wave < current_wave:
                    # If the wave number is stale, ensure the message
                    # is handled by all the engines.
                    engine_to_exclude = None
            
                    engines_running = True
                    wave_state_changed = True
                    self._send_start_wave(
                        publish_back, current_wave, engine_to_exclude
                    )
            
            
            ...
            ...
            elif (wave := outputs.start_wave) is not None and (
                wave > current_wave
                or (wave == current_wave and not engines_running)
            ):
                current_wave = wave
                engines_running = True
                wave_state_changed = True
                self._send_start_wave(publish_back, wave, eng_index)
(APIServer pid=27732) dp_engines_running False
(EngineCore_DP0 pid=27907) DEBUG 11-18 15:50:28 [v1/engine/core.py:743] EngineCore loop active.
(EngineCore_DP0 pid=27907) DEBUG 11-18 15:50:28 [v1/engine/core.py:1016] EngineCore starting idle loop for wave 0.
(EngineCore_DP1 pid=27908) DEBUG 11-18 15:50:28 [v1/engine/core.py:1016] EngineCore starting idle loop for wave 0.
(EngineCore_DP1 pid=27908) DEBUG 11-18 15:50:28 [v1/engine/core.py:743] EngineCore loop active.
1.2.3 Engine之间的同步
从上文中我们知道，前端 (AsyncLLM) 通过 first_req_send_socket 发送 "FIRST_REQ" 信号给 Coordinator，Coordinator 则通过 publish_front (XPUB) 定期向前端广播引擎负载统计 (engine_req_counts_list, current_wave, engines_running)，前端据此做负载均衡选引擎，并打上当前 wave 号发送请求。见本章2.2节。
而在后端，多个 EngineCore 通过通过在run_busy_loop中执行torch.distributed.all_reduce() 同步全局是否还有未完成请求状态。它的目的是为了让不同Engine执行的步调一致，一旦所有 Engine 都执行完了各自的请求，它们同时进入暂停状态，wave 号递增，等待下一批 FIRST_REQ 启动。我们分析一下以下的历程：
暂时无法在飞书文档外展示此内容
1. executed = self._process_engine_step() 首先就是本地Engine需要执行自己对应的请求，即便没有请求，也要执行 dummy pass，这里的目的是为了保证 all-reduce 对齐，尤其是我们在本地的测试中，一次只发送一个请求，那么势必有一个Engine是空闲的；
2. 随后通过_has_global_unfinished_reqs() 同步全局状态。这里会周期性调用 ParallelConfig.sync_dp_state()，并在其中执行 torch.distributed.all_reduce()，这个 all-reduce 聚合的是是否还有任意 rank 存在未完成请求；
3. dp_rank == 0，发送 wave_complete 信号，用来确定当前时刻是否所有 Engine 都执行完了，那么下一个时刻就要进入到新的wave当中了，所以这里需要将这个信号wave_complete告知给coordinator方便它也进行升级。
4. 随后就是推进到下一个wave，Engine中将wave号加1，Coordinate在收到信号之后不仅需要自己进行升级，还需要通过publish_front通知给前端让前端也进行升级，这样一来就防止出现三者不同步的情况。也就是说，这里完成了三方同步闭环：Engine 升级 -> 通知 Coordinator -> Coordinator 升级 -> 通知 AsyncLLM。
另外有部分零散的知识点：
1. 负载上报与前端选 Engine
run_busy_loop() 执行过程中会按需上报负载。EngineCore 在 _maybe_publish_request_counts() 中读取 scheduler 的 waiting / running 请求数；当统计发生变化且允许发布 DP LB stats 时，会通过 EngineCoreOutputs(scheduler_stats=...) 上报给 Coordinator。
Coordinator 更新各 Engine 的负载表后，通过 publish_front 将 (engine_req_counts_list, current_wave, engines_running) 转发给前端。在内部负载均衡模式下，前端根据 waiting / running 计算分数：waiting * 4 + running，选择分数最低的 Engine。选中后，前端还会临时增加本地 waiting count，以缓解统计更新间隔带来的请求倾斜。
2. idle / paused 状态下的 continue
当本轮没有执行真实请求、本地也没有未完成请求，且 engines_running == False 时，说明当前 DP wave 已处于 idle / paused 状态。这里不是 Engine 进程关闭，而是当前没有需要参与的 wave。
此时 run_busy_loop() 会直接 continue，跳过 dummy pass 和后续 all-reduce，回到循环开头继续轮询输入队列，等待下一批请求或 START_DP_WAVE 唤醒。
class DPEngineCoreProc(EngineCoreProc):
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) 获取请求 Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) 执行请求 Step the engine core.
            executed = self._process_engine_step()
            # 主动上报负载统计：如果 [waiting, running] 数量有变，通知 Coordinator。
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) 同步 All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    # 与Coordinate发送同步信号，让当前的wave前进一个单位
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # 推进到下一个wave Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0
DP 的 engine core 需要做一些定制化改造，其实现在 DPEngineCoreProc 类中，继承关系为 DPEngineCoreProc -> EngineCoreProc -> EngineCore，代码实现位于 (vllm/v1/engine/core.py) 。
1. 请求数据的整体传递路径不变：其中 EngineCoreProc 类的初始化函数会创建两个线程 input_thread 和 output_thread，对应线程函数分别是 process_input_socket 和 process_output_socket，process_input_socket 线程函数负责接收来自 LLM 进程的输入请求，然后将其加入该进程的 input_queue 队列；线程process_output_socket 负责将 output_queue 中的结果通过 ZMQ 发送到 LLM 进程。
总结：输入和输出线程函数仍然通过是主线程通过队列（input_queue/output_queue）进行数据传递。
2. DP 调度的扩展（就是上文说的DPEngineCoreProc.run_busy_loop()）它在有了请求输入后进行调度，会做以下的几件事情：按需上报负载（_maybe_publish_request_counts()）、判断本地未完成请求、必要时 execute_dummy_batch()，再通过 all-reduce 判断全局是否仍有未完成请求。wave 结束时，dp_rank == 0 经 output_queue 发送 wave_complete，由输出线程转发给 Coordinator。
3. DP的EngineCore请求处理也分两层：基类 _handle_client_request() 处理 WAKEUP、ADD、ABORT、UTILITY 等通用类型；DP 子类额外拦截 START_DP_WAVE——消息未被排除且 wave 号不过旧时，更新 current_wave，必要时将 engines_running 置为 True，唤醒 Engine 进入对应 wave。
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs], bytes]]()
        # 省略代码 ... 
        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            # 省略代码 ... 
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(addresses.inputs, addresses.coordinator_input, identity, ready_event),
                daemon=True)
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(addresses.outputs, addresses.coordinator_output, self.engine_index),
                daemon=True)
            self.output_thread.start()  
1.2.4 Engine 发送【同步】异常信号
由于 DP（Data Parallelism）中异步架构的特性，以及网络传输和消息转发带来的延迟，Engine、Coordinator 与 AsyncLLM 三者之间的状态不一定在每个时刻都完全一致。AsyncLLM 会在发送请求时携带自己当前记录的 current_wave，而 Engine 收到请求后，会将这个 request_wave 与本地的 current_wave 进行比较。
因此，系统需要处理两类同步场景：
1. 正常同步：当前 wave 已经结束，Engine 通知 Coordinator 推进 wave。
2. 补偿同步：请求携带的 wave 与 Engine 当前 wave 不一致，需要通过 Coordinator 补发启动信号，使其他 Engine 跟上。
前文主要描述的是第一类：所有 Engine 完成当前 wave 的请求后，通过 DP 同步确认全局已经没有未完成请求，然后各自推进本地 wave。此时 dp_rank == 0 会发送 wave_complete，通知 Coordinator 当前 wave 已结束。
暂时无法在飞书文档外展示此内容
    class DPCoordinatorProc:
        def process_input_socket(
            self,
            front_publish_address: str,
            back_output_address: str,
            back_publish_address: str,
        ):
            if (wave := outputs.wave_complete) is not None:
                # 2. Notification from rank 0 engine that we've
                # moved into the global paused state
                # (engines_running==False).
                if current_wave <= wave:
                    new_wave = wave + 1
                    logger.debug(
                        "Moving DP wave from %d to %d.", current_wave, new_wave
                    )
                    # 1. coordinator更新自身的wave
                    current_wave = new_wave
                    engines_running = False
                    wave_state_changed = True
             
            # 2. 同步到前端
            if wave_state_changed:
                message = (None, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(message))

在系统运行过程中，可能存在一种典型但复杂的情况：前端（AsyncLLM）发出请求的速度与 Engine 的状态推进存在不一致，从而引入潜在的时序冲突。
具体而言，系统中可能运行着多个前端客户端（即多个 AsyncLLM 进程），每个进程独立维护自己的 current_wave 状态。当某个前端恰巧在 Engine 刚完成当前 wave 并进入暂停状态、即将推进至下一 wave 时刻，发出了一个携带旧 wave 号（如 wave=0）的请求。由于网络传输延迟或消息队列排队延迟，该请求可能滞后到达。
此时，问题便显现出来：
- Engine 的 current_wave 已经成功推进至 wave + 1，并已进入暂停状态；
- 而该请求仍携带原始的 request_wave = wave，且此时引擎已关闭，无法正常处理该请求；
- 因此，Engine 必须主动向 Coordinator 反馈这一异常情况，以触发协调机制进行状态修复。
这里有两个分支：
1. request_wave > self.current_wave：说明请求携带的 wave 比当前 Engine 本地记录更新。此时 Engine 只推进自己的 current_wave，不会在这个分支里发送 start_wave 给 Coordinator。
2. request_wave < self.current_wave 且 not self.engines_running：说明 Engine 已完成旧 wave 并进入了暂停状态，但又收到了属于旧 wave 的请求。为了避免仅有当前 Engine 被请求唤醒而其他 Engine 仍停留在暂停状态，Engine 会将 engines_running 置为 True，并通过 output_queue 发送 EngineCoreOutputs(start_wave=self.current_wave) 给 Coordinator。
后续Coordinator 需要补发 START_DP_WAVE，唤醒其他 Engine 进入同一轮 wave。
class DPEngineCoreProc(EngineCoreProc):
    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(=self.current_wave))
                )

        super().add_request(request, request_wave)
1.3 DP 如何部署模型
1.3.1 单机多卡场景的数据并行
在一个节点的 8-GPU 上运行 MODEL，并行配置 DP=8，用法示例命令：
vllm serve $MODEL --data-parallel-size 8
上述命令会启动 8 个独立进程（每个进程对应一张 GPU），每个 DP 进程都独立加载一份完整模型副本，分别处理不同子集的输入请求。这些进程并行执行推理任务，并各自维护独立的 KV 缓存（Cache），因此模型参数和中间计算在不同进程间不共享。请求分配方面，vLLM 将输入批次切分为若干部分，逐一交给各个进程（或 GPU）处理；通过 NVIDIA NCCL 库广播或分发输入，确保每个 GPU 得到正确的请求子集。
1.3.2 单机多卡场景的数据并行 + 张量并行
在一个节点的 8-GPU 上运行 MODEL，并行配置 DP=4，TP=2，用法示例命令：
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
单节点多 GPU 的 DP + TP 部署 llm，vLLM 会自动启动多个工作进程（workers）来并行处理请求，进程数由服务参数 --tensor-parallel-size 与 --data-parallel-size 的乘积决定。
例如，当--tensor-parallel-size=2, --data-parallel-size=4时，vLLM 会创建 4 个独立的模型副本（Replica）。每个副本由一个工作组（Worker Group）管理，这个工作组内部使用 2 个 GPU 来共同执行张量并行。这 4 个工作组（模型副本）之间通过数据并行的方式处理不同的请求。
1.3.3 多机分布式场景的数据并行
用法示例命令。一个头节点上运行 DP=4，DP 排名 0 和 1，在第二个节点上运行排名 2 和 3 的 rank：
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
如果想和 Ray 分布式框架一起使用，此 DP 模式也可以通过指定 --data-parallel-backend=ray。
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
使用 Ray 时有一些显著的不同：
- 只需要单个启动命令（在任何节点上）来启动所有本地和远程 DP 等级，因此与在每个节点上启动相比更为方便；
- 无需指定 --data-parallel-address ，运行命令的节点将用作 --data-parallel-address；
- 无需指定 --data-parallel-rpc-port；
- 远程 DP 排名将根据 Ray 集群的节点资源进行分配。
DP 推理服务流程总结：多进程加载模型副本 -> 划分请求 -> 并行生成 -> 汇总结果。
1.4 混合张量并行和数据并行
张量（模型）并行性与数据并行性是正交的，可以同时使用二者来训练/推理大型模型。下图显示了一组用于混合模型并行和数据并行性的 GPU。
- 一个完整模型权重切分到了 8 张卡（TP8），模型被复制了 64 份（DP64），一共启动了 512（TP8 × DP64 = 512）个 GPU。
- 张量（模型）并行。同一 TP 组内的多个 GPU 共同承载一份模型副本（例如图中的 GPU 1 到 8）。vLLM 中这称为 tensor parallel group（TP 组）。组内通信包括 all-reduce、all-gather、reduce-scatter 等，具体取决于算子实现（如列并行/行并行 Linear、词表并行 Embedding），并非每层都只做 all-reduce。
- 数据并行。在每个 TP 组中处于相同 TP 位置的 GPU（例如图中的 GPU 1、9、…、505）形成数据并行组（DP 组）。这些 rank 各自持有参数相同的完整模型副本（仅在 TP 维度上切分不同）。在 vLLM 推理中，DP 组之间的 all-reduce 主要用于调度协同（如全局未完成请求、Wave 状态等），而不是训练时的梯度同步；且 MoE 与稠密模型的 DP 行为不同（见下文场景说明）。
- 通信底层通常通过 PyTorch 的 torch.distributed 发起，默认 NCCL backend；同节点上 TP 还可能走 custom all-reduce 等优化路径。
vLLM 推理场景（512 GPU，MoE 模型，DP>1，内部负载均衡）
1. 任务分发：启用内部/混合 DP 负载均衡时，API Server（经 DPLBAsyncMPClient）根据各 DP engine 的 waiting/running 队列长度选择目标 engine。若 DP 组 5（0-indexed 为 rank 4，对应 GPU 33–40）当前负载最低，请求会被路由到该 engine。
2. TP 计算：该 engine 内 TP 组的 8 张 GPU（GPU 33–40）协同执行前向。组内按层进行 all-gather / all-reduce / reduce-scatter 等通信；Logits 在输出层通过 all-gather 或 gather 汇总（通常到 TP rank 0）。
3. DP 同步：当MoE 模型使用 DPEngineCoreProc 时，为优化性能，每 32 个 engine step 才执行一次 DP 组内的 all-reduce（sync_dp_state），检查所有 DP rank 是否仍有关 unfinished 请求。
4. 参与 all-reduce 的是 DP 组内全部 64 个 rank（同一 TP 位置），不是单个 GPU 代表整组。当全局确认无未完成请求时，结束 wave 并递增计数。稠密（非 MoE）模型在 vLLM 中各 DP engine 基本独立处理请求，不走上述 wave 同步机制。
暂时无法在飞书文档外展示此内容
[图片]
二 vLLM 负载均衡模式解析
vLLM 在 Data Parallel (DP) 场景下，尤其是 MoE 模型和大规模部署中，提供了三种请求负载均衡（Load Balancing, LB）部署模式，用于把请求尽量分配到更空闲的 DP Rank。这里的 DP Rank 可以理解为一个独立的 Engine Core 实例，通常对应一组 GPU。internal / hybrid LB 会参考各 Engine 的 waiting / running 请求数进行选择；external LB 则主要把请求分发交给外部系统完成。
2.1 三种模式的对比总结
暂时无法在飞书文档外展示此内容
1. --data-parallel-size N：全局 DP Rank 总数。在 Internal LB（默认）模式下，vLLM 会在这些 Rank 之间自动进行负载均衡。
2. --data-parallel-size-local：当前节点上的本地 DP Rank 数量。在 Hybrid LB 模式下，外部 LB 先将请求分发到节点，节点内的 vLLM 再在本地 Rank 之间进行负载均衡。
3. --data-parallel-rank：当前实例对应的 DP Rank。设置后会隐式启用 External LB；也可通过 --data-parallel-external-lb 显式启用 External LB。
  两种写法不能和 Hybrid LB 混用。启用后通常配合 data_parallel_size_local = 1 使用，每个 Rank 往往由独立 vLLM 实例对外提供服务，跨 Rank 分发由外部 Router / LB 完成；
4. --data-parallel-hybrid-lb 或 --data-parallel-start-rank：启用 Hybrid LB，Hybrid LB 需要同时配置 data_parallel_size_local。
请求负载均衡功能涉及的核心组件包括：
- DP Coordinator：在 internal / hybrid LB 中，Coordinator 收集各 Engine 的 waiting / running 请求数，并发布给前端用于选择 Engine。在 MoE + DP 场景下，Coordinator 还负责协调 wave，控制整个 DP 组中 Engine 的启停状态，也就是变量engine_running。Coordinator 并非在所有模式下都承担相同职责，external LB 下请求分发主要由外部系统完成，但 MoE + DP 仍需要 Coordinator 做 wave 的同步。
- ZMQ 通信：负责前端、EngineCore、Coordinator 之间的异步通信。除了 API Server / AsyncLLM 与 EngineCore 的请求和输出通道，还包括 Coordinator 向前端发布 stats / wave 状态，以及 Coordinator 向 Engine 广播 START_DP_WAVE。在 external LB 下，Engine 通常不向 Coordinator 发布用于负载均衡的 stats；Coordinator 主要发布 wave / engine running 状态变化。
- Scheduler：每个 Engine 内部的 Continuous Batching 调度器，负责在单个 Engine 内部组织 waiting / running 请求，并生成每一步要执行的批次。internal / hybrid LB 使用这些 waiting / running 计数做跨 Engine 选路；当前实现中，前端会按 waiting * 4 + running 计算负载分数，并优先选择分数最低，也就是负载最小的 Engine。
三种负载均衡的区别：
1. Internal LB 的特点是：vLLM 自己管理所有 DP Rank，并在内部完成请求分发。前端会从 Coordinator 接收各 Engine 的 waiting / running 统计，再根据负载选择目标 Engine。这个模式配置最简单，适合单节点或少量节点的部署。
2. Hybrid LB 的特点：外部 LB 先将请求分发到某个节点，节点内的 vLLM 再在本地 DP Rank 之间做负载均衡（DPLBAsyncMPClient + local_engines_only=True）。适用于每个节点有多个本地 DP Rank、且全局 DP Rank 分布在多个节点上的部署。通常通过 --data-parallel-hybrid-lb 或 --data-parallel-start-rank 启用，并配合 --data-parallel-size-local 指定节点内 rank 数。
  需要注意：
  - 如果 data_parallel_size_local == 1，节点内只有一个 DP Rank，无节点内负载均衡的空间，系统会自动切换至 External LB（data_parallel_external_lb=True，data_parallel_hybrid_lb=False），前端退化为 DPAsyncMPClient；
  - 如果 data_parallel_size_local == data_parallel_size，说明当前节点已包含全部 DP Rank，不再需要「节点间 external LB + 节点内 vLLM LB」的混合结构，系统会关闭 Hybrid LB（data_parallel_hybrid_lb=False），实际退化为 Internal LB 模式。
3. External LB 的特点：vLLM 不再负责跨 DP Rank 的请求选择，外部 Router、Nginx、K8s Service 或自定义调度系统负责决定请求发到哪个实例或 DP Rank。每个 vLLM 实例使用 DPAsyncMPClient，只对接本地单个 Engine（get_core_engine_for_request 固定返回 self.core_engine）。
  该模式在 vLLM 的 DP 框架下仅支持 MoE 模型，同时Coordinator 仍会启动，但其职责是 wave / engines_running 同步（广播 START_DP_WAVE 等），Engine 不上报用于 LB 的 stats，不再参与请求负载均衡；非 MoE 应启动多个不带 --data-parallel-* 参数的独立实例，由外部 Router 分发。
2.2 内部负载均衡（Internal Load Balancing）—— 代码流程详解
启动流程：
1. vllm serve ... --data-parallel-size N
2. API Server 创建 AsyncLLM，其中 EngineCoreClient.make_async_mp_client() 在 Internal LB 模式下返回 DPLBAsyncMPClient。
3. 启动 DPCoordinator 进程的同时启动多个 Engine 进程（每个 DP rank 一个）：
  - MoE 模型：使用 DPEngineCoreProc
  - 非 MoE 模型：使用 EngineCoreProc（进程内将 DP 视为 1，各 rank 独立运行）
4. 各 Engine Core 与 Coordinator 握手，注册 ZMQ 通道。
请求处理 + LB 核心流程（get_core_engine_for_request）：
# 简化自 vllm/vllm/v1/engine/core_client.py # (DPLBAsyncMPClient / DPAsyncMPClient 类的 get_core_engine_for_request 方法)
def get_core_engine_for_request(self, request):# 从 DP Coordinator 定期获取最新 stats（run_engine_stats_update_task 异步任务）
    stats = self.dp_coordinator_stats  # 包含每个 engine 的 waiting、running 队列长度# 简单加权打分（可扩展 KV cache 感知）
    scores = []for eng in engines:
        score = len(eng.waiting) * 4 + len(eng.running)   # 等待队列权重更高
        scores.append(score)
    
    chosen_idx = argmin(scores)   # 选负载最低的 Enginereturn engines[chosen_idx]
完整请求生命周期（Internal LB）：
1. HTTP 请求 → FastAPI → OpenAIServingCompletion.create_completion
2. Tokenize + 创建 Request 对象 → AsyncLLM.generate()
3. DPLBAsyncMPClient.add_request_async() → 调用 get_core_engine_for_request() 选 Engine
4. 通过 ZMQ Input Socket 将请求发送到选中的 Engine Core 的 Input Thread
5. Engine Core：
  - Input Thread → 放入 input_queue
  - Main Thread → Scheduler 加入 waiting queue → Continuous Batching（step()）
  - 执行 Forward（Prefill/Decode）→ Output Thread 通过 ZMQ 返回结果
  - MoE 场景：Engine 通过 Coordinator output socket 上报 SchedulerStats（waiting/running 计数）
6. API Server 收集输出 → 返回给客户端
DP Coordinator 作用：
- Internal / Hybrid LB（非 external LB）：接收各 Engine 上报的 SchedulerStats，维护每个 Engine 的 waiting / running 请求数，并通过 publish_front 发布给 前端 AsyncLLM，供 Internal LB 选 Engine。
- MoE 专用 — request wave 协调（enable_wave_coordination=True）：
  - 当 Engine 通过 wave_complete 表示当前 wave 结束时，Coordinator 更新 current_wave 和 engines_running，并同步给前端。
  - 在新请求到来（前端发送 FIRST_REQ）或 Engine 上报 start_wave 时，广播 START_DP_WAVE，让其他 Engine 从 paused 状态进入 running wave。
  - 对 MoE 场景尤其重要：当某个 Engine 收到真实请求后，其他 Engine 即使没有本地真实请求，也需要通过 START_DP_WAVE 进入 running 状态，并在必要时执行 execute_dummy_batch()，以保持 DP / EP 通信路径一致，避免部分 rank 不参与通信导致等待。
- External LB（MoE + DP）：Coordinator 仍可能存在，但 Engine 不上报 LB stats，Coordinator 主要发布 wave / engines_running 状态变化，不参与请求负载均衡。
2.3 混合负载均衡（Hybrid Load Balancing）
Hybrid LB 采用两层选路：节点间由外部 LB 分流，节点内由 vLLM 在本地多个 DP Rank 之间做 Internal LB。
每个 Node 运行 API Server / AsyncLLM，默认进程数等于 data_parallel_size_local，并仅在本地 DP Ranks 之间做负载均衡。上游由外部 LB（如 K8s Service / Ingress）将流量分发到各 Node。该模式通常用于多节点部署：节点间流量分配交给外部 LB，节点内请求分配交给 vLLM 的 DP Internal LB。
代码上仍使用 DPLBAsyncMPClient 的内部选路逻辑：基于 [waiting, running] 计分，score = waiting * 4 + running，选择分数最低的本地 Engine。
与纯 Internal LB 的区别在于，Hybrid LB 下 client 只管理本地 EngineCores（local_engines_only == True），LB 范围限制在本节点 [start_rank, start_rank + size_local) 对应的 DP ranks 内，从而减少前端跨节点管理 EngineCore 和跨节点 ZMQ 通信的成本。
关键参数说明
- --data-parallel-size：全局 DP rank 总数。例如两台机器、每台 4 个本地 DP rank，则 data_parallel_size = 8。
- --data-parallel-size-local：当前 Node 上启动并管理的本地 DP rank 数量。Hybrid LB 的节点内负载均衡范围由这个参数决定，且为 Hybrid LB 的必要配置。
- --data-parallel-start-rank：当前 Node 负责的全局 DP rank 起始编号。例如 Node 0 管理 rank 0–3，Node 1 管理 rank 4–7，则 Node 1 的 data_parallel_start_rank = 4。设置该参数也会触发 Hybrid LB 模式。
- --data-parallel-hybrid-lb：显式启用 Hybrid LB。需与 --data-parallel-size-local 配合使用；节点间流量仍由外部 LB 负责，节点内由 vLLM 做 Internal LB。
启用与约束：Hybrid LB 与 External LB 互斥。若 data_parallel_size_local == 1，会自动退化为 External LB；若单节点且 size_local == data_parallel_size，Hybrid LB 会被关闭，行为等价于纯 Internal LB。
2.4. 外部负载均衡（External Load Balancing）
External LB 把 DP Rank 的选路交给外部系统，vLLM 实例内部不再在多个 DP Rank 之间做负载均衡。典型场景是 MoE  的 one-pod-per-rank 部署：每个 Pod 或实例通常只对应一个 DP Rank，上游由 K8s Service、Ingress 等外部 LB 分发请求，vLLM 只负责本实例对应 rank 上的推理执行。
启用方式有两种：显式设置 --data-parallel-external-lb；或在 vllm serve 中显式设置 --data-parallel-rank，由 vLLM 自动推断进入 External LB 模式。External LB 与 Hybrid LB 互斥，不能同时启用。进入 External LB 后，data_parallel_size_local 会被固定为 1，每个实例只管理一个本地 EngineCore。
代码路径上，External LB 使用 DPAsyncMPClient，而不是 DPLBAsyncMPClient。get_core_engine_for_request() 固定返回 self.core_engine，因此不会计算 waiting * 4 + running，也不会在多个 DP ranks 之间做内部选路：
def get_core_engine_for_request(self, request: EngineCoreRequest):
    return self.core_engine
Coordinator 行为与 Internal LB 不同。External LB 下，Engine 不再上报 waiting/running 队列统计，Coordinator 因此也不承担多 rank 间的负载均衡职责。
MoE 且 DP > 1 时仍需要 DPCoordinator（通常由 dp_rank=0 的节点启动）做 wave 协调：维护整个 DP 集群在 running / paused 之间的全局状态（engines_running），并在新请求到来时推进、同步各 DP rank 的 wave。
API Server / AsyncLLM 仍会订阅 Coordinator，但拿到的主要是 wave 与 running 状态，而不是 Internal LB 所需的队列负载数据。
MoE 与非 MoE 的适用方式也有区别。External DP LB 主要面向 MoE 的 one-pod-per-rank 部署，因为 MoE 的多个 DP ranks 需要通过 EP 通信协作完成 forward，必须作为一个逻辑上的 DP 集群运行。非 MoE 模型若设置 External DP LB，vLLM 会直接报错；这类模型更适合启动多个彼此独立的 vLLM 实例，不带 --data-parallel-* 参数，再由普通外部 LB 在实例之间分流。
三、DP 在线/离线实现流程
vLLM 给用户提供了同步推理（一般用于离线）LLM 和异步推理（一般用于在线）AsyncLLM 两种模式，先看下在线推理方案的流程。
3.1 数据并行工作流程图
以下流程图，简单描述了一个推理请求从进入 API 服务器到最终返回给客户端的基本过程:
[图片]
vLLM 数据并行的工作流程总结如下所示:
1. 接收请求: 客户端通过标准的 HTTP 请求将推理任务发送到 vLLM 的 API 服务器。
2. 负载均衡: API 服务器内部的负载均衡器接收到请求，如果使用 DPLB 方案，所有新来的请求都进入一个全局共享的等待队列。它会查询所有数据并行（DP）Rank 的状态，特别是它们当前正在处理的请求数和等待队列的长度，只要 DP Rank 完成，就会立即从全局队列中“认领”新的任务进行处理，实现了“能者多劳”。
3. 分发请求: 负载均衡器会选择当前最空闲（即排队请求最少）的 DP Rank，并将该请求转发给它。
4. 引擎处理: 选定的 DP Rank（一个独立的 AsyncLLMEngine 进程）接收请求，并将其加入到自己的调度队列中。
5. 模型推理: Engine 内部的调度器根据其调度策略（如 FCFS）和 PagedAttention 机制，将请求批处理后交由其管理的 GPU Worker 执行模型推理。如果该 DP Rank 内部还配置了张量并行（tensor_parallel_size > 1），那么推理任务会由一个 GPU Worker 组协同完成。
6. 返回结果: 推理完成后，结果会返回给对应的 AsyncLLMEngine 进程。
7. 结果回传: AsyncLLMEngine 进程将生成的文本流通过进程间通信（IPC，由 ZMQ 实现）回传给 API 服务器。
8. 响应客户端: API 服务器将收到的结果流作为 HTTP 响应返回给客户端。
AsyncLLM/Engine相关的内容可以参考本课程第2课：第2课-vLLM 核心组件-引擎模块和流式执行
vLLM 数据并行的调用流程代码解析：
通过 vllm serve 命令启动 llm 推理服务时，本质上是调用了 ServeSubcommand 类的 cmd 函数，代码位于 vllm/entrypoints/cli/serve.py中。这几个个分支的主要区别在于有没有启动 API Server 以及启动多少个。这是因为单个 API Server 在处理大量并发请求时，可能成为瓶颈，所以多个 API Server 可以：
- 负载均衡地分散客户端请求
- 提高 API 端的吞吐量（因为有多个独立的异步事件循环）
- 更好地利用 CPU 多核，减轻推理系统本来就繁重的CPU压力
具体关于API Server的内容并不是AI Infra课程的重点，所以也就不展开了，大家感兴趣的可以自行阅读APIServerProcessManager类。
class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless or args.api_server_count < 1:
            run_headless(args)
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)
            else:
                # Single API server (this process).
                uvloop.run(run_server(args))
run_multi_api_server 函数调用 launch_core_engines 函数启动核心 engine，而launch_core_engines 函数按需启动 Engine 和 DP coordinator 进程。
@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    num_api_servers: int = 1,
) -> Iterator[tuple[
        Optional[Union[CoreEngineProcManager, CoreEngineActorManager]],
        Optional[DPCoordinator],
        EngineZmqAddresses,
]]:
    """Launch engine and DP coordinator processes as needed."""
    addresses = EngineZmqAddresses(
        inputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
        outputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
    )

    # Run the DP Coordinator process with rank 0 when in# online DP mode.
    # 1. 启动coordinator进程
    run_coordinator = dp_size > 1 and not offline_mode and dp_rank == 0if run_coordinator:
        coordinator = DPCoordinator(parallel_config)

    # 省略代码 ...
    from vllm.v1.engine.core import EngineCoreProc

    # Start local engines.
    # 2. 启动本地 Engine 进程
    if local_engine_count:
        local_engine_manager = CoreEngineProcManager(
            EngineCoreProc.run_engine_core,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            handshake_address=handshake_address,
            client_handshake_address=client_handshake_address,
            local_client=True,
            local_engine_count=local_engine_count,
            start_index=dp_rank,
            local_start_index=local_start_index or 0)
    else:
        local_engine_manager = None
        yield local_engine_manager, coordinator, addresses
    # 等待所有 Engine 和 Coordinator 进程启动完毕、连接就绪
    # Now wait for engines to start.
    wait_for_engine_startup(
        handshake_socket,
        addresses,
        engines_to_handshake,
        parallel_config,
        vllm_config.cache_config,
        local_engine_manager,
        coordinator.proc if coordinator else None,
    )
CoreEngineProcManager 在初始化时会为每个本地 Engine 启动一个后台进程，进程的 target 是 EngineCoreProc.run_engine_core，并通过 kwargs 传入 dp_rank 和 local_dp_rank。也就是：
# 2. 启动本地 Engine 进程
if local_engine_count:
   local_engine_manager = CoreEngineProcManager(EngineCoreProc.run_engine_core,
run_engine_core 函数会先判断 data_parallel_size > 1 or dp_rank > 0 是否成立；若成立，再进一步区分：
- MoE 模型：构造 DPEngineCoreProc，保留完整 DP 拓扑，各 rank 需要跨进程同步。
- 非 MoE 模型：仍构造 EngineCoreProc，但会把 parallel_config 的 DP 字段重置为 1，各 rank 独立运行，互不同步。
暂时无法在飞书文档外展示此内容
@staticmethod
def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
    if data_parallel and vllm_config.model_config.is_moe:
        engine_core = DPEngineCoreProc(*args, **kwargs)
    else:
        parallel_config.data_parallel_size = 1
        parallel_config.data_parallel_size_local = 1
        parallel_config.data_parallel_rank = 0
        engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
    engine_core.run_busy_loop()
DPEngineCoreProc 的 run_busy_loop 在本章前文中已介绍。它重写了父类的 run_busy_loop，因此不再仅是简单的本地请求处理循环，而是 MoE DP 场景下各 Engine 之间的同步协调枢纽：通过 dummy batch、all-reduce 判断全局是否还有 unfinished requests，并在 wave 结束时向 coordinator 或前端发送 wave_complete 通知。但是，这一机制仅适用于 MoE + DP；非 MoE 的 DP rank 走普通 EngineCoreProc.run_busy_loop，不存在跨 rank 的 wave 同步。
默认情况下，vLLM 采用 内部负载均衡模式，由客户端自动选择最优的 EngineCore 来处理请求，从而实现高效的资源利用。在内部负载均衡模式下，前端的 EngineCoreClient （也就是AsyncLLM和Engine通信的中介，我们在前面的章节中介绍过）实际上是一个 DPLBAsyncMPClient实例。
总的来说，AsyncLLM 是对外的公开 API 类。它内部通过 EngineCoreClient.make_async_mp_client 创建并持有一个 EngineCoreClient 子类，作为与后端 EngineCore 通信的中介，负责将用户的推理请求翻译成 EngineCore 能理解的形式。
具体类型取决于 DP 配置：
- data_parallel_size == 1：AsyncMPClient（单 Engine，无跨 Engine 负载均衡）
- data_parallel_size > 1 且 internal / hybrid LB：DPLBAsyncMPClient（客户端在多个 EngineCore 间负载均衡；hybrid 模式下仅均衡本地 ranks）
- data_parallel_size > 1 且 external LB：DPAsyncMPClient（每个客户端实例只对接一个 DP rank，负载均衡由外部完成）
暂时无法在飞书文档外展示此内容
class DPLBAsyncMPClient(DPAsyncMPClient):
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        print('inner get_core_engine_for_request')
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count
选择的 Engine 逻辑如下（仅当请求未指定 data_parallel_rank，且不属于 late interaction 特殊路由时）：
根据 coordinator 下发的负载统计 lb_engines（每个 Engine 的 [waiting, running] 计数），计算加权分数 score = waiting × 4 + running，选择 score 最小的 Engine。waiting 的权重是 running 的 4 倍——即优先降低 waiting 队列，waiting 相同时再选 running 更少的 Engine。另外，搜索并不是每次都从 0 号 Engine 开始，而是从 self.eng_start_index 开始轮询：
self.eng_start_index = (len(self.core_engines) * self.client_index) // client_count
多个 API Server 客户端各自有不同的起始偏移，避免在负载相同（例如刚启动时全是 0）时，所有请求同时涌向 0 号 Engine。
waiting, running = current_counts[idx]
score = waiting * 4 + running
3.2 单 rank 内 Continuous Batching 数据流
一条请求到来后 vLLM 系统是如何工作的呢？单 GPU 系统的数据工作流程图总结如下所示：
[图片]
3.3 DP 离线代码解析
vLLM 使用 DP 进行离线/在线 LLM 推理时，每个 rank 都应处理不同的提示词（请求），即每个 rank 都处理数据集的不同部分。以简单的离线推理为例，来逐功能分析 DP 如何实现每个 rank 分别处理不同的提示词：
1. 批处理样本提示词定义如下，总共 400 个请求。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 100
2. 把 prompts 在 DP ranks 之间划分（均匀分配），即实现数据并行负载均衡。如果prompts在DP ranks之间不能被平均分配，还剩 remainder = 1 个 prompt，那么就要分配给前 remainder 个 rank
floor = len(prompts) // dp_size # 计算每个 group 的基础大小（向下取整）
remainder = len(prompts) % dp_size # 计算除不尽时的余数，表示有 remainder 个 group 会比 floor 多 1 个元素

def start(rank):
    """
    start(rank) 返回第 rank 个组在原列表中的起始索引（inclusive）
    """
    return rank * floor + min(rank, remainder)

prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]

# 如果当前 rank 被分配到 0 条 prompt（例如 prompt 数 < dp_size），
# 为避免后续逻辑报错，放一个占位 prompt。
if len(prompts) == 0:
    prompts = ["Placeholder"]
# 输出每个 rank 实际要处理的 prompt 数
print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")
上述代码是常见的把 N（提示词数量）个元素尽量均匀分成 K（DP SIZE）组”的做法：
  - 先把每组分 floor = N // K 个；
  - 余下 remainder = N % K 个，把它们分别分给前 remainder 个组（每组 +1）。用 start(rank) 的形式可以直接得到每组的起止索引（结束索引用 start(rank+1)）。
举例理解，假设 len(prompts)=10, dp_size=4：
  - floor = 10 // 4 = 2
  - remainder = 10 % 4 = 2
  前面的两个rank需要额外多承担一条prompt
  - start(0) = 0*2 + min(0,2) = 0 → slice [0:3) -> 3 个元素（因为 start(1)=3）
  - start(1) = 1*2 + min(1,2) = 3 → slice [3:6) -> 3 个元素
  - start(2) = 2*2 + min(2,2) = 6 → slice [6:8) -> 2 个元素
  - start(3) = 3*2 + min(3,2) = 8 → slice [8:10) -> 2 个元素
这种 floor + remainder + min(rank, remainder) 的分片算法在并行/分片任务分配、MPI 等场景是常见的。在 len(prompts) % dp_size 计算除不尽时，其效果是有 remainder 个 group 会比 floor 多 1 个元素。
3. 创建 LLM 引擎实例
# 使用 SamplingParams 指定生成策略
# 因为使用 DP，所以使用 global_dp_rank 的奇偶来给不同的 max_tokens 以演示“每个 rank 可以有不同采样参数”。
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=[16, 20][global_dp_rank % 2]
)

# Create an LLM.
llm = LLM(
    model=model,
    tensor_parallel_size=GPUs_per_dp_rank,
    enforce_eager=enforce_eager,
    enable_expert_parallel=True,
    trust_remote_code=trust_remote_code,
    max_num_seqs=max_num_seqs,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    enable_dbo=enable_dbo,
    quantization=quantization,
    compilation_config=compilation_config,
)
outputs = llm.generate(prompts, sampling_params)

# 调用 generate(...) 并打印前若干结果
# Print the outputs.
for i, output in enumerate(outputs):
    if i >= 5:
        # print only 5 outputs
        break
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
        f"Generated text: {generated_text!r}"
    )

# Give engines time to pause their processing loops before exiting.
sleep(1)
4. 用 multiprocessing.Process 启动多个本地 DP rank 的进程。for 循环部分的代码实现了下述目的：
  - 在每个节点内启动 dp_per_node 个进程；global_dp_rank 的值通过 range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node) 计算得到（保证节点间不重复的全局 ranks）。
  - 每个 Process 都运行前面定义的 main(...)，因此每个子进程会独立设置自己的 VLLM_* 环境变量并创建一个 LLM 实例来驱动该 rank 的推理。
if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    # dp_size % node_size == 0 保证每个节点能分配到整数个 DP rank（每节点 DP 数相同）
    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        # 每个 Process 都运行前面定义的 main(...)，因此每个子进程会独立设置自己的 VLLM_* 环境变量并创建一个 LLM 实例来驱动该 rank 的推理。
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size, # 把 tp_size 当作“每个 DP rank 的 GPU 数
                args.enforce_eager,
                args.trust_remote_code,
                args.max_num_seqs,
                args.max_model_len,
                args.compilation_config,
                args.gpu_memory_utilization,
                args.enable_dbo,
                args.quantization,
            ),
        )
        proc.start()
        procs.append(proc)
5. 等待进程退出与超时处理代码。
exit_code = 0
for proc in procs:
    proc.join(timeout=args.timeout)
    if proc.exitcode is None:
        print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
        proc.kill()
        exit_code = 1
    elif proc.exitcode:
        exit_code = proc.exitcode

exit(exit_code)
上述代码的作用：
- 父进程遍历所有子进程并等待它们结束：proc.join(timeout=args.timeout) 阻塞至子进程结束或超时（timeout 默认为 300 秒）。
- if proc.exitcode is None: 如果子进程在 timeout 内仍未结束，则认为挂起，打印信息并调用 proc.kill() 强制结束。
- elif proc.exitcode:：如果子进程退出码非 0（表示子进程出现异常或通过 sys.exit(n) 返回非 0），父进程把这个码作为最终 exit_code（仅记录最后一个非零的 exitcode）。
- 最后调用 exit(exit_code) 使整个脚本返回相应的退出码。
启动方式如下：
python /root/vllm_learn/code/course11/data_parallel.py  \ 
--model="Qwen/Qwen3-1.7B"   \
--dp-size=2   \
--tp-size=1   \
--gpu-memory-utilization=0.8    \
--trust-remote-code \
--disable-expert-parallel

DP rank 1, Prompt: 'Hello, my name is', Generated text: " Kiko, and I'm a young boy who loves to play with my toys and read stories."
DP rank 0, Prompt: 'Hello, my name is', Generated text: " Kiko, and I'm a young boy who loves to play with my toys"
DP rank 1, Prompt: 'The president of the United States is', Generated text: ' elected by the Electoral College. The Electoral College is a group of 538 electors who'
DP rank 0, Prompt: 'The president of the United States is', Generated text: ' elected by the Electoral College. The Electoral College is a group of 53'
DP rank 1, Prompt: 'The capital of France is', Generated text: ' Paris. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of Italy'
DP rank 0, Prompt: 'The capital of France is', Generated text: ' Paris. The capital of Spain is Madrid. The capital of Germany is Berlin.'
DP rank 1, Prompt: 'The future of AI is', Generated text: ' likely to be shaped by major advancements in machine learning, particularly in areas such as natural language processing,'
DP rank 0, Prompt: 'The future of AI is', Generated text: ' likely to be shaped by major advancements in machine learning, particularly in areas such as'
DP rank 1, Prompt: 'Hello, my name is', Generated text: ' Lily, I am 18 years old and I am studying to become a teacher. I have'
DP rank 0, Prompt: 'Hello, my name is', Generated text: ' Lily, I am 18 years old and I am studying to become a'
代码示例：
下述代码在单进程里直接演示如何把 prompts 划分给 dp_size 个 rank（适合快速验证与理解）。详见sim_dp_split_prompts.py。
# sim_dp_split_prompts.py
# 演示 floor/remainder 分片算法 —— 单进程模拟
def get_chunk(prompts, dp_size, global_dp_rank):
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    chunk = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(chunk) == 0:
        chunk = ["Placeholder"]
    return chunk

if __name__ == "__main__":
    prompts = [f"prompt_{i}" for i in range(10)]  # 10 条示例 prompt
    dp_size = 3

    print(f"All prompts ({len(prompts)}): {prompts}\n")
    for rank in range(dp_size):
        assigned = get_chunk(prompts, dp_size, rank)
        print(f"rank {rank} -> assigned {len(assigned)} prompts: {assigned}")
上述代码运行后输出如下:
All prompts (10): ['prompt_0', 'prompt_1', 'prompt_2', 'prompt_3', 'prompt_4', 'prompt_5', 'prompt_6', 'prompt_7', 'prompt_8', 'prompt_9']

rank 0 -> assigned 3 prompts: ['prompt_0', 'prompt_1', 'prompt_2']
rank 1 -> assigned 3 prompts: ['prompt_3', 'prompt_4', 'prompt_5']
rank 2 -> assigned 2 prompts: ['prompt_6', 'prompt_7']
rank 3 -> assigned 2 prompts: ['prompt_8', 'prompt_9']
用 multiprocessing.Process 启动多个DP rank进程（CPU 可运行）
下述代码用于在单机上用进程模拟每个 DP rank 的运行（不会使用 GPU、不会依赖 vLLM），方便理解和演示真实的多进程分配逻辑以及主进程如何等待子进程结束（并可设置超时）。
# split_multiproc_demo.py
# 使用 multiprocessing 启动多个进程模拟 DP ranks（CPU 可运行）

from multiprocessing import Process
from time import sleep

def worker(prompts, dp_size, global_dp_rank):
    """子进程函数：按给定公式取出当前 rank 的 prompts 并打印"""
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    chunk = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(chunk) == 0:
        chunk = ["Placeholder"]
    print(f"[PID {__import__('os').getpid()}] DP rank {global_dp_rank} -> {len(chunk)} prompts: {chunk}", flush=True)

if __name__ == "__main__":
    # 示例数据
    prompts = [f"prompt_{i}" for i in range(7)]  # 7 条 prompt，故会产生 remainder 情况
    dp_size = 4  # 分成 4 个 rank（会出现某些 rank 被分到 0 条）
    procs = []

    # 在单节点模拟启动 dp_size 个进程，每个进程被视作一个 global_dp_rank
    for rank in range(dp_size):
        p = Process(target=worker, args=(prompts, dp_size, rank))
        p.start()
        procs.append(p)

    # 等待所有子进程结束，单个子进程最长等 5 秒
    timeout_seconds = 5
    for p in procs:
        p.join(timeout=timeout_seconds)
        if p.exitcode is None:
            print(f"Process {p.pid} didn't exit within {timeout_seconds}s, killing it.")
            p.kill()
上述代码运行后输出结果如下:
[PID 51379] DP rank 2 -> 2 prompts: ['prompt_6', 'prompt_7']
[PID 51377] DP rank 0 -> 3 prompts: ['prompt_0', 'prompt_1', 'prompt_2']
[PID 51378] DP rank 1 -> 3 prompts: ['prompt_3', 'prompt_4', 'prompt_5']
[PID 51380] DP rank 3 -> 2 prompts: ['prompt_8', 'prompt_9']
因为是 4 个进程同时运行，所以程序运行输出的打印信息，并不一定是 DP rank 0 在最前面。因为 len(prompts)=10, dp_size=4，计算得 floor=2, remainder=2：
- rank0: start(0)=0 → slice [0:2) => 2 elements
- rank1: start(1)=2 → slice [2:4) => 2 elements
- rank2: start(2)=4 → slice [4:6) => 2 elements
- rank3: start(3)=6 → slice [6:7) => 1 element
- 所以 4 个进程会分别打印自己被分配的 prompts。
参考资料
- LLM推理数据并行负载均衡(DPLB)浅析
- vLLM DP特性与演进方案分析
- vLLM Data Parallel Deployment Options