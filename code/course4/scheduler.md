本文深入解析 vLLM 调度器核心机制，涵盖请求接入、优先级策略、资源分配与抢占逻辑，揭示其如何实现高并发、低延迟的高效推理调度。
调度器 Demo
为了帮助大家更直观地理解请求调度的整个过程，我们特别设计并实现了一个调度演示Demo。建议在完成本章学习后，再动手运行该示例，深入体会调度机制的工作原理。相关代码位于课程项目目录下的 code/course3/sche.py。
调度器
增加请求
我们可以把 vLLM 的调度理解为以下几条规则：
1. RUNNING 请求优先：vLLM 不是简单地谁先来谁先执行。在每个 scheduler step 中，调度器会先尝试继续调度已经处于 RUNNING 状态的请求。这样做的目的是让已经进入执行流水线的请求持续推进，减少生成过程中的停顿。这里的 RUNNING 不只表示已经开始输出 token 的 decode 请求，也可能包括 chunked prefill、spec decode，或者被抢占后恢复执行的请求。
  1. chunked prefill：如果 prompt 很长，vLLM 可能不会在一个 step 中处理完整个 prompt，而是把 prompt 分成多段处理。
  2. spec decode：投机解码会先由 draft model 或相关机制预测多个候选 token，再由目标模型验证。这个过程中，一个 RUNNING请求在某个 step 中可能不只是生成单个 token，而是调度和验证一批投机解码获得的token。
2. waiting 队列按策略选择请求：当 RUNNING 请求调度完成，并且当前 step 仍然有 token budget 和并发名额时，调度器才会从 waiting 队列中选择请求。waiting 队列的顺序取决于 scheduler policy：
  - 在 FCFS 模式下，请求按照进入队列的先后顺序处理。
  - 在 priority 模式下，priority 数值越小优先级越高；如果优先级相同，则 arrival_time 更早的请求优先。
  这里的token budget 可以理解为：一个 scheduler step 里最多允许调度多少个 token。举个例子：token_budget = 8，当前 waiting 队列里有 3 个请求：
  - 请求 A：prompt 需要 5 个 token
  - 请求 B：prompt 需要 4 个 token
  - 请求 C：prompt 需要 3 个 token
   如果先调度 A，会消耗 5 个 token budget：剩余 token_budget = 3，这时 B 需要 4 个 token，可能就会放不下。
3. 新请求次之：新请求刚进入调度器时，并不会直接执行，而是先进入 waiting 队列。只有当调度器处理完当前可继续推进的 RUNNING 请求，并且资源预算允许时，新请求才会被取出并加入 running 队列。因此，新请求通常排在已进入执行的请求之后。这可以保证正在生成的请求更加平滑，也能避免频繁切换导致执行效率下降。
4. 抢占规则：当 KV cache 等资源不足，当前请求无法继续分配 block 时，调度器会从 running 队列中选择一个请求进行抢占。抢占规则取决于调度策略：
  - 在 FCFS 模式下，调度器会抢占 running 队列末尾的请求。
  - 在 priority 模式下，调度器会抢占优先级最低的请求；如果优先级相同，则抢占到达时间更晚的请求。
被抢占的请求会释放已经占用的 KV cache，同时状态变为 PREEMPTED，然后重新放回 waiting 队列，等待后续 step 再次被调度。
如下代码所示：
- 在 AsyncLLMEngine 中，真正驱动推理循环的是 EngineCoreProc。它通常运行在独立进程中，负责接收前端请求、维护调度器状态，并不断执行 engine step。
- 前端通过 ZMQ ROUTER-DEALER 模式把请求异步发送给 EngineCoreProc。请求到达后，EngineCoreProc 不会立刻执行推理，而是先完成反序列化和预处理，再写入内部队列 self.input_queue。
- 随后，EngineCoreProc 的主循环会消费 input_queue。如果当前没有运行中的请求，调度器中也没有待处理请求，它会阻塞等待新输入；一旦请求到达，就交给 _handle_client_request() 处理。
def run_busy_loop(self):
    """EngineCore 的主循环。"""
    while True:
        # 先处理前端发来的请求，例如 ADD、ABORT、UTILITY。
        self._process_input_queue()

        # 再执行一次 engine step，包括调度、模型前向和输出处理。
        self._process_engine_step()


def _process_input_queue(self):
    """处理 input_queue 中的请求，直到需要执行一次 engine step。"""
    waited = False


    while (
        not self.engines_running
        and not self.scheduler.has_requests()
        and not self.batch_queue
    ):
        if self.input_queue.empty():
            # 当前没有输入请求，EngineCore 进入等待状态。
            logger.debug("EngineCore waiting for work.")
            waited = True

        # 阻塞等待一个新请求到达。处理该请求，例如把 ADD 请求注册到 scheduler。
        req = self.input_queue.get()
        self._handle_client_request(*req)

    # 如果 input_queue 中还有已经到达的请求，
    while not self.input_queue.empty():
        # 非阻塞地取出队列中的请求。
        req = self.input_queue.get_nowait()

        # 处理请求
        self._handle_client_request(*req)
对于新增请求，_handle_client_request() 会调用父类的EngineCore.add_request()，EngineCore.add_request() 会先做基础校验，例如检查 request_id 类型、pooling task 是否支持、KV transfer 配置是否可用。校验完成后，请求才会被交给调度器self.scheduler。
def _handle_client_request(
    self, request_type: EngineCoreRequestType, request: Any
) -> None:
    """Dispatch request from client."""

    if request_type == EngineCoreRequestType.ADD:
        req, request_wave = request
        self.add_request(req, request_wave)
        
class EngineCore:
    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """

        self.scheduler.add_request(request)
总的来说，前端请求不会直接触发模型推理，而是先经过 ZMQ 通信、输入线程、内部队列和 EngineCore 校验，最终注册到 Scheduler 的waiting 队列中，等待后续 engine step 调度执行。
调度器结构
调度器内部维护两个核心结构：
self.waiting = create_request_queue(self.policy)
self.running: list[Request] = []
至此，请求才正式进入 vLLM 的调度生命周期。不过，进入调度器并不意味着请求会立刻执行。每个 scheduler step 都会根据当前资源状态做一次决策：优先推进哪些 RUNNING 请求，是否还有资源接纳新的 waiting 请求，是否需要抢占已有请求，以及如何在吞吐量、延迟和资源利用率之间取得平衡。
当请求刚被加入调度器时，它的初始状态是 WAITING，表示该请求还没有开始执行。此时，请求只是在等待队列中排队，等待后续 scheduler step 选择它。
def add_request(self, request: Request) -> None:
    self.waiting.add_request(request)
    self.requests[request.request_id] = request
需要注意的是，一个请求通常包含多个 token。例如用户输入 "Hi!"，可能会被 tokenizer 切分成 "Hi" 和 "!" 两个 token。vLLM 不要求一次性处理完整个请求，而是允许按 token budget 分批调度：每个 scheduler step 可以只处理其中一部分 token。
如下图所示在waiting队列中每个请求均未开始调度执行，它们各自有一段prompt，也就是用户的提示输入。其中，每个请求都有一个唯一的请求id（request id）对应。如下图中有3个不同的队列，请求3的prompt输入是"Hi !"。
暂时无法在飞书文档外展示此内容
在实际实现中，请求中的 token 并不是以字符串形式保存的，而是已经被 tokenizer 转换为整型 ID，也就是词表中的索引。模型真正接收和计算的是这些 token ID。
因此，图中的字符串 token 只是为了便于理解。真实的 Request 内部保存的是 token ID 序列，例如：[1234, 5678]。接下来，我们来看调度器如何添加一个新的请求。
class Scheduler(SchedulerInterface):
    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
调度的规则
调度器调度running队列
计算本轮可调度 Token 数量
在上述步骤中，系统已经接收到来自客户端的新请求。接下来，调度器会在每个 scheduler step 中，对当前所有未完成的请求进行统一调度，并根据调度策略和资源状态，选择本轮可以执行的一个或多个请求。
需要注意的是，一个 scheduler step 并不是只能处理单个请求。只要资源允许，多个请求可以在同一轮中并行推进。这些请求可能处于不同状态：有些已经完成 prompt 处理，正在继续生成后续 token；有些还没有开始执行，仍在等待首轮计算。
当前调度主要受两类资源约束：
1. KV cache 显存资源
请求执行过程中需要为 Key/Value cache 分配物理 block。如果可用 block 不足，请求可能无法被调度，甚至需要触发抢占。
2. 单步 token budget
每个 scheduler step 能处理的 token 总数是有限的，不能超过系统设定的上限，例如 max_num_batched_tokens，也就是上文说过的token budget。这个限制用于控制单轮 batch 的规模，避免显存和计算负载过大。
如图所示，当前系统中的请求可以简化理解为两类：
1. 已经进入执行中的请求，通常位于 running 队列
这类请求已经被调度器接纳，并在之前的 step 中执行过一部分 token。典型情况是 decode 请求：prompt 已经处理完成，当前只需要继续生成下一个 token。
这类任务通常每轮只新增少量 token，但需要频繁读取已有的 KV cache，因此更偏访存密集型，主要受显存带宽影响。
2. 尚未开始或等待恢复执行的请求，通常位于 waiting 队列
新请求进入调度器后，会先进入 waiting 队列。典型情况是 prefill 请求：它的 prompt token 尚未被模型完整处理，需要等待调度器为其分配 token budget 和 KV cache block。
对 prefill 来说，GPU 需要对 prompt token 做前向计算。这个过程可以一次性处理完整 prompt，也可以在启用 chunked prefill 时分批处理。prefill 通常计算量更大，能更充分利用 GPU 算力，但也会快速消耗 KV cache block。
因此，调度器每一轮都要做权衡：既要尽量保持 running 请求的输出连续性，又要在资源允许时接纳新的 waiting 请求，同时避免单轮 token 数量和 KV cache 占用超过系统上限。
优先级：vLLM 在每个 scheduler step 中会优先调度已经处于 RUNNING 状态的请求。这样做是为了尽量保持已开始执行请求的连续性，尤其是已经进入 decode 阶段的请求。如果这类请求长时间得不到调度，用户端会明显感受到流式输出卡顿。
暂时无法在飞书文档外展示此内容
class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.
        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        ...
        ...
在前面的例子中，我们提到：一次 scheduler step 可以同时调度来自不同请求的多个 token。比如，一个请求可能正在继续生成下一个 token，另一个请求可能正在处理 prompt token。只要资源允许，它们可以在同一轮中一起执行。但是，调度器不能无限制地把 token 加入同一个 batch。为了控制单轮负载，vLLM 会使用 token budget 限制每个 scheduler step 最多能调度多少个 token。这个上限通常由 max_num_batched_tokens 决定。
如下图所示，假设当前一轮的 token_budget = 3，也就是本轮最多只能调度 3 个 token。由于调度器会优先推进已经进入执行流程的请求，因此会先处理 Request 1。假设 Request 1 本轮只需要再处理 1 个 token，那么：剩余 token_budget = 3 - 1 = 2
接着，调度器检查 waiting 队列中的 Request 2。如果 Request 2 的 prompt 正好需要处理 2 个 token，那么它也可以在本轮被调度：剩余 token_budget = 2 - 2 = 0。此时，本轮 token budget 已经用完。即使这时又来了一个 Request 3，并且它需要处理 5 个 token，也不能在当前 step 中继续执行。它会留在 waiting 队列中，等待下一轮 scheduler step。
因此，这一轮的调度结果是：
- Request 1: 调度 1 个 token  
- Request 2: 调度 2 个 token  
- Request 3: 暂不调度，继续等待
token_budget 的作用，就是把每轮调度的 token 总数控制在一个上限内，从而避免单个 step 的 batch 过大，影响显存占用、计算负载和整体延迟。
暂时无法在飞书文档外展示此内容
首先，调度器会优先处理已经在执行流程中的请求，也就是 running 队列中的请求。前面我们提到，vLLM 可以在一个 scheduler step 中同时调度多个请求。它们可能处于不同的执行状态：有的请求已经处理完 prompt，正在继续生成下一个 token；有的请求还在处理 prompt。只要资源允许，这些请求可以被组合到同一个 batch 中一起执行。
为了控制单步负载和显存压力，vLLM 会使用 token_budget 限制当前 scheduler step 最多能调度多少个 token。这个预算的初始值来自：
token_budget = self.max_num_scheduled_tokens
而 self.max_num_scheduled_tokens 通常对应配置中的 max_num_batched_tokens。
例如，假设当前系统设置一轮最多处理 3 个 token：
1. Request 1 已经在 running 队列中，本轮只需要继续处理 1 个 token。
2. Request 2 还在等待处理 prompt，例如 prompt "Hi!" 被切成 2 个 token。
3. 由于 1 + 2 = 3，没有超过当前 step 的 token budget，因此这两个请求可以在同一轮中一起被调度。
对应到源码中，schedule() 会先遍历 running 队列：
class Scheduler(SchedulerInterface):
    def schedule(self) -> SchedulerOutput:
        ...
        # 当前 step 最多还能调度多少个 token。
        token_budget = self.max_num_scheduled_tokens

        # 首先调度 RUNNING 请求。
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            # 取出 running 队列中的一个请求。
            request = self.running[req_index]

            # 计算该请求还需要继续处理多少个 token。
            # num_tokens_with_spec 表示当前请求期望被模型计算到的位置，
            # num_computed_tokens 表示已经完成计算的位置。
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            # 如果配置了 long_prefill_token_threshold，对单个请求本轮可处理的 token 数做额外限制。
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

            # 不能超过当前 step 剩余的 token budget。
            num_new_tokens = min(num_new_tokens, token_budget)

            # 也不能超过模型最大长度限制。
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

            # 如果该请求本轮没有 token 可以调度，
            # 则跳过它，继续检查 running 队列中的下一个请求。
            if num_new_tokens == 0:
                req_index += 1
                continue
            ...
这段逻辑的核心可以整理为：
1. 调度器从 running 队列中取出请求。  
2. 计算该请求本轮还需要处理的 token 数量。  
3. 用当前剩余的 token_budget 对该数量进行截断：  
  - 若 token_budget 充足，请求本轮可处理全部所需 token；  
  - 若 token_budget 不足，请求本轮只能处理部分 token，剩余部分留到后续调度步（scheduler step）继续处理。
4. 如果num_new_tokens等于0，那么可能有两类原因：
  1. 请求已经没有新的 token 需要计算，比如它已经接近 max_tokens 或 max_model_len，本轮不需要再调度。
  2. 当前资源限制导致无法调度。
暂时无法在飞书文档外展示此内容
对于 running 队列中的每个请求，调度器首先会计算它在当前 step 中还需要处理多少个 token。可以简化理解为：
当前还需要处理的 token 数量 = 请求当前目标 token 总数 - 已经计算过的 token 数
在源码中，对应的是：
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
这里的 num_tokens_with_spec 可以理解为请求当前希望模型计算到的位置，num_computed_tokens 表示已经完成计算的位置。因此二者的差值，就是当前还需要补上的 token 数量。不过，num_new_tokens 并不是算出来多少就一定处理多少。它还要受到几类限制：
1. 不能超过当前 step 剩余的 token_budget
token_budget 表示本轮调度最多还能处理多少个 token。调度器会用它限制当前请求本轮可处理的 token 数。
num_new_tokens = min(num_new_tokens, token_budget)
2. 不能超过模型最大长度限制
请求当前位置不能超过模型允许的最大序列长度，因此还需要受到 max_model_len 约束。
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - 1 - request.num_computed_tokens,
)
3. 可能受到长 prefill 切分限制
如果配置了 long_prefill_token_threshold，并且当前请求一次需要处理的 token 数超过这个阈值，调度器会把它截断到该阈值以内。
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
也就是说，当前 step 中真正可调度的 token 数量，是在“请求剩余工作量”的基础上，经过 long_prefill_token_threshold、token_budget 和 max_model_len 共同限制后的结果。
当调度器确定了本轮要为该请求处理多少个 token（即 num_new_tokens）之后，下一步就需要为这些 token 分配 KV cache block。
对应源码是：
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
这里的 allocate_slots() 会尝试为当前请求新增的 token 分配足够的 KV cache 空间。  
- 如果分配成功，说明该请求本轮可以继续被调度；  
- 如果分配失败，则说明当前 KV cache 资源不足，调度器可能需要触发抢占，释放其他请求占用的 block。
为本轮调度的token分配显存block
在 vLLM 的内存管理机制中，KV cache 的分配单位不是单个 token，而是固定大小的 block。每个 block 可以容纳一定数量的 token，这个容量由 block_size 决定。因此，调度器在决定某个请求本轮要处理 num_new_tokens 个 token 之后，还需要继续判断：当前 KV cache 中是否有足够的物理 block 可以承载这些新增 token 对应的 Key/Value 缓存。
需要注意的是，实际的 block 分配并不是简单地用 num_computed_tokens / block_size 向上取整。vLLM 还会综合考虑当前请求已经占用的 block、本轮新增 token、lookahead token、prefix cache 命中情况，以及是否存在外部 KV cache 等因素。
简单理解，block 是 vLLM 管理 KV cache 的物理存储单位，allocate_slots() 就是在为当前请求即将计算的 token 预留 KV cache 空间。
在调度过程中，block 分配可能出现两种结果：
- 资源充足：allocate_slots() 成功返回 new_blocks，表示已经为当前请求本轮新增的 token 分配到所需 KV cache 空间，请求可以继续执行。
- 资源不足：allocate_slots() 返回 None，表示当前可用 block 不足，无法满足该请求本轮的调度需求。此时调度器会进入后续的抢占逻辑，尝试释放其他请求占用的 KV cache block。
暂时无法在飞书文档外展示此内容
当调度器已经计算出请求 x 本轮需要处理的 num_new_tokens 后，下一步需要为这些 token 申请 KV cache block：
new_blocks = self.kv_cache_manager.allocate_slots(...)
如果 allocate_slots() 返回 None，说明当前可用的 KV cache block 不足，无法满足请求 x 的本轮调度需求。此时调度器会尝试抢占其他 running 请求，释放它们占用的 KV cache block，把资源让给当前请求。
在 FCFS 调度策略下，源码会从 running 队列尾部选择一个请求进行抢占：
preempted_req = self.running.pop()
这里的 pop() 会把 running 队列最后一个请求取出。可以理解为：优先保护更早进入 running 队列的请求，抢占相对靠后的请求。在 FCFS 策略下选择 running 队列末尾的请求，源于 vLLM 默认的先来先服务调度逻辑。越早进入 running 队列的请求已运行时间更长，理应获得更高的持续执行保障；而队尾请求接入调度较晚，对其发起抢占对整体执行连续性的影响最小。
被选中的请求会交给 _preempt_request() 处理：
self._preempt_request(preempted_req, scheduled_timestamp)
在 _preempt_request() 内部，调度器会释放该请求占用的 KV cache，并把请求状态改为 PREEMPTED：
self.kv_cache_manager.free(request)
self.encoder_cache_manager.free(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
request.spec_token_ids.clear()
request.num_preemptions += 1
最后，被抢占的请求会重新放回 waiting 队列，等待后续 scheduler step 再次被调度：
self.waiting.prepend_request(request)
也就是说，抢占流程可概括为：首先从 running 队列中选定目标请求，释放其占用的 KV cache block，并将状态改为 PREEMPTED；随后将该请求移回 waiting 队列，等待后续 step 重新调度。
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
class Scheduler(SchedulerInterface):
...
...
    while True:
        # 用于为一个请求分配若干个 KV Cache Block。
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens)
        # 资源不足，无法分配block
        if new_blocks is None:
            # The request cannot be scheduled.
            # Preempt the lowest-priority request.
            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(
                    self.running,
                    key=lambda r: (r.priority, r.arrival_time),
                )
                self.running.remove(preempted_req)
            else:
                preempted_req = self.running.pop()
            # 踢出请求preempted_req
            self.kv_cache_manager.free(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            preempted_req.num_computed_tokens = 0
            if self.log_stats:
                preempted_req.record_event(
                    EngineCoreEventType.PREEMPTED, scheduled_timestamp)
            # 放回 waiting 队列等待恢复，放在队首
            self.waiting.prepend_request(preempted_req)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                # 发现被踢的是自己
                # No more request to preempt.
                can_schedule = False
                break
        else:
            # The request can be scheduled.
            can_schedule = True
            break
    if not can_schedule:
        break
当 allocate_slots() 成功返回 new_blocks 时，表明当前请求已成功获取本轮执行所需的 KV cache block。随后，调度器会将其标记为本轮已调度的 running 请求，并记录其本轮分配的 block 信息以及需要计算的 token 数量。
scheduled_running_reqs.append(request)
req_to_new_blocks[request.request_id] = new_blocks
num_scheduled_tokens[request.request_id] = num_new_tokens
token_budget -= num_new_tokens
req_index += 1
此处关键变量的含义如下：
- scheduled_running_reqs：记录本轮成功调度且准备执行的 running 请求列表。
- req_to_new_blocks：建立请求 ID 与其本轮新分配 KV cache block 的映射关系。
- num_scheduled_tokens：记录每个请求本轮实际参与计算的 token 数量。
- token_budget：扣减本轮已消耗的 token 预算上限。
- req_index：指向 running 队列中下一个待处理的请求索引。
至此，该 running 请求在当前 scheduler step 中的调度工作即告完成。
调度逻辑总结
整体而言，running 队列的调度逻辑可概括为：调度器沿队列顺序逐一检查请求，计算其本轮待推进的 token 数，并尝试分配 KV cache block。在 token budget 与 block 资源均充足的前提下，该请求将被纳入本轮执行计划。
若 KV cache block 资源不足，调度器将触发抢占机制。遵循默认的 FCFS（先来先服务）策略，系统会优先选取 running 队列末尾的请求作为抢占目标。由于队尾请求介入执行的时间较晚，对其实施抢占对已长时间运行的请求造成的连续性中断影响最小。
特殊情况处理
此外，还需注意一种边界情况：若最终选定的抢占对象恰好是当前正在尝试调度的 request，代码中将体现为：
if preempted_req == request:
    break
此情形意味着调度器已无其他就绪请求可供抢占以腾挪资源。换言之，即便当前请求无法通过交换或抢占自身来满足本轮的资源分配需求，本次调度也将直接失败。
此时 new_blocks 仍保持为 None，调度器将中断当前的 running 队列扫描流程，转入后续的状态处理环节。
 class Scheduler(SchedulerInterface):
     def schedule(self) -> SchedulerOutput:
               
            while req_index < len(self.running) and token_budget > 0:
                # 开始遍历 running 队列，按 FCFS 顺序处理。
                request = self.running[req_index]
                ...
                ...
                # 跳过中间的代码
                # Schedule the request.
                scheduled_running_reqs.append(request)
                if request.use_structured_output:
                  # PERF: in case of chunked prefill,
                  # request might not include any new tokens.
                  # Therefore, we might introduce some additional
                  # cycle to fill in the bitmask, which could be a big no-op.
                  structured_output_request_ids[request.request_id] = req_index
                # 记录新分配的 Block IDs
                req_to_new_block_ids[request.request_id] = (
                    new_blocks.get_block_ids())
                # 记录本次调度的 token 数量
                num_scheduled_tokens[request.request_id] = num_new_tokens
                # 更新 token 预算
                token_budget -= num_new_tokens
                req_index += 1
暂时无法在飞书文档外展示此内容
调度器调度waiting队列
对应prefill阶段的请求
调度规则
在前面的流程中，调度器已经完成了对 running 队列的调度，也就是处理了当前已经处于运行状态的请求。接下来，调度器会尝试调度还没有开始执行的请求，也就是 waiting 队列中的请求。这一部分流程可以分为以下几个步骤。这里的编号与后面代码中的编号保持一致：
1. 调度 waiting 队列有一个前提：在调度 running 队列时没有发生资源抢占。也就是说，只有 preempted_reqs 为空时，调度器才会继续处理 waiting 队列。如果有请求被抢占，说明当前资源已经不足，调度器就不会再接纳新的 waiting 请求。
2. 另一个前提是当前系统负载还没有达到上限，也就是 token_budget > 0。这表示当前调度步中仍然有剩余 token 预算，可以继续为 waiting 队列中的请求分配计算资源。
3. 对于每一个 waiting 请求，调度器同样需要获取它已经计算过的 token 数量，也就是 num_computed_tokens。例如上图中的 Request 2 当前处于 WAITING 状态，如果它是一个全新的请求，并且没有命中前缀缓存，那么它的 num_computed_tokens 等于 0。但如果启用了 Prefix Caching，即使 Request 2 刚刚进入 WAITING 队列，只要它的 prompt 前缀和之前请求的 prompt 前缀重合，例如都以“请翻译以下段落：...”开头，那么这部分前缀就可以直接复用缓存。此时，num_computed_tokens 就不再是 0，而是缓存命中的 token 数量。
4. 接下来，调度器会计算当前请求还需要新增计算的 token 数量，也就是 num_new_tokens。它通常等于请求总 token 数减去 num_computed_tokens。同时，调度器还要确保这部分新增 token 数不会超过当前剩余的 token_budget，否则就不能在当前调度步中完整调度该请求。
暂时无法在飞书文档外展示此内容
[图片]
接上，在调度完 running 队列之后，调度器会判断是否可以继续处理 waiting 队列。这里的关键条件是 if not preempted_reqs：只有在本轮调度中没有发生请求抢占时，才会继续从 waiting 队列中取出新请求。因为一旦发生抢占，就说明当前系统资源已经比较紧张，调度器需要优先保证已有请求的执行状态，而不是继续接纳新的请求。
进入 waiting 调度流程后，还需满足3个条件：
1. waiting 队列不为空，并且 token_budget > 0。
2. 其中，token_budget 表示当前调度步剩余的 token 预算，只有预算还有余量时，调度器才会尝试为新的 waiting 请求分配计算资源。
3. 同时，如果当前 running 队列中的请求数量已经达到 max_num_running_reqs，调度器也会停止接纳新的请求。
对于每一个 waiting 请求，调度器首先要判断它是否已经有一部分 token 被计算过。
如果request.num_computed_tokens == 0，说明该请求尚未记录已完成的计算量。
此时，调度器会调用 kv_cache_manager.get_computed_blocks(request) 在本地 KV Cache 中查找可复用的前缀缓存。若启用了 connector，调度器还会进一步查询外部 KV 缓存（如远端存储或分布式 KV 传输中的缓存结果）。最终，num_computed_tokens 将由本地命中的 token 数与外部命中的 token 数相加得到，这部分 token 无需重新计算。
如果请求已具备 num_computed_tokens，通常说明异步 KV 接收已完成，调度器将直接复用该值。随后，若当前仍在异步加载远端 KV，调度器不会为其分配新的计算量；否则，将计算 num_new_tokens = request.num_tokens - num_computed_tokens，即该请求本轮实际需要计算的 token 数量，并通过 token_budget 对其进行限制，确保不超过当前调度步的资源预算。
 class Scheduler(SchedulerInterface):
     def schedule(self) -> SchedulerOutput:
        # 1. 如果现在调度running的时候，没有对其他请求进行剥夺的情况下才会调度
        if not preempted_reqs:
            # 2. 对waiting请求进行调度
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                   
                request = self.waiting.peek_request()
               
                num_external_computed_tokens = 0
                load_kv_async = False
                # Get already-cached tokens.
                # 3.1 判断当前请求是否有已经计算过的token
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    # 是否能够匹配中已经计算过的其他显存块
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)
                    # Get externally-cached tokens if using a KVConnector.
                    # 如果有外部存储（比如分布式存储或内存交换），还会去查外部缓存
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))
                    # Total computed tokens (local + external).
                    # 3.2 计算当前请求已经计算过的token数量，本地加外部，这部分Token是不需要重新计算的
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    # 3.2 计算当前请求已经计算过的token数量
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
               
                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # 4. 计算当前请求还需要计算的token数量
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    ...
                    ...
                    num_new_tokens = min(num_new_tokens, token_budget)
[图片]
对应上方代码处的注释：
1. 检查 if not preempted_reqs：在调度 waiting 队列之前，调度器会先判断此前处理 running 队列时是否发生了资源抢占。
  只有当 preempted_reqs 为空（即未发生任何请求抢占）时，才会继续调度 waiting 队列。若已发生抢占，说明当前 KV Cache 或系统资源已趋于紧张，调度器将不再接纳新的 waiting 请求。
2. 循环条件 while self.waiting and token_budget > 0：进入 waiting 队列调度逻辑后，调度器会持续从队首取出待执行请求。循环持续的条件包括：
  1. waiting 队列非空，且当前调度步仍有剩余的 Token 预算（token_budget > 0）。
  2. 此外，若 running 队列中的请求数量已达到上限 max_num_running_reqs，调度器同样会停止接纳新的 waiting 请求。
3. 获取当前 waiting 请求已计算的 Token 数量：调度器取出队首请求后，首先判断其是否已有可复用的计算结果。
  1. 若 request.num_computed_tokens == 0，说明该请求尚未记录已完成计算量，此时会调用 kv_cache_manager.get_computed_blocks(request) 查询本地 Prefix Cache 的命中情况；若配置了 connector，还会进一步查询外部 KV Cache。
  2. 最终，num_computed_tokens 为本地与外部命中的 Token 数之和。若 request.num_computed_tokens > 0，则通常表明该请求已完成异步 KV 接收，调度器将直接复用此值作为已计算的 Token 数量。
4. 计算当前请求本轮需新计算的 Token 数量：
  1. 若请求正处于异步加载远端 KV 的状态（即 load_kv_async == True），调度器不会为其分配新的计算任务，此时 num_new_tokens = 0。否则，调度器会先计算基础增量：num_new_tokens = request.num_tokens - num_computed_tokens，即总 Token 数扣除已计算或缓存命中部分后的剩余需求。
  2. 随后执行截断操作：num_new_tokens = min(num_new_tokens, token_budget)，利用当前剩余的 token_budget 限制本轮实际可调度的 Token 数量，确保不超出当前调度步的资源预算。
5. 这里的 num_computed_tokens 表示当前请求中已经计算完成、可直接复用的 token 数量，因此这部分 token 无需重新进入模型计算。它可以来自本地 KV Cache，也可以来自远程 KV Cache：
  1. 其中，num_new_local_computed_tokens 表示当前 vLLM 实例在本地 KV Cache 中命中的前缀 token，对应的 KV 数据已存在于本地显存中，可直接调
  2. num_external_computed_tokens 表示通过 KVConnector 在外部 KV 存储或其他节点中命中的 token，这些 token 虽已在逻辑上计算完成，但通常还需异步加载到本地后方可使用。
  3. 因此，最终的 num_computed_tokens 通常等于本地命中数与远程命中数之和，该数值主要用于确定当前请求在本轮调度中仍需实际计算的剩余 token 数量。
暂时无法在飞书文档外展示此内容
封装调度信息返回
class Scheduler(SchedulerInterface):
     def schedule(self) -> SchedulerOutput:
        if not preempted_reqs:
          
        while self.waiting and token_budget > 0:
            ...
            ...
            # 1. 为选中的请求申请显存块
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
            )
            if new_blocks is None:
                # The request cannot be scheduled.
                break
            
            # 2. 从waiting队列中删除这一请求
            request = self.waiting.pop_request()
            
            # 3. 加入到running队列中
            req_index += 1
            self.running.append(request)
     
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")
            # 4. 修改对应的状态和系统的负载数量
            req_to_new_block_ids[request.request_id] = (
                self.kv_cache_manager.get_block_ids(request.request_id))
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens # 更新系统当前step的负载
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
在前面的调度流程中，schedule() 已经完成了本轮 step 的核心调度决策：它会从 running 队列和 waiting 队列中选择可以执行的请求，并保证本轮要计算的 token 数量不超过 token_budget，同时也不会超过系统允许的最大运行请求数。
当请求被选中后，调度器会先为它申请 KV Cache 所需的显存块，即 block。如果显存块申请失败，请求就无法在当前 step 被调度；如果申请成功，请求会进入可执行状态。对于来自 waiting 队列的请求，调度器通常会将其移动到 running 队列，并把状态更新为 RUNNING。
在 schedule() 的当前 step 中，调度器还会用一个临时字典 req_to_new_blocks 记录本轮要传给执行端的 block 信息：
1. 对于已经在 running 队列中继续执行的请求，req_to_new_blocks 通常保存本轮新增的 block；
2. 对于刚从 waiting 队列进入 running 的请求，req_to_new_blocks 通常保存该请求当前关联的完整 block 列表。这个字典不会长期保存状态，它主要用于后面构造 SchedulerOutput。
可以把 SchedulerOutput 理解为调度器传给执行端的“任务清单”。它告诉后续的 worker：这一轮要执行哪些请求、每个请求要计算多少 token、这些请求对应哪些 KV Cache block，以及还有哪些辅助信息需要同步。SchedulerOutput 中最核心的信息主要包括三类。
1. 第一类是本轮每个请求需要计算的 token 数量。这个信息保存在 num_scheduled_tokens 中，形式为 req_id -> token 数量。调度器还会计算 total_num_scheduled_tokens，它等于所有请求本轮调度 token 数量的总和。这个总数必须不超过系统配置的 max_num_scheduled_tokens，否则当前 step 的计算负载就会超出限制。
2. 第二类是本轮调度到的请求。vLLM 会区分“首次被调度的请求”和“之前已经调度过的请求”。
  1. scheduled_new_reqs 表示本轮第一次被调度的请求。它们通常来自 waiting 队列，主要对应 Prefill 阶段。对于这类请求，worker 端还没有缓存过完整的请求元数据，因此调度器需要把请求信息、token 信息以及对应的 KV Cache block 信息一起发送过去。需要注意的是，即使是新请求，也可能通过 Prefix Caching 复用已有 block，此时它仍然属于 scheduled_new_reqs。
  2. scheduled_cached_reqs 表示之前已经调度过、worker 端已经缓存过元数据的请求。这里面包括本来就在 running 队列中继续执行的请求，也包括之前被抢占、现在恢复执行的请求。
3. 第三类是 KV Cache block 信息。vLLM 的 KV Cache 以 block 为单位管理，一个 block 可以存放多个 token 对应的 KV Cache。这里调度器保存的并不是 KV Cache 张量本身，而是用于定位这些缓存位置的 block 编号信息。
在调度器内部，这部分信息会先暂存在 req_to_new_blocks 中。它的形式是：
request_id -> KVCacheBlocks
调度器不会直接传递 KV Cache 数据本身，而是传递 KV Cache block 的编号信息。Worker 根据这些 block 编号，找到对应的显存位置，完成本轮计算中的 KV Cache 读写。执行端收到 SchedulerOutput 后，会根据这些 block_ids 更新自己的 block table。
- 对于新请求，worker 会用 NewRequestData.block_ids 初始化请求状态；对于已有请求，worker 会把 CachedRequestData.new_block_ids 追加到已有的 block table 中；
- 如果请求是被抢占后恢复的，则可能会用新的 block 信息替换旧的 block 信息。
[图片]
除此之外，SchedulerOutput 还会携带一些辅助信息，例如 speculative decoding 的 token、encoder input、公共前缀 block 数量、被抢占的请求 ID、已经完成的请求 ID，以及需要清零的新 block ID 等。这些信息不是调度主流程的核心，但会影响后续的模型执行、缓存管理和 worker 状态同步。总体包含以下信息：
1. 做什么：本轮参与执行的请求，即 scheduled_new_reqs 与 scheduled_cached_reqs.req_ids。  
2. 做多少：每个请求在本轮调度中生成的 token 数量，即 num_scheduled_tokens。  
3. 用哪些缓存资源：新请求的完整 block_ids，缓存请求的 new_block_ids，以及必要时需要清零的 new_block_ids_to_zero。
没有这个结构化的输出，执行引擎就无法知道该如何高效、正确地执行模型计算。它是连接调度逻辑和计算执行的关键桥梁。
class Scheduler(SchedulerInterface):
     def schedule(self) -> SchedulerOutput:
        ...
        ...
        # 1. 统计本轮中需要计算的token数量
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))
        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))
        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )
        # 2. 统计要执行的block块
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )
       
       
        self._update_after_schedule(scheduler_output)
        return scheduler_output
在完成 token 数量和运行队列约束检查后，调度器还需要整理本轮执行所依赖的 KV Cache block 信息。这里需要注意，调度器并不会把 KV Cache 的实际数据传给执行端，而是传递 block 的编号信息。
这些 block 编号会先保存在 req_to_new_block_ids 中。req_to_new_block_ids 是调度器在 schedule() 内部创建的一个临时字典，通常在本轮调度开始时初始化为空：req_to_new_block_ids: dict[str, list[int]] = {}
它的作用是记录：request_id -> 本轮为该请求分配或关联到的 KV Cache block IDs，也就是说，key 是请求的 request_id，value 是这个请求本轮需要使用的 block 编号列表。前面调度请求时，调度器会为请求申请新的 block，或者查询请求当前已经关联的 block，然后把结果写入 req_to_new_block_ids。
接下来，调度器会根据请求类型把这些 block 信息封装到不同的数据结构里：对于本轮首次调度的新请求，block 信息会写入 NewRequestData；对于已经运行过或被抢占后恢复的请求，block 信息会写入 CachedRequestData。最后，这些信息会统一放进 SchedulerOutput，由执行端根据其中的 block_ids 更新自己的 block table，从而知道本轮模型计算产生的 KV Cache 应该写入哪些显存块。
调度执行之后
我们在之前的步骤中选择当前step需要计算出哪些请求，以及这些请求需要用到的显存块，那么随后我们就要对请求的状态进行更新，包括对某个请求已经执行计算过的token数量进行更新，也就是_update_after_schedule对应的流程，它主要的工作就是更新当前被调度的请求中已经计算的token数量。
class Scheduler(SchedulerInterface):
    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
      
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
num_scheduled_tokens 来自本轮刚刚构造好的 SchedulerOutput，它记录的是 request_id -> 本轮为该请求调度的 token 数量。
因此，_update_after_schedule() 的核心作用就是把这些已经被调度出去的 token 数量累加到对应请求的 num_computed_tokens 上。这里的 computed 可以理解为调度器视角下已经被安排计算：
虽然在 schedule() 结束时 GPU 还没有真正完成本轮模型计算，但调度器会先把这些 token 视为即将完成，并提前推进请求状态。这样下一轮调度开始时，调度器才能知道每个请求已经推进到了哪里，避免重复调度同一段 token，造成 token 进度错误、KV Cache 写入位置混乱，甚至影响 Prefill / Decode 阶段的判断。
总结
1. 请求接入与初始状态：客户端请求通过 ZMQ 异步发送到 EngineCoreProc，随后由 input_queue 消费，并调用 add_request() 加入调度器。新请求会先进入 waiting 队列，状态被设置为 WAITING。
2. 调度优先级原则：调度器整体遵循“已激活请求优先、新请求次之”的思路。也就是说，已经进入 running 队列的请求会优先被继续调度，以保证生成过程的连续性；随后才会从 waiting 队列中选择新请求或恢复的请求。在同一队列内部，通常按照请求到达顺序进行处理，从而减少请求长期等待的问题。
3. running 队列调度：调度器会优先处理已经在运行中的请求，这些请求通常处于 Decode 或后续增量计算阶段（Chunked Prefill）。调度器会根据本轮剩余的 token_budget 为它们安排要计算的 token，并申请对应的 KV Cache block。如果资源不足，调度器可能会触发抢占，释放某些低优先级或队尾请求占用的 KV Cache，并把被抢占的请求放回 waiting 队列，等待后续重新调度。
4. waiting 队列调度：当 running 队列处理完成后，如果本轮仍有剩余 token_budget，并且没有因为抢占导致调度受阻，调度器会继续从 waiting 队列中取出请求。对于能够成功分配 KV Cache block 的请求，会将其加入 running 队列，并把状态更新为 RUNNING。
5. 资源与负载控制：调度器通过 token_budget 控制单个 step 中最多调度多少 token，同时还会受最大运行请求数、KV Cache 可用 block 数等条件约束。对于较长 prompt，vLLM 可以通过 chunked prefill 将其拆分到多个 step 中执行，从而在吞吐、延迟和显存占用之间取得平衡。
6. 调度输出与状态更新：调度完成后，调度器会构造 SchedulerOutput，其中包含本轮要执行的请求、每个请求要计算的 token 数量、对应的 KV Cache block 编号，以及一些辅助信息。随后，调度器会调用 _update_after_schedule() 更新请求状态，例如累加 num_computed_tokens，使下一轮调度能够基于最新的请求进度继续推进。