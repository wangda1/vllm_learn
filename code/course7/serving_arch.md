和视频内容比，新增了 vLLM v1 架构总览章节。
从前文可以看出，ModelRunner 是每个工作进程中实际执行模型计算的核心组件。在启用了张量并行的推理服务中，每个 ModelRunner 实例都会持有当前 rank 对应的模型权重分片，并维护与该分片配套使用的 KV Cache。除此之外，它的另一项关键职责，是在接收到调度器输出的 SchedulerOutput 之后，整理本轮 forward 所需的输入数据，并驱动实际的模型推理执行。
而ModelRunner 在收到调度结果后，如何基于请求当前已经分配的 KV Cache blocks，构造注意力计算所需的 block tables，以及如何进一步计算每个 token 对应的 slot mapping。这里需要特别说明，block table 记录的是“一个请求当前使用了哪些物理 block”；而后续 attention kernel 真正直接使用的，是由 token 位置进一步映射得到的 slot mapping，也就是每个 token 在 KV Cache 中对应的物理存储槽位。
这两类信息都是后续推理过程，尤其是 PagedAttention 机制，能够正确访问 KV Cache 的基础。
一 vLLM v1 架构总览
1.1 vLLM v1 架构核心类介绍
vLLM 对外提供了多个系统入口，最常见的有两类：一类是面向离线推理的 LLM 类，另一类是面向在线服务的 OpenAI 兼容 API 服务器，即 vllm serve 命令。下图展示了它们之间的调用关系。
LLM 类和 vllm serve 命令的基本使用方式都比较直接，第一章已经介绍过，这里不再展开。
从图中可以看到，离线推理入口 LLM 类会进一步构造 LLMEngine；而 vllm serve 对应的 OpenAI 兼容服务入口，则会构造 AsyncLLM，用于承接异步请求处理与服务化推理流程。
二者分别对应离线批处理和在线服务两种典型场景，因此可以认为，LLMEngine 和 AsyncLLM 是 vLLM v1 在执行层最核心的两个上层运行组件：前者主要负责离线推理任务的组织与执行，后者则在此基础上进一步承担异步请求接入、调度和服务化处理等职责。
[图片]
从上图可以看到，离线推理入口 LLM 类会进一步调用 LLMEngine，而在线服务入口 vllm serve 则会构造 AsyncLLM。因此，LLMEngine 和 AsyncLLM 可以看作是 vLLM v1 中两个最核心的上层运行组件：前者主要负责离线推理流程的组织与执行，后者主要负责在线场景下的异步请求处理与服务化推理。
[图片]
1. LLMEngine 和 AsyncLLM 介绍
  LLMEngine 是 vLLM v1 中面向离线推理场景的核心执行引擎。它负责接收上层提交的推理请求，并组织一次完整的推理流程，包括输入处理、请求提交到 EngineCore、模型执行结果回收，以及最终输出结果的整理。
  从职责上看，LLMEngine 主要包含以下几个部分：
  - 输入处理：将用户输入转换为内部可执行的请求表示，必要时结合 tokenizer、renderer 和 input processor 完成预处理。
  - 请求提交与调度衔接：将处理后的请求送入底层 EngineCore，由后者负责实际的调度与执行推进。
  - 模型执行：通过 EngineCore 和 executor 驱动模型运行，底层执行可以分布在多个 GPU，甚至多个进程中。
  - 输出处理：接收 EngineCore 返回的结果，并将模型输出整理为上层可直接使用的格式，例如可读文本或结构化结果。
  对应代码位于 vllm/v1/engine/llm_engine.py。
  AsyncLLM 则是 vLLM v1 中面向在线服务场景的异步执行入口。它并不只是对 LLMEngine 做一层简单的异步封装，而是独立维护了一套异步请求处理流程：一方面接收并发请求，另一方面通过异步方式与底层 EngineCore 交互，并持续回收输出结果，在需要时以流式方式返回给客户端。AsyncLLM 的几个关键特点是：
  - 面向并发请求：适合在线推理和服务化部署场景。
  - 异步输出处理：内部会启动异步 output handler，持续处理底层返回结果。
  - 支持流式返回：能够将生成结果按 token 或按阶段逐步推送给客户端。
  - 底层仍依赖同一套核心执行体系：它同样包含 input processor、output processor 和 engine core，只是整体控制流改成了异步模式。
  对应代码位于 vllm/v1/engine/async_llm.py。
2. EngineCore 介绍
  EngineCoreProc 是 vLLM v1 中用于在后台进程内运行 EngineCore 的核心进程类。它继承自 EngineCore，在保留调度与执行主循环的基础上，进一步增加了多进程部署所需的进程间通信机制。具体来说，EngineCoreProc 会在独立后台进程中运行，并通过 ZeroMQ 套接字与前端组件交换请求和输出结果；同时，它还维护输入线程和输出线程，用于在 socket 与进程内队列之间搬运数据，从而将通信、序列化和模型执行尽可能重叠起来。
  EngineCoreProc 内部包含一个持续运行的 busy loop。这个循环会不断处理输入队列、推进调度器执行，并调用底层 executor 完成模型 forward，随后再将执行结果写回输出队列，由输出线程发送给前端。也就是说，它实际上承担了 vLLM 后台推理主循环的驱动职责。
  在数据并行场景下，vLLM 还提供了 DPEngineCoreProc。它是在 EngineCoreProc 基础上的扩展版本，增加了数据并行相关的协调逻辑，例如全局 unfinished requests 状态同步、wave 推进，以及多 DP rank 之间的运行状态协同。需要注意的是，在当前 vLLM v1 实现中，DPEngineCoreProc 主要用于 MoE 模型的数据并行执行场景。
3. Scheduler Executor 介绍
  Scheduler 是在 EngineCore 内部的核心调度组件，负责决定每一步应该执行哪些请求，以及这些请求在当前 step 中各自能够推进多少 token。它一方面要管理请求状态，另一方面还要结合 KV Cache 使用情况、token budget、最大并发请求数等约束，生成可执行的批次。
  在 vLLM v1 中，Scheduler 主要维护的请求集合包括 waiting、skipped_waiting 和 running。其中，waiting 表示尚未进入执行阶段的请求，running 表示已经在持续推进的请求，而 skipped_waiting 则用于暂存那些由于异步依赖或资源约束而暂时不能进入本轮调度的请求。调度时，Scheduler 会根据策略和资源限制，综合处理 prefill 与 decode 请求，从而尽量提高吞吐并控制延迟。
  Executor 是实际执行推理计算的抽象层，它位于 Scheduler 和 Worker 之间，主要作用是将调度器（Scheduler）决定的计算任务分发给底层的 Worker 进程，并协调它们的执行。它会直接管理和协调 Worker 进程或者或 Ray Actor（如果是分布式 Ray 运行）
4. Worker 介绍
  Worker 是 vLLM 中实际执行模型推理的基本执行单元。在常见的 GPU 部署方式下，vLLM 通常采用“一个 worker 对应一个加速设备”的组织方式；在多进程执行模式下，这通常就体现为“一个进程控制一个 GPU”。例如，当张量并行度为 2、流水线并行度为 2 时，整体会有 4 个 worker 共同参与模型执行。
  每个 Worker 都会负责本 rank 上的模型加载、设备初始化、KV Cache 管理以及具体的 forward 执行。因此，Scheduler 产出的执行任务最终都会被下发到各个 worker，由它们在各自负责的设备上真正完成计算。
  在标识方式上，Worker 通常会涉及两个重要概念：
  - rank：全局分布式 rank，用于整个分布式执行过程中的协调与通信。
  - local_rank：本地 rank，更准确地说是本地设备索引，用于确定当前 worker 应该绑定哪一个本地 GPU，并访问对应的本地资源。
5. Model Runner 介绍
  每个 Worker 内部都会维护一个 ModelRunner 对象，它负责承载当前 worker 上的模型执行逻辑。ModelRunner 不仅负责驱动模型的实际 forward 计算，还负责准备执行所需的输入张量、维护与本 worker 对应的请求状态和 KV Cache 状态，并在需要时完成 CUDA Graph 的捕获与复用。因此，vLLM 中大量与模型执行直接相关的核心逻辑，都会集中在 ModelRunner 中实现。
6. 模型对象
  每个 Model Runner 对象都含有一个模型对象，即实际的 torch.nn.Module 实例。
  LLMEngine、Executor、Worker、ModelRunner、Model 类 5 个类对应的类层次结构关系图如下所示:
[图片]
1.2 vLLM 核心组件关系图解
vLLM v1 在典型多进程部署下，整体链路可以概括为：API Server → EngineCoreProc → Executor → Worker。
其中，EngineCoreProc 是后台引擎进程，内部包含 Scheduler 和 Executor：Scheduler 负责调度请求，Executor 负责协调底层 Worker 执行模型；在多 GPU 场景下，通常每个 Worker 对应一张 GPU。
以下是 vLLM v1 的典型架构图，展示了这些组件的交互：
[图片]
以及另一个视角的组件关系图，聚焦于 DPEngineCoreProc 在数据并行下的多副本结构，每个包含 EngineCore和 MultiProcExecutor：
[图片]
以及 EngineCore 的内部流程图，可以突出 schedule() 和 execute() 的循环：
[图片]
图中，EngineCore 内部的关键交互流程总结如下:
1. 上层组件将请求发送到 EngineCoreProc，请求首先经由输入 socket 线程进入进程内输入队列。
2. EngineCoreProc 的 busy loop 持续处理输入队列，并驱动 Scheduler 执行调度，决定本轮需要推进哪些请求以及各自推进多少 token。
3. Scheduler 生成 SchedulerOutput 后，EngineCore 将其交给 Executor。
4. Executor 负责协调底层 worker，并在各个 worker 上触发实际的模型执行。
5. 每个 Worker 在自己负责的设备上完成本轮计算，再将结果逐步返回给 Executor 和 EngineCore。
6. EngineCoreProc 将结果写入输出队列，并由输出线程发送回上层组件，例如 AsyncLLM。
暂时无法在飞书文档外展示此内容
1.3 vLLM 整体运行流程图
前面的内容让我们对 vLLM 架构的核心组件（模块）、相互关系已经有了个大致了解，下面我们再通过一张图了解vLLM v1 架构的整体运行流程，从全局的角度来理解关键模块的运行逻辑，方便我们后续学习关键特性。
图片来源知乎文章-vLLM不知如何开始？看这篇：vLLM框架快速入门引导。截止到 2026.2.1 vLLM 已经升级到了 0.15.0 版本，其中的部分类可能已经更新名称或者弃用了，但是大致运行逻辑和用到的技术方案没有变。
[图片]
上图中的 TokenizerGroup 是 vLLM 0.10 及之前版本有的，最新版 0.13.0+ 已经弃用，通过 engine.get_tokenizer() 函数获取 tokenizer 实例，类型为 AnyTokenizer。
AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerBase]
上图中的，AsyncLLM 与 engine core 运行在不同的进程中，前者属于客户端进程，后者属于服务端进程，两者通过队列(queue)交互。engine core 的任务由 executor 下发，多个 worker 共同完成 LLM 的端到端推理。一般情况下，每个worker 拥有一张 GPU 卡，多 worker 可实现 TP/SP/EP/PP 等并行策略。
二 ModelRunner 执行模型推理
2.1 模型的执行
在之前的课程中，我们讲解了调度器如何根据请求的等待时长及其所处的状态进行请求调度，最终生成当前步骤的调度结果，即 SchedulerOutput。该结构体包含多个重要字段，我们在此回顾其中几个关键部分：
1. scheduled_new_reqs：表示本轮首次被调度的请求，通常是刚从 waiting 队列中进入执行阶段的新请求；
2. scheduled_cached_reqs：表示之前已被调度过的请求，包括正处于 decoding 阶段的请求，或曾被抢占但在当前步骤恢复执行的请求。这类请求与新请求不同，需采用不同的处理方式；
3. 此外还需关注的是 num_scheduled_tokens 字段，记录本轮中每个请求被调度执行的 token 数；同时还有 finished_req_ids，表示在当前步骤已完成的请求 ID 列表。
暂时无法在飞书文档外展示此内容
因此，在模型执行开始之前，需要先依据本轮 SchedulerOutput 对 ModelRunner 内部维护的请求状态进行同步更新。对于本轮首次被调度的请求，ModelRunner 需要为其建立初始状态，包括请求状态、模型侧缓存状态以及对应的 block table 记录；而对于已经执行过的请求，则需要根据本轮新增分配的 block，对已有的 block table 进行追加更新。比如，若某个请求在当前 step 新增使用了 block_id = 31, 33，那么 ModelRunner 中该请求对应的 block 记录也需要同步追加这两个 block。
这一系列状态同步操作均发生在 self._update_states 方法中。
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, IntermediateTensors]:
    self._update_states(scheduler_output)
2.2 更新请求
删除过时的请求
在 ModelRunner 中维护的 input_batch 用于保存当前批次中各个请求的核心执行状态，包括 token 序列、block table、采样参数以及已生成的输出 token 等信息。它的设计目标不是在每一步重新构造一个全新的 batch，而是尽量复用已有的数据结构和槽位，从而降低输入准备阶段的 CPU 开销。
1. 在初始化阶段，input_batch 会预先创建一组固定上界的 CPU/GPU 缓冲区。例如，token_ids_cpu 会分配为 max_num_reqs × max_model_len 的 CPU 张量，用于保存各请求的 token 数据；与之相关的 num_prompt_tokens、num_computed_tokens、采样参数缓冲区以及 block table 等结构，也会在这一阶段建立。
2. 在后续每一轮调度中，系统通常只对发生变化的部分做增量更新，例如写入新请求的 prompt、追加上一步新生成的 token、补充 block table 中新增的 block，以及更新对应的采样参数。也就是说，input_batch 会尽量复用已有槽位，而不是在每一步重新分配整批内存。
3. 真正执行模型前，ModelRunner 会根据本轮调度结果，从 input_batch 中整理出本 step 实际需要执行的输入，并准备相应的执行元数据，例如 input_ids、block table、position、slot mapping 以及已计算 token 数等。随后，这些本步实际需要参与计算的输入和元数据会被同步到 GPU，并驱动本轮前向计算。因此，拷贝到 GPU 的并不是所有请求的完整历史 token 序列，而是当前步真正需要使用的那一部分数据。
在 gpu_model_runner.py 的 _update_states 方法中，首要任务之一是清理已经结束的请求。系统会先移除这些请求对应的 cached state，再将其从 input_batch 中删除，同时清理与之关联的批次内部记录，例如 block table 中对应的行、token 序列槽位以及输出 token 列表等。
这样做的原因在于，这些已完成请求不会再参与当前步及后续步骤的调度与执行；如果不及时回收它们占用的槽位，就会影响新请求加入、被抢占请求恢复，以及其他活跃请求的持续解码。下图展示的就是释放 req1 及其相关 batch 资源的过程。
暂时无法在飞书文档外展示此内容
被终止的请求主要包括两类：
1. 自然结束的请求：在当前生成过程中输出了终止符（如 end_token）或达到最大长度限制而正常完成的请求；
2. 强制终止的请求：因用户主动中止、超时、被抢占且无法恢复，或其他异常情况导致提前退出的请求。
这类请求一旦完成或被取消，其在 input_batch 中占用的资源必须及时释放，以确保后续调度能够高效复用内存空间，避免资源泄漏和状态混淆，如 block table、输出 token 列表和采样状态等。这是已经完成的请求。
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            # 移除本次没有被调度的请求
            self.input_batch.remove_request(req_id)
一些请求在当前步长中未被调度，因此也需要像已结束的请求一样，需要从 input_batch 中移除，但仍保留在 self.requests，以便未来某一步重新被调度时再加回 batch。
为了识别这些请求，系统将请求集合划分为两类：
- scheduled_req_ids：表示本轮中被实际调度的请求（即有 token 被安排执行的请求）；
- cached_req_ids：表示当前 input_batch 中已记录的、处于活跃状态的请求 ID 集合。
所以这些 unscheduled_req_ids 通常包括两类：
1. 被抢占的请求
2. 仍处于运行状态、但本轮没有获得调度的请求
通过计算 cached_req_ids - scheduled_req_ids，即可得到本轮未被调度的请求集合，记为 unscheduled_req_ids。这些请求在之前的步骤中曾被调度并保留在批处理中，但在当前步骤由于资源限制或其他原因未能继续执行。因此，在模型执行前，必须将这些未被调度的请求从 input_batch 中清除，也就是input_batch.remove_request(req_id)。
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    ...
    ...
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)
处理新增请求
暂时无法在飞书文档外展示此内容
本轮新增的请求包含两类：
1. 全新请求：指在当前步骤中首次被调度的请求，通常来自之前处于 waiting 状态的等待队列；
2. 恢复类请求：指此前已出现过但因抢占、暂停等原因未连续执行的请求，包括被剥夺资源后重新提交的请求，以及此前已部分执行（如已完成 prefill 阶段）但尚未完成的请求。
在 ModelRunner._update_states() 中，系统首先处理 scheduled_new_reqs，但不会立即把请求写入 input_batch。相反，它会先根据调度结果构造 CachedRequestState，并将请求的关键运行状态缓存到 self.requests 中。这些状态包括请求 ID、prompt token、block table 对应的 block_ids、已计算 token 数 num_computed_tokens，以及用于保存后续生成结果的 output_token_ids。其中，对于全新请求，output_token_ids 会初始化为空列表。
在完成状态初始化之后，系统再统一调用 input_batch.add_request()，把这些请求真正加入当前的InputBatch。
remove_request删除本轮没有被调度的请求，add_request增加本轮新增需要执行的请求。由于这些请求尚未在 input_batch 中分配位置，因此无需修改现有条目，而是在后续阶段通过 add_request 等操作将其正式加入批处理上下文中。这里比较重点的几个参数：
1. prompt_token_ids：表示该请求的原始输入 token 序列，也就是用户 prompt 经过 tokenizer 之后得到的 token；
2. num_computed_tokens: 这个请求通过前向计算已经写入kv cache的上下文长度，因为有的step中不能一次处理完所有的prompt_tokens，也需要分批进行处理计算，也就是大家常说的chunked prefill；
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
现在处理的是新增的请求（之前没有被处理/计算过的请求）
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    # 在我们的例子，就是对新增的请求2相关数据
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params
    
        if sampling_params and \
        sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None
            # 需要被增加到input_batch中的新增请求
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_kwargs=new_req_data.mm_kwargs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            req_ids_to_add.append(req_id)
现在需要处理的是之前已被调度过的请求，即 scheduled_cached_reqs。这类请求与新增请求不同：它们在之前的步骤中已经被加入到 self.requests 中，因此当前不需要重新创建状态，而是对 self.requests[req_id] 中已存在的请求状态进行更新。
其中最关键的更新项是显存页表（block table），即 req_state.block_ids 字段。该字段记录了当前请求所使用的物理 block ID 列表，必须根据本次调度结果进行同步更新。具体分为两种情况：
1. 如果请求是被剥夺然后再恢复运行的也就是resumed_from_preemption等于true的情况，此类请求在之前因资源不足被抢占，其原有的显存块可能已被系统回收（free），甚至被其他请求复用。因此，在恢复执行时，它通常会被分配一组全新的 block。此时不能简单地在原有 block_ids 上追加，而应直接替换整个页表为新的 new_block_ids。
2. 相对的如果当前请求没有被剥夺过，此类请求一直保留在批处理中，未发生中断。其原有的 block 资源仍然有效，只需将本次调度新分配的 block（new_block_ids）追加到现有页表末尾即可，采用 extend 操作完成扩容。
此外，若该请求当前尚未存在于 input_batch 中（即不在 req_id_to_index 映射中），说明它是从抢占状态恢复或此前未被包含在当前批次中，则需将其请求 ID 加入 req_ids_to_add 列表，以便后续通过 add_request 重新纳入批处理结构。
例如，对于 req_id = 1 的请求，假设它当前已经使用了 3 个 block，编号分别为 31、32、33。如果本轮调度又为它分配了一个新的 block，例如 45，那么系统会先把这个新 block 追加到该请求的 req_state.block_ids 中。之后：
1. 如果该请求当前已经在 input_batch 中，那么新的 block 会通过 input_batch.block_table.append_row(...) 直接同步到 batch 的块表结构中。
2. 如果该请求当前不在 input_batch 中，那么它会先被加入 reqs_to_add，随后在 input_batch.add_request() 时把更新后的 block 信息整体写回 batch。
暂时无法在飞书文档外展示此内容
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    req_data = scheduler_output.scheduled_cached_reqs
    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_data.resumed_from_preemption[i]
        # Update the cached states.
        # 更新请求中已经计算过的tokens
        req_state.num_computed_tokens = num_computed_tokens
        
        if not resumed_from_preemption:
            # 如果当前请求是之前执行过
            # Append the new blocks to the existing block IDs.
            for block_ids, new_ids in zip(req_state.block_ids,
                                          new_block_ids):
                block_ids.extend(new_ids)
        else:
            # 如果当前请求是之前执行过但是被剥夺的
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids
        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            req_ids_to_add.append(req_id)
            continue
增加 block 页表
对于一个已存在的请求（如下图所示），最重要的一项操作就是在每次调度执行前对其页表进行更新。例如，假设该请求在之前的调度中使用了 3 个 block，且每个 block 的大小为 2，则其最多可容纳 6 个 token。若本次调度需要处理一个新的 token（即第 7 个 token），而当前的页表容量已不足以容纳该 token 所需的 KV Cache 存储空间，就必须为该请求分配一个新的 block，并将其 block ID 添加到 InputBatch 中对应请求的页表记录中。
因此，系统会执行相应的 block 分配与页表更新流程，确保该请求在后续推理过程中能够正确访问新增的显存块。整个过程如图所示。
暂时无法在飞书文档外展示此内容
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    ...
    ...
    self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens)
    self.input_batch.block_table.append_row(new_block_ids, req_index)
在新一轮调度开始前，系统可能需要为请求分配新的显存块并更新页表。对于全新请求，调度器会先生成该请求对应的 block_ids，随后在 add_request() 中连同 prompt_token_ids、num_computed_tokens 等状态一起写入 input_batch，完成块表初始化。
对于已存在请求，若本轮新分配了 block，则系统会先更新 req_state.block_ids，随后再将新增 block 同步到 InputBatch 的块表结构中。
def _update_states(...):
    if not resumed_from_preemption:
        if new_block_ids is not None:
            # 将新块追加到现有块ID中
            for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                block_ids.extend(new_ids)
                
    self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
    if new_block_ids is not None:
        self.input_batch.block_table.append_row(new_block_ids, req_index)
暂时无法在飞书文档外展示此内容
无论是本轮首次被调度的新请求，还是此前已存在但当前不在 input_batch 中、需要重新加入 batch 的恢复请求，都会通过调用 add_request() 被写入 InputBatch，从而在 persistent batch 中重新建立其批级状态。
这些关键信息主要包括：token 序列、块表映射、采样参数、已生成 token 列表以及对应的内存索引等。具体实现细节可参考 InputBatch 类中的 add_request 函数。
暂时无法在飞书文档外展示此内容
2.3 模型输入构建
这一节关注 class GPUModelRunner 中的 _prepare_inputs 函数，输入就是上个阶段更新完毕的 inputBatch
当请求在上一阶段通过 _update_states() 完成状态更新后，ModelRunner 接下来会调用 _prepare_inputs()，根据最新的批次状态准备模型执行所需的输入数据。此时，一个关键步骤是将 InputBatch 中维护的块表信息提交到 GPU。对应代码是：
self.input_batch.block_table.commit_block_table(num_reqs)
这里的作用是把当前 batch 中各个请求对应的 block table 同步到 GPU，使后续 Attention 算子能够根据 token 所属位置，查找到对应 KV Cache 在显存中的物理 block 位置。
对于某个请求来说，Attention 不仅需要知道它当前占用了哪些 block，还需要知道本轮究竟有多少个 token 要参与计算，以及这些 token 在整个 batch 展平后的输入序列中处于什么位置。因此，在 _prepare_inputs() 中，系统还会进一步构造一组执行元数据，其中非常关键的一项就是 query_start_loc。query_start_loc 表示每个请求在本轮 batch 展平后，其 token 区间起始位置的前缀和。源码中的构造方式是：
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
self.query_start_loc.copy_to_gpu()
其中，cu_num_tokens 是每个请求本轮调度 token 数量的前缀和。
例如，query_start_loc = [0, 1, 4, 6, 9, 16]，则表示：
- 第 1 个请求的 token 范围是 [0, 1)，共 1 个 token
- 第 2 个请求的 token 范围是 [1, 4)，共 3 个 token
- 第 3 个请求的 token 范围是 [4, 6)，共 2 个 token
- 第 4 个请求的 token 范围是 [6, 9)，共 3 个 token
- 第 5 个请求的 token 范围是 [9, 16)，共 7 个 token
准备输入
如下图所示，Attention 算子会接收一个关键输入 query_start_loc。它表示当前 batch 中各个请求本轮待计算 token 数量的前缀和，因此可以用来确定每个请求在 batch 展平 token 序列中的起始位置与结束位置。s
例如，若 query_start_loc = [0, 1, 4, 6, 9, 16]，则说明：
- 第一个请求的 token 从索引 0 开始；
- 第二个请求从索引 1 开始；
- 第三个请求从索引 4 开始；
- 以此类推。
暂时无法在飞书文档外展示此内容
因此，query_start_loc 的作用是为每个请求划分出其在整体序列中的“边界”，从而实现多请求并行处理时的精确寻址。接下来我们分析 Attention 算子的计算流程：
1. 假设本次调度共需处理 16 个 token，这些 token 来自 5 个不同的序列（seq），系统会将所有 token ID 以一维扁平化的方式存储在一个连续数组中。也就是说，该一维数组按顺序拼接了所有请求的 token IDs。
2. 经过词嵌入（embedding）操作后，这些 token IDs 被映射为对应的嵌入向量，形成一个形状为 [total_num_tokens, hidden_dim] 的张量，其中 hidden_dim 表示每个 token 映射后的向量维度。
3. 为了从这个统一的 embedding 张量中提取属于各个请求的子序列，就需要借助 query_start_loc。以第二个请求（index=1）为例，其对应的 embedding 向量范围为：
embeddings[query_start_loc[1] : query_start_loc[2], :]  # 即 embeddings[1:4, :]
其形状为 [3, hidden_dim]，恰好对应该请求的 3 个 token 的嵌入表示。由此，query_start_loc 成功实现了对不同请求 token 的高效切分和定位。
暂时无法在飞书文档外展示此内容
综上所述，上述所有准备工作都是为了支持 PagedAttention 机制的高效运行。PagedAttention 需要准确知道每个请求的 token 在全局序列中的起始和结束位置，从而能够从扁平化的一维 input_token_ids 序列中精确定位并提取出该请求对应的 embedding 向量。
1. 具体而言，query_start_loc 描述的是当前 batch 中各请求本轮待计算 token 数量的前缀和，它用于标记每个请求在本轮展平 token 序列中的起始位置和结束位置，是 Attention 划分各请求 query 区间的重要依据。
2. input_ids 则是本轮真正送入模型执行的扁平化 token ID 序列。这些 token 并不是直接把所有请求完整拼起来，而是根据当前调度结果，从 input_batch 保存的 token 缓存中按需抽取出来的。
接下来，我们重点分析 input_ids 和 query_start_loc 的构造过程。其中特别关键的一步，是先将 token_ids_cpu_tensor 展平成一维数组，再通过 token_indices 把每个请求内部的 token 位置映射为全局线性下标，从而抽取出当前步中真正需要执行的输入 token：
token_indices = (positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1])
随后，系统通过 token_indices 计算出本轮每个待执行 token 在这个一维数组中的位置。
- 请求0的第5个token: position=5, 请求0的block id =0 → index = 5 + 0×M = 5
- 请求1的第3个token: position=3, 请求1的block id =1 → index = 3 + 1×M = 3 + 32 = 35
- 请求2的第7个token: position=7, 请求2的block id = 2  → index = 7 + 2×M = 7 + 64 = 71
这里的M就是每个请求最长的 max_model_len 长度的token序列空间，这里假设M等于32，所以在index_select的时候我们直接用以上的5, 35, 71这几个索引从tokens_ids_cpu_tensor中获取对应的token id放入到input_ids_cpu。
举例
1. req_indices 是什么？
源码里会先展开出每个待执行 token 属于哪个请求：
req_indices = [0, 0, 1, 1, 1]
含义：
- 前 2 个 token 属于 req 0
- 后 3 个 token 属于 req 1
含义：
- 前 2 个 token 属于 req 0
- 后 3 个 token 属于 req 1
2. positions_np 是什么？
再假设这两个请求当前的：num_computed_tokens_cpu = [3, 2]
表示：
- req 0 前 3 个 token 已经计算过了，所以本轮从位置 3 开始算，req 0 本轮两个 token 对应请求内位置 3, 4
- req 1 前 2 个 token 已经计算过了，所以本轮从位置 2 开始算，req 1 本轮三个 token 对应请求内位置 2, 3, 4
那么本轮每个 token 在各自请求内部的位置就是：
positions_np = [3, 4, 2, 3, 4]
3. token_ids_cpu_tensor 是什么？
它就是保存所有请求 token 的二维表：
token_ids_cpu_tensor =
[
  [101, 102, 103, 201, 202,   0,   0,   0],
  [111, 112, 113, 114, 301,   0,   0,   0],
  [  0,   0,   0,   0,   0,   0,   0,   0],
]
每一行一个请求，每一列是该请求里的 token 位置。
4. token_indices 是什么？
现在把二维坐标映射成一维下标。
因为：
- 每一行长度是 max_model_len = 8
- 所以一维展开后：
flatten(token_ids_cpu_tensor) = [101, 102, 103, 201, 202, 0, 0, 0, 111, 112, 113, 114, 301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
计算公式是：
token_indices = positions_np + req_indices * max_model_len
代入得：
token_indices =
[
  3 + 0*8,
  4 + 0*8,
  2 + 1*8,
  3 + 1*8,
  4 + 1*8,
]
= [3, 4, 10, 11, 12]
5. 最终抽出的 input_ids 是什么？
再用这些 token_indices 去索引展平后的一维数组：
flattened = [101, 102, 103, 201, 202, 0, 0, 0, 111, 112, 113, 114, 301, 0, 0, 0,...]
取：
flattened[3] = 201
flattened[4] = 202
flattened[10] = 113
flattened[11] = 114
flattened[12] = 301
所以本轮真正送去执行的：
input_ids = [201, 202, 113, 114, 301]
暂时无法在飞书文档外展示此内容
torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                   0,
                   torch.from_numpy(token_indices),
                   out=self.input_ids_cpu[:total_num_scheduled_tokens])
此处的 input_ids_cpu 即为前述用于存放 token 的一维数组。如图所示，系统通过 token_indices 索引，将 token_ids_cpu_tensor 中有效的 token 按调度顺序提取并搬运到 input_ids_cpu 中，从而构成当前调度步实际需要处理的输入 token 序列。我们举个简单的例子：
 token_ids_cpu_tensor是一个在 CPU 上的大数组，存着所有正在处理的请求的 Token ID。
- 请求 A：目前存了 4 个 Token: [101, 102, 103, 104]
- 请求 B：目前存了 3 个 Token: [201, 202, 203]
在 CPU 内存里，它们可能是这样连续排布的：[101, 102, 103, 104, 201, 202, 203, 0, 0, ...]，
假设这一轮调度（Batch），我们要处理以下内容，也就是数组中高亮的部分。
1. 请求 A 的最后 2 个 Token（需要做 Prefill 后的最后计算或 Decode）。
2. 请求 B 的全部 3 个 Token。
系统会算出当前 Batch 需要的 Token 在缓存池中的“绝对位置”：
- 请求 A 的索引 2 和 3（即 103, 104）
- 请求 B 的索引 4, 5, 6（即 201, 202, 203）
所以，token_indices_tensor = [2, 3, 4, 5, 6]，下一步就是用这里的索引token_indices_tensor去token_ids_cpu_tensor中获取对应的token。而 query_start_loc 则如同每个请求所对应 token 区域之间的“界碑”，记录了各个请求在该扁平化序列中的起始位置，用于在后续计算中区分不同请求的数据范围，实现高效、边界透明的并行处理。
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
准备 token 和显存块的映射
需要明确的是，显存分块管理主要是针对 KV Cache 的管理。每个 block 的大小为 block_size（单位：token 数），每个 block 可存储最多 block_size 个 token 的 KV 值。整个 KV Cache 的显存空间大小为：num_layers × num_blocks × block_size × num_kv_heads × head_size × 2 × dtype，其中 2 表示 K 和 V 两个矩阵，dtype 为数据类型所占字节数，num_layers表示层数，num_blocks表示系统当前分配的 block 总数， block_size 表示每个 block 最多容纳的 token 数。
我们定义单个 page 的大小为：page_size = block_size × num_kv_heads × head_size × 2，即每一 layer 中一个 block 所占用的 KV Cache 空间。每层中所有 block 的总空间为 num_blocks × page_size。
由于一个 block 最多可存放 block_size 个 token 的 KV Cache，而每个 token 占用连续的一行空间，因此要定位某个 token 在其所在 block 中的存储位置，还必须知道它在该 block 内的偏移量（offset），即该 token 是当前 block 中的第几个 token。
暂时无法在飞书文档外展示此内容
我们还是以这里的请求作为一个例子，每个请求有唯一的 req_id，且最多可占用 self.max_num_blocks_per_req 个 block，对应图中的划分。
要获取当前请求（如 req1）中被调度 token 对应的 block，确定当前请求对应的 block 表索引。其中 positions 是待调度 token 在序列中的偏移量。例如，对于请求 req1，当前处理第 3 到第 5 个 token，positions = [3, 4, 5]。
1. 先通过 positions // self.block_size 计算每个 token 所属的逻辑 block 编号。也就是说，把序列位置按 block_size 分段，就能知道每个 token 落在该请求页表中的第几个 block。
2. 再结合 req_idx，从该请求对应的块表行中查找物理 block ID。也就是根据逻辑 block 编号，到 block_table[req_idx] 中取出真正分配到的显存 block 编号，从而完成“逻辑 block -> 物理 block”的映射。
3. 最后通过 positions % self.block_size 计算 token 在 block 内的偏移量。因为一个 block 内最多容纳 block_size 个 token，所以除了知道它在哪个 block 中，还必须知道它是该 block 中的第几个 token，才能最终定位到对应的 KV Cache 存储位置。
暂时无法在飞书文档外展示此内容
def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
    # where K is the max_num_blocks_per_req and the block size is 2.
    # NOTE(woosuk): We can't simply use `token_indices // block_size`
    # here because M (max_model_len) is not necessarily divisible by
    # block_size.
    # 确定块偏移
    block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions // self.block_size)
    block_numbers = self.block_table_np.ravel()[block_table_indices]
    block_offsets = positions % self.block_size
    np.add(block_numbers * self.block_size,
           block_offsets,
           out=self.slot_mapping_np[:req_indices.shape[0]])
至此，我们已确定以下关键变量：
1. q_start_loc：每个请求中当前请求 token 的起始位置（相对于请求的总序列）
2. num_computed_tokens：每个请求中已被计算过的 token 数量（即已处理的历史 token 数）
3. num_reqs：当前批次中的请求数量
4. block_table：请求到 KV Cache 显存块的映射表，记录每个请求的逻辑 block 到物理 block 的映射
5. slot_mapping：每个 token 对应的物理 block ID 及其在 block 内偏移的映射关系，用于在 PagedAttention 中定位实际存储位置
compute_slot_mapping() 的作用，就是把请求内 token 位置转换成KV Cache 物理地址索引，从而让 Attention 在读写 KV Cache 时知道每个 token 应该访问哪一个 block、哪一个 offset。
暂时无法在飞书文档外展示此内容
另外还有一点刚才已经说的，就是一个请求会和多个block绑定，请求和block块的映射关系被存放在block_table这一结构当中。
下图中的墨迹结合视频一起看。
暂时无法在飞书文档外展示此内容
2.4 总结
在本节课程中，请求调度的结果 scheduler_output 经历了从 _update_states 到 _prepare_input，再到模型执行阶段的完整流程。
1. 在 _update_states 阶段，系统根据 SchedulerOutput 更新 ModelRunner 内部维护的请求状态和批处理结构（主要是 input_batch），确保其准确反映当前调度轮次的实际请求集合与资源分配情况。具体操作包括：
  1. 清理已完成的请求
  2. 移除本轮未被调度的请求
  3. 更新 block_table，并处理新增或恢复的请求等。该阶段的核心目标是确定每个请求所使用的物理 block ID，以及各个 token 在对应 block 中的块内偏移，为后续 PagedAttention 机制提供地址映射基础。
2. 进入 _prepare_input 阶段后，系统基于 _update_states 所更新的状态，构建并填充模型前向传播所需的所有 GPU 张量，包括 input_ids、block_table、slot_mapping、query_start_loc 等关键元数据。这些张量将作为模型执行时的输入，驱动 Attention 算子正确访问分布式存储的 KV Cache。
for req_i in range(seq_lens.size(0)):
    start = query_start_loc[req_i].item()
    end   = query_start_loc[req_i + 1].item()
    length = seq_lens[req_i].item()
    # 切分每个请求中对应token
    req_k = k[start:end]         
    req_v = v[start:end]
    # 获取该请求所有的block
    blocks = block_table[req_i]   
      
    # 按 block_size 将该请求的 token 分块处理
    for blk_idx in range((length + block_size - 1) // block_size):
        # 获取当前的
        blk_id = blocks[blk_idx].item()
        pos_in_blk = blk_idx * block_size 
        # 当前 block 中实际要写入多少个 token，最后一个block可能装不满        
        chunk_len = min(block_size, length - pos_in_blk)
        if chunk_len <= 0:
            break
        
        # 当前 block 对应 token 在整个 batch 展平序列中的全局起始位置
        global_token_offset = start + pos_in_blk         
        slot = slot_mapping[global_token_offset : global_token_offset + chunk_len]  
        # 只保留 block 内偏移位置
        slot_in_block = slot % block_size      
        # 逐个 token 将本轮生成的 K/V 写入对应物理 block 的正确位置          
        for token_idx in range(chunk_len):
            k_cache[blk_id, 0, :, slot_in_block[token_idx]] = req_k[pos_in_blk + token_idx]
            v_cache[blk_id, 1, :, slot_in_block[token_idx]] = req_v[pos_in_blk + token_idx]
为了更深入理解 PagedAttention 为何需要上述信息，下面我们通过一段伪代码展示其整体执行流程。第一部分模拟的是 Prefill 阶段对 KV Cache 的写入过程：
1. 使用 query_start_loc 确定每个请求的起始和结束位置当前输入 input_token_ids 经过 embedding 后得到形状为 [total_num_tokens, hidden_dim] 的嵌入向量。经过线性变换 $$ \text{query} = \text{input} \times W_q $$ 和 $$ \text{key} = \text{input} \times W_k $$ 后，得到的 key 和 value 张量维度为 [total_num_tokens, dim]，其中 W_k 的维度为 [hidden_dim, dim]。由于多个请求的 token 被扁平化拼接在一起，必须借助 query_start_loc 来切分出每个请求对应的子序列（即 req_k, req_v）。
2. 获取每个请求的 Key/Value 并写入对应的 KV Cache block对于每个请求，将其计算出的 req_k 和 req_v 写入由 block_table 和 slot_mapping 指定的物理显存位置。由于 req_k 包含该请求所有新 token 的 Key 值，因此写回过程中需依据每个 token 的逻辑位置，通过 slot_mapping 找到其对应的物理 slot（即 block_id * block_size + offset），并将数据写入 kv_cache[block_id] 的相应偏移处。
以上只是我们为了理解流程写的伪代码，实际的Attention实现过程中加载是和计算同时进行的（FlashAttention），并不会先将所有Key/Value值进行加载再计算。
参考资料
- vLLM 架构概览
- 图解Vllm V1系列2：Executor-Workers架构
- vLLM V1: A Major Upgrade to vLLM's Core Architecture | vLLM Blog
