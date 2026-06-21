本节深入解析 vLLM 中 MultiprocExecutor 与 Worker 间的通信机制，聚焦 RPC 广播、共享内存与 ZMQ 混合传输、连接握手及结果回传，揭示高效分布式推理背后的底层设计。
从零入门的Ai-Infra课程
1. 《自制大模型推理框架》课程目录-支持LLama3和Qwen3
2. 《速通OpenAI Triton：手写轻量级多模态推理框架》课程目录
3. 目录-面向AI-Infra的Cuda零基础入门
课程介绍请点击上方链接
前言内容
[图片]
在上一节课中，我们分析了 AsyncLLM 客户端与 EngineProc 之间的通信机制：客户端通过 ZMQ 的 ROUTER/DEALER 模式将请求推送到引擎 Engine，而 EngineProc 在接收到请求后，将其转发至实际的推理执行组件。
我们提到，当模型产生部分输出时（即 EngineProc 的输出队列非空），数据会通过 ZMQ 的 PUSH 套接字实时发送回客户端。这种设计天然支持流式输出——只要推理结果生成，就能立即推送。相应地，客户端只需使用 PULL 套接字接收即可。
客户端到 EngineProc 的方向：
- 客户端使用 ROUTER 套接字发送请求。
- EngineProc 使用 DEALER 套接字连接并接收请求。
- ROUTER/DEALER 组合适合“前端 client 与后端 engine 之间的异步通信”：前端可以根据DEALER 的 identity 将请求定向发给对应的 EngineProc，而 EngineProc 收到请求后再交给内部推理组件处理。
EngineProc 到客户端的方向：
- 当推理尚未结束、但已经产生部分 token 时，EngineProc 会先将部分结果放入输出队列。
- 输出线程不会等待整次生成完成，而是会立即把这部分结果通过 PUSH 套接字发送出去，实现流式返回。
- 客户端侧使用 PULL 套接字持续接收这些输出结果。
暂时无法在飞书文档外展示此内容
在推理架构中，EngineCoreProc这一层并不直接执行模型的前向计算。它更像一个总调度台，主要做三件事：
1. 接收来自前端的请求；
2. 维护调度逻辑——例如哪些请求该优先执行、哪些请求可以合并为一个 batch；
3. 调用 self.model_executor，把真正的执行任务交出去。
这里的 Executor 是执行管理层。它接到 EngineCoreProc 下发的任务后，不一定自己亲自计算，而是根据当前运行模式决定如何组织底层算力：
- 单进程模式：可能直接驱动一个本地 Worker；
- 多进程模式：可能把任务发给多个 Worker 进程；
- 分布式模式：还可能把任务分发到不同设备甚至不同机器上的工作单元。
而 Worker 才更接近真正干活的人。它通常负责：
- 持有模型权重；
- 管理本设备上的 KV cache；
- 调用模型执行前向计算；
- 返回本轮计算结果。
[图片]
当 EngineCoreProc 执行 step() 时，会先对当前请求进行调度，生成本轮执行计划，然后通过 self.model_executor 将该执行计划交给 Executor，由后者组织后续的推理计算。
[图片]
本节课我们将聚焦这个更接近 GPU 执行层的二房东，重点讲解 MultiprocExecutor和 Worker组件作用及协作机制！
MultiprocExecutor通过主从进程结构实现高效的本地并行控制：主进程负责任务分发与协调，各个 Worker 子进程则绑定独立 GPU，它适用于单机多卡场景，执行具体的模型计算。这样的分层架构使得各组件职责清晰、解耦明确：Engine 负责接口与调度，Executor 负责分布式执行管理，而最终的模型推理落在 Worker 上。这不仅提升了系统可维护性，也为性能优化提供了灵活空间。
也就是说，MultiprocExecutor 通过控制进程 + 多个 Worker 进程的结构实现高效的本地并行执行。控制层负责任务分发、状态协调与结果汇总，各个 Worker 进程通常绑定独立 GPU，负责具体的模型计算，因此这种模式特别适合单机多卡场景。
一 Worker 组件介绍及初始化、执行
在 vLLM 中，Worker 是负责实际模型执行的底层工作单元。不同 Executor 对 Worker 的创建与管理方式略有不同；例如在多进程模式下，Worker 通常运行在独立进程中。Worker 的核心职责主要包括：
1. 调用 execute_model 函数执行模型前向推理：接收调度器的输出（待处理的请求）SchedulerOutput、准备输入张量、执行模型前向传播、返回模型输出 ModelRunnerOutput 或中间张量（流水线并行）。
2. 调用 sample_tokens 从模型输出中采样 Tokens。
3. 调用 load_model 函数执行模型加载和管理。
4. 调用 initialize_cache 执行 KV Cache 管理。
5. 以及调用 add_lora、remove_lora、pin_lora 函数完成 LoRA 管理。
WorkerBase 基类的核心接口函数如下图：
[图片]
1.1 Executor-Workers 架构
Executor-Workers 关系架构图如下：
[图片]
可以看出 Executor-Workers 关系如下:
- 一个 Executor 下管理着若干 workers，每个 workers 位于独立的进程上，可以简单理解成一个 worker 占据着一张卡；
- Executor 负责把请求 broadcast 到各个 workers 上；
- 各个 workers 接收到请求，负责执行实际的推理过程，并将推理结果返回给 Executor。
1.2 Executor 类中创建 Worker
vLLM 在 v1 架构中引入了多种 Executor 类型，用于管理 Worker 的创建和模型执行。这些 Executor 根据并行配置（如 tensor_parallel_size、pipeline_parallel_size 和 prefill_context_parallel_size）以及分布式后端（distributed_executor_backend）动态选择，以适应不同规模的部署场景，主要有 3 种 Executor:
1. 单进程（UniProcExecutor，用于单 GPU 或简单测试）
2. 多进程单节点（MultiprocExecutor，用于多 GPU 单机）
3. 分布式多节点（RayDistributedExecutor，用于跨节点扩展）。
Executor 的选择发生在引擎初始化阶段，通过 Executor.get_class(vllm_config) 方法确定：
1. 如果 distributed_executor_backend == "uni"，使用 UniProcExecutor（默认单 GPU）。
2. 如果 distributed_executor_backend == "mp"，使用 MultiprocExecutor（多 GPU 单节点）。
3. 如果 distributed_executor_backend == "ray"，使用 RayDistributedExecutor（多节点分布式）。
每个 Executor 负责创建和管理 Worker（通常是 GPUWorker），Worker 是实际执行模型计算的单元，每个 Worker 对应一个 GPU 或分片。总 Worker 数量（world_size）由 tp_size * pp_size * pcp_size 计算。
在实际部署中，vLLM 通过 --distributed-executor-backend CLI 参数选择后端（如 "mp" 或 "ray"）。
以 MultiprocExecutor 为例，描述 Worker 创建过程。MultiprocExecutor 用于单节点多 GPU 场景，利用 Python 的 multiprocessing（mp）启动独立进程，避免 Ray 的开销。Worker 创建发生在本地 world_size 内（本地可用 GPU 数），每个 Worker 进程通过 WorkerProc 包装启动，支持数据并行（DP）和 tensor/ pipeline 并行。
关键步骤：
1. 计算 world_size 并验证（tp * pp * pcp）。
2. 获取本地 world_size（本地 GPU 数）。
3. 为每个本地 rank 创建 UnreadyWorkerProcHandle（未就绪 Worker 句柄），通过 WorkerProc.make_worker_process 启动进程。
4. 进程启动后，进入就绪状态，执行 Worker 初始化（init_device、load_model 等）。
5. 支持终止等待（wait_for_termination），确保进程安全退出。
精简后的代码如下：
from dataclasses import dataclass
from multiprocessing import Process, Queue
# ... 其他导入

@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: Process
    # ... 其他字段如 request_mq, response_mq

class MultiprocExecutor(Executor):
    def __init__(self, vllm_config: VllmConfig, ...):
        # 初始化配置
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self._init_executor()

    def _init_executor(self) -> None:
        # 获取并验证 world_size
        self.world_size = self.parallel_config.world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to "
            f"tens or_parallel_size ({tp_size}) x pipeline_parallel_size ({pp_size}) "
            f"x prefill_context_parallel_size ({pcp_size})."
        )

        # 获取本地 worker 数量
        self.local_world_size = self.parallel_config.local_world_size

        # 创建 workers（使用列表存储 UnreadyWorkerProcHandle）
        unready_workers = []
        global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp
        distributed_init_method = get_distributed_init_method(...)  # 如 TCP URI

        for local_rank in range(self.local_world_size):
            global_rank = global_start_rank + local_rank
            unready_workers.append(
                WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    # 其他参数如 queues for IPC、model_config 等
                )
            )

        # 等待 workers 就绪（例如通过 polling queues）
        self.workers = []  # 最终就绪 workers 列表
        for handle in unready_workers:
            # 处理就绪信号，转换为 WorkerProc
            worker = WorkerProc.from_handle(handle)
            self.workers.append(worker)

        # 初始化模型到 workers
        self.init_model(...)
1.3 Worker 初始化
1.3.1 Worker 初始化流程
在 vLLM v1 中，Worker（通常指 GPUWorker）的初始化发生在 ModelExecutor 的构建过程中，无论是单 GPU 的 UniProcExecutor 还是多 GPU 的 MultiProcExecutor，每个 Worker 进程都会独立执行相同的初始化步骤。这些步骤确保 Worker 准备好设备、模型和 KV 缓存，以支持高效的 LLM 推理。
Worker 初始化主要分为三个核心阶段：Init Device、Load Model 和 Initialize KV Cache。
1. Init Device（初始化设备）
  - 初始化当前 Worker 的设备环境与分布式上下文，包括数据并行（DP）、张量并行（TP）、流水线并行（PP）和专家并行（EP）等相关配置与通信组。
  - 实例化 model_runner，用于管理模型执行流程，包括采样器、KV 缓存访问以及前向传播所需的 GPU 侧缓冲区（如 input_ids、positions）。
  - 实例化 InputBatch，用于管理批处理输入状态，包括 CPU 侧前向传播缓冲区、KV 缓存块表以及采样元数据等。
2. Load Model（加载模型）
  - 实例化模型结构。
  - 加载模型权重，并按并行策略完成参数切分与设备放置。
  - 调用 model.eval() 将模型切换到 PyTorch 的推理模式。
  - 可选地调用 torch.compile() 对模型执行图进行优化，以提升后续推理性能。
3. Initialize KV Cache（初始化 KV 缓存）
  - 调用 get_kv_cache_spec 确定各层 KV Cache 的规格。
  - 通过 dummy run 或 profiling 前向传播估算可用显存，并据此计算可容纳的 KV cache block 数量。
  - 分配、reshape 并将 KV cache tensor 绑定到各注意力层。
  - 准备注意力相关元数据与后端配置（例如 FlashAttention），供后续前向传播内核使用。
  - 除非显式指定 --enforce-eager，否则还会执行 warmup 批次，并为常见批次形状捕获 CUDA graphs，以减少 kernel launch 开销、降低首轮抖动并优化推理延迟。
在 MultiprocExecutor 中，上述初始化步骤会在每个 Worker 进程内独立执行。Executor 会为每个 rank 启动对应的子进程（通常通过 WorkerProc.make_worker_process），每个子进程随后进入 WorkerProc.worker_main，完成设备、模型以及 KV Cache 等初始化流程。
初始化结束后，Worker 进程不会退出，而是进入持续运行的 busy loop，等待来自 Executor 的任务下发，并在收到工作项后执行相应的推理或控制逻辑。
Worker 的完整初始化流程及函数调用关系链可总结如下:
1. WorkerWrapperBase.__init__()
   └─> 创建 WorkerWrapper，但尚未创建实际的 Worker

2. WorkerWrapperBase.init_worker()
   ├─> 加载插件
   ├─> 解析 worker 类名
   ├─> 动态继承 worker_extension_cls（如果存在）
   └─> 创建 Worker 实例 (WorkerBase.__init__)
   
3. WorkerWrapperBase.init_device()
   └─> Worker.init_device()
       ├─> 设置 CUDA 设备
       ├─> 初始化分布式环境 (NCCL)
       ├─> 设置随机种子
       ├─> 创建内存快照
       └─> 创建 ModelRunner

4. WorkerWrapperBase.load_model()
   └─> Worker.load_model()
       └─> ModelRunner.load_model()
           └─> 加载模型权重到 GPU

5. WorkerWrapperBase.initialize_from_config()
   └─> Worker.initialize_from_config()
       └─> ModelRunner.initialize_kv_cache()
           └─> 初始化 KV cache

6. WorkerWrapperBase.compile_or_warm_up_model()
   └─> Worker.compile_or_warm_up_model()
       ├─> 模型预热（dummy run）
       ├─> 内核预热
       └─> CUDA Graph 捕获（如果启用）
1.3.2 Worker 数量初始化
world_size 表示一个 DP 组内（同一个模型副本）内的 Worker 数量，world_size 是根据并行策略（张量并行、流水线并行、上下文并行等）和 Executor 类计算得出的。
1，不开启 DP 时：
world_size 计算代码在 ParallelConfig.post_init 函数中实现，代码如下:
# vllm/vllm/config/parallel.py: __post_init__() 函数
@config
class ParallelConfig:
    def __post_init__(self) -> None:
        self.world_size = (
            self.pipeline_parallel_size           # 流水线并行大小
            * self.tensor_parallel_size           # 张量并行大小
            * self.prefill_context_parallel_size  # Prefill 上下文并行大小
        )
        
    # 如果使用 external_launcher，还需要乘以 data_parallel_size
    if self.distributed_executor_backend == "external_launcher":
        logger.info("Using external launcher for distributed inference.")
        self.world_size *= self.data_parallel_size
MultiprocExecutor 会校验这一点：
class MultiprocExecutor(Executor):
    def _init_executor(self) -> None:
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            ...
        )
一些参数说明:
- pipeline_parallel_size (PP): 流水线并行组数，默认值为 1
- tensor_parallel_size (TP): 张量并行组数，默认值为 1
- prefill_context_parallel_size (PCP): Prefill 上下文并行组数，默认值为 1
2，开启 DP 后：
ParallelConfig 类的 world_size_across_dp 属性定义:
# # vllm/vllm/config/parallel.py:
@config
class ParallelConfig:
    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        # world_size (TP * PP) * data_parallel_size (DP)。
        # 如果 vLLM 内部的 world_size 不包含 data_parallel_size
        return self.world_size * self.data_parallel_size 
多节点（机）环境中运行了多个数据并行（Data Parallel, DP）副本的情况下，LLM 通过引入了 world_size_across_dp 属性，来真正表示多节点集群中所有 Worker 的总数量（也说 world_size 代表“全局总进程数）。
乘 data_parallel_size 的原因：让 vLLM 内部认知的 world_size 与外部 launcher 实际启动的进程总数完全一致，从而正确设置 RANK / WORLD_SIZE 和分布式通信组。
[图片]
world_size 和 world_size_across_dp 的区别总结如下:
3，为什么跨机 DP 需要 Ray 或 Slurm 等分布式框架？
DP = 多个模型副本（多个 Engine）；每个副本内部仍用 multiproc 管 TP×PP×PCP 个 worker。
跨机器时，需要额外机制（分布式框架）去在不同节点上拉起多个 EngineCore，并协调 DP rank、地址、设备可见性。
单机限制与多机协同的矛盾是根本原因。
- multiprocessing 的局限：Python 的 multiprocessing 只能在当前物理机的操作系统内核中创建子进程。它无法跨越网络，去另一台机器上分配显存、拉起进程或同步状态。
- 分布式框架的作用：当启用跨机 DP（例如 2 台机器，每台机器运行一个 TP=8 的 Replica，组成 DP=2）时，需要解决跨机资源发现、网络握手和进程对齐的问题。
  - Ray：充当分布式对象存储和 Actor 调度器。它在多台机器上启动 Ray Daemon，跨机传递 Python 对象和控制信号，从而在不同机器上拉起 vLLM 实例。
  - Slurm / Torchrun：在物理层通过 SSH/PMI 接口，同时在多台机器上拉起指定数量的进程，并为每个进程注入环境变量（如 MASTER_ADDR、MASTER_PORT、RANK）。
4，World Size 的核心作用是什么？
在 vLLM（及 PyTorch 分布式）中，world_size 主要有以下两个作用：
1. 构建全局通信网（Rendezvous）：
让 NCCL 知道整个集群一共有多少个 GPU 节点，以便它们之间建立 TCP/RDMA 连接，生成全局通信拓扑图。
2. 切分通信子组（Sub-Process Groups）：
在全局通信网建立后，vLLM 会根据 world_size、tensor_parallel_size 和 pipeline_parallel_size，将全局进程切分成不同的子通信组。
  - TP 组：负责权重切分后的 All-Reduce 通信。
  - PP 组：负责层与层之间的 Send/Recv 激活值传递。
  - DP 组：在推理场景中，DP 组通常不进行通信，但需要被隔离出来，确保 TP/PP 的通信不会串线。
1.4 initialize_model_parallel 函数理解
在 vllm/distributed/parallel_state.py（负责管理所有并行状态的核心代码）中，vLLM 接收全局的 world_size 和 rank，然后将其切分为不同的子组：
# vllm/vllm/distributed/parallel_state.py
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.
    """
    ############# 代码省略 #############
使用一个具体的 8 卡 (GPU) 实例进行 3D/4D 空间的可视化来理解 initialize_model_parallel 函数作用。
1. 核心数学模型：5D 张量布局
代码中最巧妙的设计是将一维的全局 GPU 列表（torch.arange(world_size)）重塑（Reshape）为一个 5 维张量。其布局顺序为：
$$\text{all\_ranks} \in [\text{ExternalDP}, \text{DP}, \text{PP}, \text{PCP}, \text{TP}]$$
all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa
假设我们有 8 张 GPU (g0 ~ g7)，配置如下：
- tensor_model_parallel_size (TP) = 2
- pipeline_model_parallel_size (PP) = 2
- prefill_context_model_parallel_size (PCP) = 1 (暂不启用上下文并行)
- data_parallel_size (DP) = 2
- ExternalDP = 1 (表示没有外部 DP 包装)
通过 all_ranks = torch.arange(8).reshape(1, 2, 2, 1, 2)，我们在内存中构建了如下的 5D 网格：
all_ranks 结构：
[[[[[0, 1]],     <-- DP=0, PP=0, PCP=0, TP=[0,1]
   [[2, 3]]],    <-- DP=0, PP=1, PCP=0, TP=[2,3]
  [[[4, 5]],     <-- DP=1, PP=0, PCP=0, TP=[4,5]
   [[6, 7]]]]]   <-- DP=1, PP=1, PCP=0, TP=[6,7]
我们可以将这个结构简化为一个 2D 矩阵网格（行表示 DP x PP，列表示 TP）：

TP_0 (Shard 0)
TP_1 (Shard 1)
DP_0, PP_0
g0
g1
DP_0, PP_1
g2
g3
DP_1, PP_0
g4
g5
DP_1, PP_1
g6
g7
2. 各并行组（Groups）的分组可视化
代码通过 “转置（Transpose）+ 重塑（Reshape）” 的组合拳，快速将特定的维度挤压到一起，从而完成分组。
❶ Tensor Parallel (TP) 组：[g0, g1], [g2, g3], [g4, g5], [g6, g7]
对应源码实现:
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]
_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )
"""
view(-1, tensor_model_parallel_size)：
    将 5D 张量重塑为 2D 矩阵，且每一行的大小必须为 TP_size (2)。
变换结果：
    [[0, 1],
     [2, 3],
     [4, 5],
     [6, 7]]  (形状为 [4, 2])
unbind(0)：
    将 2D 矩阵沿着第 0 维（行）剥离，生成一个包含数个 1D 张量的元组。
变换结果：
    ([0, 1], [2, 3], [4, 5], [6, 7])，每一个元素代表一个独立的 TP 进程组。
"""
直观理解：由于 TP 在最内侧维度，直接按顺序每 TP_size (2) 个划分为一组，TP 分组结果如下
[g0, g1]  --> TP Group 0
[g2, g3]  --> TP Group 1
[g4, g5]  --> TP Group 2
[g6, g7]  --> TP Group 3
❷ Pipeline Parallel (PP) 组：[g0, g2], [g1, g3], [g4, g6], [g5, g7]
对应源码实现：
group_ranks = (
    all_ranks.transpose(2, 4)
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
"""
transpose(2, 4)：
    目标：交换维度 2 (PP) 和 4 (TP)。
    交换前形状：[1, 2, PP_size=2, 1, TP_size=2]
    交换后形状：[1, 2, TP_size=2, 1, PP_size=2]
.reshape(-1, pipeline_model_parallel_size)：
    目标：将张量拍平为 2D 矩阵，每行为 PP_size (2)。由于使用了 transpose 导致内存不连续，这里必须使用 reshape（底层会进行一次内存拷贝以使其连续）。
变换结果（以 DP_0 区域为例）：
    原始局部：g0 (PP_0, TP_0)，g1 (PP_0, TP_1)，g2 (PP_1, TP_0)，g3 (PP_1, TP_1)。
    转置并重塑后，g0 与 g2 排在同一行，g1 与 g3 排在同一行：
[[0, 2],   # 对应 TP_0
 [1, 3],   # 对应 TP_1
 [4, 6],   # 另一个 DP 副本的 TP_0
 [5, 7]]   # 另一个 DP 副本的 TP_1
"""
- 代码逻辑：all_ranks.transpose(2, 4).reshape(-1, PP_size)
- 转置过程（以 DP_0 为例）：
  - 交换第 2 维 (PP) 和第 4 维 (TP)。
  - 原始局部矩阵：
    $$\begin{matrix} 0 & 1 \\ 2 & 3 \end{matrix}$$
  - 转置（交换行列）后：
    $$\begin{matrix} 0 & 2 \\ 1 & 3 \end{matrix}$$
  - .reshape(-1, 2) 展平得到：
[g0, g2]  --> PP Group 0 (对应 TP_0 链条)
[g1, g3]  --> PP Group 1 (对应 TP_1 链条)
[g4, g6]  --> PP Group 2 (对应另一个 DP 副本)
[g5, g7]  --> PP Group 3 (对应另一个 DP 副本)
❸ Data Parallel (DP) 组：[g0, g4], [g1, g5], [g2, g6], [g3, g7]
对应源码实现:
group_ranks = (
    all_ranks.transpose(1, 4)
    .reshape(-1, data_parallel_size)
    .unbind(0)
)
- 代码逻辑：all_ranks.transpose(1, 4).reshape(-1, DP_size)
- 直观理解：
  - 交换第 1 维 (DP) 和第 4 维 (TP)，使同一位置但在不同数据副本（DP_0 和 DP_1）上的 GPU 对齐。
[g0, g4]  --> DP Group 0 # 对应 PP_0, TP_0 维度上的 DP 对齐
[g1, g5]  --> DP Group 1 # 对应 PP_0, TP_1 维度上的 DP 对齐
[g2, g6]  --> DP Group 2 # 对应 PP_1, TP_0 维度上的 DP 对齐
[g3, g7]  --> DP Group 3 # 对应 PP_1, TP_1 维度上的 DP 对齐
❹ Expert Parallel (EP) 组（MoE 模型专属）
对应源码实现:
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(
        -1,
        data_parallel_size
        * prefill_context_model_parallel_size
        * tensor_model_parallel_size,
    )
    .unbind(0)
)
在 MoE 混合专家模型中，专家并行通常跨越 DP、PCP 和 TP 维度（即 data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size）。
- 代码逻辑：all_ranks.transpose(1, 2).reshape(-1, DP * PCP * TP)
- 本例参数下的计算：$$EP\_size = 2 \times 1 \times 2 = 4$$
- 分组结果：
  - EP Group 0: [g0, g1, g4, g5]
  - EP Group 1: [g2, g3, g6, g7]
- 设计意图：将同一 Pipeline Stage 上不同 DP 副本内的所有 TP 节点联合起来，共同切分和承载 MoE 的专家权重。
3. 特殊机制说明
1. enable_elastic_ep（弹性专家并行）：
  - 开启后，代码使用 local_all_ranks（只包含单机内的 TP/PP 关系），并通过 _init_stateless_group 与外部 TCP Store 通信。这是为了支持在不中断整体服务的情况下，动态弹性伸缩专家并行的规模。
2. prefill_context_model_parallel_size (PCP)：
  - 上下文并行（Context Parallel）。它通过 .transpose(3, 4) 将 PCP 维度与 TP 维度对齐，用于在超长文本 Prefill 阶段，把 Prompt 沿着 Sequence 维度切分到多个 GPU 上计算。
二 Executor 和 Worker 组件通信 Demo-RPC 过程
Executor 与 Worker 之间的协作，本质上可以理解为一种类 RPC 的进程间调用机制（vLLM 源码中正是将其核心方法命名为 collective_rpc）。Executor 负责发送待执行的方法名及其参数，Worker 负责在自身进程中接收请求、执行对应逻辑，并将执行结果返回给 Executor。在这个抽象下，Executor 相当于调用方，Worker 相当于服务端。
在 Demo 中，我们分别启动多个独立进程来模拟 Executor 和 Worker（Executor 运行在引擎主进程中，每个 Worker 各占一个独立子进程）。它们之间通过队列进行通信：Executor 通过一个共享的广播队列向所有 Worker 统一下发命令（所有 Worker 收到相同内容），Worker 在执行完成后，再通过各自独立的结果队列将返回值传回 Executor。
由于 Executor 与 Worker 通常是一对多关系，请求下发采用广播方式，保证所有 Worker 拿到一致的任务；而每个 Worker 拥有独立的结果返回通道，以保证结果回收和状态管理彼此隔离、不发生混淆。
图中的 queue1、queue2、queue3 正是各个 Worker 向 Executor 回传结果的独立单向队列。
暂时无法在飞书文档外展示此内容
代码详见code/course2/executor_worker_demo.py
import multiprocessing as mp
import time
import random
import os

def dummy_execute_model(args):
    """模拟 execute_model 计算"""
    scheduler_output, = args
    time.sleep(random.uniform(0.2, 0.5))
    return f"logits_from_rank{os.getpid()}_{scheduler_output}"

def worker(rank: int, work_q: mp.Queue, result_q: mp.Queue):
    pid = os.getpid()
    print(f"[Worker {rank}] 启动，pid={pid}")

    # 1. 握手
    result_q.put("READY")

    # 2. 映射方法名 → 本地可调用对象
    method_table = {
        "execute_model": dummy_execute_model,
    }

    # 3. 主循环
    while True:
        item = work_q.get()
        if item is None:                     
            result_q.put(None)
            break
        method_name, args, kwargs = item
        func = method_table[method_name]
        print(f"[Worker {rank}] 执行 {method_name}{args}")
        output = func(args, **(kwargs or {}))
        result_q.put(output)

def master(num_workers: int = 2):
    print("=== Master 启动 ===")
    work_q = mp.Queue()
    result_queues = [mp.Queue() for _ in range(num_workers)]
    procs = [mp.Process(target=worker, args=(rank, work_q, result_queues[rank]))
             for rank in range(num_workers)]
    for p in procs:
        p.start()

    # 等待 READY
    while sum(rq.get() == "READY" for rq in result_queues) < num_workers:
        pass
    print(">>> 所有 Worker 就绪 <<<")

    # 下发 RPC 调用
    calls = [
        ("execute_model", ("scheduler_output_0",), {}),
        ("execute_model", ("scheduler_output_1",), {}),
        ("execute_model", ("scheduler_output_2",), {}),
    ]
    for call in calls * num_workers:        # 让每个 Worker 都执行一遍
        work_q.put(call)

    # 发结束信号
    for _ in range(num_workers):
        work_q.put(None)

    # 收集结果
    results = {rank: [] for rank in range(num_workers)}
    finished = 0
    while finished < num_workers:
        for rank, rq in enumerate(result_queues):
            if not rq.empty():
                item = rq.get()
                if item is None:
                    finished += 1
                else:
                    results[rank].append(item)
    for p in procs:
        p.join()
    print("=== 最终结果汇总 ===")
    for rank, outs in results.items():
        print(f"Worker {rank}: {outs}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    master(num_workers=2)
1. 首先是Executor与两个Worker分别建立连接，连接确立的方式是：Worker通过通信队列向Executor发送一个"Ready"字符串，表示自身已就绪。在Executor端，会等待来自两个Worker的"Ready"信号，当收到两个"Ready"消息后，即判定两个Worker均已就绪。
def master(num_workers: int = 2):
    # 等待 READY
    while sum(rq.get() == "READY" for rq in result_queues) < num_workers:
        pass
        
        
def worker(rank: int, work_q: mp.Queue, result_q: mp.Queue):
    pid = os.getpid()
    print(f"[Worker {rank}] 启动，pid={pid}")

    # 1. 握手
    result_q.put("READY")
2. 当Executor成功接收到来自两个Worker的“Ready”信号，确认两者均已就绪后，即可向它们发送执行指令，指令中包含待执行的方法及其相关参数。
calls = [
     ("execute_model", ("scheduler_output_0",), {}),
     ("execute_model", ("scheduler_output_1",), {}),
     ("execute_model", ("scheduler_output_2",), {}),
]
for call in calls * num_workers:        # 让每个 Worker 都执行一遍
    work_q.put(call)
3. 当 Worker 从广播队列中接收到 Executor 下发的方法名及参数后，便开始执行指定方法，并获取执行结果。
4. 当指定方法执行完毕后，Worker 将结果通过自身专属的结果队列返回给 Executor。
while True:
    item = work_q.get()
    method_name, args, kwargs = item
    func = method_table[method_name]
    print(f"[Worker {rank}] 执行 {method_name}{args}")
    output = func(args, **(kwargs or {}))
    
    result_q.put(output)
总结一下，Executor 与各 Worker 分别建立连接。连接确立的方式为：Worker 在完成自身初始化（包括消息队列的创建）后，通过一个通信管道向 Executor 发送就绪消息。消息内容除 "READY" 状态标记外，还附带该 Worker 结果回传队列的句柄（handle），供 Executor 后续收集结果时使用。
在 Executor 端，会等待来自两个 Worker 的就绪消息；收齐后，双方还需分别在各自的消息队列上调用 wait_until_ready()，完成底层 ZMQ socket 的连接配对。至此才判定两个 Worker 均已就绪，Worker 随即进入 busy loop，等待 Executor 下发 RPC 命令。
三 Executor 和 Worker 组件的协作
3.1 MultiprocExecutor 工作流程
3.1.1 获取 Executor 类
暂时无法在飞书文档外展示此内容
MultiprocExecutor 的初始化发生在 vLLM 启动阶段，在接收用户请求之前就已完成。因此，无法通过进程附加（attach）的方式调试初始化过程，因为当附加时，初始化早已结束。在 VSCode 中，应直接将vllm.entrypoints.openai.api_server作为启动模块进行调试。
这样，Python 调试器才能在初始化阶段命中断点，断点可设在 vllm/v1/executor/abstract.py 第 9 行，以捕获 Executor 创建的起始位置。Executor 有多个派生类，多卡推理对应的派生类是MultiprocExecutor。
class Executor(ExecutorBase):
    """
    Abstract class for v1 executors, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in ExecutorBase"""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
    
为了方便调试，我们可以在.vscode/launch.json 中修改配置。具体配置如下，完整文件位于 code/course2/launch.json，如需使用，请复制到课程目录下的.vscode/launch.json中。就可以调试到如上的断点中，我们格式化的打印一下parallel_config。
其中最重要的一点，'mp' 代表 Multiprocessing（多进程），这是 vLLM 最基础、最常用的分布式执行后端。也就是说，每个子进程对应一个 GPU Worker（Worker），负责处理模型推理请求，worker_cls='vllm.v1.worker.gpu_worker.Worker'
ParallelConfig(
    # ========================
    # 并行策略配置
    # ========================
    pipeline_parallel_size=1,           # 流水线并行度（默认：1）
    tensor_parallel_size=2,             # 张量并行度（默认：1），现在配置的是tp=2
    data_parallel_size=1,               # 数据并行度（总节点数）
    data_parallel_size_local=1,         # 当前节点内数据并行度
    data_parallel_rank=0,               # 当前节点在全局数据并行中的排名
    data_parallel_rank_local=0,         # 当前节点内数据并行排名

    # ========================
    # 数据并行通信配置
    # ========================
    data_parallel_master_ip='127.0.0.1',  # 数据并行主节点地址
    data_parallel_rpc_port=29550,         # RPC 通信端口（用于进程间通信）
    data_parallel_master_port=0,          # 主节点监听端口（0 表示自动分配）
    data_parallel_backend='mp',           # 数据并行后端（如 'mp' 表示 multiprocessing）
    data_parallel_external_lb=False,      # 是否启用外部负载均衡
    data_parallel_hybrid_lb=False,        # 是否启用混合负载均衡

    # ========================
    # 专家并行（MoE）相关配置
    # ========================
    enable_expert_parallel=False,         # 启用专家并行（MoE）
    enable_eplb=False,                    # 启用专家负载均衡（Expert Load Balancing）
    num_redundant_experts=0,              # 冗余专家数量（用于容错）
    eplb_window_size=1000,                # EPLB 负载均衡窗口大小（样本数）
    eplb_step_interval=3000,              # EPLB 步骤间隔（处理步数）
    eplb_log_balancedness=False,          # 是否记录负载均衡状态

    # ========================
    # 资源与性能优化
    # ========================
    max_parallel_loading_workers=None,    # 最大并行模型加载工作线程数（默认为系统自动）
    disable_custom_all_reduce=False,      # 是否禁用自定义 AllReduce 实现（使用原生 PyTorch）

    # ========================
    # Ray 与分布式运行环境
    # ========================
    ray_workers_use_nsight=False,         # 是否对 Ray worker 使用 Nsight 进行性能分析
    ray_runtime_env=None,                 # Ray 运行时环境（如容器镜像、依赖包等）
    placement_group=None,                 # Ray Placement Group 配置（用于资源调度）

    # ========================
    # 执行器与工作节点配置
    # ========================
    distributed_executor_backend='mp',    # 分布式执行后端（'mp' = multiprocessing）
    worker_cls='vllm.v1.worker.gpu_worker.Worker',  # 工作节点类（GPU Worker）
    sd_worker_cls='auto',                 # 模型存储/加载专用工作节点类（自动选择）
    worker_extension_cls='',              # 工作节点扩展类（可选插件）

    # ========================
    # 全局并行规模信息
    # ========================
    world_size=2,                         # 总共参与并行的进程数（tensor_parallel_size × data_parallel_size）
    rank=0,                               # 当前进程的全局排名（从 0 开始）
    enable_multimodal_encoder_data_parallel=False,  # 是否开启多模态编码器的数据并行支持
)

以下是本课时的 launch.json 启动文件：
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "vLLM: API Server Debug",
            "type": "debugpy",
            "request": "launch",
            "module": "vllm.entrypoints.openai.api_server",
            "args": [
                "--model",
                "Qwen/Qwen3-1.7B",
                "--dtype",
                "float16",
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.95",
                "--max-num-batched-tokens",
                "8192",
                "--max-num-seqs",
                "256",
                "--port",
                "13333",
                "--tensor-parallel-size",
                "2",
                "--pipeline-parallel-size",
                "1"
            ],
            "env": {
                // 可选：设置 CUDA_VISIBLE_DEVICES 控制 GPU 使用
                // "CUDA_VISIBLE_DEVICES": "0,1",
                // 开启 vLLM 调试日志
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
            },
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "stopOnEntry": false
        }
    ]
}
[图片]
只要点击 Run → Debug，即可开始调试。稍等片刻，程序会自动停在断点处。从启动命令可以看出，我们使用了两张显卡进行张量并行（TP=2），因此后端自动选择 
MultiprocExecutor 会根据并行配置创建对应数量的 Worker 实例。以 TP=2 为例，Executor 创建两个 Worker 进程，每个 Worker 独占一张 GPU。Executor 与 Worker 之间通过 collective RPC 机制进行任务分发和结果回收，本质上是 Executor 广播方法名 + 参数，Worker 执行后通过独立通道返回结果。具体分工如下：
- Worker 负责持有模型权重的分片（TP=2 时，每个 Worker 持有每层权重的一半分片，而非模型的部分层），接收 Executor 统一下发的执行请求，运行前向传播，并通过 NCCL 在层内自动同步中间结果。Worker 在初始化时根据分配的 local_rank 绑定对应 GPU，根据 rank 加入 NCCL 通信组。
- Executor 接收 EngineCore 的调度输出（SchedulerOutput），通过共享内存广播队列统一下发给所有 Worker，并收集返回的推理结果。
每个 Worker 启动后会进入 worker_busy_loop，持续监听广播队列并执行推理，直到收到终止信号。以此处为例，Worker 在 worker_busy_loop 中收到了 Executor 下发的 method 调用。
[图片]
[图片]
回到之前的流程。由于我们配置了双 GPU，Executor.get_class() 会根据 vllm_config 中 world_size > 1 的配置返回 MultiprocExecutor 类（而非实例）。
具体路径如下：AsyncLLM.from_vllm_config() 调用 Executor.get_class(vllm_config) 确定对应的 Executor 类，随后将其作为构造参数传入 AsyncLLM.__init__()。
class AsyncLLM(EngineClient):
    def from_vllm_config(
            cls,
            vllm_config: VllmConfig,
            start_engine_loop: bool = True,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: Optional[list[StatLoggerFactory]] = None,
            disable_log_requests: bool = False,
            disable_log_stats: bool = False,
            client_addresses: Optional[dict[str, str]] = None,
            client_index: int = 0,
        ) -> "AsyncLLM":
            
            ...
            ...
            # Create the LLMEngine.
            return cls(
                vllm_config=vllm_config,
                executor_class=Executor.get_class(vllm_config),
                start_engine_loop=start_engine_loop,
                stat_loggers=stat_loggers,
                log_requests=not disable_log_requests,
                log_stats=not disable_log_stats,
                usage_context=usage_context,
                client_addresses=client_addresses,
                client_index=client_index,
            )
3.1.2 对 Executor 中通信队列的初始化
Executor.get_class 会获取到 executor 的具体实现类，并将其保存在 AsyncLLM 类的 executor_class 属性中。在 AsyncLLM 的 init 方法里，AsyncLLM 自身并不实例化 Executor，它只是将这个类传递给 EngineCoreClient.make_async_mp_client。随后，在 EngineCore 后台进程启动时，Executor 实例化会发生在 EngineCore 后台进程内部，而非 AsyncLLM.init 中。
[图片]
# vllm/vllm/v1/engine/llm_engine.py：AsyncLLM.__init__() 函数
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class, # 传递给 EngineCore 客户端
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_index=client_index,
)
注意：Executor.get_class() 返回的是一个类，不是实例。真正的实例化发生在更深的调用链中：
AsyncLLM.__init__ → EngineCoreClient.make_async_mp_client() → launch_core_engines() → EngineCoreProc.__init__ → executor_class(vllm_config)。
也就是说，Executor 实例最终在引擎进程内部完成创建，而不是在 from_vllm_config 或 AsyncLLM 初始化的调用栈里直接实例化。
rpc_broadcast_mq 通信队列
在 Executor 的初始化过程中，最关键的步骤是建立与 Worker 之间的通信机制。Executor 与多个 Worker 之间是一对多的关系，通过两类消息队列进行数据和指令的传输：
1. 一类是 rpc_broadcast_mq，由 Executor 与所有 Worker 共享，负责统一下发命令；
2. 另一类是 worker_response_mq，每个 Worker 各自持有独立的一个实例（而非所有 Worker 共用一个），负责将本 Worker 的执行结果单独回传给 Executor。
因此，队列总数为 1 + N（N 为 Worker 数量），例如 TP=2 时共有 3 个队列：1 个广播队列和 2 个结果回传队列。
暂时无法在飞书文档外展示此内容
rpc_broadcast_mq 是 Executor 向所有 Worker 统一下发任务的唯一广播通道。Executor 将方法名及参数写入该队列，每个 Worker 被动监听，从中取出相同的任务数据。在张量并行场景下，所有 Worker 需要完全一致的 SchedulerOutput，因此广播只需一次，所有 Worker 同时收到，无需逐一点对点分发。
该队列内部采用共享内存 + ZMQ PUB/SUB 两层混合架构（distributed/device_communicators/shm_broadcast.py）：
- 小数据（方法名、调度元数据等）走共享内存环形缓冲区 ShmRingBuffer，Writer 写入后多个 Reader 各自偏移读取，避免进程间数据复制，实现零拷贝。
- 大数据（如结构化输出的 grammar bitmask 张量、KV connector 元数据等）走 ZMQ 的 XPUB/XSUB 模式——Writer 将数据发布到 XPUB 套接字，Worker 通过 XSUB 套接字订阅接收。对于跨节点 Worker，这部分走 TCP PUB/SUB。
以上的Writer也就是Executor。综上，Executor 只需一次写入，所有 Worker——无论本地还是远程——统一从同一个 rpc_broadcast_mq 获取相同指令，结构清晰且高效。
暂时无法在飞书文档外展示此内容
rpc_broadcast_mq 在 Executor 初始化过程中被创建，其中一个关键参数是 max_chunk_bytes。在 MultiprocExecutor 中，该值取自环境变量 VLLM_MQ_MAX_CHUNK_BYTES_MB（默认 16 MB），并传递给 MessageQueue 构造函数。
1. max_chunk_bytes 用于控制进程间通信的数据传输方式：当序列化后的数据总大小小于 max_chunk_bytes 时，通过共享内存环形缓冲区（ShmRingBuffer）直接传输，实现零拷贝；
2. 当数据总大小达到或超过 max_chunk_bytes 时，共享内存仅写入一个溢出标记（overflow flag = 1），实际数据改由 rpc_broadcast_mq 关联的 ZMQ XPUB/SUB 套接字进行传输，以避免共享内存预分配过大。
class MultiprocExecutor(Executor):
    def _init_executor(self) -> None:
        
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
因此，MessageQueue 采用两种通信方式：
- 小数据通过共享内存（SHM）传输，高效且零拷贝；
- 大数据则使用 ZeroMQ 套接字传输。
传递参数和方法在压缩后占用的字节数来区分的。
在 MessageQueue 的初始化阶段，self.buffer 对应共享内存通道 ShmRingBuffer，self.local_socket 对应 ZeroMQ 通道 XPUB。为了让 Executor 感知各个 Worker 是否已完成连接，此处将 XPUB_VERBOSE 设为 True。启用后，每次有新的订阅者接入并发起订阅时，XPUB 会向上层暴露一条订阅通知消息。
每个 Worker 在连接并完成订阅后，都会各自产生一条这样的通知。Executor 在 wait_until_ready() 中通过 self.local_socket.recv() 逐条接收这些通知；当预期数量的通知全部收齐后，即可认为所有 Worker 均已上线。
借助这一机制，Executor 能够在真正开始广播之前确认所有 Worker 已准备就绪，从而保证后续广播发布语义的可靠性。下面给出用于确认连接成功的握手示意图。
暂时无法在飞书文档外展示此内容
# 代码有删减，只保留关键的部分
class MessageQueue:
    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: Optional[list[int]] = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,
        max_chunks: int = 10,
        connect_ip: Optional[str] = None,
    ):

        if n_local_reader > 0:
            # 初始化共享内存
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes,
                                        max_chunks)
            # 初始化套接字
            self.local_socket = context.socket(XPUB)
            # XPUB_VERBOSE 使 XPUB 在每次有新的订阅者连入时，                     
            # 都接收一条订阅通知（而非仅第一条），便于 wait_until_ready 计数
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_addr = get_open_zmq_ipc_path()
            self.local_socket.bind(local_subscribe_addr)
在 MultiprocExecutor 中，rpc_broadcast_mq 被初始化为 Executor 向所有 Worker 统一下发调度请求和执行数据的广播通道，其内部根据 payload 大小自动选择通信路径：小于 max_chunk_bytes 走共享内存环形缓冲区，否则走 ZMQ 套接字。
暂时无法在飞书文档外展示此内容
workerProc 连接 rpc_broadcast_mq 通信队列
此前我们已在 Executor 中创建了 rpc_broadcast_mq，其内部持有共享内存区域和 ZMQ XPUB 套接字。接下来需要让两个 Worker 连接到同一个队列。
由于 rpc_broadcast_mq 内部包含不可直接跨进程传递的资源（共享内存文件描述符、套接字地址等），Executor 通过 export_handle() 将这些连接信息导出为一个可序列化的 Handle 对象（shm_broadcast.py:263-269）：
scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
Handle 是一个数据容器，包含以下连接元数据：
字段
含义
buffer_handle
共享内存区域的标识（供 ShmRingBuffer 重建）
local_subscribe_addr
ZMQ XPUB 的 IPC 地址（本地 Reader 连接用）
remote_subscribe_addr
ZMQ XPUB 的 TCP 地址（远端 Reader 连接用）
local_reader_ranks
哪些 rank 是本地 Reader
之后 scheduler_output_handle 被传入每个 Worker 的实例化过程（make_worker_process → WorkerProc.__init__），Worker 在内部调用 MessageQueue.create_from_handle(handle, rank) 重建 Reader 端的 rpc_broadcast_mq，从而与 Executor 的 Writer 端建立完整的广播通道。
class MultiprocExecutor(Executor):
  def _init_executor(self) -> None:

    self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                         self.world_size,
                                         max_chunk_bytes=max_chunk_bytes)    
    # 将self.rpc_broadcast_mq作为句柄传递给worker进程，用于为worker进程打开通信队列
    scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
    for rank in range(self.world_size):
      unready_workers.append(
        WorkerProc.make_worker_process(
          vllm_config=self.vllm_config,
          local_rank=rank,
          rank=rank,
          distributed_init_method=distributed_init_method,
          input_shm_handle=scheduler_output_handle,
        ))
通过初始化流程，Worker 在 WorkerProc.__init__ 中才真正连接 rpc_broadcast_mq 所传递的共享内存和套接字。
从第9行开始，Worker 依次：
1. 根据 handle 中的 buffer_handle 重建共享内存 ShmRingBuffer，成为本地 Reader；
2. 创建 ZMQ SUB 套接字，设置 SUBSCRIBE 为空字符串（订阅所有消息），连接到 handle 中的 local_subscribe_addr。
Executor 侧持有一个 XPUB 套接字并设置了 XPUB_VERBOSE，因此每个 Worker 连接并订阅后，Executor 会收到一条对应的订阅通知。在后续的 wait_until_ready() 调用中，Executor 先循环收取所有本地和远端 Reader 的订阅通知，收齐后才向所有已连接的 Worker 统一回发 b"READY" 消息，验证 PUB-SUB 通道双向畅通。
对应的各个 Worker 此时正阻塞在 self.local_socket.recv() 上等待这条 READY，一旦收到即可确认通信双方均已就绪，握手完成。
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
除了建立通信连接，Worker 在初始化阶段还需要完成 init_device()（根据 local_rank 绑定 GPU 并初始化 NCCL 通信组）和 load_model()（从磁盘加载权重到 GPU 显存），不过这两步不是本节的重点。
@staticmethod
def create_from_handle(handle: Handle, rank) -> "MessageQueue":
  self = MessageQueue.__new__(MessageQueue)
  self.handle = handle
  self._is_writer = False

  context = Context()

  if rank in handle.local_reader_ranks:
    assert handle.buffer_handle is not None
    # 用executor端的共享内存
    self.buffer = ShmRingBuffer(*handle.buffer_handle)
    self.current_idx = 0
    self.local_reader_rank = handle.local_reader_ranks.index(rank)
    self._is_local_reader = True
    self._is_remote_reader = False

    self.local_socket = context.socket(SUB)
    self.local_socket.setsockopt_string(SUBSCRIBE, "")
    socket_addr = handle.local_subscribe_addr
    # 连接executor端中的套接字 
    self.local_socket.connect(socket_addr)

3.1.3 通过 rpc_broadcast_mq 通信队列传递命令
暂时无法在飞书文档外展示此内容
在 TP=2 的设置下，一个 Executor 对应两个 Worker，并通过 rpc_broadcast_mq 向它们统一下发任务。以 execute_model 为例，Executor 分发调度请求的流程如下：
1. 调用 collective_rpc 向所有 Worker 发起远程执行请求：
# multiproc_executor.py:254-264
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc(
        "execute_model",                           # 方法名
        args=(scheduler_output,),                  # 参数：本次调度结果
        unique_reply_rank=self.output_rank,        # 只等这个 rank 的回复
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
    )
- method："execute_model"，Worker 在 busy loop 中通过 getattr(self.worker, method) 查表执行。
- args：传入 scheduler_output，即 Scheduler 产出的本轮调度请求序列。
- unique_reply_rank：指定只需等待哪一个 Worker 返回结果。值为 world_size - tp_size（最后一个 PP stage 的第一个 TP rank）。TP=2 且无 PP 时恰好为 0，这是因为在 TP 模式下，每层 linear 内部已经通过 NCCL all-reduce 同步了中间结果，所有 TP rank 产出的 logits 是完全一致的，无需额外的 AllReduce 或 Gather。unique_reply_rank 只是挑任意一个持有完整结果的 Worker 回传，避免重复接收 N 份相同数据。
2. collective_rpc 内部将 (method, args, kwargs, output_rank) 写入已初始化的 self.rpc_broadcast_mq，广播到所有 Worker。
3. 每个 Worker 在 worker_busy_loop 中从同一队列取出相同的任务，各自执行 self.worker.execute_model(scheduler_output)，但只有 rank == output_rank 的那个 Worker 将结果写回 response_mq 返回给 Executor。
# 代码有删减
def collective_rpc(self,
                   method: Union[str, Callable],
                   timeout: Optional[float] = None,
                   args: tuple = (),
                   kwargs: Optional[dict] = None,
                   non_block: bool = False,
                   unique_reply_rank: Optional[int] = None) -> list[Any]:

  deadline = None if timeout is None else time.monotonic() + timeout
  kwargs = kwargs or {}

  # 需要执行的方法
  send_method = method
  
  # 向self.rpc_broadcast_mq传递
  self.rpc_broadcast_mq.enqueue(
    (send_method, args, kwargs, unique_reply_rank))
  
  # 等待worker的返回
  workers = (self.workers[unique_reply_rank],
            ) if unique_reply_rank is not None else self.workers
  responses = []


def execute_model(
  self,
  scheduler_output,
) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
  ...
  ...
  (output, ) = self.collective_rpc(
    "execute_model",
    args=(scheduler_output, ),
    unique_reply_rank=self.output_rank,
    non_block=non_block,
    timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
  return output

通过 self.rpc_broadcast_mq.enqueue，Executor 将 (方法名, 参数, kwargs, output_rank) 四元组广播给所有 Worker。enqueue 内部根据数据量选择通信路径（shm_broadcast.py）：
- 数据序列化后，若 total_bytes + serialized_size < self.buffer.max_chunk_bytes，走共享内存。缓冲区的首字节写入 0，后续字节写入 buffer 数量及实际载荷。
- 若超出阈值，则走 ZMQ 套接字。此时共享内存缓冲区的首字节写入 1（溢出标记），实际数据通过 self.local_socket.send_multipart() 走 XPUB 发送。
Worker 在 dequeue 时先读取首字节：0 表示数据在共享内存中直接解析，1 表示数据已溢出至 ZMQ 通道，需从套接字接收。
class MessageQueue:
    def enqueue(self, obj, timeout: Optional[float] = None):
        """ Write to message queue with optional timeout (in seconds) """
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0  # not overflow
                    buf[1:len(serialized_obj) + 1] = serialized_obj
        if self.n_remote_reader > 0:
            self.remote_socket.send(serialized_obj)
 需要执行的方法及参数会被序列化并压缩，然后写入共享内存的buf段中。
暂时无法在飞书文档外展示此内容
3.2 Worker 工作流程
3.2.1 接收指令并执行
Worker 进程初始化完成后，不仅连接了来自 Executor的rpc_broadcast_mq消息队列，还会启动 worker_busy_loop，持续监听并处理 Executor 发来的指令。我们来看worker_busy_loop的实现：它通过循环调用 dequeue() 从 rpc_broadcast_mq 中读取命令和参数，这正是之前 enqueue 操作的逆过程。取出数据后，根据缓冲区首字节判断传输方式：
1. 若 buf[0] == 0：表示数据通过共享内存发送，直接对 buf[1:] 进行反序列化，还原出方法名和参数；
2. 若 buf[0] == 1：表示数据过大，通过 ZeroMQ 套接字发送，需调用 ZMQ 接收接口获取完整数据。
本地 Reader（与 Executor 同节点，_is_local_reader = True）：
1. 从共享内存缓冲区中读取首字节 buf[0] 判断传输方式：
  - buf[0] == 0：数据在共享内存中。缓冲区格式为：[0] [2字节: buffer数量] [4字节: buffer 0长度] [buffer 0数据] [4字节: buffer 1长度] [buffer 1数据] ...
  - 按此结构解析出 all_buffers，再通过 pickle.loads(all_buffers[0], buffers=all_buffers[1:]) 反序列化还原出完整的 Python 对象。
  - buf[0] == 1：数据溢出至 ZMQ 通道。调用 self.local_socket.recv_multipart() 从 SUB 套接字接收，同样用 pickle.loads 还原。
2. 远端 Reader（跨节点，_is_remote_reader = True）：远端 Worker 没有共享内存通道，所有数据始终通过 self.remote_socket.recv_multipart() 走 TCP PUB/SUB 接收，无需判断首字节。
# 篇幅原因，代码有删改
class MessageQueue:
    def dequeue(self,
            timeout: Optional[float] = None,
            cancel: Optional[Event] = None):
    """ Read from message queue with optional timeout (in seconds) """
    if self._is_local_reader:
        with self.acquire_read(timeout, cancel) as buf:
            overflow = buf[0] == 1
            if not overflow:
                obj = pickle.loads(buf[1:])
        if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
                    
class WorkerProc:   
    def worker_busy_loop(self):
            """Main busy loop for Multiprocessing Workers"""
            while True:
                # 读取来自队列中的方法和参数
                method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
                func = getattr(self.worker, method)
                # 执行
                output = func(*args, **kwargs)
3.2.2 Worker 与 Executor 建立连接
每个 Worker 都有一个 self.worker_response_mq 消息队列，用于将推理结果返回给 Executor。如注释所示：
- self.rpc_broadcast_mq：接收来自 Executor 的调度请求；
- self.worker_response_mq：回传执行结果。
两者的初始化方式不同：
1. self.rpc_broadcast_mq 通过 Executor 传递的句柄（包含套接字地址或共享内存）创建，直接连接已存在的通信通道；
2. self.worker_response_mq 由 Worker 在本地新建套接字和共享内存，并将其句柄返回给 Executor，供其连接和监听。Executor 侧用这个句柄重建 Reader 端，加入 response_mqs，后续在 collective_rpc 中从对应 rank 的 response_mq 读取结果。
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
在上述流程中，rpc_broadcast_mq 用于 Executor 向 Worker 发送方法和参数。与此同时，Worker 也通过 worker_response_mq 向 Executor 回传执行结果。worker_response_mq通道的建立同样需要双向握手：
- Worker 创建 worker_response_mq 并将其句柄通过create_from_handle方法返回给 Executor；
- Executor 在调用create_from_handle获取worker_response_mq时会连接该队列，并自动发送 ZMQ 订阅消息（触发 XPUB 事件）；
- Worker 在 wait_until_ready() 中检测到订阅后，向 Executor 发送 "READY" 消息；
- Executor 收到 "READY" 后，确认通道建立。
这一过程需两端协同完成，类似于 TCP 三次握手，确保通信可靠，具体表现为：
1. Executor（读端）：调用 wait_until_ready()，等待接收来自 Worker 的 "READY" 消息；
2. Worker（写端）：调用 wait_until_ready()，接收 Executor 的订阅通知，并发送 "READY" 响应。
只有通信双方都完成这一步握手，结果回传通路才算真正建立。此后，Executor 才会持有来自所有 Worker 的 worker_response_mq 队列数组，并能够依次获取各个 Worker 返回的输出。
与之相近但作用不同的另一个函数是 wait_for_ready。它是 Executor 用来等待所有 WorkerProc 子进程完成初始化的进程级同步握手。只有在这一步成功之后，Executor 才能获取到各个 Worker 的消息队列句柄；随后再调用 wait_until_ready，等待通信双方对应的消息队列真正建立连接，从而完成广播通道和结果回传通道的就绪确认。
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
# 代码有省略
class MultiprocExecutor(Executor):
    def _init_executor(self) -> None:
        # 发送订阅消息XPUB给Worker端
        self.workers = WorkerProc.wait_for_ready(unready_workers)
        for w in self.workers:
                w.worker_response_mq.wait_until_ready() # 等待Worker端发送确认消息（"Ready"）
                # 连接确立
        
class WorkerProc:
    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:
        worker_response_mq = MessageQueue.create_from_handle(response["handle"], 0)
        ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(unready_proc_handle, worker_response_mq))
        return ready_proc_handles
        
    @staticmethod
    def worker_main(*args, **kwargs):
        ...
        ...
        # 等待订阅段的XPUB订阅消息，收到后向订阅端发送"READY"消息
        worker.worker_response_mq.wait_until_ready()
        worker.worker_busy_loop()
3.2.3 Worker返回数据
Worker 端通过 enqueue 和 dequeue 配合，将模型执行结果写入 worker_response_mq。具体流程如下图所示。Executor 端需等待所有相关 Worker 返回结果后，才能从worker_response_mq队列中获取完整输出。
暂时无法在飞书文档外展示此内容
总结
1. 本节课深入剖析了 vLLM 中 MultiprocExecutor 的通信机制与执行流程，聚焦于在单机多卡场景下的核心作用，MultiprocExecutor 通过主从架构协调多个绑定 GPU 的 Worker 进程，自身不直接计算，而是通过高效的进程间通信（IPC）分发任务并收集结果。
2. 关键通信通道 rpc_broadcast_mq 采用混合传输策略：小数据（≤16MB）通过共享内存零拷贝传输，大数据则回退至 ZeroMQ 套接字，兼顾性能与灵活性。Executor 广播执行指令（如 execute_model）后，各 Worker 在 worker_busy_loop 中监听并反序列化命令，调用本地模型执行。随后，结果通过独立的 worker_response_mq 回传，该通道建立过程包含双向握手——Worker 检测到 Executor 订阅后发送 "READY" 确认，确保连接可靠。
3. 该设计实现了职责解耦：Engine 调度、Executor 分发、Worker 计算，为高性能推理提供了清晰且可扩展的分布式执行框架。
