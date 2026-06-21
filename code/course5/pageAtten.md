前言
vLLM 以分块化、哈希缓存与引用计数机制，革新 KV Cache 管理，显著提升大模型推理效率与显存利用率。
在大语言模型（LLM）的自回归生成（autoregressive decoding）过程中，每一步都会生成一个 token，并将其作为下一步的输入。为了提升推理效率，避免重复计算注意力机制中的 Key 和 Value 向量，系统会将这些中间结果缓存起来——这就是 Key-Value Cache（KV Cache）。
然而，传统的 KV Cache 管理方式存在若干关键问题，严重制约了推理性能：
1. 动态序列长度增长：每个请求的输入提示词长度不同，且在生成过程中逐步增长（尤其在采样解码时），导致 KV Cache 的大小不固定，难以统一管理；
2. 内存碎片化：若采用连续内存分配策略（如 PyTorch 的 aten::empty），随着序列动态扩展，频繁的内存重新分配会引发严重的内部和外部碎片，降低 GPU 显存利用率；
3. 冗余存储：在提示词共享或树状推测解码（speculative decoding）等场景中，多个请求可能具有相同前缀，但传统方法仍独立缓存，造成显著的存储浪费；
4. 批处理能力受限：由于 KV Cache 占用高占用时间长且内存碎片严重，实际可并行处理的 batch size 被大幅压缩，直接影响推理吞吐量。
暂时无法在飞书文档外展示此内容
self.cache_k = torch.zeros((
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,)).cuda()
值得注意的是，上述问题中的第 1、2、4 点本质上都指向同一个核心挑战：如何高效管理变长序列下的 KV Cache 内存，以减少碎片、提升显存利用率和批处理能力,这也是 vLLM 等现代推理框架重点优化的方向。针对上述 KV Cache 管理的挑战，开发人员尝试了多种解决方案，但每种方法都有其局限性。
传统策略及其缺陷
KV Cache 的显存占用
在 Transformer 模型中，每个注意力层都会维护独立的 Key（K）和 Value（V）缓存。对于长度为 $$L$$的序列，每一层的 KV Cache 所需显存大致与
$$L\times d_k \times num\_heads$$
成正比。随着自回归生成的进行，序列长度$$L$$逐步增加，KV Cache 也随之动态增长。因此，如何高效分配和管理这部分内存成为推理系统设计的核心问题。目前常见的 KV Cache 内存管理方式主要有两种：静态预分配 和 动态扩容，但二者均存在显著缺点。
静态预分配（固定最大长度）
一种简单做法是为每个请求预先分配最大支持长度（如 max_seq_len=8192）的 KV Cache 空间。这意味着即使当前序列仅生成了十几个 token，系统仍会占用长达数千 token 的缓存空间。
这种方式的主要问题包括：
- 显存浪费严重：大量预留空间长期未被使用，导致显存利用率极低；
- 加剧内存碎片：大块连续显存被提前锁定，其他新请求即使所需空间较小也无法复用这些“闲置但不可分割”的区域；
- 降低并发能力：由于显存无法高效共享，系统可同时处理的请求数量（batch size）受限，导致吞吐下降，请求拒绝率上升。
例如，一个最多生成 4096 个 token 的请求，在整个生命周期内独占对应大小的 KV Cache 资源，即便实际只用了几十个 token。在此期间，其他潜在请求可能因无法获得足够显存而被拒绝。
动态扩容（按需增长）
另一种思路是初始时为请求分配较小的 KV Cache，随后根据生成长度逐步扩展。虽然这种方法空间利用率更高，但也带来新的开销：
- 频繁显存分配/释放：每次扩容都需要调用类似cudaMalloc或cudaFree的操作，带来较高的 GPU 开销；
- 数据拷贝开销大：扩容通常需要申请更大的连续空间，并将原有数据复制过去，触发昂贵的 GPU 显存传输（memcpy），消耗带宽并拖慢推理速度；
- 加剧碎片化：频繁的分配与释放容易产生外部碎片，进一步限制大块连续内存的可用性。
综上，无论是静态预分配还是动态扩容，都无法兼顾高显存利用率、低延迟和高并发的需求。这也促使了更先进的 KV Cache 管理机制的发展——例如 vLLM 中提出的 PagedAttention，借鉴操作系统虚拟内存的思想，将物理显存划分为固定大小的页，实现高效、灵活且低碎片的 KV 缓存管理。
暂时无法在飞书文档外展示此内容
分页显存的管理办法
分页管理的优势
1. 消除连续显存依赖
传统方式通常需要为每个请求分配一段连续显存来存储 KV Cache。当请求长度不同、生命周期不同，或者请求频繁进出时，显存很容易产生碎片，即使总空闲显存足够，也可能因为没有足够大的连续空间而分配失败。
PagedAttention 允许一个请求使用多个非连续的物理 block。只要显存池中还有足够数量的空闲 block，就可以完成 KV Cache 分配，从而提高显存利用的灵活性和分配成功率。
2. 显著减少内存浪费
由于 KV Cache 按固定大小的 block 分配，一个请求只会在最后一个 block 中产生少量未使用空间。最坏情况下，每个请求最多浪费 block_size - 1 个 token 的空间。
相比传统静态预分配方式可能一次性为请求保留大量 token 空间，分页式管理把内部碎片控制在较小范围内，同时也避免了连续分配带来的外部碎片问题。
3. 支持高效共享机制（Prefix Cache）
当多个请求拥有相同的 prompt 前缀时，它们可以共享已经计算好的 KV Cache block，而不需要重复存储相同的 KV 值。
共享机制带来两个好处：
- 节省显存：相同前缀对应的 KV Cache 只需要保存一份。
- 减少计算：共享前缀已经完成计算，后续请求可以复用对应结果。
当不同请求的生成路径开始分歧时，系统再为新的分支分配独立的物理 block。这样既能复用公共前缀，又能保证不同请求后续生成内容互不影响。
工作机制简述
结合了静态和动态分配的方法
系统启动时，将可用的 KV Cache 显存预先划分为一个物理块池。每个请求的 KV Cache 不再需要一段连续的物理地址，而是通过块表（Block Table）记录其所使用的各个物理块。这种虚拟化管理方式使上层调度器只需处理块 ID，无需关心块在显存中的实际物理位置。
PagedAttention 引入分块管理机制，从根本上解决了传统 KV Cache 管理中的两大痛点：显存利用率低与碎片化严重。这一机制为高效的大模型推理提供了关键支撑。从示意图中可见，每个请求的 KV Cache 由多个非连续的物理块组成，请求仅需维护一个记录块 ID 的映射表即可。
暂时无法在飞书文档外展示此内容
计算可分配显存块的数量
在分块式显存管理中，KV Cache 被划分为一系列固定大小的物理块。请求本身维护逻辑块到物理块的映射关系，模型执行时再根据这张映射表访问对应的 KV Cache。
系统中可用于 KV Cache 的物理块数量，主要取决于显存中还剩下多少空间可以分配给 KV Cache。这个空间通常受以下因素影响：
1. 用户显存限制
  用户可以通过配置限制推理系统最多使用多少显存，例如只允许使用 GPU 总显存的一定比例。这样可以避免推理服务占满整张 GPU，给系统、其他进程或框架运行时预留必要空间。
2. 模型自身占用
  模型权重会长期驻留在显存中，推理过程中的中间激活值、临时 buffer、CUDA graph、通信 buffer 等也会占用显存。模型越大，或者单步计算所需的中间状态越多，留给 KV Cache 的显存就越少，可分配的物理块数量也就越少。
为了更准确地评估这些非 KV Cache 部分的显存开销，系统通常不会只依赖理论公式估算，而是会执行一次 profiling：使用随机输入数据跑一遍前向传播，观察实际显存峰值占用。这样可以把权重、激活值以及运行时临时 buffer 等因素都纳入统计，得到更接近真实推理过程的显存使用情况。
在完成 profiling 后，系统会根据用户允许使用的显存上限，扣除模型权重、激活值和其他运行时开销，剩余部分用于建立 KV Cache block 池。最终可用的物理块数量大致可以理解为：
1. 可用于 KV Cache 的显存 = 用户允许使用的总显存 - 模型权重显存 - 推理中间激活值/临时 buffer 显存 - 其他运行时预留开销
2. 物理块数量 = 可用于 KV Cache 的显存 / 单个 KV Cache block 大小
暂时无法在飞书文档外展示此内容
1. 维护空闲显存块 根据 物理块数量 也就是num_gpu_blocks 初始化一批物理块，并维护一个空闲队列。调度器需要为请求分配 KV Cache 时，就从空闲队列中取出可用 block；当请求结束或某些 block 不再被引用时，再将其归还到空闲队列。
2. 维护哈希到显存块的映射 为了支持 prefix cache，系统会为已经计算过的完整 block 建立哈希映射。这个映射关系可以理解为： block_hash -> physical block
  当新的请求拥有相同 prompt 前缀时，系统可以通过哈希快速查找是否已经存在可复用的 KV Cache block。如果命中，就不需要重新分配和重新计算对应 block，只需要复用已有物理块，并将该 block 的引用计数 ref_cnt 加 1。
3. 支持引用计数与复用 每个物理块都会维护引用计数。引用计数表示当前有多少请求或逻辑块正在使用这个物理块。
  - ref_cnt > 0：block 正在被使用，不能回收
  - ref_cnt = 0：block 当前无人引用，可以重新进入空闲队列
  当请求结束、被抢占，或者某个逻辑块不再需要时，对应物理块的 ref_cnt 会减少。如果引用计数降为 0，该 block 就会被重新挂入空闲队列，等待后续重新分配。
4. 支持延迟清理哈希映射 对于曾经参与 prefix cache 的 block，即使它的 ref_cnt 降为 0，系统也不一定立即删除它的哈希映射。这样做可以保留复用机会：如果后续请求再次命中相同前缀，就可以直接复用该 block。
  这样设计的核心目的是保留复用机会：假设 block A 曾存储某段 prompt 前缀的 KV Cache。当最后一个引用它的请求结束时，ref_cnt 变为 0，block A 被放入空闲队列末尾等待回收。若此时恰好有新请求携带相同前缀，由于哈希映射尚未失效，该请求仍能通过 block_hash 查找到 block A。系统会将其 ref_cnt 从 0 重置为 1，并将其从空闲队列中移除，实现零开销复用。
  也就是说，当该 block 真正从空闲队列中被重新分配给新的内容时，系统才会清除旧的哈希映射，避免哈希表指向已经被覆盖的 KV Cache 内容。
暂时无法在飞书文档外展示此内容
获取显存块的数量
一个block 存放16个token的kv cache
一个token需要的kv cache大小就是num_heads × head_size × 2 × data_type，一个 block 的大小则是：单层单 block KV Cache 大小 = block_size × 2 × num_kv_heads × head_size × dtype_size
在分配 KV Cache 显存块之前，系统需要先确定当前还能拿出多少显存用于 KV Cache。这一流程对应 _initialize_kv_caches，主要包括以下阶段：
1. 获取 KV Cache 配置
系统会先通过 get_kv_cache_specs 收集每个 attention 层的 KV Cache 规格信息，包括：
  - num_kv_heads：KV head 的数量
  - head_size：每个 head 的维度
  - block_size：每个 KV Cache block 中包含的 token 数量
  - dtype：KV Cache 使用的数据类型
这些信息决定了一个 KV Cache block 应该长什么样，也就是每层每个 block 需要占用多少显存。
2. 通过模拟推理确定 block 数量
此时系统还不知道最多能分配多少个 block，也就是num_gpu_blocks。这个值不能只靠理论估算，因为模型权重、临时 buffer、中间激活、CUDA graph 等都会影响实际可用显存。
  因此，系统会使用 dummy/random input 执行一次模拟前向传播，统计实际显存占用，再根据用户设定的显存使用比例，计算剩余可用于 KV Cache 的显存，最终得到可分配的 GPU block 数量。
3. 分配并组织 KV Cache 显存
拿到 num_gpu_blocks 后，系统会真正申请 KV Cache 显存，并按照 block 组织起来。底层张量通常会为某一层或某一组 KV Cache 分配一片连续存储，然后按 block 切分；但从请求视角看，一个请求的逻辑 KV Cache 不要求连续，可以映射到多个非连续的物理 block。不同层、不同 cache group 的底层存储也不要求彼此连续。
kv_cache_specs 描述的是每一层 KV Cache block 的规格。它告诉系统：如果要为这个模型建立 KV Cache block 池，每个 block 应该包含多少 token、每个 token 的 K/V 张量维度是多少、用什么 dtype 存储。至于一共能建立多少个 block，即 num_gpu_blocks，需要在 profiling 之后根据实际可用显存动态确定。
FullAttentionSpec(block_size=16, 
                  num_kv_heads=2, 
                  head_size=128, 
                  dtype=torch.bfloat16, ...) 
在 _initialize_kv_caches() 中，对应变量定义和获取方式为：
kv_cache_specs = self.model_executor.get_kv_cache_specs()
它既不是一个单独的 KVCacheSpec，而是一个 list：list[i] 就是第 i 个 worker 所辖所有层的dict[str, KVCacheSpec]，这个 dict 里包含了该 worker 上所有层的 KV Cache 规格。
kv_cache_specs: list[dict[str, KVCacheSpec]]
kv_cache_specs的数据结构大致如下：
kv_cache_specs = [
    {
        "model.layers.0.self_attn": FullAttentionSpec(...),
        "model.layers.1.self_attn": FullAttentionSpec(...),
        ...
    },
    ...
]
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    # 1. 获取搜集每个attention层的基本信息
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    # 2. 通过模拟运行的方式获取可以使用的显存数
    available_gpu_memory = (self.model_executor.determine_available_memory())
    self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    
    kv_cache_configs = [
        get_kv_cache_config(vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker)
    for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(kv_cache_specs, available_gpu_memory)]
在收集完各个 attention 层的 KV Cache 规格信息后，系统可以开始计算需要为 KV Cache 分配多少物理块。此处关键的并非立即分配显存，而是先明确当前 GPU 上还有多少显存可用于 KV Cache。
该过程主要在 determine_available_memory() 中完成。它通过一次模拟推理（即 profile_run()）统计模型在真实执行路径下的显存占用。相较于纯理论估算，这种方法更为可靠，因为中间激活、临时 buffer、CUDA graph 等开销难以通过公式精确计算。
@dataclass
class MemoryProfilingResult:
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot
    before_profile: MemorySnapshot   # ← free_memory 在这里面
    after_profile: MemorySnapshot    # ← free_memory 在这里面
    profile_time: float = 0.0
此过程中需要关注几个关键变量：
1. profile_result
profile_result 是模拟推理后的显存 profiling 结果，包含模型执行过程中的实际显存占用信息。其中重点关注：
- free_gpu_memory：profiling 结束后 GPU 的空闲显存，取自 profile_result.after_profile.free_memory。
- non_kv_cache_memory：非 KV Cache 部分的显存占用，主要包括模型权重、中间激活、临时 buffer 等运行时开销。
2. self.requested_memory
self.requested_memory 表示 vLLM 允许自己使用的最大显存预算，通常由 GPU 总显存乘以用户配置的 gpu_memory_utilization 得到：
self.requested_memory = total_gpu_memory × gpu_memory_utilization
它不是整张显卡的全部显存，而是用户允许 vLLM 使用的显存上限。
3. 可用于 KV Cache 的显存
完成 profiling 后，系统用允许使用的显存预算减去非 KV Cache 的显存占用，得到剩余可分配给 KV Cache 的显存：
available_kv_cache_memory = self.requested_memory - profile_result.non_kv_cache_memory - cudagraph_memory_estimate_applied
若不考虑 CUDA graph 额外预留，可简化理解为：
available_kv_cache_memory ≈ self.requested_memory - non_kv_cache_memory
因此，整体逻辑可概括为：
用户允许 vLLM 使用的最大显存  
 – 模型权重和推理运行时开销  
 = 可用于 KV Cache 的显存预算  
获得这部分显存预算后，系统再结合每个 KV Cache block 的大小，进一步计算可分配的物理块数量，即后续的 num_blocks / num_gpu_blocks。
可用 KV Cache 显存 = self.requested_memory - profile_result.non_kv_cache_memory
如果系统中有两张显卡，且对应两个 worker，则 model_executor.determine_available_memory() 返回的 available_gpu_memory 列表长度通常为 2，其中每个元素分别表示对应 worker/GPU 可用于 KV Cache 的显存大小。
def determine_available_memory(self) -> int:
  """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  GiB = lambda b: b / GiB_bytes

  # Execute a forward pass with dummy inputs to profile the memory usage
  # of the model.
  with memory_profiling(self.init_snapshot, \ 
                        weights_memory=int(self.model_runner.model_memory_usage)) as profile_result:
    self.model_runner.profile_run()
    
  free_gpu_memory = profile_result.after_profile.free_memory
    
  available_kv_cache_memory = self.requested_memory - profile_result.non_kv_cache_memory
  gc.collect()

  return int(available_kv_cache_memory)
暂时无法在飞书文档外展示此内容
get_kv_cache_config 会根据模型各层的 KV Cache 规格以及当前可用于 KV Cache 的显存大小，生成 KVCacheConfig。对于普通的 uniform full attention 模型，各层 KV Cache 规格相同，因此可以使用统一的 page_size 来计算每层可分配的 block 数量。
所以page_size 表示单层、单个 KV Cache block 占用的字节数，另外因为在 vLLM 里，一个 block 不是只存 1 个 token，而是存 block_size 个 token 的 KV Cache。对普通 attention 来说，一个 block 需要同时存Key cache和Value cache，所以公式是：
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
这里的 2 表示同时存储 Key 和 Value；block_size 表示每个 block 中包含多少个 token。
1. block_size：表示每个 KV Cache block 包含的 token 数量。例如 block_size = 16，即一个 block 可存储 16 个 token 的 KV Cache。
2. num_kv_heads ：指 KV head 的数量，注意其不一定等于 attention head 数。在普通 MHA 中，num_kv_heads = num_attention_heads；在 GQA/MQA 中，num_kv_heads 会更小。
3. head_size：指每个 KV head 的维度大小。例如 head_size = 128，即每个 head 的 Key 或 Value 向量长度为 128。
4. dtype_size：指每个元素占用的字节数。
在得到 page_size 后，可以计算 num_blocks：
num_blocks = available_memory // page_size // num_layers
这里的 num_blocks 表示每个 attention layer 可分配的 KV Cache block 数量，所以单个 attention layer 的 KV Cache 容量上限可表示为：
per_layer_size = page_size × num_blocks
在完成各 attention 层 KV Cache 规格收集并通过 profiling 得到可用于 KV Cache 的显存大小后，系统调用 get_kv_cache_configs() 为每个 worker 生成 kv_cache_configs，其中包含各 worker 的 KV Cache 配置。
申请显存块
[图片]
接下来，Engine 会通过 self.model_executor.initialize_from_config(kv_cache_configs) 将这些配置下发到 Worker。Worker 取出自己 rank 对应的 KVCacheConfig，并进入实际的 KV Cache 显存初始化流程：
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    self.model_runner.initialize_kv_cache(kv_cache_config)
initialize_from_config() 是 Worker 侧 KV Cache 初始化的入口，Engine 在完成 KV Cache 规格收集、显存 profiling 和 KVCacheConfig 生成后，会将 kv_cache_configs 下发给各个 worker。每个 worker 根据自己的 global_rank 取出对应的 kv_cache_config，然后进入 GPUWorker.initialize_from_config()。
如上文所述，kv_cache_config会告诉每个 worker：总共分配多少个物理 block（num_blocks）、每个 block 多大、要创建哪些 GPU tensor 以及它们各自服务于哪些 attention 层。Worker 拿到后据此分配显存并初始化本地的 block 池和块表。
在 GPUWorker.initialize_from_config() 中，首先会将配置中的 block 数量写入本地 cache 配置：
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
这表示当前 worker 上可用的 GPU KV Cache block 数量已经确定。随后，worker 会调用：
self.model_runner.initialize_kv_cache(kv_cache_config)
把真正的 KV Cache 初始化工作交给 model runner。
- initialize_kv_cache() 是 model runner 侧的 KV Cache 初始化主流程。它的主要职责包括：保存 kv_cache_config、初始化 attention backend、准备 backend 需要的 block size、初始化BlockTable、申请 KV Cache 显存，并将原始显存整理成 backend 可以直接使用的张量格式。
- 常见情况下，普通 attention 会使用 FlashAttention 相关 backend，例如 FlashAttentionBackend；但具体使用哪个 backend 并不是固定的，会受到硬件平台、attention 类型、模型结构和 vLLM 配置影响。不同 backend 对 KV Cache 的布局要求可能不同，因此后续 reshape 的格式也会有所差异。
- 实际申请显存的逻辑在 _allocate_kv_cache() 中，它会遍历 kv_cache_config.kv_cache_tensors，根据每个 tensor 记录的 size 在 GPU 上申请原始 buffer（torch.zeros(size, dtype=torch.int8, device=device)）。之后，_reshape_kv_cache() 再根据 attention backend 的要求将这些原始 buffer 转换和 reshape 成最终执行时使用的 KV Cache 张量。
def _allocate_kv_cache_tensors(
    self,
    kv_cache_config: KVCacheConfig,
) -> dict[str, torch.Tensor]:
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(
            kv_cache_tensor.size,
            dtype=torch.int8,
            device=self.device,
        )
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    return kv_cache_raw_tensors
这里的 kv_cache_tensor.size 是前面根据 page_size 和 num_blocks 计算出来的。对于普通 uniform full attention 模型，可近似理解为：
per_layer_size = page_size × num_blocks
其中：
page_size = 单层单个 KV Cache block 占用的字节数
num_blocks = 每层可用的物理 block 数量
此处的torch.zeros(..., dtype=torch.int8) 并不表示 KV Cache 最终以 int8 存储。这里先用 int8 申请原始字节 buffer，是为了按 kv_cache_tensor.size 精确控制申请的字节数。随后系统根据每层的 KVCacheSpec，将这段 buffer 重新解释为实际的 KV Cache dtype（例如 bfloat16），并 reshape 成 attention backend 期望的布局。后续 reshape 的逻辑可概括为：
raw_tensor = kv_cache_raw_tensors[layer_name]
raw_tensor = raw_tensor.view(dtype)
raw_tensor = raw_tensor.view(kv_cache_shape)
kv_caches[layer_name] = raw_tensor.permute(*inv_order)
其中 kv_cache_shape 由 attention backend 根据 num_blocks、block_size、num_kv_heads、head_size 和 cache_dtype 等信息生成。最后调整维度顺序，不同 backend 可能对 KV Cache 的维度排列有不同的期望，例如有的 backend 希望 K/V 维度在最前，有的则希望 block 维度或 head 维度更靠前。permute(*inv_order) 就是将前面为满足 stride/layout 要求而调整过的维度顺序，再转换回 backend 使用时约定的逻辑顺序。
最终，每个 attention layer 都会关联到一块已分配并按 backend 要求整理好的 KV Cache 张量。后续模型执行时，请求本身并不要求占用连续的 KV Cache 空间，而是通过 block table 记录“逻辑 block 到物理 block”的映射关系。PagedAttention 在计算 attention 时，根据这张映射表定位对应的物理 KV Cache block，从而在请求视角下实现非连续 block 的灵活分配与高效访问。
[图片]
每个 GPUModelRunner 实例都会维护自己的一组 KV Cache 张量，即自己的self.kv_caches。在张量并行场景下，例如 tp=2 使用两张 GPU，通常会启动两个 worker，每个 worker 对应一个 GPU，并各自拥有一个独立的 GPUModelRunner 实例，因此也会有两组独立的 KV Cache：
GPU 0 / Worker 0 / GPUModelRunner 0 -> kv_caches[0]
GPU 1 / Worker 1 / GPUModelRunner 1 -> kv_caches[1]
这两组 KV Cache in 物理显存上相互独立，分别存储在各自的 GPU 之上。由于张量并行（Tensor Parallelism）会将 attention heads 及相关权重切分至不同 GPU，每个 GPU 通常仅保存本 rank 计算所需的 KV Cache 分片，而非完整复制一份全量 KV Cache。
从调度视角看，请求的 block id 与 block table 逻辑在各 TP rank 间保持一致；但从存储视角看，同一个 block id 所对应的实际上是不同 GPU 本地 KV Cache tensor 中的对应 block。因此，KV Cache 的逻辑分配是统一的，而实际存储则为 per-worker、per-GPU 的分布形式。
暂时无法在飞书文档外展示此内容
至此，我们通过四个步骤完成了注意力层基本信息的收集，主要包括其维度信息和对应的计算后端（即使用哪种实现方式）。基于这些信息，我们申请了相应大小的显存块，并将其划分为 num_blocks 个块，供 PageAttention 使用。此时，我们可以直接使用 KV Cache 中的物理块，如下图所示。但为了更高效地管理逻辑块并支持前缀共享等机制，vLLM设计了 BlockPool 来统一管理逻辑块。
暂时无法在飞书文档外展示此内容
注意：在 vLLM v0.13 中，原有的 get_kv_cache_config 已更新为 get_kv_cache_configs，用于同时为多个 worker 生成 KVCacheConfig。
get_kv_cache_configs 首先将所有 worker 各自的 KV cache spec 合并为全局 spec，生成 KV cache groups，随后将全局 groups 投射到各 worker，委托 get_kv_cache_config_from_groups() 为每个 worker 生成完整的 KV cache 配置（含 num_blocks 和 tensor 分配）。
无论是 uniform 模型还是 hybrid 模型，最终都走同一通用路径，使用统一的 num_blocks 计算公式：
num_blocks = available_memory // page_size // group_size
其中 page_size 是单个 layer 的物理页尺寸（各 group 统一后相同），group_size 是每个 group 中的 layer 槽位数（对齐后各组一致）。
两者的差异在于 group 的组织方式：
- uniform 模型：所有 layer 归入同一个 group，group_size 即总层数，每个 layer 独占一个底层 tensor
- hybrid 模型：不同 attention 类型的 layer 分入不同 group，通过 padding 将各组 layer 数对齐为统一值（即 group_size），然后按槽位分配底层 tensor：不同 group 处于同一槽位的 layer 绑定到同一个底层 tensor，从而复用同一个显存池
需要强调的是，共享底层 tensor 不等于共享同一份 KV 数据。同一个 KV cache group 内的 layer 共享 block table，而不同 group 拥有各自独立的 block table。即使不同 group 的 layer 绑定到同一个底层 tensor，它们也会通过各自的 block table 映射到不同的 physical block，最终读写的是同一 tensor 中不同的物理区域。
KVCacheSpec 用于描述单层的 KV Cache 格式，核心字段包括：
- block_size：每个 block 容纳的 token 数量（常为 16）
- page_size_bytes：该层单个 block 的物理字节数，通常计算公式为 2 × block_size × num_kv_heads × head_size × dtype_size（2 代表 K、V 两份缓存）
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]: 

    kv_cache_configs.append(
        get_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_groups_one_worker,
            kv_cache_spec_one_worker,
            available_memory_one_worker,
        )
    )
    
def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_specs: dict[str, KVCacheSpec],
    available_memory: int,
) -> KVCacheConfig:

    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    # 步骤1：计算出块的大小
    page_size = get_uniform_page_size(kv_cache_specs)
    assert group_size > 0, "group_size must be greater than 0"
    # 步骤2：计算出块的数量
    num_blocks = get_num_blocks(
        vllm_config, group_size, available_memory, page_size
    )
    kv_cache_tensors = []
    for i in range(group_size):
        shared_by = []
        for j in range(len(kv_cache_groups)):
            if i < len(kv_cache_groups[j].layer_names):
                shared_by.append(kv_cache_groups[j].layer_names[i])
        kv_cache_tensors.append(
            KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
        )
以下是完整的申请kv cache block流程
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    # 1. 获取搜集每个attention层的基本信息
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    # 2. 通过运行的方式获取可以使用的显存数
    available_gpu_memory = (self.model_executor.determine_available_memory())
    self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    # 3. 为所有层生成统一的kv cache配置，计算的方法在_get_kv_cache_config_uniform_type中
    kv_cache_configs = [
        get_kv_cache_config(vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker)
    for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(kv_cache_specs, available_gpu_memory)]
    # 4. 申请显存并划分 调用后续的initialize_kv_cache
    self.model_executor.initialize_from_config(kv_cache_configs) # 
    
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    """
    Initialize KV cache based on `kv_cache_config`.
    Args:
        kv_cache_config: Configuration for the KV cache, including the KV
        cache size of each layer
    """
    self.kv_cache_config = kv_cache_config
    self.may_reinitialize_input_batch(kv_cache_config)
    # 为每个attention申请kv cache空间
    self.initialize_attn_backend(kv_cache_config)
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
    
def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
    
def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """
    Initializes the KV cache buffer with the correct size. The buffer needs
    to be reshaped to the desired shape before being used by the models.
    Args:
        kv_cache_config: The KV cache config
    Returns:
        dict[str, torch.Tensor]: A map between layer names to their
        corresponding memory buffer for KV cache.
     """
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    # 为每个attention层申请需要用到的kv cache空间
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size,
                             dtype=torch.int8,
                             device=self.device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    ...
    return kv_cache_raw_tensors
BlockPool
Prefix caching
为了高效管理 KV Cache 并屏蔽底层显存分配的复杂性，vLLM 引入了基于 PagedAttention 的内存管理机制。其核心是 BlockPool 类：系统初始化时，根据此前算出的 num_blocks 预创建等量的 KVCacheBlock 元数据对象（每个含 block_id、引用计数、hash 等），所有 block 以链表形式构成一个逻辑块池。
实际 GPU 显存则由上层 KVCacheConfig 按 tensor 粒度一次性分配，BlockPool 仅负责维护"哪个 block_id 已分配、被多少请求引用、是否命中前缀缓存"等逻辑状态。运行时，KVCacheManager 作为对外接口，委托内部的 KVCacheCoordinator 协调 BlockPool 与一组 SingleTypeKVCacheManager（每种 attention 类型一个），共同维护请求→block 的映射关系。核心包括：
- BlockPool：统一逻辑块池，维护所有 KVCacheBlock 元数据的分配/回收/引用计数/prefix hash 映射，所有 attention 类型共享同一个实例
- SingleTypeKVCacheManager：每种 attention 类型对应一个实例，持有 BlockPool 引用，通过 req_to_blocks 维护该类型下请求到 block 的映射
- KVCacheCoordinator：掌管 BlockPool 和一组 SingleTypeKVCacheManager，协调跨类型的 prefix 匹配与 block 分配
- KVCacheManager：对外接口，Engine 通过它操作 KV cache，实际全部委托给 KVCacheCoordinator
1. 请求到 block 的映射
SingleTypeKVCacheManager 内部维护 req_to_blocks，记录每个请求当前占用的 KVCacheBlock 列表：
request_id → [KVCacheBlock, KVCacheBlock, ...]
这些 block 的 block_id 会随调度结果下发到 worker，写入对应 KV cache group 的 block table。模型执行时，通过"请求序列中的第 i 个 block 位置 → block table[i] → 物理 block_id"完成寻址。
2. Prefix Caching 的 hash 映射
BlockPool 内部维护 cached_block_hash_to_block，将已满 block 的 hash 映射到对应的 KVCacheBlock：
block_hash → KVCacheBlock
新请求的 prompt token 在 Request 构造时即被预计算为若干 block hash。调度阶段，系统按顺序逐一在这些 hash 中查找命中：如果某个 block hash 已存在于 cached_block_hash_to_block，则直接复用该 block并增加引用计数），无需重新计算该段 token 的 KV Cache；一旦出现 miss，后续 block 即便有 hash 也不再复用。
block hash 的计算有两个关键特点：
- 只缓存完整 block：只有长度达到 block_size 的完整 block 才会计算并加入 prefix cache。未填满的尾部 block 不参与缓存，因为它还可能继续追加 token，内容尚未稳定。
- hash 具有链式依赖：当前 block 的 hash 不只依赖本 block 内的 token ids，还依赖前一个 block 的 hash，也就是父 block hash。这样可以把“到当前 block 为止的完整前缀路径”编码进 hash 中。因此，即使两个 block 内部 token ids 相同，只要它们前面的上下文不同，父 hash 不同，最终 block hash 也不同，从而避免错误复用。
为一个请求计算 block hash 的核心逻辑在 kv_cache_utils.py 的 get_request_block_hasher() 中。它返回闭包 request_block_hasher(request)，它在 Request 构造时和每次新生成 token 后增量调用，为新产生的满 block 计算 hash 并追加到 request.block_hashes。计算时从上次已计算的 token 位置开始，以 block_size 为粒度逐块推进，同时将前序 block 的 hash 作为父 hash 传入，形成链式前缀 hash。
class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management 
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
        """
        # 每个块中token的数量
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # 请求到该请求所有block的映射
        self.req_to_blocks: defaultdict[str,
                                        list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        # 记录每个请求中被缓存的full块
        self.num_cached_block: dict[str, int] = {}

具体而言，以某个请求为例，当请求中的 token 数量增长到 block_size 的整数倍时，就会形成一个新的满 block。判断标准是纯计数：num_full_blocks = num_tokens // block_size，与推理是否完成无关。
block hash 的计算和缓存分为两步：
Step 1 — 计算 hash（Request 侧）：request_block_hasher() 在请求创建时以及每次追加新 token 后被调用，增量地检查是否有新的满 block 产生。每产生一个满 block，就以该 block 内的 token ID 序列和其父 block 的 hash 作为输入，计算链式 hash，追加到 request.block_hashes 列表中。
# kv_cache_utils.py:605-608
block_tokens = request.all_token_ids[start:end]
block_hash = hash_block_tokens(hash_fn, prev_block_hash, block_tokens, extra_keys)
Step 2 — 缓存 hash（BlockPool 侧）：由 cache_full_blocks() 执行。它在每次调度时 allocate_slots() 的末尾被调用。
cache_full_blocks 接收该请求的 block 列表和已缓存的 block 数量，计算出本次新满的 block 区间，然后从 request.block_hashes 中取出对应的哈希值，依次写入每个物理块的 block_hash 字段，并将该块插入 cached_block_hash_to_block 全局哈希映射。从此，其他请求可以通过相同的 block hash 命中该物理块，实现前缀复用。

Vllm BlockPool 内部维护两个核心数据结构：
BlockPool 内部维护两个核心数据结构：
1. self.free_block_queue（FreeKVCacheBlockQueue）：管理当前系统中所有空闲物理块的双向链表队列，支持 O(1) 弹出、追加和中间删除。队列按 LRU 驱逐顺序排列——最近最少使用的块在队头，分配时优先从队头弹出；归还时追加到队尾。
2. self.cached_block_hash_to_block（BlockHashToBlockMap）：块哈希到物理块 KVCacheBlock 的映射表，用于前缀缓存查找。当多个请求共享相同前缀时，通过哈希快速定位已有物理块。
字段
含义
block_id
唯一标识，取值范围 0 ~ num_gpu_blocks - 1
ref_cnt
引用计数，表示该块当前被多少个请求使用
block_hash
该块被填满并缓存后的哈希值，空块为 None
ref_cnt 的生命周期：
- 分配时：get_new_blocks() 从队头弹出 block，ref_cnt 加 1
- 前缀命中时：touch() 将已命中 block 从空闲链表摘除，ref_cnt 加 1
- 请求结束时：free_blocks() 将请求持有的 block 逐块 ref_cnt 减 1；一旦某块 ref_cnt 降为 0，即从请求的块列表中移除，归还到 free_block_queue 队尾
以图示为例：block_1 被请求 A 和请求 B 同时使用（共享相同前缀），因此 ref_cnt = 2。若请求 A 先结束，block1 的 ref_cnt 减为 1，仍由请求 B 持有；待请求 B 也结束后，ref_cnt 降为 0，block1 才归还空闲队列。
暂时无法在飞书文档外展示此内容
申请新的显存块
当确定某个请求需要新的显存块时，分配流程在 BlockPool.get_new_blocks() 中完成：
ret = self.free_block_queue.popleft_n(num_blocks)   # Step 1: 从队头弹出 n 个空闲块
for block in ret:
    self._maybe_evict_cached_block(block)             # Step 2: 若块曾被缓存，清除哈希映射
    assert block.ref_cnt == 0                         # Step 3: 断言块处于空闲状态
    block.ref_cnt += 1                                # Step 4: 标记为已占用
_maybe_evict_cached_block() 的驱逐逻辑并不会清理所有空闲块。只有当空闲块的 block.block_hash is not None 时，才说明该 block 仍保留着 prefix cache 的哈希身份，需要在重新分配前清理旧映射。
这类 block 通常满足以下过程：
1. 该 block 上一次使用时已被填满，即包含 block_size 个 token，因此系统为其计算了 block hash，并将其注册到 cached_block_hash_to_block 哈希表中。
2. 后续该 block 的 ref_cnt 降为 0，被归还到空闲队列。但为了保留 prefix cache 的复用机会，vLLM 没有立即删除它的哈希映射。
3. 当该 block 即将被重新分配给新内容时，旧的 KV Cache 内容会被覆盖，原来的 hash → block 映射不再有效，因此必须先清理。
清理过程包括两步：
1. 从哈希表中移除旧映射：
cached_block_hash_to_block.pop(block_hash, block.block_id)
2. 重置 block 自身状态：调用 block.reset_hash()，即将 block.block_hash 置为 None。
  如果取出的 block 从未被缓存过，即 block.block_hash is None，_maybe_evict_cached_block() 会直接返回 False，不做任何清理。
class BlockPool:
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.
        Note that we do not check block cache in this function.
        Args:
            num_blocks: The number of blocks to allocate.
        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                # 清除掉这个块之前的hash信息
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret
        
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its has
        metadata and evict it from the cache.
        Args:
            block: The block to evict.
        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False
        # 从全局缓存映射中弹出该 block 的 hash 条目
        blocks_by_id = self.cached_block_hash_to_block.get(block_hash)
        if blocks_by_id is None:
            # block_hash not found in cached_block_hash_to_block,
            # eviction is not needed
            return False
        # 清除 block 上的 hash 元数据
        block.reset_hash()
        blocks_by_id.pop(block.block_id, None)
释放显存块
相对的，有申请显存块就有释放显存块，如果你是vLLM的开发者当然需要分为以下的两步：
1. 当一个请求完成并需要释放其占用的块时，首先将该块的引用计数 ref_cnt 减 1。如果某个块的引用计数减到 0，说明它不再被任何请求使用，随后需要将这个块重新放回 free_block_queue 中，以供后续请求分配使用。
2. free_blocks 只操作 ref_cnt 和空闲队列，不改cached_block_hash_to_block。即使 ref_cnt=0，该块在哈希表中的条目仍然保留，给后续同前缀的请求一个复用的窗口期。只有当该块从空闲队列被 get_new_blocks 重新分配时，才会通过 _maybe_evict_cached_block 彻底清除
3. 反向释放：块按逆序进入空闲队列，队头为最早归还的块，实现 LRU 驱逐顺序——前缀 block（序列头部）相对于后缀 block 排名更靠后，更有机会保留在队列中等待复用，后缀 block 在显存紧张时优先被驱逐。
  简单来说，逆序释放的本质就是让前缀块在空闲队列中活得最久，因为只有前缀才可能被后续请求复用，后缀块没有保留价值。
class BlockPool:
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        # 如果当前块的使用次数减为0那么就把它放到self.free_block_queue中
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])
暂时无法在飞书文档外展示此内容
计算前缀
Prefix Caching 是一种优化大模型推理效率的技术。它的目标是：当多个请求中有相同的前缀时，避免重复计算这部分内容。比如请求1的输入prompt是：
I'am vllm and this is a very very interesting course, so, hello,what is you name?
请求2的prompt是：
I'am vllm and this is a very very interesting course, so, hello,what is you name?my name is fss.
当请求1比请求2先到达时，如果请求1已经执行并生成了对应的KV Cache，并且这些缓存块已写满（成为full block），那么对于请求2，如果它的前缀与请求1相同，系统只需复用这些已缓存的block即可。
但如果前缀存在差异，例如请求2的prompt在原有基础上增加了“以下是我的输入”这样的额外前缀，那么两个请求的前缀就不再完全相同。此时，系统无法直接复用已有的KV Cache，因为新增的前缀部分尚未计算。在这种情况下，只有从第一个不同位置之前的部分可以共享，而新增部分需要重新计算并生成对应的KV Cache。
三种前缀缓存复用情况
1. 情况一：前缀完全相同
请求 1: [A, B, C, D, E, ...]
请求 2: [A, B, C, D, E, ...]
请求 2 所有 block 的 hash 在 cached_block_hash_to_block 中全部命中。请求 2 无需重新计算 KV Cache，find_longest_cache_hit() 直接返回已缓存的 block 列表，随后 touch() 将每个 block 的 ref_cnt 加 1 即可复用。
2. 情况二：前缀多了一段新内容（例如新增 system prompt）
请求 1: [A, B, C, D, E, ...]                            ← prompt 不含前缀
请求 2: [以下是我的输入, A, B, C, D, E, ...]   ← 在头部新增内容
请求 2 的第一个 block 内容与请求 1 完全不同 → hash 匹配失败。从第二个 block 开始，虽然 token 与请求 1 完全一致，但由于其父 hash（第一块的 hash）已经不同，后续所有 hash 均改变，整条链全部无法命中。
3. 情况三：前缀相同、中间分叉
请求 1: [A, B, C, D, E, ...]
请求 2: [A, B, C, X, Y, ...]                            ← 前三块公共，之后分叉
前三个 block（A, B, C）的 hash 链完全一致，全部命中缓存。从第四个 block 开始 token 不同，hash 匹配失败。此时请求 2 复用前三个块的 KV Cache，剩余的 [X, Y, ...] 需要新分配 block 并重新计算。
前缀缓存能否命中，取决于从第一个 block 开始的哈希链是否逐块匹配。一旦链上某个 block 的 hash 不匹配，其后的所有 block 都无法复用——因为每个 block 的 hash 都包含前一个 block 的 hash，断一处则全链断裂。
以下是我的输入：I'am vllm and this is a very very interesting course, so, hello,what is you name?my name is fss.
我们需要借助的就是 cached_block_hash_to_block 这个映射表，根据 block hash 找到系统中已缓存的 block。查找流程是从当前请求的第一个 block（block 0）开始，逐个匹配，直到遇到第一个未命中的 block 或遍历完所有 block 为止。这是一种贪心匹配——尽可能匹配最长的前缀。具体实现思路如下：
1. 传入当前请求的 hash 列表 block_hashes: list[BlockHash]；
2. 从第一个 block 开始，逐块在 cached_block_hash_to_block 中查找：
  -  命中 → 将该物理 block 追加到 computed_blocks 列表；
  - 未命中 → 立即 break 退出循环。
3. 循环结束后返回 computed_blocks（即所有被命中的 block 列表，可能为空）。
为什么遇到未命中就直接 break？因为 block hash 是链式依赖的——当前 block 的 hash 包含父 block 的 hash。一旦某一块查不到，后续所有块的 hash 必然也无法匹配，继续遍历没有意义。
暂时无法在飞书文档外展示此内容
class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, (FullAttentionSpec, ChunkedLocalAttentionSpec)
        ), "FullAttentionManager can only be used for full attention " \
            "and chunked local attention groups"
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        max_num_blocks = max_length // kv_cache_spec.block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # 遍历一个请求中的所有块
            # 相当于从self.cached_block_hash_to_block列表中获取块（block）
            if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    # 将当前请求命中的block添加到computed中
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks
hash的计算方式
我们需要对满载的 full block 计算 hash。也就是说，当一个 block 的 block_size 个槽位全部放满 token（对应的 KV Cache 都已就绪）时，就可以计算它的 hash。
另外，由于需要求最长公共前缀，每个 block 的 hash 计算方式不仅仅与当前 block 存储的 token 有关，还与它之前的所有 block 有关——换句话说，hash 值以链式方式包含了前面所有 block 的信息。
BlockHash 的类型定义很简单：它就是 bytes，存储了当前 block 的 hash 值。有了这个 hash 值，在判断公共前缀时就可以通过比较 hash 是否相等来快速判定。
暂时无法在飞书文档外展示此内容
BlockHash 本质上是一个 bytes 类型别名，而非包含多个字段的结构体：
BlockHash = NewType("BlockHash", bytes)
它不存储 token_ids 原文，而是将以下三要素一起哈希后生成的不透明指纹(key)：
hash_function((parent_block_hash, token_ids_tuple, extra_keys))
- parent_block_hash：前一个 block 的 hash（首块为随机种子 NONE_HASH）
- token_ids_tuple：当前 block 内的 token ID 序列
- extra_keys：多模态或 LoRA 场景下的额外键（普通文本请求为 None）
Hash 在每次请求产生新的满 block 时计算（num_tokens 跨越 block_size 的整数倍边界），由 request_block_hasher() 增量生成并追加到 request.block_hashes。
class BlockHash(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None
    
def hash_block_tokens(
        hash_function: Callable,
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[tuple[Any, ...]] = None) -> BlockHash:
    # 如果没有父亲块，也就是它很有可能是某个请求的第一个block，则parant_block_hash等于NONE
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    # 计算当前block的hash时还需要放入父亲块的hash
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)
这里还需要注意的是，一个满的block才能去计算hash值，这点在hash计算函数中亦有限制。在 vLLM v1 中，计算的时候并非每次都从头遍历所有 block，而是从已计算 hash 的下一位置开始，也就是增量遍历该 request 新形成的满 block，这里的start就是指当前 request 中还没有被 hash 过的 token 起始位置。
start = len(request.block_hashes) * block_size   # 从上次结束处续接
然后逐块推进，每凑满一个 [start, start + block_size] 区间，就计算该块的 hash：
while True:
    block_tokens = request.all_token_ids[start:end]
    block_hash = hash_block_tokens(hash_fn, prev_block_hash, block_tokens, extra_keys)
    new_block_hashes.append(block_hash)
    prev_block_hash = block_hash    # 当前块 hash 成为下一个块的父 hash
三个输入，分别是prev_block_hash（前一个满块的 hash）、block_tokens（当前块的 token ID 序列）、extra_keys（多模态/LoRA 的额外标识）——合成当前块的 BlockHash。每次迭代结束后，当前 hash 自动成为下一轮的 prev_block_hash，形成链式向前推进。
def request_block_hasher(request: Request) -> list[BlockHash]:
    # 增量起点：从已 hash 过的下一个 token 开始
    # 已 hash 过的 token 数 = block_hashes 数量 × block_size
    start_token_idx = len(request.block_hashes) * block_size
    num_tokens = request.num_tokens
    # 若剩余 token 凑不满一个 block，不计算 hash
    if start_token_idx + block_size > num_tokens:
        return []
  
    # 父 hash：上一个 block 的 hash，首位为 None
    prev_block_hash_value = (
        request.block_hashes[-1] if request.block_hashes else None
    )
    new_block_hashes: list[BlockHash] = []
    while True:
        end_token_idx = start_token_idx + block_size
        if end_token_idx > num_tokens:
            break   # 只对满 block 计算 hash

        # 取当前 block 内的 token ID 序列
        block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
        # 三要素哈希：parent_hash + token_ids + extra_keys → 当前块的不透明指纹
        block_hash = hash_block_tokens(
            caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
        )
        new_block_hashes.append(block_hash)
        start_token_idx += block_size
        prev_block_hash_value = block_hash   # 链式推进：当前 hash 成为下一块的父 hash
    return new_block_hashes
总结一下以上的内容。
vLLM 在初始化时预先将可用于 KV Cache 的 GPU 显存划分为固定大小的物理块（Block）。每个 Block 存储 block_size 个 token 的 Key 和 Value 向量。例如 block_size=16 时，一个 Block 存放 16 个 token 的 K 和 V。Block 是 KV Cache 管理和分配的基本单位。
当一个新的推理请求到达时，vLLM 会尝试复用系统中已存在的 KV Cache。请求的 token 序列以 block_size 为粒度切分——例如长度 50 的 prompt 在 block_size=16 下被分为 4 个片段：[0~15]、[16~31]、[32~47]、[48~50]。
但是有两点需要注意：
1. 不满的 block 不参与前缀匹配：上述第 4 个片段只有 3 个 token 不满一个 block，不会计算 hash，自然也谈不上前缀命中。
2. block hash 是链式依赖的：计算当前 block 的 hash 时，输入不仅是当前 block 内的 token ID，还包含其前一个 block 的 hash。也就是说，每个 block 的 hash 隐式携带了它之前所有 block 的信息。
  vLLM 对一个请求从第一个 block 开始，逐块在全局哈希表 cached_block_hash_to_block 中查询其 hash。如果所有 KV cache group 中均存在命中，则该 block 对应的物理块被复用，继续匹配下一个 block；一旦某个 block 的 hash 查询失败，匹配立即终止——因为链式依赖保证了后续 block 的 hash 必然也不存在。
关于"命中"还需要注意：
即使 hash 仍在哈希表中，对应的物理块也可能已经释放回空闲队列（ref_cnt=0），此时会将其从空闲队列中摘除、恢复使用，只有当该块后续被 get_new_blocks 重新分配给其他内容时，哈希映射才会被清除
如果哈希值不存在，或者对应的 Block 已经被回收或者根本没存在过，因此对于一个请求而言，它的前缀匹配在此中断。注意点2我们举一个例子来说明，请求1：[block 1]--[block2]--[block4] 请求2：[block 1]--[block3]--[block4']，请大家思考下，block4和block4哪怕所有的token值都是对应的，那么它们的hash值相同吗？请在评论区给出自己的分析。
block的淘汰
在 BlockPool 分配新 block 时，若需淘汰旧 block，并非直接释放其对应的显存，而是仅从 cached_block_hash_to_block 哈希表中移除该 block 的 hash 索引。
例如，当一个新请求到来，需要复用先前由请求1释放的 block5 时，在将其分配给新请求前，必须先清除 block5 中残留的旧 hash 信息。但是，系统只清除哈希映射，不会也不需要在分配时立即计算新请求的 hash。
此时的 get_new_blocks 只是从空闲队列中弹出原始 block 并将其标记为已占用（ref_cnt = 1）。该 block 此时还空白是——旧数据已经被 hash 映射断开，新内容尚未写入。之后新请求的 token 经模型推理填入该 block，待其被填满时，cache_full_blocks() 才会从 request.block_hashes 读取预计算的 hash 写入该 block 的 block_hash 字段，并重新注册到 cached_block_hash_to_block。
具体流程如下：
1. 检查该 block 是否当前持有有效的 block hash；
2. 若存在且该 hash 仍存在于 cached_block_hash_to_block 中，则将其从该哈希表中删除；
3. 从 request.block_hashes 读取预计算好的新 hash，写入该 block 的 block_hash 字段；
4. 将新 hash 与该 block 的映射关系重新插入 cached_block_hash_to_block。
暂时无法在飞书文档外展示此内容
总结
1. vLLM 通过 PagedAttention 与 BlockPool 分块管理机制，革新了 LLM 推理中 KV Cache 的内存管理方式。传统方式通常为每个请求预分配一段连续显存，受限于序列长度动态变化和显存碎片，利用率低、吞吐受限。vLLM 将显存划分为固定大小的 Block，支持非连续物理显存分配，从根源上消除了外部碎片。
2. 跨请求的前缀复用通过 cached_block_hash_to_block 哈希索引实现：系统逐块匹配请求的 hash 链，命中后通过 touch() 将物理块 ref_cnt 加 1。
  即使该块已释放回空闲队列，只要哈希映射尚未被 _maybe_evict_cached_block() 清除，仍可命中复用。释放时采用逆序归还（后缀块先入队），确保前缀块在 LRU 队列中存活更久，最大化复用机会。
3. 每个 block 仅在 token 数达到 block_size 后才由 request_block_hasher() 计算 hash——计算在推理之前、基于 token ID 完成。hash 值包含前驱 block 的 hash，形成链式依赖，保证"断一处则全链断裂"，避免不同前缀路径下的相同 token 内容被错误匹配。
  分配新块时，get_new_blocks 先清除旧块的哈希映射，将其恢复为"空白"状态；新 hash 待该块再次被填满后由 cache_full_blocks() 写入并注册。
4. 该机制兼顾高性能、高并发与低显存浪费，为高吞吐、低延迟的大模型推理服务提供了关键支撑。
