Cuda Graph 是什么
原理
在深入实践之前，我们先来搞清楚：CUDA Graph 到底是什么？它为什么能提升性能？如果你曾经调试过 GPU 上的深度学习推理或高性能计算任务，可能注意到了一个现象：即使 GPU 利用率不高，CPU 却一直在忙碌地调度各种内核启动和内存拷贝操作。这些频繁的 API 调用不仅占用 CPU 资源，还会引入延迟——这就是所谓的CPU 开销瓶颈。
NVIDIA 引入 CUDA Graph 正是为了解决这个问题。它的核心思想很简单：把一连串重复执行的 GPU 操作录下来，然后打包成一个整体，以后直接一键运行，不再需要 CPU 一次次地下达指令。那么，它是如何做到这一点的呢？下面我们分三步来理解它的核心原理。
[图片]
1. 捕获操作序列
  传统的 CUDA 编程模型中，每一次 kernel 启动、每一次内存拷贝，都需要通过主机端（CPU）调用相应的运行时 API 来触发。这些调用虽然异步执行在流（stream）中，但仍然需要驱动程序进行解析、排队和调度——这个过程会带来一定的 CPU 开销，尤其是在高频率重复执行相同操作序列的场景下（例如推理任务中的逐层计算），这种开销就显得尤为冗余。
  CUDA Graph 的设计正是为了消除这类重复性的调度成本。它允许你在一次执行过程中，将一组已经定义好的 CUDA 操作录制下来，包括内核启动（Kernel Launch）、异步内存拷贝（cudaMemcpyAsync）、事件同步等，并记录它们之间的依赖关系。
  这个过程被称为图的捕获（Graph Capture）。你可以把它理解为：先让程序跑一遍示例流程，系统自动记下每一步做了什么、先后顺序如何，而无需每次都通过 CPU 重新调度各个操作。这减少了 CPU 与 GPU 之间的交互开销。
2. 构建成静态执行计划
  当这些操作被记录完成后，CUDA 运行时会将它们组织成一张有向无环图（DAG, Directed Acyclic Graph），这就是 CUDA Graph。图中不仅包含有哪些操作，更重要的是还记录了操作之间的依赖关系。例如，某个 kernel 必须等前面的数据拷贝完成后才能启动，而另外一些彼此独立的操作则可以并行执行。正因为依赖关系被显式表达出来，系统在后续执行时就能够更准确地控制执行顺序。
  更关键的是，一旦图被构建完成，后续重复执行时就不需要 CPU 再逐个调用 API、逐条提交操作、逐步建立依赖关系，而是只需要发起一次图执行请求。这样可以显著减少 CPU 的参与和提交开销。
  换句话说，CUDA Graph 并不是简单地把多个 CUDA 操作打包在一起，而是把操作内容和依赖关系一起固化为可重复执行的执行计划。这样做的主要收益，是降低频繁重复执行相同计算流程时的调度成本，并让运行时基于完整的依赖信息更高效地组织执行。
3. 优化并高效执行
  由于整个执行流程在图中是预先定义好的，CUDA 运行时在后续 replay 时可以基于这份固定的执行计划，减少逐操作提交带来的调度开销，并更高效地组织这些操作的执行。例如：
  - 减少重复的提交与依赖建立的开销：相比传统方式每次都由 CPU 逐个发射 kernel、逐次提交 memcpy 并显式维护依赖关系，CUDA Graph 可以把这整套流程作为一个整体重复执行，从而降低 CPU 端负担。
  - 更容易利用操作之间已经明确的依赖关系：如果图中某些内存传输和计算本身不存在依赖，并且硬件资源允许，它们就可以更自然地形成重叠执行。
  - 降低部分运行时管理成本：由于整条执行路径在 replay 时是稳定的，运行时不必在每次执行时重复处理大量调度与依赖管理逻辑，这对延迟敏感、重复模式明显的场景尤其有价值。
  更准确地说，CUDA Graph 的主要价值在于把一组固定的 CUDA 操作及其依赖关系提前固化下来，使后续执行时能够以更低的 CPU 提交成本完成同样的 GPU 工作。
基本流程
CUDA Graph 实现的基本流程：
1. 创建图：在捕获模式下执行一组 CUDA 操作，例如 kernel launch、异步内存拷贝和事件同步。运行时会将这些操作及其依赖关系记录下来，形成一张 CUDA Graph。也就是说，图中的节点并非一开始就全部存在，而是在捕获过程中随着操作的执行被逐步记录进去的。
2. 实例化图：捕获完成后，需要将这张 graph 转换成可执行的图实例。只有实例化之后，这张图才能被后续高效地重复执行。
3. 执行图：对于已经实例化的图，可以多次重复执行，而不需要重新捕获，也不需要由 CPU 逐个重新提交其中的 CUDA 操作。实际使用时，通常只需要更新输入数据，或者在允许的范围内更新图中相关节点的参数，然后再次 replay 这张图，即可完成同一执行流程的重复计算。
性能优化的关键点
1. 减少 kernel launch 带来的 CPU 开销：CUDA Graph 可以将多个 kernel 及其他 CUDA 操作捕获成一张图，后续执行时只需一次图启动，而无需由 CPU 逐个提交这些操作。因此，在重复执行相同计算流程的场景下，可以显著降低 kernel launch 和调度带来的 CPU 开销。
2. 降低运行时管理成本  
  - 减少动态提交开销：图中的操作顺序和依赖关系在 capture 后已经固定，后续 replay 时无需每次重新建立这套提交流程。  
  - 稳定数据传输与计算流程：当数据传输、kernel 执行和同步关系被提前固化后，运行时可以按既定依赖关系重复执行这条路径，从而减少频繁调度带来的额外开销。
3. 提高 GPU 利用率  
  - 依赖关系更清晰：图中已显式描述哪些操作必须串行、哪些可以并行，从而减少不必要的主机端干预。  
  - 更容易形成传输与计算的重叠：如果图中的操作本身不存在数据依赖，并且硬件资源允许，内存传输和计算就更容易重叠执行。  
  - 有助于减少 GPU 空转：对于重复模式明显的工作负载，CUDA Graph 有助于让内核及相关操作以更稳定的方式被发射和执行，从而降低部分调度间隙。
实践中的注意事项
Pytorch CUDA Graph 的实践，如果违反以下条件，可能会导致无声的数值错误或未定义行为：
1. capture 期间要保证执行环境干净：在一次 CUDA Graph capture 进行时，不要让进程内其他线程插入与当前 graph 无关的 CUDA 工作，否则会破坏录制过程，导致 capture 失败，或让 replay 出现错误结果。
2. capture 期间只能包含当前 graph 的设备侧操作：被录制的应是一段连续、稳定的 GPU 执行序列。额外的 kernel 启动、设备拷贝，或其他未被当前 graph 纳入的 CUDA 操作，都可能破坏图的正确性。
3. CPU 逻辑不会被捕获：CUDA Graph 只记录设备侧工作，不记录 Python 控制流、print、普通 CPU 计算等主机逻辑。这些代码只在 capture 当下执行一次，replay 时不会重新执行。
4. replay 依赖稳定的内存地址：CUDA Graph 记录的是执行所使用的内存地址，而不是 Python 层的张量对象。输入、输出和相关缓冲区通常需要在 capture 前预先分配好，replay 时复用同一批地址；如果要喂入新数据，应拷贝到这些固定缓冲区中。
5. graph 的执行结构必须保持静态：capture 和 replay 之间，张量的 shape、stride、内存布局以及执行路径都应保持一致。依赖 .item()、.cpu() 或数据值决定分支的动态控制流，以及动态 shape，都会破坏 graph 的可重放性。
6. 多流可以使用，但同步关系必须明确：如果 capture 涉及多个 CUDA stream，需要确保这些流上的工作都属于同一次合法 capture，并且流间依赖关系是清晰且稳定的，否则容易导致 capture 失败或 replay 错误。
简单 Demo
import torch
import time

assert torch.cuda.is_available(), "CUDA not available"

device = 'cuda'
N = 1024
num_warmup = 5
num_iter = 100
# 创建固定大小的输入（必须在捕获前分配好）
x = torch.randn(N, N, device=device)
y = torch.randn(N, N, device=device)
z = torch.empty(N, N, device=device)

# 创建了一个空的图
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    for _ in range(num_warmup):
        z = torch.mm(x, y) + torch.sin(x)
torch.cuda.synchronize()
# 捕获的操作，也就是要把我需要的op放入到空图当中。
with torch.cuda.graph(graph):
    z = torch.mm(x, y) + torch.sin(x)
torch.cuda.synchronize()

start = time.time()
# 我这个graph当中已经有了op，我现在只要进行重放就可以得到结果。
# 没有对输入进行赋值
# 后续只要对图进行重放，就可以对结果计算
for _ in range(num_iter):
    graph.replay()
torch.cuda.synchronize()
graph_time = time.time() - start
print(f"CUDA Graph time:  {graph_time:.4f} s")
1. graph = torch.cuda.CUDAGraph()
  - 创建一个空的 CUDA Graph 对象。
  - 后续会把一段固定的 GPU 执行序列 capture 到这个对象里，并通过 graph.replay() 反复重放。
2. stream = torch.cuda.Stream()：创建一个 自定义 CUDA 流（stream）。
3. 图捕获阶段：将括号内的所有 GPU 操作录制到 graph 中，形成一个后续可重复执行的静态执行图。
  - with torch.cuda.graph(graph): 代码块中的 GPU 操作会在 capture 时真实执行一遍，并同时记录到 graph 中。
  - 被记录的是一段可重复执行的设备侧工作序列，例如 kernel 启动和设备侧张量读写。
  - capture 完成后，后续调用 graph.replay() 时，CPU 不需要再逐个发起这些 op，而是直接重放整段已记录的执行序列，从而减少 CPU 调度开销。
with torch.cuda.graph(graph):
    z = torch.mm(x, y) + torch.sin(x)
torch.cuda.synchronize()
4. 重放 CUDA 图（relay）： 在推理时，将新的输入数据复制到输入占位符中，然后重放捕获的 CUDA 图以获得输出。修改了 x、y 这两个输入张量里的内容之后，replay 会读取它们当前地址上的新数据；但这些张量本身的形状、地址和整体执行结构必须保持稳定。
Cuda Graph 中的注意事项
在使用 CUDA Graph 进行大模型推理优化时，有三个核心注意事项需要特别关注。理解这些约束条件，对于正确使用 CUDA Graph 至关重要。
必须使用异步接口
在 CUDA C++ 中，应避免在 capture 路径里使用会引入主机同步的同步接口，例如同步版的 cudaMemcpy。如果确实需要做数据传输，应使用适合流执行的异步方式，并确保这些操作能够被 graph capture 正确记录。
更常见、也更稳妥的做法是：在 capture 之前就完成输入输出缓冲区和工作区的分配，让 graph replay 反复复用这些固定内存，而不是依赖 capture 期间临时分配内存。
在 PyTorch 中，同样应优先使用预先分配好的静态输入/输出张量，并让拷贝和计算在 CUDA stream 上以设备侧方式执行。像 copy_(non_blocking=True) 可以用于合适的异步拷贝场景，但重点不是这个参数本身，而是不要在 capture 期间插入 .item()、.cpu()、synchronize() 这类会把 CPU 拉回执行链路的同步操作。
核心原因：同步调用会破坏图捕获
以 cudaMemcpy 为例，它是一个同步函数。当调用：
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
为什么要在 CUDA Graph 捕获期间避免任何同步操作？
因为一旦调用 .item()、.cpu()、torch.cuda.synchronize() 这类让主机等待设备结果的 API，就会同时带来正确性和性能上的问题：
1. CPU 被强行阻塞
主机线程必须等待当前流上相关的 GPU 工作完成后才能继续执行。这会把原本异步推进的执行链路变成先等结果，再往下走，在推理服务这类高并发场景中会直接拖慢调度循环。
2. 打断图的捕获流程
CUDA Graph 要记录的是一段连续、可重放的设备侧执行序列。.item()、.cpu() 这类操作会把 CPU 显式拉回执行路径中，使这段序列不再是纯设备侧工作。这样会导致 capture 失败，或者让 replay 不再满足捕获时的执行假设。
3. 破坏流水线并行
原本可以与计算重叠推进的数据传输、kernel 提交和后续调度，一旦被同步点打断，就更容易退化为串行执行。这会降低 GPU 利用率，并增加端到端延迟，在多 batch 循环、持续解码或 pipeline parallel 场景下尤其明显。
正确的做法是什么？
- 所有数据预处理、输入准备以及需要的主机侧逻辑，都应在 capture 之前完成；capture/replay 使用预先分配好的静态输入缓冲区和输出缓冲区存放数据。
- capture 阶段应只包含可被 CUDA Graph 捕获的设备侧操作，例如 matmul、attention，以及设备侧张量读写；不要在这期间插入 .item()、.cpu()、synchronize() 这类需要主机参与的操作。
- 如果需要获取结果，例如返回 logits，应先让输出张量留在 GPU 上，等 graph replay 完成后再统一执行设备到主机的拷贝或同步；如果后续计算仍在 GPU 上继续进行，则通常不需要立刻同步。
实际示例：同步调用导致的捕获失败
下面是一个错误示例，在捕获过程中添加了同步的 D2H（Device to Host）操作，详见code/course10/cuda_graph_blocking.py，会直接报错。
(py312) root@3bccafb1bf3d:~/vllm_learn#  cd /root/vllm_learn ; /usr/bin/env /usr/local/miniconda3/envs/py312/bin/python /root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 40017 -- /root/vllm_learn/code/course9/cuda_graph_blocking.py 
Traceback (most recent call last):
  File "/root/vllm_learn/code/course9/cuda_graph_blocking.py", line 28, in <module>
    x_cuda.copy_(x_host)  
    ^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `cudaErrorStreamCaptureUnsupported' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/vllm_learn/code/course9/cuda_graph_blocking.py", line 27, in <module>
    with torch.cuda.graph(graph, stream=stream):
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/miniconda3/envs/py312/lib/python3.12/site-packages/torch/cuda/graphs.py", line 265, in __exit__
    self.cuda_graph.capture_end()
  File "/usr/local/miniconda3/envs/py312/lib/python3.12/site-packages/torch/cuda/graphs.py", line 128, in capture_end
    super().capture_end()
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
import torch

# 初始化数据
x_host = torch.randn(1024, 1024, device='cpu')
y_host = torch.randn(1024, 1024, device='cpu')
x_cuda = torch.randn(1024, 1024, device='cuda')
y_cuda = torch.randn(1024, 1024, device='cuda')
z = torch.zeros(1024, 1024, device='cuda')

# 创建图和流
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

# 捕获图（错误示例）try:
with torch.cuda.graph(graph, stream=stream):
    x_cuda.copy_(x_host)  # 同步的拷贝，也是非法的，需要改成non_blocking = true
    y_cuda.copy_(y_host)  # 同步的拷贝
    z_cuda = torch.mm(x_cuda, y_cuda) + torch.sin(x_cuda)
        
    # 在捕获过程中添加同步操作
    z_cpu = z_cuda.cpu()  # 这会触发同步 D2H 拷贝except RuntimeError as e:
print(f"捕获失败: {e}")
捕获阶段只允许异步 GPU 操作
捕获阶段只能包含可被 CUDA Graph 捕获的设备侧工作，例如 GPU kernel 启动以及设备侧数据读写。需要特别避免任何会让主机介入或触发 CPU-GPU 同步的操作，比如 .item()、.cpu()，以及显式调用 synchronize()。这类操作会打断 graph capture，导致 capture 失败，或像上面的例子那样让 replay 行为出错。
如果需要把结果传回主机，例如将 logits 返回给上层应用，应当先完成 graph replay，再统一执行设备到主机的拷贝。不要在图内部或在 replay 尚未结束时，提前把中间结果拉回 CPU。这样可以保证捕获路径保持纯设备侧执行，也更符合 vLLM 对 CUDA graph 的使用方式。
固定的张量地址
CUDA Graph 在捕获（capture）阶段记录的，其实是一系列对固定虚拟显存地址的操作。换句话说，它记下的不是对某个张量做运算，而是从地址 0x7fb1c0000000 读数据，写结果到 0x7fb1d0000000这样的底层指令。
举个例子吗，在捕获时：
- 输入张量 x 被分配在显存地址0x7fb1c0000000
- 输出张量 z 被分配在显存地址0x7fb1d0000000
那么在后续 replay 时，CUDA Graph 会继续从 capture 时记录下来的那块输入地址读取数据，并将结果写回 capture 时对应的输出地址。它不会重新分配这些缓冲区，也不会因为 Python 变量后来指向了别的张量而自动改用新的地址。
因此，正确的做法通常是：
- 在 capture 之前预先分配好输入、输出及相关工作缓冲区
- replay 时复用这些张量对应的固定地址
- 每次推理前只更新缓冲区里的内容，而不替换张量对象本身
常见的更新方式是原地写入，例如 fill_()，或者用 copy_() 将新数据拷贝到原有输入缓冲区中。

这种模式在 code/course9/cuda_graph_input.py 中有标准实现：先预分配好所有张量，capture 一次，之后每次推理前只更新数据内容，不改变张量本身。
从场景 2 可以看出，当 x 修改为别的值后，没有使用 in-place 操作，而是直接创建了一个新的张量，最终导致第二次重放没有得到正确的结果。也就是说，虽然 x 的值已经修改为 5，但结果 z 在重放后计算出的结果仍然是 4。
z = torch.mm(x, y) 需要改变输入 x、y 中的值，只能对 x 进行 in-place 操作，即每次推理前只更新数据内容，不改变张量本身。例如 x.fill_(3.f) 没有重新创建一个新的张量，而是直接修改原有张量中的值。因为 CUDA Graph 是从一个显存地址读取数据，而不是从某个张量对象中读取数据，所以一旦改变了地址，graph 就会读取旧地址。
因此，即使新张量的 shape 和 dtype 完全一样，它通常也会对应一块新的显存地址，这样虽然 Python 变量 x 指向了新张量，但 graph replay 仍然会读取 capture 时旧地址上的数据，结果就可能仍然是旧值，甚至在更复杂的场景下导致错误或崩溃。
import torch
import time

device = 'cuda'
N = 2

# 1. 初始状态：x 为全 1
x = torch.ones(N, N, device=device)
y = torch.ones(N, N, device=device)
z = torch.empty(N, N, device=device)

# 捕获图
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

for _ in range(3):
    temp = torch.mm(x, y)
torch.cuda.synchronize()  # 确保预热彻底完成

with torch.cuda.stream(stream):
    # 捕获：z = x * y (即 1 * 1) 
    with torch.cuda.graph(graph):
        z = torch.mm(x, y)
torch.cuda.synchronize()

print("--- 初始捕获完成 ---")
print(f"初始 x 地址: {x.data_ptr()}, 初始 z 结果:\n{z}")

print("\n--- 场景 1: 原地修改内容 (x.fill_(2)) ---")
x.fill_(2.0)  # 地址没变，内容变了
graph.replay()
print(f"x 地址: {x.data_ptr()}, z 结果 (预期应该是 4):\n{z}")

print("\n--- 场景 2: 改变变量地址 (x = torch.full...) ---")
x = torch.full((N, N), 5.0, device=device)
# 重新创建一个张量之后，graph其实还是从原来的地方显存指针地址进行数据读取。

graph.replay()

print(f"图重放后的 z 结果:")
print(z)
禁止动态形状
使用 CUDA Graph 时，有几个关键原则：
1. 提前分配好所有张量
在 capture 之前，把这条执行路径中会用到的输入、输出和中间缓冲区都预先分配好。capture/replay 期间不应再动态创建、扩容或替换这些张量对应的底层存储。
2. 只更新数据内容，不替换张量对象
capture 之后，应通过 .copy_()、.fill_()、.zero_()、.normal_() 这类 in-place 操作更新张量里的数值，而不要写成 x = x + 1、x = torch.randn(...) 这类会生成新张量并改变底层地址的写法。
3. 张量的底层内存地址和布局必须保持稳定
从 capture 到后续多次 replay，同一输入、输出和工作缓冲区通常都应复用 capture 时的那块内存。对应的 shape、stride、contiguous 状态和 dtype 也应保持与 capture 时一致。
为什么这么严格？因为 CUDA Graph 依赖“静态性”。
一旦完成 capture，后续 replay 使用的就是同一条固定的设备侧执行序列。这个序列默认假设参与计算的张量在 replay 时仍然具有与 capture 时一致的：
- shape
- stride / 内存布局
- dtype
- 底层内存地址
这些条件如果发生变化，replay 的前提就会被破坏，轻则结果错误，重则触发未定义行为。
但有一件事是可以变的：  
张量里的具体数值可以变。只要地址、shape 和布局不变，就可以在每次 replay 前把新数据写进原来的输入缓冲区，例如用 .copy_() 更新输入，或用 .zero_() 清空缓存。
如果需要支持不同 batch size 或不同序列长度怎么办？CUDA Graph 本身不擅长直接处理动态 shape。通常做法有两种：
1. 为不同 shape 或不同 bucket 分别 capture 多张 graph
例如分别为 batch=1、batch=2，或 seq_len=512、1024、2048 建立不同的 graph，运行时按输入规模选择合适的一张。
2. 使用 padding / mask，把较小输入映射到一个较大的固定 shape 上
这种方式本质上仍然是在复用一张固定 shape的 graph，只是通过 padding 让不同输入共享同一执行结构。它不是自动兼容，而是你在模型执行路径里显式设计出来的。
在大模型推理里，这也是 prefill 阶段更难使用 CUDA Graph 的原因之一：  不同请求的 prompt 长度差异很大，导致序列长度和相关中间张量形状经常变化，因此很难用少量固定 graph 覆盖全部情况。
# 为不同形状创建不同的图
shapes = [512, 1024, 2048]
graphs = {}

for size in shapes:
    # 为每个形状分配独立的张量
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    z = torch.empty(size, size, device=device)
    
    # 创建独立的图
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(5):
            z = torch.mm(x, y) + torch.sin(x)
    torch.cuda.synchronize()
    
    with torch.cuda.graph(graph):
        z = torch.mm(x, y) + torch.sin(x)
    torch.cuda.synchronize()
    
    # 存储图及其对应的张量
    graphs[size] = {
        'graph': graph,
        'x': x,
        'y': y,
        'z': z
    }

# 使用时根据形状选择对应的图
def run_with_shape(size):
    g = graphs[size]
    g['x'].normal_()  # 更新数据
    g['graph'].replay()  # 重放对应的图
    return g['z']
CudaGraph + Model Forward 实战
在实战开始之前，可以先回忆一下 CUDA Graph 的几个关键步骤：
1. 先做预热
在正式 capture 之前，通常先执行几轮相同 shape 的前向推理。这样可以让相关 kernel、缓存和运行时状态先稳定下来，避免把首次执行的额外开销带入 capture。
2. 预先准备固定的输入缓冲区
在 capture 前，需要先分配好输入占位符张量。它们的 shape、dtype 和设备位置要与实际 replay 时一致；后续推理时只更新这些缓冲区中的数据内容，不替换张量对象本身。
3. capture 一次固定的前向执行路径
用 torch.cuda.CUDAGraph 录制一次前向过程。capture 期间应只包含稳定的设备侧执行路径，避免动态控制流、动态 shape，以及 .item()、.cpu()、synchronize() 这类会把主机拉回执行链路的操作。
4. 推理时 replay graph
后续推理时，只需把新输入写入预分配的输入缓冲区，然后调用 graph.replay()。这样就不需要再由 Python 和 CPU 逐个提交前向里的 CUDA op，因此能明显降低 launch 开销。这种方式特别适合 shape 稳定的场景，例如 decode 阶段常见的固定 seq_len=1 和 batch bucket。
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from dataclasses import dataclass
import time

@dataclass
class ModelConfig:
    num_layers: int = 12
    embedding_dim: int = 768
    num_heads: int = 12
    vocab_size: int = 50257

class SimpleGPT2(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(SimpleGPT2, self).__init__()
        self.num_layers = model_config.num_layers
        self.embedding_dim = model_config.embedding_dim
        self.num_heads = model_config.num_heads
        self.vocab_size = model_config.vocab_size

        self.embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        h = self.embed_layer(x)
        for transformer_block in self.transformer_blocks:
            h = transformer_block(h)
        logits = self.lm_head(h)
        return logits

class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = None
        self.graph_output = None

    def capture(self, x):
        # 捕获 CUDA 图
        assert self.cuda_graph is None, "CUDA graph has already been captured."
        torch.cuda.synchronize()
        
        # 创建图的输入输出占位符
        # 创建占位符的目的就是方便我们更新输入和获取输出
        # 只要把数据用copy_的方式考入到graph_input当中就可以了。
        self.graph_input = x.clone().detach().cuda()
        self.graph_output = torch.empty_like(self.model(self.graph_input))

        # 开始捕获 CUDA 图
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.graph_output = self.model(self.graph_input)

        torch.cuda.synchronize()

    def forward(self, x):
        # 就是在往输入中拷贝数据
        # relay重放
        self.graph_input.copy_(x) # 用的是inplace_操作
        self.cuda_graph.replay()
        return self.graph_output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class ModelRunner:
    def __init__(self, model, seq_len=64):
        self.model = model
        self.seq_len = seq_len
        self.graph_runners = {}

    def capture_decode_graph(self):
        # 在 decode 阶段捕获 CUDA 图
        for batch in [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128]:  # 设置一些常用 batch size
            input = torch.randint(0, self.model.vocab_size, (batch, self.seq_len)).cuda()
            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(input)
            self.graph_runners[batch] = graph_runner
            # 因为cudagraph只支持静态形状，所以我们为常见的shape都创建了cudagraph

    def decode(self, x):
        batch_size = x.shape[0] # 只要根据bs的大小获取到对应的cudagraph就可以了
        if batch_size in self.graph_runners:
            model_executable = self.graph_runners[batch_size]
        else:
            print("Warning: CUDA graph not captured for this batch size, falling back to original model.")
            model_executable = self.model
        return model_executable(x)

# 主程序入口
if __name__ == "__main__":
    # 配置模型并构造
    config = ModelConfig()
    model = SimpleGPT2(config).cuda().eval()

    # 测试用例输入（先确定 seq_len，再进行捕获）
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = torch.tensor(tokenizer.encode("Hello, how are you?", add_special_tokens=True)).unsqueeze(0).cuda()
    seq_len = 1
    runner = ModelRunner(model, seq_len=seq_len)
    runner.capture_decode_graph()

    # 模拟 decode：通常每步 seq_len=1，这里可替换为 input_ids[:, :1]
    input_ids = input_ids[:, :1].expand(128, -1)

    # 推理时间对比
    # 不使用 CUDA 图推理时间
    start = time.time()
    output_no_graph = model(input_ids)
    end = time.time()
    print(f"不使用 CUDA 图推理时间: {end - start:.4f} 秒")

    # 使用 CUDA 图推理时间
    start = time.time()
    output_with_graph = runner.decode(input_ids)
    end = time.time()
    print(f"使用 CUDA 图推理时间: {end - start:.4f} 秒")

    # 检查输出是否匹配
    torch.testing.assert_close(output_no_graph, output_with_graph, rtol=1e-03, atol=1e-03)
1. 从代码可以看出，这里对 GPT 模型在 decode 阶段的一次固定前向执行路径进行了 CUDA Graph capture。
2. 在后续推理时，只需要把新的输入数据通过 copy_() 写入已捕获 graph 对应的输入缓冲区，再调用 replay()，就可以重放整段前向计算流程。
3. 这种方式避免了每次推理都由 Python / CPU 重新逐个提交 CUDA op，从而减少了 kernel launch 和 CPU 调度开销。
从这个示例的结果可以看到，在输出与原始 eager 执行保持高度一致的前提下，使用 CUDA Graph 明显降低了推理延迟。对于 shape 稳定、需要重复执行相同前向路径的场景，例如decode 阶段的连续生成，这种优化通常更容易带来可观收益。
不使用 CUDA 图推理时间: 0.0206 秒
使用 CUDA 图推理时间: 0.0004 秒
Cuda Graph小结
使用 CUDA Graph 时，需要事先准备好固定的 input buffer 和 output buffer。在 capture 到后续 replay 的过程中，这些 buffer 的底层地址必须保持不变。如果要用新的输入数据执行同一张 graph，通常不能直接替换输入张量对象，而是需要先把新数据拷贝到预分配好的 input buffer 中，然后再调用 graph replay。执行完成后，结果也会写回预先固定好的 output buffer。
这种机制在一定程度上简化了上层模型调用者的工作。调用者主要负责把输入数据写入约定好的输入缓冲区，并在需要时读取输出缓冲区中的结果；而这条执行路径内部使用到的中间缓冲区、kernel 执行顺序以及相关依赖关系，通常由 CUDA Graph capture/replay 机制和底层执行后端统一管理，上层代码不需要在每次推理时逐个干预。
vLLM 中的 Cuda Graph
在早期的 vLLM 设计中，CUDA Graphs 和 compile 相关的代码没有解耦，vLLM v1 对该机制进行了系统性重构，新的 CUDAGraph 设计原则可概括如下：
1. 区分统一解码批次与混合批次，并为其选择不同的图执行策略
对于纯 decode 且形状满足条件的 uniform decode batch，vLLM 优先采用执行效率更高的 FULL CUDAGraph；而对于 prefill 或 mixed prefill-decode batch，则优先使用 PIECEWISE CUDAGraph，必要时退回常规执行路径。
2. 强化模块化设计，尽量使 CUDAGraph 与编译逻辑保持正交  
  - 同一套模型执行逻辑可服务于不同的图捕获策略，避免为不同捕获方式维护多套割裂的执行分支。  
  - FULL CUDAGraph 可与 model compile 组合使用，也可独立使用；但 PIECEWISE CUDAGraph 依赖于 piecewise compilation，因此二者并非在所有模式下都完全独立。  
  - 因此，v1 的重点不在于将 compile 与 graph 强绑定，而是明确两者的职责边界与依赖关系。
3. 通过 CudagraphDispatcher 在运行时动态分派不同的 CUDAGraph
在每个调度周期，vLLM 根据当前 batch 的描述信息动态决定运行模式。该判定不仅考虑 batch 中是否为纯 decode，还会结合 token 数量、是否属于 uniform decode、LoRA 状态以及当前 backend 的支持能力，最终在 FULL、PIECEWISE 和 NONE 之间选择最合适的执行路径。
4. 集中化管理 CUDAGraph 的核心行为
vLLM v1 将运行时的 mode 判定、可捕获 batch key 的初始化、batch descriptor 到 graph 的映射与分派等关键逻辑收敛到更清晰的独立组件中，从而降低了与常规模型执行路径的耦合度，也提升了可维护性与可调试性。

CUDAGraph 的捕获与执行在 vLLM v1 中主要由 vLLM 自身的运行时组件负责发起，具体的，GpuModelRunner 负责预热和触发捕获（capture），CudagraphDispatcher 负责根据当前 batch 的描述信息选择运行模式，而 CUDAGraphWrapper 则负责根据对应的 key 执行捕获或回放（replay）。
这里的匹配条件并非仅限于 Batch Size。实际上，vLLM 使用了一组更完整的 batch 描述信息，例如 token 数量、请求数量、是否为 uniform decode、以及 LoRA 状态等。仅当新的推理任务与某已捕获图的描述符匹配时，系统才会直接回放该图；否则，将切换至其他 CUDAGraph 模式，或退回到常规执行路径。
因此，v1 的改进并非让上层完全不再关注图执行，而是将图管理逻辑从模型执行主路径中抽离出来，统一交由 dispatcher、wrapper处理。上层调用者仍需准备输入、维护运行时上下文，并确保相关缓冲区和 metadata 满足回放条件，但不再需要将 capture/replay 的细节放在普通执行逻辑中。
vLLM v1 的 CUDAGraph 机制相比之前版本更灵活，主要具备以下优点：
- 优点 1：模式更丰富。 vLLM v1 提供了 5 种 cudagraph_mode 取值：NONE、PIECEWISE、FULL、FULL_DECODE_ONLY 和 FULL_AND_PIECEWISE。用户可根据模型特性、后端能力及业务负载，选择更为合适的执行模式。
- 优点 2：支持显式控制。 用户可通过 cudagraph_capture_sizes 参数，在服务启动时显式指定需要预捕获的图尺寸，从而针对目标负载进行更主动、可观察的性能优化。
- 优点 3：模块化更强。 vLLM v1 将 CUDAGraph 的捕获、分派与执行逻辑集中管理，并在设计上尽量与编译逻辑保持正交。其中，FULL CUDAGraph 可较为独立地使用，而 PIECEWISE CUDAGraph 仍依赖于 piecewise compilation（分段编译），也就是把一次完整前向执行拆成多个较小的计算片段分别编译，而不是要求整个模型前向一次性编译成单一完整计算图。
CUDAGraph 捕获形式
我们上节课提到过，在 prefill 场景下，不同请求的 seq_len 或 query_len 往往不一致，因此 batch 的形状不像纯 decode 那样规整。也正因为如此，prefill 或 mixed prefill-decode batch 通常需要比纯 decode 更灵活的执行策略。CUDAGraphMode（共 5 种）可理解为 vLLM v1 中针对不同 CUDA Graph 执行策略的配置开关：
1. NONE：禁用 CUDAGraph。
模型不会使用 cudagraph capture/replay，但是否仍使用 compile 取决于其他编译配置。若设置 enforce_eager=True，则会进一步强制关闭更广泛的优化路径，包括 compile 和 cudagraph。
2. PIECEWISE（分段图）：
将模型中适合静态图捕获的部分拆分为多个片段，分别进入 CUDAGraph，而不适合进入同一个完整图的部分则留在图外执行。该模式兼容性更强，尤其适合 prefill 和混合 batch，但由于执行过程中需要在图内与图外路径间切换，通常会有额外开销。
3. FULL（完整图）：
尝试让所有 batch 都使用完整的 CUDAGraph。该模式对 batch 形状和 backend 支持要求更高，适合较为规整、稳定的工作负载，但在通用场景下往往不如混合策略灵活。
4. FULL_DECODE_ONLY（仅解码完整图）：
仅为纯 decode batch 使用完整的 CUDAGraph，而 prefill 或 mixed prefill-decode batch 则不使用 cudagraph，转而走常规执行路径。该模式适用于 decode 优先的场景。
5. FULL_AND_PIECEWISE（默认模式，双轨图策略）：
对纯 decode batch 使用效率更高的 FULL CUDAGraph；对 prefill 和 mixed prefill-decode batch 使用兼容性更强的 PIECEWISE CUDAGraph。这是 vLLM v1 的默认模式，通常更适合通用在线服务场景。
vLLM v1 现在的 CUDAGraph 设计架构图如下：
[图片]
1. PIECEWISE: 多个cuda graph，然后这里的两个cg并不是同一个cuda graph
2. FULL：整个模型被捕获在同一个cuda graph当中
对 Cuda Graph 的封装
在深入了解 Cudagraph 执行流程之前，先介绍下 vLLM 中 CUDA Graph 的核心组件及其作用。
核心类关系图：vLLM 的 CUDA Graph 实现围基于以下核心组件：
暂时无法在飞书文档外展示此内容
各组件作用：
1. GpuModelRunner：模型运行器，负责组织模型执行流程，包括准备输入、触发图捕获以及执行推理。
2. CudagraphDispatcher：运行时调度器，根据当前 batch 的描述信息（如 token 数、请求数、是否为 uniform decode、是否启用 LoRA 等），决定本轮应使用 FULL、PIECEWISE 还是 NONE。
3. CUDAGraphWrapper：CUDAGraph 的通用包装器，负责对被包装的 runnable 执行图捕获（capture）和图重放（replay）。
4. ForwardContext：前向执行期间的运行时上下文，用于保存当前的 cudagraph_runtime_mode、batch_descriptor 以及其他前向相关信息，通常由 GpuModelRunner 设置，并由后续执行路径读取。
5. BatchDescriptor：运行时 batch 描述符，用于描述当前 batch 的关键形状特征，并作为 cudagraph 分派与匹配的重要依据。
以离线推理为例，假设运行过程中可能出现 bs=2、bs=4、bs=8、bs=16 等不同规模的 batch。由于这些 batch 在 token 数、请求数以及形状特征上可能不同，系统通常不会只使用一张固定的 CUDA Graph，而是会为不同的 batch descriptor 维护多个候选图。运行时，CudagraphDispatcher 先判断当前 batch 更适合走哪种图模式，再由后续执行组件选择能够匹配该 batch 的具体 CUDA Graph 进行重放。
因此，从整体上看，vLLM v1 的 CUDAGraph 机制并非一个模型只对应一张图，而是同一条执行路径下，为多种不同输入形状准备多张候选图，并在运行时动态选择最合适的一张。
关键的数据结构
BatchDescriptor
批次描述符，用于描述当前 batch 的关键形状特征，并作为 CUDAGraph 分派与匹配时使用的重要 key。例如，在纯 decode 场景下，如果 bs = 2，且每个请求本轮都只处理 1 个 token，那么num_tokens = 2。其中，uniform 表示当前 batch 中各请求的 token 数是否一致。对于纯 decode 场景，如果每个请求本轮处理的 token 数都相同，则该 batch 可视为 uniform。
@dataclasses.dataclass(frozen=True)
class BatchDescriptor:
    """批次描述符，用作 CUDAGraph 分派和匹配的键"""
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
各字段含义如下：
- num_tokens：当前 batch 的 token 总数。
- num_reqs：当前 batch 的请求数。在 PIECEWISE 模式下，该字段可为 None，表示匹配时不强制要求请求数完全一致。
- uniform：当前 batch 中各请求的 token 数是否一致。
- has_lora：当前 batch 是否启用了 LoRA。
- num_active_loras：当前 batch 中活跃的 LoRA 适配器数量。
因此，BatchDescriptor用于描述当前 batch 的运行时形状特征，从而帮助系统匹配合适的 CUDAGraph。
CUDAGraphEntry
用于保存某个 BatchDescriptor 对应的已捕获 CUDA Graph，以及该图重放时需要复用的相关信息。其基本工作流程可理解为：
1. 根据当前 batch 的形状特征构造一个 BatchDescriptor。
2. 以该 BatchDescriptor 为 key，在图缓存中查找是否已存在可重放的 CUDA Graph。
3. 若存在，则直接 replay；若不存在，则先 capture，再将结果缓存起来。
例如，在纯 decode 场景下，如果 bs = 2，且每个请求本轮都只处理 1 个 token，那么系统可以构造出对应的 BatchDescriptor，并据此查找是否已有匹配的 cudagraph。
@dataclasses.dataclass
class CUDAGraphEntry:
    """CUDA Graph 条目，保存已捕获图及其相关运行时信息"""
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None
各字段含义如下：
- batch_descriptor：该条目对应的 batch 描述符。
- cudagraph：已捕获完成的 CUDA Graph；若为 None，表示该 key 尚未完成 capture。
- output：该图执行后的输出缓存。实现中通常以弱引用形式保存，以减少额外内存占用。
- input_addresses：用于记录输入张量地址，用于在 replay 时校验输入地址是否与 capture 时一致。
创建调度器
在 GpuModelRunner 初始化阶段，会先创建 CudagraphDispatcher。但此时调度器内部的 key 尚未真正初始化完成，因为它需要等待 attention backend 初始化结束后，才能确定当前后端实际支持哪些 cudagraph mode，以及应该生成哪些可分派的 BatchDescriptor。
# vllm/v1/worker/gpu_model_runner.py
class GpuModelRunner:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        # 创建 CUDA Graph 调度器。
        # 此时 dispatcher 已创建，但其内部 keys_initialized 仍为 False。
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
这里需要注意，CudagraphDispatcher 在构造时仅完成对象创建和基础配置读取，并不会立刻生成所有调度 key。真正的 key 初始化通常发生在后续 attention backend 能力解析完成之后。
键的初始化
在注意力后端初始化完成后，系统会调用 initialize_cudagraph_keys(...)，为当前配置和后端能力生成一组有效的 cudagraph 调度 key。这些预先生成的 BatchDescriptor 会按照运行时模式分别存放在 self.cudagraph_keys 中。当前主要分为两类：
- CUDAGraphMode.PIECEWISE
- CUDAGraphMode.FULL
这样设计的目的是让运行时分派不需要临时重新构造整套候选图集合，而是可以直接在这些预初始化的 key 中快速匹配合适的 cudagraph 描述符。后续在推理阶段，dispatcher 会根据当前 batch 的特征（例如 num_tokens、是否为 uniform decode、LoRA 状态以及允许使用的 mode）返回两项关键信息：
- runtime_mode
- batch_descriptor
其中，runtime_mode 用于描述本轮应采用的图类型，例如 FULL 或 PIECEWISE；batch_descriptor 用于描述当前 batch 的关键形状特征。后续执行组件再根据这两个信息去查找是否已有已捕获的 CUDA Graph 可供重放。
initialize_cudagraph_keys(...) 所做的事情可以理解为：
- 为 mixed/prefill 路径预生成一批 PIECEWISE key
- 为 uniform decode 路径预生成一批 FULL key
- 将这些 key 按 mode 分类保存，供运行时分派使用
class CudagraphDispatcher:
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int):
  
        print(f"initialize_cudagraph_keys: cudagraph_mode={cudagraph_mode}")
        print(f"compilation_config.cudagraph_capture_sizes={self.compilation_config.cudagraph_capture_sizes}")
        # 创建分段捕获，uniform_decode=false的键
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs in self.compilation_config.cudagraph_capture_sizes:
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(num_tokens=bs, uniform_decode=False))
        # 创建全图捕获，uniform_decode=true的键         
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL \
            and cudagraph_mode.separate_routine():
            max_num_tokens = uniform_decode_query_len * \
                self.vllm_config.scheduler_config.max_num_seqs
            cudagraph_capture_sizes_for_decode = [
                x for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs in cudagraph_capture_sizes_for_decode:
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    BatchDescriptor(num_tokens=bs, uniform_decode=True))
        self.keys_initialized = True
1. 首先，系统会为分段捕获（piecewise模式）中指定的每个 batch size，预先创建 非均匀解码模式（uniform_decode=False）的 CUDA Graph 键。如前例所示，这类键主要用于 Prefill 阶段 的推理，因为 Prefill 阶段的输入长度通常各不相同，难以满足 uniform 条件，也能处理部分带有 decode 但形状不规整的批次。
2. 接着，系统会为全图捕获（full graph模式）中指定的每个 batch size，创建 均匀解码模式（uniform_decode=True）的 CUDA Graph 键。这类键专用于 Decode 阶段，因为在该阶段每个请求每次仅生成一个 token，行为一致，符合 uniform 执行的前提。
当然这里只创建了键，而实际图捕获是惰性的，具体的CUDA Graph还没有捕获。这里所说的指定的 batch size，实际上是指系统会针对一系列常见的 token 数量（例如 1、2、4、8、16……）提前生成对应的 CUDA Graph 键。在后续运行过程中，当调度器遇到某个特定 token 数量的批次时：
- 若该数量对应的 CUDA Graph 尚未捕获，则在首次执行时进行图捕获（capture），并将结果缓存；
- 之后再次遇到相同 token 数量的批次时，即可直接重放（replay）已捕获的 CUDA Graph，从而避免重复的 kernel 启动开销，显著提升推理效率。
后续运行时，调度器会根据当前批次的实际形状将其 dispatch 到合适的 CUDA Graph 键；如果实际规模落在可复用范围内，vLLM 会优先复用已经为该描述符捕获好的图。
查找匹配的键
[图片]
为了实现系统中的高效索引，每个 CUDA Graph 对应一个 BatchDescriptor，其关键字段如下所示。
class BatchDescriptor:
      num_tokens: int                                                                                                            
      num_reqs: int | None = None                                                                                                
      uniform: bool = False                                                                                                      
                            
其中：                                                                                                                         
- num_tokens 表示当前批次参与执行的总 token 数（padding 后）
- num_reqs 表示批次中的 request 数量
- uniform 为 True 表示批次中所有 request 具有相同的 token 数  
如上图所示，CudagraphDispatcher 会先根据当前配置以及注意力后端的支持情况，提前准备一组可用的 dispatch key。等到实际执行时，再由 dispatch() 方法根据当前批次的特征，查找匹配的运行时模式和对应的 BatchDescriptor。
从后面的代码可以看出，调度时的优先级通常是：FULL > PIECEWISE > NONE。也就是说，如果当前批次能够匹配 FULL 模式，则优先使用 FULL；否则再尝试 PIECEWISE；如果都无法匹配，就回退到 NONE，即不使用 CUDA Graph，而是按普通 eager 方式执行。由于 graph mode 下调试不太方便，我们可以先在下面的代码中添加一处打印提示。随后，再使用 curl 向推理服务发送一条请求，观察实际命中的执行路径。
curl -s --noproxy '*' http://127.0.0.1:13333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' | jq -r '.choices[0].message.content'

随后，终端将输出如下内容：其中的 16 对应 Prefill 阶段，因为此时请求数量恰好为 1，且该请求包含 16 个 token。而当 num_tokens 等于 1 时，则对应 Decode 阶段。关于 uniform_decode 的具体含义，我们将在下一小节中详细说明。
batch descriptor: BatchDescriptor(num_tokens=16, uniform_decode=False)
batch descriptor: BatchDescriptor(num_tokens=1, uniform_decode=True)
batch descriptor: BatchDescriptor(num_tokens=1, uniform_decode=True)
batch descriptor: BatchDescriptor(num_tokens=1, uniform_decode=True)
其中，BatchDescriptor(num_tokens=16, uniform=False) 对应的是本次请求的 Prefill 阶段。这里的 16 表示该批次本轮执行的总 token 数为 16；在这个例子中，只有 1 条请求，且输入长度恰好为 16，因此总 token 数同样是 16。
后面连续出现的 BatchDescriptor(num_tokens=1, uniform=True) 对应的是 Decode 阶段。在这个例子中，每一轮 decode 只处理 1 个 token，并且批次满足 uniform 条件（所有请求 token 数相同），因此会命中这样的描述符。
从调度角度看，可以把系统中的 CUDA Graph key 近似理解成一张按 [runtime mode][batch descriptor] 组织的索引表，例如 [FULL, batch_desc] 或 [PIECEWISE, batch_desc]。当运行时收到一个新批次，dispatcher 就会按优先级查找可复用的图。
# vllm/v1/cudagraph_dispatcher.py
def dispatch(
    self, batch_descriptor: BatchDescriptor
) -> tuple[CUDAGraphMode, Optional[BatchDescriptor]]:
    """根据批处理描述符返回合适的运行时模式和图键"""
    print('batch_descriptor:{}'.format(batch_descriptor))
    if not self.keys_initialized:
        return CUDAGraphMode.NONE, None
    
    # 优先级1: 检查 FULL 模式（精确匹配）
    if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
        return CUDAGraphMode.FULL, batch_descriptor
    
    # 优先级2: 检查非均匀版本的 FULL 模式（均匀批次可匹配非均匀图）
    non_uniform_key = batch_descriptor.non_uniform 
    # 这边尝试一下，有没有一个图能够不要求所有的seq_len都等于1
    if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
        return CUDAGraphMode.FULL, non_uniform_key
    
    # 优先级3: 检查 PIECEWISE 模式（更通用）
    if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
        return CUDAGraphMode.PIECEWISE, non_uniform_key
    
    # 没有匹配的图，返回 NONE
    return CUDAGraphMode.NONE, None
一旦找到匹配项，该机制会通过前向传播上下文（forward context），将选定的运行时模式和最终的 BatchDescriptor 传递给 CUDAGraphWrapper 实例，以选择并激活对应的 CUDA Graph。
调度逻辑可以概括为两步：
- 先尝试 FULL 精确匹配：用包含完整形状信息（num_tokens、num_reqs、uniform 等）的 BatchDescriptor 在 FULL 图集合中精确查找。FULL 图是按固定 batch 尺寸录制的，要求每个维度都匹配。
- FULL 不命中则放松条件尝试 PIECEWISE：将 num_reqs 置为 None、uniform 置为 False，用宽松后的描述符在 PIECEWISE 图集合中查找。PIECEWISE 不要求按 request 数量 padding，命中门槛更低。
如果两者都未命中，回退到 NONE，即不走 CUDA Graph，转为普通 eager 执行。
调度策略可以总结为：
- FULL 优先：优先尝试性能最高的 FULL 模式
- PIECEWISE 兜底：FULL 不满足时降级到 PIECEWISE
- 逐级回退：两种图都不命中则退回 NONE（eager 执行）
键的组成
在上述流程中，传递给 dispatch 方法的 BatchDescriptor 来源于一个名为 preprocess 的预处理阶段，在流程图中已有清晰体现。具体而言，它由 GPUModelRunner._preprocess 方法生成，用于确定当前批次调度所需处理的 token 数量，并判断该批次是否满足 uniform 条件。
该判断主要基于以下两个条件：
1. num_scheduled_tokens == self.input_batch.num_reqs * max_query_len此条件用于检查所有请求的 query 长度是否完全一致。也就是在验证总 token 数是否恰好等于「每请求 token 数 × 请求数」，即所有请求的 token 数量完全一致。 
2. max_query_len == self.uniform_decode_query_len其中 self.uniform_decode_query_len 通常预设为 1。该条件用于验证当前是否处于标准的单步解码（decode）场景——即每个请求仅需生成一个新 token。
当上述两个条件同时满足时，系统会将 CUDA Graph 键中的 uniform 字段设为 True。这正是我们在 Decode 阶段将 uniform 设为 True 的根本原因：因为在该阶段，每个请求每次固定只处理一个 token，行为高度一致且可预测，从而满足 uniform 执行的前提。
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    (
        num_scheduled_tokens,
        num_input_tokens,
        num_tokens_across_dp,
        input_ids,
        inputs_embeds,
        positions,
        intermediate_tensors,
        model_kwargs,
    ) = self._preprocess(scheduler_output, intermediate_tensors,
                         ubatch_slices, num_tokens_after_padding)
    
    # 判断是否是uniform的
    uniform_decode = (max_query_len
                      == self.uniform_decode_query_len) and (
        num_scheduled_tokens
        == self.input_batch.num_reqs * max_query_len)
    # 创建批处理描述符
    batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                       uniform_decode=uniform_decode)
    # 调度到合适的 CUDA Graph 模式
    cudagraph_runtime_mode, batch_descriptor = \
    self.cudagraph_dispatcher.dispatch(batch_descriptor)
上下文携带
那么在找到对应的 key，也就是 BatchDescriptor 之后，接下来需要在推理过程中以执行上下文的方式将其携带传递。具体而言，该描述符会被封装进前向传播上下文（forward context）中，由 set_forward_context 进行设置，并随推理流程贯穿整个模型执行过程。
后续的模型执行阶段可以通过调用 get_forward_context() 来获取当前线程局部存储的上下文信息，从而取出其中的 BatchDescriptor。特别地，在模型执行过程中，有一个关键组件 —— CudaWrapper，它负责管理 CUDA Graph 的捕获与重放。该组件会在每次前向执行时从上下文中提取 batch_descriptor。
- 如果发现该 BatchDescriptor 对应的键已存在且其 CUDA Graph 实例已完成注册，则直接尝试复用该图实例；
- 若对应键未命中缓存（即尚未注册或图为空），则触发 CUDA Graph 的捕获流程，执行一次完整的前向计算以录制计算图，并将此次的执行配置与生成的 CUDA Graph 实例以键值对形式注册到内部的调度表中，供后续具有相同输入模式的批次高效复用
BatchDescriptor 作为 key 写入 forward context 后，CUDAGraphWrapper 在每次执行时以它为索引查找已录制的图——有则回放、无则捕获并缓存，实现相同形状批次的图复用。
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    # 接上
    cudagraph_runtime_mode, batch_descriptor = \
        self.cudagraph_dispatcher.dispatch(batch_descriptor)
    # 设置前向传播上下文
    with set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,  # 调度器返回的模式
        batch_descriptor=batch_descriptor,  # 调度器返回的键
        ubatch_slices=ubatch_slices,
    ):
    
# vllm/compilation/cuda_graph.py
def __call__(self, *args, **kwargs):
    """CUDA Graph 包装器的调用方法"""
    # 从线程局部上下文获取运行时模式和批处理描述符
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
对图的捕获和重放
从上文中我们可以得知，CUDAGraphWrapper 是 CUDA Graph 的核心包装器类，承担着在运行时动态管理图捕获与重放的关键职责。其核心机制是：在首次执行时记录（capture）模型前向传播过程中的 CUDA 操作序列，构建成一个高效的 CUDA Graph；在后续满足相同执行条件的调用中，则直接重放（replay）该图，从而避免内核启动开销和 CPU 端调度瓶颈，显著提升推理吞吐与响应延迟。
更重要的是，CUDAGraphWrapper 并非无条件启用图执行，而是根据当前上下文中的运行时模式（cudagraph_runtime_mode）进行判断，如上文所述，通常分为三种类型：
- FULL：对整个模型前向过程进行完整图捕获；
- PIECEWISE：按网络模块分段捕获多个子图，适用于长序列或内存受限场景；
- NONE：禁用 CUDA Graph，退化为普通 eager 执行；
CUDAGraphWrapper 会结合 get_forward_context() 获取的 batch_descriptor 以及当前的运行模式，判断是否命中已缓存的 CUDA Graph。已经捕获过的图实例就保存在 wrapper 自身的字典concrete_cudagraph_entries 中（以 batch_descriptor 为键）。
1. 若命中，则直接重放该图以加速执行；
2. 若未命中，则触发新图的捕获流程，在完成录制后，将生成的 CUDA Graph 实例以 batch_descriptor 作为键存入全局图池中，从而实现一次捕获、多次复用的高效推理机制。
# vllm/compilation/cuda_graph.py
def __call__(self, *args, **kwargs):
    """CUDA Graph 包装器的调用方法"""
    # 从线程局部上下文获取运行时模式和批处理描述符
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
    
    # 如果运行时模式是 NONE 或与包装器模式不匹配，直接调用底层函数
    if (
        cudagraph_runtime_mode == CUDAGraphMode.NONE
        or cudagraph_runtime_mode != self.runtime_mode
    ):
        return self.runnable(*args, **kwargs)
    
    # 查找或创建对应的图条目
    if batch_descriptor not in self.concrete_cudagraph_entries:
        self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
            batch_descriptor=batch_descriptor
        )
    
    entry = self.concrete_cudagraph_entries[batch_descriptor]
先，CUDAGraphWrapper 会根据当前运行时模式（cudagraph_runtime_mode）以及从执行上下文中获取的 batch_descriptor，在内部维护的 CUDA Graph 缓存表中查找是否存在对应的图实例。若命中成功，则直接重放该图，实现高效推理。
如果运行模式为 NONE，即未启用 CUDA Graph 的场景（例如性能分析阶段（profile run）、预热阶段（warmup run）或显式禁用 CUDA Graph 时），包装器会跳过所有图相关的处理逻辑，直接以 PyTorch eager 模式执行模型前向计算。
当运行模式允许使用 CUDA Graph（如 FULL 或 PIECEWISE），但缓存中尚未存在对应 batch_descriptor 的图实例时，系统将触发一次新的图捕获流程。该过程与典型的捕获方式一致：在 torch.cuda.graph 上下文中完整执行一次模型前向传播，期间 GPU 上的所有异步操作将被记录，并构建成一个静态的、可重放的 CUDA 图结构。
整个捕获流程可细分为以下几个关键步骤：
1. 记录输入张量地址：在捕获前记录各输入张量的内存地址（data_ptr()）。由于 CUDA Graph 在 replay 时要求输入必须位于与 capture 时相同的内存位置，该地址列表将用于后续的一致性校验（调试模式下启用）；
2. 创建 CUDAGraph 结构：通过 torch.cuda.CUDAGraph() 构造一个新的图容器，用于承接即将记录的操作序列；
3. 执行模型推理以捕获图：在 torch.cuda.graph(cudagraph, pool=...) 上下文中运行完整的模型前向传播，此时所有 CUDA 内核启动等操作均被记录至图中，而非立即提交到流上；
4. 保存图实例至缓存表：捕获完成后，将 CUDAGraph 实例与对应输出写入 entry.cudagraph，该 entry 以 batch_descriptor 为键存放于 wrapper 自身的 concrete_cudagraph_entries 字典中，供后续相同批次特征的请求复用。
这一机制实现了按需捕获、按键复用的高效策略，既避免了重复开销，又保证了对多样化输入批次的灵活支持，是 vLLM 在高吞吐场景下实现低延迟推理的核心优化之一。
def __call__(self, *args, **kwargs):
    if entry.cudagraph is None:
        # Step 1: 记录输入张量的数据指针
        input_ptrs = [
            x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        ]
        entry.input_addresses = input_ptrs
    
        # Step 2: 创建 CUDA Graph
        cudagraph = torch.cuda.CUDAGraph()
    
        # Step 3: 捕获计算图（使用上下文管理器确保资源正确释放）
        with ExitStack() as stack:
            # 进入 CUDA Graph 编译上下文
            with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                # 手动执行前向推理并捕获操作
                output = self.runnable(*args, **kwargs)
    
        # Step 4: 将最终输出和图存入 entry
        entry.output = output
        entry.cudagraph = cudagraph

 到下一次调用时，只要运行时模式允许且 batch_descriptor 匹配，系统便能在缓存表中快速查找到已捕获的图实例，随后直接调用 .replay() 重放整个前向计算过程，无需再次触发内核启动或重复执行模型函数。
def __call__(self, *args, **kwargs):
    ...
    ...
    entry.cudagraph.replay()
    return entry.output
总的来说，这里的执行流程与我们在第9课-cuda graph中 PyTorch CUDAGraph Demo 实战一节所介绍的基本模式高度一致：先进行一次预热并捕获计算图，后续通过输入拷贝和图重放来实现高效推理。
唯一的区别在于，当前实现针对大模型推理场景进行了大量工程封装与优化——不仅对模型输入（如键、值缓存等）的管理和传递做了精细化设计，还为不同序列长度和批处理模式（如 uniform 与 non-uniform 输入）提供了灵活的支持接口。
完整流程总结
总的来说，以 FULL 模式为例，从 model.forward 到 CUDAGraphWrapper.__call__ 的路径如下：
1. 在初始化阶段（has_full_cudagraphs() 为 True 时），真实的 nn.Module 会被 CUDAGraphWrapper 包裹并赋值给 self.model，后续所有对 self.model(...) 的调用实际上都会进入 CUDAGraphWrapper.__call__；
2. 在 execute_model 中，通过 CudagraphDispatcher.dispatch(num_tokens, has_lora, uniform_decode, ...) 同时获得 cudagraph_runtime_mode 和由 dispatcher 生成的 batch_descriptor；
3. 随后以 set_forward_context(..., cudagraph_runtime_mode=..., batch_descriptor=...) 将调度结果写入线程局部的 ForwardContext，从而避免在层层函数参数中显式传递；
4. 进入 CUDAGraphWrapper.__call__ 后，第一步便调用 get_forward_context() 取出刚才写入的 cudagraph_runtime_mode 与 batch_descriptor，据此决定走 capture、replay 还是 eager 路径。
主动捕获阶段：批量预热和录制
阶段目标：在服务启动时（capture_model 方法），主动批量捕获常用的批处理大小对应的 CUDA Graph，避免首次请求时的捕获延迟。
# vllm/v1/worker/gpu_model_runner.py
def capture_model(self) -> int:
    """批量捕获 CUDA Graph 的入口方法"""
    if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0
    
    # 启用 CUDA Graph 捕获
    set_cudagraph_capturing_enabled(True)
    
    # 冻结 GC 以优化捕获性能
    with freeze_gc(), graph_capture(device=self.device):
        cudagraph_mode = self.compilation_config.cudagraph_mode
        
        # 1. 捕获混合批次（prefill + decode）
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            compilation_cases = list(reversed(self.cudagraph_batch_sizes))
            self._capture_cudagraphs(
                compilation_cases,
                cudagraph_runtime_mode=cudagraph_mode.mixed_mode(),
                uniform_decode=False
            )
        
        # 2. 捕获纯解码批次（如果配置了单独路由）
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
           cudagraph_mode.separate_routine():
            max_num_tokens = self.scheduler_config.max_num_seqs * \
                self.uniform_decode_query_len
            decode_sizes = [
                x for x in self.cudagraph_batch_sizes
                if x <= max_num_tokens and x >= self.uniform_decode_query_len
            ]
            compilation_cases_decode = list(reversed(decode_sizes))
            self._capture_cudagraphs(
                compilation_cases_decode,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                uniform_decode=True
            )
    
    # 禁用捕获（防止意外捕获）
    set_cudagraph_capturing_enabled(False)
捕获逻辑：
# vllm/v1/worker/gpu_model_runner.py
def _capture_cudagraphs(
    self, compilation_cases: list[int],
    cudagraph_runtime_mode: CUDAGraphMode,
    uniform_decode: bool
):
    """为每个编译用例捕获 CUDA Graph"""
    
    for num_tokens in compilation_cases:
        # 预热运行（多次，不捕获图）
        for _ in range(self.compilation_config.cudagraph_num_of_warmups):
            self._dummy_run(
                num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,  # 预热时不捕获
                uniform_decode=uniform_decode
            )
        
        # 实际捕获运行（触发 CUDAGraphWrapper 进行图捕获）
        self._dummy_run(
            num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,  # 指定运行时模式
            uniform_decode=uniform_decode
        )
关键点：
1. 预热阶段：不捕获图，仅用于初始化 CUDA 内核和稳定内存分配
2. 捕获阶段：使用指定的运行时模式，触发CUDAGraphWrapper进行图捕获
3. 捕获顺序：按批处理大小从大到小捕获，确保小批次可以复用大批次分配的内存池
对输入的填充
vLLM 会在启动时针对一组预设的批处理大小（如 4、8、16、32）提前录制 CUDA Graph。推理阶段，若实际 batch size 恰好命中某个已录制尺寸，则直接取出对应的图进行 replay。
但实际进入的请求数量是不固定的，随时可能是 5、7、9 这类未录制的值。由于 CUDA Graph 在录制时张量形状和内存地址已固定，无法直接用于不同形状的输入。对此，vLLM 的解法是向上 padding：将实际 batch size 对齐到最近的已录制尺寸（如 5 → 8、9 → 16），多出的位置填充占位数据，再复用已有的图。
为此，系统在初始化时预先构建一张映射表，将 [0, max_size] 范围内的每个整数都对应到它应该使用的录制尺寸，推理时直接查表即可，无需实时计算。之所以不为每个可能的 batch size 都单独录制一张图，是因为图的数量直接影响显存占用和管理开销——用有限的几档录制尺寸加 padding 对齐，是性能与资源之间合理的工程取舍。
class CompilationConfig:
    def post_init_cudagraph_sizes(self) -> None:
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_cudagraph_capture_size + 1)
        ]
        for end, start in zip( 
            self.cudagraph_capture_sizes + [self.max_cudagraph_capture_size + 1],
            [0] + self.cudagraph_capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
完整流程总结-图示
下面通过时序图总结 CUDA Graph 从初始化到执行的全流程：
暂时无法在飞书文档外展示此内容
端到端性能实测对比：Eager vs CUDAGraph
以下使用 vLLM 官方 vllm bench latency 基准工具，在同一模型、同一硬件上，对比 Eager 模式与 CUDAGraph 模式的性能差异。
测试环境
暂时无法在飞书文档外展示此内容
测试命令
python bench_cudagraph.py

###############实际的模型推理命令对比如下####################
# Eager 模式（禁用 CUDAGraph 和 torch.compile）
# vllm bench latency --model Qwen/Qwen2.5-1.5B \
#     --input-len 128 --output-len 128 --batch-size 1 \
#     --enforce-eager --gpu-memory-utilization 0.85

# # CUDAGraph 模式
# vllm bench latency --model Qwen/Qwen2.5-1.5B \
#     --input-len 128 --output-len 128 --batch-size 1 \
#     --gpu-memory-utilization 0.85
完整 python 测试代码如下所示:
"""
Performance Benchmark: Eager vs CUDAGraph/Compile
"""
import subprocess
import sys
import json
import os

MODEL = "Qwen/Qwen2.5-1.5B"
NUM_WARMUP = 3
NUM_ITERS = 10

SCENARIOS = [
    {"label": "small_batch",  "input_len": 128, "output_len": 128, "batch_size": 1},
    {"label": "medium_batch", "input_len": 256, "output_len": 128, "batch_size": 16},
    {"label": "large_batch",  "input_len": 256, "output_len": 128, "batch_size": 64},
    {"label": "high_load",    "input_len": 512, "output_len": 256, "batch_size": 32},
]

MODES = [
    {"label": "eager",       "extra_args": ["--enforce-eager"]},
    {"label": "cudagraph",   "extra_args": []},
]

def run_bench(scenario, mode):
    out_json = f"/tmp/bench_{scenario['label']}_{mode['label']}.json"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL,
        "--input-len", str(scenario["input_len"]),
        "--output-len", str(scenario["output_len"]),
        "--batch-size", str(scenario["batch_size"]),
        "--num-iters-warmup", str(NUM_WARMUP),
        "--num-iters", str(NUM_ITERS),
        "--gpu-memory-utilization", "0.85",
        "--output-json", out_json,
        "--disable-detokenize",
    ] + mode["extra_args"]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    desc = f"{scenario['label']}_{mode['label']}"
    print(f"\n{'='*60}")
    print(f"  Running: {desc}")
    print(f"  batch={scenario['batch_size']}, input={scenario['input_len']}, "
          f"output={scenario['output_len']}, mode={mode['label']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-10:]:
            print(f"  stderr: {line}")
        return None

    if os.path.exists(out_json):
        with open(out_json) as f:
            data = json.load(f)
        avg = data["avg_latency"]
        p50 = data["percentiles"]["50"]
        p99 = data["percentiles"]["99"]
        total_tokens = scenario["batch_size"] * scenario["output_len"]
        throughput = total_tokens / avg
        print(f"  Avg latency: {avg:.4f}s  P50: {p50:.4f}s  P99: {p99:.4f}s")
        print(f"  Throughput: {throughput:.1f} tokens/s")
        return {
            "avg_latency": avg, "p50": p50, "p99": p99,
            "throughput": throughput,
            "total_tokens": total_tokens,
        }
    else:
        print("  No output JSON found")
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-15:]:
            print(f"  stdout: {line}")
        return None

def main():
    print(f"Model: {MODEL}")
    print(f"Warmup: {NUM_WARMUP} iters, Bench: {NUM_ITERS} iters")
    print(f"Modes: {[m['label'] for m in MODES]}")

    all_results = {}
    for scenario in SCENARIOS:
        for mode in MODES:
            key = f"{scenario['label']}_{mode['label']}"
            result = run_bench(scenario, mode)
            all_results[key] = result

    print(f"\n{'='*80}")
    print("SUMMARY: MRV2 Eager vs CUDAGraph Performance")
    print(f"{'='*80}")
    print(f"{'Scenario':<20} {'Mode':<12} {'Avg Lat(s)':<12} {'P50(s)':<10} "
          f"{'P99(s)':<10} {'Tput(tok/s)':<14} {'Speedup':<10}")
    print("-" * 88)

    for scenario in SCENARIOS:
        eager_key = f"{scenario['label']}_eager"
        cg_key = f"{scenario['label']}_cudagraph"
        eager_r = all_results.get(eager_key)
        cg_r = all_results.get(cg_key)

        for mode_label, r in [("eager", eager_r), ("cudagraph", cg_r)]:
            if r:
                speedup = ""
                if mode_label == "cudagraph" and eager_r and cg_r:
                    sp = eager_r["avg_latency"] / cg_r["avg_latency"]
                    speedup = f"{sp:.2f}x"
                print(f"{scenario['label']:<20} {mode_label:<12} {r['avg_latency']:<12.4f} "
                      f"{r['p50']:<10.4f} {r['p99']:<10.4f} {r['throughput']:<14.1f} {speedup:<10}")
            else:
                print(f"{scenario['label']:<20} {mode_label:<12} {'FAILED':<12}")

if __name__ == "__main__":
    main()
实测结果
[图片]
性能总结
1. CUDAGraph 在低 batch 场景收益最大（1.47x 加速）
当 batch_size=1 时，模型 Decode 阶段每一步的 GPU 计算量极小（仅 1 个 token 的 forward），此时 CPU-GPU 之间的 kernel launch 开销 成为主要瓶颈。在 Eager 模式下，每一步都需要通过 CUDA Driver 逐个下发几十乃至上百个 CUDA Kernel（Attention、FFN、LayerNorm 等），而每次 launch 的 CPU 开销约 5~15μs。
CUDAGraph 模式将整个 execute_model 的 Kernel 序列捕获为一张静态 Graph，后续每步只需一次 cudaGraphLaunch（开销 ~1μs），彻底消除了 kernel launch 的累积延迟。这在 batch_size=1 的在线推理场景中效果最为显著。
Eager (BS=1):    ┃─launch─┃─launch─┃─launch─┃... × 100+ kernels per step
                 ↑ 每个 launch ~10μs，100 个 = 1ms/step CPU 开销

CUDAGraph (BS=1):┃─graph_launch─┃ ← 仅 1 次 launch ~1μs
                 ↑ 节省 ~1ms/step → 128 steps 共节省 ~128ms
2. 高 batch 场景 GPU 计算占主导，CUDAGraph 收益递减
当 batch_size=64 时，GPU forward 的计算时间远大于 kernel launch 开销，因此 CUDAGraph 的加速比下降到 1.07x。这符合 Amdahl 定律：被优化的部分（launch overhead）在总时间中占比越小，加速效果越不明显。
3. P99 延迟方差大幅降低
暂时无法在飞书文档外展示此内容
CUDAGraph 组合不仅提升了平均吞吐，更关键的是极大地稳定了尾部延迟。这对在线服务的 SLA 保证至关重要——P99 的"毛刺"减少了 70~80%。
通过实测数据可以看到：
- 低 batch（在线推理）场景：CUDAGraph 带来 47% 的延迟降低
- 高 batch（离线批处理）场景：CUDAGraph 带来 7~12% 的吞吐提升和 70~80% 的尾部延迟降低
