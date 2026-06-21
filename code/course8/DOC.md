我们在之前的课程中，已经介绍了 Attention 计算前的准备工作，包括模型加载、服务启动，以及模型执行前必要信息的计算。接下来，我们正式进入模型执行阶段。在这个阶段，我们可以将模型中的各个 Layer 近似看作不同的算子。例如，在 Qwen 模型中，常见的就有 RMSNorm 算子、MoE 算子和 Attention 算子。
其中，最关键的算子之一就是 Attention。Attention 内部包含多个计算步骤，并伴随着大量访存操作。在推理过程中，K 和 V 会随着对话不断累积，形成 KV Cache。因此，随着上下文变长，Attention 中的访存开销占比会越来越高，并逐渐成为性能瓶颈。
也正因为如此，当我们需要实现一个高性能新算子时，直接使用 PyTorch 这种“搭积木”式的方式去拼接现有算子，通常很难达到极致性能。原因在于，这种方式往往不便于我们精细控制以下关键细节：算子融合、特定维度上的并行粒度、SIMD 加载方式，以及计算与访存之间的流水线重叠（overlap）。
此外，对于高性能算子实现，很多参数都需要针对硬件和数据形状进行调优，例如分块大小（tile size）。这类参数通常需要借助 AutoTune 自动搜索最优配置。以矩阵分块为例，不同的分块大小会直接影响算子的吞吐率，而仅用 PyTorch 原生算子堆叠得到的实现，通常难以达到同等级别的优化效果。
因此，下面我们将以 Attention 算子为例，对比基于 OpenAI Triton 编写的高性能实现与 PyTorch 原生实现之间的性能差异。
算子为例，对比 OpenAI Triton 实现与 PyTorch 原生实现在性能上的差异。
OpenAI Triton 定义
OpenAI Triton 定义：Triton 是一种面向 GPU 并行计算的编程语言与编译器，提供基于 Python 的编程环境，帮助开发者高效编写自定义深度神经网络（DNN）Kernel，并在现代 GPU 上获得较高性能。
与 CUDA 的异同：Triton 在语法风格上与 Numpy、PyTorch 较为接近，因此更容易上手；但它本质上仍是一种面向硬件优化的编程语言，开发者需要显式关注底层细节，尤其是内存加载与存储（Load/Store）、数据分块（Tile）以及访存模式，因为这些因素直接影响 Kernel 的执行效率。
与 CUDA 相比，Triton 屏蔽了部分底层线程组织细节，减少了样板代码；但在高性能场景下，开发者依然需要主动设计计算组织方式和内存访问方式。
Triton 的核心要素包括：
- Domain-Specific Language（DSL）：一种嵌入在 Python 中的高性能领域专用语言。
- Tile-based Compute：以分块（Tile/Block）为核心组织计算，开发者通常需要显式设计和调优分块策略。
- JIT Compiler：基于 MLIR/LLVM 的即时编译后端，负责将高级描述编译为优化后的 GPU 机器码。
因此，Triton 既是一种语言，也是一套编译工具链。总体来看，Triton 的关键思想是“围绕 Tile 组织计算”。与 PyTorch 更偏向全局张量视图的编程方式不同，Triton 更接近指针和偏移量驱动的编程模型，开发者需要主动管理数据布局、访存方式和分块策略，这些设计会直接决定 Kernel 的性能表现。相较于 CUDA，Triton 通常能以更短的开发周期实现高性能自定义算子。
Triton 有 3 个重要特性：
1. 以 Block（Tile）为编程粒度：Triton 更关注块级别的数据组织与计算。开发者通常描述“一个程序实例如何处理一个数据块”，而不需要像 CUDA 那样直接处理复杂的 grid -> block -> thread 层次细节。
2. 强大的优化 Pass：编译器通过一系列内置的优化 Pass（如内存聚合、计算融合），其生成的算子性能可达到与 cuBLAS 等官方库相近的水平。
3. 与 PyTorch 生态无缝衔接：直接接受 Torch 张量作为输入，通过其起始指针与偏移量进行灵活的存储访问。下文将展示 Triton 如何通过简单的装饰器与 PyTorch 环境高度融合。
Triton 与 Cuda 的差异
下图总结了 Triton 和 CUDA 的一些关键区别。可以看到，Triton 的简洁性来自更高层次的抽象，但这也意味着开发者对部分底层细节的直接控制会减少。以 Shared Memory（共享内存）为例，在 CUDA 中，开发者通常需要显式管理这类片上存储资源；而在 Triton 中，开发者更多是描述块级计算和访存模式，具体的底层资源组织与优化通常由编译器完成。
另一个重要区别是同步与执行组织方式。CUDA 编程往往需要开发者显式处理线程索引、线程协作以及部分同步细节；而 Triton 采用更高层的编程模型，开发者主要关注一个 Block（Tile）上的计算逻辑，许多底层执行细节由编译器结合硬件后端负责映射与优化。这样可以明显降低线程级编程复杂度，也减少一部分因手工管理线程协作而引入错误的风险。
换句话说，Triton 编程模型的一大优势在于其 Block-wise（分块级）编程方式。用户重点考虑的是 Block（Tile）维度上的数据组织、访存方式和计算过程，而不必像 CUDA 那样手动维护复杂的线程索引（Thread Indexing）。至于 Block 内部如何调度，以及如何映射到具体硬件执行单元，则主要由 Triton 编译器负责。
同时，Triton 在内存访问优化、向量化以及部分硬件特性的利用方面提供了较强的编译器支持，能够减少 CUDA 编程中大量繁琐的手动优化工作。在合适的场景下，这种更高层次的抽象既能提升开发效率，也能获得较好的性能表现。
[图片]
在这里我们也简单的过一下Triton和Pytorch在寻址上的区别：
[图片]
triton 编译过程简单总结
Triton kernel → Triton IR → LLVM IR → PTX，最终配合 runtime 运行。
上述编译过程主要由 @triton.jit 装饰器触发。具体而言，Triton 会对 Python kernel 进行解析和追踪，生成 Triton IR（TTIR，一种基于 MLIR 的高层表示）。
随后，编译器后端会对 IR 执行一系列优化和降级（Lowering）过程，将其转换为 TritonGPU IR（TTGIR），以处理与硬件相关的访存布局和执行映射。接着，代码会继续下沉到 LLVM IR，并生成目标平台的 PTX。
最后，PTX 由 NVIDIA 驱动加载到 GPU 上执行，runtime 负责参数传递、kernel launch 和结果管理。
[图片]
第一个 Triton 算子
预备知识
我们将通过实践来掌握使用 Triton 编写算子的基本方法。第一个示例是一维向量加法。在这个示例中，会用到 3 个核心接口和一类重要配置参数：
1. tl.load：从指针指定的内存位置加载数据。
2. tl.store：将数据写回指针指定的内存位置。
3. tl.program_id(axis)：返回当前 program instance 在指定轴上的编号。axis 是编译期常量，用于指定查询哪一个启动维度。
4. 编译时元参数（Meta-parameters）：在 Triton 中，经常需要向 kernel 传入一些编译期已知的配置参数，例如 BLOCK_SIZE。这类参数通常在 kernel 函数签名中声明为 tl.constexpr，编译器可以基于它们执行循环展开、死代码消除等优化。
```Python
META = {
    'BLOCK_SIZE': 128,  # 每个块的大小
    'ANOTHER_PARAM': 42,  # 其他参数
    # 其他配置参数...
}```
一维向量加法是理解 Triton 编程模型的经典入门例子。下面给出示例代码。此时不需要完全理解每一行细节，只需要先对 kernel 的整体执行流程建立一个基本印象，后续我们会再逐步拆解。
暂时无法在飞书文档外展示此内容
算子实现
完整代码见code/course8/demo.py
```Python
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):

    # 1，定义每个线程在全局数据中的具体索引
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets # 有时直接写 block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # 2，加载数据，并执行内核算法（向量加法）
    x = tl.load(X_ptr + idx, mask=mask)
    y = tl.load(Y_ptr + idx, mask=mask)
    
    # 3，执行向量加法(内核算法)
    z = x + y                                     
    
    # 4，存储结果
    tl.store(Z_ptr + idx, z, mask=mask)  ```
只有正确理解内核中的 pid、block_start、offsets、mask和 idx 这5个概念，才能编写出正确且高效的内核代码。
1. pid（Program ID）：当前块（Block）的唯一标识符，代表该块在整个网格（Grid）中的位置（即第几个块）。在一维网格中，可通过 pid = tl.program_id(0) 获取。我们将输入数据（大小为 N）划分为 B 个块，pid = tl.program_id(0) 即用于确定当前正在处理的是第几个块，每个块负责将两个输入向量中对应位置的元素相加得到结果。也就是说，每个块的唯一标识通过 pid = tl.program_id(axis=0) 获取。
2. block_start：当前块在全局数据中的起始位置索引，用于确保各块处理的数据范围互不重叠且完整覆盖整个数据集。计算公式为 block_start = pid * BLOCK_SIZE。
3. offsets：表示当前块内每个线程相对于块起始位置的偏移量，用于帮助每个线程计算其在全局数据中的具体索引。通过 offsets = tl.arange(0, BLOCK_SIZE) 生成，用于索引块内的所有位置。
4. idx：表示每个线程在全局数据中的具体索引，用于加载和存储数据，确保每个线程处理唯一的数据元素。计算公式为 idx = block_start + offsets。注意，这里的 idx 是一个向量，代表了该分块负责的一组全局位置。
5. mask = idx < N：用于创建掩码，防止线程访问超出数据范围之外的元素。例如，若总数据量为 999，分为 4 个块，每个块处理 256 个数据，则最后一个块若不使用掩码便会出现越界访问。
为了更好的理解上述 4 个变量的关系和值意义，可通过一个实例。假设我们有一个向量长度为 N = 10，BLOCK_SIZE = 4，则内核执行后，各变量内容如下。从下表可以看出一共处理了N=10个元素，mask的作用就是在分块加载的时候屏蔽掉超出范围的索引。
Block ID (pid)
block_start
offsets
idx (global index)
0
0×4=0
[0,1,2,3]
[0,1,2,3]
1
1×4=4
[0,1,2,3]
[4,5,6,7]
2
2×4=8
[0,1,2,3]
[8,9,10,11] (mask applied for N=10)
总的来说，以上流程的关键步骤可归纳为：加载两个输入操作数、执行计算操作、将结果写回目标位置。
```Python
 # 加载数据
x = tl.load(X_ptr + idx, mask=mask)           # 加载 X 的值
y = tl.load(Y_ptr + idx, mask=mask)           # 加载 Y 的值
# 执行向量加法(内核算法)
z = x + y                                     # 执行加法
# 存储结果
tl.store(Z_ptr + idx, z, mask=mask)           # 存储结果到 Z```
main 函数调用
编写完上述 kernel 后，我们需要通过 main 函数进行调用。调用时需传入两个作为向量加法输入的 torch.Tensor 和一个用于存放结果的 torch.Tensor。也就是说，main 函数的作用包括：初始化张量、进行 GPU 预热、执行 Triton 向量加法并记录时间、执行 PyTorch 向量加法并记录时间，以及验证结果并输出性能对比。
完整代码见 code/course8/demo.py
```Python
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
    ...
    
    
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python wrapper for the Triton kernel."""
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    N = x.numel()
    z = torch.empty_like(x)

    # 配置 grid：每个 block 处理 BLOCK_SIZE 个元素
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    # 启动 kernel
    vector_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z```
kernel 在实际执行时会被重复调用多次，每次处理输入数据的一部分，直至所有输入处理完成。然而，kernel 内部并未使用 for 循环实现上述过程，原因在于这些不同数据段的处理实际上是并行执行的。
- program_id 可以理解为当前是第几个线程块。axis=0 表示查询第 0 个网格维度上的编号，并不是表示“有几层循环”。在这个示例中，program_id(0) 用来获取当前块的索引。
- grid 用于指定分块计算的总块数。若数据总量为 N，每个块的大小为 BLOCK_SIZE，则分块数量 grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE，在 Triton 中可写作 triton.cdiv(N, BLOCK_SIZE)。
- BLOCK_SIZE 用于定义每次内核执行时加载的内存/元素数量。
在 Triton 中，内核函数是并行执行的，每个内核负责处理不同的数据范围。内核的执行数量与分块数量相关，多内核并行实际上就是多块并行——即多个块可在不同的多处理器（SMs）上同时运行，而每个块内的线程也在其所属的 SM 上并行执行。
小结
Triton巧妙地将Python的直观语法与GPU的高性能计算能力相结合，为传统GPU开发模式带来了革新。与直接编写复杂的CUDA代码不同，Triton使开发者能够以类似Python的高级抽象方式描述并行计算逻辑，而无需深入底层硬件细节。
更复杂的 Softmax 算子
Softmax 函数是一种广泛应用于机器学习，特别是在多分类问题中的激活函数。它的作用是将任意实数向量转换为概率分布，并确保所有输出概率之和为 1。给定输入 $$x\in R^{M\times N}$$，执行逐行 Softmax（对于二维张量，逐行计算对应维度 dim = 1）。
给定输入张量 $$x \in R^{M \times N}$$，执行逐行 Softmax（即针对二维张量的维度 dim = 1 进行计算）。为了保证计算的数值稳定性（防止 $$\exp$$ 函数溢出），在底层实现中通常采用 Safe Softmax 形式，其公式如下：
pytorch 中 softmax 实现的 c++ 代码在 Softmax.cpp
$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)} \\
\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)$$
其中：
- $$x_i$$ 是输入向量中的第 $$i$$ 个元素。
- $$n$$ 是输入向量的长度。
- 输出的每个值都是在 0 到 1 之间，并且所有输出值的总和为 1，表示概率分布。
torch实现
原生 softmax 算子如下：
```Python
import torch

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    """
    # 先对每行求一个最大值
    x_max = x.max(dim=1)[0] # read  MN elements ; write M  elements
    # 每行各自减去最大值
    z = x - x_max[:, None] # read MN + M elements ; write MN elements
    numerator = torch.exp(z) # read  MN elements ; write MN elements
    # 对指数后的结果进行累加，得到分母denominator
    denominator = numerator.sum(dim=1) # read  MN elements ; write M  elements
    ret = numerator / denominator[:, None]  # read MN + M elements ; write MN elements
    
    return ret```
naive_softmax 函数实现了行级（row-wise）的 softmax 计算，具体步骤如下：
1. x_max = x.max(dim=1)[0]：x.max(dim=1) 返回每一行的最大值及对应索引，[0] 表示仅取最大值部分。
2. z = x - x_max[:, None]：为提升数值稳定性，减去每行的最大值，避免计算 exp 时出现溢出。[:, None] 将 x_max 从形状 (M,) 扩展为 (M, 1)，以便进行广播减法。
3. numerator = torch.exp(z)：计算 exp(z) 作为分子部分。
4. denominator = numerator.sum(dim=1)：按行求和，得到分母。
5. ret = numerator / denominator[:, None]：分子除以分母，得到 softmax 结果。
需要注意的是，代码注释中给出的数据访问量是：在将每一步都视为独立张量操作的前提下，计算 y = naive_softmax(x) 时，总读取量为 5MN + 2M 个元素，总写入量为 3MN + 2M 个元素。
之所以会有 5MN + 2M 次读取和 3MN + 2M 次写入，是因为这个实现将 softmax 拆成了 5 个张量操作。这样一来，每一步都会产生额外的中间结果读写；而在融合后的 kernel 中，这些中间结果很多可以保留在更快的片上存储中，例如寄存器或 shared memory，从而显著减少访存开销。
Triton实现
上述原生实现的内存访问量较大，因此效率并不高。更好的做法是使用自定义的融合 kernel，将一整行 softmax 所需的计算尽量放在同一个 kernel 中完成，减少中间结果的回写和重复读取。这样理想情况下只需要读取一次输入 X，并写出一次输出结果。这里的 2MN 就表示融合实现的总内存访问量：输入矩阵共有 MN 个元素，读取一次是 MN；输出矩阵同样有 MN 个元素，写出一次也是 MN，因此总访问量约为 2MN。
若把性能瓶颈近似看作内存带宽，那么相较于前面的原生实现，理论上可获得约$$\frac{8MN + 4M}{2MN}$$
倍的带宽收益；当 N 足够大时，这个值可近似看作 4 倍。需要注意，这只是理想化的上界估计，实际加速效果还会受到 kernel 启动开销、访存模式和硬件资源利用率等因素影响。
那么问题来了：和前面的一维向量不同，二维矩阵该如何读取和处理？一种常见做法是让 Triton 的程序实例按行处理数据，在给定步幅（stride）的条件下定位每一行的起始地址，然后完成该行的加载、归约和写回。
需要特别说明的是，在这个 softmax 示例里，通常会将每个程序实例处理的列数设置为不小于实际列数的最小 2 的幂。这样做有利于实现高效的块内计算，也便于配合 Triton 的向量化和归约操作。因此，当输入矩阵的列数不是 2 的幂时，往往需要在 kernel 内部通过掩码（mask）处理“填充出来的位置”，以保证访存安全和计算正确性。
暂时无法在飞书文档外展示此内容
```Python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, 
                output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # 一个块处理一行元素，idx 表示第几行，每行之间的处理是并行的
    row_idx = tl.program_id(0) 
    # 步幅表示我们需要增加指针多少才能前进 1 行
    row_start_ptr = input_ptr + row_idx * input_row_stride 
    # 块大小是大于 n_cols 的下一个 2 的幂，因此我们可以将每一行放在一个块中
    col_offsets = tl.arange(0 , BLOCK_SIZE) 
    input_ptrs = row_start_ptr + col_offsets 
    
    # 这一行中所有元素的内存地址向量。
    # using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols）

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 将结果行数据写入到指定地址范围中
    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # grid = lambda meta: (triton.cdiv(n_rows*n_cols, meta['BLOCK_SIZE']),)

    # 增加每行分配的 warp 数量（num_warps）
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[n_rows](x, 
        y, 
        x.stride(0), 
        y.stride(0), 
        n_cols, 
        num_warps=num_warps,
        BLOCK_SIZE = BLOCK_SIZE)

    return y```
这段代码的含义是：Triton 会启动 n_rows 个程序实例，每个程序实例负责对输入矩阵的一行执行 softmax。因此，这个实现本质上是按行并行的。后续的优化方向可以包括：让一个程序实例处理更多数据、提高访存效率，或者根据矩阵形状调整并行映射方式，但是否让单个程序实例处理多行，需要结合具体场景进一步分析。
另外，为什么列偏移使用 tl.arange(0, BLOCK_SIZE)，而不是 tl.arange(0, n_cols)？
- 在这个 softmax kernel 中，每个程序实例负责处理输入矩阵的一整行，而 BLOCK_SIZE 决定了该程序实例一次按多宽的向量形式处理这一行。
- 之所以使用 BLOCK_SIZE 而不是实际列数 n_cols，是因为 BLOCK_SIZE 通常会被设置为不小于 n_cols 的最小 2 的幂，例如 32、64、128。这样更有利于 Triton 编译器生成高效的块级加载、归约和向量化代码。
- 由于 BLOCK_SIZE 可能大于实际列数，因此需要在 tl.load 和 tl.store 中配合 mask，只让前 n_cols 个位置参与有效计算，超出范围的位置则被屏蔽掉。这样既能兼顾统一的块大小，又能保证访存安全和结果正确。
暂时无法在飞书文档外展示此内容
最终，我们的比较结果体现在 code/course8/softmax.py 文件中。Triton 实现与前述 PyTorch 实现的计算结果完全一致，并且获得了显著的性能提升，其速度比 PyTorch 实现高出 4 倍以上。
```Python
============================================================
⏱️  Performance Benchmark (shape=1024x1024)
============================================================
PyTorch:   0.672 ms
Triton V1: 0.163 ms
Speedup V1 over Torch: 4.11x```
实现 Matmul 算子
矩阵计算的原理
矩阵乘法是线性代数中的基本操作之一，定义如下：
给定两个矩阵 A 和 B，其中 A 的维度为 (M × K)，B 的维度为 (K × N)，则它们的乘积 C = A × B 的维度为 (M × N)，且 C 中的元素 C[i][j] 由以下公式计算：
$$C[i][j] = \sum_{k=1}^{K} A[i][k] \times B[k][j]$$
矩阵计算过程举例如下图所示：
[图片]
不带任何优化的矩阵乘法 python 代码如下所示:
```Python
def matrix_multiply(A, B):
    # A B 都是二维列表
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    assert cols_A == rows_B
    # 初始化矩阵 C，形状为 [rows_A, cols_B]
    C = [0 for _ in range(cols_B)] for _ in range(rows_A)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(rows_B):
                C[i][j] += A[i][k] * B[k][j]

    return C```
在进行矩阵乘法时，通常需要使用三重嵌套循环来组合矩阵元素。实现通用矩阵乘法（GEMM）时，需要在 M 和 N 维度上遍历输出矩阵，同时在共享维度 K 上完成累加计算。
这里的“共同迭代”是指：在计算 C[i, j] 时，需要同时访问 A[i, k] 和 B[k, j]，并沿共享维度 K 进行累加。根据 K 所在循环层次的不同，矩阵乘法会呈现出不同的数据流特征：
- 内积法（IP，Inner Product）：将共享维度 K 放在最内层循环。该方法聚焦于结果矩阵中的单个元素，通过沿 K 维做累加，直接得到一个完整的输出元素。它的特点是一次产生一个结果值，不需要额外合并部分和。
- 外积法（OP，Outer Product）：将共享维度 K 放在最外层循环。每次固定一个 k，取 A[:, k] 和 B[k, :] 做一次外积，生成一个中间结果矩阵，并将其加到结果矩阵上。它的特点是一次更新结果矩阵中的一大片区域，因此会产生较多部分和累积。
- Gustavson 算法：将共享维度 K 放在中间层循环。它介于内积法和外积法之间，通常围绕某一行或某一列组织部分结果的累积，以兼顾数据重用和部分和管理。
除了共享维度 K 的位置之外，M 和 N 两个维度的遍历顺序也会影响计算模式。对于每一种数据流形式（IP、OP 和 Gustavson），如果再考虑 M/N 两种遍历顺序，就会得到 3 x 2 = 6 种可能的循环变体。
选择哪一种变体，核心取决于数据重用效率。在矩阵乘法中，如果能够让已经加载到高速存储中的数据被更多次复用，就能显著减少访存开销并提升性能。因此，不同的数据流本质上是在权衡：让哪一维的数据尽量驻留在缓存、寄存器或片上存储中，以获得更高的整体计算效率。
[图片]
分块矩阵乘法
虽然经验证，上述矩阵乘法函数的结果正确，但其性能明显不足。此时可考虑采用分块矩阵乘法（Tiled Matrix Multiplication），即将矩阵划分为多个块，每次仅计算其中部分内容。分块的主要目的是优化访存模式，通过将计算限制在特定数据区域内，提高数据局部性，从而提升缓存利用率并改善整体性能。分块矩阵乘法的伪代码如下所示：
```Python
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc```
注意，这里的代码更多的是为了表现矩阵乘法的思路，更详细的 c++ 实现可以参考这里。
用分块（Block）的方法，将大矩阵划分为更小的子块，逐块进行乘法计算并累加结果。矩阵子块乘法（acc += dot(a, b)）：
其中：
- dot(a, b) 表示两个矩阵子块 a 和 b 的乘法结果，这里是子块级矩阵乘法，而不是向量点积。
- acc 是累加器，用于保存当前输出子块的部分和。
- 在实际计算中，通常会沿着 K 维不断读取新的子块对 (a, b)，并将每次得到的结果持续累加到 acc 中，从而得到最终的输出子块。
```Python
+——--+—-——+     +——--+—-——+     +——————————---+—-—————————-—+
| A1 | A2 |     | B1 | B2 |     | A1.B1+A2.B3 | A1.B2+A2.B4 |
+——--+—-——+ dot +——--+—-——+  =  +——————————---+—-—————————-—+
| A3 | A4 |     | B3 | B4 |     | A3.B1+A4.B3 | A3.B2+A4.B4 |
+——--+—-——+     +——--+—-——+     +——————————---+—-—————————-—+```
分块矩阵乘法原理的如下图所示：​
[图片]
小块矩阵结果如下所示：
[图片]
Python 完整矩阵乘法及分块矩阵乘法优化的代码如下所示:
```Python
import torch,time
import numpy as np


def matrix_multiply(A, B):
    # A B 都是二维列表
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    assert cols_A == rows_B
    # 初始化矩阵 C，形状为 [rows_A, cols_B]
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(rows_B):
                C[i][j] += A[i][k] * B[k][j]

    return C

def block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k):
    # 获取矩阵 A 和 B 的维度
    M, K = A.shape
    K_b, N = B.shape
    
    assert K == K_b, "矩阵 A 的列数必须等于矩阵 B 的行数"

    # 初始化结果矩阵 C
    C = np.zeros((M, N), dtype=np.float32)

    # 分块矩阵乘法
    for m in range(0, M, block_size_m):
        for n in range(0, N, block_size_n):
            # 初始化累加器块
            acc = np.zeros((block_size_m, block_size_n), dtype=np.float32)
            for k in range(0, K, block_size_k):
                # 取矩阵 A 和 B 的子块
                a_block = A[m:m+block_size_m, k:k+block_size_k]
                b_block = B[k:k+block_size_k, n:n+block_size_n]
                
                # 累加块的矩阵乘法结果
                acc += np.dot(a_block, b_block) # 本质上就是小块矩阵乘法
            
            # 将累加结果赋值给结果矩阵 C 的对应子块
            C[m:m+block_size_m, n:n+block_size_n] = acc

    return C

if __name__ == "__main__":
    # 示例矩阵
    M, K, N = 9, 12, 15  # A 是 MxK 矩阵，B 是 KxN 矩阵
    A = np.random.rand(M, K).astype(np.float32)  # 生成随机矩阵 A
    B = np.random.rand(K, N).astype(np.float32)  # 生成随机矩阵 B
    
    # 分块大小
    block_size_m = 3
    block_size_n = 3
    block_size_k = 4
    
    start_time = time.time()
    C_python = matrix_multiply(A, B) # 普通矩阵乘法
    matmul_time = time.time() - start_time
    
    start_time = time.time()
    C_block = block_matrix_multiply(A, B, block_size_m, block_size_n, block_size_k) # 调用分块矩阵乘法
    block_matmul_time = time.time() - start_time
    
    start_time = time.time()
    C_np = np.dot(A, B) # numpy 矩阵乘法
    np_matmul_time = time.time() - start_time
    
    # print("NumPy 矩阵乘法结果:\n", C_python)
    # print("分块矩阵乘法结果:\n", C_block)
    # print("NumPy 矩阵乘法结果:\n", C_np)
    
    # 验证两者结果是否相等
    if np.allclose(C_block, C_np, atol=1e-6) and np.allclose(C_python, C_np, atol=1e-6) :
        print("\n结果验证通过: 分块矩阵乘法和普通矩阵乘法与 NumPy 结果一致！")
    else:
        print("\n结果验证失败: 分块矩阵乘法普通矩阵乘法与 NumPy 结果不一致。")
        
    # 输出时间
    print(f"python matmul 时间: {matmul_time * 1000:.2f} ms")
    print(f"Python block matmul 时间: {block_matmul_time * 1000:.2f} ms")
    print(f"numpy matmul 时间: {np_matmul_time * 1000:.2f} ms")```
缓存命中率（Cache Hit Rate）是衡量缓存系统性能的一个重要指标，表示缓存请求中成功命中的比例，即从缓存中直接读取数据的次数占总访问次数的百分比。
本图结合视频一起看
暂时无法在飞书文档外展示此内容
triton 版本实现如下所示，完整代码见code/course8/matmul.py
```Python
@triton.jit
def _fused_linear_kernel_fwd(
        x_ptr,  # 输入数据矩阵首元素指针
        w_ptr,  # 权重矩阵首元素指针
        z_ptr,  # 输出结果地址
        M, N, K,  # Matrix dimensions
        BLOCK_SIZE_M: tl.constexpr = 128,  # 块大小
        BLOCK_SIZE_N: tl.constexpr = 128,
        BLOCK_SIZE_K: tl.constexpr = 64,
):
    # 对于每个triton block的二维坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 一个triton block的处理范围（在M,N轴上)
    # 生成当前块处理的行索引 (offs_m) 和 列索引 (offs_n)
    # [:, None] 将向量转为 (BLOCK_SIZE_M, 1) 的列向量
    # [None, :] 将向量转为 (1, BLOCK_SIZE_N) 的行向量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]  # 形状为 (1, BLOCK_SIZE_N)。

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 在k轴上进行一个分块归约
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        # tl.load()加载了一个块，加载的是一个范围(offs_m，xk)

        # x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k，k是现在处理块的一个起始地址，
        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)
        x = x.to(tl.float16)

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        # tl.load加载的是(w_k,offs_n)
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
        w = w.to(tl.float16)
        # 分块相乘
        z = tl.dot(x, w, acc=z)
    # 一个triton block计算的结果大小是block_m×block_n
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    tl.store(z_ptr + z_offset, z, mask=z_mask)
```
1. 先计算 M 轴和 N 轴上的偏移，用于确定当前输出子块在结果矩阵 Z 中的位置。通过 [:, None] 和 [None, :]，offs_m 与 offs_n 分别扩展成列向量和行向量，二者在广播后共同描述当前子块的二维坐标。
```Python
offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :] # 形状为 (1, BLOCK_SIZE_N)。```
2. 确定 M 轴和 N 轴的位置后，再沿共享维度 K 逐块滑动并完成累加。矩阵乘法的本质，是沿 K 维对乘积项求和。也就是说，为了计算输出块中的每个元素，都需要遍历 K 维上的对应分块：
```Python
for k in range(0, K, BLOCK_SIZE_K):
    x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
    # tl.load()加载了一个块，加载的是一个范围(offs_m，xk)```
这里的 x_k 表示当前 K 维分块中的列偏移。
3. 接下来加载矩阵 A 的一个子块。这个子块由 offs_m 和 x_k 共同确定：offs_m 负责行坐标，x_k 负责列坐标，因此加载到的是 A 中当前 M x K 小块的数据。
```Python
x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)
x = x.to(tl.float16)```
4. 同理，再加载矩阵 B 的对应子块。对于 B 而言，w_k * N 用于定位到对应行，offs_n 用于定位该行中的列位置：
```Python
w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
# tl.load加载的是(w_k,offs_n)
w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
w = w.to(tl.float16)```
这里加载的是 B 中当前 K x N 小块的数据。
5. 将分块 A 与分块 B 相乘，结果写入 Z 中。
```Python
z = tl.dot(x, w, acc=z)```
注意，这一步并不是立即写回输出矩阵，而是继续在寄存器中的累加器 z 上累加当前 K 分块的贡献。
6. 当 K 维上的所有分块都遍历完成后，z 中就保存了当前输出子块的完整结果。最后，再根据 M 轴和 N 轴的偏移，将它写回到输出矩阵 Z 的对应位置：
```Python
z_offset = offs_m * N + offs_n
z_mask = (offs_m < M) & (offs_n < N)

tl.store(z_ptr + z_offset, z, mask=z_mask)```
因此，这段代码的整体流程可以概括为：先定位输出子块的位置，再沿 K 维逐块加载 A 和 B 的子块，持续累加乘法结果，最后将完整的子块结果写回输出矩阵。
实现 Attention 算子
单头注意力机制
由于部分同学可能不熟悉自注意力机制，我们先简要介绍其计算公式。以下是自注意力机制的核心公式，其中 Q、K、V 为输入矩阵，d_k 表示 K 的维度，softmax 为 softmax 函数，Attention 为自注意力机制的输出。Q、K、V 的维度均为 (batch_size, seq_length, dim)：
$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
在计算 Q 矩阵与 K 矩阵相乘时，需要先对 K 矩阵进行转置，再进行矩阵乘法。由于本课程侧重于工程实践，我们将不过多涉及数学推导，重点在于理解公式的来源及其实际应用方法。
多头自注意力机制
另外还有一个多头自注意力机制，这个机制是说，我们把Q，K，V 矩阵分成多个头，然后每个头都进行自注意力机制的计算，最后把每个头的注意力计算结果拼接起来。有公式为： (batch_size, seq_length, num_head, head_dim)。
$$\text{Attention}(Q, K, V) = \text{concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)$$
$$head_i = \text{Attention}(Q_i, K_i, V_i)，Q_i, K_i, V_i 是 Q, K, V 矩阵的第i个头。$$
另外值得说明的是，在计算Attention的时候，我们常常会使用一个mask矩阵，这个矩阵的作用是防止模型看到未来的信息，mask的维度为(batch_size, seq_length, seq_length)，其中seq_length是Q矩阵的序列长度，它的值是一个下三角矩阵，其中对角线及以下的值为1，其他值为0，加上mask的公式为：
$$\text{Attention}(Q, K, V, mask) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + mask)V$$
暂时无法在飞书文档外展示此内容
torch实现
我们现在将使用pytorch实现一个多头自注意力机制，这里我们只需要跟着公式一步步实现就可以了，代码如下，如公式所说的那样，该函数传入分别传入Q, K, V矩阵，以及softmax的缩放因子sm_scale，以及mask矩阵。
1. 计算Q和K的乘积，得到注意力分数矩阵attn_scores，然后乘以softmax的缩放因子sm_scale，对应公式的部分为$$\frac{QK^T}{\sqrt{d_k}}$$
2. mask矩阵如果存在，它将用于遮挡未来信息，也就是说将attn_scores矩阵中mask为0的位置填充为负无穷。
3. 使用softmax函数对注意力分数矩阵attn_scores进行归一化处理。对应公式为：
$$softmax(\frac{QK^T}{\sqrt{d_k}}+mask)$$
4. 计算注意力输出out，将注意力分数矩阵attn_weights与V矩阵相乘。对应公式为：
```Python
def standard_attention(Q, K, V, sm_scale, mask=None):
    """
    标准的 PyTorch 实现的自注意力机制。
    
    Args:
        Q (torch.Tensor): 查询张量，形状 (batch_size, num_heads, seq_length, head_dim)
        K (torch.Tensor): 键张量，形状 (batch_size, num_heads, seq_length, head_dim)
        V (torch.Tensor): 值张量，形状 (batch_size, num_heads, seq_length, head_dim)
        sm_scale (float): Softmax 缩放因子
        mask (torch.Tensor, optional): 遮罩张量，形状 (batch_size, num_heads, seq_length, seq_length)
    
    Returns:
        torch.Tensor: 注意力输出，形状与 Q 相同
    """
    # 计算 QK^T
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale  # (batch_size, num_heads, seq_length, seq_length)
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)
    
    return out```
Flash Attention 实现拆解
配套代码见：code/course8/attention.py
论文中的公式
[图片]
Triton 实现
在公式中我们可以看出，Q被分为T_r块，K和V被分为T_c块，O被分为T_r块，切分的维度是Q,K,V矩阵的seq_length维度，B_r和B_c是分块的大小，M是SRAM的大小，d是Q, K, V的dim维度。虽然都是沿着seq_length维度进行切分，但Q和（K, V）的切分方式不相同。
从本章的 flashattention.py 中可以看出来，我们配置二维网格，第 0 维网格用于配置在 seq_length 维度上的分块大小，第 1 维网格用于配置 batch_size 和 head 的数量。所以在当前执行的 kernel 中，我们用 pid 的不同维度来索引 Q 矩阵在 seq_length 维度上的分块。见代码：
```Python
 # 这里有一个隐形的遍历（用triton的并行实现），对seq_len维度上的一个遍历
 m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs
 ...
 q_offs = (cur_batch_idx * q_batch_stride 
        + cur_head_idx * q_heads_stride
        + (m_offs[:, None] * q_seq_stride + dhead_range_offs[None,:] * q_dim_stride))

 o_offs = (cur_batch_idx * out_batch_stride 
        + cur_head_idx * out_heads_stride
        + (m_offs[:,None] * out_seq_stride + dhead_range_offs[None,:] * out_dim_stride))
        
 # m_offs[:, None] * q_seq_stride 这里相当于在对query做一个分块，分块的大小是br```
从代码中可以看出，我们取得了q矩阵在seq_len维度上一块长度为BLOCK_M_SIZE的块，分块的范围(i × br, i × br + br - 1)，对于o矩阵同理。因为q, o矩阵划分的大小是一样的。
```Python
q_ptrs = q_ptr + q_offs
o_ptrs = o_ptr + o_offs ```
[图片]
在上面的这个公式中，我们进行了一个二维遍历，针对以j为索引的遍历，我们在triton中继续采用循环的方式，而对于以i为索引的遍历，我们则采用triton的网格来配置，就像我们上面说的在q维上我们用第0维上的pid来确定seq_len维度上的分块位置。根据流程：
1. 我们先需要把Q, K分别加载到SRAM中，有如下的代码。此处的循环就是我们在上面所说的以j作为索引的循环，我们将block_n_offs作为索引，将k_ptrs作为加载的指针，同时以k_seq_stride作为步长，对k值进行加载。
[图片]
```Python
 # 对j轴的一个遍历，BLOCK_N_SIZE一个bc
 for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
    block_n_offs = block_n_start_idx + n_range_offs
    k_mask = block_n_offs[:, None] < n_size
    # 加载的k大小是(bc, dim)
    k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)
    
    qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))```
1. 计算注意力分数矩阵，已知计算公式$$S_{i,j} = Q_i K_j^T$$
$$S_{i,j}$$ 是注意力分数矩阵
$$Q_i$$ 是查询矩阵的第i个块
$$K_j$$ 是键矩阵的第j个块
```Python
qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
qk += tl.dot(q, tl.trans(k))```
2. 计算注意力分数qk中每行的最大值，已知计算公式$$m̃_{i,j} = \text{rowmax}(S_{i,j})$$
$$m̃_{i,j}$$是注意力分数（当前子块）每行的最大值
$$S_{i,j}$$ 是注意力分数矩阵，等于Q和K子块的矩阵乘
```Python
# 计算每行的最大值，用于数值稳定性
# qk (S_{i,j}): Q和K的乘积子块，形状为 [BLOCK_M_SIZE, BLOCK_N_SIZE]，维度等于br×bc
# l_j (m̃_{i,j}): 每行的最大注意力分数，形状为 [BLOCK_M_SIZE]
l_j = tl.max(qk, 1) # l_j 等于m_{ij}```
3. 计算缩放后的注意力分数$$P̃_{i,j}

$$ 和注意力分数每行之和（后续用于归一化）的 $$ℓ̃_{i,j}$$，有对应公式为：
$$P̃_{i,j} = \exp(S_{i,j} - m̃_{i,j})
\\[1em]
ℓ̃_{i,j} = \text{rowsum}(P̃_{i,j})$$
已知：
$$P̃_{i,j}$$ 是经过缩放后的注意力分数
$$S_{i,j}$$ 是原始注意力分数矩阵
$$m̃_{i,j}$$ 是$$S_{ij}$$每行的最大值（当前子块）
$$ℓ̃_{i,j}$$  是缩放后每行注意力分数的和（当前子块）
```Python
# 计算经过缩放的注意力分数
# qk (S_{i,j}): Q和K的乘积子块
# l_j (m̃_{i,j}): 子块每行的最大值
# numerators (P̃_{i,j}): 经过指数化和缩放的注意力分数
numerators = tl.exp(qk - l_j[:, None]) # 得到分块的归一化项

# 计算归一化项
# d_j (ℓ̃_{i,j}): 注意力分数每行的和，用于后续的softmax归一化
d_j = tl.sum(numerators, 1)```
4. 更新计算注意力分数中的行最大值$$m^{new}_i$$，对应公式为：$$m^{new}_i = \max(m_i, m̃_{i,j})$$
```Python
# 更新每行的最大值
# l_i (m_i): 当前历史的最大值
# l_j (m̃_{i,j}): 新计算的子块最大值
# l_new (m^new_i): 更新后的最大值，取两者中的较大者
l_new = tl.maximum(l_i, l_j)```
5. 随后我们就需要更新$$\ell^{new}_i $$，也就是更新归一化项，对应公式为：
$$\ell^{new}_i = e^{m_i - m^{new}_i} \ell_i+e^{m^{new}_i-m_i} \tilde{\ell}_{i,j}$$
```Python
l_new = tl.maximum(l_i, l_j)  # m^new_i = max(m_i, m̃_{i,j})

# 计算缩放因子
# alpha: 对历史数据的缩放系数 exp(m_i - m^new_i)
alpha = tl.exp(l_i - l_new)
# beta: 对新数据的缩放系数 exp(m̃_{i,j} - m^new_i)
beta = tl.exp(l_j - l_new)

# 更新归一化项
# d_i: 历史累积的归一化项
# d_j: 新计算的归一化项
# d_new: 更新后的归一化项 = alpha * d_i + beta * d_j
d_new = alpha * d_i + beta * d_j```
6. 更新输出O矩阵
$$\begin{aligned}
&  O_i \leftarrow \text{diag}(\ell^{new}_i)^{-1}(\text{diag}(\ell_i)e^{m_i - m^{new}_i} O_i 
 +\, e^{\tilde{m}_{i,j}-m^{new}_i}\tilde{P}_{i,j}V_j) 
\end{aligned}$$
我们根据公式对代码进行分析，上述对Oi的更新公式我们拆分为两个部分，第一部分是$$diag(ℓ_i)e^{m_i - m^{new}_i} O_i$$，第二部分是$$e^{\tilde{m}_{i,j} - m^{new}_i} P̃_{i,j} V_j$$：
1. 第二部分：$$beta = e^{m̃_{i,j} - m^{new}_i}，numerators = P̃{i,j}$$，p = numerators * p_scale[:, None]对应公式中就为：$$e^{\tilde{m}_{i,j} - m^{new}_i} P̃_{i,j}$$，随后还需要乘以V矩阵的分块部分
2. 第一部分：根据表格已知$$alpha = e^{m_i - m^{new}_i}$$，sigma = d_i / d_new * alpha，所以原先的结果acc与sigma相乘，就可以得到第一部分的结果。综上acc就是更新了当前子块的结果矩阵$$O_i$$
```Python
p_scale = beta / d_new
# 第二部分中的P_{i,j} × e^{m̃_{i,j} - m^{new}_i} / d_new
p = numerators * p_scale[:, None]

# 第一部分中的e^{m_i - m^{new}_i}
sigma = d_i / d_new * alpha
acc = acc * sigma[:, None]
v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
p = p.to(q_ptr.dtype.element_ty)
acc += tl.dot(p, v)```
vLLM 中的 Triton Attention 后端
[图片]
vLLM 中的 Triton Attention 实现
```Python
class TritonAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
        self.unified_attention(...)
   ```
在 Triton Attention 后端中，forward 会负责组织 Attention 计算所需的数据，并调用 unified_attention(...) 完成核心计算。传入的基础张量包括 query，以及用于更新或配合缓存管理的 key、value。对于 decoder attention，真正参与注意力计算的历史 K/V 数据通常来自 kv_cache：代码会先将其拆分为 key_cache 和 value_cache，再将它们传入 unified_attention。
除了这些张量之外，Attention 计算还需要一组描述序列结构和缓存映射关系的元信息。这些信息保存在 attn_metadata 中，并在 forward 中提取后统一传给 unified_attention。其中比较关键的字段包括：
1. query_start_loc：用于描述每个请求在当前打包查询张量中的起始位置。在调用 unified_attention 时，它以 cu_seqlens_q 的形式传入，用于定位每个请求对应的 query 区间。
2. seq_lens：表示每个请求当前可用的总序列长度，也就是当前可参与注意力计算的 KV 长度。在调用 unified_attention 时，它以 seqused_k 的名字传入，用于确定每个请求实际可读取的历史 KV 范围。
3. max_query_len：当前批次中请求的最大 query 长度，用于 kernel 的执行配置与边界控制。
4. max_seq_len：当前批次中请求的最大序列长度，用于 kernel 的执行配置，以及注意力计算时的长度上界处理。
5. block_table：请求到 KV Cache 物理块的映射表，记录逻辑 block 到物理 block 的对应关系。Attention kernel 会借助它在分页的KV Cache 中定位每个请求对应的历史 K/V 数据。
暂时无法在飞书文档外展示此内容
q_start_loc 的值：
- q_start_loc[0] = 0：第一个请求永远从 0 开始。
- q_start_loc[1] = 3：因为请求 0 长度是 3，所以请求 1 从索引 3 开始。
- q_start_loc[2] = 5：因为前两个请求总长度是 $$3+2=5$$，所以请求 2 从索引 5 开始。
- q_start_loc[3] = 9：通常最后会多存一个总长度，方便计算最后一个请求的范围（即 q_start_loc[i+1] - q_start_loc[i] 得到第 $$i$$ 个请求的长度）。
block_table (块表)：
定义：记录每个请求（Request）逻辑上的块，对应物理显存中哪一个真实的块 ID。例子：
- Request A 长度为 7 个 Token。它需要 $$7 \div 4 = 1.75 \rightarrow$$ 2 个块。
- Request B 长度为 3 个 Token。它需要 $$3 \div 4 = 0.75 \rightarrow$$ 1 个块。
在显存池中，系统可能随机分配了物理块 ID 7, 12, 5，前两个块用于Request A。
slot_mapping
针对每一个 Token，直接计算出它的 KV 数据应该存放在物理显存池中的具体绝对索引位置，Slot_Index = 物理块 ID × block + 块内偏移

FlashAttentionMetadata结构中字段的详情见第7课-ModelRunner调用模型推理。
在处理大模型的变长序列时，vLLM 通常不会将输入补齐到统一长度，而是采用 Packed 的组织方式。这样一来，所有请求的 query token 会被拼接在同一个连续张量中，因此 q.shape[0] 表示当前批次中 query token 的总数。
对于 unified_attention 来说，query 方向的并行不是按单个序列分别精确切分后再逐个启动 kernel，而是基于总 token 数构造一个启动规模的上界。当前实现中的计算方式为：
```Python
total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs```
这里的 BLOCK_Q 表示一个程序实例在 query 维度上处理的 token 数，num_seqs 表示当前 batch 中的序列数量。之所以不直接计算 $$\sum_i ceil(query\_len[i] / BLOCK\_Q)，$$是因为那样需要在 CPU 侧显式恢复每个请求的 query_len，会引入额外开销。当前实现转而使用一个更容易计算的上界。这种做法虽然会引入少量“空转”的程序实例，但可以减少 CPU 端调度成本，从整体上提高执行效率。
在 unified_attention 函数中，Triton kernel 的 launch grid 被配置为二维结构：
- 第一维（Grid 0）：对应 query 分块并行。它并不是简单对应某一个固定序列，而是对应整个 packed query 空间上的分块，因此可以看作是 batch 维度与序列内 token 维度组合后的并行。
- 第二维（Grid 1）：对应 kv head 维度的并行。由于不同注意力头之间的计算彼此独立，因此可以沿这一维度并行分发计算任务，以提高 GPU 的利用率。
对应的代码形式如下：
```Python
def unified_attention(...):
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    
    # if batch contains a prefill
    if max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128:
        kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads,)] ```
需要注意的是，这里配置的是 Triton kernel 的二维 launch grid，而不是 CUDA 编程中手工组织的线程块层次。
因此，kernel_unified_attention_2d 的并行方式可以理解为：在 query 方向上，将 packed 后的 token 按 BLOCK_Q 进行分块；在另一维上，对不同的 kv heads 并行处理。最终，每个程序实例负责某个 query 子块 + 某个 kv head对应的一部分 Attention 计算。这样既适配了变长序列的 Packed 布局，也兼顾了较高的并行度。
所以，第一维 grid 表示某个序列内的某个 query block的全局编号。同时，kernel 内部会再把这个全局编号解析回具体序列和该序列内的局部block；如果某个实例落在上界多出来的空块上，就会直接返回，不参与实际计算。
暂时无法在飞书文档外展示此内容

```Python
@triton.jit
def kernel_unified_attention_2d(...):
    # 获取当前分块在全局 Q 范围内的 ID
    q_block_global_idx = tl.program_id(0)
    # 获取当前处理的注意力头 ID
    kv_head_idx = tl.program_id(1)
    # 正在处理的序列是哪个？
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)```
1. 根据全局索引定位序列：q_block_global_idx 表示当前程序实例在整个query中的第几个 query block。由于不同请求的 query 长度不同，各个序列对应的 block 数也不同，因此不能通过简单除法直接确定当前 block 属于哪个序列。为了解决这个问题，kernel 会调用 find_seq_idx，结合 query_start_len_ptr 来定位所属序列。
  例如，假设 3 个序列在 query 维度上被分成如下 block：
  序列0: [block0, block1, block2] ，3个有效block；
  序列1: [block3, block4] ，2个有效block；
  序列2: [block5, block6, block7, block8] ，4个有效block；
  那么：
  - q_block_global_idx = 0, 1, 2 时，属于 序列0
  - q_block_global_idx = 3, 4 时，属于 序列1
  - q_block_global_idx = 5, 6, 7, 8 时，属于 序列2
2. 计算局部块索引：在确定了所属序列 seq_idx 后，还需要进一步计算当前 query block 在该序列内部的相对位置。
暂时无法在飞书文档外展示此内容
```Python
@triton.jit
def kernel_unified_attention_2d(...):
    # 1. 计算当前序列的第一个分块在全局 Grid 0 中的起始索引
    # 注意：这里的逻辑需与 Grid 启动时的分块策略保持一致
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q
    
    # 2. 计算当前分块在所属序列内的相对索引（从 0 开始）
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    
    # 3. 获取当前请求序列在 Token 总数中的起始和结束位置
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    
    # 4. 计算当前请求的实际序列长度
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    
    # 5. 边界保护（Masking）：如果当前分块的起始位置已超出序列实际长度，则直接退出。
    # 这是为了处理由于向上取整（Ceil Division）导致的边缘 Program 实例。
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return```
  为了确定当前处理的 query block 在序列中的具体位置，首先需要定位该序列在全局 q-block 空间中的起始 block 索引，也就是 q_block_start_idx。这里之所以写成tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx，是因为 Grid 0 使用的并不是准确的分块数，而是一个带有上界性质的启动规模；因此，序列起始位置除了和前面累计的 token 数有关，还要额外加上前面序列带来的偏移项 seq_idx。
  在得到 q_block_start_idx 之后，通过计算 q_block_local_idx = q_block_global_idx - q_block_start_idx，就可以得到当前 block 在所属序列内的局部索引。比如，假设当前处理的全局 block 索引 q_block_global_idx = 4，而某个序列的起始 block 索引为 3，那么当前处理的就是该序列中的第 2 个 block，也就是局部索引 q_block_local_idx = 1。
  此外，还需要获取当前序列的 query 长度信息。做法是读取 query_start_len_ptr 中当前序列和下一序列的起始 token 位置，两者之差就是当前序列包含的 query token 数量，即：
```Python
cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index```
  这里的 cur_batch_query_len 表示的是当前序列的 query 长度，如果当前 block 的起始位置已经超出了该序列的实际 query 长度，kernel 就会直接返回，从而跳过这些由于上界式 grid 启动而产生的无效程序实例。
3. 在确定了所属序列以及当前 block 在序列内的位置之后，下一步就是计算 Query 矩阵的精确内存偏移。Query 张量在内存中的逻辑形状通常为[Total Tokens, Num Query Heads, Head Size]。因此，除了要定位当前处理的是哪一段 token 之外，还需要进一步确定它在 Heads 维度和 Dim 维度上的访问偏移，这样才能准确完成后续的 Query 数据加载。
暂时无法在飞书文档外展示此内容
  在此基础上，kernel 需要进一步定位当前 block 中每个元素在 token、head 和 dim 三个维度上的位置。对应代码如下：
```Python
# 1. 生成块内局部偏移张量
offs_m = tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, HEAD_SIZE_PADDED)

# 2. 计算当前 Token 在序列内的位置
# 注意：这里考虑了 GQA 情况，num_queries_per_kv 是 Query 头与 KV 头的比例
query_pos = q_block_local_idx * BLOCK_Q + (offs_m // num_queries_per_kv)

# 3. 计算全局内存偏移坐标
# query_offset_0: 对应全局 Token 维度的索引
query_offset_0 = cur_batch_in_all_start_index + query_pos
# query_offset_1: 对应特定的 Query Head 索引（将 KV 组索引映射回具体的 Query 头）
query_offset_1 = kv_head_idx * num_queries_per_kv + (offs_m % num_queries_per_kv)

# 4. 利用步幅（Stride）计算最终的物理内存指针偏移
# 采用广播机制将 1D 索引扩展为 2D 坐标映射
query_offset = (query_offset_0[:, None] * query_stride_0 +
                query_offset_1[:, None] * query_stride_1 + 
                offs_d[None, :])

# 5. 生成掩码（Mask）以确保访存安全
# dim_mask: 用于屏蔽为了对齐而填充的额外维度（当 HEAD_SIZE 不是 2 的幂次时）
dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
# query_mask_0: 用于屏蔽超出当前请求实际长度的 Token
query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
# query_mask_1: 用于确保 Query Head 访问不越界
query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)```
  为了确定当前需要加载的 $$Q$$ 分块在显存中的具体位置，我们首先需要定位该序列在全局 Token 序列中的起始偏移量。这个起始偏移量可以通过 cur_batch_in_all_start_index 获取，它表示在当前请求之前，所有序列一共占用了多少个 token，可以理解为当前序列在全局 token 轴上的基地址。
  在此基础上，为了获取特定序列中某个分块（由 q_block_local_idx 标识）的所有 Token 数据，我们采用的索引逻辑是：全局行索引 = 序列起始基址 + 块内局部偏移
  具体来说，索引过程逻辑上涉及三个维度：
  - Token 维度：通过 cur_batch_in_all_start_index + (q_block_local_idx * BLOCK_Q + 块内偏移) 定位到当前 block 中各个 token 在全局 packed token 空间中的位置。
  - Head 维度：结合当前程序实例对应的 kv_head_idx，再配合 num_queries_per_kv，映射到具体的 query head 索引。
  - Dim 维度：通过 offs_d 定位到单个 attention head 内部的特征维位置。
```Python
query_offset_0 = cur_batch_in_all_start_index + query_pos```
  对于q的其他维自然还有对num_head维度的索引，和dim维度的索引，最终讲这些索引按照多维的方式组织起来就是query_offset，另外还有它的mask防止在加载query矩阵时出现越界的情况。
  这一步给出了当前 Q 数据在 token 维度上的全局索引。除此之外，Q 的访存定位还需要结合另外两个维度：
  - query_offset_1：表示 num_query_heads 维度上的索引；
  - offs_d：表示 head_size 维度上的索引。
  例如，假设 num_queries_per_kv = 4、BLOCK_Q = 2，并且当前 q_block_local_idx = 0。如果 offs_m = [0,1,2,3,4,5,6,7]，那么：
  - offs_m // num_queries_per_kv = [0,0,0,0,1,1,1,1]
  - offs_m % num_queries_per_kv = [0,1,2,3,0,1,2,3]
  这说明 offs_m 同时编码了两层信息：整除部分表示当前是第几个 token，取余部分表示该 token 对应的第几个 query head。
  也就是说，前 4 个位置对应第 0 个 token 的 4 个 query heads，后 4 个位置对应第 1 个 token 的 4 个 query heads。因此，query_pos 用来确定 token 在序列中的位置，而 query_offset_1 用来确定具体访问的是哪个 query head。
4. 加载 Query 分块并定位物理块映射表：在完成索引计算后，kernel 会通过 tl.load 将当前 Program 实例对应的 Query 分块从全局显存加载到更快的片上访问层次中，以便后续参与 Attention 计算。与此同时，为了后续能够从分页管理的 KV Cache 中读取对应的 Key 和 Value，还需要先定位当前序列在 block_table 中对应的位置。
  - dim_mask[None, :]: 处理特征维度 (Head Size) 的对齐
  - query_mask_0[:, None]: 处理 Token 序列维度的边界 (Sequence Length)
  - query_mask_1[:, None]: 处理注意力头维度的边界 (Num Heads)
```Python
Q = tl.load(
    query_ptr + query_offset,
    mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    other=0.0,
)
block_table_offset = seq_idx * block_table_stride```
  而 block_table_offset = seq_idx * block_table_stride 则用于计算当前序列在 block_table 中对应行的基址偏移。后续 kernel 就可以基于这个偏移，从 block_table 中查出当前序列的逻辑 block 到物理 KV Cache block 的映射关系。
5. 初始化flashattention的最大值和归一化项目，也就是公式中的$$m_i$$和$$l_i$$以及累加结果acc。
```Python
M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)```
  这里：
  - M 用于记录当前已经处理过的所有 K 块中，每一行 query 对应的最大 attention score，也就是在线 softmax 里的 m_i；
  - L 用于记录归一化相关的累计量，也就是在线 softmax 里的 l_i；
  - acc 用于累计当前 query block 对应的输出结果，最终会得到这一块的 attention 输出。
6. 在正式计算注意力分数之前，我们需要根据当前分块的需求，动态加载对应的 Key 和 Value 矩阵分块（即公式中的 $$k_j$$ 和 $$v_j$$）。这些数据并非连续存储，而是按照 PagedAttention 机制，分布在由 块映射表（Block Table） 管理的非连续物理显存块中。
  总体来说，这一步的核心任务就是：借助 block_table，从分页式 KV Cache 中把当前 query block 需要访问的 K/V 数据准确取出来，为后续的 QK^T、softmax 和 PV 累加做好准备。
```Python
# 开始遍历该序列对应的所有历史 KV 块
for j in range(0, num_blocks):
    # 1. 查表：根据序列偏移和循环索引 j，获取当前逻辑块对应的物理块 ID
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

    # 2. 生成块内 Token 维度的偏移 (0 ~ BLOCK_SIZE-1)
    offs_n = tl.arange(0, BLOCK_SIZE)

    # 3. 计算物理显存中的 Value 块偏移
    # 维度通常为 [物理块 ID, 块内 Token, 头索引, 特征维度]
    v_offset = (physical_block_idx * stride_v_cache_0 +
                kv_head_idx * stride_v_cache_2 +
                offs_d[None, :] * stride_v_cache_3 +
                offs_n[:, None] * stride_v_cache_1)

    # 4. 计算物理显存中的 Key 块偏移
    # 注意：K 的索引布局通常经过优化（如转置），以适配高效的向量化矩阵乘法
    k_offset = (physical_block_idx * stride_k_cache_0 +
                kv_head_idx * stride_k_cache_2 +
                offs_d[:, None] * stride_k_cache_3 +
                offs_n[None, :] * stride_k_cache_1)```
7. 计算局部注意力分数 $$S = Q_i K_j^T$$，利用片上 SRAM 中加载好的 $$Q$$和 $$K$$ 分块进行矩阵乘法。为了防止数值过大导致梯度失效，通常会在此步骤乘以缩放因子 scale（即 $$1/\sqrt{d_k}$$）。
```Python
S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
S += scale * tl.dot(Q, K)```
8. 更新局部最大值$$m_{ij} = rowmax(S_{ij})$$
```Python
# 8. 计算局部注意力分数
# S 的形状为 [BLOCK_M, BLOCK_SIZE]，存储当前分块的注意力得分
S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
# tl.dot 执行矩阵乘法 Q * K.T，scale 为缩放因子
S += scale * tl.dot(Q, K)

# 9. 更新运行中的行最大值 
# M 是前 j-1 个块的运行最大值，tl.max(S, axis=1) 是当前第 j 个块的行最大值
m_j = tl.maximum(M, tl.max(S, axis=1))

# 特殊处理：防止负无穷导致后续计算异常，确保数值计算的鲁棒性
m_j = tl.where(m_j > float("-inf"), m_j, 0.0)```
9. 计算指数化分值与局部累加和：为了实现数值稳定的 Softmax，我们首先计算指数化的分值（即 Softmax 的分子部分），并统计其局部累加和
```Python
# 10. 计算当前块的指数化分值（未归一化概率）
# 利用当前行最大值 m_j 进行平移，防止 exp(S) 发生数值溢出
P = tl.exp(S - m_j[:, None])

# 计算局部归一化因子（Softmax 分母的局部贡献）
# 对每行指数化后的权重求和，得到当前 KV 块对应的局部累加值 l_j
l_j = tl.sum(P, axis=1)```
10. 首先，计算缩放因子 alpha = tl.exp(M - m_j)，该因子用于调整之前累积结果的权重，确保数值稳定性。具体来说，由于每一轮循环都可能产生新的局部最大值$$m_j$$，我们需要将之前累积的中间结果“对齐”到新的数值基准上，以确保全局一致性。
```Python
# 1. 计算重缩放因子 (Rescale Factor)
#     alpha = exp(旧最大值 - 新最大值)。由于新最大值 m_j >= 旧最大值 M，
# 因此 alpha <= 1，这一步能有效防止计算过程中的数值溢出。
alpha = tl.exp(M - m_j)

# 2. 对齐旧的累加结果
# 将之前步骤计算的未归一化加权和 acc 乘以 alpha，使其基准与当前块对齐
acc = acc * alpha[:, None]

# 3. 更新运行中的统计量
# 更新累加的指数和 L：先缩放旧和，再加入当前块的贡献 l_j
L = L * alpha + l_j
# 更新运行最大值 M 为当前最新的最大值
M = m_j```
11. 在完成所有 KV 块的遍历后，累加器 acc 中存储的是加权后的指数和。最后一步需要除以全局归一化因子 $$L$$（即所有分块指数和的总和），以得到最终的注意力输出。
```Python
# 12. 最终归一化：将累加结果除以全局累加和，得到符合概率分布的加权平均值
acc = acc / L[:, None]```
  总的来说，这里的Attention计算过程是一个非常标准的FlashAttention，没特别的地方，要说特别就是注意力分数和块表相结合了。
开启 triton 后端
```Python
"env": {
    // 可选：设置 CUDA_VISIBLE_DEVICES 控制 GPU 使用
    // "CUDA_VISIBLE_DEVICES": "0,1",
    // 开启triton后端
    "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
    // 开启 vLLM 调试日志
    "VLLM_LOGGING_LEVEL": "DEBUG",
    "VLLM_USE_TRITON_FLASH_ATTN": "True",
    "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
},```
如果用命令行开启服务就是：
```Bash
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3-0.6B --attention-backend TRITON_ATTN```
