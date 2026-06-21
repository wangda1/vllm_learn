# CUDA C++ 手写算子 vs Triton 写算子

本文以 `add.py`（Triton 实现的并行向量加法）为起点，对比「直接用 C++/CUDA 写算子」与「用 Triton 写算子」的核心区别，并进一步在 **warp 规约、block 规约、grid-stride loop** 三个典型并行模式上分别举例说明两者的差异。

---

## 0. 核心心智模型：thread-centric vs tile-centric

| 维度 | CUDA C++ | Triton |
| --- | --- | --- |
| 编程粒度 | **单个线程**（thread-centric）：你写的是「一个线程做什么」 | **一个程序实例处理一个 tile/block**（tile-centric）：你写的是「一个 program 处理一整块数据」 |
| 并行索引 | 手动算 `blockIdx * blockDim + threadIdx` | `tl.program_id(0)` + `tl.arange(0, BLOCK_SIZE)` 直接得到一段偏移向量 |
| 线程协作 | 显式 `__syncthreads()`、`__shfl_xor_sync` 等 | 编译器隐式处理，block 内同步基本不用手写 |
| 共享内存 / 寄存器分配 | 手动 `__shared__`、手动管理 bank conflict | 编译器自动决定 tile 放寄存器还是 shared memory |
| 向量化 / 访存合并 | 手动 `float4`、手动保证 coalescing | 编译器 pass 自动做 memory coalescing / vectorization |
| 编译链路 | `nvcc` → PTX/SASS，需写 host launch + binding | `@triton.jit` → Triton IR → LLVM IR → PTX，JIT 即时编译 |
| 调优 | 手动试 block size、tile size | `@triton.autotune` 自动搜索 |

一句话：**CUDA 让你管线程，Triton 让你管 tile**。Triton 把「block 内部如何切分到 thread、如何同步、如何用 shared memory」这层交给了编译器。

---

## 1. 向量加法：`add.py` 的 Triton 实现 vs C++ CUDA

### 1.1 Triton 版（即 `add.py`）

```python
@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                       # 当前 program 处理第几个 block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 一段连续偏移（向量）
    mask = offsets < N                           # 越界保护
    x = tl.load(X_ptr + offsets, mask=mask)      # 一次加载一整块
    y = tl.load(Y_ptr + offsets, mask=mask)
    tl.store(Z_ptr + offsets, x + y, mask=mask)  # 一次写回一整块

# launch：grid = (ceil(N / BLOCK_SIZE),)，BLOCK_SIZE 不是线程数而是「每个 program 处理的元素数」
vector_add_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x, y, z, N, BLOCK_SIZE=1024)
```

要点：
- 没有 `threadIdx`。`offsets` 是一个长度为 `BLOCK_SIZE` 的「逻辑向量」，编译器自己决定用多少线程、每线程几个元素去填这块。
- `BLOCK_SIZE` 是 `tl.constexpr`（编译期常量），编译器可据此做循环展开、死代码消除。
- 越界用 `mask` 处理，`tl.load(..., mask=mask)` 等价于 CUDA 里 `if (idx < N)` 的批量版本。

### 1.2 C++ CUDA 版（等价实现）

```cpp
// kernel：每个线程负责一个元素（grid-stride 见第 4 节）
__global__ void vector_add_kernel(const float* X, const float* Y, float* Z, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 手动算全局线程号
    if (idx < N) {                                    // 手动越界判断
        Z[idx] = X[idx] + Y[idx];                     // 一个线程只处理一个元素
    }
}

// host 端 launch
void vector_add(const float* x, const float* y, float* z, int N) {
    int block = 256;                                  // 线程数 = block size
    int grid  = (N + block - 1) / block;              // ceil
    vector_add_kernel<<<grid, block>>>(x, y, z, N);
}
```

还需要额外的 PyTorch 绑定样板（`torch.utils.cpp_extension` / pybind11），把 `at::Tensor` 的 `data_ptr<float>()` 取出来传给 kernel。

### 1.3 关键区别小结

| 关键点 | Triton (`add.py`) | C++ CUDA |
| --- | --- | --- |
| 一个执行单元处理的数据 | 一整块 `BLOCK_SIZE` 个元素 | 默认一个线程一个元素 |
| 越界处理 | `mask` 向量 | `if (idx < N)` 标量分支 |
| 访存合并 / 向量化 | 编译器自动 | 手动（`float4` 等） |
| 与 PyTorch 集成 | 直接吃 `torch.Tensor`，零绑定代码 | 需写 extension + pybind 绑定 |
| 开发成本 | 低，几十行 Python | 高，kernel + host + binding + 编译配置 |
| 极限可控性 | 受编译器约束 | 完全可控（手写 PTX 级优化） |

对「逐元素 element-wise」算子（如 add），两者性能几乎一致（都是 memory-bound，带宽打满即可），但 Triton 开发成本显著更低。差异在更复杂的、需要线程协作的算子上才真正体现出来 —— 下面三节就是这类场景。

---

## 2. Warp 规约（warp reduction）

**场景**：在一个 warp（32 线程）内部把 32 个值求和，不经过 shared memory，纯靠寄存器 + warp shuffle。

### 2.1 C++ CUDA：显式 `__shfl_down_sync`

```cpp
// 把一个 warp 内 32 个线程的 val 规约到 lane 0
__inline__ __device__ float warpReduceSum(float val) {
    // 蝶式折叠：offset = 16, 8, 4, 2, 1
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);  // 显式 warp 内寄存器交换
    }
    return val;  // lane 0 拿到全和
}
```

- 你必须**显式**知道 warp = 32、自己写折叠循环、自己管 mask `0xffffffff`（哪些 lane 参与）。
- 这是 CUDA 里最「贴硬件」的操作之一：直接操作 warp lane 间的寄存器，无需同步、无需 shared memory，最快。
- 易错点：mask 写错、divergent warp 下 `__shfl` 行为、`warpSize` 假设。

### 2.2 Triton：没有「warp」这个概念

Triton **不暴露 warp 层级**。你不会写 `__shfl`，而是直接对 tile 调用规约原语：

```python
@triton.jit
def reduce_kernel(X_ptr, Out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs, mask=offs < N, other=0.0)
    s = tl.sum(x, axis=0)          # 一行搞定整块规约
    tl.store(Out_ptr, s)
```

- `tl.sum`（以及 `tl.max` / `tl.min` 等）由编译器降级时**自动**生成最优的 warp shuffle + shared memory 组合。
- 开发者**看不到也管不到** warp 折叠的细节 —— `num_warps` 是个 launch 参数（autotune 可调），但折叠逻辑不用你写。
- 好处：代码极简、不会写错 mask；代价：无法手工微调折叠策略（比如自定义的非标准规约模式就比较别扭）。

### 2.3 区别

| | CUDA C++ | Triton |
| --- | --- | --- |
| warp 概念 | 一等公民，显式 `__shfl_*_sync` | 完全隐藏，无 warp 级 API |
| 规约写法 | 手写折叠循环 | `tl.sum(x, axis=...)` 一行 |
| mask / lane 管理 | 手动 | 编译器 |
| 可控性 | 极高（可做 segmented / 自定义规约） | 受限于内置 reduce 原语 |

---

## 3. Block 规约（block reduction）

**场景**：把一个 block（多个 warp，比如 256 线程）内所有值规约成一个。经典做法是「先 warp 规约 → 各 warp 结果写 shared memory → 再由一个 warp 做二次规约」。

### 3.1 C++ CUDA：warp 规约 + shared memory + `__syncthreads`

```cpp
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];          // 最多 32 个 warp，存各 warp 的部分和
    int lane = threadIdx.x % warpSize;           // warp 内 lane
    int wid  = threadIdx.x / warpSize;           // 第几个 warp

    val = warpReduceSum(val);                    // 1) 每个 warp 先各自规约
    if (lane == 0) shared[wid] = val;            // 2) 每个 warp 的结果写 shared
    __syncthreads();                             // 3) 显式同步，等所有 warp 写完

    // 4) 用第 0 个 warp 把各 warp 的部分和再规约一次
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;                                  // threadIdx.x == 0 拿到 block 全和
}
```

开发者要**手动**编排三层：lane/wid 索引计算、shared memory 分配与读写、`__syncthreads()` 屏障。漏写同步 = 数据竞争 + 偶发错误结果，是 CUDA 最常见的 bug 源之一。

### 3.2 Triton：还是 `tl.sum`，block 规约和 warp 规约长得一样

```python
@triton.jit
def block_reduce_kernel(X_ptr, Out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + row * n_cols + offs, mask=offs < n_cols, other=0.0)
    s = tl.sum(x, axis=0)           # 整个 tile（可能跨多个 warp）一次规约
    tl.store(Out_ptr + row, s)
```

- 对 Triton 来说，「warp 规约」和「block 规约」**没有区别** —— 都是 `tl.sum(tile)`。tile 跨多少 warp、要不要 shared memory 做二级规约，全由编译器根据 `BLOCK_SIZE` 和 `num_warps` 自动决定。
- 完全**没有** `__syncthreads()`、没有 shared memory 声明、没有 lane/wid 计算。
- 这就是 Triton 「tile-centric」的最大红利：消除了 CUDA 里最容易出错的「shared memory + 同步」编排。

### 3.3 区别

| | CUDA C++ | Triton |
| --- | --- | --- |
| 二级规约编排 | 手写：warp 规约 → shared → 再规约 | 编译器自动 |
| shared memory | 手动声明 + 读写 | 隐式 |
| 同步 | 手动 `__syncthreads()`（漏写就出错） | 隐式，无需手写 |
| 代码量 | 多（十几行模板） | 一行 `tl.sum` |
| warp/block 规约写法 | 完全不同 | **完全相同** |

> 实际工程里，`fused_rmsnorm.py`、`softmax.py` 这类需要「整行求和 / 求最大值」的算子，本质都是 block 规约。Triton 用一个 `tl.sum` / `tl.max` 就覆盖了，而 CUDA 要手写上面这套 `blockReduceSum`。

---

## 4. Grid-stride loop（网格跨步循环）

**场景**：数据量 `N` 远大于一次能启动的线程总数时，让每个线程「跨步」处理多个元素，从而用固定大小的 grid 覆盖任意大的数据，同时保证访存合并。

### 4.1 C++ CUDA：显式 stride 循环

```cpp
__global__ void add_grid_stride(const float* X, const float* Y, float* Z, int N) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;        // 全网格线程总数 = 跨步步长
    for (int i = idx; i < N; i += stride) {     // 每个线程处理 i, i+stride, i+2*stride, ...
        Z[i] = X[i] + Y[i];
    }
}
// launch 时 grid 可固定（如按 SM 数量设），不必随 N 线性增长
add_grid_stride<<<sm_count * 32, 256>>>(x, y, z, N);
```

- 经典「persistent / grid-stride」模式：grid 大小与 N 解耦，复用线程、减少 launch 开销、对 L2 cache 更友好。
- `stride = gridDim.x * blockDim.x` 必须手动算；循环边界、合并访存（连续线程访问连续地址）都要自己保证。

### 4.2 Triton：两种写法

**写法 A —— 一个 program 一个 tile（`add.py` 默认风格）**：把数据切成 `ceil(N/BLOCK_SIZE)` 个 tile，每个 program 处理一个 tile。这其实是把 grid-stride 的「跨步」交给了 **grid 维度**（program 数量随 N 增长），program 内部不循环：

```python
# 见 add.py：grid = (triton.cdiv(N, BLOCK_SIZE),)，每个 program 处理 BLOCK_SIZE 个元素
```

**写法 B —— program 内部显式跨步**（更接近 CUDA grid-stride，用于固定 grid / persistent kernel）：

```python
@triton.jit
def add_grid_stride(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)            # 类比 gridDim.x
    stride = num_programs * BLOCK_SIZE           # 跨步步长（以 tile 为单位）
    # 每个 program 跨步处理多个 tile
    for start in range(pid * BLOCK_SIZE, N, stride):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask)
        y = tl.load(Y_ptr + offs, mask=mask)
        tl.store(Z_ptr + offs, x + y, mask=mask)
```

### 4.3 区别

| | CUDA C++ | Triton |
| --- | --- | --- |
| 跨步单位 | 单个元素，`stride = gridDim*blockDim` | 一个 tile（`BLOCK_SIZE` 个元素），`stride = num_programs*BLOCK_SIZE` |
| 常见写法 | program 内 `for` 跨步循环 | 默认让 grid 维度承担跨步（写法 A），需要 persistent 时才显式循环（写法 B） |
| 访存合并 | 手动保证连续线程→连续地址 | tile 内天然连续，编译器保证合并 |
| 边界处理 | `i < N` 标量 | `mask` 向量 |
| 索引计算 | 手动 stride | `tl.num_programs` + tile 偏移 |

---

## 5. 总结：什么时候用哪个？

- **逐元素 / 标准规约 / 标准 GEMM、softmax、norm 类算子** → 优先 Triton：开发快、不易错、性能接近手写，且能用 `@triton.autotune` 自动调参。`add.py`、`softmax.py`、`fused_rmsnorm.py` 都属于这一类。
- **需要极限压榨硬件、非标准并行模式、特殊指令（如 mma/tensor core 的细粒度排布、自定义 warp specialization、复杂 shared memory 双缓冲）** → CUDA C++ 仍是上限更高的选择。
- **核心权衡**：Triton 用「编译器接管 thread/warp/shared-memory/同步」换取开发效率与正确性；CUDA 用「全手动」换取可控性与性能上限。

| 能力 | CUDA C++ | Triton |
| --- | --- | --- |
| Warp shuffle 细节 | ✅ 完全可控 | ❌ 隐藏 |
| Shared memory / 同步 | ✅ 手动 | ⚙️ 编译器 |
| 访存合并 / 向量化 | 🛠️ 手动 | ⚙️ 编译器自动 |
| 自动调优 | ❌ 自己搭 | ✅ `@triton.autotune` |
| 与 PyTorch 集成 | 🛠️ 需绑定 | ✅ 直接吃 Tensor |
| 开发成本 | 高 | 低 |
| 性能上限 | 更高 | 接近，通常足够 |
