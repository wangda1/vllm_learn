这是一个非常经典的 GPU 性能调优问题。要理解“一个 SM 可供调度的 Warp 是什么决定的”，以及如何“故意”复现一个由于资源占用过高导致 **Occupancy（SM 占用率）极低**的场景，我们需要从 GPU 的硬件限制和资源分配机制说起。

---

### 一、 一个 SM 可供调度的 Warp 是由什么决定的？

在 GPU 硬件中，一个 SM（流式多处理器）能同时驻留（Active）并供硬件调度器（Warp Scheduler）调度的最大 Warp 数量，是由**硬件物理上限**和**软件资源占用**共同决定的。

#### 1. 硬件物理上限（Hard Limits）
每代 GPU 架构都有其绝对的物理上限。一旦达到这些上限，即使其他资源再空闲，SM 也无法再容纳更多的 Warp。
> **重要：Ampere 内部分两档，别把 RTX 3090 和 A100 混为一谈！**
> 同属 Ampere，但 compute capability 不同，单 SM 的 Warp/线程/共享内存上限差很多：
>
> | 设备 | Compute Cap | 最大线程/SM | 最大 Warp/SM | 共享内存/SM(opt-in 上限) |
> |---|---|---|---|---|
> | **A100**(GA100) | 8.0 | 2048 | **64** | 164 KB |
> | **RTX 3090**(GA102) | 8.6 | 1536 | **48** | 100 KB |
>
> 本机是 **RTX 3090(sm_86)**，所以下文复现用的就是 **48 warps/SM、100 KB smem/SM** 这一档。
> （以下"以 A100 为例"的 64 warps 数字仅在 compute 8.0 上成立。）

以 **A100（compute 8.0）** 为例，单个 SM 的硬件上限为：
*   **最大驻留 Warp 数**：**64 个 Warp**（即 2048 个线程）。
*   **最大驻留 Block 数**：**32 个 Blocks**。
*   **寄存器堆（Register File）总大小**：**256 KB**（共 65,536 个 32-bit 寄存器）。
*   **共享内存（Shared Memory）最大容量**：**164 KB**（默认可配置为 100 KB 左右给单个 Block 使用）。

#### 2. 软件资源占用（限制因子）
在实际运行中，SM 能驻留多少个 Warp，取决于你的 Kernel 占用了多少资源。这被称为 **3 大限制因子**：
1.  **Block 尺寸限制**：你配置的 `blockDim` 大小。
2.  **寄存器限制**：每个线程占用的寄存器数量（由编译器决定，或通过 `__launch_bounds__` 限制）。
3.  **共享内存限制**：每个 Block 申请的 Shared Memory 字节数。

**SM 最终能驻留的 Warp 数，是上述所有限制条件计算出的“交集”（最小值）。**

---

### 二、 如何复现“Occupancy 极低”的场景？

**Occupancy（占用率）** 的定义是：
$$\text{Occupancy} = \frac{\text{SM 实际驻留的 Warp 数}}{\text{SM 硬件支持的最大 Warp 数 (64)}}$$

我们要复现 **Occupancy 极低**（例如只有 12.5% 甚至更低），可以通过**故意让某一种资源占用爆表**，从而强行卡死 SM 能够驻留的 Block 数量。

下面我们提供两个可编译、可运行的 CUDA 复现用例。

#### 场景 A：通过“共享内存占用过高”卡死 Occupancy（最经典）

*   **原理**：单个 SM 最多支持 100 KB 共享内存（A100/3090 同为 100 KB opt-in 上限）。如果我们让每个 Block 申请 **64 KB** 的共享内存：
    *   因为 $64\text{ KB} \times 2 = 128\text{ KB} > 100\text{ KB}$，所以**一个 SM 物理上只能同时容纳 1 个 Block**。
    *   如果我们的 Block 大小设为普通的 **256 个线程（8 个 Warps）**。
    *   此时，SM 实际驻留的 Warp 数就只有这 1 个 Block 带来的 **8 个 Warps**。
    *   **最终 Occupancy**取决于该 SM 最大 Warp 数：
        *   **RTX 3090（本机，48 warps/SM）**：$8 / 48 = \mathbf{16.7\%}$（已用 Occupancy API 实测，见第三节）。
        *   **A100（64 warps/SM）**：$8 / 64 = \mathbf{12.5\%}$。
    *   ⚠️ **关键坑**：Volta/Ampere 上单 Block 动态共享内存默认上限仅 **48 KB**，想用 64 KB **必须**先 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)` 显式 opt-in，否则 kernel 启动直接返回 `invalid argument`、**根本不会运行**（且若不检查返回值会静默失败，ncu 也抓不到 kernel）。下面代码已包含这一步。

##### 复现代码 (`low_occupancy_smem.cu`)：
```cuda-cpp
#include <iostream>
#include <cuda_runtime.h>

// 简单的 CUDA 错误检查宏：任何调用失败立即打印并退出，
// 避免“kernel 没跑成功却以为成功”这种静默错误。
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__         \
                      << " -> " << cudaGetErrorString(_e) << std::endl;       \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// 每个 Block 申请 64 KB 的动态共享内存
__global__ void low_occupancy_smem_kernel(float *out, const float *in, int n) {
    extern __shared__ float s_data[]; // 动态共享内存

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        // 故意读写共享内存，防止被编译器优化抹去
        s_data[tid] = in[idx];
        __syncthreads();
        out[idx] = s_data[tid] * 2.0f;
    }
}

int main() {
    const int N = 1048576;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Block 设为 256 线程 (8 个 Warps)
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 故意申请 64 KB 的共享内存 (64 * 1024 字节)
    size_t shared_mem_size = 64 * 1024;

    // 关键：>48KB 动态共享内存必须显式 opt-in，否则启动失败、kernel 不运行。
    CUDA_CHECK(cudaFuncSetAttribute(low_occupancy_smem_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem_size));

    std::cout << "启动 Kernel，每个 Block 占用 64KB 共享内存..." << std::endl;
    low_occupancy_smem_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());       // 捕获启动期错误（配置非法等）
    CUDA_CHECK(cudaDeviceSynchronize());  // 捕获执行期错误
    std::cout << "运行结束。由于共享内存超限，单个 SM 只能驻留 1 个 Block，Occupancy 极低！" << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
```
> 编译（本机 RTX 3090 = sm_86）：`nvcc -arch=sm_86 -Xptxas -v low_occupancy_smem.cu -o low_occupancy`

---

#### 场景 B：通过“寄存器占用过高”卡死 Occupancy

*   **原理**：Ampere 单个 SM 共有 65,536 个寄存器。如果我们写一个极其复杂的 Kernel，或者使用编译器指令，让每个线程强行占用最大上限 **255 个寄存器**：
    *   一个 Block 如果有 512 个线程，需要的寄存器数就是 $512 \times 255 = 130,560$ 个，这远远超过了单个 SM 的 65,536 个。
    *   因此，硬件调度器会进行限制：单个 SM 只能容纳 $65,536 / 255 \approx 256$ 个线程（即 **8 个 Warps**）。
    *   **最终 Occupancy**同样被卡死在 8 warps：本机 RTX 3090 为 $8/48=\mathbf{16.7\%}$，A100 为 $8/64=\mathbf{12.5\%}$。

##### 复现方法：
在 CUDA 中，我们可以通过在 Kernel 声明前加上 `__launch_bounds__`，或者在编译时加上 `-maxrregcount` 参数，来故意制造或观察寄存器压力。

例如，在编译时强行限制或观察寄存器：
```bash
# 编译并输出 GPU 资源占用详情（包含每个 Kernel 占用了多少个寄存器）
nvcc -Xptxas -v low_occupancy_smem.cu -o low_occupancy
```
> 注意：旧资料里常见的 `-Xptxas -v,-abi=no` 在较新 CUDA（如 CUDA 10.1）上会报
> `ptxas error : Invalid value 'no' for option -abi`，因为这些版本的 ptxas 已无 `-abi=no` 选项。
> 只用 `-v` 即可打印寄存器/smem/cmem 用量；想看寄存器溢出再加 `-warn-spills`。
控制台会输出类似：
`ptxas info    : Used 255 registers, 1024 bytes smem...` 
这能让你在不运行程序的情况下，直接通过编译器报告预测出极低的 Occupancy。

---

### 三、 用 ncu 验证 Occupancy（实测）

理论推算之后，还得用工具实测确认。这里有两条互补的路子：**ncu**（需要 root，给"理论 + 实际达到"两个值）和 **CUDA Occupancy API**（不需要 root，只给理论值，适合 CI/无权限环境）。

#### 0. 环境准备（本机多版本 CUDA）

`ncu` 在 CUDA 12.0 里（CUDA 10.1 没有），先切环境：

```bash
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
ncu --version   # 本机为 Nsight Compute 2022.4
# 按 sm_86 重新编译，确保跑在 3090 原生架构上
nvcc -arch=sm_86 -Xptxas -v low_occupancy_smem.cu -o low_occupancy
```

#### 1. 用 ncu 抓 Occupancy section（需要 root）

```bash
sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    ncu --section Occupancy --section LaunchStats -c 1 ./low_occupancy
```

* `-c 1` 只采第一个 kernel（本例就一个），避免重复 replay。
* 报告里重点看这几行（数值为 RTX 3090 预期）：
    * `Theoretical Occupancy        16.67 %`  ← 与本文推算一致
    * `Achieved Occupancy          ~16 %`     ← 实际跑出来的，略低于理论值正常
    * `Block Limit Shared Mem        1 block` ← **限制因子就是它**：smem 把每 SM 卡到 1 个 block
    * `Block Limit Warps / Registers` 会明显大于 1，说明瓶颈不在它们。
* 谁是限制因子，就看 `Block Limit *` 里**最小的那个**——本例 Shared Mem = 1 最小，故 occupancy 被共享内存卡死（对应场景 A）。

> **常见报错 `ERR_NVGPUCTRPERM`**：非 root 跑 ncu 读硬件计数器会报
> `The user does not have permission to access NVIDIA GPU Performance Counters`，
> 表现为 `==WARNING== No kernels were profiled.`。解决：用 `sudo` 跑，或让管理员把
> `NVreg_RestrictProfilingToAdminUsers=0` 写进 nvidia 驱动模块参数后重载驱动。

> **另一个静默坑（本例真实踩到）**：如果 kernel 启动本身就失败（如忘了给 >48KB 动态
> smem 做 `cudaFuncSetAttribute` opt-in），程序照样打印"运行结束"，但 ncu 报
> `No kernels were profiled`——因为压根没有 kernel 真正跑起来。先确保 `./low_occupancy`
> 自身不报 CUDA error（已加 `CUDA_CHECK`），再去 profile。

#### 2. 无 root 兜底：CUDA Occupancy API 算理论值

不想搞 root，可以直接在程序里调 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 自报家门，**无需任何权限**：

```cpp
int blocks = 0;
cudaFuncSetAttribute(low_occupancy_smem_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks, low_occupancy_smem_kernel, /*blockSize=*/256, /*dynSmem=*/64*1024);

cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
int maxWarps    = p.maxThreadsPerMultiProcessor / p.warpSize;   // 3090: 48
int activeWarps = blocks * (256 / p.warpSize);                  // 1 * 8 = 8
printf("occupancy = %.1f%%\n", 100.0 * activeWarps / maxWarps); // 16.7%
```

本机实测输出（RTX 3090）：

```
GPU: NVIDIA GeForce RTX 3090 (sm_86)
maxThreads/SM=1536  maxWarps/SM=48  smemOptin/SM=100KB
per-block: 256 threads (8 warps), dyn smem=64KB
=> max active blocks/SM = 1
=> active warps/SM = 8 / 48
=> THEORETICAL OCCUPANCY = 16.7%
```

这与 ncu 的 Theoretical Occupancy 对得上，也印证了第二节的推算（只是把 A100 的 12.5% 换成 3090 的 16.7%）。

#### 3. 限制因子速查

ncu 的 `Block Limit *` 四个值里取最小，就是当前的 Occupancy 瓶颈：

| Block Limit 最小项 | 含义 | 对应本文场景 | 调优方向 |
|---|---|---|---|
| Shared Mem | 共享内存用太多 | 场景 A | 降低每 block smem / 用更小 tile |
| Registers | 寄存器用太多 | 场景 B | `-maxrregcount` / `__launch_bounds__` 降寄存器 |
| Warps | block 尺寸/数量受 SM 硬上限 | — | 调 blockDim |
| Blocks | 达到每 SM 最大 block 数 | — | 增大 block（减少 block 数） |

通过 ncu 实测 + Occupancy API 兜底，你可以非常精准地定位到阻碍 GPU 满载计算的物理瓶颈，而不必靠猜。