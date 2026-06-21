#include <iostream>
#include <cuda_runtime.h>

// 简单的 CUDA 错误检查宏：任何调用失败都立即打印并退出，
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

    // 关键：Volta/Ampere 上单个 Block 的动态共享内存默认上限仅 48 KB，
    // 想用到 64 KB 必须显式 opt-in，否则 kernel 启动直接返回 invalid argument、
    // 根本不会运行（ncu 也就抓不到 kernel）。
    CUDA_CHECK(cudaFuncSetAttribute(low_occupancy_smem_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shared_mem_size));

    std::cout << "启动 Kernel，每个 Block 占用 64KB 共享内存..." << std::endl;
    low_occupancy_smem_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());       // 捕获启动期错误（配置非法等）
    CUDA_CHECK(cudaDeviceSynchronize());  // 捕获执行期错误
    std::cout << "运行结束。由于共享内存超限，单个 SM 只能驻留 1 个 Block，Occupancy 被卡死在 12.5%！" << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
