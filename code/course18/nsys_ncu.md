先给你一句话总结：**nsys看“系统有没有并行起来、谁在拖时间”；ncu看“单个kernel为什么慢、瓶颈在计算还是访存”**。下面直接给你**必看指标+最简demo+怎么看结果**，照着跑一遍就会。

---

## 一、NSYS（系统级）关键指标怎么看
### 1）时间线（Timeline，最核心）
打开 `nsys-ui xxx.nsys-rep`，重点看三行：
- **CPU Threads**：是否在频繁 `cudaSync`/`cudaMalloc`（红色=阻塞）
- **CUDA Kernels**：kernel 是否**连续无空隙**（有空隙=CPU跟不上或同步等待）
- **Memory Transfers**：H2D/D2H 是否和 kernel **重叠执行**（不重叠=带宽浪费）


### 2）终端汇总（--stats=true）必看表
- **cuda_gpu_kern_sum**：按**总耗时排序**，找到 Top1 热点 kernel（下一步给 ncu 分析）
- **cuda_api_sum**：`cudaLaunchKernel`/`cudaMemcpy` 耗时占比（>30% 说明CPU侧瓶颈）
- **cuda_gpu_mem_time_sum**：H2D/D2H 总耗时（看是否是带宽瓶颈）

### 3）核心指标速查
- **GPU Utilization**：>70% 正常；<50% 说明**CPU-GPU 没并行**
- **Kernel Gap**：相邻 kernel 间隔 >10us → **CPU 提交慢或同步过多**
- **Memcpy Overlap**：拷贝与 kernel 重叠率 <50% → **数据搬运阻塞计算**

---

## 二、NCU（Kernel级）关键指标怎么看
打开 `ncu-ui xxx.ncu-rep`，先看**Speed of Light（屋顶线）**，再看**Warp Stall**。

### 1）Speed of Light（一眼定瓶颈）
- **Compute Throughput（SM）**：>60%=计算饱和；<30%=算力闲置
- **Memory Throughput**：>70%=访存打满；<40%=带宽浪费。注意 SOL 这行的 Memory% 是各存储子系统（DRAM/L1/L2/共享内存等）里**取最大值**，不一定就是 DRAM 带宽——具体是哪一级要看下方 Memory Workload Analysis
- **结论**：
  - **Memory Bound**：DRAM%高、SM%低 → 优化访存（合并访问、L1/共享内存）
  - **Compute Bound**：SM%高、DRAM%低 → 优化算法/用 TensorCore
  - **Latency Bound**：两者都低 → 优化调度（Occupancy、减少依赖）

### 2）Warp Stall（最常见瓶颈）
你之前截图里的关键项：
- **Stall MIO Throttle**：**最常见** → 内存/特殊数学指令太多，流水线被压限流
- **Long Scoreboard**：指令依赖链太长，warp 等数据
- **No Eligible Warps**：Occupancy 太低，没足够 warp 可调度
- **Branch Divergence**：分支发散，warp 内线程执行路径不同

### 3）核心指标速查
- **Occupancy（SM 占用率）**：>50% 合格；<30% 严重偏低（调 block 大小、共享内存/寄存器）
- **L1/L2 Hit Rate**：L1>60%、L2>80% 正常；低 → 访存模式差（复用差、随机访问）
- **Warp Execution Efficiency**：>90% 优秀；<70% 分支发散严重

---

## 三、推荐Demo：矩阵乘法（最经典、瓶颈明显）
### 1）Demo 代码（保存为 `matmul_demo.py`）
```python
import torch
import torch.cuda.nvtx as nvtx

def main():
    # 造一个中等规模矩阵（易出瓶颈）
    N = 2048
    A = torch.randn(N, N, device="cuda")
    B = torch.randn(N, N, device="cuda")

    # 预热（ncu 用 -s 跳过这段冷启动 kernel）
    for _ in range(10):
        _ = A @ B
    torch.cuda.synchronize()

    # 正式跑：用 NVTX 打标，nsys 时间线里能直接定位这段
    nvtx.range_push("matmul_loop")
    for _ in range(20):
        _ = A @ B
    torch.cuda.synchronize()
    nvtx.range_pop()

if __name__ == "__main__":
    main()
```
> 说明：原来预分配的 `C = torch.empty(...)` 没有意义——`C = A @ B` 每次都会新分配输出张量并重新绑定，预分配的缓冲区不会被复用。这里直接去掉。NVTX `range_push/pop` 是为了让下面 `-t nvtx` 真正采到东西。

### 2）NSYS 采集（系统级）
```bash
# 采集并生成报告（含CUDA+NVTX+系统调用）
nsys profile -t cuda,nvtx,osrt -o matmul_nsys --stats=true python matmul_demo.py

# 打开GUI看时间线
nsys-ui matmul_nsys.nsys-rep
```
> 想在时间线里看到真正的「GPU Utilization / SM Active」采样行，需要加 `--gpu-metrics-device=all`（**nsys ≥ 2021.2 才支持**；本机的 2020.4.3 没有这个选项，此时时间线上的「利用率」是按 kernel 覆盖时间估算的，不是硬件计数器）。
>
> 同样的流程也适用于本课的真实负载——把上面命令换成 `nsys_offline.sh` 里的 `vllm bench latency ...` 即可。

**看什么**：
- Kernels 是否密集无空隙
- H2D 是否和 matmul kernel 重叠
- 有没有频繁同步（`cudaDeviceSynchronize` / `cudaStreamSynchronize`，旧文档里常简写成 cudaSync）

### 3）NCU 采集（Kernel级，精准抓 matmul）
```bash
# 只抓名字匹配 gemm 的 kernel，跳过前 10 个预热launch，采 1 次，全指标
# 注意：是 -k regex:gemm（不是 --kernel-regex，旧版本的写法在现代 ncu 已移除）
sudo ncu -k regex:gemm -s 10 -c 1 --set full -o matmul_ncu python matmul_demo.py

# 打开GUI分析
ncu-ui matmul_ncu.ncu-rep
```
> 几个坑：
> - **`-k regex:gemm`** 才是现代 Nsight Compute（本机 2022.4）的正则过滤写法；fp32 的 `torch.randn` 会走 `*sgemm*` kernel，能被 `gemm` 匹配到。
> - **`-s 10`** 跳过预热，避免把冷启动 kernel 当成稳态来分析。
> - **`--set full`** 会对同一个 kernel 反复 replay 采全部 section，很慢；只想快速定位瓶颈可用 `--set basic` 或只采 SOL：`--section SpeedOfLight`。
> - ncu 读硬件性能计数器通常需要 **root 或开放 GPU 计数器权限**（否则报 `ERR_NVGPUCTRPERM`），所以加了 `sudo`。
**看什么**：
- Speed of Light：matmul 通常是 **Compute Bound**（SM%高、DRAM%中低）
- Warp Stall：看是否有 **MIO Throttle** 或 **Branch Divergence**
- Occupancy：理想 >70%

---

## 四、快速分析流程（记住这5步）
1. **nsys**：看时间线 → 找热点 kernel + 检查并行度
2. **nsys stats**：确认 Top1 kernel（如 gemm）
3. **ncu**：针对该 kernel 跑 full 分析
4. **屋顶线**：判断是 **Memory/Compute/Latency Bound**
5. **Warp Stall**：定位具体阻塞原因（MIO/Scoreboard/Branch）

---

## 五、进阶 Demo：直接抓 vLLM 里的真实 kernel

matmul 是干净的教学例子；下面是**本课真实负载**（`vllm bench latency` 跑 Qwen3-0.6B）的抓法。和 matmul 最大的不同：**vLLM 默认开 CUDA Graph，会让 profiler 抓不到 kernel 名字**，必须先处理掉这一点。

### ⚠️ 0）先解决「CUDA Graph 导致 kernel 名全是 [Unknown]」

直接对当前 `nsys_offline.sh` 的产物做 kernel 汇总，会看到这样的结果（实测）：

```
 Time(%)  Total Time(ns)  Instances        Name
   99.8     148,672,872       417     [Unknown]          ← 99.8% 时间全在这，没法分析
    0.0          56,765         3     cutlass_80_wmma_tensorop_bf16_...gemm...   ← 只漏出一个 GEMM
```

99.8% 的时间落在 `[Unknown]`，因为这些 kernel 是被 **CUDA Graph 整体重放**的，nsys 默认拿不到图内 kernel 的名字。两种解法：

- **方案 A（推荐做 kernel 分析）**：跑 vLLM 时加 `--enforce-eager`，关掉 CUDA Graph，让 kernel 逐个正常 launch、名字可见。代价是性能不代表生产态。
- **方案 B（想保留 Graph 的真实性能）**：给 nsys 加 `--cuda-graph-trace=node`，让它展开图内节点、还原 kernel 名（nsys ≥ 2020.3 支持）。注意这会显著增大报告体积。

> 这正是 `nsys_offline.sh` 里那行 `--cuda-graph-trace=node` 的作用——去掉它，下游 kernel 名就会塌成 `[Unknown]`。

### 1）NSYS：先定位 vLLM 的热点 kernel

```bash
export CUDA_VISIBLE_DEVICES=6
nsys profile -t cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    -o vllm_nsys --force-overwrite=true \
    vllm bench latency --model /home/eechengyang/CX/model/Qwen3-0.6B \
    --input-len 512 --output-len 8 --batch-size 16 \
    --num-iters-warmup 5 --num-iters 1 --enforce-eager

# 从报告里拉出按总耗时排序的 kernel 列表，找 Top1
nsys stats --report gpukernsum vllm_nsys.qdrep | head -20
```
> **报告名随版本变**：本机 nsys 2020.4 用的是 `gpukernsum`/`cudaapisum`/`gpumemtimesum`；**nsys ≥ 2021.2 才改叫** `cuda_gpu_kern_sum`/`cuda_api_sum`/`cuda_gpu_mem_time_sum`（即本文第一节用的那套名）。报错说找不到 report 时先核对版本。

vLLM 解码阶段典型的热点 kernel 大类：
- **Attention**：`flash_fwd_*`（FlashAttention 后端）或 `paged_attention_v1/v2`（vLLM 自带 PagedAttention 后端）
- **GEMM**（QKV / o_proj / MLP）：`cutlass_*gemm*`、`*sm80*gemm*`——本机实测漏出的就是 `cutlass_80_wmma_tensorop_bf16_...gemm`
- **vLLM 自定义算子**：`rms_norm_kernel`、`silu_and_mul_kernel`、`rotary_embedding_kernel`

### 2）NCU：针对上一步选出的 kernel 名做精分析

把上一步 `gpukernsum` 里的 Top1 名字（截一段稳定子串）填进 `-k regex:`：

```bash
# 例：分析 attention。-s 跳过 warmup+图捕获阶段的 launch，-c 1 只采一个稳态实例
export CUDA_VISIBLE_DEVICES=6
sudo -E ncu -k regex:flash_fwd -s 20 -c 1 --set full \
    -o vllm_attn_ncu \
    vllm bench latency --model /home/eechengyang/CX/model/Qwen3-0.6B \
    --input-len 512 --output-len 8 --batch-size 16 \
    --num-iters-warmup 5 --num-iters 1 --enforce-eager

ncu-ui vllm_attn_ncu.ncu-rep
```
> vLLM 专属的坑：
> - **必须 `--enforce-eager`**：ncu 不能 profile CUDA-Graph 重放出来的 kernel，开着 Graph 会直接抓不到。
> - **`-s` 要给够大**：vLLM 启动有大量预热/捕获 launch，skip 太小会采到非稳态 kernel；先用 nsys 时间线大致估算需要跳过多少个。
> - **`sudo -E`**：保留 `CUDA_VISIBLE_DEVICES` 等环境变量（裸 `sudo` 会丢环境，跑到错误的卡上）。
> - **范围尽量收窄**：`--set full` 对每个匹配 kernel 反复 replay，vLLM 里匹配项一多就极慢；优先 `--set basic` 或先只采 `--section SpeedOfLight` 定瓶颈类型。

### 3）怎么读 vLLM 的结果
- **Attention（decode 阶段）几乎都是 Memory/Latency Bound**：batch、KV 长度小，算术强度低，DRAM% 或 stall（Long Scoreboard）高、SM% 低——别指望它 Compute Bound。
- **GEMM 在 prefill / 大 batch 下偏 Compute Bound**：看 Tensor Core 利用率（SOL 里的 `SM (TC)` 或 pipe utilization）是否吃满。
- 两个阶段瓶颈不同，这也是 vLLM 要把 **prefill 和 decode 分开调度**的根因。

---

要不要我把上面的关键指标整理成一份一页式速查表（含阈值、判断标准和优化方向），你直接对照就能用？