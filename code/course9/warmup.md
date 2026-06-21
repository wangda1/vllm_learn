# 预热（Warmup）的意义与本质

> 配套代码：[`cuda_graph.py`](./cuda_graph.py)（`bench()` 计时预热 + capture 前的 side-stream 预热）。

预热看着只是「先空跑几次」，但本质上它在做一件事：

> **把「只在第一次执行才发生、稳态不会重复」的一次性开销，以及「需要跑一会儿才稳定」的瞬态，提前在「不计时 / 不捕获 / 不对外服务」的阶段消化掉**，让后续的测量、capture、或真实请求面对一个 warm、确定、可复现的系统状态。

---

## 一、预热消除的「一次性开销」（lazy init / 延迟初始化）

GPU 软件栈里大量东西是**用到才初始化**的，第一次调用奇慢，之后不再发生：

| 一次性开销 | 第一次会发生什么 | 预热后 |
|---|---|---|
| **CUDA context** | 首个 CUDA 调用才创建上下文 | 已建好 |
| **cuBLAS / cuDNN handle** | 首次 matmul/conv 才加载库、建 handle、把库代码搬进显存 | 已加载 |
| **kernel 加载 / JIT** | PTX→SASS 编译、module load 进 GPU | 已驻留 |
| **显存分配器** | 首次分配走 `cudaMalloc`（慢且**同步**）；PyTorch caching allocator 之后从内存池复用 | 池子已就绪，后续分配几乎零成本 |
| **算法选择 (autotune)** | cuBLAS/cuDNN 首次会试跑多个候选算法挑最快的（`torch.backends.cudnn.benchmark` 模式尤甚） | heuristic 已定 |
| **torch.compile** | 首次调用才触发 Dynamo trace + Inductor 编译 | 已编译缓存 |

## 二、预热让系统进入「稳态」（steady state）

不是一次性、而是「需要跑一会儿才稳定」的瞬态：

- **GPU 时钟频率 / boost**：刚开始 GPU 在低频，负载持续才 boost 到高频（反之也可能热降频）。不预热，前几次迭代频率没稳，测出来偏慢。
- **缓存预热**：L2 cache、指令缓存被填充。

## 三、对 CUDA Graph：预热是「铁律」而非可选优化

CUDA Graph 的 capture 要求图里**只有纯异步设备侧操作**（见 TUTORIAL 第 3 章铁律一）。而上面那些一次性开销——cuBLAS 懒加载、`cudaMalloc`、autotune 试跑——很多是**同步的**。若不预热就直接 capture：

1. 轻则把这些「非纯计算」的初始化操作错误地**录进图**，污染重放；
2. 重则因含同步操作**直接让 capture 失败**（`operation not permitted when stream is capturing`）。

所以 `cuda_graph.py`（第 57–63 行）特意在 **side stream** 上预热：

```python
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())   # side stream 等当前流
with torch.cuda.stream(s):
    for _ in range(num_warmup):
        compute()                            # 在 side stream 上把懒加载全触发掉
torch.cuda.current_stream().wait_stream(s)   # 当前流再等 side stream
torch.cuda.synchronize()
```

既把懒加载全部触发掉，又**不污染默认流**（`torch.cuda.graph` 在自己的 stream 上 capture）。预热完 capture 到的就是一段干净的、纯计算的 kernel 序列。

## 四、对 benchmark：预热保证测到的是稳态

`cuda_graph.py` 的 `bench()`（第 26–35 行）先 `warmup` 再 `synchronize` 再计时，否则第一次迭代把 cuBLAS 初始化、`cudaMalloc`、频率爬升全算进去，平均值被严重拉偏——测出来的不是算子真实吞吐，而是「冷启动 + 算子」的混合值。

```python
def bench(fn, iters, warmup):
    for _ in range(warmup):   # 消化一次性开销 + 进入稳态
        fn()
    torch.cuda.synchronize()  # 等预热真正跑完
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()  # 等被测迭代真正跑完
    return (time.perf_counter() - start) / iters * 1e3
```

## 五、推理服务里的「warmup 请求」是同一回事

vLLM / TGI 这类服务启动时先打几条假请求，目的完全一致：把所有 kernel 编译、cuBLAS 初始化、allocator 预分配、**CUDA Graph capture、padding 分桶图的预捕获**全部在启动阶段做完。这样**第一个真实用户请求**不吃这些冷启动延迟，避免首 token 延迟出现尖刺（cold-start latency spike）。

---

## 一句话本质

预热 = 把「第一次才付、稳态不再付」的冷启动成本和未达稳态的瞬态，从「我们关心的阶段」（计时 / capture / 服务真实请求）里挪出去，**提前付清**，使后续行为稳定、可复现、可被干净地捕获。
