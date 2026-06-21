# Course 9 教学版：CUDA Graph 与 torch.compile

> 本文是 [`DOC.md`](./DOC.md) 的**教学重构版**：把零散的参考资料整理成可按章节讲授的课程，
> 并补齐 `DOC.md` 几乎没展开的 **torch.compile** 部分，以及它与 CUDA Graph 的协同关系。
>
> - `DOC.md`：完整参考手册（CUDA Graph 原理 + vLLM v1 CUDAGraph 内部实现逐行解析）。
> - 本文 `TUTORIAL.md`：分章节、带「原理 + demo + 练习」的教学主线。
>
> **配套代码**（均可 `python xxx.py` 直接跑，含 eager 基线与正确性校验）：
> `cuda_graph.py` · `cuda_graph_input.py` · `cuda_graph_blocking.py` · `cuda_graph_padding.py` · `cuda_graph_gpt.py` · **`torch_compile_demo.py`**（新增）
>
> **环境**：PyTorch ≥ 2.4，CUDA GPU。多卡机器建议 `CUDA_VISIBLE_DEVICES=<空闲卡>` 运行。

---

## 课程地图

| 章节 | 主题 | 核心问题 | 配套代码 |
|------|------|----------|----------|
| 第 0 章 | 为什么需要这些优化 | GPU 没满，CPU 为啥忙？ | — |
| 第 1 章 | CUDA Graph 原理 | 录制→重放怎么省开销 | — |
| 第 2 章 | 第一个 CUDA Graph | 怎么写、怎么算加速比 | `cuda_graph.py` |
| 第 3 章 | 三大铁律 | 异步 / 固定地址 / 静态形状 | `cuda_graph_blocking.py`, `cuda_graph_input.py` |
| 第 4 章 | 模型推理实战 | decode 阶段怎么用 | `cuda_graph_gpt.py` |
| 第 5 章 | 动态 shape 工程解法 | 任意 batch 怎么办 | `cuda_graph_padding.py` |
| 第 6 章 | torch.compile 原理 | Dynamo/AOT/Inductor 三段栈 | `torch_compile_demo.py` |
| 第 7 章 | Graph Break | 什么打断图、怎么观测 | `torch_compile_demo.py` |
| 第 8 章 | compile × CUDA Graph 协同 | 两者怎么配合（最关键） | `torch_compile_demo.py` |
| 第 9 章 | vLLM 落地 | 5 种 mode / piecewise / dispatcher | 见 `DOC.md` |

每章末尾有 **🧩 练习** 和 **✅ 小结**。建议顺序：先把第 1~5 章的 CUDA Graph 打牢，再进第 6~8 章的 torch.compile，最后用第 9 章把两者在 vLLM 里串起来。

---

## 第 0 章：为什么需要 CUDA Graph / torch.compile？

### 0.1 一个常见现象：GPU 没吃满，CPU 却很忙

调试 GPU 推理时你可能见过：**GPU 利用率不高，但 CPU 一直忙着下发指令**。原因是传统 CUDA 执行模型里，每一次 kernel 启动、每一次 memcpy，都要由 CPU 调用一次运行时 API，驱动再解析、排队、调度。

每次 launch 的 CPU 开销约 **5~15μs**。单看不多，但大模型一层 decoder 就有几十上百个 kernel（Attention、FFN、LayerNorm、RoPE…）：

```
Eager (batch=1 decode):  ┃─launch─┃─launch─┃─launch─┃ ... × 100+ kernels / step
                         ↑ 每个 ~10μs，100 个 ≈ 1ms/step 纯 CPU 开销
```

当 **GPU 单步算得很快**（batch=1、seq_len=1 的 decode），这 ~1ms 的 launch 开销就成了瓶颈——这叫 **launch-bound / CPU-bound**。

### 0.2 两条优化路线

| 优化 | 解决什么 | 一句话 |
|------|---------|--------|
| **CUDA Graph** | kernel **launch 开销** | 把一串 kernel 录成一张图，一次 launch 重放整段 |
| **torch.compile** | kernel **数量 + 访存** | 把多个算子**融合**成更少、更高效的 kernel |

两者**正交、可叠加**：compile 先把图变小变快，CUDA Graph 再把剩下的 launch 开销也吃掉。
vLLM 默认两者一起用（第 8、9 章），这是本课的主线。

### ✅ 小结
- launch 开销在 **小 batch / decode** 场景占主导，这是这两个优化的主战场。
- CUDA Graph 砍 launch 次数；torch.compile 砍 kernel 数量与访存；二者可组合。

---

## 第 1 章：CUDA Graph 原理

> NVIDIA 引入 CUDA Graph 的核心思想：**把一连串重复执行的 GPU 操作录下来，打包成整体，以后一键重放**，不再让 CPU 一次次下指令。

三步走：

1. **捕获（Capture）**：在捕获模式下跑一遍目标流程，运行时把所有设备侧操作（kernel launch、异步拷贝、事件同步）**及其依赖关系**记录下来。
2. **实例化（Instantiate）**：把记录转成可执行的图实例（DAG，有向无环图）。节点是操作，边是依赖。
3. **重放（Replay）**：之后每次只需发起**一次**图执行请求，CPU 不再逐个提交 op。

```
传统：CPU → launch k1 → launch k2 → ... → launch kN   （N 次 CPU 提交）
Graph：CPU → graphLaunch（1 次提交）→ GPU 按 DAG 重放 k1..kN
```

**关键认知**：CUDA Graph 省的是 **CPU 端的提交/调度开销**，不是 GPU 的计算时间。所以它只在 **launch 开销占比大**时才有明显收益（第 0 章）。

### ✅ 小结
- 三步：capture → instantiate → replay。
- 图固化的是「操作内容 + 依赖关系」，replay 时一次提交。
- 收益来源 = 减少 CPU launch/调度开销，**不加速 GPU 计算本身**。

---

## 第 2 章：第一个 CUDA Graph

> 配套：[`cuda_graph.py`](./cuda_graph.py)。**先跑一遍看加速比**。

### 2.1 最小骨架

```python
# 1) 固定输入/输出缓冲区（capture 前分配好，replay 复用同一地址）
x = torch.randn(N, N, device="cuda")
y = torch.randn(N, N, device="cuda")

# 2) 预热（在 side stream 上，避免把首次 cuBLAS/cuDNN 懒加载带进 capture）
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(10): compute()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

# 3) 捕获
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    z = compute()          # z 是捕获时分配的输出缓冲区
torch.cuda.synchronize()

# 4) 重放
graph.replay()             # 之后每次只要 replay，CPU 不再逐个发 kernel
```

### 2.2 两个最容易踩的计时坑

`cuda_graph.py` 的注释专门强调了：
1. **CUDA 是异步的**：计时前后必须 `torch.cuda.synchronize()`，否则 `time.time()` 只测到「launch 派发」耗时，`replay()` 会在 GPU 没算完时就返回，得到**假的超大加速比**。
2. **必须有 eager 基线 + 数值校验**：和 eager 对比才知道省了多少；`assert_close` 确认 replay 结果正确。

### 2.3 为什么要刻意用「小张量 + 多个串行小 kernel」

`cuda_graph.py` 用 `N=256, NUM_OPS=50`（50 个串行小算子）。这是**故意制造 launch-bound 场景**：单个 kernel 计算量小，CPU launch 开销占主导，CUDA Graph 收益才明显。反过来，如果用一个超大矩阵乘，GPU 算得久，launch 开销可忽略，CUDA Graph 几乎看不出加速。

### 2.4 为什么预热要放在 side stream？（stream 隔离顺序，不隔离资源）

骨架第 2 步把预热放在一条新建的 **side stream** 上跑（`cuda_graph.py` 第 57–63 行）。一个常见疑问是：side stream 和 default stream 是两条不同的流，凭什么在 side stream 上预热能让 default stream（以及后面的 capture）受益？流之间不是隔离的吗？

**核心结论：Stream 只隔离「执行顺序」，不隔离「资源」。**

预热要初始化的东西——加载的 kernel、cuBLAS/cuDNN 库、显存池、GPU 时钟——全是 **context / 进程 / 硬件级的全局资源**，不属于某一条流。所以在哪条流上预热，这些全局状态就为**所有**流（含 default 与 capture 流）热好了。

| | 是否跨 stream 共享 | 说明 |
|---|---|---|
| **执行顺序 / 队列** | ❌ 不共享（这就是隔离点） | 不同 stream 是独立队列，彼此无隐式先后，可并发重叠 |
| **CUDA context** | ✅ 共享 | 一进程一卡通常只一个 context，所有 stream 都挂在它下面 |
| **已加载的 kernel / module（cubin）** | ✅ 共享 | kernel 加载进的是 context，不是 stream；任何 stream 都能 launch |
| **cuBLAS/cuDNN 库 + autotune 启发式缓存** | ✅ 共享 | 库加载进 context 地址空间，算法选择缓存是进程级 |
| **显存 + PyTorch caching allocator 内存池** | ✅ 共享（进程级） | 预热时 `cudaMalloc` 撑大的池子，default stream 之后能复用 |
| **物理 GPU：SM、L2、时钟/功耗状态** | ✅ 共享 | 硬件只有一套，频率 boost 是设备级 |

**那为什么偏要用 side stream，而不直接在 default stream 上预热？** 这是专为后面的 CUDA Graph capture 服务的：

1. **避开特殊的 legacy default 流**：传统默认流（NULL 流）与所有其他流有隐式同步语义，而 capture 必须在**非默认流**上进行。`torch.cuda.graph()` 内部就在自己的流上 capture，所以约定预热也放 side stream，整条链路都避开 legacy default 流，保持干净。
2. **让显存池/依赖处于 capture 期望的状态**：capture 期间分配器有特殊行为（private pool），在 side stream + 显式依赖的语境里预热，避免 default stream 的残留状态混进 capture。

#### 当前流的切换时间线

把整段「预热 → capture」连起来看，**当前流（current stream）** 经历了 `default → side(预热) → default → graph capture 流 → default` 的切换：

```
① default stream        ← 起始
        │  with torch.cuda.stream(s):        ← 切换
② side stream s          ← 预热 compute() 在这里跑
        │  退出 with                          ← 切回
③ default stream         ← wait_stream / synchronize 在这条流上发生
        │  with torch.cuda.graph(graph):     ← 切换（torch 内部 capture 流）
④ graph capture stream   ← z = compute() 被录制
        │  退出 with                          ← 切回
⑤ default stream
```

两个精确点（容易搞错）：

- **只有 `with` 块切换当前流；`wait_stream` / `synchronize` 不切流。** 它们是「建依赖 / 等待」，作用在「当前是哪条流」之上，本身不改当前流：
  ```python
  s.wait_stream(torch.cuda.current_stream())   # ❌ 不切流：让 s 等 default，当前流仍是 default
  with torch.cuda.stream(s):                   # ✅ 切流：current → s（预热）
      ...
                                               # ✅ 退出 with：current → default
  torch.cuda.current_stream().wait_stream(s)   # ❌ 不切流：让 default 等 s
  torch.cuda.synchronize()                     # ❌ 不切流：CPU 等 GPU 全干完
  ```
- **预热的 `s` 和 capture 流是两条不同的 side 流**：`s` 是你 `torch.cuda.Stream()` 建的；第 ④ 步的 capture 流是 `torch.cuda.graph()` 内部自己持有的另一条流（`graph.default_capture_stream`，未传 `stream=` 时）。所以本质是 **default + 两条不同 side 流**（预热流 / 捕获流）交替。

**`wait_stream` 正是「跨流不隔离」的证据**：流之间能用 event 显式架依赖边（`s.wait_stream(...)` 让 side 等 default 干完再预热；`current.wait_stream(s)` 让 default 等预热全做完才进 capture）。没有这两行，两条流各跑各的（顺序隔离），capture 可能在预热没跑完时就开始，把懒加载/`cudaMalloc` 错误录进图。
c
> 一句话：**资源全局共享 → side stream 预热惠及 default 与 capture 流；顺序彼此隔离 → 必须用 `wait_stream`/`synchronize` 显式焊死「预热完才 capture」。** 详见 [`warmup.md`](./warmup.md)。

### 🧩 练习
1. 跑 `python cuda_graph.py`，记录加速比。把 `N` 改成 `4096`，加速比变大还是变小？为什么？

变了，因为计算量变大了。

2. 删掉计时函数里的 `torch.cuda.synchronize()`，观察加速比变成多少——体会「假加速比」。

变大了。

### ✅ 小结
- 骨架四步：固定缓冲 → 预热 → capture → replay。
- 计时必同步、必有 eager 基线、必校验数值。
- CUDA Graph 的收益场景 = launch-bound（小算子、小 batch）。
- **Stream 隔离顺序、不隔离资源**：side stream 预热惠及 default 与 capture 流；当前流走 `default → side(预热) → default → capture 流`，只有 `with` 块切流，`wait_stream`/`synchronize` 只建依赖。

---

## 第 3 章：CUDA Graph 三大铁律

CUDA Graph 的「静态性」要求很硬，违反会**静默算错**或直接崩。三条铁律：

### 铁律一：捕获期只能有异步设备侧操作（不能同步）

捕获期间出现 `.item()`、`.cpu()`、同步 `cudaMemcpy`、`synchronize()` 这类**把 CPU 拉回执行链路**的操作，会直接破坏 capture。

[`cuda_graph_blocking.py`](./cuda_graph_blocking.py) 就是反例：在 capture 块里做 `x_cuda.copy_(x_host)`（H2D 同步拷贝）和 `.cpu()`，运行直接报错：

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

**正确做法**：所有主机侧逻辑、数据准备放在 capture 之前；要取结果就等 replay 结束后再统一 D2H。

### 铁律二：固定张量地址（replay 绑定的是显存地址，不是 Python 对象）

CUDA Graph 记录的是「从地址 `0x...A` 读、写到地址 `0x...B`」这种底层指令，**不是**「对张量对象 x 运算」。

[`cuda_graph_input.py`](./cuda_graph_input.py) 用两个场景把这点讲透：

```python
# 场景 1：in-place 改内容（地址不变）✅
x.fill_(2.0); graph.replay()      # z 正确更新为 mm(2,1)=4

# 场景 2：重新赋值（生成新张量，地址变了）❌
x = torch.full((N,N), 5.0)        # x 指向新地址
graph.replay()                    # graph 仍读旧地址 → z 还是 4，不是期望的 10
```

**结论**：换对象 = 换地址，replay 读不到新数据。更新输入只能用 in-place：`copy_()` / `fill_()` / `zero_()` / `normal_()`。

#### ❓ 常见问题：`cuda_graph.py` 里一定要固定输入张量 x、y 的地址吗？

**是的，必须固定地址——这不是 demo 的可选写法，而是 CUDA Graph 的机制决定的。**

capture 时，CUDA Graph 把每个 kernel 的实参「显存虚拟地址」原样录进了静态图——记录的是指针值（`x.data_ptr()`、`y.data_ptr()`、`z.data_ptr()` 这些 GPU 地址），而不是「变量 x」这个 Python 名字。`replay()` 不再走 CPU 端算子调度，直接让 GPU 按图里录好的地址去读输入、写输出。

关键是区分 **地址（指针）** 和 **内容（数据）**：

```python
# ✅ 可以：in-place 改内容，地址不变 → replay 读得到新值
x.fill_(0.0)          # 对应 cuda_graph.py 第 83 行：sin(0)=0 → z 全为 0
x.copy_(new_data)     # 把新数据拷进原缓冲区
x[:] = new_data
graph.replay()

# ❌ 不行：重新绑定 x，地址变了 → 图仍读老地址
x = torch.randn(N, N, device=device)   # x 指向新显存
graph.replay()        # 等于没改，甚至读到已释放显存
```

所以 `cuda_graph.py` 第 38–40 行注释强调「必须在捕获前分配好，replay 时复用同一批地址」；输出 `z`（第 67 行）同理是 capture 时定下的固定缓冲区，replay 把结果写回这块地址，后面才能直接断言 `z`。

> **联系 vLLM**：这正是 vLLM 用 CUDA Graph 的标准模式——预先按最大 batch/token 数分配**固定的 input buffer**，每步推理把当前数据 `copy_` 进去再 replay；形状变化则用 **padding 分桶 + 多张预捕获图**覆盖（见第 5 章 `cuda_graph_padding.py`）。
>
> **一句话**：地址（指针）必须固定，内容必须 in-place 更新——这是 CUDA Graph「把动态调度换成静态重放」付出的代价。

### 铁律三：静态形状（shape/stride/dtype/布局在 capture 与 replay 间必须一致）

只有**张量里的数值**可以变；shape、stride、dtype、地址都不能变。
依赖 `.item()` 的动态控制流、动态 shape 都会破坏可重放性。

**那不同 batch / seq_len 怎么办？** 两条路（详见第 5 章）：
- 为不同 shape **分别 capture 多张图**（分桶 bucketing）；
- 用 **padding** 把小输入对齐到一个较大的固定 shape，复用同一张图。

> 这也是 **prefill 阶段难用 CUDA Graph** 的根因：不同请求 prompt 长度差异大，形状多变。
> 而 decode 阶段每步 seq_len=1，形状稳定，天然适合。

### 🧩 练习
1. 跑 `python cuda_graph_blocking.py`，复现报错，再把同步拷贝改成「capture 前准备好数据」让它跑通。
2. 跑 `python cuda_graph_input.py`，解释场景 2 为什么 z 不是 10。

### ✅ 小结
- 三铁律：**捕获期无同步**、**固定地址（只 in-place 更新）**、**静态形状**。
- 违反不一定报错，可能静默算错——这是 CUDA Graph 最大的坑。

---

## 第 4 章：CUDA Graph + 模型推理实战

> 配套：[`cuda_graph_gpt.py`](./cuda_graph_gpt.py)（SimpleGPT2 + CUDAGraphRunner）。

### 4.1 封装模式：CUDAGraphRunner

把「capture 一次 + replay 多次」封装成一个 runner，对外暴露和 `model(x)` 一样的接口：

```python
class CUDAGraphRunner:
    def capture(self, x):
        self.graph_input = x.clone()                    # 固定输入缓冲区
        self.graph_output = torch.empty_like(self.model(self.graph_input))
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.graph_output = self.model(self.graph_input)  # capture 前向

    def forward(self, x):
        self.graph_input.copy_(x)    # in-place 写入新输入（铁律二）
        self.cuda_graph.replay()     # 重放整段前向
        return self.graph_output
```

调用方只需把新数据 `copy_` 进固定缓冲区、`replay()`、读固定输出缓冲区。

### 4.2 多 batch 分桶：decode 阶段的标准玩法

decode 阶段 batch size 不固定，于是**为一组常用 batch 各 capture 一张图**，运行时按 batch 选图：

```python
for batch in [1, 2, 4, 8, 16, 32, 64, 128]:     # 预设档位
    runner = CUDAGraphRunner(model); runner.capture(dummy_input(batch))
    self.graph_runners[batch] = runner
# 推理时：self.graph_runners[x.shape[0]](x)
```

`cuda_graph_gpt.py` 实测：decode 一步从 **0.0206s → 0.0004s**，且输出与 eager 一致（`assert_close`）。这正是第 0 章「batch=1 launch-bound」收益最大的体现。

### 🧩 练习
1. 跑 `python cuda_graph_gpt.py`，确认输出一致且明显加速。
2. 给一个**没 capture 过**的 batch（如 7），观察它如何 fallback 到 eager。
3. 思考：为什么 capture 要按 batch **从大到小**做（提示：内存池复用，见第 5 章）。

### ✅ 小结
- 用 runner 封装 capture/replay，对外接口和普通 model 一致。
- decode 多 batch → 分桶各 capture 一张图，按 batch 选图 replay。

---

## 第 5 章：动态 shape 的工程解法 —— 分桶 + padding

> 配套：[`cuda_graph_padding.py`](./cuda_graph_padding.py)（复刻 vLLM 的 `bs_to_padded_graph_size`）。

线上 batch 是任意值（5、7、9…），但为每个值都 capture 一张图，显存和管理开销会爆炸。vLLM 的解法：

1. **只为少数档位 capture**：如 `[1,2,4,8,16,32]`。
2. **建查表**：把 `[0, max]` 内每个 bs 映射到「应使用的录制档位」（向上取整到最近档），如 `5→8`、`9→16`。
3. **运行时 padding**：实际 bs 向上对齐到该档，多出的行填占位数据，复用那张图，最后只取真实行的结果。

```python
def run(self, x):
    bs = x.shape[0]
    padded = self.bs_to_padded[bs]   # 查表得档位
    buf = self.inputs[padded]
    buf[:bs].copy_(x); buf[bs:].zero_()  # 真实数据 + padding
    self.graphs[padded].replay()
    return self.outputs[padded][:bs]     # 只取真实行
```

一个工程细节：**从大到小 capture**，让小图复用大图分配的内存池（`g.pool()`），省显存。

`cuda_graph_padding.py` 验证了 bs ∈ {1,3,5,7,9,…} 经 padding 后，真实行结果与 eager 全部一致。

### 🧩 练习
1. 跑 `python cuda_graph_padding.py`，看 `bs→档位` 映射表与正确性校验。
2. 把 `capture_sizes` 改成 `[1,4,16,64]`，观察 padding 浪费（如 bs=5 要 padding 到几？）。

### ✅ 小结
- 动态 batch = 分桶 + 向上 padding 复用固定图。
- 用查表 O(1) 选档；从大到小 capture 共享内存池。

---

## 第 6 章：torch.compile 原理

> 配套：[`torch_compile_demo.py`](./torch_compile_demo.py) Part 1。
> 这一块在 `DOC.md` 里几乎没展开，是本教学版重点补齐的内容。

### 6.1 torch.compile 是什么

一行 `torch.compile(model)` 就能把 eager 模型即时编译加速。它和 CUDA Graph 解决的是**不同问题**：CUDA Graph 砍 launch 次数，torch.compile **砍 kernel 数量与访存**（靠算子融合 + 代码生成）。

### 6.2 三段式编译栈

```
你的 Python 函数 / nn.Module
   │
   ├─ ① TorchDynamo    抓 Python 字节码 → FX Graph（计算图）
   │                    遇到不认识的 Python 操作就「graph break」（第 7 章）
   ├─ ② AOTAutograd    拆出前向/反向，把复合算子分解成更基础的 aten 算子
   │
   └─ ③ TorchInductor  把图编译成融合后的 kernel：
                        GPU → 生成 Triton kernel；CPU → 生成 C++/OpenMP
```

- **Dynamo**：在 Python 解释器层面拦截字节码，符号化执行，抽出一张静态图。它给图加「**守卫(guard)**」（如输入形状/类型），下次输入满足守卫就复用已编译产物，否则**重编译**（第 8 章动态 shape）。
- **AOTAutograd**：提前把前向和反向都 trace 出来，并做算子分解（decomposition），为后端提供更规整的算子集。
- **Inductor**：默认后端。核心能力是**算子融合**——把一串逐元素算子合成一个 Triton kernel，中间结果留在寄存器，省掉反复读写显存（和 Course 8 讲的融合是同一个道理）。

### 6.3 Part 1 实测：融合带来的加速

`torch_compile_demo.py` Part 1 对一串逐元素算子 `sin*cos + tanh*2 - exp.clamp` 做对比：

```
eager          :   6.667 ms      # 每个算子一个 kernel，中间结果落显存
torch.compile  :   0.640 ms      # Inductor 融合成单个 Triton kernel
speedup        :   10.42x
```

**10x 来自访存压缩**：eager 要为每个中间结果读写显存，融合后整条链只读一次输入、写一次输出。

### 🧩 练习
1. 跑 `python torch_compile_demo.py`（Part 1），换不同算子链，观察加速比变化。
2. 设环境变量 `TORCH_LOGS=output_code python ...`，看 Inductor 生成的 Triton 代码长什么样。

### ✅ 小结
- torch.compile = Dynamo（抓图）→ AOTAutograd（拆算子）→ Inductor（生成融合 kernel）。
- 主要收益来自**算子融合**（减少 kernel 数 + 访存），与 CUDA Graph 正交。

---

## 第 7 章：Graph Break（计算图被打断）

> 配套：`torch_compile_demo.py` Part 2。

### 7.1 什么是 graph break

Dynamo 抓图时，遇到**无法纳入静态图的 Python 操作**就会「断图」：把函数切成 `图1 → Python 原生执行 → 图2`。常见触发点：

- `.item()` / `.cpu()` / `.tolist()`：把张量值拉回 CPU；
- **依赖张量值的控制流**：`if x.sum() > 0:`；
- 调用 Dynamo 不支持的第三方库函数、打印、复杂 Python 对象操作。

### 7.2 怎么观测：`torch._dynamo.explain`

`torch_compile_demo.py` Part 2 用一个带 `.item()` 的 `if` 对比：

```python
def with_break(x):
    x = x * 2
    if x.sum().item() > 0:   # ← 数据依赖控制流，触发 graph break
        x = x + 1
    return x.sin()

exp = torch._dynamo.explain(with_break)(x)
# 输出：graph 数=2, graph_break 数=1     （被切成两张图）
# 对照 no_break：graph 数=1, graph_break 数=0
```

### 7.3 为什么 graph break 是性能杀手

每个断点意味着：图变小 → **融合机会变少**、要在「图内/图外」之间来回切、而且 **CUDA Graph 也没法跨断点连续捕获**。断点越多，compile 和 CUDA Graph 的收益越差。

这正引出 vLLM 的 **piecewise（分段）模式**：模型里有些断点（比如 attention 这种带自定义 kernel、需要特殊处理的算子）无法避免，于是干脆**在断点处把模型切成多段，每段各自编译 + 各自 CUDA Graph**（第 8、9 章）。

### 🧩 练习
1. 跑 Part 2，复现 graph 数/break 数。
2. 把 `if x.sum().item()>0` 换成 `torch.where(...)`（无数据依赖控制流），看 break 是否消失。

### ✅ 小结
- graph break = Dynamo 把图切断，由 `.item()` / 数据依赖控制流等触发。
- 断点降低融合 + CUDA Graph 收益；vLLM 的 piecewise 就是对断点的工程应对。

---

## 第 8 章：torch.compile × CUDA Graph 协同（最关键的一章）

> 配套：`torch_compile_demo.py` Part 3、Part 4。

### 8.1 两者是叠加关系

| | 解决 | 怎么做 |
|---|------|--------|
| torch.compile（Inductor） | kernel 数量 + 访存 | 融合算子，但融合后的 kernel **仍逐个 launch** |
| CUDA Graph | launch 开销 | 把 kernel 序列录成图，一次 launch 重放 |

所以最优组合：**先 compile 把图变小变快，再用 CUDA Graph 把剩下的 launch 开销吃掉**。

### 8.2 `mode="reduce-overhead"`：compile 内置 CUDA Graph

torch.compile 的 `reduce-overhead` 模式会在 Inductor 编译之上，**自动套一层 CUDA Graph**：

```python
m_default = torch.compile(model)                          # 只融合
m_reduce  = torch.compile(model, mode="reduce-overhead")  # 融合 + CUDA Graph
```

Part 3 在一个 launch-bound 小模型（12×[Linear+GELU]，batch=8）上实测：

```
eager                       :   0.406 ms
compile (default)           :   0.332 ms      # 融合，但仍逐个 launch
compile (reduce-overhead)   :   0.109 ms      # 再叠 CUDA Graph，吃掉 launch 开销
reduce-overhead vs eager    :   3.71x
```

可见 default 只快了一点（这模型本就 launch-bound，融合空间有限），**真正的大头是 CUDA Graph**。

### 8.3 动态 shape 与重编译：为什么 decode 才好用

CUDA Graph 要静态形状，compile 也对形状敏感。Part 4 喂入形状序列 `[128,256,512,256,128]`：

```
默认(自动动态)  : 编译出 2 张图（第一次按 128 静态编译；遇到 256 时把维度升级为动态，
                                 之后 512/256/128 全复用这张动态图）
dynamic=True   : 编译出 1 张图（一上来就编译形状无关的动态图）
```

**要点**：形状抖动会触发重编译/多图；而 **decode 阶段 seq_len=1、形状恒定**，最适合 compile + CUDA Graph。prefill 形状多变，所以要么走 piecewise、要么靠 padding 分桶（第 5 章）。

### 8.4 串起来：vLLM 的两种组合

- **FULL CUDAGraph**：整个模型前向 capture 成一张图。要求形状规整（uniform decode），可独立用，也可叠在 compile 上。
- **PIECEWISE CUDAGraph**：依赖 **piecewise compilation**——在 graph break 处把前向切成多段，**每段分别编译 + 分别 CUDA Graph**，断点处（如 attention）走图外。兼容性强，适合 prefill / 混合 batch。

### 🧩 练习
1. 跑 Part 3，比较 default 与 reduce-overhead，解释差距来自哪。
2. 跑 Part 4，把序列改成全 `256`，观察是否只编译 1 张图。
3. 思考：为什么 vLLM 对 decode 用 FULL、对 prefill 用 PIECEWISE？

### ✅ 小结
- compile 与 CUDA Graph **叠加**：融合 + 减 launch。
- `reduce-overhead` = compile 自带 CUDA Graph；收益大头常来自 CUDA Graph。
- 形状恒定（decode）最适合；形状多变（prefill）走 piecewise / padding。

---

## 第 9 章：vLLM 落地 —— 5 种 CUDAGraphMode 与动态分派

vLLM v1 把 CUDA Graph 与编译逻辑解耦，提供 5 种 `cudagraph_mode`：

| mode | 含义 | 适用 |
|------|------|------|
| `NONE` | 禁用 CUDA Graph | 调试 / `enforce_eager` |
| `PIECEWISE` | 分段图（断点处走图外） | prefill / 混合 batch，兼容性强 |
| `FULL` | 整个前向一张图 | 规整、稳定负载 |
| `FULL_DECODE_ONLY` | 仅 decode 用 FULL | decode 优先 |
| `FULL_AND_PIECEWISE` | **默认**：decode 用 FULL、prefill 用 PIECEWISE | 通用在线服务 |

运行时分派（`CudagraphDispatcher.dispatch`）按 **FULL > PIECEWISE > NONE** 优先级，用 `BatchDescriptor`（`num_tokens` / `uniform_decode` / `has_lora` 等）作 key 匹配已捕获的图：

```
decode（num_tokens=1, uniform=True）   → 命中 FULL 图 → replay
prefill（num_tokens=16, uniform=False）→ 命中 PIECEWISE 图 → 分段 replay
都不命中                                → NONE，eager 执行
```

**这部分（CudagraphDispatcher / BatchDescriptor / CUDAGraphWrapper / capture_model / padding 表）在 [`DOC.md`](./DOC.md) 有逐行源码解析**，本章只给地图，细节看 DOC。

开启方式：

```bash
# 默认就开 CUDAGraph + compile；用 --enforce-eager 关掉对照
vllm serve Qwen/Qwen3-1.7B
vllm serve Qwen/Qwen3-1.7B --enforce-eager        # 纯 eager
vllm bench latency --model Qwen/Qwen2.5-1.5B --input-len 128 --output-len 128 --batch-size 1
```

### 实测结论（来自 `DOC.md` 基准）
- **低 batch（在线推理）**：CUDA Graph 带来 ~**1.47x**（47%）延迟降低——launch-bound 主场。
- **高 batch（离线批处理）**：~7~12% 吞吐提升（GPU 计算占主导，符合 Amdahl 定律），但 **P99 尾延迟降低 70~80%**，对 SLA 极重要。

### ✅ 全课总结
1. **launch 开销**在小 batch / decode 场景是瓶颈。
2. **CUDA Graph**：capture→replay，一次提交重放整段，砍 launch 开销；三铁律=异步/固定地址/静态形状。
3. **动态 shape**：分桶 + padding 复用固定图。
4. **torch.compile**：Dynamo→AOTAutograd→Inductor，靠**融合**砍 kernel 数与访存；注意 graph break。
5. **协同**：compile + CUDA Graph 叠加（`reduce-overhead`）；形状恒定的 decode 最受益。
6. **vLLM**：FULL（decode）+ PIECEWISE（prefill）双轨，dispatcher 按 BatchDescriptor 动态选图。

---

## 附录：跑通所有示例

```bash
cd /home/eechengyang/CX/vllm_learn/code
# 多卡机器建议指定空闲卡：CUDA_VISIBLE_DEVICES=1
python course9/cuda_graph.py            # 第 2 章：最小 CUDA Graph + 加速比
python course9/cuda_graph_blocking.py   # 第 3 章：同步操作导致 capture 失败（反例）
python course9/cuda_graph_input.py      # 第 3 章：固定地址铁律
python course9/cuda_graph_padding.py    # 第 5 章：分桶 + padding
python course9/cuda_graph_gpt.py        # 第 4 章：模型推理实战（decode 提速）
python course9/torch_compile_demo.py    # 第 6~8 章：torch.compile 四个实验
```
