# Course 8 教学版：用 Triton 写大模型高性能算子

> 本文是 [`DOC.md`](./DOC.md) 的**教学重构版**：把零散的参考资料整理成可按章节讲授的课程，
> 并补齐三块核心内容——**Triton 核心入门**、**算子融合**、**FlashAttention**。
>
> - `DOC.md`：完整参考手册（softmax / matmul / attention / vLLM Triton 后端逐行解析）。
> - 本文 `TUTORIAL.md`：分章节、带「为什么」、带练习的教学主线，配套可运行代码。
>
> **配套代码**：`add.py`(向量加) · `softmax.py` · `matmul.py` · `fused_rmsnorm.py`(算子融合) · `attention.py`(FlashAttention)
>
> **环境**：Triton ≥ 3.0，CUDA GPU。每个代码文件都能 `python xxx.py` 直接跑出正确性 + 性能对比。

---

## 课程地图

| 章节 | 主题 | 核心问题 | 配套代码 |
|------|------|----------|----------|
| 第 0 章 | 为什么要手写算子 | GPU 慢在哪？访存 vs 计算 | — |
| 第 1 章 | Triton 核心入门 | 编程模型、三板斧、寻址、调优 | `add.py` |
| 第 2 章 | 算子融合 | 为什么融合能加速？怎么融？ | `fused_rmsnorm.py`, `softmax.py` |
| 第 3 章 | 在线 Softmax | FlashAttention 的数学基石 | `softmax.py` |
| 第 4 章 | FlashAttention | 怎么把 attention 装进 SRAM | `attention.py` |
| 第 5 章 | 工程落地 | vLLM 里的 Triton 后端 | 见 `DOC.md` |

每章末尾有 **🧩 练习** 和 **✅ 小结**，适合课堂讲授 + 课后实操。

---

## 第 0 章：为什么要手写算子？

### 0.1 一个反直觉的事实：大模型推理大多是「访存瓶颈」

我们习惯认为 GPU 是「算力怪兽」，但在 LLM 推理里，**很多算子的瓶颈不是算力，而是显存带宽**。

衡量一个算子是「算力受限」还是「访存受限」，用 **算术强度 (Arithmetic Intensity)**：

```
算术强度 = 浮点运算次数 (FLOPs) / 访存字节数 (Bytes)
```

- 算术强度**高** → compute-bound（瓶颈是算力），例如大矩阵乘 GEMM。
- 算术强度**低** → memory-bound（瓶颈是带宽），例如 RMSNorm、Softmax、逐元素加、Decode 阶段的 Attention。

以 RMSNorm 为例：每个元素只做几次乘加，却要把整个张量从 HBM 读出、写回。算力闲着，带宽跑满——这类算子的优化核心就是**少搬数据**。

### 0.2 PyTorch「搭积木」为什么不够快

```python
# RMSNorm 的 PyTorch 写法 = 一串独立算子
variance = x.pow(2).mean(-1, keepdim=True)  # 读 x，写中间结果
x = x * torch.rsqrt(variance + eps)         # 再读 x，再写
return x * weight                           # 又读，又写
```

每一行 `.pow()` / `.mean()` / `.rsqrt()` / `*` 都是一次独立的 GPU kernel：

1. **每个 kernel 都要把数据从 HBM 读进来、算完写回去**——中间结果在显存里来回搬。
2. **每次启动 kernel 都有固定开销**（launch overhead，几微秒级，小算子上占比可观）。
3. **无法精细控制**分块、并行粒度、计算/访存重叠、tile 大小等。

手写融合算子能把这些中间结果**留在寄存器/共享内存**里，一次读入、一次写出。`fused_rmsnorm.py`
实测在 `8192×4096` 上比 PyTorch 搭积木快 **~3x**，原因正是省掉了中间张量的 HBM 往返。

### ✅ 小结
- 先判断算子是 compute-bound 还是 memory-bound，再决定优化方向。
- LLM 里的归一化 / 激活 / 逐元素 / decode attention 大多是 memory-bound，**融合 = 省访存 = 加速**。

---

## 第 1 章：Triton 核心入门

### 1.1 Triton 是什么

> Triton 是一门嵌入在 Python 里的 GPU 编程 DSL + 编译器。语法像 NumPy/PyTorch，但本质是面向硬件的并行语言。

和 CUDA 的关键区别——**编程粒度不同**：

| | CUDA | Triton |
|---|------|--------|
| 编程粒度 | 单个 **thread** | 一个 **program（处理一个 Tile/Block）** |
| 线程索引 | 手动管理 `threadIdx/blockIdx` | 不直接碰线程，只写「一块数据怎么算」 |
| 共享内存 | 手动 `__shared__` + 同步 | 编译器自动管理 |
| 向量化/访存合并 | 手动优化 | 编译器 Pass 自动做 |
| 开发周期 | 长 | 短 |

**心智模型**：在 Triton 里你写的是「**第 pid 个程序实例，负责处理哪一块数据**」。
至于这块数据内部怎么拆给 warp、怎么用共享内存，交给编译器。

### 1.2 三板斧 + 两个概念

写任何 Triton kernel 都离不开这几个 API：

| API | 作用 |
|-----|------|
| `tl.program_id(axis)` | 我是第几个程序实例（在第 `axis` 维网格上的编号） |
| `tl.arange(0, BLOCK)` | 生成块内偏移向量 `[0,1,...,BLOCK-1]` |
| `tl.load(ptr + offs, mask=...)` | 按指针+偏移从 HBM 读一块数据 |
| `tl.store(ptr + offs, val, mask=...)` | 把一块数据写回 HBM |

两个必须理解的概念：

- **`mask`（掩码）**：数据量通常不是 BLOCK 的整数倍，`mask = offs < N` 屏蔽越界访问。**忘了 mask = 非法访存 / 错误结果**。
- **`tl.constexpr`（编译期常量）**：如 `BLOCK_SIZE`，编译器据此做循环展开、死代码消除等优化。

### 1.3 第一个 kernel：向量加法（逐行讲）

完整代码见 [`add.py`](./add.py)。核心五个变量：

```python
@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                       # ① 我是第几块
    block_start = pid * BLOCK_SIZE               # ② 这块在全局的起点
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # ③ 这块覆盖的全局索引（向量）
    mask = offsets < N                           # ④ 屏蔽越界
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    tl.store(Z_ptr + offsets, x + y, mask=mask)  # ⑤ 算完写回
```

启动时用 **grid** 指定要起多少个程序实例：

```python
grid = (triton.cdiv(N, BLOCK_SIZE),)   # ceil(N / BLOCK_SIZE) 个 block
vector_add_kernel[grid](x, y, z, N, BLOCK_SIZE=1024)
```

**关键直觉**：kernel 里没有 `for` 循环遍历所有数据，因为不同 block 是**并行**跑在不同 SM 上的。
`pid` 不是「第几次循环」，而是「我是哪一块」。

> 形象例子（`N=10, BLOCK_SIZE=4`）：
>
> | pid | block_start | offsets | mask |
> |-----|-------------|---------|------|
> | 0 | 0 | [0,1,2,3] | 全有效 |
> | 1 | 4 | [4,5,6,7] | 全有效 |
> | 2 | 8 | [8,9,10,11] | [T,T,F,F]（屏蔽 10,11）|

### 1.4 多维寻址：stride 与广播 `[:, None]`

一维很简单，二维呢？GPU 显存是**一维线性**的，多维张量靠 **stride（步幅）** 换算地址：

```
元素 [i, j] 的线性地址 = i * stride(0) + j * stride(1)
```

Triton 用 `[:, None]` / `[None, :]` 把一维偏移广播成二维坐标网格（和 NumPy 完全一致）：

```python
offs_m = tl.arange(0, BLOCK_M)[:, None]   # 列向量 (BLOCK_M, 1) —— 行坐标
offs_n = tl.arange(0, BLOCK_N)[None, :]   # 行向量 (1, BLOCK_N) —— 列坐标
# 广播相加得到 (BLOCK_M, BLOCK_N) 的地址块：
ptrs = base_ptr + offs_m * stride_row + offs_n * stride_col
```

这套「指针 + stride + 广播」是 Triton 寻址的**通用套路**，matmul / attention 全靠它。
建议把它当成肌肉记忆：**行坐标用 `[:, None]`，列坐标用 `[None, :]`**。

### 1.5 常用 `tl` API 速查

| 类别 | API | 说明 |
|------|-----|------|
| 索引 | `tl.program_id` / `tl.arange` / `tl.cdiv` | 定位 + 偏移 |
| 访存 | `tl.load` / `tl.store`（`mask=`, `other=`） | 读写 HBM |
| 归约 | `tl.sum` / `tl.max` / `tl.min`（`axis=`） | 块内归约 |
| 数学 | `tl.exp` / `tl.sqrt` / `tl.log` / `tl.where` | 逐元素 |
| 矩阵 | `tl.dot(a, b, acc=)` | 块级矩阵乘（走 Tensor Core）|
| 变形 | `tl.trans` / `x[:, None]` / `.to(dtype)` | 转置/广播/类型转换 |

### 1.6 编译流程 与 `num_warps` / `num_stages`

```
@triton.jit 触发：Triton kernel → TTIR(MLIR) → TTGIR → LLVM IR → PTX → GPU 执行
```

启动 kernel 时除了业务参数，还有两个**性能旋钮**：

- **`num_warps`**：每个 program 用多少个 warp（32 线程）干活。块越大通常需要越多 warp。
  `softmax.py` 里就按 `BLOCK_SIZE` 大小调 `num_warps`（4→8→16）。
- **`num_stages`**：软件流水线级数，让「加载下一块」和「计算当前块」重叠（overlap），隐藏访存延迟。matmul/attention 这类有 K 维循环的 kernel 上很有用。

### 1.7 AutoTune：让编译器帮你搜最优配置

tile 大小、`num_warps` 等最优值**依赖硬件和数据形状**，手调很累。用 `@triton.autotune` 自动搜索：

```python
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],   # 这些形状变了才重新搜索；否则复用上次最优
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

Triton 会在首次遇到某组 `key` 时实测所有 config，缓存最快的那个。**生产级算子（包括 vLLM 的 Triton attention）几乎都用 autotune**。

### 🧩 练习
1. 改写 `add.py`，把 `BLOCK_SIZE` 改成 `tl.constexpr` 之外的普通参数会发生什么？为什么不行？
2. 实现一个 **逐元素 `y = a*x + b`（AXPY）** 的 Triton kernel。
3. 给 `add.py` 加上 `@triton.autotune`，搜索 `BLOCK_SIZE ∈ {256,512,1024,2048}`，观察不同 N 下的最优值。

### ✅ 小结
- Triton 的粒度是 **program/Tile**，不是 thread；你只描述「一块怎么算」。
- 五件套：`pid / block_start / offsets / mask / load-store`。
- 多维寻址 = **指针 + stride + `[:,None]`/`[None,:]` 广播**。
- 性能旋钮：`BLOCK_*`、`num_warps`、`num_stages`，用 `autotune` 自动搜。

---

## 第 2 章：算子融合（Operator Fusion）

> 本章配套可运行代码：[`fused_rmsnorm.py`](./fused_rmsnorm.py)（实测 ~3x 加速）、[`softmax.py`](./softmax.py)。

### 2.1 什么是算子融合

**算子融合 = 把多个本来分开执行的算子合并进一个 kernel**，让中间结果不落显存。

两类收益：
1. **省访存**（主要收益）：中间结果常驻寄存器/共享内存，省掉一次次 HBM 读写。对 memory-bound 算子是数倍加速。
2. **省 launch 开销**：N 个 kernel 合成 1 个，少 N-1 次启动。对小算子/小 batch 友好。

### 2.2 算账：以 Softmax 为例

朴素 Softmax（`naive_softmax`）拆成 5 个张量操作，对 `M×N` 输入：

```
读取 ≈ 5MN + 2M 元素，写入 ≈ 3MN + 2M 元素     (8MN 量级访存)
```

融合后理想情况：**只读一次 X、只写一次输出**：

```
读取 ≈ MN，写入 ≈ MN                            (2MN 量级访存)
```

带宽收益上界 ≈ `(8MN+4M) / 2MN ≈ 4x`（N 大时）。`softmax.py` 实测正是 **~4x**。
这说明：**memory-bound 算子的加速比，约等于访存量的压缩比**。

### 2.3 融合的三种典型形态

| 形态 | 含义 | 例子 |
|------|------|------|
| **逐元素链式融合** | 一串 element-wise 接龙 | `bias → GELU → dropout` 合一 |
| **归约融合** | element-wise + reduction 合一 | Softmax、RMSNorm、LayerNorm |
| **生产者-消费者融合** | 前一步输出直接喂下一步 | `residual add → RMSNorm`、`matmul → activation` |

### 2.4 案例：手写 Fused RMSNorm

RMSNorm：`y = x / sqrt(mean(x²) + eps) * weight`，是**归约融合**的代表。

核心思路（完整见 `fused_rmsnorm.py`）：**一个 program 处理一整行，只读一次、只写一次**，
中间的 square / mean / rsqrt / 乘法全在寄存器里完成：

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, x_row_stride, y_row_stride,
                   n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(tl.float32)  # 只读一次

    var  = tl.sum(x * x, axis=0) / n_cols       # ┐
    rstd = 1.0 / tl.sqrt(var + eps)             # │ 全程不回写显存
    w    = tl.load(w_ptr + col, mask=mask)      # │
    y    = x * rstd * w                         # ┘

    tl.store(y_ptr + row * y_row_stride + col, y, mask=mask)  # 只写一次
```

两个工程细节，课堂要强调：
- **归约用 fp32 累加**（`x.to(tl.float32)`）：fp16 直接 `sum(x*x)` 会丢精度。先升 fp32 再算。
- **按行并行**：每行独立，天然适合一个 program 一行；行内归约用 `tl.sum(..., axis=0)`。

### 2.5 案例：Add + RMSNorm 的生产者-消费者融合

Transformer 残差结构里反复出现这一对：

```python
residual = residual + hidden      # 生产者：残差相加
hidden   = rmsnorm(residual) * w  # 消费者：紧接着归一化
```

`add_rmsnorm_kernel` 把两步融进一个 kernel：相加结果 `h` 留在寄存器里直接喂给归一化，
**省掉一次完整中间张量的读写**，同时把更新后的 `residual` 写回（下一层残差分支要用）。
这正是 **vLLM `RMSNorm` 算子支持 `residual` 入参** 的底层逻辑。

### 2.6 融合的边界（什么时候不该融）

- **会爆寄存器/共享内存**：融太多导致 occupancy（占用率）下降，可能反而变慢。
- **compute-bound 算子收益有限**：大 GEMM 本身算力受限，融个 bias 收益不大（但 `matmul → activation` 仍常见）。
- **数据复用模式冲突**：两个算子并行/分块策略天然矛盾时，强融得不偿失。

经验法则：**优先融 memory-bound 的逐元素/归约算子**；融之前先想清楚瓶颈在哪。

### 🧩 练习
1. 跑 `python fused_rmsnorm.py`，记录加速比；把 shape 改成 `(8192, 8192)` 再看，加速比变大还是变小？为什么？
2. 实现 **Fused LayerNorm**（比 RMSNorm 多一个减均值 + bias）。
3. 实现 **Fused Bias + GELU**（逐元素链式融合），与 `F.gelu(x + bias)` 对拍。
4. 思考题：为什么 `add_rmsnorm_kernel` 要把 `residual` 写回，而不只是输出归一化结果？

### ✅ 小结
- 融合的本质：**让中间结果不落显存**。memory-bound 算子加速 ≈ 访存压缩比。
- 三种形态：逐元素链式、归约融合、生产者-消费者。
- 注意 fp32 归约、寄存器压力、瓶颈判断。

---

## 第 3 章：在线 Softmax —— FlashAttention 的数学基石

> 先吃透这一章，FlashAttention 就只剩工程问题了。配套：`softmax.py`。

### 3.1 Safe Softmax 回顾

直接 `exp(x)` 会溢出，所以先减每行最大值（不改变结果，softmax 平移不变）：

```
softmax(x_i) = exp(x_i - max) / Σ_j exp(x_j - max)
```

朴素实现需要**遍历整行 3 遍**：① 求 max ② 求分母 Σexp ③ 算每个输出。
问题：attention 里这一行可能长达几万（seq_len），**整行放不进 SRAM**，必须分块。

### 3.2 在线 Softmax：一遍扫描，边读边更新

核心思想：**分块读入，维护两个「运行统计量」**，每来一个新块就「修正」之前的结果。

维护：当前最大值 `m`、当前分母 `l`。来一个新块、它的局部最大值 `m_j`、局部分母 `l_j`：

```
m_new = max(m, m_j)                       # 更新全局最大值
l_new = exp(m - m_new) * l                # 旧分母「重缩放」对齐到新基准
      + exp(m_j - m_new) * l_j            # 加上新块的贡献
```

关键是那个 **重缩放因子 `exp(m_old - m_new)`**：当出现更大的最大值时，之前基于旧最大值算的所有
中间量都要乘上这个 ≤1 的因子「打折」，对齐到新基准。这样无需回头重算，**单遍扫描就能得到正确结果**。

### 3.3 代码对照

`softmax.py` 因为整行能放进一个 block，用的是 two-pass 的简化版（`tl.max` + `tl.sum` 直接对整行）。
真正的「在线分块更新」逻辑在 `attention.py` 里（下一章），那里序列被切成多个 N 块，
必须用上面的 `m_new / l_new` 递推。**这正是从 softmax 走向 FlashAttention 的桥梁。**

### 🧩 练习
1. 用纸笔推一遍：两个块 `[1, 3]` 和 `[5, 2]`，用在线公式算 softmax，验证和直接算一致。
2. 在 `softmax.py` 基础上，手动把一行拆成两半，用在线递推合并，验证结果不变。

### ✅ 小结
- Safe softmax 减最大值防溢出。
- 在线 softmax 用「运行最大值 + 运行分母 + 重缩放因子」实现**单遍分块**计算。
- 重缩放因子 `exp(m_old - m_new)` 是整个机制的灵魂。

---

## 第 4 章：FlashAttention 拆解

> 配套代码：[`attention.py`](./attention.py)（含 prefill + decode 测试，全部通过）。
> 逐行公式对照见 `DOC.md`「实现 Attention 算子」一节；本章讲**整体设计直觉**。

### 4.1 标准 Attention 的痛点

```
Attention(Q,K,V) = softmax(QKᵀ / √d) · V
```

朴素实现要**显式构造 `S = QKᵀ`**，它的大小是 `seq_len × seq_len`。
序列 8K 时这就是 64M 个元素的中间矩阵，写出再读回——**访存爆炸 + 显存爆炸**。

### 4.2 FlashAttention 的核心招数

**永远不把完整的 `S` 矩阵写进 HBM**。把 Q/K/V 沿 seq_len 分块，在 SRAM 里：
分块算 `QKᵀ` → 在线 softmax 边算边更新 → 直接累加到输出 `O`。`S` 只在片上短暂存在。

分块策略（对照 `attention.py`）：
- **Q 沿 seq_len 切成 `BLOCK_M` 块**，用 **grid 第 0 维** 并行（不同 Q 块互相独立）。
- **K/V 沿 seq_len 切成 `BLOCK_N` 块**，在 kernel 内部用 **`for` 循环**遍历。
- **grid 第 1 维** 并行 `batch × head`。

```python
# attention.py：二维 grid
grid = lambda meta: (triton.cdiv(m_size, meta["BLOCK_M_SIZE"]),  # Q 分块
                     bs * n_heads,                               # batch×head
                     1)
```

### 4.3 内层循环：在线 softmax 实战

`attention.py` 的 `for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE)` 就是第 3 章的在线递推：

```python
qk = tl.dot(q, tl.trans(k))           # 分块 QKᵀ（片上，不落显存）
l_j = tl.max(qk, 1)                   # 当前块行最大值
numerators = tl.exp(qk - l_j[:, None])
d_j = tl.sum(numerators, 1)

l_new = tl.maximum(l_i, l_j)          # ← 第 3 章的 m_new
alpha = tl.exp(l_i - l_new)           # ← 旧结果的重缩放因子
beta  = tl.exp(l_j - l_new)           # ← 新块的缩放因子
d_new = alpha * d_i + beta * d_j      # ← l_new 递推

acc = acc * sigma[:, None]            # 旧的输出累加器先打折对齐
acc += tl.dot(p, v)                   # 再加上新块 P·V 的贡献
```

**直觉总结**：每读一个 K/V 块，就把已经攒下的输出 `acc` 按重缩放因子「打折」，
再加上新块的贡献。扫完所有 K/V 块，`acc` 就是正确的 attention 输出——全程没碰过完整的 `S`。

### 4.4 Causal Mask（因果遮罩）

自回归生成不能看未来 token。`attention.py` 在分块算完 `qk` 后施加下三角 mask：

```python
mask = offs_m[:, None] >= offs_k[None, :]      # 行 >= 列 才可见
qk = tl.where(mask, qk * sm_scale, -1.0e8)     # 屏蔽位置填 -inf，softmax 后≈0
```

### 4.5 Prefill vs Decode

- **Prefill**（首次处理 prompt）：Q 的 seq_len > 1，是大矩阵，compute-bound，FlashAttention 分块计算。
- **Decode**（逐 token 生成）：Q 的 seq_len = 1，但 K/V 很长（历史 KV Cache），memory-bound，瓶颈在读 KV。

`attention.py` 的 `test_prefill_stage` / `test_decode_stage` 分别覆盖这两个阶段，都与 PyTorch 对拍通过。

### 🧩 练习
1. 跑 `python attention.py`，确认 prefill / decode 都 PASS。
2. 把 `BLOCK_M_SIZE / BLOCK_N_SIZE` 从 32 改成 64，观察正确性与速度变化。
3. 关掉 causal mask（`causal_mask=False`），结果会怎样？为什么 decode（seq_len=1）时不需要它？
4. 进阶：给 kernel 加 `@triton.autotune` 搜索 `BLOCK_M/BLOCK_N`。

### ✅ 小结
- FlashAttention = **分块 + 在线 softmax + 永不物化 S 矩阵**。
- Q 用 grid 并行、K/V 用循环遍历；batch×head 走 grid 另一维。
- 内层循环就是第 3 章在线 softmax 的工程实现。

---

## 第 5 章：工程落地 —— vLLM 的 Triton Attention 后端

真实推理框架里，Attention 还要处理**变长序列打包（Packed）** 和 **分页 KV Cache（PagedAttention）**。
vLLM 的 `TritonAttentionImpl` / `kernel_unified_attention_2d` 在第 4 章基础上多了两件事：

1. **Packed 变长**：所有请求的 query 拼成一个大张量，靠 `query_start_loc`(cu_seqlens) 定位每个请求；
   grid 用「总 token 数的上界」启动，多余的空块直接 `return`。
2. **分页 KV Cache**：K/V 不连续存放，靠 `block_table` 把逻辑块映射到物理块，在 kernel 内查表取数。

这部分**逐行解析见 [`DOC.md`](./DOC.md)「vLLM 中的 Triton Attention 后端」一节**（含 `query_start_loc` /
`block_table` / `slot_mapping` 详解）。开启方式：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG \
  vllm serve Qwen/Qwen3-0.6B --attention-backend TRITON_ATTN
```

### ✅ 全课总结
1. **判断瓶颈**：memory-bound 还是 compute-bound，决定优化方向。
2. **Triton 心智模型**：写「一块怎么算」，靠 `pid + offsets + mask + load/store`。
3. **算子融合**：让中间结果不落显存，memory-bound 算子可得数倍加速。
4. **在线 Softmax**：用运行统计量 + 重缩放，实现单遍分块计算。
5. **FlashAttention**：分块 + 在线 softmax + 不物化 S，是前面所有知识点的集大成。
6. **工程落地**：vLLM 在此之上叠加 Packed 变长与分页 KV Cache。

---

## 附录：跑通所有示例

```bash
cd /home/eechengyang/CX/vllm_learn/code
python course8/add.py            # 第 1 章：向量加
python course8/softmax.py        # 第 2/3 章：融合 softmax，~4x
python course8/fused_rmsnorm.py  # 第 2 章：融合 RMSNorm，~3x
python course8/matmul.py         # 分块矩阵乘
python course8/attention.py      # 第 4 章：FlashAttention（prefill+decode）
```
