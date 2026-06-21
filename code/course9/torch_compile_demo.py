"""
torch.compile 教学 demo：配合 course9/TUTORIAL.md 第 6~8 章使用。

本文件用 4 个独立小实验，把 torch.compile 的核心概念讲清楚：
  Part 1  基础加速      —— compile vs eager，算子融合带来的收益
  Part 2  Graph Break   —— 什么操作会打断计算图、怎么观测、为什么影响性能
  Part 3  reduce-overhead—— torch.compile + CUDA Graph 的组合拳（最关键）
  Part 4  动态 shape     —— 形状变化触发重编译，以及 dynamic=True 的作用

运行（建议指定一张空闲卡）：
    CUDA_VISIBLE_DEVICES=1 python course9/torch_compile_demo.py

torch.compile 的三段式编译栈（原理见 TUTORIAL 第 6 章）：
    Python 字节码
      └─ TorchDynamo  : 抓取字节码 → FX Graph，遇到不认识的就 graph break
          └─ AOTAutograd : 拆出前/反向，做算子分解
              └─ TorchInductor : 生成融合后的 Triton(GPU)/C++(CPU) kernel
"""
import time
import torch

assert torch.cuda.is_available(), "需要 CUDA"
torch.set_float32_matmul_precision("high")
DEV = "cuda"


def bench(fn, iters=50, warmup=20):
    """同步计时：先预热（compile 的首次编译开销也在这里被吃掉），再 sync 计时。返回 ms/iter。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# ============================================================
# Part 1：基础加速 —— Inductor 把一串逐元素算子融合成一个 kernel
# ============================================================
def part1_basic_speedup():
    print("\n" + "=" * 60)
    print("Part 1: torch.compile 基础加速（算子融合）")
    print("=" * 60)

    # 一串逐元素算子：eager 下每个都是独立 kernel + 中间结果落显存；
    # Inductor 会把它们融合成一个 Triton kernel，省掉大量访存。
    def f(x):
        return torch.sin(x) * torch.cos(x) + torch.tanh(x) * 2.0 - x.exp().clamp(max=10)

    x = torch.randn(8192, 8192, device=DEV)
    f_compiled = torch.compile(f)

    eager_ms = bench(lambda: f(x))
    comp_ms = bench(lambda: f_compiled(x))

    torch.testing.assert_close(f(x), f_compiled(x), rtol=1e-3, atol=1e-3)
    print(f"  eager          : {eager_ms:7.3f} ms")
    print(f"  torch.compile  : {comp_ms:7.3f} ms")
    print(f"  speedup        : {eager_ms / comp_ms:7.2f}x   （逐元素链被融合成单个 kernel）")


# ============================================================
# Part 2：Graph Break —— 数据依赖的控制流会把图切断
# ============================================================
def part2_graph_break():
    print("\n" + "=" * 60)
    print("Part 2: Graph Break（计算图被打断）")
    print("=" * 60)

    # .item() 把 GPU 标量拉回 CPU 做 if 判断 —— Dynamo 无法把它纳入静态图，
    # 只能在这里「断图」：前半段编译成 graph1，python 执行 if，后半段是 graph2。
    def with_break(x):
        x = x * 2
        # Q: 这里触发graph break有两点：1. cpu标量，存在GPU->CPU的内存拷贝；2. if 动态分支
        if x.sum().item() > 0:   # ← 数据依赖控制流：触发 graph break
            x = x + 1
        return x.sin()

    # 无数据依赖：整段可以编译成一张图
    def no_break(x):
        x = x * 2
        x = x + 1
        return x.sin()

    x = torch.randn(1024, 1024, device=DEV)

    # 用 torch._dynamo.explain 直接数出 graph break 个数（教学观测手段）
    exp = torch._dynamo.explain(with_break)(x)
    print(f"  with_break : graph 数={exp.graph_count}, graph_break 数={exp.graph_break_count}")
    exp2 = torch._dynamo.explain(no_break)(x)
    print(f"  no_break   : graph 数={exp2.graph_count}, graph_break 数={exp2.graph_break_count}")
    print("  结论：.item()/依赖张量值的 if 会断图；断点越多，融合/CUDA Graph 收益越差。")
    print("  （vLLM 的 piecewise 模式正是在不可避免的断点处把模型切成多段分别编译）")


# ============================================================
# Part 3：reduce-overhead —— torch.compile 内置 CUDA Graph
# ============================================================
def part3_reduce_overhead():
    print("\n" + "=" * 60)
    print("Part 3: mode='reduce-overhead' = compile + CUDA Graph")
    print("=" * 60)

    # 一个 launch-bound 的小模型：很多小 kernel 串行，CPU launch 开销占主导。
    # 这正是 CUDA Graph 的主场。reduce-overhead 模式会在 Inductor 编译之上，
    # 自动用 CUDA Graph 把 kernel 序列录下来重放，进一步砍掉 launch 开销。
    import torch.nn as nn

    model = nn.Sequential(
        *[layer for _ in range(12)
          for layer in (nn.Linear(512, 512), nn.GELU())]
    ).to(DEV).eval()

    x = torch.randn(8, 512, device=DEV)

    with torch.inference_mode():
        eager_ms = bench(lambda: model(x))
        # Q: 为什么这里的实验结果：compile 后的反而比 eager 的差？
        # A: 
        m_default = torch.compile(model)                              # 仅融合
        m_reduce = torch.compile(model, mode="reduce-overhead")       # 融合 + CUDA Graph
        with torch.no_grad():
            default_ms = bench(lambda: m_default(x))
            reduce_ms = bench(lambda: m_reduce(x))

    print(f"  eager                       : {eager_ms:7.3f} ms")
    print(f"  compile (default)           : {default_ms:7.3f} ms")
    print(f"  compile (reduce-overhead)   : {reduce_ms:7.3f} ms  ← 额外吃掉 launch 开销")
    print(f"  reduce-overhead vs eager    : {eager_ms / reduce_ms:7.2f}x")
    print("  原理：default 融合算子但仍逐个 launch；reduce-overhead 再叠一层 CUDA Graph 重放。")


# ============================================================
# Part 4：动态 shape —— 形状变了会重编译
# ============================================================
def part4_dynamic_shape():
    print("\n" + "=" * 60)
    print("Part 4: 动态 shape 与重编译")
    print("=" * 60)

    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    def f(x):
        return (x @ x.T).sin()

    # 每遇到没见过的 shape，Dynamo 的守卫(guard)失败，会重新编译。
    # 但默认开启「自动动态」：第一次按具体形状编译；第二次遇到新形状时，
    # 它会把该维度标记为动态，重编译出一份「形状无关」的图，之后各种形状都复用它。
    # 这就是为什么 CUDA Graph / compile 在 prefill（长度多变）阶段难用、
    # 而 decode（seq_len=1，形状恒定）阶段好用。
    dynamo.reset()
    counters.clear()
    f_auto = torch.compile(f)
    for n in [128, 256, 512, 256, 128]:
        f_auto(torch.randn(n, n, device=DEV))
    auto_graphs = counters["stats"]["unique_graphs"]

    # 对比：dynamic=True 从一开始就编译形状无关的图，省掉那次「升级」重编译
    dynamo.reset()
    counters.clear()
    f_dyn = torch.compile(f, dynamic=True)
    for n in [128, 256, 512, 256, 128]:
        f_dyn(torch.randn(n, n, device=DEV))
    dyn_graphs = counters["stats"]["unique_graphs"]

    print(f"  喂入形状序列 [128, 256, 512, 256, 128]（3 个不同形状）")
    print(f"  默认(自动动态)     : 编译出 {auto_graphs} 张图"
          f"（1 张静态 + 1 张升级后的动态图，后续形状全复用）")
    print(f"  dynamic=True       : 编译出 {dyn_graphs} 张图（一上来就是形状无关的动态图）")
    print(f"  结论：形状抖动会触发重编译；decode 阶段形状恒定，最适合 compile + CUDA Graph。")


if __name__ == "__main__":
    print("torch:", torch.__version__)
    part1_basic_speedup()
    part2_graph_break()
    part3_reduce_overhead()
    part4_dynamic_shape()
    print("\n全部 demo 跑完。建议配合 course9/TUTORIAL.md 第 6~8 章阅读。")
