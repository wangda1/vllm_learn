"""
最小 CUDA Graph demo：捕获一段固定的 GPU 计算，replay 重放，并和 eager 对比。

要点（也是这份 demo 相比"只测 graph 时间"的旧版改进的地方）：
1. CUDA 是异步的：计时前后必须 torch.cuda.synchronize()，否则 time.time() 只测到
   "launch 派发"耗时，replay() 会在 GPU 还没算完时就返回，得到假的超大加速比。
2. 要有 eager 基线对比，才能看出 CUDA Graph 到底省了多少。
3. 要验证 replay 的结果和 eager 一致（数值正确性）。
4. 演示"改输入要用 in-place 写回固定缓冲区"，replay 才能读到新数据。
"""
import torch
import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

assert torch.cuda.is_available(), "CUDA not available"

device = "cuda"
# 关键：CUDA Graph 省的是 CPU 端 kernel launch 开销。
# 只有当「launch 开销」在总时间里占比大时（小算子、多算子）才有明显收益。
# 所以这里用「小张量 + 几十个串行小 kernel」来制造 launch-bound 场景。
N = 256
NUM_OPS = 50
num_warmup = 10
num_iter = 200


def bench(fn, iters=num_iter, warmup=num_warmup):
    """同步计时：先预热，再 sync，再计时，最后再 sync。返回每次迭代的毫秒数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


# 固定大小的输入/输出缓冲区（必须在捕获前分配好，replay 时复用同一批地址）
x = torch.randn(N, N, device=device)
y = torch.randn(N, N, device=device)


def compute():
    # 几十个串行小算子：每个 kernel 计算量都很小，CPU launch 开销占主导
    h = x
    for _ in range(NUM_OPS):
        h = torch.tanh(h @ y + x)
    return h


# ---------- 1. eager 基线 ----------
eager_ms = bench(compute)

# ---------- 2. 捕获 CUDA Graph ----------
# 预热放在 side stream 上，避免把首次 kernel 的 cuBLAS/cuDNN 懒加载开销带进捕获，
# 也避免污染默认流。torch.cuda.graph 内部会在自己的 stream 上完成 capture。
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(num_warmup):
        compute()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    z = compute()  # z 是捕获时分配的输出缓冲区；replay 会把结果写回这块地址
torch.cuda.synchronize()

graph_ms = bench(graph.replay)

# ---------- 3. 数值正确性校验 ----------
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(z, compute(), rtol=1e-3, atol=1e-3)

print(f"eager       : {eager_ms*1e3:8.2f} us/iter")
print(f"CUDA Graph  : {graph_ms*1e3:8.2f} us/iter")
print(f"speedup     : {eager_ms / graph_ms:8.2f}x")
print("数值校验通过：replay 结果与 eager 一致")

# ---------- 4. 改输入：必须 in-place 写回原缓冲区，graph 才读得到新数据 ----------
x.fill_(0.0)  # in-place 修改 x 的内容（地址不变）
graph.replay()
torch.cuda.synchronize()
# 此时 sin(0)=0，mm(0,y)=0，所以 z 应当全为 0
assert torch.allclose(z, torch.zeros_like(z), atol=1e-5), "in-place 更新后 replay 结果不对"
print("in-place 更新输入后 replay 生效：z 全为 0")
