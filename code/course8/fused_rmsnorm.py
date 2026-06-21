"""
算子融合教学示例：Fused RMSNorm（含 residual-add 融合版本）

本文件配合 course8/TUTORIAL.md 第 2 章「算子融合」使用。

RMSNorm 是现代大模型（LLaMA / Qwen 等）最常用的归一化算子：
    y = x / sqrt(mean(x^2) + eps) * weight

用 PyTorch「搭积木」实现时，会被拆成多个独立算子（pow / mean / add / rsqrt / mul ...），
每一步都要把整行数据从显存读出、再把中间结果写回显存。RMSNorm 本质是 **访存密集型
(memory-bound)** 算子：计算量很小，瓶颈在带宽。因此把这些步骤「融合」进一个 kernel、
让中间结果常驻寄存器，就能省下大量的 HBM 往返，带来明显加速。

运行：
    python course8/fused_rmsnorm.py
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Baseline 1：PyTorch「搭积木」实现（多次访存）
# ============================================================
def rmsnorm_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    # 每一行都要被反复读写：x^2 -> mean -> rsqrt -> x*... -> *weight
    variance = x.pow(2).mean(dim=-1, keepdim=True)   # 读 x，写中间结果
    x = x * torch.rsqrt(variance + eps)              # 再读 x，再写
    return x * weight                                # 又读一遍，再写


# ============================================================
# Triton：Fused RMSNorm（一个 kernel 搞定整行）
# ============================================================
@triton.jit
def rmsnorm_kernel(
    x_ptr,            # 输入 [n_rows, n_cols]
    w_ptr,            # 权重 [n_cols]
    y_ptr,            # 输出 [n_rows, n_cols]
    x_row_stride,
    y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 处理一整行（与 softmax 一致的「按行并行」策略）
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ---- 只读一次 x，整行进寄存器 ----
    x = tl.load(x_ptr + row * x_row_stride + col_offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)  # 归约用 fp32 累加，保证数值精度

    # ---- 下面的 square / mean / rsqrt / mul 全部在片上完成，零中间回写 ----
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    y = x * rstd * w

    # ---- 只写一次输出 ----
    tl.store(y_ptr + row * y_row_stride + col_offsets, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.is_cuda and weight.is_cuda
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    rmsnorm_kernel[(n_rows,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


# ============================================================
# 进阶：Fused Add + RMSNorm（producer-consumer 融合）
# ------------------------------------------------------------
# vLLM / Transformer 残差结构里极常见的一段：
#     residual = residual + hidden          # 残差相加
#     hidden   = rmsnorm(residual) * weight  # 再做归一化
# 把「相加」和「归一化」融进同一个 kernel：残差相加的结果直接留在寄存器里喂给
# 归一化，省掉一次完整的中间张量读写。同时把更新后的 residual 写回（后续层还要用）。
# ============================================================
@triton.jit
def add_rmsnorm_kernel(
    x_ptr,            # 本层输入 hidden
    res_ptr,          # 残差 residual（原地更新为 x + res）
    w_ptr,
    y_ptr,            # 归一化输出
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    base = row * row_stride + col_offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # producer：残差相加
    h = x + res
    # 把更新后的 residual 写回（下一层的残差分支要用）
    tl.store(res_ptr + base, h, mask=mask)

    # consumer：紧接着做 RMSNorm，h 仍在寄存器里，无需重新从显存读
    var = tl.sum(h * h, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    tl.store(y_ptr + base, h * rstd * w, mask=mask)


def add_rmsnorm_triton(x, residual, weight, eps: float = 1e-6):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    residual = residual.clone()  # 演示用：避免污染输入；实际 vLLM 是原地更新
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    add_rmsnorm_kernel[(n_rows,)](
        x, residual, weight, y,
        x.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, residual


# ============================================================
# 正确性 + 性能对比
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    eps = 1e-6

    print("=" * 60)
    print("正确性验证")
    print("=" * 60)
    for n_rows, n_cols in [(4, 123), (16, 512), (1024, 4096)]:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)

        y_ref = rmsnorm_naive(x, w, eps)
        y_tri = rmsnorm_triton(x, w, eps)
        ok = torch.allclose(y_ref, y_tri, atol=1e-3, rtol=1e-3)
        print(f"  RMSNorm     shape=({n_rows},{n_cols}):  match={ok}")

        res = torch.randn_like(x)
        y_fused, res_new = add_rmsnorm_triton(x, res, w, eps)
        y_fref = rmsnorm_naive(x + res, w, eps)
        ok2 = torch.allclose(y_fref, y_fused, atol=1e-3, rtol=1e-3)
        print(f"  Add+RMSNorm shape=({n_rows},{n_cols}):  match={ok2}")

    print("\n" + "=" * 60)
    print("性能对比 (shape = 8192 x 4096, fp16)")
    print("=" * 60)
    try:
        from triton.testing import do_bench

        x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
        w = torch.randn(4096, device="cuda", dtype=torch.float16)

        ms_naive = do_bench(lambda: rmsnorm_naive(x, w, eps))
        ms_triton = do_bench(lambda: rmsnorm_triton(x, w, eps))
        # 同时跟官方融合算子比一下（若 torch 版本支持）
        print(f"PyTorch 搭积木 (naive): {ms_naive:.4f} ms")
        print(f"Triton  融合 kernel    : {ms_triton:.4f} ms")
        print(f"加速比 Speedup         : {ms_naive / ms_triton:.2f}x")
    except Exception as e:
        print(f"Benchmark skipped: {e}")
