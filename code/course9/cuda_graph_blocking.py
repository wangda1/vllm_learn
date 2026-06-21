"""
演示：capture 期间出现「主机同步」操作会让图捕获失败，以及正确的修法。

错误示范：从 pageable CPU 内存做同步 H2D 拷贝（copy_(..., non_blocking=False)），
          以及在图内 .cpu() 这类 D2H 同步，都会在 capture 时报：
          "operation not permitted when stream is capturing"。

正确修法：capture 路径里只允许「可被图记录的设备侧异步操作」。如果确实要喂主机数据：
          1) 把 host 张量 pin_memory()（页锁定），
          2) 用 copy_(..., non_blocking=True) 做异步 H2D，
          这样拷贝是可捕获的 cudaMemcpyAsync。要拿结果回主机，则等 replay 结束后再统一 D2H。

注意：一次失败的 capture 会污染本进程的 CUDA 状态，导致后续 capture 出现
      "captures_underway INTERNAL ASSERT" 之类的连锁错误。所以这里把「错误」和
      「正确」两个场景各放到独立子进程里跑，互不干扰。
"""
import sys
import subprocess

N = 1024


def run_wrong():
    import torch
    device = "cuda"
    x_host = torch.randn(N, N)              # pageable（非页锁定）
    x_cuda = torch.empty(N, N, device=device)
    y_cuda = torch.randn(N, N, device=device)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x_cuda.copy_(x_host)            # 同步 H2D（预热阶段合法）
            _ = torch.mm(x_cuda, y_cuda)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    try:
        with torch.cuda.graph(g):
            x_cuda.copy_(x_host)            # ❌ 同步 H2D，capture 期间非法
            z = torch.mm(x_cuda, y_cuda)
            _ = z.cpu()                     # ❌ 同步 D2H，同样非法
        torch.cuda.synchronize()
        print("[wrong] 居然捕获成功了？（不应该）")
    except RuntimeError as e:
        print(f"[wrong] 捕获失败（符合预期）: {str(e).splitlines()[0]}")


def run_right():
    import torch
    device = "cuda"
    x_host = torch.randn(N, N).pin_memory()  # ✅ 页锁定内存，才能真正异步拷贝
    x_cuda = torch.empty(N, N, device=device)
    y_cuda = torch.randn(N, N, device=device)
    z_out = torch.empty(N, N, device=device)  # 预分配输出缓冲区

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x_cuda.copy_(x_host, non_blocking=True)
            z_out.copy_(torch.mm(x_cuda, y_cuda))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    with torch.cuda.graph(g):
        x_cuda.copy_(x_host, non_blocking=True)  # ✅ 可捕获的异步 H2D
        z_out.copy_(torch.mm(x_cuda, y_cuda))    # 计算结果写回固定输出缓冲区
    torch.cuda.synchronize()

    # 喂新数据：in-place 更新页锁定 host buffer，再 replay
    x_host.normal_()
    g.replay()
    # 需要结果回主机？等 replay 结束后再统一 D2H（不要放进图里）
    z_cpu = z_out.cpu()
    print(f"[right] 捕获+重放成功，z_out 已可用，z[0,0]={z_cpu[0, 0].item():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("wrong", "right"):
        {"wrong": run_wrong, "right": run_right}[sys.argv[1]]()
    else:
        # 各起一个干净子进程，避免失败的 capture 污染后续
        for mode in ("wrong", "right"):
            subprocess.run([sys.executable, __file__, mode])
