"""
CUDA Graph + 模型前向实战（decode 阶段）。

相比旧版的关键修正：
1. 计时 bug：旧版用 time.time() 包住 model(x) / replay()，但 CUDA 是异步的，
   不 synchronize 就只测到 launch 派发耗时——replay() 会在 GPU 还没算完就返回，
   于是得出 0.0004s vs 0.0206s（≈50x）这种假加速比。
   正确做法：warmup + synchronize + 多次迭代取平均（见 bench()）。
2. 捕获前缺预热：旧版 capture() 只跑了一次前向就捕获，cuBLAS/cuDNN 的懒加载、
   autotune 可能发生在 capture 期间，既不稳又会把首次开销带进图。这里在 side stream
   上预热若干次再捕获。
3. eager 基线也要预热后再计时，否则首次调用的 autotune 开销会让对比严重失真。

真实加速比通常是 1.5~3x（取决于 batch、模型大小、kernel 数），而不是几十倍。
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
import time


@dataclass
class ModelConfig:
    num_layers: int = 12
    embedding_dim: int = 768
    num_heads: int = 12
    vocab_size: int = 50257


class SimpleGPT2(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.vocab_size = model_config.vocab_size
        self.embed_layer = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_config.embedding_dim,
                nhead=model_config.num_heads,
                batch_first=True,
            )
            for _ in range(model_config.num_layers)
        ])
        self.lm_head = nn.Linear(model_config.embedding_dim, model_config.vocab_size)

    def forward(self, x):
        h = self.embed_layer(x)
        for block in self.transformer_blocks:
            h = block(h)
        return self.lm_head(h)


class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = None
        self.graph_output = None

    def capture(self, x, num_warmup=3):
        assert self.cuda_graph is None, "CUDA graph has already been captured."

        # 固定的输入占位符（地址固定，后续只往里 copy_ 新数据）
        # Q: 这里 clone 并 detach 的作用是什么？
        # A: 
        self.graph_input = x.clone().detach()

        # 在 side stream 上充分预热：让 cuBLAS/cuDNN 完成懒加载和 autotune，
        # 否则这些开销可能发生在 capture 期间，导致捕获不稳或把首次开销固化进图。
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(num_warmup):
                self.model(self.graph_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # 捕获
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.graph_output = self.model(self.graph_input)
        torch.cuda.synchronize()

    def forward(self, x):
        self.graph_input.copy_(x)   # in-place 写回固定输入缓冲区
        self.cuda_graph.replay()
        return self.graph_output

    __call__ = forward


class ModelRunner:
    def __init__(self, model, seq_len=1):
        self.model = model
        self.seq_len = seq_len
        self.graph_runners = {}

    def capture_decode_graph(self, batch_sizes):
        # decode 阶段每步 seq_len=1，shape 规整，最适合 CUDA Graph。
        # 为每个常用 batch size 各捕获一张图（CUDA Graph 只支持静态 shape）。
        for batch in batch_sizes:
            x = torch.randint(0, self.model.vocab_size, (batch, self.seq_len), device="cuda")
            runner = CUDAGraphRunner(self.model)
            runner.capture(x)
            self.graph_runners[batch] = runner

    def decode(self, x):
        bs = x.shape[0]
        runner = self.graph_runners.get(bs)
        if runner is None:
            print(f"Warning: batch={bs} 未捕获，回退到 eager。")
            return self.model(x)
        return runner(x)


def bench(fn, iters=100, warmup=10):
    """同步计时，返回每次迭代毫秒数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    torch.manual_seed(0)

    config = ModelConfig()
    model = SimpleGPT2(config).cuda().eval()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    runner = ModelRunner(model, seq_len=1)
    runner.capture_decode_graph(batch_sizes)

    print(f"{'batch':>6} | {'eager(ms)':>10} | {'graph(ms)':>10} | {'speedup':>8} | 数值校验")
    print("-" * 60)
    with torch.inference_mode():
        for bs in batch_sizes:
            x = torch.randint(0, config.vocab_size, (bs, 1), device="cuda")

            # 正确性：graph replay vs eager
            out_eager = model(x)
            out_graph = runner.decode(x)
            ok = torch.allclose(out_eager, out_graph, rtol=1e-3, atol=1e-3)

            eager_ms = bench(lambda: model(x))
            graph_ms = bench(lambda: runner.decode(x))
            print(f"{bs:>6} | {eager_ms:>10.3f} | {graph_ms:>10.3f} | "
                  f"{eager_ms / graph_ms:>7.2f}x | {'通过' if ok else '失败!'}")
