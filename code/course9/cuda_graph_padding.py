"""
动态 batch size 的工程解法：分桶（bucketing）+ padding。

背景：CUDA Graph 只支持静态 shape，但线上请求的 batch size 是任意的（5、7、9…）。
为每个可能的 bs 都捕获一张图 → 显存和管理开销爆炸。
vLLM 的做法（见 DOC 的 CompilationConfig.post_init_cudagraph_sizes / bs_to_padded_graph_size）：
  1. 只为少数几档 size 捕获图（capture_sizes，如 [1,2,4,8,16,32]）；
  2. 预先建一张查表：把 [0, max] 内每个 bs 映射到「应使用的录制档位」（向上取整到最近档）；
  3. 运行时实际 bs 向上 padding 到该档位，多出来的行填占位数据，复用那张图。

这份 demo 复刻了这套机制，并验证 padding 后的结果对「真实行」与 eager 一致。
"""
import torch
import torch.nn as nn


def build_bs_to_padded(capture_sizes, max_size):
    """复刻 vLLM 的 bs_to_padded_graph_size：bs -> 向上取整到的录制档位。"""
    capture_sizes = sorted(capture_sizes)
    table = [0] * (max_size + 1)
    # 对每个区间 (start, end]，bs 落在其中就 padding 到 end
    for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
        for bs in range(start, end):
            table[bs] = start if bs == start else end
    return table


class TinyModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class PaddedGraphRunner:
    """为若干档 size 各捕获一张图；运行时按查表把任意 bs padding 到对应档位再 replay。"""

    def __init__(self, model, dim, capture_sizes):
        self.model = model
        self.dim = dim
        self.capture_sizes = sorted(capture_sizes)
        self.max_size = self.capture_sizes[-1]
        self.bs_to_padded = build_bs_to_padded(self.capture_sizes, self.max_size)
        self.graphs = {}      # size -> CUDAGraph
        self.inputs = {}      # size -> 固定输入缓冲区
        self.outputs = {}     # size -> 固定输出缓冲区
        self._capture_all()

    def _capture_all(self):
        # Q: 按从大到小捕获，便于小图复用大图分配的内存池（与 vLLM 一致），为什么？
        # A: 多张图共享同一内存池（pool=g.pool()）时，每张图捕获期的「临时分配」
        #    （forward 内部的中间激活，在 capture 内被分配又释放）都从这个池子里借块。
        #    关键：从大到小捕获，最大那张图先把池子撑到峰值工作集，之后每张更小的图
        #    需要的临时块都能装进「已经预留好的大块」里 → 池子不必再增长，也不会留下
        #    「小到装不下后续大请求」的零碎块。
        #    反过来从小到大：先按小图预留小块，轮到大图时小块装不下、只能再额外申请，
        #    既抬高峰值显存又制造碎片。
        #    所以大→小让总预留显存 ≈ 最大图（而非各图之和），这就是「吃掉碎片」的本质。
        pool = None
        for size in sorted(self.capture_sizes, reverse=True):
            inp = torch.zeros(size, self.dim, device="cuda")
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(inp)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            ctx = torch.cuda.graph(g, pool=pool) if pool else torch.cuda.graph(g)
            with ctx:
                # Q: 结合 inp 和 out，为什么没有cuda cpp算子中显式地内存在host和device之间的搬运？
                # A: 因为数据从头到尾都在 GPU 上，根本不需要 H2D/D2H 搬运。
                #    - inp 是用 device="cuda" 建的设备缓冲区；权重在 cuda；out 也落在设备上。
                #    - 喂数据走的是 run() 里的 buf[:bs].copy_(x)，x 也是 cuda 张量 → 这是
                #      设备内 D2D 拷贝（把数据搬进固定缓冲区），不碰主机。
                #    - 取结果 self.outputs[padded][:bs] 返回的也是设备张量，连校验 allclose
                #      都在 GPU 上做，全程没有回主机。
                #    手写 CUDA C++ demo 常见 cudaMemcpy H2D/D2H，是因为那种例子从 CPU 上准备
                #    数据、最后要在 CPU 打印；这里 PyTorch 把张量直接建在 cuda 上，省去显式搬运。
                #    更关键：CUDA Graph 捕获期禁止同步 H2D/D2H（铁律一），所以这里也只能是纯设备计算。
                out = self.model(inp)
            torch.cuda.synchronize()
            pool = g.pool()  # 共享内存池给下一张（更小的）图
            self.graphs[size], self.inputs[size], self.outputs[size] = g, inp, out

    def run(self, x):
        bs = x.shape[0]
        assert bs <= self.max_size, f"bs={bs} 超过最大录制档 {self.max_size}"
        padded = self.bs_to_padded[bs]          # 查表得到要用的档位
        buf = self.inputs[padded]
        buf[:bs].copy_(x)                        # 真实数据填前 bs 行
        buf[bs:].zero_()                         # padding 行填占位
        self.graphs[padded].replay()
        return self.outputs[padded][:bs]         # 只取真实行的结果


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    dim = 512
    capture_sizes = [1, 2, 4, 8, 16, 32]
    model = TinyModel(dim).cuda().eval()

    runner = PaddedGraphRunner(model, dim, capture_sizes)

    print("bs -> padding 档位 映射表（部分）:")
    for bs in [1, 3, 5, 7, 9, 15, 17, 31, 32]:
        print(f"  bs={bs:>2}  ->  graph={runner.bs_to_padded[bs]:>2}")
    print()

    print(f"{'bs':>4} | {'用到的图':>8} | 数值校验(真实行 vs eager)")
    print("-" * 45)
    # Q: torch.inference_mode 的作用是什么？
    # A: 它是「只做前向推理」的上下文，关闭 autograd，相当于更激进、更省的 no_grad：
    #    - 不构建反向计算图、不存梯度元数据 → 省显存、降每个算子的开销。
    #    - 比 no_grad 更彻底：还关掉张量的 version counter / view 追踪，所以在它里面新建的
    #      张量无法再参与 autograd，换来更低的运行时开销（适合纯推理/benchmark）。
    #    - 与 model.eval() 分工不同：eval() 管的是 dropout/BN 等「训练 vs 推理行为差异」的层；
    #      inference_mode() 管的是 autograd 开关。推理要两者配合。
    #    - 对本 demo：保证前向路径干净、低开销，数值对比更贴近真实部署。
    with torch.inference_mode():
        for bs in [1, 3, 5, 7, 9, 13, 16, 20, 32]:
            x = torch.randn(bs, dim, device="cuda")
            out_graph = runner.run(x)
            out_eager = model(x)
            ok = torch.allclose(out_graph, out_eager, rtol=1e-3, atol=1e-3)
            print(f"{bs:>4} | {runner.bs_to_padded[bs]:>8} | {'通过' if ok else '失败!'}")
