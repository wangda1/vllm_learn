"""
dp_tp_orthogonal_demo.py
========================
教学示例 2：DP 与 TP 的正交关系，以及数据如何在多机多卡间流动

对应 DOC.md 1.3 ~ 1.4 节（混合张量并行与数据并行）。

核心结论（务必先建立这张图）：
---------------------------------------------------------------
张量并行(TP) 与 数据并行(DP) 是【正交】的两个维度：

    - TP（纵向：切模型）：同一份模型权重被【切开】放到一个 TP 组的多张卡上，
      组内多张卡【协同】算同一条/同一批请求的同一次前向，靠 all-gather / all-reduce
      把切片结果拼/加起来。一个 TP 组 = 一个完整的“模型副本(replica)”。

    - DP（横向：复制模型）：把整份模型【复制】成多个副本(replica)，
      不同副本处理【不同】的请求子集，互相独立。

把 GPU 排成一个二维网格，就能看清正交关系（下例 DP=2, TP=2，共 4 张卡）：

                 TP 维度 (组内切模型, 协同算一次 forward)
                 tp_rank=0        tp_rank=1
              +---------------+---------------+
   dp_rank=0  |    GPU 0      |    GPU 1      |  <- DP 组0 = 副本0 (一个 TP 组)
              +---------------+---------------+
   dp_rank=1  |    GPU 2      |    GPU 3      |  <- DP 组1 = 副本1 (一个 TP 组)
              +---------------+---------------+
                    ^
                    |__ 竖着看：处于相同 TP 位置的 GPU(如 GPU0 与 GPU2) 构成一个 DP 组，
                        它们持有“参数相同”的同一切片，处理不同请求。

   全局 GPU 编号公式：  gpu_id = dp_rank * tp_size + tp_rank

数据流动（一条请求的旅程）：
   请求 --(前端按负载选副本)--> 某个 DP 副本
        --(副本内 TP 组协同 forward: column-parallel -> all-gather,
                                      row-parallel    -> all-reduce)-->
        --> logits --> 采样 --> 返回
   不同 DP 副本之间在“计算上互相独立”（推理时 DP 组的 all-reduce 主要用于
   调度协同/wave 同步，不是把不同请求的激活值加在一起）。

本示例用 numpy 把一层 MLP 真正按 TP 切开计算，并验证：
   “TP 切分多卡协同的结果” == “单卡不切分的结果”，
从而让你确信 TP 的 all-gather / all-reduce 到底在拼什么、加什么。

运行：
    python dp_tp_orthogonal_demo.py
"""

import numpy as np

np.random.seed(0)


# ============================================================
# 0. 一个最小“模型层”：Transformer 里的 MLP 块
#    y = (GELU(x @ W1) @ W2)
#      W1: [hidden, ffn]   —— 第一层，做列并行 (Column Parallel)
#      W2: [ffn, hidden]   —— 第二层，做行并行 (Row Parallel)
#    这正是 vLLM/Megatron 里 MLP 的经典 TP 切法：
#      列并行的输出 + 行并行的输入 天然对齐，组内只需在 W2 之后做一次 all-reduce。
# ============================================================
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class MLPWeights:
    """完整（未切分）的一层 MLP 权重，作为“标准答案”用于对拍。"""
    def __init__(self, hidden, ffn):
        self.hidden = hidden
        self.ffn = ffn
        self.W1 = np.random.randn(hidden, ffn) * 0.1   # [hidden, ffn]
        self.W2 = np.random.randn(ffn, hidden) * 0.1   # [ffn, hidden]

    def forward_single_gpu(self, x):
        """单卡参考实现（不切分）。x: [tokens, hidden] -> [tokens, hidden]"""
        h = gelu(x @ self.W1)     # [tokens, ffn]
        y = h @ self.W2           # [tokens, hidden]
        return y


# ============================================================
# 1. 一张 GPU：只持有“自己那一份”权重切片
# ============================================================
class GPU:
    def __init__(self, gpu_id, dp_rank, tp_rank, tp_size):
        self.gpu_id = gpu_id
        self.dp_rank = dp_rank      # 属于哪个 DP 副本
        self.tp_rank = tp_rank      # 在 TP 组内的位置
        self.tp_size = tp_size
        # 权重切片（在 build_replica 中按 TP 切好放进来）
        self.W1_shard = None        # 列并行：[hidden, ffn/tp]
        self.W2_shard = None        # 行并行：[ffn/tp, hidden]

    def __repr__(self):
        return f"GPU{self.gpu_id}(dp={self.dp_rank},tp={self.tp_rank})"


# ============================================================
# 2. 一个 TP 组 = 一个 DP 副本(replica)：组内多张卡协同算一次 forward
# 注意：这里以replicae作为一个DP副本的抽象，其中包含了多个tp rank，总的叫一个TP组
# ============================================================
class Replica:
    """
    一个数据并行副本。内部由 tp_size 张 GPU 组成一个 TP 组，
    它们各持权重的一部分，协同完成一次前向。
    """
    def __init__(self, dp_rank, tp_size, full_weights: MLPWeights):
        self.dp_rank = dp_rank
        self.tp_size = tp_size
        self.hidden = full_weights.hidden
        self.ffn = full_weights.ffn
        self.gpus = []

        # —— 按 TP 切分权重，分发到组内每张 GPU ——
        ffn_per = self.ffn // tp_size
        # 列并行：W1 沿“列”(ffn 维) 切成 tp_size 份
        W1_shards = np.split(full_weights.W1, tp_size, axis=1)   # each [hidden, ffn/tp]
        # 行并行：W2 沿“行”(ffn 维) 切成 tp_size 份
        W2_shards = np.split(full_weights.W2, tp_size, axis=0)   # each [ffn/tp, hidden]

        for tp_rank in range(tp_size):
            gpu_id = dp_rank * tp_size + tp_rank   # 全局编号公式
            g = GPU(gpu_id, dp_rank, tp_rank, tp_size)
            g.W1_shard = W1_shards[tp_rank]
            g.W2_shard = W2_shards[tp_rank]
            self.gpus.append(g)

    # ---- 副本内一次完整前向：演示 TP 的 all-gather 与 all-reduce ----
    def forward(self, x, verbose=False):
        """
        x: [tokens, hidden]，这一份输入在 TP 组内是【复制】的（每张卡都有完整 x）。
        返回 y: [tokens, hidden]
        """
        tp = self.tp_size

        # === 第一层：列并行 Linear (Column Parallel) ===
        # 每张卡用自己的 W1_shard 算出一段 ffn 维的输出，彼此不重叠。
        local_h = []
        for g in self.gpus:
            h_part = gelu(x @ g.W1_shard)          # [tokens, ffn/tp]，仅本卡负责的那几列
            local_h.append(h_part)
            if verbose:
                print(f"      {g}: 列并行算出隐藏维切片 shape={h_part.shape}")

        # 在 vLLM 里，列并行的输出若要给后面“需要完整维度”的算子用，会做 all-gather。
        # 但这里紧接着是行并行(W2)，行并行恰好【按相同维度】吃这些切片，
        # 所以工程上常常【不 all-gather】，而是各卡保留自己的切片直接喂给 W2。
        # 为了演示 all-gather 的含义，我们也展示“拼起来 == 完整 h”：
        h_full_gathered = np.concatenate(local_h, axis=1)  # all-gather: 沿 ffn 维拼接
        if verbose:
            print(f"      [all-gather] 组内拼接列并行切片 -> 完整隐藏层 shape={h_full_gathered.shape}")

        # === 第二层：行并行 Linear (Row Parallel) ===
        # 每张卡用自己的 W2_shard，吃自己那段 h 切片，算出一个【部分和】(partial sum)。
        partial_y = []
        for g, h_part in zip(self.gpus, local_h):
            y_part = h_part @ g.W2_shard           # [tokens, hidden]，是“部分和”
            partial_y.append(y_part)
            if verbose:
                print(f"      {g}: 行并行算出 partial sum shape={y_part.shape}")

        # 行并行的多张卡各自只算了 sum 的一部分，必须 all-reduce 把它们【相加】，
        # 才能得到正确的完整输出。这是 TP 组内最关键的一次通信。
        y = np.sum(partial_y, axis=0)              # all-reduce(SUM)
        if verbose:
            print(f"      [all-reduce SUM] 组内累加 {tp} 份 partial sum -> 最终输出 shape={y.shape}")

        return y


# ============================================================
# 3. 整个部署：DP 个副本（横向复制），每个副本是一个 TP 组（纵向切分）
# ============================================================
class Deployment:
    def __init__(self, dp_size, tp_size, hidden, ffn):
        self.dp_size = dp_size
        self.tp_size = tp_size
        # 所有副本共享“同一份逻辑权重”（DP 的定义：每个副本权重相同）
        self.full_weights = MLPWeights(hidden, ffn)
        self.replicas = [Replica(dp, tp_size, self.full_weights) for dp in range(dp_size)]

    def print_grid(self):
        print(f"\nGPU 二维网格 (DP={self.dp_size} × TP={self.tp_size} = "
              f"{self.dp_size * self.tp_size} 张卡)，gpu_id = dp_rank*TP + tp_rank：")
        header = "            " + "".join(f"  tp_rank={t}   " for t in range(self.tp_size))
        print(header)
        for dp in range(self.dp_size):
            row = f"  dp_rank={dp} |"
            for g in self.replicas[dp].gpus:
                row += f"   GPU{g.gpu_id:<2}    |"
            print(row + f"   <- DP副本{dp} (一个 TP 组, 协同算 forward)")
        print("              \\__ 竖看: 相同 tp_rank 的卡(如 "
              f"GPU0 与 GPU{self.tp_size}) 持相同权重切片, 构成一个 DP 组")


# ============================================================
# 4. 演示
# ============================================================
def demo_tp_correctness(dep: Deployment):
    """演示 TP 组内 all-gather/all-reduce 协同的结果 == 单卡参考结果。"""
    print("\n" + "=" * 70)
    print("第一部分：TP 组内一次 forward 的数据流（列并行→all-gather, 行并行→all-reduce）")
    print("=" * 70)

    tokens = 3
    x = np.random.randn(tokens, dep.full_weights.hidden)
    print(f"\n输入 x: shape={x.shape} (在 TP 组内每张卡都持有完整副本)")
    print(f"\n>>> 用 DP副本0 的 TP 组（{dep.tp_size} 张卡）协同计算：")
    y_tp = dep.replicas[0].forward(x, verbose=True)

    # 单卡参考答案
    y_ref = dep.full_weights.forward_single_gpu(x)
    max_err = np.max(np.abs(y_tp - y_ref))
    print(f"\n  TP 协同结果   y_tp[0]  = {np.round(y_tp[0], 4)}")
    print(f"  单卡参考结果  y_ref[0] = {np.round(y_ref[0], 4)}")
    print(f"  最大误差 = {max_err:.2e}  ->  {'✔ 完全一致' if max_err < 1e-10 else '✘ 不一致'}")
    print("\n  结论：TP 把权重切开放到多卡，靠 all-gather/all-reduce 拼/加回来，")
    print("        数学上等价于单卡，但显存与算力被分摊到了组内多张卡。")


def demo_dp_tp_dataflow(dep: Deployment):
    """演示请求如何在 DP × TP 网格中流动：前端选副本 -> 副本内 TP 协同。"""
    print("\n" + "=" * 70)
    print("第二部分：DP × TP 正交 —— 多条请求如何在网格中并行流动")
    print("=" * 70)

    # 模拟一批请求，以及前端(内置负载均衡)把它们分发到不同 DP 副本
    requests = ["req-A", "req-B", "req-C", "req-D", "req-E"]
    # 简单轮询/按负载分发：这里用轮询代表“前端把不同请求路由到不同 DP 副本”
    print(f"\n前端收到 {len(requests)} 条请求，按负载均衡路由到 {dep.dp_size} 个 DP 副本：")
    routing = {dp: [] for dp in range(dep.dp_size)}
    for i, r in enumerate(requests):
        dp = i % dep.dp_size
        routing[dp].append(r)

    for dp, reqs in routing.items():
        gpu_ids = [g.gpu_id for g in dep.replicas[dp].gpus]
        print(f"  DP副本{dp} (GPU {gpu_ids}) <- {reqs}")

    print("\n各 DP 副本【相互独立、并行】处理自己分到的请求；")
    print("每个副本内部，TP 组的多张卡【协同】完成 forward：\n")

    hidden = dep.full_weights.hidden
    for dp, reqs in routing.items():
        if not reqs:
            print(f"  DP副本{dp}: 本轮无请求 -> 仍需执行 dummy batch 以对齐(见示例1)")
            continue
        # 把该副本分到的请求拼成一个 batch（DP 的本质：不同副本吃不同数据）
        batch = np.random.randn(len(reqs), hidden)
        gpu_ids = [g.gpu_id for g in dep.replicas[dp].gpus]
        y = dep.replicas[dp].forward(batch, verbose=False)
        print(f"  DP副本{dp} (GPU {gpu_ids}) 处理 {reqs}: "
              f"组内 TP 协同 forward -> 输出 shape={y.shape} ✔")

    print("\n  关键正交关系：")
    print("   - 横向(DP): 副本之间数据不同、计算独立 -> 提升【并发吞吐】。")
    print("   - 纵向(TP): 组内切同一份权重、协同算同一批 -> 降低【单卡显存/延迟】。")
    print("   - 两者可同时叠加：总卡数 = DP × TP，互不干扰，正交组合。")


def main():
    DP_SIZE = 2     # 2 个数据并行副本
    TP_SIZE = 2     # 每个副本用 2 张卡做张量并行
    HIDDEN = 8
    FFN = 16        # 必须能被 TP_SIZE 整除

    print("=" * 70)
    print(f"DP × TP 正交关系演示    DP={DP_SIZE}, TP={TP_SIZE}, "
          f"共 {DP_SIZE * TP_SIZE} 张 GPU")
    print("=" * 70)

    dep = Deployment(DP_SIZE, TP_SIZE, HIDDEN, FFN)
    dep.print_grid()
    demo_tp_correctness(dep)
    demo_dp_tp_dataflow(dep)

    print("\n" + "=" * 70)
    print("一句话总结：TP 是“把一个模型摊开给几张卡一起算”，")
    print("            DP 是“把整个模型复制几份各自算不同请求”，")
    print("            二者正交，vLLM 用 --tensor-parallel-size × --data-parallel-size 同时启用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
