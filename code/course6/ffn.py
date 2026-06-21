"""
FFN(Feed-Forward Network,前馈网络)教学示例
====================================================

Transformer 的每个 block 由两个子层组成:
    1. Attention 子层   —— token 之间交换信息(混 token)
    2. FFN 子层         —— 每个 token 各自做一次非线性变换(混 channel)

本文件聚焦 FFN 子层,给出两种实现,它们在 Transformer 里扮演的角色完全相同:

    A. 经典 FFN  = 2 层 MLP,带"先升维再降维"的瓶颈结构
    B. 现代 FFN  = SwiGLU,用 gate/up/down 三个 Linear,是当下主流 LLM 的标配

核心要点:
    - FFN 是逐 token(position-wise)的:对序列里每个位置独立、用同一套权重做变换。
    - "瓶颈"指中间维度 d_ff 远大于输入维度 d(经典取 d_ff = 4d),
      先把表示升到高维空间做非线性,再压回原维度。
    - FFN 通常是 Transformer 里参数量最大的部分(约占 2/3),也是 MoE 要替换的对象。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A. 经典 FFN:Linear(d -> d_ff) -> 激活 -> Linear(d_ff -> d)
# ---------------------------------------------------------------------------
class ClassicFFN(nn.Module):
    """经典 Transformer 的 FFN(原始 "Attention is All You Need" 用的就是它)。

    结构:  x --(升维)--> d_ff --(激活)--> --(降维)--> d
    形状:  [.., d] -> [.., d_ff] -> [.., d_ff] -> [.., d]

    通常 d_ff = 4 * d,所以叫"瓶颈":两头细(d)、中间粗(4d)。
    早期用 ReLU,后来 BERT/GPT 系列多用 GELU。
    """

    def __init__(self, d: int, d_ff: int | None = None, activation=F.gelu):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d  # 经典取 4 倍
        self.fc1 = nn.Linear(d, d_ff)   # 升维:d -> d_ff
        self.fc2 = nn.Linear(d_ff, d)   # 降维:d_ff -> d
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d]
        h = self.fc1(x)        # -> [batch, seq, d_ff]   升到高维
        h = self.act(h)        # -> [batch, seq, d_ff]   非线性
        y = self.fc2(h)        # -> [batch, seq, d]       压回原维度
        return y


# ---------------------------------------------------------------------------
# B. 现代 FFN:SwiGLU(gate / up / down 三个 Linear)
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """现代 LLM(LLaMA、Qwen、Mistral 等)普遍使用的 FFN 变体。

    GLU(Gated Linear Unit)思想:把"升维"拆成两路
        gate 路:  Linear(d -> d_ff),过激活函数(SiLU/Swish)当"门控"
        up   路:  Linear(d -> d_ff),不过激活,当"内容"
    两路逐元素相乘(门控内容),再由 down 路压回 d:
        down 路:  Linear(d_ff -> d)

    公式:  FFN(x) = down( SiLU(gate(x)) * up(x) )

    注意:它的"角色"仍然是 FFN 子层 —— 逐 token 的非线性变换,
         只是把经典 FFN 的 fc1 换成了"门控双路"。
         因为多了一路,为保持参数量相当,d_ff 常取约 (8/3)*d 而非 4*d。
    """

    def __init__(self, d: int, d_ff: int | None = None):
        super().__init__()
        # 多了一路 gate,所以中间维度通常缩到 ~8/3 d 以对齐总参数量
        d_ff = d_ff if d_ff is not None else int(8 / 3 * d)
        self.gate_proj = nn.Linear(d, d_ff, bias=False)  # 门控路
        self.up_proj = nn.Linear(d, d_ff, bias=False)    # 内容路
        self.down_proj = nn.Linear(d_ff, d, bias=False)  # 降维路

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d]
        gate = F.silu(self.gate_proj(x))  # -> [.., d_ff]  门控(带激活)
        print("gate shape: ", tuple(gate.shape))
        up = self.up_proj(x)              # -> [.., d_ff]  内容(不带激活)
        print("up shape: ", tuple(up.shape))
        h = gate * up                     # -> [.., d_ff]  逐元素门控
        print("h shape: ", tuple(h.shape))
        y = self.down_proj(h)             # -> [.., d]      压回原维度
        return y


# ---------------------------------------------------------------------------
# 演示:两种 FFN 输入输出形状一致,可在 Transformer 里互换
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch, seq, d = 2, 4, 16
    x = torch.randn(batch, seq, d)

    classic = ClassicFFN(d)               # d_ff = 4d = 64
    swiglu = SwiGLUFFN(d)                 # d_ff ≈ 8/3 d ≈ 42

    y1 = classic(x)
    y2 = swiglu(x)

    print("输入  x      :", tuple(x.shape))
    print("经典 FFN 输出:", tuple(y1.shape), "  d_ff =", classic.fc1.out_features)
    print("SwiGLU  输出 :", tuple(y2.shape), "  d_ff =", swiglu.gate_proj.out_features)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    print("\n参数量对比(故意把 d_ff 选成参数量相当):")
    print("  ClassicFFN :", n_params(classic))
    print("  SwiGLUFFN  :", n_params(swiglu))

    print("\n要点:两者输入输出都是 [.., d],角色都是 FFN 子层,可直接互换。")
