"""
MoE(Mixture of Experts,混合专家)教学示例
====================================================

承接 ffn.py:一个 FFN 就是一个"专家"。

MoE 的核心思想一句话:
    把 Transformer block 里的 "1 个 FFN 子层" 换成 "N 个并列的 FFN(专家)+ 1 个路由器(router)"。
    每个 token 只被送进其中 top-k 个专家(而不是全部),从而:
        - 参数量(总专家数)可以做得很大 —— "知识容量"大;
        - 但每个 token 实际计算量只取决于 k —— "激活参数"小、推理便宜。

对比维度:
                  Dense(稠密)              MoE(稀疏)
    FFN 子层      1 个 FFN                  N 个 FFN(专家)+ router
    每 token 计算 走这唯一 1 个 FFN          只走被 router 选中的 top-k 个专家
    参数利用      全部参数都参与每次前向      总参数大,但每 token 只激活一小部分
    典型代表      LLaMA、Qwen3-Dense         Mixtral、Qwen3-MoE、DeepSeek-MoE

本文件用同一个隐藏维度 d,分别搭一个 Dense FFN 和一个 MoE 层,直观对比二者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接复用 ffn.py 里的 SwiGLU 当作"专家";每个专家就是一个独立 FFN。
from ffn import SwiGLUFFN


# ---------------------------------------------------------------------------
# 路由器 Router(也叫 gating network / gate)
# ---------------------------------------------------------------------------
class Router(nn.Module):
    """决定"每个 token 该交给哪些专家"。

    本质是一个很小的 Linear:把 token 的表示 [.., d] 映射成 N 个专家的打分 [.., N],
    再对打分做 softmax / top-k,得到"选哪几个专家 + 各自权重"。
    """

    def __init__(self, d: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d, num_experts, bias=False)  # d -> N 个专家打分

    def forward(self, x: torch.Tensor, top_k: int):
        # x: [num_tokens, d]
        logits = self.gate(x)                       # [num_tokens, N] 每个专家的分
        # 取每个 token 分最高的 k 个专家
        topk_val, topk_idx = logits.topk(top_k, dim=-1)   # [num_tokens, k]
        # 只在被选中的 k 个专家之间做 softmax,得到归一化的组合权重
        topk_weight = F.softmax(topk_val, dim=-1)         # [num_tokens, k]
        return topk_idx, topk_weight                       # 选谁 + 各自权重


# ---------------------------------------------------------------------------
# MoE 层:N 个专家(FFN)+ router,top-k 稀疏激活
# ---------------------------------------------------------------------------
class MoELayer(nn.Module):
    """用 N 个 FFN 专家替换掉单个 FFN 子层。

    前向流程:
        1. router 给每个 token 选出 top-k 个专家及其权重;
        2. 每个 token 只送进这 k 个专家(其余专家对它不计算);
        3. 把 k 个专家的输出按权重加权求和,作为该 token 的最终输出。

    形状上对外仍是 [.., d] -> [.., d],和一个普通 FFN 完全可互换。
    """

    def __init__(self, d: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d = d
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(d, num_experts)
        # N 个并列的、各自独立的 FFN —— 每个就是一个专家
        self.experts = nn.ModuleList([SwiGLUFFN(d) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d];先摊平成 token 序列便于逐 token 路由
        batch, seq, d = x.shape
        x_flat = x.reshape(-1, d)                       # [T, d],T = batch*seq

        topk_idx, topk_weight = self.router(x_flat, self.top_k)  # [T, k], [T, k]

        y = torch.zeros_like(x_flat)                    # [T, d] 输出累加器
        # 教学写法:按专家遍历,把"被分到该专家的 token"挑出来算一次。
        # (生产实现会用分组/批处理 kernel,但语义等价。)
        for e in range(self.num_experts):
            # 找出本专家在哪些 (token, 槽位) 被选中
            mask = topk_idx == e                        # [T, k] bool
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue                                # 没 token 选它,跳过
            selected = x_flat[token_ids]                # [m, d] 选中的 token
            out = self.experts[e](selected)             # [m, d] 过该专家 FFN
            w = topk_weight[token_ids, slot_ids].unsqueeze(-1)  # [m, 1] 权重
            y.index_add_(0, token_ids, out * w)         # 按权重累加回去

        return y.reshape(batch, seq, d)                 # 还原形状 [.., d]


# ---------------------------------------------------------------------------
# 演示:同一个 d 下,Dense FFN 与 MoE 层的对比
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch, seq, d = 2, 4, 16
    x = torch.randn(batch, seq, d)

    # Dense:就是单个 FFN 子层
    dense = SwiGLUFFN(d)

    # MoE:4 个专家,每个 token 只激活 top-2
    moe = MoELayer(d, num_experts=4, top_k=2)

    y_dense = dense(x)
    y_moe = moe(x)

    print("输入  x         :", tuple(x.shape))
    print("Dense FFN  输出 :", tuple(y_dense.shape))
    print("MoE 层     输出 :", tuple(y_moe.shape))

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    dense_p = n_params(dense)
    moe_p = n_params(moe)
    # 每 token 实际"激活"的专家参数 ≈ top_k 个专家;router 很小可忽略
    active_p = n_params(moe.experts[0]) * moe.top_k

    print("\n参数量对比:")
    print(f"  Dense FFN 总参数        : {dense_p}")
    print(f"  MoE 总参数(4 专家+router): {moe_p}")
    print(f"  MoE 每 token 激活参数(≈top2): {active_p}")

    print("\n要点:")
    print("  - 一个专家 = 一个独立 FFN;MoE = N 个 FFN 并列 + router 选 top-k。")
    print("  - 总参数随专家数变大(容量大),但每 token 只激活 top-k 个(计算省)。")
    print("  - 对外仍是 [.., d] -> [.., d],可直接替换 Transformer 里的 FFN 子层。")
