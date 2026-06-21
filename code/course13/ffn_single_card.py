import torch
import torch.nn as nn
import torch.nn.functional as F
from atten_single_card import SingleCardMultiHeadAttention


class FeedForward(nn.Module):
    """
    swiGLU = down(silu(gate(x)) * up(x))
    """
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()

        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)


    def forward(self, x):
        """
        输入：[2, 128, 512]
        """

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # 逐元素门控
        h = F.silu(gate) * up
        return self.down_proj(h)


class TransformerBlock(nn.Module):
    """
    x = x + Atten(Norm(x))
    x = x + FFN(Norm(x))
    """
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.attn = SingleCardMultiHeadAttention(embed_dim, num_heads)
        self.ffn_norm = nn.RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)

    def forward(self, x, mask=None):
        ## attention 子层
        attn_out, _ = self.attn(self.attn_norm(x), mask)
        x = x + attn_out

        ## ffn 子层
        ffn_out, _ = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x


if __name__ == "__main__":
    
    batch_size, seq_len = 2, 128
    embed_dim, num_heads, ffn_dim = 512, 8, 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    block = TransformerBlock(embed_dim, num_heads, ffn_dim).to(device)
    out = block(x, mask=causal_mask)

    print(f"Block 输入: {x.shape}")    # [2, 128, 512]
    print(f"Block 输出: {out.shape}")  # [2, 128, 512] —— 形状不变，可堆叠 N 层
