import torch
import torch.nn as nn
import torch.nn.functional as F

# batch_size=2, seq_len=128, hidden_size=512, head_num=8, head_dim=64

class SingleCardMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.embed_dim = embed_dim # 512
        self.num_heads = num_heads # 8
        self.head_dim = embed_dim // self.num_heads

        assert self.embed_dim == self.num_heads * self.head_dim # 必须整除

        ## q, k, v 投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        ## 输出投影层
        self.o_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x, mask=None):
        
        batch_size, seq_len, embed_dim = x.shape

        # 计算q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 将q/k/v切分多头
        # a. 按照head拆分
        # q[2, 128, 8, 64]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # b. 交换维度, 把head和seq_len交换位置
        # q [2, 8, 128, 64]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        ## 此时 [batch_size, seq_len, num_heads, head_dim]
        # 计算注意力分数
        # [2, 8, 128, 64] @ [2, 8, 64, 128] -> [2, 8, 128, 128]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)


        # 应用mask
        # mask shape 是 [1, 1, 128, 128]
        # mask 在这里的作用是什么？如何参与运算的
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))


        # Softmax 归一化得到注意力权重
        # atten_weights shape 是 [2, 8, 128, 128]
        atten_weights = F.softmax(scores, dim=-1)


        # 注意力权重和V相乘
        # attent_weights shape 是 [2,8,128,128]
        # v shape 是 [2,8,128,64]
        # context shape 是 [2,8,128,64]
        context = torch.matmul(atten_weights, v)


        # 合并多头 -> shape 变为 [2, 128, 8, 64]
        context = context.transpose(1, 2).contiguous()
        # flatten
        context = context.view(batch_size, seq_len, embed_dim)


        # 输出投影
        output = self.o_proj(context)

        return output, atten_weights


if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 2
    seq_len = 128
    embed_dim = 512
    num_heads = 8
    
    # 确保在GPU上运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")


    # 输入 x
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)

    # 创建因果掩码矩阵，防止看到未来的信息
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
    # 扩展为 [1, 1, 128, 128]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    print(causal_mask.shape)

    # 初始化模型
    attention_layer = SingleCardMultiHeadAttention(embed_dim, num_heads).to(device)
    print(attention_layer)

    # forward
    output, atten_weights = attention_layer.forward(x, causal_mask)
    print(f"output shape: {output.shape}")
    print(f"atten_weights shape: {atten_weights.shape}")

    
    print("\n--- Shape Verification ---")
    print(f"Input Shape:          {x.shape}")
    print(f"Causal Mask Shape:    {causal_mask.shape}")
    print(f"Attention Weights:    {atten_weights.shape}  <- [B, H, S, S]")
    print(f"Output Shape:         {output.shape}  <- [B, S, E]")
    
    # 验证 mask 是否生效：右上三角应该全部为 0
    print(f"Sample head weight (top-right should be 0):\n{atten_weights[0, 0, :3, :3]}")