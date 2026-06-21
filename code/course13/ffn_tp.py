import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist

from atten_tp import ColumnParallelLinear, RowParallelLinear, TensorParallelAttention



### 1. 分布式环境初始化
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} 初始化完成，使用 GPU: {torch.cuda.current_device()}")


def cleanup():
    dist.destroy_process_group()




class TensorParallelFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, world_size):
        super().__init__()
        # 将 gate 和 up 放在一起进行投影，使用列并行
        self.gate_up_proj = ColumnParallelLinear(embed_dim, ffn_dim * 2, world_size)
        # 完整的shape [2048, 512], 每张卡 shape [1024, 512] 需要进行一次all reduce，op=sum
        self.down_proj = RowParallelLinear(ffn_dim, embed_dim, world_size)


    def forward(self, x):
        # 输入：每张卡相同 [2, 128, 512]
        gate_up = self.gate_up_proj(x)
        # gate, up [2, 128, 1024]
        gate, up = torch.chunk(gate_up, 2, dim=-1)

        h = F.silu(gate) * up

        return self.down_proj(h)

class TensorParallelTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, world_size, rank):
        super().__init__()

        self.attn_norm = nn.RMSNorm(embed_dim)

        self.attn = TensorParallelAttention(embed_dim, num_heads, world_size, rank)

        self.ffn_norm = nn.RMSNorm(embed_dim)

        self.ffn = TensorParallelFeedForward(embed_dim, ffn_dim, world_size)

    def forward(self, x, mask=None):
        
        attn_out = self.attn(self.attn_norm(x))

        x = x + attn_out

        ffn_out = self.ffn(self.ffn_norm(x))

        x = x + ffn_out

        return x



def run_worker(rank, world_size):
    setup(rank, world_size)

    batch_size = 2
    seq_len = 128
    embed_dim = 512
    num_heads = 8
    ffn_dim = 2048

    # 初始化TP模型，并搬运至GPU（漏了 .cuda(rank) 会导致权重在 CPU、输入在 GPU 而报 device 不一致）
    model = TensorParallelTransformerBlock(embed_dim, ffn_dim, num_heads, world_size, rank).cuda(rank)

    # 模拟输入数据
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim).cuda(rank)

    # 构造 Casual Mask 防止看到未来信息
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).cuda(rank)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 前向传播
    output = model(x, mask=causal_mask)

    # 收集两卡的输出验证是否一致
    # 关键点：o_proj 内部做了 all_reduce(SUM)，每张卡都拿到「完整且相同」的最终输出，
    # 所以正确实现下两卡输出应当逐元素一致。这正是验证 TP 正确性的核心依据。
    # for _ in world_size 会报错（int 不可迭代），必须用 range(world_size)
    output_list = [torch.empty_like(output) for _ in range(world_size)]
    dist.all_gather(output_list, output)

    if rank == 0:
        # 先校验输出维度回到完整的 embed_dim（行并行 all_reduce 后维度还原）
        assert output.shape == (batch_size, seq_len, embed_dim), \
            f"输出维度异常: {output.shape}, 期望 {(batch_size, seq_len, embed_dim)}"

        are_identical = torch.allclose(output_list[0], output_list[1], atol=1e-5)

        print(f"[验证结果] GPU 0 和 GPU 1 的最终输出是否完全一致: {are_identical}")
        if are_identical:
            print("✅ Tensor Parallelism 维度与数值验证成功！")
        else:
            # 打印最大误差，方便定位是数值精度问题还是逻辑错误
            max_diff = (output_list[0] - output_list[1]).abs().max().item()
            print(f"❌ 验证失败，两卡输出不一致！最大误差: {max_diff:.6e}")
            
    cleanup()

if __name__ == "__main__":
    world_size = 2
    # 启动 2 个进程分别控制 2 张 GPU
    torch.multiprocessing.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )        