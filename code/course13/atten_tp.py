
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# tp=2 切分，每张卡输入是一样的，q/k/v只保留一半的head_num按照列切分，输出投影矩阵按照行切分

### 1. 分布式环境初始化
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} 初始化完成，使用 GPU: {torch.cuda.current_device()}")


def cleanup():
    dist.destroy_process_group()



## 张量并行的组件定义

# 列切分
class ColumnParallelLinear(nn.Module):
    """
    列并行线性层（用于QKV投影）
    输入X: [B, S, E] 每张卡上相同
    权重W: [E, 3 * E / TP_SIZE]（按列切分）
    输出Y: [B, S, 3 * E / TP_SIZE]
    """
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.randn(self.out_features_per_partition, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features_per_partition))


    def forward(self, x):
        # x shapes is [B, S, E]
        return F.linear(x, self.weight, self.bias)
        

# 行切分
class RowParallelLinear(nn.Module):
    """
    行并行线性层（用于 O 投影）
    输入X：[B, S, E / TP_SIZE]（每个 GPU 上的本地多头拼接结果）
    权重W: [E / TP_SIZE, E]（按行切分）
    输出Y: [B, S, E]（经过 All-Reduce后的完整输出）
    """
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.randn(self.out_features))

    def forward(self, x):
        """
        需要注意：这里有一次 All-Reduce 并且进行的是sum，汇总两卡的结果
        """
        partial_output = F.linear(x, self.weight, self.bias)
        # 这里的dist process group 因为要进行 all-reduce 因此会死等直到传完
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)

        return partial_output


# TP Attention Layer 定义
class TensorParallelAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, world_size, rank):
        super().__init__()
        self.embed_dim = embed_dim # 512
        self.num_heads = num_heads # 8
        self.world_size = world_size # 2
        self.rank = rank #

        self.head_dim = embed_dim // num_heads
        self.num_heads_per_partition = self.num_heads // world_size
        self.local_embed_dim = self.num_heads_per_partition * self.head_dim


        # QKV 投影层，列并行
        # 输入 512，输出为 512 * 3 = 1536，切分后每张卡输出 3 * 256 = 768
        self.qkv_proj = ColumnParallelLinear(embed_dim, 3 * embed_dim, world_size)

        # O 投影层，行并行
        # 输入 512，输出512。每张卡切分后输入为 256，按行切分，输出512
        self.o_proj = RowParallelLinear(embed_dim, embed_dim, world_size)

    def forward(self, x, mask = None):
        
        batch_size, seq_len, _ = x.shape

        # 1. QKV 投影
        # qkv_local shape [2, 128, 256 * 3]
        qkv_local = self.qkv_proj(x)

        # 2. 拆分出的q/k/v
        q_local, k_local, v_local = torch.chunk(qkv_local, 3, dim=-1)

        # 3. 切分多头并且转置
        # 变换：[2, 128, 256] -> [2, 128, 4, 64] -> [2, 4, 128, 64]
        q_local = q_local.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k_local = k_local.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v_local = v_local.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        if self.rank == 0:
            print(f"\n--- [Rank 0 内部维度追踪] ---")
            print(f"1. 输入 X 维度:               {x.shape}")
            print(f"2. QKV 投影后 (qkv_local):    {qkv_local.shape}")
            print(f"3. 拆分后 Q_local 维度:        {q_local.shape}  <- [B, Local_H, S, D]")

        if self.rank == 0:
            print("")

        # 4. 本地自注意力计算
        # scores 的shape: [2, 4, 128, 128]
        # 缩放因子是 sqrt(head_dim)，即 head_dim ** 0.5，与单卡版本保持一致；
        # 注意 head_dim 是按 head 切分的局部维度，TP 不改变 head_dim，所以缩放因子不变
        scores = torch.matmul(q_local, k_local.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # 注意是 masked_fill（in-place 版本是 masked_fill_），不是 mask_filled
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # shape is [2, 4, 128, 128]
        atten_weights = F.softmax(scores, dim=-1)

        # shape is [2, 4, 128, 64]
        context_local = torch.matmul(atten_weights, v_local)
        
        # 5. 合并本地的多头
        # [2, 4, 128, 64] -> [2, 128, 4, 64] -> [2, 128, 256]
        context_local = context_local.transpose(1, 2).contiguous()
        context_local = context_local.view(batch_size, seq_len, self.local_embed_dim)

        if self.rank == 0:
            print(f"4. 注意力分数 Scores 维度:     {scores.shape}  <- [B, Local_H, S, S]")
            print(f"5. 本地多头合并后维度:         {context_local.shape}  <- [B, S, Local_E]")

        # 6. 输出投影
        # [2, 128, 256] -> output: [2, 128, 512]，内部会 All-Reduce
        output = self.o_proj(context_local)

        if self.rank == 0:
            print(f"6. O_proj (All-Reduce后) 维度: {output.shape}  <- [B, S, E]")
            print(f"-----------------------------\n")

        return output


def run_worker(rank, world_size):
    setup(rank, world_size)

    batch_size = 2
    seq_len = 128
    embed_dim = 512
    num_heads = 8

    # 初始化TP模型，并搬运至GPU（漏了 .cuda(rank) 会导致权重在 CPU、输入在 GPU 而报 device 不一致）
    model = TensorParallelAttention(embed_dim, num_heads, world_size, rank).cuda(rank)

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