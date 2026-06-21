import torch                        # 导入 PyTorch 核心库
import torch.distributed as dist   # 导入分布式通信模块

def init_process():
    dist.init_process_group(backend='nccl')  # 用 NCCL 后端初始化进程组（GPU 间通信）
    torch.cuda.set_device(dist.get_rank())   # 每个进程绑定自己序号对应的 GPU

def example_reduce_scatter():
    rank = dist.get_rank()            # 当前进程编号
    world_size = dist.get_world_size()  # 总进程数

    # 构造本进程的输入：world_size 个张量组成的列表，每个张量长度为 2
    # 第 j 个张量 = [(rank+1)*1, (rank+1)*2] 的 (j+1) 次方
    # 不同 rank、不同块编号 j 得到不同的数值，模拟各进程持有不同的分块数据
    input_tensor = [
        torch.tensor(
            [(rank + 1) * i for i in range(1, 3)],  # 值为 (rank+1)*1, (rank+1)*2
            dtype=torch.float32
        ).cuda() ** (j + 1)           # 对第 j 块做 (j+1) 次方，使各块数值有所区分
        for j in range(world_size)    # 共 world_size 个块，对应 world_size 个进程
    ]

    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()  # 预分配输出缓冲区（长度=每块元素数）

    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")

    # ReduceScatter：所有 rank 的第 i 块按元素求和，结果只发给 rank i
    # 等价于先 AllReduce 再每个 rank 只保留自己那块，但通信量更少
    # 常用于张量并行中的梯度聚合：每张卡只需维护全局梯度的一个分片
    dist.reduce_scatter(
        output_tensor,               # 输出：接收规约后属于本 rank 的那块结果
        input_tensor,                # 输入：本进程贡献的 world_size 个块（列表）
        op=dist.ReduceOp.SUM)        # 规约操作为求和（也可用 MAX、PRODUCT 等）

    print(f"After ReduceScatter on rank {rank}: {output_tensor}")

init_process()
example_reduce_scatter()
