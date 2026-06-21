import torch                        # 导入 PyTorch 核心库
import torch.distributed as dist   # 导入分布式通信模块

def init_process():
    dist.init_process_group(backend='nccl')  # 用 NCCL 后端初始化进程组（专为 GPU 间通信优化）
    torch.cuda.set_device(dist.get_rank())   # 每个进程绑定自己序号对应的 GPU，避免争抢

def example_scatter():
    if dist.get_rank() == 0:                 # 只有 rank 0 是"根节点"，负责准备并分发数据
        scatter_list = [
            torch.tensor([i + 1] * 5,        # 构造长度为 5、值全为 (i+1) 的张量
                dtype=torch.float32).cuda()  # 放到 GPU 上；rank i 收到的是值全为 (i+1) 的张量
            for i in range(dist.get_world_size())  # 为每个 rank 准备一份独立的数据
        ]
        print(f"Rank 0: Tensor to scatter: {scatter_list}")  # 打印待分发的数据列表
    else:
        scatter_list = None              # 非根节点无需提供 scatter_list，传 None 即可

    tensor = torch.empty(5, dtype=torch.float32).cuda()  # 所有 rank 预分配接收缓冲区（未初始化）

    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")  # 打印接收前的（未初始化）值
    dist.scatter(tensor, scatter_list, src=0)  # Scatter：rank 0 把 scatter_list[i] 发给 rank i
                                               # 每个进程只收到属于自己的那一份
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")   # 打印各自收到的数据

# Scatter 是"一对多"操作：根节点把不同的数据切片分别发给各进程，各进程只收到自己那份
init_process()
example_scatter()
