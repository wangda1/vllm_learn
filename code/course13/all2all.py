from mpi4py import MPI   # 导入 MPI 绑定库（Python 接口，封装底层 MPI 标准）
import numpy as np       # 导入 NumPy，用于构造发送/接收缓冲区

comm = MPI.COMM_WORLD                              # 获取全局通信子（包含所有进程）
rank, size = comm.Get_rank(), comm.Get_size()      # rank=当前进程编号；size=总进程数

# 准备发送缓冲区：形状 [size, k]，第 i 行将被发送给 rank i
k = 2
sendbuf = np.arange(
    rank * size * k,          # 起始值：各进程起点不同，保证每个进程的数据互不重叠
    (rank + 1) * size * k,    # 结束值（不含），共生成 size*k 个连续整数
    dtype='i'                 # 整型
).reshape(size, k)            # reshape 为 [size, k]：sendbuf[i] 是发给 rank i 的数据

print(f"Rank {rank} sendbuf=\n{sendbuf}")

# 预先分配接收缓冲区，形状与 sendbuf 相同
recvbuf = np.empty_like(sendbuf)  # recvbuf[i] 将存放从 rank i 收到的数据

# Alltoall 核心操作：全量交换
# 每个进程把 sendbuf[i] 发给 rank i，同时从 rank i 接收数据存入 recvbuf[i]
# 结果：recvbuf[i] = rank i 的 sendbuf[rank]（即对方专门给本进程准备的那行）
# 在 MoE EP 并行中，这是将 token 路由到对应专家所在 GPU 的关键原语（dispatch/combine 步骤）
comm.Alltoall(sendbuf, recvbuf)

print(f"Rank {rank} recvbuf=\n{recvbuf}")
