import torch
import time

device = 'cuda'
N = 2

# 1. 初始状态：x 为全 1
x = torch.ones(N, N, device=device)
y = torch.ones(N, N, device=device)
z = torch.empty(N, N, device=device)

# 捕获图
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

for _ in range(3):
    temp = torch.mm(x, y)
torch.cuda.synchronize()  # 确保预热彻底完成


with torch.cuda.stream(stream):
    # 捕获：z = x * y (即 1 * 1)
    with torch.cuda.graph(graph):
        z = torch.mm(x, y)
torch.cuda.synchronize()

captured_ptr = x.data_ptr()  # graph 记录的就是这个地址
# 注意：capture 只记录操作，不保证把结果写进输出缓冲区，要 replay 一次 z 才有意义的值
graph.replay()
torch.cuda.synchronize()
print("--- 初始捕获完成 ---")
print(f"捕获时 x 地址: {captured_ptr}, 首次 replay 后 z 结果(mm(1,1)=2):\n{z}")

print("\n--- 场景 1: 原地修改内容 x.fill_(2.0)（地址不变，内容变）---")
x.fill_(2.0)
graph.replay()
torch.cuda.synchronize()
print(f"x 地址: {x.data_ptr()} (==捕获地址: {x.data_ptr() == captured_ptr})")
print(f"z 结果(预期 mm(2,1)=4):\n{z}")

print("\n--- 场景 2: 重新赋值 x = torch.full(...)（生成新张量，地址变了）---")
x = torch.full((N, N), 5.0, device=device)
print(f"新 x 地址: {x.data_ptr()} (==捕获地址: {x.data_ptr() == captured_ptr})")
graph.replay()
torch.cuda.synchronize()
# graph 仍从旧地址（still 持有场景1写入的 2.0）读数据，所以 z 仍是 4，而不是 mm(5,1)=10
print(f"z 结果(仍是旧地址算出的 4，而非期望的 10):\n{z}")
print("结论：graph 绑定的是显存地址而非 Python 张量对象；换对象=换地址，replay 读不到新数据。")
