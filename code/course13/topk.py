import torch                          # 导入 PyTorch 核心库
import torch.nn.functional as F       # 导入函数式 API（含 softmax）

# 输入 router_logits 张量（来自 gate 层输出），形状: [num_tokens, num_experts]
router_logits = torch.tensor([
    [0.1, 0.2, 0.5, 0.1, 0.05, 0.03, 0.01, 0.01],  # token 0 对 8 个专家的打分
    [0.3, 0.4, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01],  # token 1 对 8 个专家的打分
    [0.05, 0.1, 0.1, 0.1, 0.3, 0.25, 0.05, 0.05],  # token 2 对 8 个专家的打分
    [0.2, 0.1, 0.4, 0.2, 0.05, 0.03, 0.01, 0.01],  # token 3 对 8 个专家的打分
])

# 1. Top-k 选择：每个 token 选出分数最高的 k 个专家
top_k = 2                                            # MoE 超参：每个 token 路由到 2 个专家
topk_logits, topk_ids = torch.topk(                  # 沿最后一维取最大的 k 个值及其索引
    router_logits, k=top_k, dim=-1)                  # topk_logits: [4,2] 原始分；topk_ids: [4,2] 专家编号

# 2. 权重重新归一化：仅在选出的 top-k 内做 softmax（不是全部 8 个专家）
topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
# 每行两个权重之和 = 1，代表该 token 分配给每个选中专家的融合比例

# --- 输出结果 ---
# topk_weights 是对 topk_logits 应用 softmax 的结果
# tensor([[0.5744, 0.4256],  <-- softmax([0.5, 0.2])  token0 → 专家2(57%), 专家1(43%)
#         [0.5250, 0.4750],  <-- softmax([0.4, 0.3])  token1 → 专家1(52%), 专家0(48%)
#         [0.5125, 0.4875],  <-- softmax([0.3, 0.25]) token2 → 专家4(51%), 专家5(49%)
#         [0.5498, 0.4502]]) <-- softmax([0.4, 0.2])  token3 → 专家2(55%), 专家0(45%)
print("topk_weights:\n", topk_weights)

# topk_ids 是 router_logits 中最高分的专家索引
# tensor([[2, 1],  <-- 专家 2 (0.5), 专家 1 (0.2)
#         [1, 0],  <-- 专家 1 (0.4), 专家 0 (0.3)
#         [4, 5],  <-- 专家 4 (0.3), 专家 5 (0.25)
#         [2, 0]]) <-- 专家 2 (0.4), 专家 0 (0.2)
# 注意：token 3 中有两个分数为 0.2 的专家（索引 0 和 3），torch.topk 返回索引较小的那个
print("\ntopk_ids:\n", topk_ids)
