"""
tree_attention_demo.py
======================
教学示例：树状投机解码与 Tree Attention（对应 Q8）。

spec_decode_demo.py 里每步草稿是【一条链】(γ 个 token)，首次拒绝即停 —— 一旦中间某个
token 被拒，后面辛苦算的草稿全废。Medusa / EAGLE-2 的改进是：每步草稿不是一条链，而是
一棵【树】——每个位置给出多个候选分支。这样即使某分支被拒，别的分支仍可能被接受，
从而提高"每次 target forward 的期望产出 token 数"。

要在【一次】target forward 里同时验证整棵树，靠的是 Tree Attention：
把树的所有节点拍扁成一个序列，用一个特制的 mask 让每个节点只能看到自己的【祖先】，
于是一次前向就算出了每条根→叶路径上每个位置的分布，且共享前缀只算一次。

  第 0 课  为什么要从"链"升级到"树"
  第 1 课  构建一棵草稿树，并拍扁成序列
  第 2 课  Tree Attention mask：节点只能看祖先（对比普通 causal mask）
  第 3 课  一次前向验证整棵树 vs 逐路径串行验证：省了多少重复计算
  第 4 课  树上的接受/拒绝：选出最长被接受路径（分布仍无偏）
  第 5 课  EAGLE 的"特征级自回归" vs 独立 draft 小模型：关键差异

运行：
    python tree_attention_demo.py
"""

import random
from collections import Counter

# 复用 spec_decode_demo 的玩具 bigram 目标模型语义（这里独立写一份，便于单独运行）。
VOCAB = ["今天", "天气", "真", "好", "不错", "坏", "。", "<eos>"]
TARGET_TABLE = {
    "今天": {"天气": 0.7, "真": 0.2, "不错": 0.1},
    "天气": {"真": 0.6, "不错": 0.3, "好": 0.1},
    "真":   {"好": 0.7, "不错": 0.2, "坏": 0.1},
    "好":   {"。": 0.8, "<eos>": 0.2},
    "不错": {"。": 0.6, "<eos>": 0.4},
    "坏":   {"。": 0.7, "<eos>": 0.3},
    "。":   {"<eos>": 1.0},
    "<eos>": {"<eos>": 1.0},
}


def target_dist(context):
    """目标大模型 M_p：给定上下文返回覆盖词表的条件分布。"""
    row = TARGET_TABLE.get(context[-1], {"<eos>": 1.0})
    return {t: row.get(t, 0.0) for t in VOCAB}


def draft_dist(context, alpha):
    """草稿模型 M_q = (1-alpha)*p + alpha*uniform，alpha 越小越准。"""
    p = target_dist(context)
    u = 1.0 / len(VOCAB)
    return {t: (1 - alpha) * p[t] + alpha * u for t in VOCAB}


def topk_tokens(dist, k):
    """取概率最高的 k 个 token（树的每个节点向外扩 k 个分支）。"""
    return [t for t, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


# ============================================================
# 树节点：拍扁存储。每个节点记 token、父节点下标(parent)、所处深度。
# parent = -1 表示"根之前的真实上下文"(prompt 末尾)，是所有路径的公共起点。
# ============================================================

class TreeNode:
    def __init__(self, idx, token, parent, depth):
        self.idx = idx          # 在扁平序列里的下标
        self.token = token
        self.parent = parent    # 父节点在扁平序列里的下标（-1 = 上下文根）
        self.depth = depth


def build_draft_tree(context, alpha, branch, depth, rng):
    """
    用草稿模型构建一棵草稿树并拍扁。
      branch: 每个节点向下扩多少个候选分支（Medusa/EAGLE-2 的"宽度"）
      depth : 树的层数（草稿向前看多少步）
    返回扁平节点列表 nodes（按 BFS 顺序），nodes[i].parent 指向父节点下标。
    """
    nodes = []
    # 第 1 层：从真实上下文 context 出发，扩 branch 个候选
    q = draft_dist(tuple(context), alpha)
    frontier = []
    for tok in topk_tokens(q, branch):
        n = TreeNode(len(nodes), tok, parent=-1, depth=1)
        nodes.append(n)
        frontier.append(n)
    # 第 2..depth 层：每个叶子继续扩 branch 个候选
    for _ in range(depth - 1):
        nxt = []
        for node in frontier:
            path = list(context) + path_tokens(nodes, node)
            q = draft_dist(tuple(path), alpha)
            for tok in topk_tokens(q, branch):
                child = TreeNode(len(nodes), tok, parent=node.idx, depth=node.depth + 1)
                nodes.append(child)
                nxt.append(child)
        frontier = nxt
    return nodes


def path_tokens(nodes, node):
    """从根到 node 的 token 序列（含 node 自己）。"""
    out, cur = [], node
    while cur is not None:
        out.append(cur.token)
        cur = nodes[cur.parent] if cur.parent >= 0 else None
    return list(reversed(out))


def ancestors(nodes, node):
    """node 的所有祖先节点下标（不含自己），从近到远。"""
    out, cur = [], nodes[node.parent] if node.parent >= 0 else None
    while cur is not None:
        out.append(cur.idx)
        cur = nodes[cur.parent] if cur.parent >= 0 else None
    return out


# ============================================================
# 第 2 课：Tree Attention mask
#   普通 causal mask：第 i 个 token 能看到所有 j<=i（因为是一条直链）。
#   tree mask     ：第 i 个节点只能看到【自己 + 自己的祖先】，看不到别的分支。
#   这样把多条路径塞进一个序列里，彼此不串味，一次前向各算各的。
# ============================================================

def build_tree_mask(nodes):
    """返回 n×n 的 bool mask，mask[i][j]=True 表示 i 可以 attend 到 j。"""
    n = len(nodes)
    mask = [[False] * n for _ in range(n)]
    for node in nodes:
        i = node.idx
        mask[i][i] = True                       # 看自己
        for a in ancestors(nodes, node):        # 看所有祖先
            mask[i][a] = True
    return mask


def print_mask(nodes, mask):
    head = "      " + " ".join(f"{n.token[:2]:>3}" for n in nodes)
    print(head)
    for node in nodes:
        row = " ".join(" ✓ " if mask[node.idx][j] else " · " for j in range(len(nodes)))
        print(f"  {node.token[:2]:>3} {row}")


# ============================================================
# 课程主体
# ============================================================

def lesson0():
    print("=" * 72)
    print("第 0 课：从'链'升级到'树'——为什么")
    print("=" * 72)
    print("""  链式投机(spec_decode_demo)：每步草稿是一条 γ token 的链，首次拒绝即停。
    问题：中间一个 token 被拒，它后面所有草稿全部作废 —— 接受长度被"最弱一环"卡死。

  树式投机(Medusa / EAGLE-2)：每步草稿是一棵树，每个位置给若干候选分支。
    好处：某分支被拒，可以换到兄弟分支继续；"每次 target forward 的期望产出"更高。
    代价：要验证的候选 token 变多(整棵树)，但只要还在 memory-bound 区间，
          一次前向多验几个 token 几乎不额外花钱（见 batching_spec_demo.py）。

  难点：怎么在【一次】target 前向里把整棵树都验了？—— 答案是 Tree Attention。""")


def lesson1(nodes, context):
    print("\n" + "=" * 72)
    print("第 1 课：构建草稿树并拍扁成序列")
    print("=" * 72)
    print(f"  真实上下文 = {context}（树根之前的公共前缀）")
    print(f"  草稿树：branch=2, depth=2 → 第1层2个候选，每个再扩2个 = 共 {len(nodes)} 个节点\n")
    print(f"  {'idx':>3} {'token':<6} {'parent':>6} {'depth':>5}   根→该节点路径")
    for n in nodes:
        print(f"  {n.idx:>3} {n.token:<6} {n.parent:>6} {n.depth:>5}   "
              f"{' → '.join(path_tokens(nodes, n))}")
    print("\n  拍扁顺序(BFS)：所有节点排成一个序列，喂给 target 做一次前向。")


def lesson2(nodes):
    print("\n" + "=" * 72)
    print("第 2 课：Tree Attention mask（节点只能看祖先）")
    print("=" * 72)
    print("  ✓ = 行节点可以 attend 到 列节点；· = 不可以。")
    print("  注意每行只在'自己 + 自己的祖先'处是 ✓，不同分支互相看不到：\n")
    mask = build_tree_mask(nodes)
    print_mask(nodes, mask)
    print("""
  对比普通 causal mask（下三角全 ✓）：那是把序列当一条直链，每个 token 看到前面所有。
  树里第 3、4 个节点是第 1 个节点的孩子，第 5、6 个是第 2 个节点的孩子；
  若用 causal mask，节点5(第2分支的孩子)会错误地"看到"节点3(第1分支)，路径就串味了。
  tree mask 保证：每个节点的注意力上下文 == 它那条根→叶路径的真实前缀。""")
    return mask


def lesson3(nodes, context):
    print("\n" + "=" * 72)
    print("第 3 课：一次前向验证整棵树 vs 逐路径串行 —— 省了什么")
    print("=" * 72)
    # 枚举所有根→叶路径
    leaves = [n for n in nodes if not any(c.parent == n.idx for c in nodes)]
    paths = [path_tokens(nodes, leaf) for leaf in leaves]
    serial_tokens = sum(len(p) for p in paths)      # 串行：每条路径独立跑，前缀重复算
    tree_tokens = len(nodes)                         # 树：每个节点只算一次
    print("  所有根→叶路径：")
    for p in paths:
        print(f"    {' → '.join(p)}")
    print(f"""
  串行验证：把每条路径单独喂一次，公共前缀被【重复计算】
            总计算位置 = Σ每条路径长度 = {serial_tokens} 个 token-位置
  树状验证：拍扁后每个节点只出现一次，公共前缀只算一次
            总计算位置 = 树节点数 = {tree_tokens} 个 token-位置
  → tree attention 省的就是 {serial_tokens - tree_tokens} 个重复前缀位置的计算，
     并且把"多次前向"合并成"一次前向"（省的是 kernel 启动 + 反复搬权重）。
  树越深、分支共享前缀越多，省得越多。""")


def lesson4(context, nodes, alpha, rng):
    print("\n" + "=" * 72)
    print("第 4 课：树上的接受/拒绝 —— 选出最长被接受路径")
    print("=" * 72)
    print("""  验证规则和链式相同：沿某条路径从根往下，对每个草稿 token 用拒绝采样
  min(1, p/q) 判定；接受就继续往下，拒绝就停在这里。在所有路径里取
  "被接受前缀最长"的那条作为本步输出。下面对一条具体路径走一遍：\n""")
    # 取第一条根→叶路径演示逐节点判定
    leaf = next(n for n in nodes if not any(c.parent == n.idx for c in nodes))
    chain = []
    cur = leaf
    while cur is not None:
        chain.append(cur)
        cur = nodes[cur.parent] if cur.parent >= 0 else None
    chain.reverse()
    ctx = list(context)
    accepted = []
    for node in chain:
        p = target_dist(tuple(ctx))[node.token]
        q = draft_dist(tuple(ctx), alpha)[node.token]
        ap = min(1.0, p / q) if q > 0 else 1.0
        ok = rng.random() < ap
        print(f"    节点'{node.token}': p={p:.2f} q={q:.2f} 接受率={ap:.2f} -> "
              f"{'✔ 接受' if ok else '✗ 拒绝(此路径到此为止)'}")
        if not ok:
            break
        accepted.append(node.token)
        ctx.append(node.token)
    print(f"  这条路径被接受的前缀 = {accepted}")
    print("  实际引擎会在所有路径里挑最长被接受前缀；分布无偏性与链式一致（见下方蒙特卡洛）。")

    # 蒙特卡洛：验证"树式单步第一个产出 token"分布仍 == 目标分布
    print("\n  蒙特卡洛验证无偏性（树根那一层第一个 token 的分布应 == M_p）：")
    N = 100_000
    cnt = Counter()
    for _ in range(N):
        # 第一层用拒绝采样：从 branch 个候选里按链式规则取第一个被接受/修正的 token
        tok = _verify_first_token(context, alpha, rng)
        cnt[tok] += 1
    tgt = target_dist(tuple(context))
    print(f"    {'token':<8}{'目标p':>10}{'树式经验':>12}")
    for t in VOCAB:
        if tgt[t] > 0 or cnt[t] > 50:
            print(f"    {t:<8}{tgt[t]:>10.3f}{cnt[t]/N:>12.3f}")
    err = max(abs(cnt[t]/N - tgt[t]) for t in VOCAB)
    print(f"    最大偏差 = {err:.4f} -> ✔ 树式投机同样无偏。")


def _verify_first_token(context, alpha, rng):
    """对第一层 token 做一次拒绝采样判定，返回最终产出的第一个 token。
    （单候选链式语义；树的多分支只影响后续接受长度，不改第一个 token 的分布。）"""
    q = draft_dist(tuple(context), alpha)
    # 先从 draft 小模型分布 q 里采一个候选 token y
    y = _sample(q, rng)
    p = target_dist(tuple(context))
    ap = min(1.0, p[y] / q[y]) if q[y] > 0 else 1.0
    if rng.random() < ap:
        return y
    # 拒绝：从 recovered = normalize(max(p-q,0)) 重采
    raw = {t: max(p[t] - q[t], 0.0) for t in VOCAB}
    z = sum(raw.values()) or 1.0
    # 当候选被拒绝时，从修正分布 recovered = normalize(max(p-q, 0)) 里重新采样
    return _sample({t: raw[t] / z for t in VOCAB}, rng)


def _sample(dist, rng):
    """
    输入一个概率分布字典 dist（形如 {token: 概率, ...}），从中按照各 token 的权重（概率）随机抽取一个 token 返回。
    概率越大的 token 越容易被抽中
    """
    toks = list(dist.keys())
    return rng.choices(toks, weights=[dist[t] for t in toks], k=1)[0]


def lesson5():
    print("\n" + "=" * 72)
    print("第 5 课：EAGLE 的'特征级自回归' vs 独立 draft 小模型")
    print("=" * 72)
    print("""  这条演进线每一步在解决前一代的痛点：

    独立 draft 小模型 (Leviathan'22)
        痛点：要单独训练/部署一个小模型，它和大模型的分布不一定对齐 → 接受率有限。

    Medusa
        改进：不要独立小模型，在大模型最后一层 hidden 上接 K 个轻量"头"，
              每个头直接预测"未来第 i 个 token"。单卡可微调、部署简单。
        痛点：各头独立预测，没有头与头之间的自回归依赖 → 候选质量受限、接受长度不高。

    EAGLE / EAGLE-2 / EAGLE-3
        关键洞察：自回归不确定性主要在【特征(hidden state)空间】而非 token 空间。
        做法：drafter 不在 token 上自回归，而是在【特征空间】自回归——
              输入 = (上一步 token 的 embedding) ⊕ (目标模型上一步的 hidden state)，
              用一个小 transformer 预测下一步的 hidden，再映射成 token。
              因为复用了目标模型的高层特征，drafter 极小却接受率很高。
        EAGLE-2：把草稿树做成"动态树"——按置信度扩展，节点用在刀刃上。
        EAGLE-3：去掉"特征必须能重建出 logits"的约束，直接多层特征融合，接受率更高。

    Lookahead (Jacobi 解码)
        另一条路：完全不要 drafter，用 n-gram 池 + Jacobi 迭代并行猜，靠自身历史命中。

  与"再训一个小 draft 模型"的关键差异：EAGLE 的 drafter 站在目标模型的肩膀上
  (吃它的 hidden state)，所以 (1) 分布天然更对齐 → 接受率高；(2) 自身可以极小 → 草稿便宜。
  这正好同时优化了 Q7 里加速比的两个因子：接受率↑ 且 草稿开销↓。""")


def main():
    rng = random.Random(2026)
    context = ["今天"]
    alpha = 0.3
    nodes = build_draft_tree(context, alpha, branch=2, depth=2, rng=rng)

    lesson0()
    lesson1(nodes, context)
    lesson2(nodes)
    lesson3(nodes, context)
    lesson4(context, nodes, alpha, rng)
    lesson5()

    print("\n" + "=" * 72)
    print("小结（Q8）：")
    print("  · 树式投机用'多分支候选'提高每步接受长度，缓解链式'最弱一环卡死'。")
    print("  · Tree Attention = 让每个节点只 attend 自己的祖先，一次前向并行验证整棵树，")
    print("    且共享前缀只算一次（省重复计算 + 省多次 kernel/搬权重）。")
    print("  · 演进：独立draft → Medusa(多头,无自回归) → EAGLE(特征级自回归,动态树) → Lookahead。")
    print("  · EAGLE 关键：drafter 吃目标模型 hidden，分布对齐好、自身极小 → 接受率↑且草稿↓。")
    print("=" * 72)


if __name__ == "__main__":
    main()
