#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
course19 · Demo 2 —— KV cache 是怎么从 P「交接」到 D 的（1P1D 核心机制）
========================================================================

对应 DOC.md：第二章「设计方案」、第三章「端到端流程解析」

这是整门课最核心的一个 demo。它要证明一件事：

    ★ Decode 阶段的输出，与「KV cache 是本机算的，还是从远端传来的」**完全无关**。
      只要把 prefill 产生的 KV cache 原样搬到 D 节点，D 接着 decode 出来的 token
      和「单机一把跑完 prefill+decode」**逐 token 完全一致**。

这正是 PD 分离能成立的根本原因：KV cache 是 prefill 阶段唯一需要交接的「中间产物」，
把它交接好，decode 就对「KV 来自哪里」完全透明。

实现方式：用纯 Python 手写一个玩具 Transformer（带真实的 KV cache + 注意力），
不依赖 numpy / torch，直接运行：

    python kv_handoff_demo.py

我们会跑两条路径并逐 token 对比：
    (A) 单机路径   ：prefill + decode 都在一个对象里完成（基准答案）
    (B) PD 分离路径：P 对象只做 prefill 并把 KV「序列化发送」，
                     D 对象「接收并注入」KV 后接着 decode
"""

import math
import random

# --------------------------------------------------------------------------
# 0. 一个玩具 Transformer 的超参（小到可以手算，但结构是真的）
# --------------------------------------------------------------------------
VOCAB = 32      # 词表大小
DIM = 16        # 隐藏维度
LAYERS = 3      # 层数
SEED = 20240613


def make_matrix(rng, rows, cols, scale=0.3):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def matvec(mat, vec):
    """矩阵 (rows×cols) 乘向量 (cols) → (rows)。"""
    return [sum(mat[r][c] * vec[c] for c in range(len(vec))) for r in range(len(mat))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


# --------------------------------------------------------------------------
# 1. 模型权重：固定随机种子，保证两条路径用的是同一套权重
# --------------------------------------------------------------------------
class ToyWeights:
    def __init__(self, seed=SEED):
        rng = random.Random(seed)
        self.embed = make_matrix(rng, VOCAB, DIM)          # token → 向量
        self.Wq, self.Wk, self.Wv, self.Wo = [], [], [], []
        for _ in range(LAYERS):
            self.Wq.append(make_matrix(rng, DIM, DIM))
            self.Wk.append(make_matrix(rng, DIM, DIM))
            self.Wv.append(make_matrix(rng, DIM, DIM))
            self.Wo.append(make_matrix(rng, DIM, DIM))
        self.lm_head = make_matrix(rng, VOCAB, DIM)        # 向量 → logits


W = ToyWeights()


# --------------------------------------------------------------------------
# 2. KV Cache：每层、每个 token 存一份 (k, v)
#    这正是真实 vLLM 里 paged KV cache 存的东西（这里不分页，简化为 list）
# --------------------------------------------------------------------------
class KVCache:
    """一个请求的 KV cache：kv[layer] = [(k_0,v_0), (k_1,v_1), ...]，按 token 顺序追加。"""
    def __init__(self):
        self.kv = [[] for _ in range(LAYERS)]

    def append(self, layer, k, v):
        self.kv[layer].append((k, v))

    def length(self):
        return len(self.kv[0])

    # ↓↓↓ 下面两个方法就是「KV 交接」的本质：序列化 / 反序列化 ↓↓↓
    def serialize(self):
        """模拟 P 节点的 save_kv_layer / send_tensor：把 KV 打包成可传输的数据。
        真实 vLLM 里这一步是 extract_kv_from_layer(按 block_ids 取出切片) + NCCL 发送。"""
        return [list(layer) for layer in self.kv]

    @classmethod
    def deserialize(cls, payload):
        """模拟 D 节点的 start_load_kv / inject_kv_into_layer：把收到的 KV 注入本地缓存。
        真实 vLLM 里这一步是 NCCL 接收 + 按 D 自己分配的 block_ids 写进本地 KV pool。"""
        obj = cls()
        obj.kv = [list(layer) for layer in payload]
        return obj


# --------------------------------------------------------------------------
# 3. 单层注意力前向：给定输入向量 x 和该层的 KV cache，算出输出向量
#    （causal 注意力：当前 token 能看到自己和之前所有 token）
# --------------------------------------------------------------------------
def attention_layer(layer, x, cache, write_cache=True):
    q = matvec(W.Wq[layer], x)
    k = matvec(W.Wk[layer], x)
    v = matvec(W.Wv[layer], x)

    if write_cache:
        cache.append(layer, k, v)          # 把当前 token 的 K,V 写入 cache

    # 注意力：当前 q 对 cache 里所有 (k,v) 做 softmax 加权
    keys = [kv[0] for kv in cache.kv[layer]]
    vals = [kv[1] for kv in cache.kv[layer]]
    scale = 1.0 / math.sqrt(DIM)
    scores = [sum(q[i] * kk[i] for i in range(DIM)) * scale for kk in keys]
    weights = softmax(scores)

    ctx = [0.0] * DIM
    for w_ij, vv in zip(weights, vals):
        for i in range(DIM):
            ctx[i] += w_ij * vv[i]

    out = matvec(W.Wo[layer], ctx)
    # 残差 + 一个非线性，让层与层之间真正耦合
    return [math.tanh(a + b) for a, b in zip(x, out)]


def forward_token(token_id, cache, write_cache=True):
    """跑完整个模型的一个 token，返回最后一层的隐藏向量。"""
    x = list(W.embed[token_id])
    for layer in range(LAYERS):
        x = attention_layer(layer, x, cache, write_cache=write_cache)
    return x


def logits_to_token(hidden):
    """用 lm_head 把隐藏向量映射成 logits，取 argmax（贪心解码）。"""
    logits = matvec(W.lm_head, hidden)
    return max(range(VOCAB), key=lambda t: logits[t])


# --------------------------------------------------------------------------
# 4. prefill：把整段 prompt 喂进去，产出最后一个 token 的隐藏态 + 完整 KV cache
# --------------------------------------------------------------------------
def prefill(prompt_ids, cache):
    hidden = None
    for tid in prompt_ids:
        hidden = forward_token(tid, cache, write_cache=True)
    return hidden   # 最后一个 prompt token 的隐藏态 → 用来预测第一个生成 token


# --------------------------------------------------------------------------
# 5. decode：从一个已有的 KV cache 出发，自回归生成 N 个 token
# --------------------------------------------------------------------------
def decode(first_hidden, cache, n_tokens):
    out = []
    hidden = first_hidden
    for _ in range(n_tokens):
        next_tok = logits_to_token(hidden)
        out.append(next_tok)
        hidden = forward_token(next_tok, cache, write_cache=True)  # 把新 token 也写进 KV
    return out


# ==========================================================================
# 路径 A：单机一把梭（基准答案）
# ==========================================================================
def run_monolithic(prompt_ids, n_gen):
    cache = KVCache()
    first_hidden = prefill(prompt_ids, cache)
    return decode(first_hidden, cache, n_gen)


# ==========================================================================
# 路径 B：PD 分离 —— P 只 prefill 并交接 KV，D 接收后 decode
# ==========================================================================
class PrefillWorker:
    """P 节点：只负责 prefill，产出 KV，然后「发送」给 D。对应 kv_producer。"""
    def prefill_and_handoff(self, prompt_ids):
        cache = KVCache()
        first_hidden = prefill(prompt_ids, cache)
        # 真实 vLLM：max_tokens 被 proxy 强制改成 1，P 只算 prefill + 吐 1 个 token 就停。
        first_token = logits_to_token(first_hidden)
        payload = cache.serialize()          # ← save_kv_layer + NCCL send
        print(f"  [P] prefill 完成：{len(prompt_ids)} 个 prompt token，"
              f"KV 长度={cache.length()}，打包 {len(payload)} 层准备发送")
        return first_token, payload


class DecodeWorker:
    """D 节点：接收 KV，注入本地缓存，接着 decode。对应 kv_consumer。"""
    def receive_and_decode(self, prompt_ids, first_token, payload, n_gen):
        # ① 注入收到的 KV（这里 D 直接复用 P 的 KV，跳过了对 prompt 的重复 prefill）
        cache = KVCache.deserialize(payload)     # ← start_load_kv + inject_kv_into_layer
        print(f"  [D] 收到并注入 KV，本地 KV 长度={cache.length()}，"
              f"无需重新 prefill {len(prompt_ids)} 个 token")
        # ② 关键细节：D 需要先把「第一个 token」过一遍模型拿到它的隐藏态，再继续生成。
        #    （真实 vLLM 中 get_num_new_matched_tokens 返回 len(prompt)-1，
        #     最后 1 个 token 在 D 上重算，正是为了拿到这个用于 decode 的隐藏态。）
        hidden = forward_token(first_token, cache, write_cache=True)
        rest = decode(hidden, cache, n_gen - 1)
        return [first_token] + rest


def run_pd_disaggregated(prompt_ids, n_gen):
    p = PrefillWorker()
    d = DecodeWorker()
    first_token, payload = p.prefill_and_handoff(prompt_ids)
    return d.receive_and_decode(prompt_ids, first_token, payload, n_gen)


# ==========================================================================
# main：两条路径对比，逐 token 验证一致
# ==========================================================================
if __name__ == "__main__":
    print(__doc__)

    rng = random.Random(7)
    prompt = [rng.randrange(VOCAB) for _ in range(12)]
    N_GEN = 16

    print("=" * 70)
    print(f"prompt (token ids) = {prompt}")
    print(f"要生成 {N_GEN} 个 token\n")

    print("路径 A：单机一把梭 (prefill + decode 同处)")
    out_mono = run_monolithic(prompt, N_GEN)
    print(f"  → 生成: {out_mono}\n")

    print("路径 B：PD 分离 (P 只 prefill 并交接 KV，D 接收后 decode)")
    out_pd = run_pd_disaggregated(prompt, N_GEN)
    print(f"  → 生成: {out_pd}\n")

    print("=" * 70)
    if out_mono == out_pd:
        print("✅ 逐 token 完全一致！")
        print("""
这证明了 PD 分离的根本前提：
  · prefill 阶段唯一需要交接给 decode 的，就是 KV cache（+ 最后一个 token）。
  · 只要 KV 被原样搬到 D，decode 对「KV 来自本机还是远端」完全无感知，
    输出与单机逐 token 相同。
  · 因此可以放心地把 P/D 拆到不同硬件上，用 NCCL/RDMA 把 KV 搬过去即可。

真实 vLLM 的对应关系：
  serialize()   ≈ P 侧 save_kv_layer() → extract_kv_from_layer() → NCCL send_tensor()
  deserialize() ≈ D 侧 start_load_kv() → 后台线程 recv → inject_kv_into_layer()
  「第一个 token 在 D 上重算」≈ get_num_new_matched_tokens() 返回 len(prompt)-1""")
    else:
        # 不应发生
        print("❌ 不一致，demo 有 bug")
        diff = [i for i, (a, b) in enumerate(zip(out_mono, out_pd)) if a != b]
        print(f"   首个不一致位置: {diff[:5]}")

    print("\n下一课 → block_remap_demo.py：P 和 D 的物理块号不同，KV 怎么对上号。\n")
