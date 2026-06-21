# PagedAttention 的非连续物理块如何参与计算

> 主题:vLLM PagedAttention 下,KV cache 用逻辑块组织、物理块可能不连续,这些非连续的 tensor 如何送进 GEMM 类算子计算?decode 阶段是否通过 NCCL 收集所有 `num_computed_tokens` 的 KV 再拼成一块大连续 tensor 送 GEMM?

---

## 0. 一句话结论

问题里藏着**两个反了的前提**,纠正完答案就清楚:

1. **非连续物理块不会被拼成大连续 tensor 再送 cuBLAS GEMM**——PagedAttention 的全部意义就是【不物化连续 KV】,而是用 `block_table` 在自定义融合核内部逐块间接寻址。
2. **decode 阶段实例内 KV 从不通过 NCCL 收集组装**——NCCL 要么传**激活**(TP all-reduce),要么是 PD 分离时**一次性**交接 KV,绝不是"每步收集 KV 拼连续再送 GEMM"。

---

## 1. 先分清:哪些算子碰 paged KV,哪些根本不碰

一个 decoder layer 里有一堆 GEMM,但**只有 attention 一处碰分页 KV cache**,其余 GEMM 全在**连续的激活张量**上做,跟分页毫无关系。

| 算子 | 输入 | 是否碰 paged KV |
|---|---|---|
| QKV proj (`q/k/v_proj`) | `hidden_states [num_tokens, d]` 连续 | ❌ 普通 cuBLAS GEMM |
| **attention (QK^T · softmax · V)** | Q(连续) + K/V(**分页、物理不连续**) | ✅ 唯一一处 |
| O proj (`o_proj`) | attn 输出 `[num_tokens, d]` 连续 | ❌ 普通 GEMM |
| MLP (gate / up / down) | `[num_tokens, d]` 连续 | ❌ 普通 GEMM |

**关键推论**:所谓"物理块不连续怎么喂 GEMM",**只发生在 attention 这一步**。QKV / MLP 那些大 GEMM 拿的是当前这一拍要算的 token 的隐藏态,永远是规整连续张量,cuBLAS 直接吃。

**时序细节**:
- QKV proj 先算出**当前 token** 的连续 K/V;
- `reshape_and_cache` kernel 按 `slot_mapping` 把新 K/V **scatter 写进** paged cache 的物理槽位;
- attention 再去读整段历史 KV。

---

## 2. 前提一纠正:非连续物理块不拼连续,而是 block_table 在 kernel 内部 gather

这正是 PagedAttention 的**全部意义**——避免"拼一块大连续 tensor"。如果先 gather 拼连续再算,就退回到没分页的世界,显存碎片/浪费问题原样回来。

**实际做法**:attention **不是调 cuBLAS GEMM**,而是一个**自定义融合 CUDA 核**:
- vLLM 的 `paged_attention` kernel(来自 `_C`),或
- FlashAttention / FlashInfer 的 paged 变体(来自 `_vllm_fa2_C` / `_vllm_fa3_C`)。

它多吃一个输入参数:**`block_table`(逻辑块号 → 物理块号的映射表)**。kernel 内部逻辑:

```
for 每个逻辑块 i in 这条序列的 block_table:
    phys = block_table[i]                    # 查物理块号
    load K_block, V_block from cache[phys]   # 直接按物理地址取这一块
    QK^T(这一块) → online-softmax 累加 → 乘 V_block 累加
```

要点:
- **"非连续"被 block_table 的一层指针间接寻址吸收掉**,gather 发生在 kernel 的寄存器 / 共享内存层面;
- **全程不在显存里物化出一个连续的大 KV 张量**;
- Q@K^T 和 softmax@V 是 **kernel 内部的小块矩阵乘**,不是一次 cuBLAS 大 GEMM(FlashAttention 式 online softmax 逐块累加)。

**与 PD 分离的"gather 成连续"不矛盾**:
- PD 分离里的 gather 是**跨节点传输前**为了走 NCCL 才临时拼连续,传到 D 节点又 scatter 回各自的物理块;
- **真正算 attention 时仍然是 paged、不连续**;
- 一句话:**传输 ≠ 计算**。

---

## 3. 前提二纠正:decode 阶段没有"NCCL 收 KV 再组装"这回事

在**单个推理实例内部**,KV cache **从不**通过 NCCL 跨卡收集。两种 NCCL 用法必须分开:

### (1) 张量并行(TP)内的 NCCL
- TP 把 attention 的 **head 切到不同卡**,每张卡只持有**自己那几个 head 的 KV**,就在本地 paged cache 里;
- decode 时每卡各算各 head 的 attention,**KV 全程不动、不传**;
- NCCL 只在 **o_proj 之后做一次 all-reduce**,把各卡的部分输出加起来——**传的是激活** `[num_tokens, d]`,不是 KV。

### (2) PD 分离的 NCCL
- 那是 **prefill 节点 → decode 节点**一次性交接 KV;
- **只发生在请求刚进 decode 前**,不是 decode 每一步;
- 交接完,decode 就在本地 paged cache 上自回归,跟 KV 来自哪儿无关。

---

## 4. decode 阶段的真实计算流

```
新 token hidden_state [1, d]   (连续)
  → QKV proj (cuBLAS GEMM)      → q [1,h,dh], 新 k/v [1,h,dh]
  → reshape_and_cache           : 把新 k/v 写进本地 paged cache 的尾块(scatter)
  → paged attention kernel      : q 去读「本地、分页、不连续」的全部历史 KV
                                  (block_table 间接寻址,kernel 内逐块 gather)
  → o_proj (GEMM)               → [TP 时] NCCL all-reduce 激活
  → MLP (GEMM)                  → all-reduce
```

**没有任何一步把 `num_computed_tokens` 的 KV 收集组装成大连续 tensor 再送 GEMM**。恰恰相反,整个设计就是为了不做这件事。

---

## 5. 总结(两条记牢)

1. **碰分页 KV 的只有 attention**,它走**自定义融合核 + block_table 间接寻址**,kernel 内部逐块 gather,**不物化连续 KV、不调 cuBLAS GEMM**;其余所有 GEMM 都在连续激活上跑,与分页无关。
2. **decode 里 NCCL 传的是激活(TP all-reduce)或一次性交接 KV(PD 分离)**,**绝不是每步把 KV 收集拼连续**。

---

## 附:相关源码 / 概念出处

- attention 核与编译扩展:`_C`(PagedAttention)、`_vllm_fa2_C` / `_vllm_fa3_C`(FlashAttention paged 变体)。
- 写 KV 进 cache:`reshape_and_cache` kernel + `slot_mapping`。
- 逻辑块↔物理块映射:`block_table`。
- PD 分离里 KV 的 gather/scatter 传输:`extract_kv_from_layer`(P 侧 gather)→ NCCL → `inject_kv_into_layer`(D 侧 scatter),见
  `vllm/distributed/kv_transfer/kv_connector/v1/p2p/`。
