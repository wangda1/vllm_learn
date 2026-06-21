"""
vLLM 量化实操:加载量化模型 + 显存/吞吐对比
=================================================================

前面 01–04 讲清了量化的"数值原理"。这个文件讲"在 vLLM 里到底怎么用",
并用一个可运行的基准把 fp16 vs fp8 的显存和吞吐量对比出来。

------------------------------------------------------------------------
vLLM 里用量化的三种入口
------------------------------------------------------------------------
1) 在线量化(online / on-the-fly):加载普通 fp16 模型,启动时实时量化权重。
   不需要预量化的 checkpoint,最适合快速试验。
        LLM(model=..., quantization="fp8")          # W8,权重在线压成 fp8
   局限:只能做 weight-only 的简单量化(fp8),做不了需要校准的 GPTQ/AWQ/W8A8。

2) 加载预量化 checkpoint(离线量化产物,推荐生产用):
   GPTQ / AWQ / compressed-tensors 格式的模型,vLLM 会自动识别,无需显式指定。
        LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ")    # 自动检测 quantization
   这些 checkpoint 由 llmcompressor / AutoAWQ / AutoGPTQ 离线生成(见文末 recipe)。

3) KV Cache 量化(和权重量化正交,可叠加):
        LLM(model=..., kv_cache_dtype="fp8")          # 长上下文省一半 KV 显存

------------------------------------------------------------------------
命令行等价写法(vllm serve)
------------------------------------------------------------------------
    vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8
    vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ            # AWQ 自动识别
    vllm serve <model> --kv-cache-dtype fp8

------------------------------------------------------------------------
⚠ 本机(RTX 3090 / Ampere sm_86)的重要事实
------------------------------------------------------------------------
3090 没有原生 FP8 计算单元(原生 FP8 要 Ada sm_89 / Hopper 起步)。
vLLM 会打印:
   "Your GPU does not have native support for FP8 computation ...
    Weight-only FP8 compression will be used leveraging the Marlin kernel."
也就是说:在 3090 上 fp8 退化成 weight-only(W8A16)——
   ✅ 省显存、decode(memory-bound)能受益;
   ❌ 拿不到 FP8 Tensor Core 的算力翻倍,prefill(compute-bound)不会变快。
这正好印证 DOC.md 第③问:"量化是否变快"必须分阶段、分 bound 看。

------------------------------------------------------------------------
用法
------------------------------------------------------------------------
分别跑两次,对比显存与 decode 吞吐:
    python quant/05_vllm_quant.py --quant none
    python quant/05_vllm_quant.py --quant fp8

(同进程内连续起两个 LLM 容易踩 vLLM 的资源回收坑,所以做成单配置脚本,
 跑两次人工对比最稳。)
"""

import argparse
import time

DEFAULT_MODEL = "/home/eechengyang/CX/model/Qwen3-0.6B"


def run(model: str, quant: str, n_prompts: int, gen_tokens: int):
    from vllm import LLM, SamplingParams

    quant_arg = None if quant == "none" else quant
    print(f"\n>>> 加载模型: quantization={quant_arg}")
    t0 = time.time()
    llm = LLM(
        model=model,
        quantization=quant_arg,        # None=原始 fp16/bf16;"fp8"=在线 weight-only
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        enforce_eager=True,            # 教学:关掉 cudagraph,数字更干净
    )
    load_s = time.time() - t0

    # 关于显存:vLLM V1 把模型跑在独立的 EngineCore 子进程里,主进程的
    # torch.cuda.memory_allocated() 读不到。真正的权重显存看启动日志这两行:
    #     "Model loading took X GiB memory"     <- 权重占用,量化收益就在这
    #     "GPU KV cache size: N tokens"          <- 固定显存预算下,权重越小→KV 越多
    # 本机实测(Qwen3-0.6B,gpu_memory_utilization=0.6):
    #     none: 权重 1.12 GiB,KV 109,136 tokens
    #     fp8 : 权重 0.72 GiB,KV 112,832 tokens   <- 权重 -36%,KV 容量更大
    # 这就是 weight-only 量化最实在的收益:同样的卡,装下更大的模型 / 更长的上下文。

    # decode 吞吐基准:固定输出长度,batch 一批 prompt,测每秒生成 token 数
    prompts = ["请用一句话解释什么是模型量化。"] * n_prompts
    params = SamplingParams(max_tokens=gen_tokens, temperature=0.0, ignore_eos=True)

    # 预热一次(触发 kernel 编译/缓存)
    llm.generate(["热身"], SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)

    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    dt = time.time() - t0
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)

    print("\n" + "=" * 60)
    print(f"  配置            : quantization={quant}")
    print(f"  模型加载耗时    : {load_s:.1f} s")
    print(f"  (权重/KV 显存见上面启动日志的 'Model loading took' / 'KV cache size')")
    print(f"  生成 token 总数 : {total_out}  ({n_prompts} prompts × {gen_tokens})")
    print(f"  decode 总耗时   : {dt:.2f} s")
    print(f"  decode 吞吐     : {total_out / dt:.1f} tok/s")
    print("=" * 60)
    print(f"  样例输出: {outs[0].outputs[0].text[:60]!r}")
    print("\n对比要点:")
    print("  - fp8 的'权重显存'明显小于 none(weight-only 的核心收益,本机 1.12→0.72 GiB)。")
    print("  - 但本机 decode 吞吐几乎持平:模型只有 0.6B,且 3090 上 fp8 走 Marlin")
    print("    (weight-only),拿不到原生 FP8 算力 → 没有算力红利。这正是 DOC.md 第③问:")
    print("    量化省的是显存/带宽,'变快'要看模型够不够大、阶段是不是 memory-bound。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default="none", choices=["none", "fp8"])
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=128)
    args = ap.parse_args()
    run(args.model, args.quant, args.n_prompts, args.gen_tokens)


if __name__ == "__main__":
    main()


# ===========================================================================
# 附:离线生成 GPTQ/AWQ/W8A8 checkpoint 的 recipe(用 llmcompressor)
# ===========================================================================
# 本机没装 llmcompressor;下面是标准流程,供你之后在有校准数据时复现。
# 在线 fp8 做不了的"需要校准的量化"(W4A16-GPTQ、W8A8-INT8)都靠它离线产出。
#
#   pip install llmcompressor
#
#   from llmcompressor.transformers import oneshot
#   from llmcompressor.modifiers.quantization import GPTQModifier
#   # W4A16:weight-only int4 + GPTQ,decode 省显存的主力方案
#   recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
#   oneshot(model="Qwen/Qwen2.5-7B-Instruct", dataset="open_platypus",
#           recipe=recipe, output_dir="Qwen2.5-7B-W4A16")
#   # 产物目录可直接喂给 vLLM:LLM(model="Qwen2.5-7B-W4A16")
#
#   # W8A8(SmoothQuant + INT8):prefill 想吃 INT8 算力时用,需要 SmoothQuant 预处理
#   from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
#   recipe = [SmoothQuantModifier(smoothing_strength=0.8),
#             GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
#
# 注意 ignore=["lm_head"]:对应 02 文件结论——lm_head 敏感,保留 fp16 不量化。
