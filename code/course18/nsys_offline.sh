
export CUDA_VISIBLE_DEVICES=6

nsys profile \
    --trace-fork-before-exec=true \
    --output=/home/eechengyang/CX/vllm_learn/code/course18/reports \
    --force-overwrite=true \
    vllm bench latency --model /home/eechengyang/CX/model/Qwen3-0.6B \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --max-num-seqs 32 \
    --input-len 512 \
    --output-len 8