
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

conda activate cx_vllm

vllm serve /home/eechengyang/CX/model/Qwen3-0.6B \
  --gpu-memory-utilization 0.78 \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --enable-chunked-prefill