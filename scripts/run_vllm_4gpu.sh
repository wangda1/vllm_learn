export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=3,4,5,6

vllm serve /home/eechengyang/CX/model/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.78 \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --enable-chunked-prefill
