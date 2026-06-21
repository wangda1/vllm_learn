export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=3

mkdir -p ./my_traces

# 2. 在运行 bench 命令前设置环境变量，并指定 Profiler 性能分析文件保存目录
export VLLM_TORCH_PROFILER_DIR=./my_traces

vllm serve /home/eechengyang/CX/model/Qwen3-0.6B \
  --gpu-memory-utilization 0.78 \
  --max-num-seqs 32 \
  --max-model-len 8192 \
  --enable-chunked-prefill
