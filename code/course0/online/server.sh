#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="Qwen/Qwen3-1.7B"
PORT=13312
LOG_FILE="vllm.log"
PID_FILE="vllm.pid"

echo "[INFO] Ensuring model is cached ..."
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL_NAME')"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[INFO] Killing old vLLM process $OLD_PID ..."
    kill -TERM "$OLD_PID" && sleep 3 || true
  fi
  rm -f "$PID_FILE"
fi

echo "[INFO] Starting vLLM service on port $PORT ..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

# 等待1800s，如果服务还是没起来还是没通请检查自己的网络
for i in {1..1800}; do
  if nc -z 127.0.0.1 "$PORT"; then
    echo "[OK] vLLM service is ready at http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] Port $PORT not listening after 1800 s, check $LOG_FILE"
exit 1
