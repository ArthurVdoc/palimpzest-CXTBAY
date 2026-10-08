#!/bin/bash
set -e

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# MODEL_PATHS
# MODEL1_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"
MODEL2_PATH="/scratch/hpc4ai/models/llama/Llama-3.3-70B-Instruct"
# MODEL3_PATH="/scratch/hpc4ai/models/qwen/Qwen3-235B-A22B"
# MODEL4_PATH="/scratch/hpc4ai/models/openai/gpt-oss-120b"

# MODEL_NAMES
# MODEL1_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL2_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL3_NAME="Qwen/Qwen3-235B-A22B"
# MODEL4_NAME="openai/gpt-oss-120b"

EMBEDDING_PORT=8002
# MODEL1_PORT=8005
MODEL2_PORT=8006
# MODEL3_PORT=8007
# MODEL4_PORT=8008
LOG_DIR="./var/logs"

mkdir -p "$LOG_DIR"

wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT..."
  # Aumentei o timeout para não travar o script se o modelo demorar
  for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
      echo "[INFO] Model on port $PORT is ready!"
      return 0
    fi
    sleep 5
  done
  echo "[ERROR] Model on port $PORT timed out!"
  return 1
}

echo "[INFO] Starting vLLM embedding server (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port $EMBEDDING_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.05 \
    --max-model-len 8192 > "$LOG_DIR/embedding.log" 2>&1 &
PIDEmbedding=$!
wait_for_ready $EMBEDDING_PORT

# echo "[INFO] Starting vLLM inference servers"
# # Modelo 1 na GPU 0
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL1_PATH" \
#     --port $MODEL1_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL1_NAME \
#     --max-model-len 50000 \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 2 \
#     2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
# PID1=$!
# wait_for_ready $MODEL1_PORT

# Modelo 2 na GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL2_PATH" \
    --port $MODEL2_PORT \
    --dtype bfloat16 \
    --served-model-name $MODEL2_NAME \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
PID2=$!
wait_for_ready $MODEL2_PORT

# # Modelo 3 na GPU 2 e 3
# CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL3_PATH" \
#     --port $MODEL3_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL3_NAME \
#     --max-model-len 50000 \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 2 \
#     2>&1 | tee "$LOG_DIR/model_pequeno2.txt" &
# PID3=$!
# wait_for_ready $MODEL3_PORT

# # Modelo 4 na GPU 4 e 5
# CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL4_PATH" \
#     --port $MODEL4_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL4_NAME \
#     --max-model-len 50000 \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 2 \
#     2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL4_PORT.txt" &
# PID4=$!
# wait_for_ready $MODEL4_PORT

echo "[SUCCESS] Todos os modelos iniciados!"
#wait $PID1 $PIDEmbedding

sleep 10
kill "$PID1" "$PID2" "$PID3" "$PID4" 2>/dev/null || true