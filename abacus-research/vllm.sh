#!/bin/bash
set -e

export HF_HOME="/mnt/scratch/global/huggingface_cache/huggingface"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Paths and ports1
MODEL1_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL1_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL2_NAME ="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL3_NAME="Qwen/Qwen3-4B"
# MODEL3_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL1_NAME="Qwen/Qwen2.5-7B-Instruct"
#MODEL2_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
# "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL2_NAME="Qwen/Qwen3-0.6B"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"



EMBEDDING_PORT=8002
MODEL1_PORT=8005
#MODEL2_PORT=8006
# MODEL3_PORT=8007
LOG_DIR="/home/luizaregi/palimpzest/backup_morpheu/palimpzest/abacus-research/var/logs"

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
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port $EMBEDDING_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.05 \
    --max-model-len 8192 > "$LOG_DIR/embedding.log" 2>&1 &
PIDEmbedding=$!
wait_for_ready $EMBEDDING_PORT

echo "[INFO] Starting vLLM inference servers"
# Modelo 1 na GPU 0
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL1_NAME" \
    --port $MODEL1_PORT \
    --dtype bfloat16 \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
PID1=$!
wait_for_ready $MODEL1_PORT

# Modelo 2 na GPU 1
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL2_NAME" \
#     --port $MODEL2_PORT \
#     --dtype bfloat16 \
#     --max-model-len 20000 \
#     --gpu-memory-utilization 0.9\
#     2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT

# Modelo 3 na GPU 1
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL3_NAME" \
#     --port $MODEL3_PORT \
#     --dtype bfloat16 \
#     --max-model-len 35000 \
#     --gpu-memory-utilization 0.4 \
#     2>&1 | tee "$LOG_DIR/model_pequeno2.txt" &
# PID3=$!
# wait_for_ready $MODEL3_PORT

echo "[SUCCESS] Todos os modelos iniciados!"
wait $PID1 $PIDEmbedding
# wait $PID1 $PID2 $PIDEmbedding