#!/bin/bash
set -e

export HF_HOME="/scratch/global/huggingface_cache/huggingface"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}" 
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


# Paths
# "Qwen/Qwen2.5-1.5B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct"
# "Qwen/Qwen3-0.6B"
# "Qwen/Qwen2.5-7B-Instruct"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-14B-Instruct-AWQ" 
# "casperhansen/llama-3-8b-instruct-awq"

EMBEDDING_PORT=8010

MODEL1_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODEL1_PORT=8006

# MODEL2_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL2_PORT=8005

LOG_DIR="/mnt/scratch/global/palimpzest/abacus-research/var/logs"
mkdir -p "$LOG_DIR"

wait_for_ready(){
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}

# 1. SERVER DE EMBEDDING (Porta 8010)
echo "[INFO] Starting vLLM embedding server"
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port $EMBEDDING_PORT \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.05 & # ~1.2GB
PIDEmbedding=$!
wait_for_ready $EMBEDDING_PORT

# 7B ou 14 ou 7
# 2. MODELO 1: QWEN 0.5B (Porta 8011 - O Coelho)
echo "[INFO] Starting $MODEL1_NAME"
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL1_NAME" \
    --port $MODEL1_PORT \
    --max-model-len 32000 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager & 
    # --disable-log-requests 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" & 
PID1=$!
wait_for_ready $MODEL1_PORT

# 3. MODELO 2: LLAMA 8B AWQ (Porta 8012 - A Tartaruga)
# echo "[INFO] Starting $MODEL2_NAME"
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL2_NAME" \
#     --port $MODEL2_PORT \
#   --max-model-len 8192 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.90 &
#     # --disable-log-requests 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT

echo "[SUCCESS] Todos os modelos estão rodando"
echo "Embedding: $EMBEDDING_PORT | $MODEL1_NAME: $MODEL1_PORT | $MODEL2_NAME: $MODEL2_PORT"

wait $PID1 $PID2 $PIDEmbedding