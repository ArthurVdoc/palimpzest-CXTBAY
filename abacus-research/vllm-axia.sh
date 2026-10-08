#!/bin/bash
set -e

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# MODEL_PATHS
MODEL1_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"
MODEL2_PATH="/scratch/hpc4ai/models/llama/Llama-3.3-70B-Instruct"
# MODEL3_PATH="/scratch/hpc4ai/models/qwen/Qwen3-235B-A22B"
MODEL4_PATH="/scratch/hpc4ai/models/openai/gpt-oss-120b"

# MODEL_NAMES
MODEL1_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL2_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL3_NAME="Qwen/Qwen3-235B-A22B"
MODEL4_NAME="openai/gpt-oss-120b"

EMBEDDING_PORT=8001
MODEL1_PORT=8005
MODEL2_PORT=8006
# MODEL3_PORT=8007
MODEL4_PORT=8008

LOG_DIR="${LOG_DIR:-$(pwd -P)/var/logs}"
mkdir -p "$LOG_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"


# wait_for_ready() {
#   local PORT=$1
#   echo "[INFO] Waiting for model on port $PORT..."
#   # Aumentei o timeout para não travar o script se o modelo demorar
#   for i in {1..60}; do
#     if curl -s http://localhost:$PORT/health > /dev/null; then
#       echo "[INFO] Model on port $PORT is ready!"
#       return 0
#     fi
#     sleep 5
#   done
#   echo "[ERROR] Model on port $PORT timed out!"
#   return 1
# }

# ------------ REVISED --------------------------------------
wait_for_ready() {
    # wait_for_ready <porta> <tentativas> <pid_a_vigiar>
    local port=$1 tries=${2:-180} pid=${3:-}
    echo "[INFO] aguardando porta $port (ate ~$((tries * 10))s)..."
    for ((i = 1; i <= tries; i++)); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] processo $pid morreu antes de servir na porta $port"
            return 1
        fi
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "[INFO] porta $port pronta apos ~$((i * 10))s"
            return 0
        fi
        sleep 10
    done
    echo "[ERROR] timeout esperando a porta $port"
    return 1
}
# ---------------------------------------------------------


echo "[INFO] Starting vLLM embedding server (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port $EMBEDDING_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.05 \
    --max-model-len 8192 > "$LOG_DIR/embedding.log" 2>&1 &
PIDEmbedding=$!
wait_for_ready $EMBEDDING_PORT

echo "[INFO] Starting vLLM inference servers"
# Modelo 1 na GPU 0
echo "[INFO] subindo Llama-3.1-8B (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL1_PATH" \
    --port $MODEL1_PORT \
    --dtype bfloat16 \
    --served-model-name $MODEL1_NAME \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.3 \
    >"$LOG_DIR/saida_VLLM_${MODEL1_PORT}.txt" 2>&1 &
PID1=$!
echo "[INFO]   llama8b PID=$PID1 (log: $LOG_DIR/vllm_${MODEL1_PORT}.log)"
wait_for_ready "$MODEL1_PORT" 240 "$PID1"

# Modelo 2 na GPU 0 e 1
echo "[INFO] subindo Llama-3.3-70B (GPU 0 e 1, TP=2)"
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL2_PATH" \
    --port $MODEL2_PORT \
    --dtype bfloat16 \
    --served-model-name $MODEL2_NAME \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.6 \
    --tensor-parallel-size 2 \
    >"$LOG_DIR/saida_VLLM_${MODEL2_PORT}.txt" 2>&1 &
PID2=$!
echo "[INFO]   llama70b PID=$PID2 (log: $LOG_DIR/vllm_${MODEL2_PORT}.log)"
wait_for_ready "$MODEL2_PORT" 240 "$PID2"

# # Modelo 3 na GPU 2, 3, 4 e 5
# echo "[INFO] subindo Qwen3-235B-A22B (GPUs 2, 3, 4 e 5, TP=4)"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL3_PATH" \
#     --port $MODEL3_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL3_NAME \
#     --max-model-len 40000 \
#     --gpu-memory-utilization 0.95 \
#     --tensor-parallel-size 4 \
#     >"$LOG_DIR/saida_VLLM_${MODEL3_PORT}.txt" 2>&1 &
# PID3=$!
# echo "[INFO]   qwen235B PID=$PID3 (log: $LOG_DIR/vllm_${MODEL3_PORT}.log)"
# wait_for_ready "$MODEL3_PORT" 240 "$PID3"

# Modelo 4 na GPU 6 e 7
echo "[INFO] subindo GPT-OSS-120B (GPUs 2 e 3, TP=2)"
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL4_PATH" \
    --port $MODEL4_PORT \
    --dtype bfloat16 \
    --served-model-name $MODEL4_NAME \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    >"$LOG_DIR/saida_VLLM_${MODEL4_PORT}.txt" 2>&1 &
PID4=$!
echo "[INFO]   gptoss120B PID=$PID4 (log: $LOG_DIR/vllm_${MODEL4_PORT}.log)"
wait_for_ready "$MODEL4_PORT" 240 "$PID4"

echo "[SUCCESS] Todos os modelos iniciados!"
#wait $PID1 $PIDEmbedding

wait