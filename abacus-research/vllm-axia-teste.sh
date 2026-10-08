# #!/bin/bash

# echo "==============================================================="
# echo "🔍 DIAGNÓSTICO: LISTANDO OS MODELOS DISPONÍVEIS NO SCRATCH"
# echo "==============================================================="
# echo "Conteúdo de /scratch/hpc4ai/models/:"
# ls -F /scratch/hpc4ai/models/

# echo "Conteúdo de /scratch/hpc4ai/models/qwen/:"
# ls -F /scratch/hpc4ai/models/qwen/ 2>/dev/null || echo "Pasta qwen não encontrada ou vazia!"
# echo "==============================================================="

# set -e

# export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# # --- PATHS DOS MODELOS (Revisados conforme estrutura do Scratch) ---
# MODEL1_PATH="/scratch/hpc4ai/models/qwen/Qwen3-235B-A22B"
# MODEL2_PATH="/scratch/hpc4ai/models/llama/Llama-3.3-70B-Instruct"

# # --- NAMES DOS MODELOS ---
# MODEL1_NAME="Qwen/Qwen3-235B-A22B"
# MODEL2_NAME="meta-llama/Llama-3.3-70B-Instruct"

# # --- PORTAS ---
# EMBEDDING_PORT=8002
# MODEL1_PORT=8005  # Porta do Qwen (MODEL1)
# MODEL2_PORT=8006  # Porta do Llama
# LOG_DIR="./var/logs"

# mkdir -p "$LOG_DIR"

# wait_for_ready() {
#   local PORT=$1
#   echo "[INFO] Waiting for model on port $PORT..."
#   for i in {1..120}; do
#     if curl -s http://localhost:$PORT/health > /dev/null; then
#       echo "[INFO] Model on port $PORT is ready!"
#       return 0
#     fi
#     sleep 5
#   done
#   echo "[ERROR] Model on port $PORT timed out!"
#   return 1
# }

# # =====================================================================
# # SERVIDOR DE EMBEDDING (GPU 0 - Compartilhada com o início do Qwen)
# # =====================================================================
# echo "[INFO] Starting vLLM embedding server (GPU 0)"
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model nomic-ai/nomic-embed-text-v1 \
#     --port $EMBEDDING_PORT \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.05 \
#     --max-model-len 8192 > "$LOG_DIR/embedding.log" 2>&1 &
# PIDEmbedding=$!
# wait_for_ready $EMBEDDING_PORT

# echo "[INFO] Starting vLLM inference servers"

# # =====================================================================
# # MODELO 1: QWEN3 235B (GPUs 0, 1, 2, 3 - Tensor Parallel = 4)
# # =====================================================================
# # CORREÇÃO: Alterado tensor-parallel de 5 para 4 (potência de 2 obrigatória)
# # Aloca 4 GPUs físicas para o processo do Qwen
# echo "[INFO] Starting Qwen Model (MODEL1 nas GPUs 0, 1, 2 e 3)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL1_PATH" \
#     --port $MODEL1_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL1_NAME \
#     --max-model-len 32768 \
#     --gpu-memory-utilization 0.95 \
#     --tensor-parallel-size 4 \
#     2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
# PID1=$!
# wait_for_ready $MODEL1_PORT

# # =====================================================================
# # MODELO 2: LLAMA 3.3 70B (GPUs 4 e 5 - Tensor Parallel = 2)
# # =====================================================================
# # CORREÇÃO: Movido para as GPUs 4 e 5 para isolar completamente do Qwen
# echo "[INFO] Starting Llama 3.3 70B (MODEL2 nas GPUs 4 e 5)"
# CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL2_PATH" \
#     --port $MODEL2_PORT \
#     --dtype bfloat16 \
#     --served-model-name $MODEL2_NAME \
#     --max-model-len 50000 \
#     --gpu-memory-utilization 0.90 \
#     --tensor-parallel-size 2 \
#     2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT

# echo "[SUCCESS] Todos os modelos iniciados com isolamento de GPUs!"
# wait

#!/bin/bash
set -e

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?Defina a variavel de ambiente HF_TOKEN}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Configurações do Llama 70B
MODEL1_PATH="/scratch/hpc4ai/models/llama/Llama-3.3-70B-Instruct"
MODEL1_NAME="meta-llama/Llama-3.3-70B-Instruct"

EMBEDDING_PORT=8002
MODEL1_PORT=8005

LOG_DIR="${LOG_DIR:-$(pwd -P)/var/logs}"
mkdir -p "$LOG_DIR"

wait_for_ready() {
    local port=$1 tries=${2:-180} pid=${3:-}
    echo "[INFO] Aguardando serviço na porta $port..."
    for ((i = 1; i <= tries; i++)); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] Processo $pid morreu antes de abrir a porta $port!"
            echo "[ERROR] Confira os detalhes do erro no log: $LOG_DIR/saida_VLLM_${port}.txt"
            return 1
        fi
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "[INFO] Serviço na porta $port está PRONTO!"
            return 0
        fi
        sleep 10
    done
    echo "[ERROR] Timeout esperando a porta $port!"
    return 1
}

echo "=== 1. Subindo servidor de Embedding (Nomic) na GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model nomic-ai/nomic-embed-text-v1 \
    --port $EMBEDDING_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.05 \
    --max-model-len 8192 > "$LOG_DIR/saida_VLLM_${EMBEDDING_PORT}.txt" 2>&1 &
PID_EMBED=$!

wait_for_ready "$EMBEDDING_PORT" 240 "$PID_EMBED"

echo "=== 2. Subindo Llama-70B distribuído em 2 GPUs (Tensor Parallelism = 2) ==="
# Se tiver 3 GPUs no pod, use GPU 1 e 2 aqui. Se tiver apenas 2, garanta gpu-memory-utilization menor
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL1_PATH" \
    --port $MODEL1_PORT \
    --dtype bfloat16 \
    --served-model-name "$MODEL1_NAME" \
    --tensor-parallel-size 2 \
    --max-model-len 40000 \
    --gpu-memory-utilization 0.80 > "$LOG_DIR/saida_VLLM_${MODEL1_PORT}.txt" 2>&1 &
PID1=$!

wait_for_ready "$MODEL1_PORT" 300 "$PID1"

echo "=== [SUCCESS] Embedding e LLM iniciados e prontos para uso! ==="

wait $PID_EMBED $PID1