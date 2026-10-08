#!/usr/bin/env bash

SESSION="biodex-tests"
ENV_PATH="/scratch/global/abacus"
BASE_DIR="/scratch/global/abacus/abacus-research"

BUDGETS=(30 45 60 75 90 105 120)
STRATEGIES=(mab bay)
NUM_TESTS=3

tmux new-session -d -s $SESSION -n vllm

# vllm
tmux send-keys -t $SESSION:vllm "
source ~/miniconda3/bin/activate 
source activate $ENV_PATH
cd $BASE_DIR
./vllm.sh
" C-m

echo "aguardando início do vllm..."
sleep 180

# captura PID do vllm.sh
VLLM_PIDS=$(pgrep -f "${BASE_DIR}/vllm.sh" || true)
if [[ -n "$VLLM_PIDS" ]]; then
  echo "vllm.sh PIDs: $VLLM_PIDS"
else
  echo "vllm.sh PIDs não encontrados ainda"
fi

# scheduler
tmux new-window -t $SESSION -n scheduler
tmux send-keys -t $SESSION:scheduler "
source ~/miniconda3/bin/activate 
source activate $ENV_PATH
cd $BASE_DIR
python scheduler.py
" C-m

echo "aguardando início do scheduler..."
sleep 15

echo "iniciando testes biodex..."

tmux new-window -t $SESSION -n biodex

STRATEGY_LIST="${STRATEGIES[*]}"
BUDGET_LIST="${BUDGETS[*]}"

tmux send-keys -t $SESSION:biodex "
source ~/miniconda3/bin/activate
conda activate $ENV_PATH
cd $BASE_DIR

for strategy in $STRATEGY_LIST; do
  for budget in $BUDGET_LIST; do
    for test_id in \$(seq 1 $NUM_TESTS); do
      echo \"Rodando: strategy=\$strategy budget=\$budget test=\$test_id\"

      python biodex-demo.py \
        --k 3 \
        --j 3 \
        --progress \
        --sentinel-execution-strategy \$strategy \
        --sample-budget \$budget

      sleep 5
    done
  done
done

echo \"Testes finalizados\"
tmux kill-session -t $SESSION
" C-m

tmux attach-session -t $SESSION