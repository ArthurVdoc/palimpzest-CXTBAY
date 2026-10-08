#!/usr/bin/env bash

SESSION="cuad-tests"
ENV_PATH="/scratch/global/abacus"
BASE_DIR="/scratch/global/abacus/abacus-research"

# Atualizado com os novos arrays e quantidade de testes
BUDGETS=(5 20 50)
# BUDGETS=(20)
STRATEGIES=(mab bay cxtbay)
# STRATEGIES=(cxtbay)
NUM_TESTS=5

tmux new-session -d -s $SESSION -n vllm

# vllm (Alterado para vllm.sh)
tmux send-keys -t $SESSION:vllm "
source ~/miniconda3/bin/activate 
source activate $ENV_PATH
cd $BASE_DIR
./vllm.sh
" C-m

echo "aguardando início do vllm..."
sleep 300

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

echo "iniciando testes cuad..."

tmux new-window -t $SESSION -n cuad

STRATEGY_LIST="${STRATEGIES[*]}"
BUDGET_LIST="${BUDGETS[*]}"

tmux send-keys -t $SESSION:cuad "
source ~/miniconda3/bin/activate
conda activate $ENV_PATH
cd $BASE_DIR

for strategy in $STRATEGY_LIST; do
  for budget in $BUDGET_LIST; do
    for test_id in \$(seq 1 $NUM_TESTS); do
      echo \"Rodando: strategy=\$strategy budget=\$budget test=\$test_id\"

      python cuad-demo1.py \
        --k 6 \
        --j 4 \
        --max-cost 0.36 \
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