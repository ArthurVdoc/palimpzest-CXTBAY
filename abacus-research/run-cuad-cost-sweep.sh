#!/usr/bin/env bash
# =============================================================================
# run_cuad_cost_sweep.sh
#
# Objetivo: comparar MAB vs CXTBAY via curva Qualidade × Custo.
#
# Design do experimento:
#   - sample-budget fixo = 20
#   - Política: MaxQualityAtFixedCost
#   - max-cost varia de 0.25 a 0.41 em passos de 0.02  (9 valores)
#   - 5 seeds distintas por ponto  →  5 splits de contrato diferentes
#   - 2 estratégias: mab, cxtbay
#   - Total: 9 × 5 × 2 = 90 execuções
#
# Saída: opt-profiling-data/<exp-name>-metrics.json  para cada execução.
#   O nome do experimento encoda estratégia e max-cost, facilitando o plot.
# =============================================================================

SESSION="cuad-cost-sweep"
ENV_PATH="/scratch/global/abacus"
BASE_DIR="/scratch/global/abacus/abacus-research"

SAMPLE_BUDGET=20

# Estratégias a comparar
STRATEGIES=(mab cxtbay)

# Sweep de max-cost: 0.25 → 0.41, passo 0.02
MAX_COSTS=(0.25 0.27 0.29 0.31 0.33 0.35 0.37 0.39 0.41)

# Uma seed diferente para cada um dos 5 testes  →  5 splits distintos
SEEDS=(42 43 44 45 46)

# ---------------------------------------------------------------------------
# 1. Iniciar sessão tmux e subir o vLLM
# ---------------------------------------------------------------------------
tmux new-session -d -s "$SESSION" -n vllm

tmux send-keys -t "$SESSION":vllm "
source ~/miniconda3/bin/activate
conda activate $ENV_PATH
cd $BASE_DIR
./vllm.sh
" C-m

echo "Aguardando vLLM iniciar (300s)..."
sleep 300

# ---------------------------------------------------------------------------
# 2. Subir o scheduler
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n scheduler
tmux send-keys -t "$SESSION":scheduler "
source ~/miniconda3/bin/activate
conda activate $ENV_PATH
cd $BASE_DIR
python scheduler.py
" C-m

echo "Aguardando scheduler iniciar (15s)..."
sleep 15

# ---------------------------------------------------------------------------
# 3. Janela principal: loop de experimentos
# ---------------------------------------------------------------------------
echo "Iniciando sweep de max-cost (MAB vs CXTBAY)..."
tmux new-window -t "$SESSION" -n sweep

# Expande os arrays para strings literais antes de enviar ao tmux
STRATEGY_LIST="${STRATEGIES[*]}"
MAX_COST_LIST="${MAX_COSTS[*]}"
SEED_LIST="${SEEDS[*]}"

tmux send-keys -t "$SESSION":sweep "
source ~/miniconda3/bin/activate
conda activate $ENV_PATH
cd $BASE_DIR

TOTAL=0
DONE=0

# Conta o total de runs para acompanhamento
for strategy in $STRATEGY_LIST; do
  for max_cost in $MAX_COST_LIST; do
    for seed in $SEED_LIST; do
      TOTAL=\$((TOTAL + 1))
    done
  done
done
echo \"Total de execuções planejadas: \$TOTAL\"

for strategy in $STRATEGY_LIST; do
  for max_cost in $MAX_COST_LIST; do
    for seed in $SEED_LIST; do

      DONE=\$((DONE + 1))

      # Nome do experimento embute estratégia, max-cost e seed  →  fácil de parsear
      # Exemplo: cuad-sweep_mab_cost0.27_seed43
      EXP_NAME=\"cuad-sweep_\${strategy}_cost\${max_cost}_seed\${seed}\"

      echo \"\"
      echo \"=== [\$DONE/\$TOTAL] strategy=\$strategy  max-cost=\$max_cost  seed=\$seed ===\"
      echo \"Exp name: \$EXP_NAME\"

      python cuad-demo1.py \\
        --k 6 \\
        --j 4 \\
        --sample-budget $SAMPLE_BUDGET \\
        --max-cost \$max_cost \\
        --sentinel-execution-strategy \$strategy \\
        --seed \$seed \\
        --exp-name \$EXP_NAME

      echo \"Concluído: \$EXP_NAME\"
      sleep 5

    done   # seeds
  done     # max-costs
done       # strategies

echo ''
echo '=============================='
echo ' Sweep concluído com sucesso!'
echo '=============================='
tmux kill-session -t $SESSION
" C-m

tmux attach-session -t "$SESSION"