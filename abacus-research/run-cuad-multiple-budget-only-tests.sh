#!/usr/bin/env bash

cd /app/abacus-research

SESSION="cuad-tests"
BASE_DIR="/app/abacus-research"

BUDGETS=(5 10 15 20 25 30)
STRATEGIES=(mab bay)
NUM_TESTS=3

STRATEGY_LIST="${STRATEGIES[*]}"
BUDGET_LIST="${BUDGETS[*]}"

for strategy in $STRATEGY_LIST; do
  for budget in $BUDGET_LIST; do
    for test_id in $(seq 1 $NUM_TESTS); do
      echo "Rodando: strategy=$strategy budget=$budget test=$test_id"

      python "$BASE_DIR/cuad-demo1.py" \
        --k 3 \
        --j 2 \
        --sentinel-execution-strategy "$strategy" \
        --sample-budget "$budget"

      sleep 5
    done
  done
done

echo "Testes finalizados"