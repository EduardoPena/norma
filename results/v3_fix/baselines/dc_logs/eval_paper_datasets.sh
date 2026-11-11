#!/usr/bin/env bash
# Evaluate the DC discovery grid of the paper datasets with a 60 s query limit.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=60
for ds in hospital tax restaurant; do
  .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py profiling --dataset $ds \
    --csv baselines/1_profiling/results_dc_discovery/${ds}_v3_*.csv > results/v3_fix/baselines/dc_logs/eval_$ds.log 2>&1
  echo "$ds done $(date -Iseconds)"
done
