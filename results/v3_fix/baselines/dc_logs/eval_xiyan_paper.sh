#!/usr/bin/env bash
# XiYan-SQL evaluation (3 seeds) on the paper datasets, 60 s query limit.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=60
export BASELINES_SUMMARY=results/v3_fix/baselines/summary_xiyan.csv
R=results/baselines_v3_gpu_package/xiyan_run/outputs
for ds in hospital tax restaurant sisu; do
  .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py xiyan --dataset $ds \
    --detailed $R/$ds/seed0/${ds}_xiyan_detailed_results.csv $R/$ds/seed1/${ds}_xiyan_detailed_results.csv \
               $R/$ds/seed2/${ds}_xiyan_detailed_results.csv > results/v3_fix/baselines/dc_logs/eval_xiyan_$ds.log 2>&1
  echo "$ds done $(date -Iseconds)"
done
