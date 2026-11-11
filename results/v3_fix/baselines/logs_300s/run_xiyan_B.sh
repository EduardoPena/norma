#!/usr/bin/env bash
# XiYan-SQL variant B (32B, prompt of xyyan_text2sql_inference.py, SQLite dialect, table "data"), 300 s limit.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=300
export BASELINES_SUMMARY=results/v3_fix/baselines/summary_xiyanB_300s.csv
R=results/baselines_v3_gpu_package/xiyan_run_B/outputs_B
for ds in hospital restaurant sisu tax hmdaz hmda_checks; do
  .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py xiyan --dataset $ds --table data --dialect sqlite \
    --detailed $R/$ds/seed0/${ds}_xiyan_detailed_results.csv $R/$ds/seed1/${ds}_xiyan_detailed_results.csv $R/$ds/seed2/${ds}_xiyan_detailed_results.csv \
    > results/v3_fix/baselines/logs_300s/xiyanB_$ds.log 2>&1
  echo "xiyan B $ds exit $? $(date -Iseconds)"
done
