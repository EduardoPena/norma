#!/usr/bin/env bash
# XiYan-SQL variant A (14B, 3 seeds) on the six datasets, 60 s query limit, final converter.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=60
export BASELINES_SUMMARY=results/v3_fix/baselines/summary_xiyan.csv
R=results/baselines_v3_gpu_package/xiyan_run/outputs
ev() {
  .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py xiyan --dataset $1 \
    --detailed $R/$1/seed0/$1_xiyan_detailed_results.csv $R/$1/seed1/$1_xiyan_detailed_results.csv \
               $R/$1/seed2/$1_xiyan_detailed_results.csv > results/v3_fix/baselines/dc_logs/eval_xiyanA_$1.log 2>&1
  echo "xiyan A $1 exit $? $(date -Iseconds)"
}
for ds in hospital tax restaurant sisu; do ev $ds; done
while pgrep -f "evaluate_baselines_v3.py profiling --dataset hmdaz" >/dev/null; do sleep 60; done
for ds in hmdaz hmda_checks; do ev $ds; done
