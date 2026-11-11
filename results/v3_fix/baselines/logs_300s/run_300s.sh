#!/usr/bin/env bash
# Baseline re-evaluation with the 300 s query limit used for Norma.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=300
L=results/v3_fix/baselines/logs_300s
R=results/baselines_v3_gpu_package/xiyan_run/outputs
D=baselines/1_profiling/results_dc_discovery
xiyan() {
  BASELINES_SUMMARY=results/v3_fix/baselines/summary_xiyan_300s.csv .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py xiyan --dataset $1 \
    --detailed $R/$1/seed0/$1_xiyan_detailed_results.csv $R/$1/seed1/$1_xiyan_detailed_results.csv $R/$1/seed2/$1_xiyan_detailed_results.csv > $L/xiyanA_$1.log 2>&1
  echo "xiyan A $1 exit $? $(date -Iseconds)"
}
dc() {
  BASELINES_SUMMARY=results/v3_fix/baselines/summary_300s.csv .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py profiling --dataset $1 \
    --csv $D/$1_v3_*.csv > $L/dc_$1.log 2>&1
  echo "dc $1 exit $? $(date -Iseconds)"
}
for ds in hospital restaurant sisu tax hmdaz hmda_checks; do xiyan $ds; done
for ds in hospital restaurant sisu tax hmdaz; do dc $ds; done
