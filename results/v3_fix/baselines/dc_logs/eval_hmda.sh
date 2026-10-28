#!/usr/bin/env bash
# HMDA evaluations (60 s query limit): XiYan-SQL variant A (3 seeds), then DC Discovery on HMDAZ.
cd /home/junio/source/norma
export NORMA_SQL_TIMEOUT_S=60
R=results/baselines_v3_gpu_package/xiyan_run/outputs
for ds in hmdaz hmda_checks; do
  BASELINES_SUMMARY=results/v3_fix/baselines/summary_xiyan.csv .venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py xiyan --dataset $ds \
    --detailed $R/$ds/seed0/${ds}_xiyan_detailed_results.csv $R/$ds/seed1/${ds}_xiyan_detailed_results.csv \
               $R/$ds/seed2/${ds}_xiyan_detailed_results.csv > results/v3_fix/baselines/dc_logs/eval_xiyan_$ds.log 2>&1
  echo "xiyan $ds done $(date -Iseconds)"
done
.venv/bin/python -u experiments/baselines/evaluate_baselines_v3.py profiling --dataset hmdaz \
  --csv baselines/1_profiling/results_dc_discovery/hmdaz_v3_*.csv > results/v3_fix/baselines/dc_logs/eval_hmdaz.log 2>&1
echo "dc hmdaz done $(date -Iseconds)"
