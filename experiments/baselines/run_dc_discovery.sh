#!/usr/bin/env bash
# Run DCDiscovery.jar (FDCD, baselines/1_profiling) over the round-v3 inputs
# produced by prepare_dc_discovery.py: exact discovery and the epsilon grid of
# the paper. The jar writes ./results_dc_discovery/<input>_{exact|approx_<eps>}_<ts>.csv,
# so it runs from baselines/1_profiling. Every run has a wall-clock limit; a
# timeout or crash is logged in runs.tsv, never hidden.
#
# Usage: experiments/baselines/run_dc_discovery.sh "hospital tax" [timeout] [heap]
set -u
DATASETS=${1:-"hospital restaurant sisu tax hmdaz hmda_checks"}
TIMEOUT=${2:-6h}
HEAP=${3:-8g}
JAVA=${JAVA:-$HOME/.local/jdk-21/bin/java}  # the jar needs Java >= 20 (class file 64)
EPSILONS="exact 0.001 0.005 0.01 0.05 0.1"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK="$ROOT/baselines/1_profiling"
LOGS="$ROOT/results/v3_fix/baselines/dc_logs"
mkdir -p "$LOGS"
[ -f "$LOGS/runs.tsv" ] || printf "dataset\tepsilon\tstatus\texit\tseconds\tstarted\n" > "$LOGS/runs.tsv"

cd "$WORK" || exit 1
for ds in $DATASETS; do
  input="trimmed_datasets_v3/${ds}_v3.csv"
  [ -f "$input" ] || { echo "missing $input"; continue; }
  for eps in $EPSILONS; do
    arg=""; [ "$eps" != "exact" ] && arg="$eps"
    log="$LOGS/${ds}_${eps}.log"
    started=$(date -Iseconds); t0=$(date +%s)
    echo "[$started] $ds eps=$eps"
    timeout "$TIMEOUT" "$JAVA" -Duser.language=en -Duser.country=US -Xmx"$HEAP" -jar dc_discovery/DCDiscovery.jar "$input" $arg > "$log" 2>&1
    code=$?
    secs=$(( $(date +%s) - t0 ))
    if [ $code -eq 0 ]; then status=ok
    elif [ $code -eq 124 ]; then status=timeout
    elif grep -q "OutOfMemoryError" "$log"; then status=oom
    else status=error; fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ds" "$eps" "$status" "$code" "$secs" "$started" >> "$LOGS/runs.tsv"
    echo "  -> $status (${secs}s)"
  done
done
