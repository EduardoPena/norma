#!/usr/bin/env bash
# Last attempt at DC discovery on HMDA Checks with a 12 GB heap, after the HMDAZ evaluation frees the machine.
cd /home/junio/source/norma
while pgrep -f "profiling --dataset hmdaz" >/dev/null; do sleep 120; done
echo "start $(date -Iseconds)"
experiments/baselines/run_dc_discovery.sh "hmda_checks_20k hmda_checks_10k" 2h 12g
echo "end $(date -Iseconds)"
