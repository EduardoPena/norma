#!/usr/bin/env bash
# XiYan-SQL (14B, bf16) on the six datasets, three seeds each. Logs in logs/.
set -u
mkdir -p logs outputs
run() { python run_xiyan_v3.py --dataset "$1" --csv "$2" --gt "$3" ${4:+--rows $4} --seeds 0 1 2 --out outputs 2>&1 | tee "logs/$1.log"; }
run hospital    data/hospital/hospital.csv                data/hospital/hospital-holo_text2rule_gt.json
run tax         data/tax/tax.csv                          data/tax/tax_text2rule_gt.json
run restaurant  data/restaurant/restaurant.csv            data/restaurant/restaurant-inspections_text2rule_gt.json
run sisu        data/sisu/sisu.csv                        data/sisu/sisu_text2rule_gt.json
run hmdaz       data/hmdaz/hmdaz_first1M.csv.gz           data/hmdaz/hmdaz_text2rule_gt.json             1000000
run hmda_checks data/hmda_checks/hmda_checks_first1M.csv.gz data/hmda_checks/hmda_regc_checks-text2rule-gt.json 1000000
tar -czf outputs.tar.gz outputs logs
echo "done: outputs.tar.gz"
