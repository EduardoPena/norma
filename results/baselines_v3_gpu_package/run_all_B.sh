#!/usr/bin/env bash
# Prompt variant B (xyyan_text2sql_inference.py) with XiYan-SQL 32B, six datasets, three seeds.
set -u
M=XGenerationLab/XiYanSQL-QwenCoder-32B-2504
mkdir -p logs_B outputs_B
run() { python run_xiyan_v3.py --prompt B --model $M --dataset "$1" --csv "$2" --gt "$3" ${4:+--rows $4} --seeds 0 1 2 --out outputs_B 2>&1 | tee "logs_B/$1.log"; }
run hospital    data/hospital/hospital.csv                data/hospital/hospital-holo_text2rule_gt.json
run tax         data/tax/tax.csv                          data/tax/tax_text2rule_gt.json
run restaurant  data/restaurant/restaurant.csv            data/restaurant/restaurant-inspections_text2rule_gt.json
run sisu        data/sisu/sisu.csv                        data/sisu/sisu_text2rule_gt.json
run hmdaz       data/hmdaz/hmdaz_first1M.csv.gz           data/hmdaz/hmdaz_text2rule_gt.json             1000000
run hmda_checks data/hmda_checks/hmda_checks_first1M.csv.gz data/hmda_checks/hmda_regc_checks-text2rule-gt.json 1000000
tar -czf outputs_B.tar.gz outputs_B logs_B
echo "done: outputs_B.tar.gz"
