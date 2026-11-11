"""AV sensitivity to the dominant gold rules (round v3, E2; no LLM calls).

The aggregate violation F1 (AV) is computed over the union of violating tuples, so
a single gold rule can dominate it. For each dataset this script re-evaluates the
stored E2 outputs on the GT without the dominant rule(s): on HMDA_CHECKS the two
date-range edits (S270, Q022: uniform synthetic-looking dates, 97% / 80% of rows);
elsewhere the gold rule with the most violations when it flags >= 75% of the
gold-violating tuples (max_gold_rule_violation_share). Exact CE/CE_equiv are reported on the same reduced GT.

Usage: .venv/bin/python results/v3/analysis/av_sensitivity.py [dataset ...]
Writes results/v3/analysis/av_sensitivity_<datasets>.csv; with no argument (all
datasets) also av_sensitivity.md.
"""
import ast, glob, json, os, sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, ".")
from src.data import DatasetLoader
from src.evaluation.mapping import MappingEvaluator
from src.predicates import normalize_rules

OUT = "results/v3/analysis"
GT = {"hmdaz": "hmdaz_text2rule_gt.json", "hmda_checks": "hmda_regc_checks-text2rule-gt.json",
      "hospital": "hospital-holo_text2rule_gt.json", "restaurant": "restaurant-inspections_text2rule_gt.json",
      "sisu": "sisu_text2rule_gt.json", "tax": "tax_text2rule_gt.json"}
DATE_EDITS = ("Q022", "S270")
SHARE = 0.75

# Output globs (comma-separated); default: the stored round-v3 E2 outputs.
# version_fix: SENS_GLOBS="results/v3/e2_mapping/*singleprompt*/outputs/*.json,results/v3_fix/e2_mapping/*multistep*/outputs/*.json"
SENS_GLOBS = os.environ.get("SENS_GLOBS", "results/v3/e2_mapping/*/outputs/*.json").split(",")
OUT = os.environ.get("SENS_OUT", OUT)
os.makedirs(OUT, exist_ok=True)
runs = defaultdict(list)
for f in [x for g in SENS_GLOBS for x in glob.glob(g)]:
    o = json.load(open(f))
    runs[o["dataset"]].append((o["variant"], o["outputs"], f))

rows = []
SELECTED = sys.argv[1:] or list(GT)
for ds in SELECTED:
    dfs, _, gt = DatasetLoader(f"data/{ds}", sample_rows=1000000, gt_file=GT[ds], stream_csv=True, space_version=2).load()
    gold = {s: normalize_rules(e["rules"]) for e in gt for s in e["sentences"]}
    # per-sentence gold violation counts from any detailed metrics file of this dataset
    detailed = pd.read_csv(glob.glob(f"{runs[ds][0][2].rsplit('/outputs/', 1)[0]}/metrics/*_detailed.csv")[0])
    counts = detailed.dropna(subset=["num_gold_violations"]).groupby("gold_constraint").num_gold_violations.max()
    if ds == "hmda_checks":
        raw = json.load(open(f"data/{ds}/{GT[ds]}"))["mapping"]
        drop = {s for e in raw if any(e.get("_source", "").endswith(c) for c in DATE_EDITS) for s in e["sentences"]}
        reason = "date edits S270 + Q022"
    else:
        run_dir = runs[ds][0][2].rsplit('/outputs/', 1)[0]
        union = pd.read_csv(f"{run_dir}/results_summary.csv").total_gold_violations.iloc[0]
        top, share = counts.idxmax(), counts.max() / union  # same definition as max_gold_rule_violation_share
        drop = set(detailed.loc[detailed.gold_constraint == top, "sentence"]) if share >= SHARE else set()
        reason = f"dominant rule ({share:.0%} of the gold-violating tuples)" if drop else "none (no rule >= 75%)"
    reduced = {s: r for s, r in gold.items() if s not in drop}
    df = next(iter(dfs.values()))
    for label, g in (("full", gold), ("without_dominant", reduced)):
        if label == "without_dominant" and not drop:
            continue
        evaluator = MappingEvaluator(g, df=df)
        for variant, outs, _ in runs[ds]:
            preds = [(x["sentence"], normalize_rules(ast.literal_eval(x["pred_constraint"])) if x.get("pred_constraint") else None)
                     for x in outs if x["sentence"] in g]
            m, _ = evaluator.evaluate(preds)
            mapper, llm = variant.split(" | ", 1)
            rows.append({"dataset": ds, "gt": label, "excluded": reason if label != "full" else "",
                         "mapper": mapper, "llm": llm, "AV": m.aggregate_violation_f1, "AV_P": m.aggregate_violation_precision,
                         "AV_R": m.aggregate_violation_recall, "CE": m.constraint_exact_f1, "CE_equiv": m.constraint_equiv_f1,
                         "rule_F1": m.pairwise_avg_execution_f1, "exec_equiv": m.violation_identical_rate,
                         "max_share": m.max_gold_rule_violation_share})
        evaluator.close()
    print(ds, reason, flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/av_sensitivity_{'_'.join(SELECTED)}.csv", index=False)
piv = res.pivot_table(index=["mapper", "llm"], columns=["dataset", "gt"], values="AV").round(2)
md = ["# Sensibilidade da violação agregada (E2, rodada v3)", "", __doc__.strip(), "",
      "Excluídos: " + "; ".join(sorted({f"{d}: {r}" for d, r in zip(res.dataset, res.excluded) if r})), "",
      piv.to_markdown(), ""]
open(f"{OUT}/av_sensitivity_{'_'.join(SELECTED)}.md", "w").write("\n".join(md))
print(piv.to_string())
