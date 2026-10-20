"""Re-evaluate the stored round-v3 outputs with the corrected pair semantics.

The SQL translator used ``t1.tid < t2.tid`` for two-tuple DCs, which drops
violations of asymmetric DCs whose witnesses appear in the opposite tid order
(fixed in src/dbms/sql_translator.py: ``t1.tid <> t2.tid``). This script
recomputes, without any LLM call, every metric that depends on executing DCs
(aggregate violation, per-rule violation metrics, execution equivalence,
concentration, SQL errors) and keeps the others (classification, exact and
equivalence-aware extraction) as computed by the runners.

Inputs (read only): results/v3/{e1_classifiers,e2_mapping,e3_end2end}/<run>/.
Outputs: results/v3_reeval/<same layout>/<run>/results_summary.csv (E1 copied
unchanged) and results/v3_reeval/CHANGES.csv (old vs new values).
Usage: .venv/bin/python results/v3_reeval/reevaluate.py [dataset ...]
"""
import ast
import glob
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.data import DatasetLoader  # noqa: E402
from src.evaluation.mapping import MappingEvaluator  # noqa: E402
from src.predicates import normalize_rules  # noqa: E402

SRC, DST = Path("results/v3"), Path("results/v3_reeval")
GT = {"hmdaz": "hmdaz_text2rule_gt.json", "hmda_checks": "hmda_regc_checks-text2rule-gt.json",
      "hospital": "hospital-holo_text2rule_gt.json", "restaurant": "restaurant-inspections_text2rule_gt.json",
      "sisu": "sisu_text2rule_gt.json", "tax": "tax_text2rule_gt.json"}
E2_COLS = {  # summary column -> MappingMetrics attribute
    "agg_viol_precision": "aggregate_violation_precision", "agg_viol_recall": "aggregate_violation_recall",
    "agg_viol_f1": "aggregate_violation_f1", "total_gold_violations": "total_gold_violations",
    "total_pred_violations": "total_pred_violations", "total_overlap_violations": "total_overlap_violations",
    "pairwise_viol_exact_match": "pairwise_execution_accuracy", "pairwise_viol_exact_count": "pairwise_execution_correct",
    "pairwise_viol_total": "pairwise_execution_total", "pairwise_viol_prec_mean": "pairwise_avg_execution_precision",
    "pairwise_viol_prec_std": "pairwise_std_execution_precision", "pairwise_viol_rec_mean": "pairwise_avg_execution_recall",
    "pairwise_viol_rec_std": "pairwise_std_execution_recall", "pairwise_viol_f1_mean": "pairwise_avg_execution_f1",
    "pairwise_viol_f1_std": "pairwise_std_execution_f1", "pairwise_viol_jac_mean": "pairwise_avg_execution_jaccard",
    "pairwise_viol_jac_std": "pairwise_std_execution_jaccard",
}
E3_COLS = {
    "pairwise_violation_exact_match": "pairwise_execution_accuracy",
    "pairwise_violation_exact_match_count": "pairwise_execution_correct",
    "pairwise_violation_total": "pairwise_execution_total",
    "pairwise_violation_precision": "pairwise_avg_execution_precision",
    "pairwise_violation_precision_std": "pairwise_std_execution_precision",
    "pairwise_violation_recall": "pairwise_avg_execution_recall",
    "pairwise_violation_recall_std": "pairwise_std_execution_recall",
    "pairwise_violation_f1": "pairwise_avg_execution_f1", "pairwise_violation_f1_std": "pairwise_std_execution_f1",
    "pairwise_violation_jaccard": "pairwise_avg_execution_jaccard",
    "pairwise_violation_jaccard_std": "pairwise_std_execution_jaccard",
}
COMMON = ["num_pred_execution_errors", "num_gold_execution_errors", "gold_rules_without_violations",
          "max_gold_rule_violation_share", "violation_identical_rate", "violation_identical_rules",
          "violation_identical_total"]


def rules_of(text):
    if text is None or (isinstance(text, float) and pd.isna(text)) or text == "":
        return None
    return normalize_rules(ast.literal_eval(text))


def run_dirs(exp, datasets):
    out = defaultdict(list)
    for summary in sorted(glob.glob(str(SRC / exp / "*" / "results_summary.csv"))):
        run = Path(summary).parent
        ds = pd.read_csv(summary).dataset.iloc[0]
        if ds in datasets:
            out[ds].append(run)
    return out


def main():
    datasets = sys.argv[1:] or list(GT)
    if (SRC / "e1_classifiers").exists():
        shutil.copytree(SRC / "e1_classifiers", DST / "e1_classifiers", dirs_exist_ok=True)
    e2, e3 = run_dirs("e2_mapping", datasets), run_dirs("e3_end2end", datasets)
    changes = []
    for ds in datasets:
        if not e2[ds] and not e3[ds]:
            continue
        dfs, _, gt = DatasetLoader(f"data/{ds}", sample_rows=1000000, gt_file=GT[ds], stream_csv=True,
                                   space_version=2).load()
        df = next(iter(dfs.values()))
        gold = {s: normalize_rules(e["rules"]) for e in gt for s in e["sentences"] if e.get("rules")}
        evaluator = MappingEvaluator(gold, df=df)
        for exp, runs in (("e2_mapping", e2[ds]), ("e3_end2end", e3[ds])):
            for run in runs:
                summary = pd.read_csv(run / "results_summary.csv")
                detailed = pd.read_csv(glob.glob(str(run / "metrics" / "*.csv"))[0])
                preds = [(s, rules_of(p)) for s, p in zip(detailed.sentence, detailed.pred_constraint) if s in gold]
                metrics, _ = evaluator.evaluate(preds)
                new = {col: getattr(metrics, attr) for col, attr in (E2_COLS if exp == "e2_mapping" else E3_COLS).items()}
                new.update({col: getattr(metrics, col) for col in COMMON})
                if exp == "e3_end2end":  # aggregate over the predictions of ALL sentences, as the runner does
                    outputs = json.load(open(glob.glob(str(run / "outputs" / "*.json"))[0]))["outputs"]
                    pred_v, gold_v = set(), set()
                    for o in outputs:
                        if o.get("predicted"):
                            pred_v |= evaluator._execute_rule(ast.literal_eval(o["predicted"]))
                    for rule in gold.values():
                        gold_v |= evaluator._execute_rule(rule)
                    ov = gold_v & pred_v
                    p = len(ov) / len(pred_v) if pred_v else 0.0
                    r = len(ov) / len(gold_v) if gold_v else 0.0
                    new.update({"aggregate_violation_precision": p, "aggregate_violation_recall": r,
                                "aggregate_violation_f1": 2 * p * r / (p + r) if p + r else 0.0,
                                "total_gold_violations": len(gold_v), "total_pred_violations": len(pred_v),
                                "total_overlap_violations": len(ov)})
                for col, value in new.items():
                    old = summary[col].iloc[0] if col in summary else None
                    if value is None:
                        continue
                    changed = old is None or (str(old) != str(value) and not (
                        pd.notna(pd.to_numeric(old, errors="coerce")) and abs(float(old) - float(value)) < 1e-4))
                    if changed:
                        changes.append({"experiment": exp, "dataset": ds, "run": run.name,
                                        "variant": summary.variant_name.iloc[0], "column": col,
                                        "old": old, "new": value})
                    summary.loc[0, col] = value
                out = DST / exp / run.name
                out.mkdir(parents=True, exist_ok=True)
                summary.to_csv(out / "results_summary.csv", index=False)
                print(f"{exp} {ds} {summary.variant_name.iloc[0]}: done", flush=True)
        evaluator.close()
    report = pd.DataFrame(changes)
    path = DST / f"CHANGES_{'_'.join(datasets)}.csv"
    report.to_csv(path, index=False)
    print(f"{len(report)} changed values -> {path}")


if __name__ == "__main__":
    main()
