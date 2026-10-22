#!/usr/bin/env python3
"""Evaluate the baselines (DC discovery, XiYan-SQL) with the round-v3 protocol.

Same data loading and evaluator as Norma in round v3: the first 1M rows of the
HMDA files, the revised gold standards, the v2 predicate space, and the SQL
translator with the corrected pair semantics (t1.tid <> t2.tid). Use it for the
HMDA datasets and to re-evaluate the stored baseline outputs of the paper
datasets.

DC discovery (output CSV of fdcd/DCFinder with columns DC, Coverage, NSymbols;
DCs written as "not(t1.a==t2.a and t1.b<>t2.b)"): the DCs are ranked by the
mixed coverage/succinctness score of the paper and the top-k are evaluated at
the document level (as in experiments/baselines/extractResultsProfiling.py):

  .venv/bin/python experiments/baselines/evaluate_baselines_v3.py profiling \
      --dataset hmdaz --csv path/hmdaz_approx_0.01.csv --label approx_0.01 --top-k 5 10 20

XiYan-SQL (CSV with columns sentence, pred_sql; pred_sql empty or NULL when the
model judged the sentence not to be a checkable constraint):

  .venv/bin/python experiments/baselines/evaluate_baselines_v3.py xiyan \
      --dataset hmdaz --detailed path/hmdaz_xiyan_detailed_results.csv

Results are appended to results/v3_fix/baselines/summary.csv.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import DatasetLoader  # noqa: E402
from src.evaluation.mapping import MappingEvaluator  # noqa: E402
from src.predicates import normalize_rules  # noqa: E402

GT = {
    "hospital": "hospital-holo_text2rule_gt.json",
    "restaurant": "restaurant-inspections_text2rule_gt.json",
    "sisu": "sisu_text2rule_gt.json",
    "tax": "tax_text2rule_gt.json",
    "hmdaz": "hmdaz_text2rule_gt.json",
    "hmda_checks": "hmda_regc_checks-text2rule-gt.json",
}
OUT = Path("results/v3_fix/baselines/summary.csv")


def load(dataset):
    dfs, _, gt = DatasetLoader(
        f"data/{dataset}", sample_rows=1_000_000, gt_file=GT[dataset], stream_csv=True, space_version=2
    ).load()
    df = next(iter(dfs.values()))
    sentences = [s for e in gt for s in e["sentences"]]
    gold = {s: normalize_rules(e["rules"]) for e in gt for s in e["sentences"] if e.get("rules")}
    return df, sentences, gold


def parse_dc(dc):
    """'not(t1.x==t2.x and t1.y>t2.y)' -> ['t1.x = t2.x', 't1.y > t2.y'] (paper notebook)."""
    import re
    dc = dc.strip()
    if dc.startswith("not("):
        dc = dc[4:]
    if dc.endswith(")"):
        dc = dc[:-1]
    out = []
    for p in dc.split(" and "):
        p = p.strip().replace("==", "=").replace("<>", "!=")
        for op in ["<=", ">=", "!=", "=", "<", ">"]:
            if op in p:
                p = re.sub(f"([^\\s]){re.escape(op)}([^\\s])", f"\\1 {op} \\2", p)
                break
        out.append(p)
    return out


def save(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


def profiling(args):
    df, _, gold = load(args.dataset)
    dcs = pd.read_csv(args.csv)
    dcs["NSymbols_norm"] = dcs["NSymbols"].min() / dcs["NSymbols"]
    dcs["Mixed_score"] = 0.5 * dcs["Coverage"] + 0.5 * dcs["NSymbols_norm"]
    dcs = dcs.sort_values("Mixed_score", ascending=False)
    gold_rules = list(gold.values())
    for k in args.top_k:
        # one DC = one rule whose body is the conjunction of its predicates
        # (a plain list would be read as independent rules by normalize_rules)
        predicted = [[{"AND": r}] if len(r) > 1 else [r[0]]
                     for r in (parse_dc(dc) for dc in dcs.head(k)["DC"]) if r]
        evaluator = MappingEvaluator(ground_truth=gold, df=df)
        m = evaluator.evaluate_document_level(predicted_rules=predicted, gold_rules=gold_rules)
        evaluator.close()
        save({"dataset": args.dataset, "method": "DC Discovery", "label": args.label, "top_k": k,
              "source": str(args.csv), "evaluated_at": datetime.now().isoformat(timespec="seconds"),
              **{c: m.get(c) for c in ("constraint_exact_precision", "constraint_exact_recall",
                                       "constraint_exact_f1", "aggregate_violation_precision",
                                       "aggregate_violation_recall", "aggregate_violation_f1",
                                       "total_gold_violations", "total_pred_violations")}})


def xiyan(args):
    from src.evaluation.classification import ClassificationEvaluator

    df, sentences, gold = load(args.dataset)
    pred = pd.read_csv(args.detailed)
    sql = {r.sentence: (None if pd.isna(r.pred_sql) or str(r.pred_sql).strip().upper() in ("", "NULL")
                        else str(r.pred_sql)) for r in pred.itertuples()}
    missing = [s for s in sentences if s not in sql]
    if missing:
        print(f"Warning: {len(missing)} sentences without a prediction (treated as no constraint)")
    # classification: a sentence is predicted as a constraint when a query is returned
    clf = ClassificationEvaluator({s: (gold.get(s) or []) for s in sentences})
    cm, _ = clf.evaluate([(s, sql.get(s) is not None) for s in sentences])
    # violations: union of the tuples returned by all queries vs. union of the gold violations
    evaluator = MappingEvaluator(ground_truth=gold, df=df)
    gold_v = set()
    for rule in gold.values():
        gold_v |= evaluator._execute_rule(rule)
    from src.dbms.db_interface import DBExecutor
    # XiYan-SQL queries name the table as in its prompt ("t1" in the paper runs)
    db = DBExecutor(df=df, table_name=args.table)
    pred_v, failed = set(), 0
    for s in sentences:
        q = sql.get(s)
        if not q:
            continue
        res = db.execute_violation_query(q.strip().rstrip(";"))
        if not res["success"]:
            failed += 1
            continue
        pred_v |= {int(v) for row in res["rows"] for v in row if v is not None and str(v).lstrip("-").isdigit()}
    evaluator.close()
    db.close()
    ov = gold_v & pred_v
    p = len(ov) / len(pred_v) if pred_v else 0.0
    r = len(ov) / len(gold_v) if gold_v else 0.0
    save({"dataset": args.dataset, "method": "XiYan-SQL", "label": "xiyan", "top_k": None,
          "source": str(args.detailed), "evaluated_at": datetime.now().isoformat(timespec="seconds"),
          "clf_precision": cm.precision, "clf_recall": cm.recall, "clf_f1": cm.f1_score,
          "aggregate_violation_precision": p, "aggregate_violation_recall": r,
          "aggregate_violation_f1": 2 * p * r / (p + r) if p + r else 0.0,
          "total_gold_violations": len(gold_v), "total_pred_violations": len(pred_v),
          "failed_queries": failed,
          "note": "constraint extraction (WHERE-clause predicates) is not computed here"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("profiling")
    a.add_argument("--dataset", required=True, choices=sorted(GT))
    a.add_argument("--csv", required=True, type=Path)
    a.add_argument("--label", default="approx")
    a.add_argument("--top-k", type=int, nargs="+", default=[5])
    b = sub.add_parser("xiyan")
    b.add_argument("--dataset", required=True, choices=sorted(GT))
    b.add_argument("--detailed", required=True, type=Path)
    b.add_argument("--table", default="t1", help="table name used in the generated SQL (default: t1)")
    args = parser.parse_args()
    profiling(args) if args.mode == "profiling" else xiyan(args)


if __name__ == "__main__":
    main()
