#!/usr/bin/env python3
"""Evaluate the baselines (DC discovery, XiYan-SQL) with the round-v3 protocol.

Same data loading and evaluator as Norma in round v3: the first 1M rows of the
HMDA files, the revised gold standards, the v2 predicate space, and the SQL
translator with the corrected pair semantics (t1.tid <> t2.tid). Used for the six
datasets, so that every baseline number of the paper comes from one protocol.

DC discovery (CSVs of DCDiscovery.jar with columns DC, Coverage, NSymbols; DCs
written as "not(t1.a==t2.a and t1.b<>t2.b)"): the DCs are ranked by the mixed
coverage/succinctness score of the paper and the top-k are evaluated at the
document level (as in experiments/baselines/extractResultsProfiling.py). One row
per (epsilon, k); the epsilon is read from the file name (exact | approx_<eps>):

  .venv/bin/python experiments/baselines/evaluate_baselines_v3.py profiling \
      --dataset hmdaz --csv baselines/1_profiling/results_dc_discovery/hmdaz_v3_*.csv \
      --top-k 5 10 20 50 100

XiYan-SQL (one CSV per seed with columns sentence, pred_sql; pred_sql empty or
NULL when the model judged the sentence not to be a checkable constraint):

  .venv/bin/python experiments/baselines/evaluate_baselines_v3.py xiyan \
      --dataset hmdaz --detailed out/seed0/hmdaz_xiyan_detailed_results.csv out/seed1/...

For XiYan-SQL, every query is executed as generated (any SQL; SQLite queries of
prompt variant B are first transpiled to DuckDB with sqlglot, --dialect sqlite).
Constraint extraction reads the WHERE clause of each query
(experiments/baselines/sql_rules.py) and is scored with MappingEvaluator.evaluate,
as Norma's mappers. A query with no DC reading (aggregation, NOT EXISTS, CASE,
functions, ...) is scored by execution equivalence: it is a correct extraction if
it returns exactly the (non-empty) violating tuples of the gold DC of its sentence,
and a wrong predicted constraint otherwise. Classification is reported twice: a non-null query counts as
a predicted constraint (as for Norma), and, as in the paper's evaluation script
(baselines/2_text2sql/eval_xiyan_duck.py), with failing queries excluded.

Results are appended to results/v3_fix/baselines/summary.csv.
"""
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlglot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import DatasetLoader  # noqa: E402
from src.evaluation.mapping import MappingEvaluator  # noqa: E402
from src.predicates import normalize_rules  # noqa: E402
from sql_rules import sql_to_rules  # noqa: E402

GT = {
    "hospital": "hospital-holo_text2rule_gt.json",
    "restaurant": "restaurant-inspections_text2rule_gt.json",
    "sisu": "sisu_text2rule_gt.json",
    "tax": "tax_text2rule_gt.json",
    "hmdaz": "hmdaz_text2rule_gt.json",
    "hmda_checks": "hmda_regc_checks-text2rule-gt.json",
}
OUT = Path(os.environ.get("BASELINES_SUMMARY", "results/v3_fix/baselines/summary.csv"))
METRICS = ("constraint_exact_precision", "constraint_exact_recall", "constraint_exact_f1",
           "aggregate_violation_precision", "aggregate_violation_recall", "aggregate_violation_f1",
           "total_gold_violations", "total_pred_violations")


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


def epsilon_of(path):
    m = re.search(r"_approx_([0-9.]+)_", Path(path).name)
    return float(m.group(1)) if m else 0.0


def save(row):
    """Append one row; the read-modify-write is locked, since several evaluations may run at once."""
    import fcntl

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{OUT}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        df = pd.DataFrame([row])
        if OUT.exists():
            df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
        df.to_csv(OUT, index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
           if k not in ("source", "evaluated_at")})


def profiling(args):
    df, _, gold = load(args.dataset)
    gold_rules = list(gold.values())
    for csv_path in sorted(args.csv):
        dcs = pd.read_csv(csv_path)
        eps = epsilon_of(csv_path)
        if dcs.empty:
            save({"dataset": args.dataset, "method": "DC Discovery", "epsilon": eps, "top_k": None,
                  "source": str(csv_path), "note": "no DCs discovered",
                  "evaluated_at": datetime.now().isoformat(timespec="seconds")})
            continue
        dcs["NSymbols_norm"] = dcs["NSymbols"].min() / dcs["NSymbols"]
        dcs["Mixed_score"] = 0.5 * dcs["Coverage"] + 0.5 * dcs["NSymbols_norm"]
        dcs = dcs.sort_values("Mixed_score", ascending=False)
        for k in args.top_k:
            if k > len(dcs) and k != args.top_k[0]:
                continue
            # one DC = one rule whose body is the conjunction of its predicates
            # (a plain list would be read as independent rules by normalize_rules)
            predicted = [[{"AND": r}] if len(r) > 1 else [r[0]]
                         for r in (parse_dc(dc) for dc in dcs.head(k)["DC"]) if r]
            evaluator = MappingEvaluator(ground_truth=gold, df=df)
            m = evaluator.evaluate_document_level(predicted_rules=predicted, gold_rules=gold_rules)
            evaluator.close()
            save({"dataset": args.dataset, "method": "DC Discovery", "epsilon": eps,
                  "top_k": min(k, len(dcs)), "n_discovered": len(dcs), "source": str(csv_path),
                  "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                  **{c: m.get(c) for c in METRICS}})


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def xiyan(args):
    from src.dbms.db_interface import DBExecutor

    df, sentences, gold = load(args.dataset)
    evaluator = MappingEvaluator(ground_truth=gold, df=df)
    gold_v = set()
    for rule in gold.values():
        gold_v |= evaluator._execute_rule(rule)
    # XiYan-SQL queries name the table t1, as in its prompt
    db = DBExecutor(df=df, table_name=args.table)
    gold_by_s = {s: evaluator._execute_rule(r) for s, r in gold.items()}
    for path in args.detailed:
        pred = pd.read_csv(path)
        sql = {r.sentence: (None if pd.isna(r.pred_sql) or str(r.pred_sql).strip().upper() in ("", "NULL")
                            else str(r.pred_sql)) for r in pred.itertuples()}
        missing = [s for s in sentences if s not in sql]
        if missing:
            print(f"Warning: {len(missing)} sentences without a prediction (treated as no constraint)")
        pred_v, failed, non_extractable, reasons = set(), set(), [], {}
        predictions, pred_by_s, transpiled = [], {}, 0
        for s in sentences:
            q = sql.get(s)
            rules = None
            if q:
                runnable = q.strip().rstrip(";")
                if args.dialect != "duckdb":
                    try:
                        runnable = sqlglot.transpile(runnable, read=args.dialect, write="duckdb")[0]
                        transpiled += 1
                    except Exception:  # noqa: BLE001 - run the original query and let it fail visibly
                        pass
                res = db.execute_violation_query(runnable)
                if res["success"]:
                    pred_by_s[s] = {int(v) for row in res["rows"] for v in row
                                    if v is not None and str(v).lstrip("-").isdigit()}
                    pred_v |= pred_by_s[s]
                else:
                    failed.add(s)
                rules, reason = sql_to_rules(runnable)
                if rules is None:
                    non_extractable.append(s)
                    reasons[reason.split(" (")[0].split(":")[0]] = reasons.get(reason.split(" (")[0].split(":")[0], 0) + 1
                else:
                    rules = normalize_rules(rules)
            predictions.append((s, rules))
        # classification: a non-null query is a predicted constraint (Norma's criterion) ...
        is_gold = {s: s in gold for s in sentences}
        tp = sum(1 for s in sentences if sql.get(s) and is_gold[s])
        fp = sum(1 for s in sentences if sql.get(s) and not is_gold[s])
        fn = sum(1 for s in sentences if not sql.get(s) and is_gold[s])
        cp, cr, cf = _prf(tp, fp, fn)
        # ... and as in eval_xiyan_duck.py, which drops sentences whose query fails
        ok = [s for s in sentences if s not in failed]
        tp2 = sum(1 for s in ok if sql.get(s) and is_gold[s])
        fp2 = sum(1 for s in ok if sql.get(s) and not is_gold[s])
        fn2 = sum(1 for s in ok if not sql.get(s) and is_gold[s])
        pp, pr, pf = _prf(tp2, fp2, fn2)
        # constraint extraction from the WHERE clauses, scored as Norma's mappers ...
        metrics, _ = evaluator.evaluate(predictions)
        tp, fp, fn = metrics.constraint_exact_tp, metrics.constraint_exact_fp, metrics.constraint_exact_fn
        etp = metrics.constraint_equiv_tp or 0
        efp = metrics.constraint_equiv_fp or 0
        efn = metrics.constraint_equiv_fn or 0
        matched = evaluator._canonical_rule_set([r for _, r in predictions if r])
        # ... and the queries with no DC reading by execution equivalence
        exec_equivalent = 0
        for s in non_extractable:
            same = s in gold and s not in failed and gold_by_s[s] and pred_by_s.get(s) == gold_by_s[s]
            if same and not (evaluator._canonical_rule_set([gold[s]]) <= matched):
                tp, fn, etp, efn, exec_equivalent = tp + 1, fn - 1, etp + 1, efn - 1, exec_equivalent + 1
            elif not same:
                fp, efp = fp + 1, efp + 1
        ce_p, ce_r, ce_f = _prf(tp, fp, fn)
        ceq_f = _prf(etp, efp, efn)[2]
        ov = gold_v & pred_v
        vp = len(ov) / len(pred_v) if pred_v else 0.0
        vr = len(ov) / len(gold_v) if gold_v else 0.0
        save({"dataset": args.dataset, "method": "XiYan-SQL", "run": Path(path).parent.name,
              "source": str(path), "evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "clf_precision": cp, "clf_recall": cr, "clf_f1": cf,
              "clf_precision_paper": pp, "clf_recall_paper": pr, "clf_f1_paper": pf,
              "constraint_exact_precision": ce_p, "constraint_exact_recall": ce_r, "constraint_exact_f1": ce_f,
              "constraint_equiv_f1": ceq_f,
              "constraint_exact_tp": tp, "constraint_exact_fp": fp, "constraint_exact_fn": fn,
              "constraint_exact_f1_dc_only": metrics.constraint_exact_f1,
              "aggregate_violation_precision": vp, "aggregate_violation_recall": vr,
              "aggregate_violation_f1": 2 * vp * vr / (vp + vr) if vp + vr else 0.0,
              "total_gold_violations": len(gold_v), "total_pred_violations": len(pred_v),
              "queries": sum(1 for s in sentences if sql.get(s)), "failed_queries": len(failed),
              "non_extractable": len(non_extractable), "exec_equivalent_extractions": exec_equivalent,
              "transpiled": transpiled, "dialect": args.dialect,
              "note": "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))})
    evaluator.close()
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("profiling")
    a.add_argument("--dataset", required=True, choices=sorted(GT))
    a.add_argument("--csv", required=True, type=Path, nargs="+")
    a.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    b = sub.add_parser("xiyan")
    b.add_argument("--dataset", required=True, choices=sorted(GT))
    b.add_argument("--detailed", required=True, type=Path, nargs="+")
    b.add_argument("--table", default="t1", help="table name used in the generated SQL (default: t1; variant B: data)")
    b.add_argument("--dialect", default="duckdb", choices=["duckdb", "sqlite"],
                   help="dialect of the generated SQL; sqlite queries are transpiled to DuckDB")
    args = parser.parse_args()
    profiling(args) if args.mode == "profiling" else xiyan(args)


if __name__ == "__main__":
    main()
