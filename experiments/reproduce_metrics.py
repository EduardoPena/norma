"""Recompute the sentence-level mapping metrics (experiment E2) from the stored LLM
outputs and compare them with the stored run summaries. No LLM call and no API key.

For every run directory <root>/e2_mapping/<run>/ it reads
  - config.yaml                 dataset options of the run (GT file, rows, predicate space)
  - metrics/*_detailed.csv      the predicted DC of each ground-truth sentence
  - results_summary.csv         the metrics reported by the run
and recomputes, with the same evaluator (src/evaluation/mapping.py):
  CM   constraint_exact_f1      exact constraint mapping F1
  CM≡  constraint_equiv_f1      equivalence-aware constraint mapping F1
  AV   agg_viol_f1              aggregate violation F1 (all GT sentences pooled)
  Rule pairwise_viol_f1_mean    mean per-constraint violation F1

Usage:
  python experiments/reproduce_metrics.py [--root results/v3_fix] [--datasets hospital tax ...]
                                          [--out reproduce_e2.csv] [--tol 0.001]
The data files must be unpacked first (python tools/unpack_data.py).
Exit status 1 if any metric differs from the stored value by more than --tol.
"""
import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.shared.experiment_base import _normalize_sentence_rules  # noqa: E402
from src.data import DatasetLoader  # noqa: E402
from src.evaluation.mapping import MappingEvaluator  # noqa: E402
from src.predicates import normalize_rules  # noqa: E402

METRICS = {  # summary column -> how to get it from the recomputation
    "constraint_exact_f1": lambda m, av: m.constraint_exact_f1,
    "constraint_equiv_f1": lambda m, av: m.constraint_equiv_f1,
    "agg_viol_f1": lambda m, av: av,
    "pairwise_viol_f1_mean": lambda m, av: m.pairwise_avg_execution_f1,
}


def parse_rules(text):
    if text is None or (isinstance(text, float) and pd.isna(text)) or text == "":
        return None
    return normalize_rules(ast.literal_eval(text))


def load_dataset(name, options):
    loader = DatasetLoader(
        f"data/{name}",
        sample_rows=options.get("sample_rows"),
        gt_file=options.get("gt_file"),
        stream_csv=options.get("stream_csv", False),
        space_version=2 if str(options.get("predicate_space", "v1")).lstrip("v") == "2" else 1,
    )
    dataframes, _, gt = loader.load()
    drop = options.get("drop_multi_rule_sentences", False)
    gold = {}
    for entry in gt:
        rules = _normalize_sentence_rules(entry.get("rules"), drop_multi_rule_sentences=drop)
        if rules:
            for sentence in entry["sentences"]:
                gold[sentence] = rules
    return next(iter(dataframes.values())), gold


def aggregate_violation_f1(evaluator, predictions, gold):
    predicted, golden = set(), set()
    for _, rules in predictions:
        if rules:
            predicted.update(evaluator._execute_rule(rules))
    for rules in gold.values():
        golden.update(evaluator._execute_rule(rules))
    overlap = len(predicted & golden)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(golden) if golden else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/v3_fix")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out", default=None, help="CSV with one row per run (default: <root>/reproduce_e2.csv)")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--outputs-from", nargs="*", default=["results/v3"],
                        help="roots searched for config.yaml and metrics/ when a run directory holds only "
                             "a re-evaluated results_summary.csv (Single-Prompt runs of results/v3_fix)")
    args = parser.parse_args()

    runs = {}
    for summary in sorted(Path(args.root).glob("e2_mapping/*/results_summary.csv")):
        run = summary.parent
        source = next((d for d in [run] + [Path(r) / "e2_mapping" / run.name for r in args.outputs_from]
                       if (d / "config.yaml").is_file()), None)
        if source is None:
            print(f"skipped {run}: no config.yaml/metrics found", file=sys.stderr)
            continue
        config = yaml.safe_load((source / "config.yaml").read_text())
        dataset = config["datasets"][0]
        if args.datasets and dataset not in args.datasets:
            continue
        runs.setdefault(dataset, []).append((run, source, config["dataset_options"][dataset]))

    rows = []
    for dataset, items in runs.items():
        print(f"== {dataset}: {len(items)} runs", flush=True)
        cache = {}
        for run, source, options in items:
            key = yaml.safe_dump(options, sort_keys=True)
            if key not in cache:
                cache[key] = load_dataset(dataset, options)
            df, gold = cache[key]
            detailed = next((source / "metrics").glob("*_detailed.csv"))
            stored = pd.read_csv(run / "results_summary.csv").iloc[0]
            det = pd.read_csv(detailed)
            predictions = [(s, parse_rules(p)) for s, p in zip(det.sentence, det.pred_constraint) if s in gold]
            evaluator = MappingEvaluator(gold, df=df)
            metrics, _ = evaluator.evaluate(predictions)
            av = aggregate_violation_f1(evaluator, predictions, gold)
            evaluator.close()
            row = {"dataset": dataset, "variant": stored.variant_name, "run": run.name}
            for column, get in METRICS.items():
                row[f"{column}_stored"] = float(stored[column])
                row[f"{column}_recomputed"] = float(get(metrics, av))
            row["max_abs_diff"] = max(abs(row[f"{c}_stored"] - row[f"{c}_recomputed"]) for c in METRICS)
            rows.append(row)
            print(f"  {stored.variant_name:<45} CM {row['constraint_exact_f1_recomputed']:.3f}  "
                  f"CM≡ {row['constraint_equiv_f1_recomputed']:.3f}  AV {av:.3f}  "
                  f"Rule {row['pairwise_viol_f1_mean_recomputed']:.3f}  max|Δ| {row['max_abs_diff']:.5f}", flush=True)

    if not rows:
        print("no runs found", file=sys.stderr)
        return 1
    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(args.root) / "reproduce_e2.csv"
    table.to_csv(out, index=False)
    bad = table[table.max_abs_diff > args.tol]
    print(f"\n{len(table)} runs, max |Δ| = {table.max_abs_diff.max():.6f}; "
          f"{len(bad)} above tolerance {args.tol}. Written to {out}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
