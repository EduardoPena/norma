#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_NA = 'N/A'
VARIANT_NAME = 'XIYANText2SQL'
SUMMARY_PATH = Path("baseline_results") / "summary.csv"

from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator
from src.evaluation.classification import ClassificationEvaluator


# =============================================================================
# Metric helpers (mirrors MappingEvaluator internals)
# =============================================================================

def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def _prf(gold: Set, pred: Set) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


def _exec_sql(sql: str, db: DBExecutor) -> Tuple[Set[int], bool]:
    """Execute raw SQL via DBExecutor, return (violation_id_set, success)."""
    result = db.execute_violation_query(sql.strip())
    if not result["success"]:
        return set(), False
    ids: Set[int] = set()
    for row in result.get("rows", []):
        for val in row:
            if val is not None:
                try:
                    ids.add(int(val))
                except (TypeError, ValueError):
                    pass
    return ids, True


def _exec_rule(rule: List[str], translator: SQLTranslator) -> Set[int]:
    """Execute a gold constraint rule via SQLTranslator."""
    result = translator.execute_violation(rule)
    if not result["success"]:
        return set()
    ids: Set[int] = set()
    for row in result.get("rows", []):
        for val in row:
            if val is not None:
                try:
                    ids.add(int(val))
                except (TypeError, ValueError):
                    pass
    return ids


# =============================================================================
# Main evaluator class
# =============================================================================

class Text2SQLMapperEval:

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def load_dataset(self, dataset_name: str):
        from src.data import DatasetLoader
        data_dir = Path("data") / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")
        loader = DatasetLoader(str(data_dir))
        df, sentences, sentence2dcs_map, predicate_space = loader.load()
        return df, sentences, sentence2dcs_map, predicate_space

    def load_baseline(self, dataset_name: str) -> Dict[str, Optional[str]]:
        data_dir = Path("baseline_results/text2sql_results") / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")
        df = pd.read_csv(data_dir / f"{dataset_name}_xiyan_detailed_results.csv")
        return {
            row["sentence"]: row["pred_sql"]
            if pd.notna(row["pred_sql"]) and str(row["pred_sql"]).strip() != ""
            else None
            for _, row in df.iterrows()
        }

    def run(self):
        print("-" * 80)
        print(f"Loading {self.dataset_id}...")
        try:
            df, sentences, sentence2dcs_map, predicate_space = self.load_dataset(self.dataset_id)
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return

        gt_sentence2dcs_map = {s: dcs for s, dcs in sentence2dcs_map.items() if dcs}
        num_total = len(sentences)
        num_gt    = len(gt_sentence2dcs_map)

        print(f"Total sentences:          {num_total}")
        print(f"GT constraint sentences:  {num_gt}")
        print(f"Non-constraint sentences: {num_total - num_gt}\n")

        baseline_sentence2pred_sql = self.load_baseline(self.dataset_id)

        # ------------------------------------------------------------------
        # Set up DuckDB + translator (same as main pipeline)
        # ------------------------------------------------------------------
        # table_name="t1" ensures pred SQL that references "t1" resolves correctly.
        # SQLTranslator wraps this as "t1" t1 in gold queries, which also works.
        db         = DBExecutor(df, table_name="t1")
        translator = SQLTranslator(db)

        # ------------------------------------------------------------------
        # Pre-compute gold violations for all GT sentences
        # ------------------------------------------------------------------
        print("Executing gold constraints...")
        gold_violations: Dict[str, Set[int]] = {}
        for sentence, rule in gt_sentence2dcs_map.items():
            gold_violations[sentence] = _exec_rule(rule, translator)

        # ------------------------------------------------------------------
        # Execute pred SQL and collect per-sentence results
        # ------------------------------------------------------------------
        print("Executing pred SQL from baseline...\n")

        # For classification: pred_is_constraint = tool produced a non-null SQL
        clf_predictions: List[Tuple[str, bool]] = []

        # For pairwise (GT sentences only)
        pw_exact:     List[bool]  = []
        pw_jaccard:   List[float] = []
        pw_precision: List[float] = []
        pw_recall:    List[float] = []
        pw_f1:        List[float] = []

        # For aggregate violation (all sentences)
        all_gold_violations: Set[int] = set()
        all_pred_violations: Set[int] = set()

        detail_records = []

        for sentence in sentences:
            pred_sql      = baseline_sentence2pred_sql.get(sentence)
            gold_is_const = sentence in gt_sentence2dcs_map
            tool_pred_pos = pred_sql is not None  # classification signal from the tool

            # Classification
            clf_predictions.append((sentence, tool_pred_pos))

            # Execute pred SQL
            pred_viol: Set[int] = set()
            pred_exec_ok  = False
            pred_exec_err = None

            if pred_sql:
                pred_viol, pred_exec_ok = _exec_sql(pred_sql, db)
                if not pred_exec_ok:
                    pred_exec_err = "execution failed"
                    print(f"  [WARN] pred SQL failed: {sentence[:70]!r}")

            # Accumulate aggregate violations
            gold_viol = gold_violations.get(sentence, set())
            all_gold_violations |= gold_viol
            all_pred_violations |= pred_viol

            # Pairwise (GT only)
            exec_exact = exec_jac = exec_p = exec_r = exec_f = None
            num_gold_v = num_pred_v = num_overlap_v = None

            if gold_is_const:
                exec_exact = (gold_viol == pred_viol)
                exec_jac   = _jaccard(gold_viol, pred_viol)
                exec_p, exec_r, exec_f = _prf(gold_viol, pred_viol)
                num_gold_v    = len(gold_viol)
                num_pred_v    = len(pred_viol)
                num_overlap_v = len(gold_viol & pred_viol)

                pw_exact.append(exec_exact)
                pw_jaccard.append(exec_jac)
                pw_precision.append(exec_p)
                pw_recall.append(exec_r)
                pw_f1.append(exec_f)

            detail_records.append({
                "sentence":               sentence,
                "gold_is_constraint":     gold_is_const,
                "pred_is_constraint":     tool_pred_pos,
                "pred_sql":               pred_sql,
                "pred_exec_ok":           pred_exec_ok if pred_sql else None,
                "pred_exec_error":        pred_exec_err,
                "num_gold_violations":    num_gold_v,
                "num_pred_violations":    num_pred_v,
                "num_overlap_violations": num_overlap_v,
                "pairwise_violation_exact_match": exec_exact,
                "pairwise_violation_jaccard":     exec_jac,
                "pairwise_violation_precision":   exec_p,
                "pairwise_violation_recall":      exec_r,
                "pairwise_violation_f1":          exec_f,
            })

        db.close()

        # ------------------------------------------------------------------
        # Classification metrics
        # ------------------------------------------------------------------
        clf_evaluator = ClassificationEvaluator(ground_truth=gt_sentence2dcs_map)
        clf_metrics, _ = clf_evaluator.evaluate(clf_predictions)

        # ------------------------------------------------------------------
        # Aggregate violation metrics
        # ------------------------------------------------------------------
        total_gv = len(all_gold_violations)
        total_pv = len(all_pred_violations)
        total_ov = len(all_gold_violations & all_pred_violations)
        agg_vp = total_ov / total_pv if total_pv else 0.0
        agg_vr = total_ov / total_gv if total_gv else 0.0
        agg_vf = 2 * agg_vp * agg_vr / (agg_vp + agg_vr) if (agg_vp + agg_vr) else 0.0

        # ------------------------------------------------------------------
        # Pairwise aggregation
        # ------------------------------------------------------------------
        pw_total       = len(pw_exact)
        pw_exact_count = sum(pw_exact)
        pw_exact_acc   = pw_exact_count / pw_total if pw_total else 0.0
        avg_jac, std_jac = _mean_std(pw_jaccard)
        avg_p,   std_p   = _mean_std(pw_precision)
        avg_r,   std_r   = _mean_std(pw_recall)
        avg_f,   std_f   = _mean_std(pw_f1)

        # ------------------------------------------------------------------
        # Print results
        # ------------------------------------------------------------------
        w = 70
        print("\n" + "=" * w)
        print(f"TEXT2SQL EVALUATION: {self.dataset_id}")
        print("=" * w)

        print(f"\n-- CLASSIFICATION (all {num_total} sentences) " + "-" * 20)
        print(f"   Accuracy:  {clf_metrics.accuracy:.3f}")
        print(f"   Precision: {clf_metrics.precision:.3f}")
        print(f"   Recall:    {clf_metrics.recall:.3f}")
        print(f"   F1:        {clf_metrics.f1_score:.3f}  [PRIMARY]")
        print(f"   TP: {clf_metrics.true_positives:3d}  FP: {clf_metrics.false_positives:3d}  "
              f"FN: {clf_metrics.false_negatives:3d}  TN: {clf_metrics.true_negatives:3d}")

        print(f"\n-- AGGREGATE VIOLATION DISCOVERY (all {num_total} sentences) " + "-" * 5)
        print(f"   Precision: {agg_vp:.3f}")
        print(f"   Recall:    {agg_vr:.3f}")
        print(f"   F1:        {agg_vf:.3f}  [PRIMARY]")
        print(f"   Gold violations:      {total_gv}")
        print(f"   Predicted violations: {total_pv}")
        print(f"   Overlap violations:   {total_ov}")

        print(f"\n-- PAIRWISE VIOLATION DISCOVERY (GT sentences only, n={pw_total}) " + "-" * 5)
        print(f"   Exact Match: {pw_exact_acc:.3f} ({pw_exact_count}/{pw_total})")
        print(f"   Precision:   {avg_p:.3f} (+/-{std_p:.3f})")
        print(f"   Recall:      {avg_r:.3f} (+/-{std_r:.3f})")
        print(f"   F1:          {avg_f:.3f} (+/-{std_f:.3f})  [PRIMARY]")
        print(f"   Jaccard:     {avg_jac:.3f} (+/-{std_jac:.3f})")

        print("=" * w)

        # ------------------------------------------------------------------
        # Save detailed per-row CSV
        # ------------------------------------------------------------------
        out_dir  = Path("baseline_results/text2sql_results") / self.dataset_id
        out_path = out_dir / f"{self.dataset_id}_xiyan_eval_results.csv"
        pd.DataFrame(detail_records).to_csv(out_path, index=False)
        print(f"\nDetailed results saved to: {out_path}")

        # ------------------------------------------------------------------
        # Return summary row (compatible with run_mapper_eval_full.py columns)
        # ------------------------------------------------------------------
        return {
            'dataset':              self.dataset_id,
            'variant_name':         VARIANT_NAME,
            'implementation':       'text2sql',
            'mode':                 'SENTENCE_DIRECT',
            'num_sentences_total':  num_total,
            'num_sentences_gt':     num_gt,
            'status':               'success',
            'runtime_s':            _NA,
            # Classification
            'clf_accuracy':         clf_metrics.accuracy,
            'clf_precision':        clf_metrics.precision,
            'clf_recall':           clf_metrics.recall,
            'clf_f1':               clf_metrics.f1_score,
            'clf_tp':               clf_metrics.true_positives,
            'clf_fp':               clf_metrics.false_positives,
            'clf_fn':               clf_metrics.false_negatives,
            'clf_tn':               clf_metrics.true_negatives,
            # Aggregate violation
            'aggregate_violation_precision': agg_vp,
            'aggregate_violation_recall':    agg_vr,
            'aggregate_violation_f1':        agg_vf,
            'total_gold_violations':         total_gv,
            'total_pred_violations':         total_pv,
            'total_overlap_violations':      total_ov,
            # Constraint exact match — N/A (no predicate-level output from Text2SQL)
            'constraint_exact_precision': _NA,
            'constraint_exact_recall':    _NA,
            'constraint_exact_f1':        _NA,
            'constraint_exact_tp':        _NA,
            'constraint_exact_fp':        _NA,
            'constraint_exact_fn':        _NA,
            'num_gold_constraints':       num_gt,
            'num_pred_constraints':       _NA,
            # Pairwise syntactic — N/A
            'pairwise_syntactic_exact_match':           _NA,
            'pairwise_syntactic_exact_match_count':     _NA,
            'pairwise_syntactic_total':                 _NA,
            'pairwise_syntactic_predicate_jaccard':     _NA,
            'pairwise_syntactic_predicate_jaccard_std': _NA,
            # Pairwise violation
            'pairwise_violation_exact_match':       pw_exact_acc,
            'pairwise_violation_exact_match_count': pw_exact_count,
            'pairwise_violation_total':             pw_total,
            'pairwise_violation_precision':         avg_p,
            'pairwise_violation_precision_std':     std_p,
            'pairwise_violation_recall':            avg_r,
            'pairwise_violation_recall_std':        std_r,
            'pairwise_violation_f1':                avg_f,
            'pairwise_violation_f1_std':            std_f,
            'pairwise_violation_jaccard':           avg_jac,
            'pairwise_violation_jaccard_std':       std_jac,
        }


def append_to_summary(row: Dict):
    """Append a result row to the shared summary CSV, creating it if needed."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        existing = pd.read_csv(SUMMARY_PATH)
        # Remove any prior row for this dataset + variant to avoid duplicates
        mask = (existing['dataset'] == row['dataset']) & (existing['variant_name'] == row['variant_name'])
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary updated: {SUMMARY_PATH}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Text2SQL (XIYANText2SQL) baseline results")
    parser.add_argument(
        "--datasets", nargs="+", default=["sisu", "hospital", "restaurant", "tax"],
        help="One or more dataset names to evaluate (default: tax)"
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print('=' * 80)
        try:
            experiment = Text2SQLMapperEval(dataset)
            summary_row = experiment.run()
            if summary_row:
                append_to_summary(summary_row)
        except Exception as e:
            print(f"\nError on dataset {dataset}: {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    print("Starting Text2SQL mapper evaluation...\n")
    main()