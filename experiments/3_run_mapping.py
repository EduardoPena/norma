#!/usr/bin/env python3
"""
GT-Only Mapper Evaluation Script

Evaluates mapping quality in isolation: feeds ONLY ground-truth constraint
sentences (rules != null) directly to the mapper — no classifier step.

This gives a clean upper-bound view of mapper performance, free from any
classification noise (no false positives or false negatives from a classifier).

Metrics reported (all computed over GT sentences only):
  Syntactic:
    - Predicate exact match (strict)
    - Predicate Jaccard (mean ± std)
  Pairwise violation discovery:
    - Violation exact match
    - Violation precision / recall / F1 / Jaccard (mean ± std)
  Aggregate violation discovery (pooled over all GT sentences):
    - Aggregate violation precision / recall / F1
    - Gold / predicted / overlap violation counts

Supported mapper implementations:
  - llm                   : Standard LLMMapper
  - multi_step_refinement : MultiStepRefinementMapper

Mode is always SENTENCE_DIRECT (GT sentences fed directly, no filtering step).

Usage:
    python run_mapper_eval_gt_only.py configs/mappers/gt_only.yaml
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment, mapping_diagnostics
from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.evaluation.mapping import MappingEvaluator
from src.llm.helper import PromptLoader
from src.predicates import Rule, predicate_strings, transform_predicates

_NA = "N/A"


class MapperGTOnlyExperiment(BaseExperiment):
    """
    Pure mapping quality evaluation over GT constraint sentences only.

    No classifier is involved. Every GT sentence (those with non-null rules)
    is fed directly to the mapper. Metrics cover syntactic quality, pairwise
    violation discovery, and aggregate violation discovery — all scoped to
    GT sentences.
    """

    def __init__(self, config_file: str):
        super().__init__(config_file)
        if self.config["experiment_type"] != "mapper_gt_only":
            raise ValueError(
                f"Expected experiment_type='mapper_gt_only', "
                f"got '{self.config['experiment_type']}'"
            )
        self.prompt_loader = PromptLoader()

    # =========================================================================
    # Top-level run
    # =========================================================================

    def run(self):
        print("\n" + "=" * 80)
        print(f"GT-ONLY MAPPER EVALUATION: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp:         {self.timestamp}")
        print("\nScope: GT constraint sentences only (rules != null)")
        print("No classifier — pure mapping quality measurement.")
        print("\nMetrics computed over GT sentences:")
        print("  Syntactic:   exact match, predicate Jaccard")
        print("  Pairwise:    violation exact match, P/R/F1/Jaccard")
        print("  Aggregate:   pooled violation P/R/F1")
        print("=" * 80 + "\n")

        self.save_config_yaml()
        self.save_execution_config()

        total = len(self.config["datasets"]) * len(self.config["variants"])
        completed = 0
        failed = 0

        for dataset_name in self.config["datasets"]:
            try:
                n_failed = self.run_dataset(dataset_name)
                failed += n_failed
                completed += len(self.config["variants"]) - n_failed
            except Exception as e:
                print(f"\nError processing dataset {dataset_name}: {e}")
                traceback.print_exc()
                failed += len(self.config["variants"])

        print(f"\n{'=' * 80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nProgress: {completed}/{total} variants completed")
        if failed > 0:
            print(f"Failed:   {failed}/{total} variants")
        print(f"\nResults saved to: {self.results_dir}")

    # =========================================================================
    # Dataset loop
    # =========================================================================

    def run_dataset(self, dataset_name: str) -> int:
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        print(f"Loading {dataset_name}...")
        try:
            df, sentences, sentence2dcs_map, predicate_space = self.load_dataset(
                dataset_name
            )
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return len(self.config["variants"])

        # Keep only GT constraint sentences (rules != null / non-empty)
        gt_sentences = [
            s for s in sentences if s in sentence2dcs_map and sentence2dcs_map[s]
        ]
        gt_sentence2dcs_map = {s: sentence2dcs_map[s] for s in gt_sentences}

        num_total = len(sentences)
        num_gt = len(gt_sentences)

        print(f"Total sentences in document: {num_total}")
        print(
            f"GT constraint sentences:     {num_gt}  ← only these are fed to the mapper"
        )
        print(f"Non-constraint sentences:    {num_total - num_gt}  ← skipped\n")

        results = {}
        failed_count = 0

        for i, variant in enumerate(self.config["variants"], 1):
            name = variant["name"]
            impl = variant["mapper"].get("implementation", "llm")

            print(f"\n[{i}/{len(self.config['variants'])}] Variant: {name}")
            print(f"  Implementation: {impl}")
            print(f"  Model:          {variant['mapper'].get('model', _NA)}")

            try:
                result = self.run_variant(
                    name=name,
                    variant=variant,
                    gt_sentences=gt_sentences,
                    gt_sentence2dcs_map=gt_sentence2dcs_map,
                    predicate_space=predicate_space,
                    df=df,
                    dataset_name=dataset_name,
                )
                results[name] = result
                self.append_result(
                    self._build_result_dict(
                        dataset_name, variant, result, num_total, num_gt
                    )
                )
                print(f"  Variant completed")

            except Exception as e:
                print(f"  Error: Variant failed: {e}")
                traceback.print_exc()
                failed_count += 1
                self.append_result(
                    self._build_failure_dict(
                        dataset_name, variant, str(e), num_total, num_gt
                    )
                )

        if results:
            self._print_comparison_table(results)

        return failed_count

    # =========================================================================
    # Variant runner
    # =========================================================================

    def run_variant(
        self,
        name: str,
        variant: Dict,
        gt_sentences: List[str],
        gt_sentence2dcs_map: Dict,
        predicate_space: Dict,
        df,
        dataset_name: str,
    ) -> Dict:
        mapper_config = variant["mapper"]
        use_context = variant.get("use_context", False)
        num_rows_context = variant.get("num_rows_context", 3)
        implementation = mapper_config.get("implementation", "llm")

        # Build context if needed
        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, num_rows=num_rows_context)

        # Create mapper
        print(f"  Creating mapper ({implementation})...")
        mapper = self._create_mapper(mapper_config, predicate_space, df)

        # ------------------------------------------------------------------
        # Run mapper on GT sentences only
        # ------------------------------------------------------------------
        print(f"  Mapping {len(gt_sentences)} GT sentences...")
        t0 = time.perf_counter()

        # BaseMapper.map returns List[Tuple[sentence, predicates]].
        # Keep backward compatibility if an implementation returns only
        # predicate lists.
        raw_outputs = mapper.map(gt_sentences, context=context)

        processing_errors = getattr(mapper, "processing_errors", [])
        if processing_errors:
            raise RuntimeError(
                "Mapper did not complete all sentences; "
                f"first processing error: {processing_errors[0]}"
            )

        runtime_s = time.perf_counter() - t0

        # Normalize to sentence/rules tuples.
        if (
            raw_outputs
            and isinstance(raw_outputs[0], tuple)
            and len(raw_outputs[0]) == 2
        ):
            predictions_raw: List[Tuple[str, Optional[List[Rule]]]] = raw_outputs
        else:
            predictions_raw = list(zip(gt_sentences, raw_outputs))

        # ------------------------------------------------------------------
        # Semantic normalization (minor syntactic variants → GT predicates)
        # ------------------------------------------------------------------
        predictions = self._map_semantically_valid_constraint_predicates(
            predictions_raw, gt_sentence2dcs_map
        )

        for (sent, raw_pred), (_, mapped_pred) in zip(predictions_raw, predictions):
            if raw_pred != mapped_pred:
                print(
                    f"    Normalized:\n"
                    f"      Sentence:    {sent[:80]}\n"
                    f"      Raw Pred:    {raw_pred}\n"
                    f"      Mapped Pred: {mapped_pred}"
                )

        # ------------------------------------------------------------------
        # 1. Pairwise mapping quality — all GT sentences
        # ------------------------------------------------------------------
        print(f"  Evaluating pairwise quality ({len(gt_sentences)} GT sentences)...")
        map_evaluator = MappingEvaluator(
            ground_truth=gt_sentence2dcs_map,
            df=df,
            csv_path=self._get_primary_csv_path(predicate_space),
        )
        map_metrics, sentence_results = map_evaluator.evaluate(predictions)

        # ------------------------------------------------------------------
        # 2. Aggregate violation — pool all GT sentence predictions
        # ------------------------------------------------------------------
        print(f"  Evaluating aggregate violation (GT sentences pooled)...")

        all_pred_violations: set = set()
        for _, preds in predictions:
            if preds:
                all_pred_violations.update(map_evaluator._execute_rule(preds))

        all_gold_violations: set = set()
        for rule in gt_sentence2dcs_map.values():
            all_gold_violations.update(map_evaluator._execute_rule(rule))

        overlap = all_gold_violations & all_pred_violations
        total_gv = len(all_gold_violations)
        total_pv = len(all_pred_violations)
        total_ov = len(overlap)
        agg_vp = total_ov / total_pv if total_pv else 0.0
        agg_vr = total_ov / total_gv if total_gv else 0.0
        agg_vf = (2 * agg_vp * agg_vr / (agg_vp + agg_vr)) if (agg_vp + agg_vr) else 0.0

        map_metrics.aggregate_violation_precision = agg_vp
        map_metrics.aggregate_violation_recall = agg_vr
        map_metrics.aggregate_violation_f1 = agg_vf
        map_metrics.total_gold_violations = total_gv
        map_metrics.total_pred_violations = total_pv
        map_metrics.total_overlap_violations = total_ov

        map_evaluator.close()

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------
        print(f"  Saving outputs...")
        self.save_outputs(
            outputs=[
                {
                    "sentence": sent,
                    "gold_constraint": str(gt_sentence2dcs_map[sent]),
                    "pred_constraint": str(preds) if preds else None,
                    "pred_is_non_null": preds is not None and len(preds) > 0,
                    **(
                        {"violated_when": getattr(mapper, "violation_notes", {}).get(sent)}
                        if getattr(mapper, "explain_violation", False)
                        else {}
                    ),
                }
                for sent, preds in predictions
            ],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="gt_only_mapper",
        )

        self._save_detailed_sentence_results(
            sentence_results=sentence_results,
            predictions=predictions,
            gt_sentence2dcs_map=gt_sentence2dcs_map,
            dataset_name=dataset_name,
            variant_name=name,
        )

        # ------------------------------------------------------------------
        # Print metrics
        # ------------------------------------------------------------------
        print(f"\n  ── RUNTIME " + "─" * 59)
        print(
            f"     Pipeline runtime: {runtime_s:.2f}s  ({len(gt_sentences)} GT sentences)"
        )

        print(f"\n  ── AGGREGATE VIOLATION DISCOVERY (GT sentences pooled) " + "─" * 15)
        print(f"     Precision: {agg_vp:.3f}")
        print(f"     Recall:    {agg_vr:.3f}")
        print(f"     F1:        {agg_vf:.3f}  [PRIMARY]")
        print(f"     Gold violations:      {total_gv}")
        print(f"     Predicted violations: {total_pv}")
        print(f"     Overlap violations:   {total_ov}")

        print(f"\n  ── CONSTRAINT EXACT MATCH (GT sentences) " + "─" * 29)
        print(f"     Precision: {map_metrics.constraint_exact_precision:.3f}")
        print(f"     Recall:    {map_metrics.constraint_exact_recall:.3f}")
        print(f"     F1:        {map_metrics.constraint_exact_f1:.3f}  [PRIMARY]")
        print(
            f"     TP: {map_metrics.constraint_exact_tp}  FP: {map_metrics.constraint_exact_fp}  FN: {map_metrics.constraint_exact_fn}"
        )
        print(
            f"     Predicted: {map_metrics.num_pred_constraints}, Gold: {map_metrics.num_gold_constraints}"
        )

        print(
            f"\n  ── PAIRWISE SYNTACTIC QUALITY (n={map_metrics.pairwise_strict_total}) "
            + "─" * 30
        )
        print(
            f"     Exact Match:       {map_metrics.pairwise_strict_accuracy:.3f} ({map_metrics.pairwise_strict_correct}/{map_metrics.pairwise_strict_total})"
        )
        print(
            f"     Predicate Jaccard: {map_metrics.pairwise_avg_predicate_jaccard:.3f} (±{map_metrics.pairwise_std_predicate_jaccard:.3f})"
        )

        print(f"\n  ── PAIRWISE VIOLATION DISCOVERY " + "─" * 39)
        print(
            f"     Exact Match:       {map_metrics.pairwise_execution_accuracy:.3f} ({map_metrics.pairwise_execution_correct}/{map_metrics.pairwise_execution_total})"
        )
        print(
            f"     Precision:         {map_metrics.pairwise_avg_execution_precision:.3f} (±{map_metrics.pairwise_std_execution_precision:.3f})"
        )
        print(
            f"     Recall:            {map_metrics.pairwise_avg_execution_recall:.3f} (±{map_metrics.pairwise_std_execution_recall:.3f})"
        )
        print(
            f"     F1:                {map_metrics.pairwise_avg_execution_f1:.3f} (±{map_metrics.pairwise_std_execution_f1:.3f})"
        )
        print(
            f"     Violation Jaccard: {map_metrics.pairwise_avg_execution_jaccard:.3f} (±{map_metrics.pairwise_std_execution_jaccard:.3f})"
        )

        gap = (
            map_metrics.pairwise_execution_accuracy
            - map_metrics.pairwise_strict_accuracy
        )
        print(f"\n  INSIGHT: Syntactic vs Violation Gap = {gap:.3f}")
        if gap > 0:
            print(
                f"     {gap*100:.1f}% of constraints find correct violations despite syntactic differences"
            )
        elif gap < 0:
            print(
                f"     Warning: {abs(gap)*100:.1f}% of syntactically correct constraints fail on violations"
            )

        return {
            "map_metrics": map_metrics,
            "runtime_s": runtime_s,
            "implementation": implementation,
            "model": mapper_config.get("model", _NA),
            "model_output_failures": len(getattr(mapper, "output_failures", [])),
            "diagnostics": mapping_diagnostics(
                map_metrics, [preds for _, preds in predictions], predicate_space
            ),
        }

    # =========================================================================
    # Mapper factory
    # =========================================================================

    def _create_mapper(self, mapper_config: Dict, predicate_space: Dict, df):
        implementation = mapper_config.get("implementation", "llm")
        csv_path = self._get_primary_csv_path(predicate_space)

        if implementation == "llm":
            return LLMMapper(
                model_name=mapper_config["model"],
                provider=mapper_config.get("provider"),
                prompt_name=mapper_config["prompt_key"],
                allowed_predicates=predicate_space.get("predicates"),
                predicate_generator=predicate_space.get("_generator"),
                column_context=mapper_config.get("column_context", False),
                explain_violation=mapper_config.get("explain_violation", False),
                chunk_size=mapper_config.get("batch_size", 10),
                filter_predicates=mapper_config.get("filter_predicates", False),
                predicate_tables=predicate_space["tables"],
                max_candidate_predicates=mapper_config.get(
                    "max_candidate_predicates", 1000
                ),
                max_relevant_columns=mapper_config.get("max_relevant_columns", 12),
                max_tokens=mapper_config.get("max_tokens"),
                structured_output=mapper_config.get("structured_output", True),
            )
        elif implementation == "multi_step_refinement":
            return MultiStepRefinementMapper(
                model_name=mapper_config["model"],
                provider=mapper_config.get("provider"),
                predicate_space=predicate_space,
                df=df,
                csv_path=csv_path,
                max_refinement_iterations=mapper_config.get(
                    "max_refinement_iterations", 2
                ),
                prompt_file=mapper_config.get("prompt_file", "multi_step_refinement"),
                max_tokens=mapper_config.get("max_tokens"),
                structured_output=mapper_config.get("structured_output", True),
                enforce_rule_gate=mapper_config.get("enforce_rule_gate", True),
                include_column_descriptions=mapper_config.get(
                    "include_column_descriptions", False
                ),
                max_column_description_chars=mapper_config.get(
                    "max_column_description_chars", 200
                ),
            )
        else:
            raise ValueError(f"Unknown mapper implementation: {implementation}")

    def _build_context(self, df, num_rows: int = 3):
        from src.utils.table_context import TableContextBuilder

        return TableContextBuilder(sample_strategy="head", num_rows=num_rows).build(df)

    def _get_primary_csv_path(self, predicate_space: Dict) -> Optional[str]:
        if predicate_space.get("_sample_rows") is not None:
            return None
        table_files = predicate_space.get("_table_files") or {}
        if not table_files:
            return None
        return next(iter(table_files.values()))

    # =========================================================================
    # Semantic normalization (inherited pattern from run_mapper_eval_full.py)
    # =========================================================================

    def _map_semantically_valid_constraint_predicates(
        self,
        predictions: List[Tuple[str, Optional[List[Rule]]]],
        sentence2dcs_map: Dict[str, List[Rule]],
    ) -> List[Tuple[str, Optional[List[Rule]]]]:
        mapped = []
        for sentence, pred_constraint in predictions:
            if pred_constraint is None or sentence not in sentence2dcs_map:
                mapped.append((sentence, pred_constraint))
                continue
            gt_predicates = predicate_strings(sentence2dcs_map[sentence])
            normalized = transform_predicates(
                pred_constraint,
                lambda predicate: self._match_to_gt_predicate(predicate, gt_predicates),
            )
            mapped.append((sentence, normalized))
        return mapped

    def _match_to_gt_predicate(self, pred: str, gt_predicates: List[str]) -> str:
        for gt_p in gt_predicates:
            if self._are_semantically_equivalent(pred, gt_p):
                return gt_p
        return pred

    def _are_semantically_equivalent(self, p1: str, p2: str) -> bool:
        if p1.lower() == p2.lower():
            return True
        col1, op1, val1 = self._parse_predicate(p1)
        col2, op2, val2 = self._parse_predicate(p2)
        if col1 is None or col1 != col2:
            return False
        if op1 in ("=", "LIKE") and op2 in ("=", "LIKE"):
            s1, s2 = self._extract_string(val1), self._extract_string(val2)
            if s1 and s2 and min(len(s1), len(s2)) >= 4:
                return s1.lower() in s2.lower() or s2.lower() in s1.lower()
        if op1 in ("!=", "NOT LIKE") and op2 in ("!=", "NOT LIKE"):
            s1, s2 = self._extract_string(val1), self._extract_string(val2)
            if s1 and s2 and min(len(s1), len(s2)) >= 4:
                return s1.lower() in s2.lower() or s2.lower() in s1.lower()
        if op1 == "IN" and op2 == "=":
            vals = self._extract_in_values(val1)
            s2 = self._extract_string(val2)
            if len(vals) == 1 and s2:
                return vals[0].lower() == s2.lower()
        if op2 == "IN" and op1 == "=":
            vals = self._extract_in_values(val2)
            s1 = self._extract_string(val1)
            if len(vals) == 1 and s1:
                return vals[0].lower() == s1.lower()
        if op1 in (">", ">=") and op2 in (">", ">="):
            return val1 == val2
        if op1 in ("<", "<=") and op2 in ("<", "<="):
            return val1 == val2
        return False

    def _parse_predicate(
        self, pred: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pred = pred.strip()
        for op in (
            "NOT LIKE",
            "NOT IN",
            "IS NOT NULL",
            "IS NULL",
            "LIKE",
            "IN",
            "!=",
            "<=",
            ">=",
            "<",
            ">",
            "=",
        ):
            upper = pred.upper()
            idx = upper.find(f" {op} ")
            if idx != -1:
                return pred[:idx].strip(), op, pred[idx + len(op) + 2 :].strip()
            if upper.endswith(f" {op}"):
                return pred[: -(len(op) + 1)].strip(), op, None
        return None, None, None

    def _extract_string(self, val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        val = val.strip()
        if val.startswith("'") and val.endswith("'"):
            return val[1:-1].replace("%", "").strip()
        return None

    def _extract_in_values(self, val: Optional[str]) -> List[str]:
        if not val:
            return []
        val = val.strip().strip("()")
        return [v.strip().strip("'") for v in val.split(",")]

    # =========================================================================
    # Detailed CSV output
    # =========================================================================

    def _save_detailed_sentence_results(
        self,
        sentence_results,
        predictions: List[Tuple[str, Optional[List[Rule]]]],
        gt_sentence2dcs_map: Dict,
        dataset_name: str,
        variant_name: str,
    ):
        import pandas as pd

        gt_by_sentence = {r.sentence: r for r in sentence_results}
        records = []

        for sent, preds in predictions:
            record = {
                "sentence": sent,
                "gold_constraint": str(gt_sentence2dcs_map.get(sent)),
                "pred_constraint": str(preds) if preds else None,
                "pred_is_non_null": preds is not None and len(preds) > 0,
            }

            r = gt_by_sentence.get(sent)
            if r is not None:
                record.update(
                    {
                        "pairwise_syntactic_exact_match": r.strict_correct,
                        "pairwise_syntactic_predicate_jaccard": r.predicate_jaccard,
                        "pairwise_logical_match": r.logical_match,
                        "pred_execution_errors": r.pred_execution_errors,
                    }
                )
                if r.execution_exact is not None:
                    record.update(
                        {
                            "pairwise_violation_exact_match": r.execution_exact,
                            "pairwise_violation_precision": r.execution_precision,
                            "pairwise_violation_recall": r.execution_recall,
                            "pairwise_violation_f1": r.execution_f1,
                            "pairwise_violation_jaccard": r.execution_jaccard,
                            "num_gold_violations": (
                                len(r.gold_violations) if r.gold_violations else 0
                            ),
                            "num_pred_violations": (
                                len(r.pred_violations) if r.pred_violations else 0
                            ),
                            "num_overlap_violations": (
                                len(r.gold_violations & r.pred_violations)
                                if r.gold_violations and r.pred_violations
                                else 0
                            ),
                        }
                    )

            records.append(record)

        df_out = pd.DataFrame(records)
        filename = f"{dataset_name}_{variant_name.replace(' ', '_')}_detailed.csv"
        filepath = self.metrics_dir / filename
        df_out.to_csv(filepath, index=False)
        print(f"  Saved detailed results: {filepath.name}")

    # =========================================================================
    # Result / failure dicts for summary CSV
    # =========================================================================

    def _build_result_dict(
        self,
        dataset_name: str,
        variant: Dict,
        result: Dict,
        num_total: int,
        num_gt: int,
    ) -> Dict:
        m = result["map_metrics"]
        return {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": result["implementation"],
            "model": result["model"],
            "prompt_key": variant["mapper"].get("prompt_key", _NA),
            "prompt_file": variant["mapper"].get("prompt_file", _NA),
            "use_context": variant.get("use_context", False),
            "num_rows_context": variant.get("num_rows_context", _NA),
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "runtime_s": result["runtime_s"],
            "model_output_failures": result["model_output_failures"],
            "status": "success",
            # Aggregate violation
            "agg_viol_precision": m.aggregate_violation_precision,
            "agg_viol_recall": m.aggregate_violation_recall,
            "agg_viol_f1": m.aggregate_violation_f1,
            "total_gold_violations": m.total_gold_violations,
            "total_pred_violations": m.total_pred_violations,
            "total_overlap_violations": m.total_overlap_violations,
            # Constraint exact match
            "constraint_exact_precision": m.constraint_exact_precision,
            "constraint_exact_recall": m.constraint_exact_recall,
            "constraint_exact_f1": m.constraint_exact_f1,
            "constraint_exact_tp": m.constraint_exact_tp,
            "constraint_exact_fp": m.constraint_exact_fp,
            "constraint_exact_fn": m.constraint_exact_fn,
            "num_gold_constraints": m.num_gold_constraints,
            "num_pred_constraints": m.num_pred_constraints,
            # Pairwise syntactic
            "pairwise_syn_exact_match": m.pairwise_strict_accuracy,
            "pairwise_syn_exact_count": m.pairwise_strict_correct,
            "pairwise_syn_total": m.pairwise_strict_total,
            "pairwise_syn_jaccard_mean": m.pairwise_avg_predicate_jaccard,
            "pairwise_syn_jaccard_std": m.pairwise_std_predicate_jaccard,
            # Pairwise violation
            "pairwise_viol_exact_match": m.pairwise_execution_accuracy,
            "pairwise_viol_exact_count": m.pairwise_execution_correct,
            "pairwise_viol_total": m.pairwise_execution_total,
            "pairwise_viol_prec_mean": m.pairwise_avg_execution_precision,
            "pairwise_viol_prec_std": m.pairwise_std_execution_precision,
            "pairwise_viol_rec_mean": m.pairwise_avg_execution_recall,
            "pairwise_viol_rec_std": m.pairwise_std_execution_recall,
            "pairwise_viol_f1_mean": m.pairwise_avg_execution_f1,
            "pairwise_viol_f1_std": m.pairwise_std_execution_f1,
            "pairwise_viol_jac_mean": m.pairwise_avg_execution_jaccard,
            "pairwise_viol_jac_std": m.pairwise_std_execution_jaccard,
            **result.get("diagnostics", {}),
        }

    def _build_failure_dict(
        self, dataset_name: str, variant: Dict, error: str, num_total: int, num_gt: int
    ) -> Dict:
        return {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": variant["mapper"].get("implementation", _NA),
            "model": variant["mapper"].get("model", _NA),
            "prompt_key": variant["mapper"].get("prompt_key", _NA),
            "prompt_file": variant["mapper"].get("prompt_file", _NA),
            "use_context": variant.get("use_context", False),
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "status": "failed",
            "error": error,
            "runtime_s": _NA,
            "agg_viol_f1": _NA,
            "constraint_exact_f1": _NA,
            "pairwise_syn_exact_match": _NA,
            "pairwise_viol_f1_mean": _NA,
        }

    # =========================================================================
    # Comparison table
    # =========================================================================

    def _print_comparison_table(self, results: Dict):
        def _fmt(v, width: int, precision: int = 3) -> str:
            if isinstance(v, (int, float)):
                return f"{v:<{width}.{precision}f}"
            return f"{str(v):<{width}}"

        print("\n" + "=" * 130)
        print("COMPARISON TABLE  —  GT sentences only, no classifier")
        print("=" * 130)

        hdr = (
            f"  {'Variant':<45} {'Impl':<10} {'Model':<22} "
            f"{'AggViol F1':<12} {'ConstExact F1':<15} "
            f"{'SynExact':<10} {'SynJacc':<10} "
            f"{'ViolF1':<10} {'ViolJacc':<10} {'Runtime(s)':<10}"
        )
        print(hdr)
        print("  " + "-" * 128)

        for name, r in results.items():
            m = r["map_metrics"]
            print(
                f"  {name:<45} {r['implementation']:<10} {r['model']:<22} "
                f"{_fmt(m.aggregate_violation_f1, 12)} {_fmt(m.constraint_exact_f1, 15)} "
                f"{_fmt(m.pairwise_strict_accuracy, 10)} {_fmt(m.pairwise_avg_predicate_jaccard, 10)} "
                f"{_fmt(m.pairwise_avg_execution_f1, 10)} {_fmt(m.pairwise_avg_execution_jaccard, 10)} "
                f"{_fmt(r['runtime_s'], 10, precision=2)}"
            )

        if results:
            print()
            best_agg = max(
                results.items(),
                key=lambda x: x[1]["map_metrics"].aggregate_violation_f1,
            )
            best_const = max(
                results.items(), key=lambda x: x[1]["map_metrics"].constraint_exact_f1
            )
            best_syn = max(
                results.items(),
                key=lambda x: x[1]["map_metrics"].pairwise_strict_accuracy,
            )
            best_viol = max(
                results.items(),
                key=lambda x: x[1]["map_metrics"].pairwise_avg_execution_f1,
            )
            fastest = min(results.items(), key=lambda x: x[1]["runtime_s"])

            print(
                f"  Best Aggregate Violation F1:   {best_agg[0][:55]} ({best_agg[1]['map_metrics'].aggregate_violation_f1:.3f})"
            )
            print(
                f"  Best Constraint Exact F1:      {best_const[0][:55]} ({best_const[1]['map_metrics'].constraint_exact_f1:.3f})"
            )
            print(
                f"  Best Syntactic Exact Match:    {best_syn[0][:55]} ({best_syn[1]['map_metrics'].pairwise_strict_accuracy:.3f})"
            )
            print(
                f"  Best Pairwise Violation F1:    {best_viol[0][:55]} ({best_viol[1]['map_metrics'].pairwise_avg_execution_f1:.3f})"
            )
            print(
                f"  Fastest:                       {fastest[0][:55]} ({fastest[1]['runtime_s']:.2f}s)"
            )

        print("=" * 130)


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GT-only mapper evaluation: pure mapping quality, no classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mapper_eval_gt_only.py configs/mappers/gt_only.yaml
        """,
    )
    parser.add_argument("config_file", help="Path to YAML config file")
    args = parser.parse_args()

    try:
        experiment = MapperGTOnlyExperiment(args.config_file)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting GT-only mapper evaluation...")
    main()
