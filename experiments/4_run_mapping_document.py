#!/usr/bin/env python3
"""
Document-Level Mapper Evaluation Script

Sends the entire document (all sentences) to the mapper in a single API call
and evaluates the resulting constraint set against the ground truth.

Since there is no sentence-level alignment in document mode, only aggregate
violation discovery metrics are computable:
  - Aggregate violation precision / recall / F1
  - Gold / predicted / overlap violation counts
  - Number of predicted rules

All pairwise and classification metrics are N/A by design.

Supported document mapper implementations (set via 'implementation' in YAML):
  - llm_document : LLMDocumentMapper

Usage:
    python experiments/4_run_mapping_document.py experiments/configs/4_run_mapping_document.yaml
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment
from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
from src.mappers.llm_document_mapper import LLMDocumentMapper
from src.evaluation.mapping import MappingEvaluator

_NA = "N/A"


# =============================================================================
# EXPERIMENT
# =============================================================================


class DocumentMapperExperiment(BaseExperiment):
    """
    Document-level mapper evaluation.

    The entire document is sent in a single LLM call.  The predicted rule set
    is compared to all ground-truth constraints via aggregate violation
    discovery (row-level execution against the database).

    Metrics per variant:
      Aggregate violation precision / recall / F1  [PRIMARY]
      Gold violations / Predicted violations / Overlap
      Number of predicted rules
    """

    def __init__(self, config_file: str):
        super().__init__(config_file)
        if self.config["experiment_type"] != "mapper_document":
            raise ValueError(
                f"Expected experiment_type='mapper_document', "
                f"got '{self.config['experiment_type']}'"
            )

    # =========================================================================
    # Top-level run
    # =========================================================================

    def run(self):
        print("\n" + "=" * 80)
        print(f"DOCUMENT MAPPER EVALUATION: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp:         {self.timestamp}")
        print("\nScope: entire document sent in one call — aggregate violation only.")
        print("All pairwise / classification metrics are N/A.")
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
    # Per-dataset run
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

        gt_sentence2dcs_map = {s: dcs for s, dcs in sentence2dcs_map.items() if dcs}
        num_total = len(sentences)
        num_gt = len(gt_sentence2dcs_map)

        print(f"Total sentences:          {num_total}")
        print(f"GT constraint sentences:  {num_gt}")
        print(f"Non-constraint sentences: {num_total - num_gt}\n")

        results = {}
        failed_count = 0

        for i, variant in enumerate(self.config["variants"], 1):
            name = variant["name"]
            impl = variant["document_mapper"].get("implementation", "llm_document")
            model = variant["document_mapper"].get("model", "?")

            print(f"\n[{i}/{len(self.config['variants'])}] Variant: {name}")
            print(f"  Implementation: {impl}")
            print(f"  Model:          {model}")

            try:
                result = self.run_variant(
                    name=name,
                    variant=variant,
                    all_sentences=sentences,
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
            self._print_comparison_table(results, dataset_name)

        return failed_count

    # =========================================================================
    # Per-variant run
    # =========================================================================

    def run_variant(
        self,
        name: str,
        variant: Dict,
        all_sentences: List[str],
        gt_sentence2dcs_map: Dict,
        predicate_space: Dict,
        df,
        dataset_name: str,
    ) -> Dict:
        use_context = variant.get("use_context", True)
        num_rows_context = variant.get("num_rows_context", 3)

        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, num_rows=num_rows_context)

        # Document mapper needs the predicate list in context
        predicates_list_str = "\n".join(predicate_space["predicates"])
        context["predicates_list"] = predicates_list_str

        print(f"  Processing document ({len(all_sentences)} sentences in one call)...")

        doc_mapper_config = variant["document_mapper"]
        document_mapper = LLMDocumentMapper(
            model_name=doc_mapper_config["model"],
            provider=doc_mapper_config.get("provider"),
            prompt_name=doc_mapper_config["prompt_key"],
        )
        pipeline = NormaPipeline(
            mode=ProcessingMode.DOCUMENT, document_mapper=document_mapper
        )

        t0 = time.perf_counter()
        predicted_rules = pipeline.process(all_sentences, context=context)
        runtime_s = time.perf_counter() - t0
        # Returns List[List[str]] — a flat list of predicted constraint predicate-lists

        # ------------------------------------------------------------------
        # Aggregate violation — pool all predicted constraints vs all GT
        # ------------------------------------------------------------------
        print(f"  Evaluating aggregate violation...")

        map_evaluator = MappingEvaluator(ground_truth=gt_sentence2dcs_map, df=df)

        all_pred_violations: set = set()
        for preds in predicted_rules:
            if preds:
                all_pred_violations.update(map_evaluator._execute_rule(preds))

        all_gold_violations: set = set()
        for rule in gt_sentence2dcs_map.values():
            all_gold_violations.update(map_evaluator._execute_rule(rule))

        map_evaluator.close()

        overlap = all_gold_violations & all_pred_violations
        total_gv = len(all_gold_violations)
        total_pv = len(all_pred_violations)
        total_ov = len(overlap)
        agg_vp = total_ov / total_pv if total_pv else 0.0
        agg_vr = total_ov / total_gv if total_gv else 0.0
        agg_vf = 2 * agg_vp * agg_vr / (agg_vp + agg_vr) if (agg_vp + agg_vr) else 0.0

        # ------------------------------------------------------------------
        # Save raw outputs
        # ------------------------------------------------------------------
        print(f"  Saving outputs...")
        self.save_outputs(
            outputs=[{"rule": str(rule)} for rule in predicted_rules],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="document_mapper",
        )

        # ------------------------------------------------------------------
        # Print metrics
        # ------------------------------------------------------------------
        print(f"\n  ── RUNTIME " + "─" * 59)
        print(f"     Pipeline runtime: {runtime_s:.2f}s")
        print(f"\n  ── AGGREGATE VIOLATION DISCOVERY " + "─" * 38)
        print(f"     Precision: {agg_vp:.3f}")
        print(f"     Recall:    {agg_vr:.3f}")
        print(f"     F1:        {agg_vf:.3f}  [PRIMARY]")
        print(f"     Gold violations:      {total_gv}")
        print(f"     Predicted violations: {total_pv}")
        print(f"     Overlap violations:   {total_ov}")
        print(f"     Predicted rules:      {len(predicted_rules)}")
        print(
            f"\n  ── ALL OTHER METRICS: N/A (no sentence-level alignment in DOCUMENT mode) ──"
        )

        return {
            "aggregate_violation_precision": agg_vp,
            "aggregate_violation_recall": agg_vr,
            "aggregate_violation_f1": agg_vf,
            "total_gold_violations": total_gv,
            "total_pred_violations": total_pv,
            "total_overlap_violations": total_ov,
            "num_predicted_rules": len(predicted_rules),
            "runtime_s": runtime_s,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_context(self, df, num_rows: int = 3):
        from src.utils.table_context import TableContextBuilder

        return TableContextBuilder(sample_strategy="head", num_rows=num_rows).build(df)

    def _build_result_dict(
        self,
        dataset_name: str,
        variant: Dict,
        result: Dict,
        num_total: int,
        num_gt: int,
    ) -> Dict:
        doc_cfg = variant["document_mapper"]
        return {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": doc_cfg.get("implementation", _NA),
            "model": doc_cfg.get("model", _NA),
            "prompt_key": doc_cfg.get("prompt_key", _NA),
            "mode": "DOCUMENT",
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "status": "success",
            "runtime_s": result["runtime_s"],
            "aggregate_violation_precision": result["aggregate_violation_precision"],
            "aggregate_violation_recall": result["aggregate_violation_recall"],
            "aggregate_violation_f1": result["aggregate_violation_f1"],
            "total_gold_violations": result["total_gold_violations"],
            "total_pred_violations": result["total_pred_violations"],
            "total_overlap_violations": result["total_overlap_violations"],
            "num_predicted_rules": result["num_predicted_rules"],
        }

    def _build_failure_dict(
        self, dataset_name: str, variant: Dict, error: str, num_total: int, num_gt: int
    ) -> Dict:
        doc_cfg = variant.get("document_mapper", {})
        return {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": doc_cfg.get("implementation", _NA),
            "model": doc_cfg.get("model", _NA),
            "prompt_key": doc_cfg.get("prompt_key", _NA),
            "mode": "DOCUMENT",
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "status": "failed",
            "error": error,
            "runtime_s": _NA,
            "aggregate_violation_precision": _NA,
            "aggregate_violation_recall": _NA,
            "aggregate_violation_f1": _NA,
            "total_gold_violations": _NA,
            "total_pred_violations": _NA,
            "total_overlap_violations": _NA,
            "num_predicted_rules": _NA,
        }

    def _print_comparison_table(self, results: Dict, dataset_name: str):
        print("\n" + "=" * 110)
        print(f"COMPARISON TABLE — {dataset_name.upper()}")
        print("=" * 110)
        print(f"\n  Note: Only aggregate violation is computable in DOCUMENT mode.")

        hdr = (
            f"  {'Variant':<42} {'Model':<24} "
            f"{'Agg Viol P':<12} {'Agg Viol R':<12} {'Agg Viol F1':<13} "
            f"{'Gold':<7} {'Pred':<7} {'Overlap':<9} {'Rules':<7} {'Runtime(s)'}"
        )
        print(hdr)
        print("  " + "-" * 130)

        for name, result in results.items():
            variant = next(v for v in self.config["variants"] if v["name"] == name)
            model = variant["document_mapper"].get("model", _NA)
            print(
                f"  {name:<42} {model:<24} "
                f"{result['aggregate_violation_precision']:<12.3f} "
                f"{result['aggregate_violation_recall']:<12.3f} "
                f"{result['aggregate_violation_f1']:<13.3f} "
                f"{result['total_gold_violations']:<7} "
                f"{result['total_pred_violations']:<7} "
                f"{result['total_overlap_violations']:<9} "
                f"{result['num_predicted_rules']:<7} "
                f"{result['runtime_s']:.2f}s"
            )

        if len(results) > 1:
            best = max(results.items(), key=lambda x: x[1]["aggregate_violation_f1"])
            fastest = min(results.items(), key=lambda x: x[1]["runtime_s"])
            print(
                f"\n  Best Agg Violation F1: {best[0]} ({best[1]['aggregate_violation_f1']:.3f})"
            )
            print(
                f"  Fastest Runtime:       {fastest[0]} ({fastest[1]['runtime_s']:.2f}s)"
            )

        print("=" * 110)


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Document-level mapper evaluation — aggregate violation discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/4_run_mapping_document.py experiments/configs/4_run_mapping_document.yaml
        """,
    )
    parser.add_argument("config_file", help="Path to YAML config file")
    args = parser.parse_args()

    try:
        experiment = DocumentMapperExperiment(args.config_file)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting document mapper evaluation experiment...")
    main()
