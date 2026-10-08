#!/usr/bin/env python3
"""
Full Mapper Evaluation Script

Feeds ALL sentences to the mapper and evaluates three aspects:

1. CLASSIFICATION — binary: does the mapper output any constraint for this sentence?
   Metrics (over all sentences): precision, recall, F1, accuracy, confusion matrix
   Ground truth positive = sentence in sentence2dcs_map with non-empty constraints

2. MAPPING QUALITY — how well are the correctly identified constraints translated?
   - Pairwise syntactic quality      (GT sentences only - that method predicted as constraints)
   - Pairwise violation discovery    (GT sentences only - that method predicted as constraints)
   - Aggregate violation discovery   (ALL sentences: all predicted vs all GT constraints)

Supported mapper implementations (set via 'implementation' in YAML config):
   - llm                : Standard LLMMapper
   - multi_step_refinement : MultiStepRefinementMapper
   - external           : Plug in any external tool (e.g. Text2SQL) via TODO stub below

Supported modes:
   - SENTENCE_DIRECT   : Map all sentences directly (no classifier).
                         Full metrics: classification + pairwise + aggregate violation.
   - SENTENCE_FILTERED : Classify first, then map only rule sentences.
                         Full metrics: classification + pairwise + aggregate violation.
                         Classification is evaluated over ALL sentences (same as DIRECT).
   - DOCUMENT          : Process entire document in one call.
                         Only aggregate violation is computable (no sentence alignment).
                         All other metric fields are N/A.

Usage:
    python run_mapper_eval_full.py configs/mappers/sentence_level.yaml

Example YAML variant for external tool:
    - name: "MyText2SQL"
      mode: SENTENCE_DIRECT
      mapper:
        implementation: external
        # any extra keys here are passed as config to call_external_tool_single
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment
from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
from src.classifiers.llm import LLMRuleClassifier
from src.mappers.llm import LLMMapper
from src.mappers.llm_document_mapper import LLMDocumentMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.evaluation.mapping import MappingEvaluator
from src.evaluation.classification import ClassificationEvaluator
from src.llm.helper import PromptLoader
from src.predicates import Rule, predicate_strings, transform_predicates

_NA = "N/A"


# =============================================================================
# EXTERNAL TOOL STUB
# =============================================================================


def call_external_tool_single(
    sentence: str,
    context: Dict,
    config: Dict,
) -> Optional[List[Rule]]:
    """
    Call an external constraint extraction / Text2SQL tool for a single sentence.

    This function is called once per sentence when implementation: external is set.
    Fill in the body with your tool's API call.

    Args:
        sentence : The natural language sentence to process.
        context  : Table context dict (keys depend on TableContextBuilder output,
                   e.g. context['table_context'] is a string summary of the table).
                   May be empty dict if use_context: false in the variant config.
        config   : The full mapper config dict from the YAML variant, so you can
                   pass any extra parameters (model name, endpoint URL, etc.) by
                   just adding keys to the YAML.

    Returns:
        List[Rule] — independent predicted rules
        None       — if the sentence is not classified as a constraint
    """
    raise NotImplementedError(
        "call_external_tool_single is not implemented. "
        "Edit the TODO stub in run_mapper_eval_full.py."
    )


# =============================================================================
# EXPERIMENT
# =============================================================================


class MapperEvalFullExperiment(BaseExperiment):
    """
    Full mapper evaluation: classification + mapping quality + violation discovery.
    Supports llm, multi_step_refinement, and external tool backends.

    Evaluation scope per mode:
      SENTENCE_DIRECT / SENTENCE_FILTERED:
        - Classification metrics       → ALL sentences
        - Pairwise mapping metrics     → GT constraint sentences only
        - Aggregate violation metrics  → ALL sentences (all predicted vs all GT constraints)
      DOCUMENT:
        - Aggregate violation metrics  → all predicted rules vs all GT constraints
        - All other metrics            → N/A (no sentence-level alignment available)
    """

    def __init__(self, config_file: str):
        super().__init__(config_file)
        if self.config["experiment_type"] != "mapper":
            raise ValueError(
                f"Expected experiment_type='mapper', "
                f"got '{self.config['experiment_type']}'"
            )
        self.prompt_loader = PromptLoader()

    def run(self):
        print("\n" + "=" * 80)
        print(f"FULL MAPPER EVALUATION: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("\nEvaluation scope:")
        print("  SENTENCE_DIRECT / FILTERED:")
        print("    Classification metrics     → ALL sentences")
        print(
            "    Aggregate violation        → ALL sentences (all predicted vs all GT)"
        )
        print("    Pairwise mapping metrics   → GT constraint sentences only")
        print("  DOCUMENT:")
        print(
            "    Aggregate violation        → all predicted rules vs all GT constraints"
        )
        print("    All other metrics          → N/A")
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
            mode = variant.get("mode", "SENTENCE_DIRECT")

            print(f"\n[{i}/{len(self.config['variants'])}] Variant: {name}")
            print(f"  Mode:           {mode}")
            print(
                f"  Implementation: {variant['mapper'].get('implementation', 'llm') if 'mapper' in variant else variant.get('document_mapper', {}).get('implementation', 'llm_document')}"
            )

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
            self._print_comparison_table(results)

        return failed_count

    # =========================================================================
    # Core variant router
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
        mode = variant.get("mode", "SENTENCE_DIRECT")

        if mode in ("SENTENCE_DIRECT", "SENTENCE_FILTERED"):
            return self._run_sentence_variant(
                name=name,
                variant=variant,
                mode=mode,
                all_sentences=all_sentences,
                gt_sentence2dcs_map=gt_sentence2dcs_map,
                predicate_space=predicate_space,
                df=df,
                dataset_name=dataset_name,
            )
        elif mode == "DOCUMENT":
            return self._run_document_variant(
                name=name,
                variant=variant,
                all_sentences=all_sentences,
                gt_sentence2dcs_map=gt_sentence2dcs_map,
                predicate_space=predicate_space,
                df=df,
                dataset_name=dataset_name,
            )
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Use SENTENCE_DIRECT, SENTENCE_FILTERED, or DOCUMENT."
            )

    # =========================================================================
    # SENTENCE_DIRECT and SENTENCE_FILTERED (identical post-pipeline metrics)
    # =========================================================================

    def _run_sentence_variant(
        self,
        name: str,
        variant: Dict,
        mode: str,
        all_sentences: List[str],
        gt_sentence2dcs_map: Dict,
        predicate_space: Dict,
        df,
        dataset_name: str,
    ) -> Dict:
        use_context = variant.get("use_context", False)
        num_rows_context = variant.get("num_rows_context", 3)
        implementation = variant["mapper"].get("implementation", "llm")

        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, num_rows=num_rows_context)

        # ------------------------------------------------------------------
        # Run pipeline on ALL sentences
        # ------------------------------------------------------------------
        print(f"  Processing all {len(all_sentences)} sentences (mode={mode})...")

        t0 = time.perf_counter()
        model_output_failures = _NA
        if implementation == "external":
            pipeline_results = self._run_external_tool(
                all_sentences, context, variant["mapper"]
            )
        else:
            pipeline = self._create_sentence_pipeline(
                mode, variant, predicate_space, df
            )
            pipeline_results = pipeline.process(all_sentences, context=context)
            processing_errors = getattr(pipeline.mapper, "processing_errors", [])
            if processing_errors:
                raise RuntimeError(
                    "Mapper did not complete all sentences; "
                    f"first processing error: {processing_errors[0]}"
                )
            model_output_failures = len(
                getattr(pipeline.mapper, "output_failures", [])
            ) + len(getattr(getattr(pipeline, "classifier", None), "output_failures", []))
            # Returns List[Tuple[sentence, is_rule, predicates]]
        runtime_s = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # 1. Classification evaluation — ALL sentences
        # ------------------------------------------------------------------
        print(f"  Evaluating classification (all {len(all_sentences)} sentences)...")
        clf_predictions = [
            (sent, preds is not None and len(preds) > 0)
            for sent, _is_rule, preds in pipeline_results
        ]
        clf_evaluator = ClassificationEvaluator(ground_truth=gt_sentence2dcs_map)
        clf_metrics, _ = clf_evaluator.evaluate(clf_predictions)

        # ------------------------------------------------------------------
        # 2. Pairwise mapping quality — GT sentences only
        # ------------------------------------------------------------------
        print(f"  Evaluating pairwise mapping quality (GT sentences only)...")

        gt_pipeline_results = [
            (sent, is_rule, preds)
            for sent, is_rule, preds in pipeline_results
            if sent in gt_sentence2dcs_map
        ]

        predictions_raw = [(sent, preds) for sent, _, preds in gt_pipeline_results]
        predictions = self._map_semantically_valid_constraint_predicates(
            predictions_raw, gt_sentence2dcs_map
        )

        for (sent, raw_pred), (_, mapped_pred) in zip(predictions_raw, predictions):
            if raw_pred != mapped_pred:
                print(
                    f"    Normalized:\n"
                    f"      Sentence:    {sent}\n"
                    f"      Raw Pred:    {raw_pred}\n"
                    f"      Mapped Pred: {mapped_pred}"
                )

        map_evaluator = MappingEvaluator(
            ground_truth=gt_sentence2dcs_map,
            df=df,
            csv_path=self._get_primary_csv_path(predicate_space),
        )
        map_metrics, sentence_results = map_evaluator.evaluate(predictions)

        # ------------------------------------------------------------------
        # 3. Aggregate violation — ALL sentences
        #    Pool constraints predicted from ALL sentences (not just GT),
        #    execute against the full table, compare to all gold violations.
        # ------------------------------------------------------------------
        print(
            f"  Evaluating aggregate violation (all {len(all_sentences)} sentences)..."
        )

        all_pred_violations: set = set()
        for sent, _, preds in pipeline_results:
            if preds is not None and len(preds) > 0:
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
        agg_vf = 2 * agg_vp * agg_vr / (agg_vp + agg_vr) if (agg_vp + agg_vr) else 0.0

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
                    "sentence": s,
                    "gold_is_constraint": s in gt_sentence2dcs_map,
                    "pred_is_constraint": preds is not None and len(preds) > 0,
                    "is_rule_flag": ir,
                    "predicted": str(preds) if preds else None,
                }
                for s, ir, preds in pipeline_results
            ],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="all_sentences_mapper",
        )

        self._save_detailed_sentence_results(
            sentence_results=sentence_results,
            all_pipeline_results=pipeline_results,
            gt_sentence2dcs_map=gt_sentence2dcs_map,
            dataset_name=dataset_name,
            variant_name=name,
        )

        # ------------------------------------------------------------------
        # Print metrics
        # ------------------------------------------------------------------
        print(f"\n  ── RUNTIME " + "─" * 59)
        print(f"     Pipeline runtime: {runtime_s:.2f}s")

        print(f"\n  ── CLASSIFICATION (all {len(all_sentences)} sentences) " + "─" * 30)
        print(f"     Accuracy:  {clf_metrics.accuracy:.3f}")
        print(f"     Precision: {clf_metrics.precision:.3f}")
        print(f"     Recall:    {clf_metrics.recall:.3f}")
        print(f"     F1:        {clf_metrics.f1_score:.3f}  [PRIMARY]")
        print(
            f"     TP: {clf_metrics.true_positives:3d}  FP: {clf_metrics.false_positives:3d}"
        )
        print(
            f"     FN: {clf_metrics.false_negatives:3d}  TN: {clf_metrics.true_negatives:3d}"
        )

        print(
            f"\n  ── AGGREGATE VIOLATION DISCOVERY (all {len(all_sentences)} sentences) "
            + "─" * 13
        )
        print(f"     Precision: {map_metrics.aggregate_violation_precision:.3f}")
        print(f"     Recall:    {map_metrics.aggregate_violation_recall:.3f}")
        print(f"     F1:        {map_metrics.aggregate_violation_f1:.3f}  [PRIMARY]")
        print(f"     Gold violations:      {map_metrics.total_gold_violations}")
        print(f"     Predicted violations: {map_metrics.total_pred_violations}")
        print(f"     Overlap violations:   {map_metrics.total_overlap_violations}")

        print(f"\n  ── CONSTRAINT EXACT MATCH (GT sentences only) " + "─" * 25)
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
            f"\n  ── PAIRWISE SYNTACTIC QUALITY (GT sentences only, n={map_metrics.pairwise_strict_total}) "
            + "─" * 10
        )
        print(
            f"     Exact Match:       {map_metrics.pairwise_strict_accuracy:.3f} ({map_metrics.pairwise_strict_correct}/{map_metrics.pairwise_strict_total})"
        )
        print(
            f"     Predicate Jaccard: {map_metrics.pairwise_avg_predicate_jaccard:.3f} (±{map_metrics.pairwise_std_predicate_jaccard:.3f})"
        )

        print(f"\n  ── PAIRWISE VIOLATION DISCOVERY (GT sentences only) " + "─" * 20)
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
        print(f"\n  INSIGHT: Syntactic vs Violation Gap (pairwise) = {gap:.3f}")
        if gap > 0:
            print(
                f"     {gap*100:.1f}% of constraints find correct violations despite syntactic differences"
            )
        elif gap < 0:
            print(
                f"     Warning: {abs(gap)*100:.1f}% of syntactically correct constraints fail to find correct violations"
            )

        return {
            "type": "sentence",
            "mode": mode,
            "implementation": variant["mapper"].get("implementation", "llm"),
            "clf_metrics": clf_metrics,
            "map_metrics": map_metrics,
            "runtime_s": runtime_s,
            "model_output_failures": model_output_failures,
        }

    # =========================================================================
    # DOCUMENT mode
    # =========================================================================

    def _run_document_variant(
        self,
        name: str,
        variant: Dict,
        all_sentences: List[str],
        gt_sentence2dcs_map: Dict,
        predicate_space: Dict,
        df,
        dataset_name: str,
    ) -> Dict:
        use_context = variant.get("use_context", False)
        num_rows_context = variant.get("num_rows_context", 3)

        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, num_rows=num_rows_context)

        # Document mapper needs predicates in context
        predicates_list_str = "\n".join(predicate_space["predicates"])
        context["predicates_list"] = predicates_list_str

        print(f"  Processing document ({len(all_sentences)} sentences in one call)...")
        pipeline = self._create_document_pipeline(variant, predicate_space, context)

        t0 = time.perf_counter()
        predicted_rules = pipeline.process(all_sentences, context=context)
        runtime_s = time.perf_counter() - t0
        # Returns List[List[str]] — a flat list of predicted constraint predicate lists

        # ------------------------------------------------------------------
        # Aggregate violation — no sentence alignment, just pool everything
        # ------------------------------------------------------------------
        print(f"  Evaluating aggregate violation (document mode)...")

        map_evaluator = MappingEvaluator(
            ground_truth=gt_sentence2dcs_map,
            df=df,
            csv_path=self._get_primary_csv_path(predicate_space),
        )

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
        # Save outputs
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

        print(f"\n  ── AGGREGATE VIOLATION DISCOVERY (document mode) " + "─" * 22)
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
            "type": "document",
            "mode": "DOCUMENT",
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
    # External tool runner
    # =========================================================================

    def _run_external_tool(
        self,
        sentences: List[str],
        context: Dict,
        mapper_config: Dict,
    ) -> List[Tuple[str, bool, Optional[List[Rule]]]]:
        results = []
        for i, sentence in enumerate(sentences, 1):
            if i % 10 == 0 or i == len(sentences):
                print(f"    External tool: {i}/{len(sentences)} sentences processed...")
            try:
                predicates = call_external_tool_single(sentence, context, mapper_config)
                is_rule = predicates is not None and len(predicates) > 0
                results.append((sentence, is_rule, predicates))
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"    Warning: external tool failed on sentence [{i}]: {e}")
                results.append((sentence, False, None))
        return results

    # =========================================================================
    # Pipeline factories
    # =========================================================================

    def _create_sentence_pipeline(
        self, mode: str, variant: Dict, predicate_space: Dict, df
    ):
        """Create SENTENCE_DIRECT or SENTENCE_FILTERED pipeline."""
        mapper_config = variant["mapper"]
        implementation = mapper_config.get("implementation", "llm")
        csv_path = self._get_primary_csv_path(predicate_space)

        if implementation == "multi_step_refinement":
            mapper = MultiStepRefinementMapper(
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
            )
        elif implementation == "llm":
            mapper = LLMMapper(
                model_name=mapper_config["model"],
                provider=mapper_config.get("provider"),
                prompt_name=mapper_config["prompt_key"],
                allowed_predicates=predicate_space["predicates"],
                chunk_size=mapper_config.get("batch_size", 10),
                filter_predicates=mapper_config.get("filter_predicates", False),
                predicate_tables=predicate_space["tables"],
                max_candidate_predicates=mapper_config.get("max_candidate_predicates", 1000),
                max_relevant_columns=mapper_config.get("max_relevant_columns", 12),
                max_tokens=mapper_config.get("max_tokens"),
                structured_output=mapper_config.get("structured_output", True),
            )
        else:
            raise ValueError(f"Unknown mapper implementation: {implementation}")

        if mode == "SENTENCE_FILTERED":
            classifier_config = variant["classifier"]
            classifier_prompt = self.prompt_loader.load(
                "classification", classifier_config["prompt_key"]
            )
            classifier = LLMRuleClassifier(
                model_name=classifier_config["model"],
                provider=classifier_config.get("provider"),
                prompt_template=classifier_prompt,
                chunk_size=classifier_config.get("batch_size", 10),
                structured_output=classifier_config.get("structured_output", True),
            )
            return NormaPipeline(
                mode=ProcessingMode.SENTENCE_FILTERED,
                classifier=classifier,
                mapper=mapper,
            )
        else:
            return NormaPipeline(mode=ProcessingMode.SENTENCE_DIRECT, mapper=mapper)

    def _create_document_pipeline(
        self, variant: Dict, predicate_space: Dict, context: Dict
    ):
        """Create DOCUMENT pipeline."""
        doc_mapper_config = variant["document_mapper"]
        document_mapper = LLMDocumentMapper(
            model_name=doc_mapper_config["model"],
            provider=doc_mapper_config.get("provider"),
            prompt_name=doc_mapper_config["prompt_key"],
        )
        return NormaPipeline(
            mode=ProcessingMode.DOCUMENT, document_mapper=document_mapper
        )

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
    # Semantic normalization
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
    # Detailed CSV output (sentence modes only)
    # =========================================================================

    def _save_detailed_sentence_results(
        self,
        sentence_results,
        all_pipeline_results,
        gt_sentence2dcs_map: Dict,
        dataset_name: str,
        variant_name: str,
    ):
        import pandas as pd

        gt_by_sentence = {r.sentence: r for r in sentence_results}
        records = []

        for sent, _is_rule, preds in all_pipeline_results:
            gold_is_constraint = sent in gt_sentence2dcs_map
            pred_is_constraint = preds is not None and len(preds) > 0

            record = {
                "sentence": sent,
                "gold_is_constraint": gold_is_constraint,
                "pred_is_constraint": pred_is_constraint,
                "gold_constraint": (
                    str(gt_sentence2dcs_map[sent]) if gold_is_constraint else None
                ),
                "pred_constraint": str(preds) if preds else None,
            }

            r = gt_by_sentence.get(sent)
            if r is not None and gold_is_constraint:
                record.update(
                    {
                        "pairwise_syntactic_exact_match": r.strict_correct,
                        "pairwise_syntactic_predicate_jaccard": r.predicate_jaccard,
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
        filename = (
            f"{dataset_name}_{variant_name.replace(' ', '_')}_detailed_results.csv"
        )
        filepath = self.metrics_dir / filename
        df_out.to_csv(filepath, index=False)
        print(f"  Saved detailed results to: {filepath}")

    # =========================================================================
    # Result / failure dicts
    # =========================================================================

    def _build_result_dict(
        self,
        dataset_name: str,
        variant: Dict,
        result: Dict,
        num_total: int,
        num_gt: int,
    ) -> Dict:
        mode = result["mode"]
        base = {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": (
                variant.get("mapper", variant.get("document_mapper", {})).get(
                    "implementation", _NA
                )
            ),
            "mode": mode,
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "status": "success",
            "runtime_s": result.get("runtime_s", _NA),
            "model_output_failures": result.get("model_output_failures", _NA),
        }

        if result["type"] == "document":
            # Only aggregate violation is available
            base.update(
                {
                    # Classification — N/A
                    "clf_accuracy": _NA,
                    "clf_precision": _NA,
                    "clf_recall": _NA,
                    "clf_f1": _NA,
                    "clf_tp": _NA,
                    "clf_fp": _NA,
                    "clf_fn": _NA,
                    "clf_tn": _NA,
                    # Aggregate violation — available
                    "aggregate_violation_precision": result[
                        "aggregate_violation_precision"
                    ],
                    "aggregate_violation_recall": result["aggregate_violation_recall"],
                    "aggregate_violation_f1": result["aggregate_violation_f1"],
                    "total_gold_violations": result["total_gold_violations"],
                    "total_pred_violations": result["total_pred_violations"],
                    "total_overlap_violations": result["total_overlap_violations"],
                    "num_predicted_rules": result["num_predicted_rules"],
                    # Constraint exact match — N/A
                    "constraint_exact_precision": _NA,
                    "constraint_exact_recall": _NA,
                    "constraint_exact_f1": _NA,
                    "constraint_exact_tp": _NA,
                    "constraint_exact_fp": _NA,
                    "constraint_exact_fn": _NA,
                    "num_gold_constraints": _NA,
                    "num_pred_constraints": _NA,
                    # Pairwise syntactic — N/A
                    "pairwise_syntactic_exact_match": _NA,
                    "pairwise_syntactic_exact_match_count": _NA,
                    "pairwise_syntactic_total": _NA,
                    "pairwise_syntactic_predicate_jaccard": _NA,
                    "pairwise_syntactic_predicate_jaccard_std": _NA,
                    # Pairwise violation — N/A
                    "pairwise_violation_exact_match": _NA,
                    "pairwise_violation_exact_match_count": _NA,
                    "pairwise_violation_total": _NA,
                    "pairwise_violation_precision": _NA,
                    "pairwise_violation_precision_std": _NA,
                    "pairwise_violation_recall": _NA,
                    "pairwise_violation_recall_std": _NA,
                    "pairwise_violation_f1": _NA,
                    "pairwise_violation_f1_std": _NA,
                    "pairwise_violation_jaccard": _NA,
                    "pairwise_violation_jaccard_std": _NA,
                }
            )
        else:
            # Sentence mode (DIRECT or FILTERED) — all metrics available
            clf = result["clf_metrics"]
            m = result["map_metrics"]
            base.update(
                {
                    # Classification
                    "clf_accuracy": clf.accuracy,
                    "clf_precision": clf.precision,
                    "clf_recall": clf.recall,
                    "clf_f1": clf.f1_score,
                    "clf_tp": clf.true_positives,
                    "clf_fp": clf.false_positives,
                    "clf_fn": clf.false_negatives,
                    "clf_tn": clf.true_negatives,
                    # Aggregate violation
                    "aggregate_violation_precision": m.aggregate_violation_precision,
                    "aggregate_violation_recall": m.aggregate_violation_recall,
                    "aggregate_violation_f1": m.aggregate_violation_f1,
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
                    "pairwise_syntactic_exact_match": m.pairwise_strict_accuracy,
                    "pairwise_syntactic_exact_match_count": m.pairwise_strict_correct,
                    "pairwise_syntactic_total": m.pairwise_strict_total,
                    "pairwise_syntactic_predicate_jaccard": m.pairwise_avg_predicate_jaccard,
                    "pairwise_syntactic_predicate_jaccard_std": m.pairwise_std_predicate_jaccard,
                    # Pairwise violation
                    "pairwise_violation_exact_match": m.pairwise_execution_accuracy,
                    "pairwise_violation_exact_match_count": m.pairwise_execution_correct,
                    "pairwise_violation_total": m.pairwise_execution_total,
                    "pairwise_violation_precision": m.pairwise_avg_execution_precision,
                    "pairwise_violation_precision_std": m.pairwise_std_execution_precision,
                    "pairwise_violation_recall": m.pairwise_avg_execution_recall,
                    "pairwise_violation_recall_std": m.pairwise_std_execution_recall,
                    "pairwise_violation_f1": m.pairwise_avg_execution_f1,
                    "pairwise_violation_f1_std": m.pairwise_std_execution_f1,
                    "pairwise_violation_jaccard": m.pairwise_avg_execution_jaccard,
                    "pairwise_violation_jaccard_std": m.pairwise_std_execution_jaccard,
                }
            )

        return base

    def _build_failure_dict(
        self, dataset_name: str, variant: Dict, error: str, num_total: int, num_gt: int
    ) -> Dict:
        return {
            "dataset": dataset_name,
            "variant_name": variant["name"],
            "implementation": (
                variant.get("mapper", variant.get("document_mapper", {})).get(
                    "implementation", _NA
                )
            ),
            "mode": variant.get("mode", "SENTENCE_DIRECT"),
            "num_sentences_total": num_total,
            "num_sentences_gt": num_gt,
            "status": "failed",
            "error": error,
            "runtime_s": _NA,
            "clf_f1": _NA,
            "aggregate_violation_f1": _NA,
            "constraint_exact_f1": _NA,
            "pairwise_syntactic_exact_match": _NA,
            "pairwise_violation_f1": _NA,
        }

    # =========================================================================
    # Comparison table
    # =========================================================================

    def _print_comparison_table(self, results: Dict):
        sentence_results = {k: v for k, v in results.items() if v["type"] == "sentence"}
        document_results = {k: v for k, v in results.items() if v["type"] == "document"}

        print("\n" + "=" * 120)
        print("COMPARISON TABLE")
        print("=" * 120)

        # ---- Sentence modes (DIRECT + FILTERED) ----
        if sentence_results:
            print("\nSentence Modes (DIRECT / FILTERED):")
            print("  Left  (all sentences):      Clf F1 | Agg Viol F1")
            print("  Right (GT sentences only):  Const F1 | Syn Exact | Viol F1")
            hdr = (
                f"  {'Variant':<38} {'Mode':<12} {'Impl':<10} "
                f"{'[ALL] Clf F1':<14} {'[ALL] AggViol F1':<18} "
                f"{'[GT] Const F1':<15} {'[GT] Syn Exact':<16} {'[GT] Viol F1':<12} {'Runtime(s)':<10}"
            )
            print(hdr)
            print("  " + "-" * 128)

            for name, result in sentence_results.items():
                clf = result["clf_metrics"]
                m = result["map_metrics"]
                print(
                    f"  {name:<38} {result['mode']:<12} "
                    f"{result.get('implementation', ''):<10} "
                    f"{clf.f1_score:<14.3f} {m.aggregate_violation_f1:<18.3f} "
                    f"{m.constraint_exact_f1:<15.3f} {m.pairwise_strict_accuracy:<16.3f} "
                    f"{m.pairwise_avg_execution_f1:<12.3f} {result.get('runtime_s', _NA)!s:<10}"
                )

            print()
            best_clf = max(
                sentence_results.items(), key=lambda x: x[1]["clf_metrics"].f1_score
            )
            best_agg = max(
                sentence_results.items(),
                key=lambda x: x[1]["map_metrics"].aggregate_violation_f1,
            )
            best_const = max(
                sentence_results.items(),
                key=lambda x: x[1]["map_metrics"].constraint_exact_f1,
            )
            best_viol_pw = max(
                sentence_results.items(),
                key=lambda x: x[1]["map_metrics"].pairwise_avg_execution_f1,
            )
            fastest = min(
                (
                    (k, v)
                    for k, v in sentence_results.items()
                    if isinstance(v.get("runtime_s"), float)
                ),
                key=lambda x: x[1]["runtime_s"],
                default=None,
            )

            print(
                f"  Best Classification F1 [ALL]:        {best_clf[0][:50]} ({best_clf[1]['clf_metrics'].f1_score:.3f})"
            )
            print(
                f"  Best Aggregate Violation F1 [ALL]:   {best_agg[0][:50]} ({best_agg[1]['map_metrics'].aggregate_violation_f1:.3f})"
            )
            print(
                f"  Best Constraint Exact F1 [GT]:       {best_const[0][:50]} ({best_const[1]['map_metrics'].constraint_exact_f1:.3f})"
            )
            print(
                f"  Best Violation F1 Pairwise [GT]:     {best_viol_pw[0][:50]} ({best_viol_pw[1]['map_metrics'].pairwise_avg_execution_f1:.3f})"
            )
            if fastest:
                print(
                    f"  Fastest Runtime:                     {fastest[0][:50]} ({fastest[1]['runtime_s']:.2f}s)"
                )

        # ---- Document mode ----
        if document_results:
            print("\nDocument Mode (DOCUMENT):")
            print(
                "  Note: Only aggregate violation is computable — all other metrics are N/A."
            )
            hdr = (
                f"  {'Variant':<38} {'Impl':<10} "
                f"{'[DOC] AggViol P':<16} {'[DOC] AggViol R':<16} {'[DOC] AggViol F1':<18} "
                f"{'Gold Viol':<10} {'Pred Viol':<10} {'Overlap':<10} {'Pred Rules':<10} {'Runtime(s)':<10}"
            )
            print(hdr)
            print("  " + "-" * 128)

            variant_by_name = {v["name"]: v for v in self.config["variants"]}
            for name, result in document_results.items():
                impl = (
                    variant_by_name.get(name, {})
                    .get("document_mapper", {})
                    .get("implementation", _NA)
                )
                print(
                    f"  {name:<38} "
                    f"{impl:<10} "
                    f"{result['aggregate_violation_precision']:<16.3f} "
                    f"{result['aggregate_violation_recall']:<16.3f} "
                    f"{result['aggregate_violation_f1']:<18.3f} "
                    f"{result['total_gold_violations']:<10} "
                    f"{result['total_pred_violations']:<10} "
                    f"{result['total_overlap_violations']:<10} "
                    f"{result['num_predicted_rules']:<10} "
                    f"{result.get('runtime_s', _NA)!s:<10}"
                )

            if len(document_results) > 1:
                best_doc = max(
                    document_results.items(),
                    key=lambda x: x[1]["aggregate_violation_f1"],
                )
                fastest_doc = min(
                    (
                        (k, v)
                        for k, v in document_results.items()
                        if isinstance(v.get("runtime_s"), float)
                    ),
                    key=lambda x: x[1]["runtime_s"],
                    default=None,
                )
                print(
                    f"\n  Best Aggregate Violation F1 [DOC]: {best_doc[0][:50]} ({best_doc[1]['aggregate_violation_f1']:.3f})"
                )
                if fastest_doc:
                    print(
                        f"  Fastest Runtime [DOC]:             {fastest_doc[0][:50]} ({fastest_doc[1]['runtime_s']:.2f}s)"
                    )

        print("=" * 120)


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Full mapper evaluation: classification + mapping quality + violations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mapper_eval_full.py configs/mappers/sentence_level.yaml
  python run_mapper_eval_full.py configs/mappers/document_level.yaml
        """,
    )
    parser.add_argument("config_file", help="Path to YAML config file")
    args = parser.parse_args()

    try:
        experiment = MapperEvalFullExperiment(args.config_file)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting full mapper evaluation experiment...")
    main()
