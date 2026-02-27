#!/usr/bin/env python3
"""
Ground Truth Sentence Mapper Experiment Runner

Tests mapper on ONLY sentences with ground truth constraints.
Supports only SENTENCE_DIRECT mode to evaluate the mapper's 
ability to correctly translate constraint sentences.

Usage:
    python run_sentence_mappers_gt.py configs/mappers/sentence_level.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment
from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.evaluation.mapping import MappingEvaluator


class MapperExperiment(BaseExperiment):
    """Run mapper experiments on ground truth constraint sentences only."""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        # Validate experiment type
        if self.config['experiment_type'] != 'mapper':
            raise ValueError(
                f"Expected experiment_type='mapper', "
                f"got '{self.config['experiment_type']}'"
            )
    
    def run(self):
        """Run mapper experiments across all datasets and variants."""
        print("\n" + "=" * 80)
        print(f"GROUND TRUTH MAPPER EXPERIMENT: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("\nMode: SENTENCE_DIRECT (Ground Truth Constraints Only)")
        print("  Testing mapper on sentences with known constraints")
        print("  Evaluates translation accuracy of mapper")
        print("=" * 80 + "\n")
        
        # Save configs
        self.save_config_yaml()
        self.save_execution_config()
        
        # Track progress
        total_combinations = len(self.config['datasets']) * len(self.config['variants'])
        completed = 0
        failed = 0
        
        # Run for each dataset
        for dataset_name in self.config['datasets']:
            try:
                dataset_failed = self.run_dataset(dataset_name)
                failed += dataset_failed
                completed += len(self.config['variants']) - dataset_failed
            except Exception as e:
                print(f"\nError processing dataset {dataset_name}: {e}")
                traceback.print_exc()
                failed += len(self.config['variants'])
        
        # Final summary
        print(f"\n{'=' * 80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nProgress: {completed}/{total_combinations} variants completed")
        if failed > 0:
            print(f"Failed: {failed}/{total_combinations} variants")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"\nKey files:")
        print(f"  {self.results_summary_path}")
        print(f"  {self.outputs_dir}/")
        print(f"  {self.metrics_dir}/")
    
    def run_dataset(self, dataset_name: str) -> int:
        """Run all variants on a single dataset."""
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        try:    
            df, sentences, sentence2dcs_map, predicate_space = self.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return len(self.config['variants'])
        
        # Filter to only sentences with ground truth constraints
        gt_sentences = [sent for sent in sentences if sent in sentence2dcs_map and sentence2dcs_map[sent]]
        gt_sentence2dcs_map = {sent: dcs for sent, dcs in sentence2dcs_map.items() if dcs}
        
        num_total = len(sentences)
        num_gt_constraints = len(gt_sentences)
        
        print(f"Total sentences: {num_total}")
        print(f"Sentences with GT constraints: {num_gt_constraints}")
        print(f"Testing on GT constraint sentences only\n")
        
        # Run each variant
        results = {}
        failed_count = 0
        
        for i, variant in enumerate(self.config['variants'], 1):
            name = variant['name']
            mode = variant.get('mode', 'SENTENCE_DIRECT')
            
            # Validate mode
            if mode != 'SENTENCE_DIRECT':
                print(f"\n[{i}/{len(self.config['variants'])}] Skipping variant: {name}")
                print(f"  Error: Mode '{mode}' not supported. Only SENTENCE_DIRECT is supported.")
                failed_count += 1
                self.append_result(self._build_failure_dict(
                    dataset_name, variant, f"Mode '{mode}' not supported", num_gt_constraints
                ))
                continue
            
            print(f"\n[{i}/{len(self.config['variants'])}] Processing variant: {name}")
            print(f"  Mode: {mode}")
            
            try:
                result = self.run_variant(
                    name=name,
                    variant=variant,
                    sentences=gt_sentences,
                    sentence2dcs_map=gt_sentence2dcs_map,
                    predicate_space=predicate_space,
                    df=df,
                    dataset_name=dataset_name
                )
                results[name] = result
                
                # Track result
                result_dict = self._build_result_dict(
                    dataset_name=dataset_name,
                    variant=variant,
                    result=result,
                    num_sentences=num_gt_constraints,
                    num_rules=num_gt_constraints
                )
                self.append_result(result_dict)
                
                print(f"  Variant completed")
                
            except Exception as e:
                print(f"  Error: Variant failed: {e}")
                traceback.print_exc()
                failed_count += 1
                
                # Track failure
                self.append_result(self._build_failure_dict(
                    dataset_name, variant, str(e), num_gt_constraints
                ))
        
        # Print comparison table
        if results:
            self._print_comparison_table(results)
        
        return failed_count
    

    def map_semmantically_valid_constaint_predicates()
    
    def run_variant(self, name: str, variant: Dict, sentences: List[str],
                   sentence2dcs_map: Dict, predicate_space: Dict, df, dataset_name: str) -> Dict:
        """Run a single variant on ground truth constraint sentences."""
        mode = variant.get('mode', 'SENTENCE_DIRECT')
        use_context = variant.get('use_context', False)
        num_rows_context = variant.get('num_rows_context', 3)
        
        # Only SENTENCE_DIRECT mode is supported
        if mode != 'SENTENCE_DIRECT':
            raise ValueError(f"Only SENTENCE_DIRECT mode is supported, got: {mode}")
        
        # Build context
        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, dataset_name, num_rows=num_rows_context)
        
        # Create pipeline
        pipeline = self._create_sentence_direct_pipeline(variant, predicate_space, df)
        
        # Process (only GT constraint sentences)
        print(f"  Processing {len(sentences)} GT constraint sentences...")
        results = pipeline.process(sentences, context=context)
        
        # Evaluate
        print(f"  Evaluating...")
        predictions = [(sent, preds) for sent, is_rule, preds in results]

        predictions = map_semantically_valid_constaint_predicates(
        
        evaluator = MappingEvaluator(ground_truth=sentence2dcs_map, df=df)
        metrics, sentence_results = evaluator.evaluate(predictions)
        evaluator.close()
        
        detailed_metrics = self._prepare_sentence_metrics(metrics, sentence_results)
        
        # Save outputs
        print(f"  Saving outputs...")
        outputs_file = self.save_outputs(
            outputs=[{
                'sentence': s,
                'is_rule': ir,
                'predicted': str(p) if p else None
            } for s, ir, p in results],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="sentence_mappings_gt"
        )
        
        # Display detailed metrics
        print(f"\n  CONSTRAINT EXACT MATCH (Set membership):")
        print(f"     Precision: {metrics.constraint_exact_precision:.3f}")
        print(f"     Recall:    {metrics.constraint_exact_recall:.3f}")
        print(f"     F1:        {metrics.constraint_exact_f1:.3f} [PRIMARY]")
        print(f"     TP: {metrics.constraint_exact_tp}, FP: {metrics.constraint_exact_fp}, FN: {metrics.constraint_exact_fn}")
        print(f"     Predicted: {metrics.num_pred_constraints}, Gold: {metrics.num_gold_constraints}")
        
        print(f"\n  PAIRWISE SYNTACTIC QUALITY (Known sentence->constraint pairing, n={metrics.pairwise_strict_total}):")
        print(f"     Exact Match:       {metrics.pairwise_strict_accuracy:.3f} ({metrics.pairwise_strict_correct}/{metrics.pairwise_strict_total})")
        print(f"     Predicate Jaccard: {metrics.pairwise_avg_predicate_jaccard:.3f} (±{metrics.pairwise_std_predicate_jaccard:.3f})")
        
        print(f"\n  PAIRWISE VIOLATION DISCOVERY (Known pairing):")
        print(f"     Exact Match:       {metrics.pairwise_execution_accuracy:.3f} ({metrics.pairwise_execution_correct}/{metrics.pairwise_execution_total})")
        print(f"     Violation Jaccard: {metrics.pairwise_avg_execution_jaccard:.3f} (±{metrics.pairwise_std_execution_jaccard:.3f})")
        
        print(f"\n  AGGREGATE VIOLATION DISCOVERY (All constraints pooled):")
        print(f"     Violation Precision:    {metrics.aggregate_violation_precision:.3f}")
        print(f"     Violation Recall:       {metrics.aggregate_violation_recall:.3f}")
        print(f"     Violation F1:           {metrics.aggregate_violation_f1:.3f} [PRIMARY]")
        print(f"     Gold violations:        {metrics.total_gold_violations}")
        print(f"     Predicted violations:   {metrics.total_pred_violations}")
        print(f"     Overlap violations:     {metrics.total_overlap_violations}")
        
        # Show syntactic vs violation-based gap
        gap = metrics.pairwise_execution_accuracy - metrics.pairwise_strict_accuracy
        print(f"\n  INSIGHT: Syntactic vs Violation Gap (pairwise) = {gap:.3f}")
        if gap > 0:
            print(f"     {gap*100:.1f}% of constraints find correct violations despite syntactic differences")
        elif gap < 0:
            print(f"     Warning: {abs(gap)*100:.1f}% of syntactically correct constraints fail to find correct violations")
        
        # Save detailed per-sentence results
        self._save_detailed_sentence_results(sentence_results, dataset_name, name)
        
        return {
            'type': 'sentence',
            'mode': mode,
            'metrics': metrics
        }
    
    def _create_sentence_direct_pipeline(self, variant: Dict, predicate_space: Dict, df):
        """Create SENTENCE_DIRECT pipeline."""
        mapper_config = variant['mapper']
        implementation = mapper_config.get('implementation', 'llm')
        
        if implementation == 'multi_step_refinement':
            # MultiStepRefinementMapper with built-in prompts
            mapper = MultiStepRefinementMapper(
                model_name=mapper_config['model'],
                predicate_space=predicate_space,
                df=df,
                max_refinement_iterations=mapper_config.get('max_refinement_iterations', 2),
                prompt_file=mapper_config.get('prompt_file', 'multi_step_refinement')
            )
        elif implementation == 'llm':
            # Standard LLMMapper
            mapper = LLMMapper(
                model_name=mapper_config['model'],
                prompt_name=mapper_config['prompt_key'],
                allowed_predicates=predicate_space['predicates'],
                chunk_size=mapper_config.get('batch_size', 10)
            )
        else:
            raise ValueError(f"Unknown mapper implementation: {implementation}")
        
        return NormaPipeline(mode=ProcessingMode.SENTENCE_DIRECT, mapper=mapper)
    
    def _build_context(self, df, dataset_name: str, num_rows: int = 3):
        """Build table context."""
        from src.utils.table_context import TableContextBuilder
        context_builder = TableContextBuilder(sample_strategy="head", num_rows=num_rows)
        return context_builder.build(df)
    
    def _save_detailed_sentence_results(self, sentence_results, dataset_name: str, variant_name: str):
        """Save detailed per-sentence evaluation results."""
        import pandas as pd
        
        # Convert to records for DataFrame
        records = []
        for r in sentence_results:
            record = {
                'sentence': r.sentence,
                'gold_is_constraint': r.gold_is_rule,
                'pred_is_constraint': r.pred_is_rule,
                'gold_constraint': str(r.gold_rule) if r.gold_rule else None,
                'pred_constraint': str(r.pred_rule) if r.pred_rule else None,
            }
            
            # Add pairwise syntactic metrics (only for gold constraints)
            if r.gold_is_rule:
                record.update({
                    'pairwise_syntactic_exact_match': r.strict_correct,
                    'pairwise_syntactic_predicate_jaccard': r.predicate_jaccard,
                })
            
            # Add pairwise violation discovery metrics (only for gold constraints)
            if r.execution_exact is not None:
                record.update({
                    'pairwise_violation_exact_match': r.execution_exact,
                    'pairwise_violation_jaccard': r.execution_jaccard,
                    'num_gold_violations': len(r.gold_violations) if r.gold_violations else 0,
                    'num_pred_violations': len(r.pred_violations) if r.pred_violations else 0,
                    'num_overlap_violations': len(r.gold_violations & r.pred_violations) if r.gold_violations and r.pred_violations else 0,
                })
            
            records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(records)
        filename = f"{dataset_name}_{variant_name.replace(' ', '_')}_detailed_results.csv"
        filepath = self.metrics_dir / filename
        df.to_csv(filepath, index=False)
        print(f"  Saved detailed results to: {filepath}")
    
    def _prepare_sentence_metrics(self, metrics, sentence_results):
        """Prepare detailed metrics for sentence-level."""
        return {
            'evaluation_type': 'sentence',
            'summary_metrics': {
                'constraint_exact_f1': metrics.constraint_exact_f1,
                'pairwise_syntactic_exact': metrics.pairwise_strict_accuracy,
                'pairwise_violation_exact': metrics.pairwise_execution_accuracy,
                'pairwise_violation_jaccard': metrics.pairwise_avg_execution_jaccard,
                'aggregate_violation_f1': metrics.aggregate_violation_f1
            }
        }
    
    def _prepare_document_metrics(self, metrics, predicted_rules, gold_rules):
        """Prepare detailed metrics for document-level."""
        return {
            'evaluation_type': 'document',
            'summary_metrics': {
                'constraint_exact_f1': metrics['constraint_exact_f1'],
                'aggregate_violation_f1': metrics['aggregate_violation_f1']
            },
            'num_predicted': len(predicted_rules),
            'num_gold': len(gold_rules)
        }
    
    def _build_result_dict(self, dataset_name: str, variant: Dict,
                          result: Dict, num_sentences: int, num_rules: int) -> Dict:
        """Build result dictionary for sentence-level GT evaluation."""
        mode = variant.get('mode', 'SENTENCE_DIRECT')
        
        base_dict = {
            'dataset': dataset_name,
            'variant_name': variant['name'],
            'mode': mode,
            'num_sentences': num_sentences,
            'status': 'success'
        }
        
        # Always sentence-level metrics (only GT constraints)
        metrics = result['metrics']
        base_dict.update({
            # Constraint exact match
            'constraint_exact_precision': metrics.constraint_exact_precision,
            'constraint_exact_recall': metrics.constraint_exact_recall,
            'constraint_exact_f1': metrics.constraint_exact_f1,
            'constraint_exact_tp': metrics.constraint_exact_tp,
            'constraint_exact_fp': metrics.constraint_exact_fp,
            'constraint_exact_fn': metrics.constraint_exact_fn,
            'num_gold_constraints': metrics.num_gold_constraints,
            'num_pred_constraints': metrics.num_pred_constraints,
            # Pairwise syntactic quality
            'pairwise_syntactic_exact_match': metrics.pairwise_strict_accuracy,
            'pairwise_syntactic_exact_match_count': metrics.pairwise_strict_correct,
            'pairwise_syntactic_total': metrics.pairwise_strict_total,
            'pairwise_syntactic_predicate_jaccard': metrics.pairwise_avg_predicate_jaccard,
            'pairwise_syntactic_predicate_jaccard_std': metrics.pairwise_std_predicate_jaccard,
            # Pairwise violation discovery
            'pairwise_violation_exact_match': metrics.pairwise_execution_accuracy,
            'pairwise_violation_exact_match_count': metrics.pairwise_execution_correct,
            'pairwise_violation_total': metrics.pairwise_execution_total,
            'pairwise_violation_jaccard': metrics.pairwise_avg_execution_jaccard,
            'pairwise_violation_jaccard_std': metrics.pairwise_std_execution_jaccard,
            # Aggregate violation discovery
            'aggregate_violation_precision': metrics.aggregate_violation_precision,
            'aggregate_violation_recall': metrics.aggregate_violation_recall,
            'aggregate_violation_f1': metrics.aggregate_violation_f1,
            'total_gold_violations': metrics.total_gold_violations,
            'total_pred_violations': metrics.total_pred_violations,
            'total_overlap_violations': metrics.total_overlap_violations,
        })
        
        return base_dict
    
    def _build_failure_dict(self, dataset_name: str, variant: Dict,
                           error: str, num_sentences: int) -> Dict:
        """Build failure result dictionary."""
        return {
            'dataset': dataset_name,
            'variant_name': variant['name'],
            'mode': variant.get('mode', 'SENTENCE_DIRECT'),
            'num_sentences': num_sentences,
            'status': 'failed',
            'error': error,
            'constraint_exact_f1': 'N/A',
            'pairwise_syntactic_exact': 'N/A',
            'pairwise_violation_exact': 'N/A',
            'aggregate_violation_f1': 'N/A'
        }
    
    def _print_comparison_table(self, results: Dict):
        """Print comparison table for GT constraint sentence results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE (Ground Truth Constraints Only)")
        print("=" * 80)
        
        print("\nResults (SENTENCE_DIRECT on GT Constraints):")
        print(f"{'Variant':<40} {'Const':<8} {'Syntax':<8} {'Viol/Con':<8} {'Viol Jac':<8} {'Viol Agg':<8}")
        print(f"{'':40} {'Exact F1':<8} {'Exact':<8} {'Exact':<8} {'(pair)':<8} {'F1':<8}")
        print("-" * 90)
        
        for name, result in results.items():
            m = result['metrics']
            print(f"{name:<40} "
                  f"{m.constraint_exact_f1:<8.3f} {m.pairwise_strict_accuracy:<8.3f} "
                  f"{m.pairwise_execution_accuracy:<8.3f} {m.pairwise_avg_execution_jaccard:<8.3f} {m.aggregate_violation_f1:<8.3f}")
        
        print(f"\n{'Key Metrics:':<40}")
        best_const = max(results.items(), key=lambda x: x[1]['metrics'].constraint_exact_f1)
        print(f"  Best Constraint Exact Match F1:    {best_const[0][:50]} ({best_const[1]['metrics'].constraint_exact_f1:.3f})")
        
        best_viol_pair = max(results.items(), key=lambda x: x[1]['metrics'].pairwise_execution_accuracy)
        print(f"  Best Violation Exact (pairwise):   {best_viol_pair[0][:50]} ({best_viol_pair[1]['metrics'].pairwise_execution_accuracy:.3f})")
        
        best_viol_agg = max(results.items(), key=lambda x: x[1]['metrics'].aggregate_violation_f1)
        print(f"  Best Violation F1 (aggregate):     {best_viol_agg[0][:50]} ({best_viol_agg[1]['metrics'].aggregate_violation_f1:.3f})")
        
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run GT Sentence Mapper Experiments (SENTENCE_DIRECT on GT constraints only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sentence_mappers_gt.py configs/mappers/sentence_level.yaml
        """
    )
    parser.add_argument('config_file', help='Path to YAML config file')
    
    args = parser.parse_args()
    
    try:
        experiment = MapperExperiment(args.config_file)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
