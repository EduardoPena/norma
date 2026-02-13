#!/usr/bin/env python3
"""
Mapper Experiment Runner

Supports three mapping modes:
1. SENTENCE_DIRECT: Map all sentences directly (no classifier)
2. SENTENCE_FILTERED: Classify first, then map only rule sentences
3. DOCUMENT: Process entire document in one call

Usage:
    python run_mappers.py configs/mappers/sentence_level.yaml
    python run_mappers.py configs/mappers/document_level.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment
from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
from src.classifiers.llm import LLMRuleClassifier
from src.mappers.llm import LLMMapper
from src.mappers.llm_document_mapper import LLMDocumentMapper
from src.evaluation.mapping import MappingEvaluator
from src.llm.helper import PromptLoader


class MapperExperiment(BaseExperiment):
    """Run mapper experiments in different modes."""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        # Validate experiment type
        if self.config['experiment_type'] != 'mapper':
            raise ValueError(
                f"Expected experiment_type='mapper', "
                f"got '{self.config['experiment_type']}'"
            )
        
        # Initialize prompt loader
        self.prompt_loader = PromptLoader()
    
    def run(self):
        """Run mapper experiments across all datasets and variants."""
        print("\n" + "=" * 80)
        print(f"MAPPER EXPERIMENT: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("\nSupported modes:")
        print("  SENTENCE_DIRECT:   Map all sentences (no classifier)")
        print("  SENTENCE_FILTERED: Classify first, map only rules")
        print("  DOCUMENT:          Process entire document together")
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
        
        
        
        
        num_total = len(sentences)
        num_rules = sum(1 for dc in sentence2dcs_map.values() if dc)
        
        print(f"Total sentences: {num_total}")
        print(f"Rule sentences: {num_rules}\n")
        
        # Run each variant
        results = {}
        failed_count = 0
        
        for i, variant in enumerate(self.config['variants'], 1):
            name = variant['name']
            mode = variant.get('mode', 'SENTENCE_DIRECT')
            
            print(f"\n[{i}/{len(self.config['variants'])}] Processing variant: {name}")
            print(f"  Mode: {mode}")
            
            try:
                result = self.run_variant(
                    name=name,
                    variant=variant,
                    sentences=sentences,
                    sentence2dcs_map=sentence2dcs_map,
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
                    num_sentences=num_total,
                    num_rules=num_rules
                )
                self.append_result(result_dict)
                
                print(f"  Variant completed")
                
            except Exception as e:
                print(f"  Error: Variant failed: {e}")
                traceback.print_exc()
                failed_count += 1
                
                # Track failure
                self.append_result(self._build_failure_dict(
                    dataset_name, variant, str(e), num_total
                ))
        
        # Print comparison table
        if results:
            self._print_comparison_table(results)
        
        return failed_count
    
    def run_variant(self, name: str, variant: Dict, sentences: List[str],
                   sentence2dcs_map: Dict, predicate_space: Dict, df, dataset_name: str) -> Dict:
        """Run a single variant."""
        mode = variant.get('mode', 'SENTENCE_DIRECT')
        use_context = variant.get('use_context', False)
        num_rows_context = variant.get('num_rows_context', 3)
        
        # Build context
        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, dataset_name, num_rows=num_rows_context)
        
        # Create pipeline based on mode
        if mode == 'SENTENCE_DIRECT':
            pipeline = self._create_sentence_direct_pipeline(variant, predicate_space)
        elif mode == 'SENTENCE_FILTERED':
            pipeline = self._create_sentence_filtered_pipeline(variant, predicate_space)
        elif mode == 'DOCUMENT':
            pipeline = self._create_document_pipeline(variant, predicate_space, context)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Process
        print(f"  Processing {len(sentences)} sentences...")
        results = pipeline.process(sentences, context=context)
        
        # Evaluate
        print(f"  Evaluating...")
        if mode == 'DOCUMENT':
            # Document mode: extract gold rules from sentence2dcs_map
            gold_rules = [rules for rules in sentence2dcs_map.values() if rules]
            
            evaluator = MappingEvaluator(ground_truth=sentence2dcs_map, df=df)
            metrics = evaluator.evaluate_document_level(
                predicted_rules=results,
                gold_rules=gold_rules
            )
            evaluator.close()
            
            detailed_metrics = self._prepare_document_metrics(metrics, results, gold_rules)
            
            # Save outputs
            print(f"  Saving outputs...")
            outputs_file = self.save_outputs(
                outputs=[{'rule': str(rule)} for rule in results],
                dataset_name=dataset_name,
                variant_name=name,
                output_type="document_mappings"
            )
            
            # Display metrics
            print(f"\n  CONSTRAINT DISCOVERY (How many constraints were found):")
            print(f"     Constraint Precision: {metrics['constraint_precision']:.3f}")
            print(f"     Constraint Recall:    {metrics['constraint_recall']:.3f}")
            print(f"     Constraint F1:        {metrics['constraint_f1']:.3f} [PRIMARY]")
            print(f"     Matched predicted:    {metrics['num_matched_pred_constraints']}/{metrics['num_pred_constraints']}")
            print(f"     Matched gold:         {metrics['num_matched_gold_constraints']}/{metrics['num_gold_constraints']}")
            
            print(f"\n  VIOLATION DISCOVERY - AGGREGATE (All constraints pooled, union-based):")
            print(f"     Violation Precision: {metrics['violation_precision_aggregate']:.3f}")
            print(f"     Violation Recall:    {metrics['violation_recall_aggregate']:.3f}")
            print(f"     Violation F1:        {metrics['violation_f1_aggregate']:.3f}")
            print(f"     Gold violations:     {metrics['total_gold_violations']}")
            print(f"     Pred violations:     {metrics['total_pred_violations']}")
            print(f"     Overlap violations:  {metrics['total_overlap_violations']}")
            
            return {
                'type': 'document',
                'metrics': metrics,
                'num_predicted': len(results),
                'num_gold': len(gold_rules)
            }
        else:
            # Sentence modes: full metrics
            predictions = [(sent, preds) for sent, is_rule, preds in results]
            
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
                output_type="sentence_mappings"
            )
            
            # Display detailed metrics
            print(f"\n  CONSTRAINT DETECTION (Which sentences contain constraints):")
            print(f"     Precision: {metrics.detection_precision:.3f}")
            print(f"     Recall:    {metrics.detection_recall:.3f}")
            print(f"     F1:        {metrics.detection_f1:.3f}")
            print(f"     Confusion: TP={metrics.detection_tp}, FP={metrics.detection_fp}, "
                  f"FN={metrics.detection_fn}, TN={metrics.detection_tn}")
            
            print(f"\n  CONSTRAINT SYNTACTIC QUALITY (How well constraints match syntactically, n={metrics.strict_total}):")
            print(f"     Exact Match:       {metrics.strict_accuracy:.3f} ({metrics.strict_correct}/{metrics.strict_total})")
            print(f"     Predicate Jaccard: {metrics.avg_predicate_jaccard:.3f} (±{metrics.std_predicate_jaccard:.3f})")
            print(f"     Soft Jaccard:      {metrics.avg_soft_jaccard:.3f} (±{metrics.std_soft_jaccard:.3f})")
            
            print(f"\n  VIOLATION DISCOVERY - PER-CONSTRAINT (How well each constraint finds violations):")
            print(f"     Exact Match:       {metrics.execution_accuracy:.3f} ({metrics.execution_correct}/{metrics.execution_total})")
            print(f"     Violation Jaccard: {metrics.avg_execution_jaccard:.3f} (±{metrics.std_execution_jaccard:.3f}) [PRIMARY]")
            print(f"     Avg Precision:     {metrics.avg_execution_precision:.3f} (±{metrics.std_execution_precision:.3f})")
            print(f"     Avg Recall:        {metrics.avg_execution_recall:.3f} (±{metrics.std_execution_recall:.3f})")
            print(f"     Avg F1:            {metrics.avg_execution_f1:.3f} (±{metrics.std_execution_f1:.3f})")
            
            print(f"\n  VIOLATION DISCOVERY - AGGREGATE (All constraints pooled, union-based):")
            print(f"     Violation Precision:    {metrics.union_precision:.3f}")
            print(f"     Violation Recall:       {metrics.union_recall:.3f}")
            print(f"     Violation F1:           {metrics.union_f1:.3f} [PRIMARY]")
            print(f"     Gold violations:        {metrics.total_gold_violations}")
            print(f"     Predicted violations:   {metrics.total_pred_violations}")
            print(f"     Overlap violations:     {metrics.total_overlap_violations}")
            
            # Show syntactic vs violation-based gap
            gap = metrics.execution_accuracy - metrics.strict_accuracy
            print(f"\n  INSIGHT: Syntactic Quality vs Violation Discovery Gap = {gap:.3f}")
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
    
    def _create_sentence_direct_pipeline(self, variant: Dict, predicate_space: Dict):
        """Create SENTENCE_DIRECT pipeline."""
        mapper_config = variant['mapper']
        mapper = LLMMapper(
            model_name=mapper_config['model'],
            prompt_name=mapper_config['prompt_key'],
            allowed_predicates=predicate_space['predicates'],
            chunk_size=mapper_config.get('batch_size', 10)
        )
        return NormaPipeline(mode=ProcessingMode.SENTENCE_DIRECT, mapper=mapper)
    
    def _create_sentence_filtered_pipeline(self, variant: Dict, predicate_space: Dict):
        """Create SENTENCE_FILTERED pipeline."""
        classifier_config = variant['classifier']
        mapper_config = variant['mapper']
        
        # Load prompt template for classifier
        classifier_prompt = self.prompt_loader.load(
            "classification", classifier_config['prompt_key']
        )
        
        classifier = LLMRuleClassifier(
            model_name=classifier_config['model'],
            prompt_template=classifier_prompt
        )
        
        mapper = LLMMapper(
            model_name=mapper_config['model'],
            prompt_name=mapper_config['prompt_key'],
            allowed_predicates=predicate_space['predicates'],
            chunk_size=mapper_config.get('batch_size', 10)
        )
        
        return NormaPipeline(
            mode=ProcessingMode.SENTENCE_FILTERED,
            classifier=classifier,
            mapper=mapper
        )
    
    def _create_document_pipeline(self, variant: Dict, predicate_space: Dict, context: Dict):
        """Create DOCUMENT pipeline."""
        doc_mapper_config = variant['document_mapper']
        
        # Add predicates to context (required for document mapper)
        predicates_list_str = "\n".join(predicate_space['predicates'])
        context['predicates_list'] = predicates_list_str
        
        document_mapper = LLMDocumentMapper(
            model_name=doc_mapper_config['model'],
            prompt_name=doc_mapper_config['prompt_key']
        )
        
        return NormaPipeline(mode=ProcessingMode.DOCUMENT, document_mapper=document_mapper)
    
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
                'constraint_detection_status': r.detection_status,
                'gold_constraint': str(r.gold_rule) if r.gold_rule else None,
                'pred_constraint': str(r.pred_rule) if r.pred_rule else None,
            }
            
            # Add syntactic metrics (only for gold constraints)
            if r.gold_is_rule:
                record.update({
                    'constraint_syntactic_exact_match': r.strict_correct,
                    'constraint_syntactic_predicate_jaccard': r.predicate_jaccard,
                    'constraint_syntactic_soft_jaccard': r.soft_jaccard,
                })
            
            # Add violation discovery metrics (only for gold constraints)
            if r.execution_exact is not None:
                record.update({
                    'violation_discovery_exact_match': r.execution_exact,
                    'violation_discovery_jaccard': r.execution_jaccard,
                    'violation_discovery_precision': r.execution_precision,
                    'violation_discovery_recall': r.execution_recall,
                    'violation_discovery_f1': r.execution_f1,
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
                'detection_f1': metrics.detection_f1,
                'execution_accuracy': metrics.execution_accuracy,
                'avg_execution_jaccard': metrics.avg_execution_jaccard,
                'union_f1': metrics.union_f1
            }
        }
    
    def _prepare_document_metrics(self, metrics, predicted_rules, gold_rules):
        """Prepare detailed metrics for document-level."""
        return {
            'evaluation_type': 'document',
            'summary_metrics': {
                'constraint_f1': metrics['constraint_f1'],
                'violation_f1_aggregate': metrics['violation_f1_aggregate']
            },
            'num_predicted': len(predicted_rules),
            'num_gold': len(gold_rules)
        }
    
    def _build_result_dict(self, dataset_name: str, variant: Dict,
                          result: Dict, num_sentences: int, num_rules: int) -> Dict:
        """Build result dictionary."""
        mode = variant.get('mode', 'SENTENCE_DIRECT')
        
        base_dict = {
            'dataset': dataset_name,
            'variant_name': variant['name'],
            'mode': mode,
            'num_sentences': num_sentences,
            'status': 'success'
        }
        
        if result['type'] == 'document':
            metrics = result['metrics']
            base_dict.update({
                # Constraint discovery
                'constraint_discovery_precision': metrics['constraint_precision'],
                'constraint_discovery_recall': metrics['constraint_recall'],
                'constraint_discovery_f1': metrics['constraint_f1'],
                'num_matched_pred_constraints': metrics['num_matched_pred_constraints'],
                'num_matched_gold_constraints': metrics['num_matched_gold_constraints'],
                'num_gold_constraints': metrics['num_gold_constraints'],
                'num_pred_constraints': metrics['num_pred_constraints'],
                # Violation discovery - aggregate
                'violation_precision_aggregate': metrics['violation_precision_aggregate'],
                'violation_recall_aggregate': metrics['violation_recall_aggregate'],
                'violation_f1_aggregate': metrics['violation_f1_aggregate'],
                'total_gold_violations': metrics['total_gold_violations'],
                'total_pred_violations': metrics['total_pred_violations'],
                'total_overlap_violations': metrics['total_overlap_violations'],
                # N/A for sentence-only metrics
                'constraint_detection_f1': 'N/A',
                'constraint_syntactic_quality': 'N/A',
                'violation_discovery_per_constraint_exact': 'N/A',
                'violation_discovery_per_constraint_jaccard': 'N/A'
            })
        else:
            # Sentence-level modes: include all detailed metrics
            metrics = result['metrics']
            base_dict.update({
                # Constraint Detection
                'constraint_detection_precision': metrics.detection_precision,
                'constraint_detection_recall': metrics.detection_recall,
                'constraint_detection_f1': metrics.detection_f1,
                'constraint_detection_tp': metrics.detection_tp,
                'constraint_detection_fp': metrics.detection_fp,
                'constraint_detection_fn': metrics.detection_fn,
                'constraint_detection_tn': metrics.detection_tn,
                # Constraint Syntactic Quality
                'constraint_syntactic_exact_match': metrics.strict_accuracy,
                'constraint_syntactic_exact_match_count': metrics.strict_correct,
                'constraint_syntactic_total': metrics.strict_total,
                'constraint_syntactic_predicate_jaccard': metrics.avg_predicate_jaccard,
                'constraint_syntactic_predicate_jaccard_std': metrics.std_predicate_jaccard,
                'constraint_syntactic_soft_jaccard': metrics.avg_soft_jaccard,
                'constraint_syntactic_soft_jaccard_std': metrics.std_soft_jaccard,
                # Violation Discovery - Per-Constraint
                'violation_discovery_per_constraint_exact': metrics.execution_accuracy,
                'violation_discovery_per_constraint_exact_count': metrics.execution_correct,
                'violation_discovery_per_constraint_total': metrics.execution_total,
                'violation_discovery_per_constraint_jaccard': metrics.avg_execution_jaccard,
                'violation_discovery_per_constraint_jaccard_std': metrics.std_execution_jaccard,
                'violation_discovery_per_constraint_precision': metrics.avg_execution_precision,
                'violation_discovery_per_constraint_precision_std': metrics.std_execution_precision,
                'violation_discovery_per_constraint_recall': metrics.avg_execution_recall,
                'violation_discovery_per_constraint_recall_std': metrics.std_execution_recall,
                'violation_discovery_per_constraint_f1': metrics.avg_execution_f1,
                'violation_discovery_per_constraint_f1_std': metrics.std_execution_f1,
                # Violation Discovery - Aggregate
                'violation_precision_aggregate': metrics.union_precision,
                'violation_recall_aggregate': metrics.union_recall,
                'violation_f1_aggregate': metrics.union_f1,
                'total_gold_violations': metrics.total_gold_violations,
                'total_pred_violations': metrics.total_pred_violations,
                'total_overlap_violations': metrics.total_overlap_violations,
                # Counts
                'num_gold_constraints': num_rules,
                'num_pred_constraints': 'N/A'
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
            'constraint_detection_f1': 'N/A',
            'constraint_discovery_f1': 'N/A',
            'constraint_syntactic_quality': 'N/A',
            'violation_discovery_per_constraint_exact': 'N/A',
            'violation_discovery_per_constraint_jaccard': 'N/A',
            'violation_f1_aggregate': 'N/A'
        }
    
    def _print_comparison_table(self, results: Dict):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        
        # Group by type
        sentence_results = {k: v for k, v in results.items() if v['type'] == 'sentence'}
        document_results = {k: v for k, v in results.items() if v['type'] == 'document'}
        
        if sentence_results:
            print("\nSentence-Level Results (Detailed):")
            print(f"{'Variant':<40} {'Mode':<18} {'Const':<8} {'Syntax':<8} {'Viol/Con':<8} {'Viol Jac':<8} {'Viol Agg':<8}")
            print(f"{'':40} {'':18} {'Det F1':<8} {'Quality':<8} {'Exact':<8} {'(per-C)':<8} {'F1':<8}")
            print("-" * 110)
            
            for name, result in sentence_results.items():
                m = result['metrics']
                print(f"{name:<40} {result.get('mode', 'DIRECT'):<18} "
                      f"{m.detection_f1:<8.3f} {m.strict_accuracy:<8.3f} "
                      f"{m.execution_accuracy:<8.3f} {m.avg_execution_jaccard:<8.3f} {m.union_f1:<8.3f}")
            
            print(f"\n{'Key Metrics:':<40}")
            best_det = max(sentence_results.items(), key=lambda x: x[1]['metrics'].detection_f1)
            print(f"  Best Constraint Detection F1:      {best_det[0][:50]} ({best_det[1]['metrics'].detection_f1:.3f})")
            
            best_exec = max(sentence_results.items(), key=lambda x: x[1]['metrics'].execution_accuracy)
            print(f"  Best Violation Discovery (per-C):  {best_exec[0][:50]} ({best_exec[1]['metrics'].execution_accuracy:.3f})")
            
            best_union = max(sentence_results.items(), key=lambda x: x[1]['metrics'].union_f1)
            print(f"  Best Violation Discovery (agg):    {best_union[0][:50]} ({best_union[1]['metrics'].union_f1:.3f})")
        
        if document_results:
            print("\nDocument-Level Results:")
            print(f"{'Variant':<40} {'Const F1':<10} {'Viol F1':<10} {'Matched':<15}")
            print(f"{'':40} {'(discovery)':<10} {'(aggregate)':<10} {'(C/V)':<15}")
            print("-" * 80)
            
            for name, result in document_results.items():
                m = result['metrics']
                matched_str = f"{m['num_matched_pred_constraints']}/{m['num_pred_constraints']}"
                print(f"{name:<40} {m['constraint_f1']:<10.3f} "
                      f"{m['violation_f1_aggregate']:<10.3f} {matched_str:<15}")
            
            best_constraint = max(document_results.items(), key=lambda x: x[1]['metrics']['constraint_f1'])
            best_violation = max(document_results.items(), key=lambda x: x[1]['metrics']['violation_f1_aggregate'])
            print(f"\nBest Constraint Discovery: {best_constraint[0]} ({best_constraint[1]['metrics']['constraint_f1']:.3f})")
            print(f"Best Violation Discovery:  {best_violation[0]} ({best_violation[1]['metrics']['violation_f1_aggregate']:.3f})")
        
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run mapper experiments (SENTENCE_DIRECT/FILTERED/DOCUMENT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mappers.py configs/mappers/sentence_level.yaml
  python run_mappers.py configs/mappers/document_level.yaml
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
