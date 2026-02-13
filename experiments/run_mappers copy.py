#!/usr/bin/env python3
"""
Mapper Experiment Runner

Handles both sentence-level and document-level mappers with appropriate metrics for each.

Usage:
    python run_mappers.py configs/mappers/sentence_level.yaml
    python run_mappers.py configs/mappers/document_level.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.experiment_base import BaseExperiment
from src.mappers.llm import LLMMapper
from src.mappers.llm_document_mapper import LLMDocumentMapper
from src.evaluation.mapping import MappingEvaluator
from src.llm.helper import PromptLoader


class MapperExperiment(BaseExperiment):
    """Run mapper experiments (sentence-level or document-level)."""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        # Validate experiment type
        if self.config['experiment_type'] not in ['sentence_mapper', 'document_mapper']:
            raise ValueError(
                f"Expected experiment_type 'sentence_mapper' or 'document_mapper', "
                f"got '{self.config['experiment_type']}'"
            )
        
        self.is_document_level = self.config['experiment_type'] == 'document_mapper'
        
        # Initialize prompt loader
        self.prompt_loader = PromptLoader()
        
        # Sentence filter only applies to sentence-level
        if not self.is_document_level:
            valid_filters = ['all', 'rules_only', 'non_rules_only']
            self.sentence_filter = self.config.get('sentence_filter', 'all')
            if self.sentence_filter not in valid_filters:
                raise ValueError(
                    f"Invalid sentence_filter '{self.sentence_filter}'. "
                    f"Must be one of: {valid_filters}"
                )
        else:
            self.sentence_filter = None
    
    def run(self):
        """Run mapper experiments across all datasets and variants."""
        mode = "DOCUMENT-LEVEL" if self.is_document_level else "SENTENCE-LEVEL"
        
        print("\n" + "=" * 80)
        print(f"{mode} MAPPER EXPERIMENT: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")
        
        if not self.is_document_level:
            print(f"Sentence filter: {self.sentence_filter}")
        
        print("\nEvaluation metrics:")
        if self.is_document_level:
            print("  Global execution metrics (union-based)")
            print("  Coverage analysis")
            print("  Per-sentence metrics: not applicable")
        else:
            print("  Detection metrics (TP/FP/FN/TN)")
            print("  Syntactic correctness (exact match, Jaccard)")
            print("  Execution correctness (per-rule and global)")
        
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
        """Run all mapper variants on a single dataset."""
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        try:
            df, predicate_space, ground_truth, dataframes = self.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return len(self.config['variants'])
        
        if self.is_document_level:
            # Document-level: process all sentences together
            sentences = [gt['sentences'][0] for gt in ground_truth]
            # Collect all gold rules (flatten the list)
            gold_rules = []
            for gt in ground_truth:
                if gt['rules']:
                    for rule in gt['rules']:
                        gold_rules.append(rule)
            
            print(f"Total sentences: {len(sentences)}")
            print(f"Gold constraints: {len(gold_rules)}\\n")
            
            filtered_sentences = sentences
            filtered_ground_truth = ground_truth
            num_rules = len(gold_rules)
        else:
            # Sentence-level: filter sentences
            sentences, filtered_ground_truth = self._filter_sentences(ground_truth)
            
            num_total = len(ground_truth)
            num_filtered = len(sentences)
            num_rules = sum(1 for item in filtered_ground_truth 
                           if item['rules'] is not None and len(item['rules']) > 0)
            
            print(f"Total sentences: {num_total}")
            print(f"Filtered sentences (filter={self.sentence_filter}): {num_filtered}")
            print(f"Rule sentences in filtered set: {num_rules}\n")
            
            if num_filtered == 0:
                print("Warning: No sentences to process. Skipping dataset.")
                return len(self.config['variants'])
            
            filtered_sentences = sentences
        
        # Build sentence->rules map for sentence-level evaluator
        sentence2dcs_map = {}
        for gt in filtered_ground_truth:
            sentence = gt['sentences'][0]
            if gt['rules'] and len(gt['rules']) > 0:
                sentence2dcs_map[sentence] = gt['rules'][0]
            else:
                sentence2dcs_map[sentence] = None
        
        # Create evaluator with DB for execution metrics
        evaluator = MappingEvaluator(ground_truth=sentence2dcs_map, df=df)
        
        # Run each variant
        results = {}
        failed_count = 0
        
        for i, variant in enumerate(self.config['variants'], 1):
            name = variant['name']
            print(f"\n[{i}/{len(self.config['variants'])}] Processing variant: {name}")
            
            try:
                if self.is_document_level:
                    result = self.run_document_variant(
                        name=name,
                        variant=variant,
                        sentences=filtered_sentences,
                        gold_rules=gold_rules,
                        predicate_space=predicate_space,
                        evaluator=evaluator,
                        df=df,
                        dataset_name=dataset_name
                    )
                else:
                    result = self.run_sentence_variant(
                        name=name,
                        variant=variant,
                        sentences=filtered_sentences,
                        ground_truth=filtered_ground_truth,
                        predicate_space=predicate_space,
                        evaluator=evaluator,
                        df=df,
                        dataset_name=dataset_name
                    )
                
                results[name] = result
                
                # Track results with appropriate metrics
                result_dict = self._build_result_dict(
                    dataset_name=dataset_name,
                    variant=variant,
                    result=result,
                    num_sentences=len(filtered_sentences),
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
                    dataset_name=dataset_name,
                    variant=variant,
                    error=str(e),
                    num_sentences=len(filtered_sentences)
                ))
        
        # Close DB connection
        evaluator.close()
        
        # Print comparison table
        if results:
            self._print_comparison_table(results)
        
        return failed_count
    
    def run_sentence_variant(self, name: str, variant: Dict, sentences: List[str], 
                            ground_truth: List[Dict], predicate_space: Dict,
                            evaluator, df, dataset_name: str) -> Dict:
        """Run a sentence-level mapper variant."""
        print(f"{'─' * 60}")
        print(f"  Model: {variant.get('model', 'N/A')}")
        print(f"  Implementation: {variant['implementation']}")
        print(f"  Prompt: {variant.get('prompt_key', 'N/A')}")
        
        # Extract config
        implementation = variant['implementation']
        model = variant.get('model', 'gpt-5-mini')
        prompt_key = variant.get('prompt_key')
        use_context = variant.get('use_context', False)
        num_rows_context = variant.get('num_rows_context', 3)
        
        # Create mapper
        print(f"  Creating sentence mapper...")
        mapper = self._create_sentence_mapper(implementation, model, prompt_key, variant, predicate_space)
        
        # Build context
        context = None
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, dataset_name, num_rows=num_rows_context)
        
        # Map sentences
        print(f"  Mapping {len(sentences)} sentences...")
        mapping_results = mapper.map(sentences=sentences, context=context)
        
        # Format as (sentence, constraints) pairs
        predictions = [(sent, preds) for sent, preds in mapping_results]
        
        # Evaluate with full metrics
        print(f"  Evaluating with DB execution...")
        metrics, sentence_results = evaluator.evaluate(predictions)
        
        # Prepare detailed results
        detailed_metrics = self._prepare_sentence_detailed_metrics(metrics, sentence_results)
        
        # Save outputs
        print(f"  Saving outputs...")
        outputs_file = self.save_outputs(
            outputs=[{'sentence': s, 'constraints': c} for s, c in predictions],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="sentence_mappings"
        )
        
        metrics_file = self.save_detailed_metrics(
            metrics_data=detailed_metrics,
            dataset_name=dataset_name,
            variant_name=name
        )
        
        # Display key metrics
        print(f"\n  Detection Metrics:")
        print(f"     Precision: {metrics.detection_precision:.3f}")
        print(f"     Recall:    {metrics.detection_recall:.3f}")
        print(f"     F1:        {metrics.detection_f1:.3f}")
        
        print(f"\n  Syntactic Accuracy:")
        print(f"     Exact Match:  {metrics.strict_accuracy:.3f}")
        print(f"     Pred Jaccard: {metrics.avg_predicate_jaccard:.3f}")
        
        print(f"\n  Execution Accuracy [PRIMARY]:")
        print(f"     Exact Match:  {metrics.execution_accuracy:.3f}")
        print(f"     Exec Jaccard: {metrics.avg_execution_jaccard:.3f}")
        print(f"     Union F1:     {metrics.union_f1:.3f}")
        
        print(f"\n  Output: {outputs_file.name}")
        print(f"  Metrics: {metrics_file.name}")
        
        return {
            'type': 'sentence',
            'metrics': metrics,
            'sentence_results': sentence_results
        }
    
    def run_document_variant(self, name: str, variant: Dict, sentences: List[str],
                            gold_rules: List, predicate_space: Dict, evaluator, 
                            df, dataset_name: str) -> Dict:
        """Run a document-level mapper variant."""
        print(f"{'─' * 60}")
        print(f"  Model: {variant.get('model', 'N/A')}")
        print(f"  Implementation: {variant['implementation']}")
        print(f"  Prompt: {variant.get('prompt_key', 'N/A')}")
        
        # Extract config
        implementation = variant['implementation']
        model = variant.get('model', 'gpt-5-mini')
        prompt_key = variant.get('prompt_key')
        use_context = variant.get('use_context', False)
        num_rows_context = variant.get('num_rows_context', 3)
        
        # Create document mapper
        print(f"  Creating document mapper...")
        mapper = self._create_document_mapper(implementation, model, prompt_key)
        
        # Build context
        context = {}
        if use_context:
            print(f"  Building context (num_rows={num_rows_context})...")
            context = self._build_context(df, dataset_name, num_rows=num_rows_context)
        
        # Add predicate space to context for document mapper
        predicates_list_str = "\n".join(predicate_space['predicates'])
        context['predicates_list'] = predicates_list_str
        
        # Map entire document
        print(f"  Processing document ({len(sentences)} sentences)...")
        predicted_rules = mapper.extract(sentences, context=context)
        
        print(f"  Extracted {len(predicted_rules)} constraints")
        
        # Evaluate at document level
        print(f"  Evaluating with DB execution...")
        metrics = evaluator.evaluate_document_level(
            predicted_rules=predicted_rules,
            gold_rules=gold_rules
        )
        
        # Save outputs
        print(f"  Saving outputs...")
        outputs_file = self.save_outputs(
            outputs=[{'rule': str(rule)} for rule in predicted_rules],
            dataset_name=dataset_name,
            variant_name=name,
            output_type="document_mappings"
        )
        
        detailed_metrics = self._prepare_document_detailed_metrics(
            metrics, predicted_rules, gold_rules, len(sentences)
        )
        
        metrics_file = self.save_detailed_metrics(
            metrics_data=detailed_metrics,
            dataset_name=dataset_name,
            variant_name=name
        )
        
        # Display metrics
        print(f"\n  Global Execution Metrics:")
        print(f"     Union Precision: {metrics['union_precision']:.3f}")
        print(f"     Union Recall:    {metrics['union_recall']:.3f}")
        print(f"     Union F1:        {metrics['union_f1']:.3f} [PRIMARY]")
        
        print(f"\n  Coverage:")
        print(f"     Gold violations:    {metrics['total_gold_violations']}")
        print(f"     Pred violations:    {metrics['total_pred_violations']}")
        print(f"     Overlap violations: {metrics['total_overlap_violations']}")
        
        ratio = len(predicted_rules) / len(gold_rules) if gold_rules else 0
        print(f"\n  Constraint Count:")
        print(f"     Gold: {len(gold_rules)}")
        print(f"     Pred: {len(predicted_rules)}")
        print(f"     Ratio: {ratio:.2f}")
        
        print(f"\n  Output: {outputs_file.name}")
        print(f"  Metrics: {metrics_file.name}")
        
        return {
            'type': 'document',
            'metrics': metrics,
            'predicted_rules': predicted_rules,
            'num_gold_rules': len(gold_rules)
        }
    
    def _build_result_dict(self, dataset_name: str, variant: Dict, 
                          result: Dict, num_sentences: int, num_rules: int) -> Dict:
        """Build result dictionary with appropriate metrics based on experiment type."""
        base_dict = {
            'dataset': dataset_name,
            'variant_name': variant['name'],
            'implementation': variant['implementation'],
            'model': variant.get('model', 'N/A'),
            'prompt_key': variant.get('prompt_key', 'N/A'),
            'batch_size': variant.get('batch_size', 'N/A'),
            'use_context': variant.get('use_context', False),
            'num_rows_context': variant.get('num_rows_context', 3),
            'num_sentences': num_sentences,
            'status': 'success'
        }
        
        if self.is_document_level:
            # Document-level metrics only
            metrics = result['metrics']
            base_dict.update({
                'sentence_filter': 'N/A',
                'num_gold_rules': result['num_gold_rules'],
                'num_pred_rules': metrics['num_pred_rules'],
                # Global execution metrics
                'union_precision': metrics['union_precision'],
                'union_recall': metrics['union_recall'],
                'union_f1': metrics['union_f1'],
                'total_gold_violations': metrics['total_gold_violations'],
                'total_pred_violations': metrics['total_pred_violations'],
                'total_overlap_violations': metrics['total_overlap_violations'],
                # Sentence-level metrics marked as N/A
                'detection_precision': 'N/A',
                'detection_recall': 'N/A',
                'detection_f1': 'N/A',
                'detection_tp': 'N/A',
                'detection_fp': 'N/A',
                'detection_fn': 'N/A',
                'detection_tn': 'N/A',
                'strict_accuracy': 'N/A',
                'avg_predicate_jaccard': 'N/A',
                'avg_soft_jaccard': 'N/A',
                'execution_accuracy': 'N/A',
                'avg_execution_jaccard': 'N/A',
                'avg_execution_precision': 'N/A',
                'avg_execution_recall': 'N/A',
                'avg_execution_f1': 'N/A'
            })
        else:
            # Sentence-level: all metrics available
            metrics = result['metrics']
            base_dict.update({
                'sentence_filter': self.sentence_filter,
                'num_rules': num_rules,
                # Detection metrics
                'detection_precision': metrics.detection_precision,
                'detection_recall': metrics.detection_recall,
                'detection_f1': metrics.detection_f1,
                'detection_tp': metrics.detection_tp,
                'detection_fp': metrics.detection_fp,
                'detection_fn': metrics.detection_fn,
                'detection_tn': metrics.detection_tn,
                # Syntactic metrics
                'strict_accuracy': metrics.strict_accuracy,
                'avg_predicate_jaccard': metrics.avg_predicate_jaccard,
                'avg_soft_jaccard': metrics.avg_soft_jaccard,
                # Execution metrics (per-rule)
                'execution_accuracy': metrics.execution_accuracy,
                'avg_execution_jaccard': metrics.avg_execution_jaccard,
                'avg_execution_precision': metrics.avg_execution_precision,
                'avg_execution_recall': metrics.avg_execution_recall,
                'avg_execution_f1': metrics.avg_execution_f1,
                # Global execution metrics
                'union_precision': metrics.union_precision,
                'union_recall': metrics.union_recall,
                'union_f1': metrics.union_f1,
                'total_gold_violations': metrics.total_gold_violations,
                'total_pred_violations': metrics.total_pred_violations,
                'total_overlap_violations': metrics.total_overlap_violations,
                'num_gold_rules': 'N/A',
                'num_pred_rules': 'N/A'
            })
        
        return base_dict
    
    def _create_sentence_mapper(self, implementation: str, model: str, prompt_key: Optional[str], 
                               variant: Dict, predicate_space: Dict):
        """Factory method to create sentence-level mapper."""
        
        if implementation == 'llm':
            if not prompt_key:
                raise ValueError("LLM implementation requires 'prompt_key'")
            
            return LLMMapper(
                model_name=model,
                prompt_name=prompt_key,
                allowed_predicates=predicate_space['predicates'],
                chunk_size=variant.get('batch_size', 10)
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    
    def _create_document_mapper(self, implementation: str, model: str, prompt_key: Optional[str]):
        """Factory method to create document-level mapper."""
        
        if implementation == 'llm':
            if not prompt_key:
                raise ValueError("LLM implementation requires 'prompt_key'")
            
            return LLMDocumentMapper(
                model_name=model,
                prompt_name=prompt_key
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    
    def _build_failure_dict(self, dataset_name: str, variant: Dict, 
                           error: str, num_sentences: int) -> Dict:
        """Build failure result dictionary."""
        base_dict = {
            'dataset': dataset_name,
            'variant_name': variant['name'],
            'implementation': variant['implementation'],
            'model': variant.get('model', 'N/A'),
            'num_sentences': num_sentences,
            'status': 'failed',
            'error': error
        }
        
        # Set all metrics to N/A
        na_metrics = [
            'detection_precision', 'detection_recall', 'detection_f1',
            'detection_tp', 'detection_fp', 'detection_fn', 'detection_tn',
            'strict_accuracy', 'avg_predicate_jaccard', 'avg_soft_jaccard',
            'execution_accuracy', 'avg_execution_jaccard',
            'avg_execution_precision', 'avg_execution_recall', 'avg_execution_f1',
            'union_precision', 'union_recall', 'union_f1',
            'total_gold_violations', 'total_pred_violations', 'total_overlap_violations'
        ]
        
        for metric in na_metrics:
            base_dict[metric] = 'N/A'
        
        return base_dict
    
    def _filter_sentences(self, ground_truth: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Filter sentences based on config (sentence-level only)."""
        if self.sentence_filter == 'all':
            sentences = [gt['sentences'][0] for gt in ground_truth]
            return sentences, ground_truth
        elif self.sentence_filter == 'rules_only':
            filtered = [gt for gt in ground_truth 
                       if gt['rules'] is not None and len(gt['rules']) > 0]
            sentences = [gt['sentences'][0] for gt in filtered]
            return sentences, filtered
        elif self.sentence_filter == 'non_rules_only':
            filtered = [gt for gt in ground_truth 
                       if gt['rules'] is None or len(gt['rules']) == 0]
            sentences = [gt['sentences'][0] for gt in filtered]
            return sentences, filtered
    
    def _build_context(self, df, dataset_name: str, num_rows: int = 3):
        """Build table context."""
        from src.utils.table_context import TableContextBuilder
        context_builder = TableContextBuilder(sample_strategy="head", num_rows=num_rows)
        return context_builder.build(df)
    
    def _prepare_sentence_detailed_metrics(self, metrics, sentence_results):
        """Prepare detailed metrics for sentence-level evaluation."""
        # Categorize by detection status
        by_detection = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
        for r in sentence_results:
            by_detection[r.detection_status].append({
                'sentence': r.sentence[:200],
                'gold_rule': str(r.gold_rule) if r.gold_rule else None,
                'pred_rule': str(r.pred_rule) if r.pred_rule else None,
                'strict_correct': r.strict_correct,
                'predicate_jaccard': r.predicate_jaccard,
                'execution_exact': r.execution_exact,
                'execution_jaccard': r.execution_jaccard
            })
        
        return {
            'evaluation_type': 'sentence_level',
            'summary_metrics': {
                'detection_f1': metrics.detection_f1,
                'strict_accuracy': metrics.strict_accuracy,
                'execution_accuracy': metrics.execution_accuracy,
                'avg_execution_jaccard': metrics.avg_execution_jaccard,
                'union_f1': metrics.union_f1
            },
            'detection_counts': {k: len(v) for k, v in by_detection.items()},
            'examples_by_detection': {k: v[:10] for k, v in by_detection.items()},
            'full_metrics': {
                'detection_precision': metrics.detection_precision,
                'detection_recall': metrics.detection_recall,
                'detection_f1': metrics.detection_f1,
                'detection_tp': metrics.detection_tp,
                'detection_fp': metrics.detection_fp,
                'detection_fn': metrics.detection_fn,
                'detection_tn': metrics.detection_tn,
                'strict_accuracy': metrics.strict_accuracy,
                'strict_correct': metrics.strict_correct,
                'strict_total': metrics.strict_total,
                'avg_predicate_jaccard': metrics.avg_predicate_jaccard,
                'avg_soft_jaccard': metrics.avg_soft_jaccard,
                'execution_accuracy': metrics.execution_accuracy,
                'execution_correct': metrics.execution_correct,
                'execution_total': metrics.execution_total,
                'avg_execution_jaccard': metrics.avg_execution_jaccard,
                'avg_execution_precision': metrics.avg_execution_precision,
                'avg_execution_recall': metrics.avg_execution_recall,
                'avg_execution_f1': metrics.avg_execution_f1,
                'union_precision': metrics.union_precision,
                'union_recall': metrics.union_recall,
                'union_f1': metrics.union_f1,
                'total_gold_violations': metrics.total_gold_violations,
                'total_pred_violations': metrics.total_pred_violations,
                'total_overlap_violations': metrics.total_overlap_violations
            }
        }
    
    def _prepare_document_detailed_metrics(self, metrics, predicted_rules, 
                                          gold_rules, num_sentences):
        """Prepare detailed metrics for document-level evaluation."""
        return {
            'evaluation_type': 'document_level',
            'num_sentences': num_sentences,
            'num_gold_rules': len(gold_rules),
            'num_pred_rules': len(predicted_rules),
            'summary_metrics': {
                'union_precision': metrics['union_precision'],
                'union_recall': metrics['union_recall'],
                'union_f1': metrics['union_f1']
            },
            'coverage': {
                'total_gold_violations': metrics['total_gold_violations'],
                'total_pred_violations': metrics['total_pred_violations'],
                'total_overlap_violations': metrics['total_overlap_violations']
            },
            'predicted_rules_sample': [str(r) for r in predicted_rules[:20]],
            'full_metrics': metrics
        }
    
    def _print_comparison_table(self, results: Dict):
        """Print comparison table (adapts to experiment type)."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        
        if self.is_document_level:
            # Document-level comparison
            print(f"{'Variant':<40} {'U-Prec':<10} {'U-Rec':<10} {'U-F1':<10}")
            print("-" * 80)
            
            for name, result in results.items():
                m = result['metrics']
                print(f"{name:<40} {m['union_precision']:<10.3f} "
                      f"{m['union_recall']:<10.3f} {m['union_f1']:<10.3f}")
            
            if results:
                print("\n" + "=" * 80)
                best = max(results.items(), key=lambda x: x[1]['metrics']['union_f1'])
                print(f"Best Union F1: {best[0]} ({best[1]['metrics']['union_f1']:.3f})")
                print("=" * 80)
        else:
            # Sentence-level comparison
            print(f"{'Variant':<30} {'Det F1':<10} {'Exec Acc':<10} {'Exec Jac':<10} {'Union F1':<10}")
            print("-" * 80)
            
            for name, result in results.items():
                m = result['metrics']
                print(f"{name:<30} {m.detection_f1:<10.3f} "
                      f"{m.execution_accuracy:<10.3f} "
                      f"{m.avg_execution_jaccard:<10.3f} {m.union_f1:<10.3f}")
            
            if results:
                print("\n" + "=" * 80)
                best_exec = max(results.items(), 
                              key=lambda x: x[1]['metrics'].execution_accuracy)
                best_union = max(results.items(), 
                               key=lambda x: x[1]['metrics'].union_f1)
                print(f"Best Exec Accuracy: {best_exec[0]} "
                      f"({best_exec[1]['metrics'].execution_accuracy:.3f})")
                print(f"Best Union F1:      {best_union[0]} "
                      f"({best_union[1]['metrics'].union_f1:.3f})")
                print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run mapper experiments (sentence-level or document-level)',
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