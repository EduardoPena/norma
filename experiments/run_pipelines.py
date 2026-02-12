#!/usr/bin/env python3
"""
Pipeline Experiment Runner

Runs end-to-end NORMA pipeline experiments (Classifier → Mapper).
Tests on all sentences (rules + non-rules) and evaluates both stages.

Usage:
    python run_pipelines.py configs/pipelines/norma_best.yaml
    python run_pipelines.py configs/pipelines/norma_ablations.yaml
"""

import sys
from pathlib import Path
import csv
import time
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.experiment_base import BaseExperiment
from src.classifiers import create_classifier
from src.mappers import PromptMapper
from src.pipelines import TwoPhaseNorma
from src.metrics.mapping import OneToOneMappingEvaluator


class PipelineExperiment(BaseExperiment):
    """Run pipeline (end-to-end NORMA) experiments."""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        # Validate this is a pipeline experiment
        if self.config['experiment_type'] != 'pipeline':
            raise ValueError(f"Expected experiment_type='pipeline', got '{self.config['experiment_type']}'")
    
    def run(self):
        """Run pipeline experiments across all datasets and variants."""
        self.print_header(f"PIPELINE EXPERIMENT: {self.config['experiment_name']}")
        
        # Save config copy
        self.save_config_copy()
        
        # Run for each dataset
        for dataset_name in self.config['datasets']:
            self.run_dataset(dataset_name)
    
    def run_dataset(self, dataset_name: str):
        """Run all pipeline variants on a single dataset."""
        self.print_subheader(f"Dataset: {dataset_name}")
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        df, predicate_space, ground_truth, dataframes = self.load_dataset(dataset_name)
        
        num_rules = sum(1 for item in ground_truth if item['rules'] is not None)
        num_total = len(ground_truth)
        
        print(f"Total sentences: {num_total}")
        print(f"Rule sentences: {num_rules}")
        print(f"Non-rule sentences: {num_total - num_rules}\n")
        
        # Extract sentences
        sentences = [gt_item['sentences'][0] for gt_item in ground_truth]
        table_description = self.get_table_description(dataset_name)
        
        # Prepare CSV
        csv_filename = self.get_csv_filename(dataset_name, "pipeline_results")
        
        # Run each variant
        results = {}
        for variant in self.config['variants']:
            name = variant['name']
            result_data = self.run_variant(
                name=name,
                variant=variant,
                sentences=sentences,
                ground_truth=ground_truth,
                predicate_space=predicate_space,
                df=df,
                table_description=table_description
            )
            results[name] = result_data
            
            # Save to CSV
            self.save_result_to_csv(
                csv_filename, dataset_name, name, variant, 
                result_data, num_rules, num_total
            )
        
        # Print comparison table
        self.print_comparison_table(results)
        
        print(f"\n✓ Results saved to: {csv_filename}")
    
    def run_variant(self, name: str, variant: Dict, sentences, ground_truth,
                   predicate_space, df, table_description: str):
        """Run a single pipeline variant."""
        print(f"\nRunning: {name}")
        
        classifier_cfg = variant['classifier'].copy()
        mapper_cfg = variant['mapper']
        
        print(f"  Classifier: {classifier_cfg['implementation']} - {classifier_cfg.get('model', 'N/A')}")
        print(f"  Mapper: {mapper_cfg['model']} - {mapper_cfg.get('prompt_key', 'basic')}")
        
        start_time = time.time()
        
        # Create classifier
        use_context = classifier_cfg.pop('use_context', True)
        classifier = create_classifier(classifier_cfg)
        
        # Prepare context
        context = None
        if use_context:
            context = classifier._build_context(df, table_description)
        
        # Classify
        print(f"  Classifying {len(sentences)} sentences...")
        classifications = classifier.classify(sentences, context=context)
        num_rules_pred = sum(classifications)
        print(f"  ✓ Classified {num_rules_pred}/{len(sentences)} as rules")
        
        # Compute classification metrics
        classification_tp = sum(1 for gt, pred in zip(ground_truth, classifications)
                               if (gt['rules'] is not None) and pred)
        classification_fp = sum(1 for gt, pred in zip(ground_truth, classifications)
                               if (gt['rules'] is None) and pred)
        classification_fn = sum(1 for gt, pred in zip(ground_truth, classifications)
                               if (gt['rules'] is not None) and not pred)
        
        cls_precision = classification_tp / (classification_tp + classification_fp) if (classification_tp + classification_fp) > 0 else 0.0
        cls_recall = classification_tp / (classification_tp + classification_fn) if (classification_tp + classification_fn) > 0 else 0.0
        cls_f1 = 2 * (cls_precision * cls_recall) / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0
        
        print(f"  Classification - P: {cls_precision:.2%}, R: {cls_recall:.2%}, F1: {cls_f1:.2%}")
        
        # Create mapper
        mapper = PromptMapper(
            predicate_space=predicate_space,
            df=df,
            model_name=mapper_cfg['model'],
            temperature=mapper_cfg.get('temperature', 0.0),
            prompt_key=mapper_cfg.get('prompt_key', 'basic'),
            batch_size=mapper_cfg.get('batch_size', 1)
        )
        
        # Run pipeline
        print(f"  Running mapping...")
        pipeline = TwoPhaseNorma(classifier=classifier, mapper=mapper)
        
        predictions = pipeline.map_multiple(
            sentences=sentences,
            context=context,
            classifications=classifications
        )
        
        # Format predictions
        formatted_predictions = []
        for gt_item, pred_rules in zip(ground_truth, predictions):
            formatted_predictions.append({'id': gt_item['id'], 'rules': pred_rules})
        
        # Evaluate mapping
        evaluator = OneToOneMappingEvaluator(ground_truth, df=df)
        eval_result = evaluator.evaluate(formatted_predictions)
        
        elapsed = time.time() - start_time
        
        # Display results
        mapping_metrics = eval_result.metrics
        print(f"  Mapping - Detection F1: {mapping_metrics['detection_f1']:.2%}, "
              f"Strict: {mapping_metrics['strict_accuracy']:.2%}, "
              f"Jaccard: {mapping_metrics['avg_jaccard']:.2%}")
        print(f"  Time: {elapsed:.1f}s")
        
        return {
            'classification_metrics': {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'tp': classification_tp,
                'fp': classification_fp,
                'fn': classification_fn
            },
            'mapping_result': eval_result,
            'elapsed_time': elapsed,
            'num_rules_pred': num_rules_pred
        }
    
    def save_result_to_csv(self, csv_filename: Path, dataset_name: str,
                          variant_name: str, variant_config: Dict,
                          result_data: Dict, num_rules: int, num_total: int):
        """Save a single result to CSV."""
        file_exists = csv_filename.exists()
        
        cls_metrics = result_data['classification_metrics']
        map_metrics = result_data['mapping_result'].metrics
        elapsed = result_data['elapsed_time']
        num_rules_pred = result_data['num_rules_pred']
        
        with open(csv_filename, 'a', newline='') as f:
            fieldnames = [
                'dataset', 'variant_name',
                'classifier_impl', 'classifier_model', 'classifier_prompt',
                'mapper_model', 'mapper_prompt',
                'cls_precision', 'cls_recall', 'cls_f1',
                'cls_tp', 'cls_fp', 'cls_fn',
                'map_detection_f1', 'map_strict_accuracy', 'map_avg_jaccard',
                'map_db_execution_accuracy', 'map_db_jaccard',
                'elapsed_time_seconds', 'num_rules_pred', 'num_rules_gt', 'num_total'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            classifier_cfg = variant_config['classifier']
            mapper_cfg = variant_config['mapper']
            
            writer.writerow({
                'dataset': dataset_name,
                'variant_name': variant_name,
                'classifier_impl': classifier_cfg['implementation'],
                'classifier_model': classifier_cfg.get('model', 'N/A'),
                'classifier_prompt': classifier_cfg.get('prompt_key', 'N/A'),
                'mapper_model': mapper_cfg['model'],
                'mapper_prompt': mapper_cfg.get('prompt_key', 'basic'),
                'cls_precision': f"{cls_metrics['precision']:.4f}",
                'cls_recall': f"{cls_metrics['recall']:.4f}",
                'cls_f1': f"{cls_metrics['f1']:.4f}",
                'cls_tp': cls_metrics['tp'],
                'cls_fp': cls_metrics['fp'],
                'cls_fn': cls_metrics['fn'],
                'map_detection_f1': f"{map_metrics['detection_f1']:.4f}",
                'map_strict_accuracy': f"{map_metrics['strict_accuracy']:.4f}",
                'map_avg_jaccard': f"{map_metrics['avg_jaccard']:.4f}",
                'map_db_execution_accuracy': f"{map_metrics.get('db_execution_accuracy', 0.0):.4f}",
                'map_db_jaccard': f"{map_metrics.get('avg_db_execution_jaccard', 0.0):.4f}",
                'elapsed_time_seconds': f"{elapsed:.2f}",
                'num_rules_pred': num_rules_pred,
                'num_rules_gt': num_rules,
                'num_total': num_total
            })
    
    def print_comparison_table(self, results: Dict):
        """Print comparison table of all results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Variant':<35} {'Cls-F1':<10} {'Map-F1':<10} {'Strict':<10} {'Jacc':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for name, result_data in results.items():
            cls_f1 = result_data['classification_metrics']['f1']
            map_metrics = result_data['mapping_result'].metrics
            elapsed = result_data['elapsed_time']
            
            print(f"{name:<35} {cls_f1:<10.2%} {map_metrics['detection_f1']:<10.2%} "
                  f"{map_metrics['strict_accuracy']:<10.2%} {map_metrics['avg_jaccard']:<10.2%} "
                  f"{elapsed:<10.1f}")
        
        # Best performers
        print("\n" + "=" * 80)
        best_cls = max(results.items(), key=lambda x: x[1]['classification_metrics']['f1'])
        best_map = max(results.items(), key=lambda x: x[1]['mapping_result'].metrics['detection_f1'])
        
        print(f"Best Classification F1: {best_cls[0]} ({best_cls[1]['classification_metrics']['f1']:.2%})")
        print(f"Best Mapping F1:        {best_map[0]} ({best_map[1]['mapping_result'].metrics['detection_f1']:.2%})")
        print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_pipelines.py <config_file>")
        print("\nExample:")
        print("  python run_pipelines.py configs/pipelines/norma_best.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        experiment = PipelineExperiment(config_file)
        experiment.run()
        print("\n✓ Experiment complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
