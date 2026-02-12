#!/usr/bin/env python3
"""
Mapper Experiment Runner

Compares different mapper configurations on rule sentences only.
Tests constraint extraction: sentence → denial constraint predicates.

Usage:
    python run_mappers.py configs/mappers/prompt_variants.yaml
    python run_mappers.py configs/mappers/batch_size_study.yaml
"""

import sys
from pathlib import Path
import csv
import time
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.experiment_base import BaseExperiment
from src.mappers import PromptMapper
from src.metrics.mapping import OneToOneMappingEvaluator


class MapperExperiment(BaseExperiment):
    """Run mapper comparison experiments."""
    
    def __init__(self, config_file: str):
        super().__init__(config_file)
        
        # Validate this is a mapper experiment
        if self.config['experiment_type'] != 'mapper':
            raise ValueError(f"Expected experiment_type='mapper', got '{self.config['experiment_type']}'")
    
    def run(self):
        """Run mapper experiments across all datasets and variants."""
        self.print_header(f"MAPPER EXPERIMENT: {self.config['experiment_name']}")
        
        # Save config copy
        self.save_config_copy()
        
        # Run for each dataset
        for dataset_name in self.config['datasets']:
            self.run_dataset(dataset_name)
    
    def run_dataset(self, dataset_name: str):
        """Run all mapper variants on a single dataset."""
        self.print_subheader(f"Dataset: {dataset_name}")
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        df, predicate_space, ground_truth, dataframes = self.load_dataset(dataset_name)
        
        # Filter to rule sentences only
        ground_truth_rules = [item for item in ground_truth if item['rules'] is not None]
        
        print(f"Total sentences: {len(ground_truth)}")
        print(f"Rule sentences (testing on): {len(ground_truth_rules)}\n")
        
        # Prepare CSV
        csv_filename = self.get_csv_filename(dataset_name, "mapper_results")
        
        # Run each variant
        results = {}
        for variant in self.config['variants']:
            name = variant['name']
            result_data = self.run_variant(
                name=name,
                variant=variant,
                ground_truth=ground_truth_rules,
                predicate_space=predicate_space,
                df=df
            )
            results[name] = result_data
            
            # Save to CSV
            self.save_result_to_csv(
                csv_filename, dataset_name, name, variant, 
                result_data, len(ground_truth_rules)
            )
        
        # Print comparison table
        self.print_comparison_table(results)
        
        print(f"\n✓ Results saved to: {csv_filename}")
    
    def run_variant(self, name: str, variant: Dict, ground_truth: List[Dict],
                   predicate_space, df):
        """Run a single mapper variant."""
        print(f"\nRunning: {name}")
        print(f"  Model: {variant['model']}")
        print(f"  Prompt: {variant.get('prompt_key', 'basic')}")
        print(f"  Batch size: {variant.get('batch_size', 1)}")
        
        # Create mapper
        mapper = PromptMapper(
            predicate_space=predicate_space,
            df=df,
            model_name=variant['model'],
            temperature=variant.get('temperature', 0.0),
            prompt_key=variant.get('prompt_key', 'basic'),
            batch_size=variant.get('batch_size', 1)
        )
        
        # Extract sentences
        sentences = [gt_item['sentences'][0] for gt_item in ground_truth]
        
        # Track mapping time
        print(f"  Mapping {len(sentences)} sentences...")
        start_time = time.time()
        
        # Map all sentences
        pred_rules_list = mapper.map_multiple(sentences)
        
        mapping_time = time.time() - start_time
        
        # Format predictions
        predictions = []
        for gt_item, pred_rules in zip(ground_truth, pred_rules_list):
            predictions.append({'id': gt_item['id'], 'rules': pred_rules})
        
        # Evaluate
        evaluator = OneToOneMappingEvaluator(ground_truth, df=df)
        result = evaluator.evaluate(predictions)
        
        # Display metrics
        metrics = result.metrics
        print(f"  Detection F1:  {metrics['detection_f1']:.2%}")
        print(f"  Strict Acc:    {metrics['strict_accuracy']:.2%}")
        print(f"  Avg Jaccard:   {metrics['avg_jaccard']:.2%}")
        if 'avg_db_execution_jaccard' in metrics:
            print(f"  DB Jaccard:    {metrics['avg_db_execution_jaccard']:.2%}")
        print(f"  Time:          {mapping_time:.2f}s")
        
        return {
            'result': result,
            'mapping_time': mapping_time
        }
    
    def save_result_to_csv(self, csv_filename: Path, dataset_name: str,
                          variant_name: str, variant_config: Dict,
                          result_data: Dict, num_rules: int):
        """Save a single result to CSV."""
        file_exists = csv_filename.exists()
        
        result = result_data['result']
        mapping_time = result_data['mapping_time']
        metrics = result.metrics
        
        with open(csv_filename, 'a', newline='') as f:
            fieldnames = [
                'dataset', 'variant_name', 'model', 'prompt_key', 
                'temperature', 'batch_size', 'num_rules',
                'detection_precision', 'detection_recall', 'detection_f1',
                'strict_accuracy', 'avg_jaccard', 'avg_soft_jaccard',
                'db_execution_accuracy', 'avg_db_execution_jaccard',
                'mapping_time_seconds'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'dataset': dataset_name,
                'variant_name': variant_name,
                'model': variant_config['model'],
                'prompt_key': variant_config.get('prompt_key', 'basic'),
                'temperature': variant_config.get('temperature', 0.0),
                'batch_size': variant_config.get('batch_size', 1),
                'num_rules': num_rules,
                'detection_precision': f"{metrics['detection_precision']:.4f}",
                'detection_recall': f"{metrics['detection_recall']:.4f}",
                'detection_f1': f"{metrics['detection_f1']:.4f}",
                'strict_accuracy': f"{metrics['strict_accuracy']:.4f}",
                'avg_jaccard': f"{metrics['avg_jaccard']:.4f}",
                'avg_soft_jaccard': f"{metrics.get('avg_soft_jaccard', 0.0):.4f}",
                'db_execution_accuracy': f"{metrics.get('db_execution_accuracy', 0.0):.4f}",
                'avg_db_execution_jaccard': f"{metrics.get('avg_db_execution_jaccard', 0.0):.4f}",
                'mapping_time_seconds': f"{mapping_time:.2f}"
            })
    
    def print_comparison_table(self, results: Dict):
        """Print comparison table of all results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Variant':<40} {'Det-F1':<10} {'Strict':<10} {'Jaccard':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for name, result_data in results.items():
            m = result_data['result'].metrics
            t = result_data['mapping_time']
            print(f"{name:<40} {m['detection_f1']:<10.2%} "
                  f"{m['strict_accuracy']:<10.2%} {m['avg_jaccard']:<10.2%} {t:<10.2f}")
        
        # Best performers
        print("\n" + "=" * 80)
        best_f1 = max(results.items(), key=lambda x: x[1]['result'].metrics['detection_f1'])
        best_strict = max(results.items(), key=lambda x: x[1]['result'].metrics['strict_accuracy'])
        best_jaccard = max(results.items(), key=lambda x: x[1]['result'].metrics['avg_jaccard'])
        
        print(f"Best Detection F1:    {best_f1[0]} ({best_f1[1]['result'].metrics['detection_f1']:.2%})")
        print(f"Best Strict Accuracy: {best_strict[0]} ({best_strict[1]['result'].metrics['strict_accuracy']:.2%})")
        print(f"Best Avg Jaccard:     {best_jaccard[0]} ({best_jaccard[1]['result'].metrics['avg_jaccard']:.2%})")
        print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_mappers.py <config_file>")
        print("\nExample:")
        print("  python run_mappers.py configs/mappers/prompt_variants.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        experiment = MapperExperiment(config_file)
        experiment.run()
        print("\n✓ Experiment complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
