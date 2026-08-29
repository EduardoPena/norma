#!/usr/bin/env python3
"""
Classifier Experiment Runner

Compares different classifier configurations (LLM, RAG, SLM variants).
Evaluates binary classification: is this sentence a rule or not?

Usage:
    python run_classifiers.py configs/classifiers/all_implementations.yaml
    python run_classifiers.py configs/classifiers/llm_only.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.llm.helper import PromptLoader
from src.evaluation.classification import ClassificationEvaluator
from src.classifiers.llm import LLMRuleClassifier
from experiments.shared.experiment_base import BaseExperiment


class ClassifierExperiment(BaseExperiment):
    """Run classifier comparison experiments."""

    def __init__(self, config_file: str):
        super().__init__(config_file)

        # Validate this is a classifier experiment
        if self.config["experiment_type"] != "classifier":
            raise ValueError(
                f"Expected experiment_type='classifier', "
                f"got '{self.config['experiment_type']}'"
            )

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

    def run(self):
        """Run classifier experiments across all datasets and variants."""
        print("\n" + "=" * 80)
        print(f"CLASSIFIER EXPERIMENT: {self.config['experiment_name']}")
        print("=" * 80)
        print(f"\nResults directory: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("\nDirectory structure:")
        print(f"  config.yaml           (configuration)")
        print(f"  execution_config.csv  (parameters)")
        print(f"  results_summary.csv   (results)")
        print(f"  outputs/              (predictions)")
        print(f"  metrics/              (detailed metrics)")
        print("=" * 80 + "\n")

        # Save configs
        self.save_config_yaml()
        self.save_execution_config()

        # Track progress
        total_combinations = len(self.config["datasets"]) * len(self.config["variants"])
        completed = 0
        failed = 0

        # Run for each dataset
        for dataset_name in self.config["datasets"]:
            try:
                dataset_failed = self.run_dataset(dataset_name)
                failed += dataset_failed
                completed += len(self.config["variants"]) - dataset_failed
            except Exception as e:
                print(f"\nError processing dataset {dataset_name}: {e}")
                traceback.print_exc()
                failed += len(self.config["variants"])

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
        print(f"  {self.results_dir / 'execution_config.csv'}")
        print(f"  {self.outputs_dir}/")
        print(f"  {self.metrics_dir}/")

    def run_dataset(self, dataset_name: str) -> int:
        """
        Run all classifier variants on a single dataset.

        Returns:
            Number of failed variants
        """
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        # Load dataset
        print(f"Loading {dataset_name}...")
        try:

            df, sentences, sentence2dcs_map, predicate_space = self.load_dataset(
                dataset_name
            )
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return len(self.config["variants"])

        num_rules = sum(rules is not None for rules in sentence2dcs_map.values())
        num_total = len(sentences)

        print(f"Total sentences: {num_total}")
        print(f"Rule sentences: {num_rules}")
        print(f"Non-rule sentences: {num_total - num_rules}\n")

        # Create evaluator
        evaluator = ClassificationEvaluator(ground_truth=sentence2dcs_map)

        # Create ground_truth list aligned with sentences
        ground_truth = [
            {"id": i, "sentences": [sentence], "rules": sentence2dcs_map.get(sentence)}
            for i, sentence in enumerate(sentences, 1)
        ]

        # Run each variant
        results = {}
        failed_count = 0

        for i, variant in enumerate(self.config["variants"], 1):
            name = variant["name"]
            print(f"\n[{i}/{len(self.config['variants'])}] Processing variant: {name}")

            try:
                result = self.run_variant(
                    name=name,
                    variant=variant,
                    sentences=sentences,
                    ground_truth=ground_truth,
                    evaluator=evaluator,
                    df=df,
                    dataset_name=dataset_name,
                )
                results[name] = result

                # Track result for summary CSV (saved incrementally)
                self.append_result(
                    {
                        "dataset": dataset_name,
                        "variant_name": name,
                        "implementation": variant["implementation"],
                        "model": variant.get("model", "N/A"),
                        "prompt_key": variant.get("prompt_key", "N/A"),
                        "batch_size": variant.get("batch_size", "N/A"),
                        "use_context": variant.get("use_context", False),
                        "num_rows_context": variant.get("num_rows_context", 3),
                        "num_rules": num_rules,
                        "num_total": num_total,
                        "accuracy": result.accuracy,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1_score": result.f1_score,
                        "true_positives": result.true_positives,
                        "false_positives": result.false_positives,
                        "true_negatives": result.true_negatives,
                        "false_negatives": result.false_negatives,
                        "status": "success",
                    }
                )

                print(f"  Variant completed")

            except Exception as e:
                print(f"  Error: Variant failed: {e}")
                traceback.print_exc()
                failed_count += 1

                # Still track the failure in results
                self.append_result(
                    {
                        "dataset": dataset_name,
                        "variant_name": name,
                        "implementation": variant["implementation"],
                        "model": variant.get("model", "N/A"),
                        "prompt_key": variant.get("prompt_key", "N/A"),
                        "batch_size": variant.get("batch_size", "N/A"),
                        "use_context": variant.get("use_context", False),
                        "num_rows_context": variant.get("num_rows_context", 3),
                        "num_rules": num_rules,
                        "num_total": num_total,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "true_negatives": 0,
                        "false_negatives": 0,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Print comparison table (only for successful variants)
        if results:
            self._print_comparison_table(results)

        return failed_count

    def run_variant(
        self,
        name: str,
        variant: Dict,
        sentences: List[str],
        ground_truth: List[Dict],
        evaluator,
        df,
        dataset_name: str,
    ):
        """Run a single classifier variant."""
        print(f"{'─' * 60}")
        print(f"  Model: {variant.get('model', 'N/A')}")
        print(f"  Implementation: {variant['implementation']}")
        print(f"  Prompt: {variant.get('prompt_key', 'N/A')}")

        # Extract config
        implementation = variant["implementation"]
        model = variant.get("model", "gpt-5-mini")
        prompt_key = variant.get("prompt_key")
        use_context = variant.get("use_context", False)
        num_rows_context = variant.get("num_rows_context", 3)

        # Create classifier based on implementation
        print(f"  Creating classifier...")
        classifier = self._create_classifier(implementation, model, prompt_key, variant)

        # Handle RAG training if needed
        if implementation == "rag":
            training_data = [
                {
                    "sentence": gt_item["sentences"][0] if gt_item["sentences"] else "",
                    "label": gt_item["rules"] is not None and len(gt_item["rules"]) > 0,
                }
                for gt_item in ground_truth
                if gt_item.get("sentences")
            ]
            print(f"  Training RAG with {len(training_data)} examples...")
            classifier.train(training_data)

        # Build context if needed
        context = None
        if use_context:
            print(f"  Building table context (num_rows={num_rows_context})...")
            context = self._build_context(df, dataset_name, num_rows=num_rows_context)
            context_preview = context["table_context"][:100].replace("\n", " ")
            print(f"  Context preview: {context_preview}...")

        # Get predictions
        print(f"  Classifying {len(sentences)} sentences...")
        sentence_predictions = classifier.classify(sentences, context=context)

        # Format predictions for evaluator (list of tuples)
        predictions = sentence_predictions

        # Evaluate
        print(f"  Evaluating predictions...")
        metrics, sentence_results = evaluator.evaluate(predictions)

        # Prepare detailed results with examples
        detailed_metrics = self._prepare_detailed_metrics(
            metrics, sentence_predictions, ground_truth
        )

        # Save outputs (predictions) - convert to dict format
        print(f"  Saving outputs...")
        predictions_dict = [
            {
                "id": gt_item["id"],
                "sentence": sentence,
                "predicted_is_rule": pred_is_rule,
                "gold_is_rule": gt_item["rules"] is not None
                and len(gt_item["rules"]) > 0,
            }
            for (sentence, pred_is_rule), gt_item in zip(predictions, ground_truth)
        ]
        outputs_file = self.save_outputs(
            outputs=predictions_dict,
            dataset_name=dataset_name,
            variant_name=name,
            output_type="predictions",
        )

        # Save detailed metrics
        metrics_file = self.save_detailed_metrics(
            metrics_data=detailed_metrics, dataset_name=dataset_name, variant_name=name
        )

        # Display metrics
        print(f"\n  Metrics:")
        print(f"     Accuracy:  {metrics.accuracy:.3f}")
        print(f"     Precision: {metrics.precision:.3f}")
        print(f"     Recall:    {metrics.recall:.3f}")
        print(f"     F1 Score:  {metrics.f1_score:.3f}")

        print(f"\n  Confusion Matrix:")
        print(
            f"     TP: {metrics.true_positives:3d}  |  FP: {metrics.false_positives:3d}"
        )
        print(
            f"     FN: {metrics.false_negatives:3d}  |  TN: {metrics.true_negatives:3d}"
        )

        print(f"\n  Output: {outputs_file.name}")
        print(f"  Metrics: {metrics_file.name}")
        print(f"  Summary: {self.results_summary_path.name}")

        return metrics

    def _create_classifier(
        self, implementation: str, model: str, prompt_key: Optional[str], variant: Dict
    ):
        """Factory method to create classifier based on implementation type."""

        if implementation == "llm":
            # Load prompt template
            if not prompt_key:
                raise ValueError("LLM implementation requires 'prompt_key'")

            if "_pt" in prompt_key:
                prompt_template = self.prompt_loader.load(
                    "classification_pt", prompt_key
                )
            else:
                prompt_template = self.prompt_loader.load("classification", prompt_key)

            return LLMRuleClassifier(
                model_name=model,
                provider=variant.get("provider"),
                prompt_template=prompt_template,
                chunk_size=variant.get("batch_size", 20),
            )

        elif implementation == "rag":
            # RAG classifier (not yet implemented)
            raise NotImplementedError("RAG classifier not yet implemented")

        elif implementation == "slm":
            # SLM classifier (not yet implemented)
            raise NotImplementedError("SLM classifier not yet implemented")

        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def _build_context(
        self, df, dataset_name: str, num_rows: int = 3
    ) -> Dict[str, Any]:
        """Build table context for classifiers."""
        from src.utils.table_context import TableContextBuilder

        context_builder = TableContextBuilder(sample_strategy="head", num_rows=num_rows)
        return context_builder.build(df)

    def _prepare_detailed_metrics(self, metrics, sentence_predictions, ground_truth):
        """Prepare detailed metrics with examples for each category."""
        metrics_dict = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "true_negatives": metrics.true_negatives,
            "false_negatives": metrics.false_negatives,
        }

        # Categorize predictions
        categorized = {"TP": [], "FP": [], "FN": [], "TN": []}

        for (sentence, pred_is_rule), gt_item in zip(
            sentence_predictions, ground_truth
        ):
            gold_is_rule = gt_item["rules"] is not None and len(gt_item["rules"]) > 0

            if pred_is_rule and gold_is_rule:
                category = "TP"
            elif pred_is_rule and not gold_is_rule:
                category = "FP"
            elif not pred_is_rule and gold_is_rule:
                category = "FN"
            else:
                category = "TN"

            categorized[category].append(
                {
                    "id": gt_item["id"],
                    "sentence": sentence,
                    "predicted": pred_is_rule,
                    "gold": gold_is_rule,
                    "gold_rules": gt_item.get("rules", []),
                }
            )

        # Take first 10 examples of each category
        examples = {cat: items[:10] for cat, items in categorized.items()}

        return {
            "metrics": metrics_dict,
            "confusion_matrix": {
                "true_positives": metrics_dict["true_positives"],
                "false_positives": metrics_dict["false_positives"],
                "true_negatives": metrics_dict["true_negatives"],
                "false_negatives": metrics_dict["false_negatives"],
            },
            "counts_by_category": {
                cat: len(items) for cat, items in categorized.items()
            },
            "examples_by_category": examples,
        }

    def _print_comparison_table(self, results: Dict):
        """Print comparison table of all results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Variant':<40} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10}")
        print("-" * 80)

        for name, result in results.items():
            print(
                f"{name:<40} {result.accuracy:<10.3f} {result.precision:<10.3f} "
                f"{result.recall:<10.3f} {result.f1_score:<10.3f}"
            )

        # Best performers
        if results:
            print("\n" + "=" * 80)
            best_f1 = max(results.items(), key=lambda x: x[1].f1_score)
            best_acc = max(results.items(), key=lambda x: x[1].accuracy)
            print(f"Best F1:       {best_f1[0]} ({best_f1[1].f1_score:.3f})")
            print(f"Best Accuracy: {best_acc[0]} ({best_acc[1].accuracy:.3f})")
            print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run classifier experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_classifiers.py configs/classifiers/all_implementations.yaml
  python run_classifiers.py configs/classifiers/llm_only.yaml
  python run_classifiers.py configs/classifiers/context_ablation.yaml
        """,
    )
    parser.add_argument("config_file", help="Path to YAML config file")

    args = parser.parse_args()

    try:
        experiment = ClassifierExperiment(args.config_file)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
