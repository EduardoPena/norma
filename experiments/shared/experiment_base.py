"""Base class for all experiments with shared functionality."""

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from src.predicates import normalize_rules


def _normalize_sentence_rules(
    rules: Any,
    *,
    drop_multi_rule_sentences: bool = False,
) -> Any:
    """Validate GT rules and convert only the legacy nested-list representation."""
    if rules is None:
        return None

    normalized = normalize_rules(rules)
    if drop_multi_rule_sentences and normalized and len(normalized) > 1:
        return None
    return normalized


class BaseExperiment:
    """Base class providing common experiment functionality."""

    def __init__(self, config_file: str):
        """
        Initialize experiment with config file.

        Args:
            config_file: Path to YAML config file
        """
        self.config_file = Path(config_file)
        self.config = self.load_config()

        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup results directory: {experiment_name}_{timestamp}
        experiment_name = self.config.get("experiment_name", "unnamed_experiment")
        safe_name = experiment_name.replace(" ", "_").lower()
        folder_name = f"{safe_name}_{self.timestamp}"

        # Get base directory from config (default: 'results')
        results_base_dir = self.config.get("results_dir", "results")
        base_dir = Path(results_base_dir)

        self.results_dir = base_dir / folder_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track all results
        self.all_results = []

        # Results subdirectories
        self.outputs_dir = self.results_dir / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.results_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # Path to incremental results CSV
        self.results_summary_path = self.results_dir / "results_summary.csv"

    def load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["experiment_type", "datasets", "variants"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must have '{field}' field")

        return config

    def load_dataset(self, dataset_name: str):
        """
        Load dataset using DatasetLoader.
        Returns:
            tuple: (df, sentences, sentence2dcs_map, predicate_space)
        """
        from src.data import DatasetLoader

        data_dir = Path("data") / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")

        dataset_options = (self.config.get("dataset_options") or {}).get(
            dataset_name, {}
        )
        loader = DatasetLoader(
            str(data_dir),
            sample_rows=dataset_options.get("sample_rows"),
            gt_file=dataset_options.get("gt_file"),
            stream_csv=dataset_options.get("stream_csv", False),
            profile_rows=dataset_options.get("profile_rows"),
        )
        dataframes, predicate_space, ground_truth = loader.load()

        if not dataframes:
            raise ValueError(f"Dataset has no tables: {data_dir}")

        if len(dataframes) != 1:
            table_names = ", ".join(sorted(dataframes.keys()))
            raise ValueError(
                f"Expected exactly 1 table for experiments, got {len(dataframes)}: {table_names}"
            )

        df = next(iter(dataframes.values()))
        sentences: List[str] = []
        sentence2dcs_map: Dict[str, Any] = {}
        drop_multi_rule_sentences = dataset_options.get(
            "drop_multi_rule_sentences", False
        )

        for item in ground_truth:
            item_sentences = item.get("sentences") or []
            rules = _normalize_sentence_rules(
                item.get("rules"),
                drop_multi_rule_sentences=drop_multi_rule_sentences,
            )

            for sentence in item_sentences:
                sentences.append(sentence)
                sentence2dcs_map[sentence] = rules

        predicate_space_dict = {
            "quantifiers": dict(predicate_space.quantifiers),
            "_table_files": {
                table_name: str(path) for table_name, path in loader.table_files.items()
            },
            "_stream_csv": dataset_options.get("stream_csv", False),
            "_sample_rows": dataset_options.get("sample_rows"),
            "tables": [
                {
                    "name": table_name,
                    "columns": [
                        {
                            "name": column_name,
                            "type": column_spec["type"],
                            "allowed_ops": list(column_spec["allowed_ops"]),
                            **(
                                {"description": column_spec["description"]}
                                if isinstance(column_spec.get("description"), str)
                                and column_spec["description"].strip()
                                else {}
                            ),
                        }
                        for column_name, column_spec in table_schema.items()
                    ],
                }
                for table_name, table_schema in predicate_space.schema.items()
            ],
            "predicates": sorted(predicate_space.allowed_predicates),
        }

        return df, sentences, sentence2dcs_map, predicate_space_dict

    def save_execution_config(self):
        """
        Save execution configuration as CSV.
        This shows what was run: datasets, variants, parameters.
        """
        config_rows = []

        # Add general info
        config_rows.append(
            {
                "parameter": "experiment_name",
                "value": self.config.get("experiment_name", "N/A"),
            }
        )
        config_rows.append(
            {
                "parameter": "experiment_type",
                "value": self.config.get("experiment_type", "N/A"),
            }
        )
        config_rows.append(
            {
                "parameter": "results_dir",
                "value": self.config.get("results_dir", "results"),
            }
        )
        config_rows.append({"parameter": "timestamp", "value": self.timestamp})

        # Add datasets
        datasets = self.config.get("datasets", [])
        config_rows.append({"parameter": "datasets", "value": ", ".join(datasets)})
        config_rows.append({"parameter": "num_datasets", "value": len(datasets)})

        # Add variants info
        variants = self.config.get("variants", [])
        config_rows.append({"parameter": "num_variants", "value": len(variants)})

        # Add each variant's details
        for i, variant in enumerate(variants, 1):
            prefix = f"variant_{i}"
            config_rows.append(
                {"parameter": f"{prefix}_name", "value": variant.get("name", "N/A")}
            )
            config_rows.append(
                {
                    "parameter": f"{prefix}_implementation",
                    "value": variant.get("implementation", "N/A"),
                }
            )
            config_rows.append(
                {"parameter": f"{prefix}_model", "value": variant.get("model", "N/A")}
            )
            for key, value in variant.items():
                if key not in ["name", "implementation", "model"]:
                    config_rows.append(
                        {"parameter": f"{prefix}_{key}", "value": str(value)}
                    )

        # Save to CSV
        config_file = self.results_dir / "execution_config.csv"
        df = pd.DataFrame(config_rows)
        df.to_csv(config_file, index=False)

        print(f"Execution config saved to: {config_file}")
        return config_file

    def save_config_yaml(self):
        """Save a copy of the original YAML config file."""
        config_copy = self.results_dir / "config.yaml"
        with open(config_copy, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Config YAML saved to: {config_copy}")
        return config_copy

    def save_outputs(
        self,
        outputs: List[Dict],
        dataset_name: str,
        variant_name: str,
        output_type: str = "predictions",
    ):
        """
        Save outputs (predictions, generated rules, etc.) to JSON file.

        Args:
            outputs: List of output dictionaries
            dataset_name: Name of the dataset
            variant_name: Name of the variant (will be sanitized)
            output_type: Type of output (e.g., 'predictions', 'rules', 'constraints')

        Returns:
            Path to saved file
        """
        # Sanitize variant name for filename
        safe_variant = (
            variant_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
        )

        filename = (
            self.outputs_dir / f"{dataset_name}_{safe_variant}_{output_type}.json"
        )

        with open(filename, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "variant": variant_name,
                    "output_type": output_type,
                    "timestamp": self.timestamp,
                    "count": len(outputs),
                    "outputs": outputs,
                },
                f,
                indent=2,
            )

        return filename

    def save_detailed_metrics(
        self, metrics_data: Dict[str, Any], dataset_name: str, variant_name: str
    ):
        """
        Save detailed metrics including confusion matrix and examples to JSON.

        Args:
            metrics_data: Dictionary with metrics, confusion matrix, examples
            dataset_name: Name of the dataset
            variant_name: Name of the variant

        Returns:
            Path to saved file
        """
        safe_variant = (
            variant_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
        )

        filename = self.metrics_dir / f"{dataset_name}_{safe_variant}_detailed.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "variant": variant_name,
                    "timestamp": self.timestamp,
                    **metrics_data,
                },
                f,
                indent=2,
            )

        return filename

    def append_result(self, result_dict: Dict[str, Any]):
        """
        Add a result to the tracked results and save incrementally.

        Args:
            result_dict: Dictionary containing result metrics
        """
        self.all_results.append(result_dict)

        # Save incrementally after each result
        self._save_results_incremental()

    def _save_results_incremental(self):
        """
        Internal method to save results incrementally.
        Overwrites CSV each time to keep it simple and avoid duplicates.
        """
        if not self.all_results:
            return

        # Convert current results to DataFrame
        df = pd.DataFrame(self.all_results)

        # Save (overwrite each time)
        df.to_csv(self.results_summary_path, index=False, float_format="%.4f")

    def save_results_summary(self):
        """
        Save all tracked results to summary CSV file.
        This is called at the end, but results are also saved incrementally.

        Returns:
            Path to saved file
        """
        if not self.all_results:
            print("Warning: No results to save")
            return None

        # Convert to DataFrame for easy CSV writing
        df = pd.DataFrame(self.all_results)
        df.to_csv(self.results_summary_path, index=False, float_format="%.4f")

        print(f"Results summary saved to: {self.results_summary_path}")
        return self.results_summary_path

    def run(self):
        """Run the experiment. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
