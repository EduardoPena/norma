#!/usr/bin/env python3



import sys
import time
import argparse
from pathlib import Path
from turtle import pd
from typing import Dict, List, Optional, Tuple
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))


_NA = 'N/A'


from src.evaluation.mapping import MappingEvaluator
from src.evaluation.classification import ClassificationEvaluator


class Text2SQLMapperEval():

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def load_dataset(self, dataset_name: str):
        """
        Load dataset using DatasetLoader.
        Returns:
            tuple: (df, predicate_space, ground_truth, dataframes)
        """
        from src.data import DatasetLoader
        import json

        data_dir = Path("data") / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")

        # Load using DatasetLoader (returns: df, sentences, sentence2dcs_map, predicate_space)
        loader = DatasetLoader(str(data_dir))
        df, sentences, sentence2dcs_map, predicate_space = loader.load()

        return df, sentences, sentence2dcs_map, predicate_space
    
    def load_baseline(self, dataset_name: str):
        """
        Load dataset using DatasetLoader.
        Returns:
            dict: {sentence: pred_sql or None}
        """
        from pathlib import Path
        import pandas as pd

        data_dir = Path("baseline_results/text2sql_results") / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")

        df = pd.read_csv(data_dir / f"{dataset_name}_xiyan_detailed_results.csv")

        # Map each sentence to its pred_sql (None if empty/NaN)
        sentence_to_pred_sql = {
            row["sentence"]: row["pred_sql"] if pd.notna(row["pred_sql"]) and row["pred_sql"] != "" else None
            for _, row in df.iterrows()
        }

        return sentence_to_pred_sql

    def run(self):

        print("-" * 80)

        print(f"Loading {self.dataset_id}...")
        try:
            df, sentences, sentence2dcs_map, predicate_space = self.load_dataset(
                self.dataset_id)
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return 0

        gt_sentence2dcs_map = {s: dcs for s,
                               dcs in sentence2dcs_map.items() if dcs}
        num_total = len(sentences)
        num_gt = len(gt_sentence2dcs_map)

        print(f"Total sentences:          {num_total}")
        print(f"GT constraint sentences:  {num_gt}")
        print(f"Non-constraint sentences: {num_total - num_gt}\n")

        baseline_sentence2pred_sql = self.load_baseline(self.dataset_id)

        for sentence in sentences:
            print(f"Sentence: {sentence} -> Baseline SQL: {baseline_sentence2pred_sql.get(sentence, 'N/A')}")
            


# =============================================================================

def main():
    dataset_id = "tax"

    try:
        experiment = Text2SQLMapperEval(dataset_id)
        experiment.run()
        print("\nExperiment complete")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting text 2 sql mapper evaluation experiment...")
    main()
