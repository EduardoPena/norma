"""Dataset loading from NORMA predicate, ground-truth, and CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.predicates import PredicateSpace


class DatasetLoader:
    """Load a dataset directory using its predicate specification."""

    def __init__(
        self,
        data_dir: str,
        sample_rows: Optional[int] = None,
        gt_file: Optional[str] = None,
        stream_csv: bool = False,
        profile_rows: Optional[int] = None,
        space_version: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rows = sample_rows
        self.gt_file = gt_file
        self.stream_csv = stream_csv
        self.profile_rows = profile_rows
        # 1: *_predicates.json (all templates); 2: *_schema.json (schema only,
        # predicates generated on demand, see src/predicates/on_demand.py)
        if space_version not in (1, 2):
            raise ValueError(f"Unknown predicate space version: {space_version}")
        self.space_version = space_version
        self.table_files: dict[str, Path] = {}

    def load(
        self,
    ) -> tuple[dict[str, pd.DataFrame], PredicateSpace, list[dict[str, Any]]]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        predicate_path = self._single_matching_file(
            "*_schema.json" if self.space_version == 2 else "*_predicates.json"
        )
        with predicate_path.open(encoding="utf-8") as file:
            predicate_spec = json.load(file)
        predicate_space = PredicateSpace.from_spec(predicate_spec)

        dataframes: dict[str, pd.DataFrame] = {}
        for table_name in predicate_space.schema:
            csv_path = self._resolve_table_file(table_name)
            self.table_files[table_name] = csv_path
            dataframe = pd.read_csv(
                csv_path,
                nrows=self.sample_rows,
                low_memory=False,
            )
            dataframe.attrs["declared_column_types"] = {
                column_name: column_spec["type"]
                for column_name, column_spec in predicate_space.schema[
                    table_name
                ].items()
            }
            dataframes[table_name] = dataframe

        ground_truth_path = (
            self.data_dir / self.gt_file
            if self.gt_file
            else self._single_matching_file("*_text2rule_gt.json")
        )
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_path}")
        with ground_truth_path.open(encoding="utf-8") as file:
            ground_truth_payload = json.load(file)

        ground_truth = ground_truth_payload.get("mapping")
        if not isinstance(ground_truth, list):
            raise ValueError(
                f"Ground-truth file must contain a mapping list: {ground_truth_path}"
            )
        return dataframes, predicate_space, ground_truth

    def _single_matching_file(self, pattern: str) -> Path:
        matches = sorted(self.data_dir.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {pattern} file in {self.data_dir}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def _resolve_table_file(self, table_name: str) -> Path:
        exact_path = self.data_dir / f"{table_name}.csv"
        if exact_path.is_file():
            return exact_path

        csv_files = sorted(self.data_dir.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]
        raise FileNotFoundError(
            f"Could not resolve CSV for table {table_name!r} in {self.data_dir}"
        )
