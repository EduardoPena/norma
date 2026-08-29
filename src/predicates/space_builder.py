"""
PredicateSpaceBuilder: Generates predicate space from CSV data.

Supports:
- Cross-tuple predicates: t1.col = t2.col, t1.col < t2.col
- Constant predicates: t1.col = $dtype, t1.col > $dtype
- String patterns: t1.col LIKE $pattern, t1.col NOT LIKE $pattern
- Categorical sets: t1.col IN $values, t1.col NOT IN $values
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from typing import Dict, Set, Optional
import json
import os
import re


class PredicateSpaceBuilder:
    """Builds predicate space from a DataFrame."""

    def __init__(
        self,
        table_alias: str = "t",
        categorical_threshold: int = 10,
        cross_column_overlap: float = 0.3,
        min_distinct_for_cross: int = 3,
        enable_same_type_comparisons: bool = True,
    ):
        """
        Args:
            table_alias: Prefix for tuple variables (e.g., "t" -> t1, t2)
            categorical_threshold: Max distinct values for IN/NOT IN. -1 to disable.
            cross_column_overlap: Min overlap ratio to consider columns joinable.
            min_distinct_for_cross: Min distinct values to consider cross-column (avoids binary).
            enable_same_type_comparisons: Allow comparisons between same-type columns (dates, timestamps, etc.) without overlap requirement.
        """
        self.table_alias = table_alias
        self.categorical_threshold = categorical_threshold
        self.cross_column_overlap = cross_column_overlap
        self.min_distinct_for_cross = min_distinct_for_cross
        self.enable_same_type_comparisons = enable_same_type_comparisons
        self.t1 = f"{table_alias}1"
        self.t2 = f"{table_alias}2"

    def _read_csv_sample(
        self,
        csv_path: str,
        sample_rows: Optional[int] = None,
        chunksize: int = 50000,
    ) -> pd.DataFrame:
        """Read a bounded sample from a CSV, avoiding full-file loads for large datasets."""
        if sample_rows is None:
            return pd.read_csv(csv_path, low_memory=False)

        if sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer")

        frames = []
        rows_read = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=min(chunksize, sample_rows),
            low_memory=False,
        ):
            remaining = sample_rows - rows_read
            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

            frames.append(chunk)
            rows_read += len(chunk)

            if rows_read >= sample_rows:
                break

        if not frames:
            return pd.read_csv(csv_path, nrows=0, low_memory=False)

        return pd.concat(frames, ignore_index=True)

    def build_predicates(self, df: pd.DataFrame) -> Set[str]:
        """Generate all predicates for the DataFrame."""
        predicates = set()

        for col in df.columns:
            n_distinct = df[col].dropna().nunique()

            # Determine simplified dtype
            if is_numeric_dtype(df[col]):
                # Check if integer or float
                if df[col].dtype in ["int64", "int32", "int16", "int8"]:
                    dtype = "int"
                else:
                    dtype = "float"
            else:
                dtype = "string"

            # Cross-tuple equality (all columns)
            predicates.add(f"{self.t1}.{col} = {self.t2}.{col}")
            predicates.add(f"{self.t1}.{col} != {self.t2}.{col}")

            # Constant equality (all columns)
            predicates.add(f"{self.t1}.{col} = ${dtype}")
            predicates.add(f"{self.t1}.{col} != ${dtype}")

            # is null and is not null (all columns)
            predicates.add(f"{self.t1}.{col} IS NULL")
            predicates.add(f"{self.t1}.{col} IS NOT NULL")

            if is_numeric_dtype(df[col]):
                # Cross-tuple comparison
                for op in ["<", ">", "<=", ">="]:
                    predicates.add(f"{self.t1}.{col} {op} {self.t2}.{col}")

                # Constant comparison
                for op in ["<", ">", "<=", ">="]:
                    predicates.add(f"{self.t1}.{col} {op} ${dtype}")

            elif is_string_dtype(df[col]) or df[col].dtype == object:
                # LIKE / NOT LIKE for all strings
                predicates.add(f"{self.t1}.{col} LIKE $pattern")
                predicates.add(f"{self.t1}.{col} NOT LIKE $pattern")

                # IN / NOT IN for categorical only
                if (
                    self.categorical_threshold != -1
                    and n_distinct <= self.categorical_threshold
                ):
                    predicates.add(f"{self.t1}.{col} IN $values")
                    predicates.add(f"{self.t1}.{col} NOT IN $values")

        # Cross-column predicates (same tuple, different columns)
        predicates.update(self._build_cross_column_predicates(df))

        return predicates

    def _get_column_overlap(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Calculate overlap ratio between two columns' domains."""
        vals1 = set(df[col1].dropna().unique())
        vals2 = set(df[col2].dropna().unique())

        if not vals1 or not vals2:
            return 0.0

        intersection = vals1 & vals2
        smaller = min(len(vals1), len(vals2))

        return len(intersection) / smaller if smaller > 0 else 0.0

    def _detect_column_semantic_type(self, col_name: str) -> Optional[str]:
        """
        Detect semantic type from column name patterns.

        Returns:
            'date', 'timestamp', 'time', 'location', 'coordinate', or None
        """
        col_lower = col_name.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", col_lower) if token]

        # Date/time patterns
        date_patterns = ["date", "dt", "day"]
        timestamp_patterns = ["timestamp", "ts", "datetime"]
        time_patterns = ["time", "hour", "minute", "second"]

        # Location patterns
        location_patterns = [
            "location",
            "address",
            "city",
            "state",
            "country",
            "zip",
            "postal",
        ]
        coordinate_patterns = ["lat", "lon", "lng", "latitude", "longitude", "coord"]

        # Check patterns
        if any(pattern in tokens for pattern in timestamp_patterns):
            return "timestamp"
        if any(pattern in tokens for pattern in date_patterns):
            return "date"
        if any(pattern in tokens for pattern in time_patterns):
            return "time"
        if any(pattern in tokens for pattern in coordinate_patterns):
            return "coordinate"
        if any(pattern in tokens for pattern in location_patterns):
            return "location"

        return None

    def _build_cross_column_predicates(self, df: pd.DataFrame) -> Set[str]:
        """Build predicates comparing different columns within same tuple."""
        predicates = set()
        columns = list(df.columns)

        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1 :]:
                # Skip if either column is binary
                n_distinct1 = df[col1].dropna().nunique()
                n_distinct2 = df[col2].dropna().nunique()

                if (
                    n_distinct1 < self.min_distinct_for_cross
                    or n_distinct2 < self.min_distinct_for_cross
                ):
                    continue

                # Check type compatibility
                both_numeric = is_numeric_dtype(df[col1]) and is_numeric_dtype(df[col2])
                both_string = (
                    is_string_dtype(df[col1]) or df[col1].dtype == object
                ) and (is_string_dtype(df[col2]) or df[col2].dtype == object)

                if not (both_numeric or both_string):
                    continue

                # Check semantic types (date, timestamp, location, etc.)
                sem_type1 = (
                    self._detect_column_semantic_type(col1)
                    if self.enable_same_type_comparisons
                    else None
                )
                sem_type2 = (
                    self._detect_column_semantic_type(col2)
                    if self.enable_same_type_comparisons
                    else None
                )
                same_semantic_type = sem_type1 and sem_type2 and sem_type1 == sem_type2

                # Check domain overlap (skip for same semantic types)
                if same_semantic_type:
                    # Allow comparisons between same semantic types without overlap check
                    overlap_ok = True
                else:
                    overlap = self._get_column_overlap(df, col1, col2)
                    overlap_ok = overlap >= self.cross_column_overlap

                if not overlap_ok:
                    continue

                # Add cross-column predicates (same tuple) - both orderings for symmetric ops
                predicates.add(f"{self.t1}.{col1} = {self.t1}.{col2}")
                predicates.add(f"{self.t1}.{col1} != {self.t1}.{col2}")
                predicates.add(f"{self.t1}.{col2} = {self.t1}.{col1}")
                predicates.add(f"{self.t1}.{col2} != {self.t1}.{col1}")

                if both_numeric or same_semantic_type:
                    # Allow ordering comparisons for numeric or same semantic types (e.g., dates)
                    # Generate both orderings since <> are opposites but both useful
                    for op in ["<", ">", "<=", ">="]:
                        predicates.add(f"{self.t1}.{col1} {op} {self.t1}.{col2}")
                        predicates.add(f"{self.t1}.{col2} {op} {self.t1}.{col1}")

        return predicates

    def build_schema(self, df: pd.DataFrame) -> Dict:
        """Build schema definition for the DataFrame."""
        columns = []
        for col in df.columns:
            n_distinct = df[col].dropna().nunique()

            # Determine simplified dtype
            if is_numeric_dtype(df[col]):
                if df[col].dtype in ["int64", "int32", "int16", "int8"]:
                    col_type = "int"
                else:
                    col_type = "float"
                allowed_ops = [
                    "=",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "IS NULL",
                    "IS NOT NULL",
                ]
            else:
                col_type = "string"
                # Base string operations
                allowed_ops = ["=", "!=", "LIKE", "NOT LIKE", "IS NULL", "IS NOT NULL"]

                # Add IN/NOT IN for categorical columns
                if (
                    self.categorical_threshold != -1
                    and n_distinct <= self.categorical_threshold
                ):
                    allowed_ops.extend(["IN", "NOT IN"])

            col_info = {"name": col, "type": col_type, "allowed_ops": allowed_ops}
            columns.append(col_info)
        return columns

    def build(
        self,
        csv_path: str,
        save: bool = True,
        output_path: Optional[str] = None,
        sample_rows: Optional[int] = None,
        chunksize: int = 50000,
    ) -> Dict:
        """
        Build predicate space from CSV file.

        Args:
            csv_path: Path to CSV file
            save: Whether to save the JSON spec
            output_path: Custom output path. If None, uses csv_path with _predicate_space.json
            sample_rows: Optional max number of rows to inspect. Uses chunked reads when set.
            chunksize: Chunk size used while sampling large CSVs.

        Returns:
            Dict with quantifiers, tables, and predicates
        """
        df = self._read_csv_sample(
            csv_path,
            sample_rows=sample_rows,
            chunksize=chunksize,
        )
        table_name = os.path.basename(csv_path).replace(".csv", "")

        predicates = self.build_predicates(df)
        schema = self.build_schema(df)

        spec = {
            "quantifiers": {
                self.t1: table_name,
                self.t2: table_name,
            },
            "tables": [
                {
                    "name": table_name,
                    "columns": schema,
                    "file": os.path.basename(csv_path),
                }
            ],
            "predicates": sorted(list(predicates)),
        }

        if save:
            if output_path is None:
                output_path = csv_path.replace(".csv", "_predicates.json")

            if os.path.exists(output_path):
                print(f"Warning: File {output_path} already exists. Overwriting.")
                # confirm = input(f"File {output_path} exists. Overwrite? (y/N): ").lower()
                # if confirm != "y":
                #     print("Skipped saving.")
                #     return spec

            with open(output_path, "w") as f:
                json.dump(spec, f, indent=2)
            print(f"Saved to {output_path}")

        return spec
