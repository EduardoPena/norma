from __future__ import annotations

from typing import Any, Dict, Optional
import re

import duckdb
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


class DBExecutor:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        table_name: str = "data_table",
        tid_column: str = "tid",
        csv_path: Optional[str] = None,
    ):
        self.table_name = table_name
        self.tid_column = tid_column
        self.conn = duckdb.connect(":memory:")
        self.column_type_map: Dict[str, str] = {}

        self.conn.execute("SET memory_limit='10GB'")

        if csv_path is None and df is None:
            raise ValueError("DBExecutor requires either df or csv_path")

        if csv_path is not None:
            if df is not None:
                self.column_type_map = self._build_column_type_map(df)
            self._register_csv(csv_path, df)
            return

        if tid_column not in df.columns:
            df = df.copy()
            df[tid_column] = range(1, len(df) + 1)
        else:
            df = df.copy()

        # Convert StringDtype columns to object dtype for DuckDB compatibility
        for col in df.columns:
            if is_string_dtype(df[col]):
                df[col] = df[col].astype("object")

        self.column_type_map = self._build_column_type_map(df)
        self.conn.register(table_name, df)

    def _register_csv(self, csv_path: str, df_sample: Optional[pd.DataFrame]) -> None:
        escaped_path = csv_path.replace("'", "''")
        has_tid = bool(df_sample is not None and self.tid_column in df_sample.columns)
        csv_scan = f"read_csv_auto('{escaped_path}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
        select_list = self._build_csv_select_list(df_sample, has_tid)

        sql = f'CREATE VIEW "{self.table_name}" AS SELECT {select_list} FROM {csv_scan}'

        self.conn.execute(sql)

    def _build_csv_select_list(
        self, df_sample: Optional[pd.DataFrame], has_tid: bool
    ) -> str:
        if df_sample is None:
            base = "*"
        else:
            projected_columns = []
            for column_name in df_sample.columns:
                quoted_name = self._quote_identifier(column_name)
                projected_columns.append(quoted_name)
            base = ", ".join(projected_columns)

        if has_tid:
            return base

        if base == "*":
            return (
                f"*, row_number() OVER () AS {self._quote_identifier(self.tid_column)}"
            )

        return (
            f"{base}, row_number() OVER () AS {self._quote_identifier(self.tid_column)}"
        )

    def _build_column_type_map(self, df: pd.DataFrame) -> Dict[str, str]:
        return {
            column_name: self._duckdb_type_for_series(df[column_name]) or "VARCHAR"
            for column_name in df.columns
        }

    def _duckdb_type_for_series(self, series: pd.Series) -> Optional[str]:
        if is_bool_dtype(series):
            return "BOOLEAN"
        if is_integer_dtype(series):
            return "BIGINT"
        if is_float_dtype(series):
            return "DOUBLE"
        if is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        return None

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def rewrite_predicate(self, predicate: str) -> str:
        lhs, op, rhs = self._split_predicate(predicate)
        if lhs is None or op is None:
            return predicate

        rhs_is_textual = self._rhs_is_textual(op, rhs)
        if rhs_is_textual:
            return predicate

        return self._cast_numeric_refs(predicate)

    def _split_predicate(self, predicate: str):
        multi_word_ops = [
            "IS NOT NULL",
            "IS NULL",
            "NOT LIKE",
            "NOT IN",
            "LIKE",
            "IN",
        ]
        upper = predicate.upper()
        for op in multi_word_ops:
            token = f" {op} "
            idx = upper.find(token)
            if idx != -1:
                lhs = predicate[:idx].strip()
                rhs = predicate[idx + len(token) :].strip()
                return lhs, op, rhs
            trailing = f" {op}"
            if upper.endswith(trailing):
                lhs = predicate[: -len(trailing)].strip()
                return lhs, op, None

        for op in (">=", "<=", "!=", "=", ">", "<"):
            idx = predicate.find(op)
            if idx != -1:
                lhs = predicate[:idx].strip()
                rhs = predicate[idx + len(op) :].strip()
                return lhs, op, rhs

        return None, None, None

    def _rhs_is_textual(self, op: str, rhs: Optional[str]) -> bool:
        if rhs is None:
            return False
        if op in {"LIKE", "NOT LIKE"}:
            return True
        if rhs.startswith("'") or rhs.startswith('"'):
            return True
        if rhs.startswith("(") and ("'" in rhs or '"' in rhs):
            return True
        return False

    def _cast_numeric_refs(self, expression: str) -> str:
        pattern = re.compile(r"\b([A-Za-z]\d+)\.(\w+)\b")

        def replace(match: re.Match[str]) -> str:
            alias, column_name = match.groups()
            column_type = self.column_type_map.get(column_name)
            if column_type not in {"BIGINT", "DOUBLE", "BOOLEAN", "TIMESTAMP"}:
                return match.group(0)
            return f"TRY_CAST({alias}.{column_name} AS {column_type})"

        return pattern.sub(replace, expression)

    def execute(self, sql: str):
        result = self.execute_sql(sql)
        return result.get("rows") if result.get("success") else []

    def close(self) -> None:
        self.conn.close()

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        General SQL executor.

        Returns:
          {"success": bool, "row_count": int|None, "rows": list[tuple]|None, "error"?: str}
        """
        try:
            rows = self.conn.execute(sql).fetchall()
            return {"success": True, "row_count": len(rows), "rows": rows}
        except Exception as e:
            return {"success": False, "row_count": None, "rows": None, "error": str(e)}

    def execute_violation_query(self, sql: str) -> Dict[str, Any]:
        """
        Violation executor (your contract):

        Returns:
          {"success": bool, "row_count": int|None, "rows": list[tuple]|None,
           "constraint_violated": bool|None, "message"?: str, "error"?: str}
        """
        try:
            rows = self.conn.execute(sql).fetchall()

            row_count = len(rows)
            constraint_violated = row_count > 0

            return {
                "success": True,
                "row_count": row_count,
                "rows": rows,  # <-- tuples returned
                "constraint_violated": constraint_violated,
                "message": (
                    f"Found {row_count} violation(s)"
                    if constraint_violated
                    else "No violations found"
                ),
            }
        except Exception as e:
            return {
                "success": False,
                "row_count": None,
                "rows": None,
                "constraint_violated": None,
                "error": str(e),
            }
