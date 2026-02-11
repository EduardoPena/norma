from __future__ import annotations

from typing import Any, Dict

import duckdb
import pandas as pd
from pandas.api.types import is_string_dtype


class DBExecutor:
    def __init__(self, df: pd.DataFrame, table_name: str = "data_table", tid_column: str = "tid"):
        self.table_name = table_name
        self.tid_column = tid_column
        self.conn = duckdb.connect(":memory:")

        if tid_column not in df.columns:
            df = df.copy()
            df[tid_column] = range(1, len(df) + 1)
        else:
            df = df.copy()
        
        # Convert StringDtype columns to object dtype for DuckDB compatibility
        for col in df.columns:
            if is_string_dtype(df[col]):
                df[col] = df[col].astype('object')

        self.conn.register(table_name, df)

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
                "message": f"Found {row_count} violation(s)" if constraint_violated else "No violations found",
            }
        except Exception as e:
            return {
                "success": False,
                "row_count": None,
                "rows": None,
                "constraint_violated": None,
                "error": str(e),
            }
