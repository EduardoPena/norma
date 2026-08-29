from __future__ import annotations

from typing import Any, Dict, List
import re

from .db_interface import DBExecutor


class SQLTranslator:
    def __init__(
        self, db: DBExecutor, default_alias: str = "t", quote_table_name: bool = True
    ):
        self.db = db
        self.default_alias = default_alias
        self.quote_table_name = quote_table_name
        self._tuple_pattern = re.compile(r"\b([a-z]\d+)\.")  # t1., p2., h10., ...

    def _table(self) -> str:
        name = self.db.table_name
        return f'"{name}"' if self.quote_table_name else name

    def _extract_tuple_ids(self, rule: List[str]) -> List[str]:
        ids: set[str] = set()
        for pred in rule:
            ids.update(self._tuple_pattern.findall(pred))
        return sorted(ids)

    def translate(self, rule: List[str]) -> Dict[str, Any]:
        """
        Returns {"success": bool, "sql": str|None, "error"?: str, ...}

        Multi-tuple rules add: t1.tid < t2.tid < t3.tid ... to avoid matching a row with itself
        and to avoid symmetric duplicates.
        """
        try:
            table = self._table()
            tid = self.db.tid_column

            normalized_rule = [self.db.rewrite_predicate(pred) for pred in rule]
            tuple_ids = self._extract_tuple_ids(normalized_rule)

            # If no aliases appear, treat as single-tuple rule using default alias
            if not tuple_ids:
                alias = self.default_alias
                where_clause = " AND ".join(normalized_rule)
                sql = f"SELECT {alias}.{tid} AS violation_tid FROM {table} {alias} WHERE {where_clause}"
                return {
                    "success": True,
                    "sql": sql,
                    "constraint_type": "single_tuple",
                    "tuple_identifiers": [alias],
                }

            # Single-tuple: t1.*
            if len(tuple_ids) == 1:
                alias = tuple_ids[0]
                where_clause = " AND ".join(normalized_rule)
                sql = f"SELECT {alias}.{tid} AS violation_tid FROM {table} {alias} WHERE {where_clause}"
                return {
                    "success": True,
                    "sql": sql,
                    "constraint_type": "single_tuple",
                    "tuple_identifiers": tuple_ids,
                }

            # Multi-tuple: FROM table t1, table t2, ...
            from_clause = ", ".join([f"{table} {a}" for a in tuple_ids])

            # Distinctness/order to prevent self-match and duplicates:
            # t1.tid < t2.tid AND t2.tid < t3.tid ...
            distinctness = " AND ".join(
                [
                    f"{tuple_ids[i]}.{tid} < {tuple_ids[i+1]}.{tid}"
                    for i in range(len(tuple_ids) - 1)
                ]
            )

            where_parts = [distinctness] + normalized_rule
            where_clause = " AND ".join(where_parts)

            select_cols = ", ".join([f"{a}.{tid} AS {a}_tid" for a in tuple_ids])

            sql = f"SELECT {select_cols} FROM {from_clause} WHERE {where_clause}"
            return {
                "success": True,
                "sql": sql,
                "constraint_type": "multi_tuple",
                "tuple_identifiers": tuple_ids,
            }

        except Exception as e:
            return {"success": False, "sql": None, "error": str(e)}

    def execute_violation(self, rule: List[str]) -> Dict[str, Any]:
        """Translate + execute using db.execute_violation_query()."""
        tr = self.translate(rule)
        if not tr["success"]:
            return {
                "success": False,
                "row_count": None,
                "rows": None,
                "constraint_violated": None,
                "sql": None,
                "error": tr.get("error", "translation failed"),
            }

        sql = tr["sql"]
        out = self.db.execute_violation_query(sql)
        out["sql"] = sql
        out["constraint_type"] = tr["constraint_type"]
        out["tuple_identifiers"] = tr["tuple_identifiers"]
        return out
