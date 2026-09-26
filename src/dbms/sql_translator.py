from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

from .db_interface import DBExecutor
from src.predicates.validator import (
    Arithmetic,
    Column,
    Constant,
    Expression,
    Group,
    LogicalNode,
    ParsedRule,
    PredicateNode,
    Rule,
    RuleValidationError,
    collect_aliases,
    parse_rule,
)


class SQLTranslator:
    def __init__(
        self, db: DBExecutor, default_alias: str = "t", quote_table_name: bool = True
    ):
        self.db = db
        self.default_alias = default_alias
        self.quote_table_name = quote_table_name

    def _table(self) -> str:
        name = self.db.table_name
        return f'"{name}"' if self.quote_table_name else name

    def translate(self, rule: Any) -> Dict[str, Any]:
        """
        Returns {"success": bool, "sql": str|None, "error"?: str, ...}

        Multi-tuple rules add: t1.tid < t2.tid < t3.tid ... to avoid matching a row with itself
        and to avoid symmetric duplicates.
        """
        try:
            table = self._table()
            tid = self.db.tid_column
            parsed_rule = self._parse_compatible_rule(rule)
            where_expression = self._render_rule(parsed_rule)
            tuple_ids = sorted(collect_aliases(parsed_rule))

            # If no aliases appear, treat as single-tuple rule using default alias
            if not tuple_ids:
                alias = self.default_alias
                sql = f"SELECT {alias}.{tid} AS violation_tid FROM {table} {alias} WHERE {where_expression}"
                return {
                    "success": True,
                    "sql": sql,
                    "constraint_type": "single_tuple",
                    "tuple_identifiers": [alias],
                }

            # Single-tuple: t1.*
            if len(tuple_ids) == 1:
                alias = tuple_ids[0]
                sql = f"SELECT {alias}.{tid} AS violation_tid FROM {table} {alias} WHERE {where_expression}"
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

            where_clause = f"{distinctness} AND {where_expression}"

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

    def _parse_compatible_rule(self, rule: Any) -> ParsedRule:
        """Accept one new-format rule and legacy flat conjunctions during migration."""
        if isinstance(rule, list):
            if not rule:
                raise RuleValidationError("Rule cannot be empty")
            if not all(isinstance(predicate, str) for predicate in rule):
                raise RuleValidationError(
                    "A rule must be a predicate, a logical node, or a legacy predicate list"
                )
            if len(rule) == 1:
                rule = rule[0]
            else:
                rule = {"AND": rule}
        return parse_rule(rule)

    def _render_rule(self, rule: ParsedRule) -> str:
        if isinstance(rule, PredicateNode):
            return self._render_predicate(rule)
        rendered_items = [self._render_rule(item) for item in rule.items]
        return "(" + f" {rule.operator} ".join(rendered_items) + ")"

    def _render_predicate(self, predicate: PredicateNode) -> str:
        left = self._render_expression(predicate.left)
        if predicate.operator in {"IS NULL", "IS NOT NULL"}:
            rendered = f"{left} {predicate.operator}"
        elif isinstance(predicate.right, tuple):
            values = ", ".join(
                self._render_constant(value) for value in predicate.right
            )
            rendered = f"{left} {predicate.operator} ({values})"
        else:
            assert predicate.right is not None
            right = self._render_expression(predicate.right)
            rendered = f"{left} {predicate.operator} {right}"
        return self.db.rewrite_predicate(rendered)

    def _render_expression(self, expression: Expression) -> str:
        if isinstance(expression, Column):
            return f"{expression.alias}.{expression.name}"
        if isinstance(expression, Constant):
            return self._render_constant(expression)
        if isinstance(expression, Group):
            return f"({self._render_expression(expression.expression)})"

        left = self._render_expression(expression.left)
        right = self._render_expression(expression.right)
        if expression.operator == "^":
            return f"POWER({left}, {right})"
        if expression.operator == "%":
            return f"MOD({left}, {right})"
        if expression.operator == "/":
            return f"({left} / NULLIF({right}, 0))"
        return f"({left} {expression.operator} {right})"

    @staticmethod
    def _render_constant(constant: Constant) -> str:
        if constant.is_text:
            return "'" + str(constant.value).replace("'", "''") + "'"
        value = constant.value
        assert isinstance(value, Decimal)
        normalized = format(value.normalize(), "f")
        return "0" if normalized in {"-0", ""} else normalized

    def execute_violation(self, rule: Any) -> Dict[str, Any]:
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
