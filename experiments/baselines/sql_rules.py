"""Read the violation condition of a Text-to-SQL query as Norma rules.

The paper evaluates constraint extraction for XiYan-SQL by reading the predicates
of the WHERE clause of each generated query. This module does that with sqlglot:
the WHERE clause (and any JOIN ... ON condition) becomes one rule, an AND/OR tree
of predicate strings in Norma's format ("t1.col op value"), which
src.predicates.normalize_rules and MappingEvaluator accept.

- Predicates on the row identifier (t1.tid <> t2.tid, ...) are dropped: they are
  the pair semantics of the violation query, not part of the constraint.
- CAST/TRY_CAST around a column are unwrapped (Norma casts numeric columns itself).
- BETWEEN becomes two comparisons; NOT over =, IN, LIKE and IS NULL becomes the
  negated operator.
- A correlated EXISTS that is a conjunct of the outer WHERE
  ("WHERE ... AND EXISTS (SELECT 1 FROM t1 AS t2 WHERE <cond>)") is the usual way
  of writing a two-tuple violation, so its condition is read as pair predicates
  (inner alias -> t2). Predicates between the two tuples are written with t1 on
  the left, as in the gold DCs ("t2.a = t1.a" -> "t1.a = t2.a").
- Rules that Norma's predicate grammar rejects (CASE, function calls on columns,
  ...) are non-extractable ("unsupported expression"), since they cannot be scored.
- NOT EXISTS, other subqueries (IN (SELECT ...), scalar), aggregates,
  GROUP BY/HAVING and set operations make the query non-extractable:
  sql_to_rules returns (None, reason) and the caller counts it, instead of guessing.
"""
from typing import Any, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from src.predicates.validator import RuleValidationError, parse_rules

TID = "tid"
_CMP = {exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}
_NEG = {"=": "!=", "!=": "=", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_FLIP = {"=": "=", "!=": "!=", ">": "<", ">=": "<=", "<": ">", "<=": ">="}


class NotExtractable(Exception):
    pass


def _qualify(node: exp.Expression) -> exp.Expression:
    """Column references as t1.<col>/t2.<col> (unqualified -> t1), casts unwrapped."""
    def fix(n):
        if isinstance(n, (exp.Cast, exp.TryCast)) and isinstance(n.this, exp.Column):
            return n.this
        if isinstance(n, exp.Column):
            table = (n.table or "t1").lower()
            if table not in ("t1", "t2"):
                raise NotExtractable(f"unknown table alias {table!r}")
            return exp.column(n.name, table=table)
        return n
    return node.transform(fix)


def _text(node: exp.Expression) -> str:
    return _qualify(node).sql(dialect="duckdb", identify=False)


def _side(node: exp.Expression) -> str:
    """Operand text; arithmetic in parentheses, as in the gold DCs ("t1.a >= (5 * t1.b)")."""
    while isinstance(node, exp.Paren):
        node = node.this
    text = _text(node)
    return f"({text})" if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)) else text


def _tables(node: exp.Expression) -> set:
    return {(c.table or "t1").lower() for c in node.find_all(exp.Column)}


def _oriented(left: exp.Expression, op: str, right: exp.Expression) -> str:
    """Comparison text with t1 on the left when it relates t2 (left) to t1 (right)."""
    if _tables(left) == {"t2"} and _tables(right) == {"t1"}:
        left, right, op = right, left, _FLIP[op]
    return f"{_side(left)} {op} {_side(right)}"


def _conjuncts(node: exp.Expression) -> List[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    return [node]


def _exists(node: exp.Exists) -> List[Any]:
    """Pair predicates of 'EXISTS (SELECT .. FROM t1 AS <a> WHERE cond)' (alias -> t2)."""
    sub = node.this
    if not isinstance(sub, exp.Select) or sub.args.get("joins") or sub.args.get("group") \
            or sub.args.get("having") or sub.find(exp.AggFunc) \
            or any(s is not sub for s in sub.find_all(exp.Select)) or sub.args.get("where") is None:
        raise NotExtractable("subquery")
    source = sub.args.get("from").this if sub.args.get("from") else None
    if not isinstance(source, exp.Table) or source.name.lower() != "t1":
        raise NotExtractable("subquery")
    alias = (source.alias or source.name).lower()

    def rename(n):
        if isinstance(n, exp.Column):
            table = (n.table or alias).lower()
            if table == alias:
                return exp.column(n.name, table="t2")
            if table == "t1":
                return exp.column(n.name, table="t1")
            raise NotExtractable(f"unknown table alias {table!r}")
        return n

    cond = sub.args["where"].this.transform(rename)
    return [p for p in (_convert(c) for c in _conjuncts(cond)) if p is not None]


def _mentions_tid(node: exp.Expression) -> bool:
    return any(c.name.lower() == TID for c in node.find_all(exp.Column))


def _convert(node: exp.Expression) -> Optional[Any]:
    """Expression -> predicate string, {"AND": [...]}, {"OR": [...]} or None (dropped)."""
    if isinstance(node, exp.Paren):
        return _convert(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        key = "AND" if isinstance(node, exp.And) else "OR"
        items = []
        for side in (node.left, node.right):
            sub = _convert(side)
            if sub is None:
                continue
            if isinstance(sub, dict) and key in sub:
                items.extend(sub[key])
            else:
                items.append(sub)
        if not items:
            return None
        return items[0] if len(items) == 1 else {key: items}
    if _mentions_tid(node):
        return None
    if isinstance(node, exp.Not):
        return _negate(node.this)
    if type(node) in _CMP:
        return _oriented(node.left, _CMP[type(node)], node.right)
    if isinstance(node, exp.Is):
        return f"{_text(node.this)} IS NULL"
    if isinstance(node, (exp.Like, exp.ILike)):
        return f"{_text(node.this)} LIKE {_text(node.expression)}"
    if isinstance(node, exp.In):
        return _in(node, negate=False)
    if isinstance(node, exp.Between):
        col = _text(node.this)
        return {"AND": [f"{col} >= {_text(node.args['low'])}", f"{col} <= {_text(node.args['high'])}"]}
    raise NotExtractable(f"unsupported predicate {type(node).__name__}")


def _negate(node: exp.Expression) -> Optional[Any]:
    """NOT pushed down (De Morgan); NOT BETWEEN becomes an OR of two comparisons."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Not):
        return _convert(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        key = "OR" if isinstance(node, exp.And) else "AND"
        items = [x for x in (_negate(node.left), _negate(node.right)) if x is not None]
        if not items:
            return None
        return items[0] if len(items) == 1 else {key: items}
    if _mentions_tid(node):
        return None
    if isinstance(node, exp.In):
        return _in(node, negate=True)
    if isinstance(node, (exp.Like, exp.ILike)):
        return f"{_text(node.this)} NOT LIKE {_text(node.expression)}"
    if isinstance(node, exp.Is):
        return f"{_text(node.this)} IS NOT NULL"
    if type(node) in _CMP:
        return _oriented(node.left, _NEG[_CMP[type(node)]], node.right)
    if isinstance(node, exp.Between):
        col = _text(node.this)
        return {"OR": [f"{col} < {_text(node.args['low'])}", f"{col} > {_text(node.args['high'])}"]}
    raise NotExtractable(f"NOT over {type(node).__name__}")


def _in(node: exp.In, negate: bool) -> str:
    if node.args.get("query") or node.find(exp.Select):
        raise NotExtractable("IN with subquery")
    values = ", ".join(_text(v) for v in node.expressions)
    return f"{_text(node.this)} {'NOT IN' if negate else 'IN'} ({values})"


def sql_to_rules(sql: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Return ([rule], None) for an extractable query, or (None, reason)."""
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception as error:  # noqa: BLE001 - any parse failure is reported
        return None, f"parse error: {error}"
    if not isinstance(tree, exp.Select):
        return None, f"not a single SELECT ({type(tree).__name__})"
    if tree.args.get("group") or tree.args.get("having") or any(
            a.find_ancestor(exp.Select) is tree for a in tree.find_all(exp.AggFunc)):
        return None, "aggregation"
    conditions = []
    for join in tree.args.get("joins") or []:
        if join.args.get("on") is not None:
            conditions.append(join.args["on"])
    if tree.args.get("where") is not None:
        conditions.append(tree.args["where"].this)
    if not conditions:
        return None, "no WHERE clause"
    parts = []
    try:
        for condition in conditions:
            for conjunct in _conjuncts(condition):
                if isinstance(conjunct, exp.Exists):
                    parts.extend(_exists(conjunct))
                elif isinstance(conjunct, exp.Not) and conjunct.find(exp.Exists):
                    raise NotExtractable("NOT EXISTS")
                elif conjunct.find(exp.Select) or conjunct.find(exp.Exists):
                    raise NotExtractable("subquery")
                else:
                    part = _convert(conjunct)
                    if part is not None:
                        parts.append(part)
    except NotExtractable as error:
        return None, str(error)
    if not parts:
        return None, "only tid predicates"
    flat = []
    for part in parts:
        flat.extend(part["AND"] if isinstance(part, dict) and "AND" in part else [part])
    rule = flat[0] if len(flat) == 1 else {"AND": flat}
    try:
        parse_rules([rule])
    except RuleValidationError as error:
        return None, f"unsupported expression ({str(error)[:60]})"
    return [rule], None
