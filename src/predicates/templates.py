"""Match concrete predicates against predicate-space templates ($int, $string, ...)."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Optional

from .validator import (
    Arithmetic,
    Column,
    Constant,
    Group,
    PredicateNode,
    iter_predicates,
    parse_rules,
)

_NUMERIC_PLACEHOLDERS = ("$int", "$float", "$number")


def _variants(expression: Any) -> list[list[str]]:
    """Renderings of an expression; each rendering is a list of tokens where
    numeric constants appear as "#" (to be replaced by a placeholder)."""
    if isinstance(expression, Column):
        return [[f"{expression.alias}.{expression.name}"]]
    if isinstance(expression, Constant):
        return [["$string"]] if expression.is_text else [["#"]]
    if isinstance(expression, Group):
        inner = _variants(expression.expression)
        return [["(", *tokens, ")"] for tokens in inner] + inner
    assert isinstance(expression, Arithmetic)
    out = []
    for left, right in product(_variants(expression.left), _variants(expression.right)):
        out.append([*left, expression.operator, *right])
        if expression.operator in {"+", "*"}:
            out.append([*right, expression.operator, *left])
    return out


def predicate_templates(predicate: PredicateNode) -> set[str]:
    """Template strings a concrete predicate could have been instantiated from."""
    left_variants = _variants(predicate.left)
    if predicate.operator in {"IS NULL", "IS NOT NULL"}:
        renderings = [[*left, predicate.operator] for left in left_variants]
    elif predicate.operator in {"IN", "NOT IN"}:
        renderings = [[*left, predicate.operator, "$values"] for left in left_variants]
    elif predicate.operator in {"LIKE", "NOT LIKE"}:
        renderings = [[*left, predicate.operator, "$pattern"] for left in left_variants]
    else:
        renderings = [
            [*left, predicate.operator, *right]
            for left, right in product(left_variants, _variants(predicate.right))
        ]
    templates = set()
    for tokens in renderings:
        text = " ".join(tokens).replace("( ", "(").replace(" )", ")")
        holes = text.count("#")
        if not holes:
            templates.add(text)
            continue
        for choice in product(_NUMERIC_PLACEHOLDERS, repeat=holes):
            filled = text
            for placeholder in choice:
                filled = filled.replace("#", placeholder, 1)
            templates.add(filled)
    return templates


def in_space(predicate: PredicateNode, allowed: Iterable[str]) -> bool:
    allowed = allowed if isinstance(allowed, (set, frozenset)) else set(allowed)
    return bool(predicate_templates(predicate) & allowed)


def out_of_space_counts(
    outputs: Iterable[Any], allowed: Optional[Iterable[str]]
) -> tuple[int, int]:
    """(predicates outside the space, total predicates) over sentence outputs."""
    if not allowed:
        return 0, 0
    allowed = set(allowed)
    outside = total = 0
    for rules in outputs:
        if not rules:
            continue
        try:
            parsed = parse_rules(rules)
        except ValueError:
            continue
        for rule in parsed or ():
            for predicate in iter_predicates(rule):
                total += 1
                outside += not in_space(predicate, allowed)
    return outside, total
