"""Logical normal form of denial-constraint rules, for equivalence-aware evaluation.

The exact-match metric compares canonical rule trees, so logically identical
rules written differently (``x IN (1,3)`` vs ``x = 1 OR x = 3``, ``a * 0.05``
vs ``0.05 * a``, distributed ORs, one rule with an inner OR vs several
independent rules) count as errors. ``logical_form`` maps a sentence output
(list of independent rules, i.e. a disjunction of violation conditions) to a
disjunctive normal form over normalized literals:

- ``IN`` / ``NOT IN`` expand to ORs of ``=`` / ANDs of ``!=``;
- comparisons are oriented (column before constant, then lexicographic order),
  flipping the operator accordingly;
- ``+`` and ``*`` operands are sorted (commutativity), redundant parentheses
  around a single column or constant are dropped;
- AND/OR are flattened and distributed (DNF), duplicates and absorbed terms
  (``A OR (A AND B)`` = ``A``) are removed.

It does not use data or domain knowledge: ``x >= 1 AND x <= 4`` and
``x IN (1,2,3,4)`` stay different even for integer codes.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional

from .validator import (
    Arithmetic,
    Column,
    Constant,
    Group,
    LogicalNode,
    PredicateNode,
    _canonical_constant,
    parse_rules,
)

Literal = str
Term = FrozenSet[Literal]
LogicalForm = FrozenSet[Term]

MAX_TERMS = 4096  # DNF size guard; larger rules fall back to None (no equivalence claim)

_FLIP = {"=": "=", "!=": "!=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}


class _TooLarge(Exception):
    pass


def _expression(expression: Any) -> str:
    if isinstance(expression, Column):
        return f"{expression.alias}.{expression.name}"
    if isinstance(expression, Constant):
        return _canonical_constant(expression)
    if isinstance(expression, Group):
        inner = expression.expression
        if isinstance(inner, (Column, Constant, Group)):
            return _expression(inner)
        return f"({_expression(inner)})"
    assert isinstance(expression, Arithmetic)
    if expression.operator in {"+", "*"}:
        operands = sorted(_flatten(expression, expression.operator))
        return f" {expression.operator} ".join(operands)
    return f"{_operand(expression.left)} {expression.operator} {_operand(expression.right)}"


def _flatten(expression: Any, operator: str) -> list[str]:
    """Operands of a chain of the same commutative operator (a * b * c)."""
    while isinstance(expression, Group) and isinstance(expression.expression, Arithmetic):
        if expression.expression.operator != operator:
            break
        expression = expression.expression
    if isinstance(expression, Arithmetic) and expression.operator == operator:
        return _flatten(expression.left, operator) + _flatten(expression.right, operator)
    return [_operand(expression)]


def _operand(expression: Any) -> str:
    text = _expression(expression)
    if isinstance(expression, Arithmetic) or (
        isinstance(expression, Group) and isinstance(expression.expression, Arithmetic)
    ):
        return f"({text})"
    return text


def _is_constant(expression: Any) -> bool:
    if isinstance(expression, Constant):
        return True
    if isinstance(expression, Group):
        return _is_constant(expression.expression)
    if isinstance(expression, Arithmetic):
        return _is_constant(expression.left) and _is_constant(expression.right)
    return False


def _comparison(left: Any, operator: str, right: Any) -> Literal:
    left_text, right_text = _expression(left), _expression(right)
    left_const, right_const = _is_constant(left), _is_constant(right)
    flip = (left_const and not right_const) or (
        left_const == right_const and right_text < left_text
    )
    if flip:
        left_text, right_text, operator = right_text, left_text, _FLIP[operator]
    return f"{left_text} {operator} {right_text}"


def _predicate_form(predicate: PredicateNode) -> LogicalForm:
    left = predicate.left
    if predicate.operator in {"IS NULL", "IS NOT NULL"}:
        return frozenset({frozenset({f"{_expression(left)} {predicate.operator}"})})
    if predicate.operator in {"IN", "NOT IN"}:
        assert isinstance(predicate.right, tuple)
        literals = {_comparison(left, "=" if predicate.operator == "IN" else "!=", value)
                    for value in predicate.right}
        if predicate.operator == "IN":
            return frozenset(frozenset({literal}) for literal in literals)
        return frozenset({frozenset(literals)})
    if predicate.operator in {"LIKE", "NOT LIKE"}:
        return frozenset({frozenset({
            f"{_expression(left)} {predicate.operator} {_expression(predicate.right)}"
        })})
    return frozenset({frozenset({_comparison(left, predicate.operator, predicate.right)})})


def _absorb(terms: set[Term]) -> LogicalForm:
    """Drop terms that are supersets of another term (A OR (A AND B) = A)."""
    ordered = sorted(terms, key=len)
    kept: list[Term] = []
    for term in ordered:
        if not any(other <= term for other in kept):
            kept.append(term)
    return frozenset(kept)


def _rule_form(rule: Any) -> LogicalForm:
    if isinstance(rule, PredicateNode):
        return _predicate_form(rule)
    assert isinstance(rule, LogicalNode)
    children = [_rule_form(item) for item in rule.items]
    if rule.operator == "OR":
        return _absorb(set().union(*children))
    terms: set[Term] = {frozenset()}
    for child in children:
        terms = {left | right for left in terms for right in child}
        if len(terms) > MAX_TERMS:
            raise _TooLarge
    return _absorb(terms)


def logical_form(rules: Any) -> Optional[LogicalForm]:
    """DNF of a sentence output (independent rules = disjunction); None if empty/too large."""
    parsed = parse_rules(rules)
    if not parsed:
        return None
    try:
        terms: set[Term] = set()
        for rule in parsed:
            terms |= _rule_form(rule)
            if len(terms) > MAX_TERMS:
                raise _TooLarge
        return _absorb(terms)
    except _TooLarge:
        return None


def logically_equivalent(left: Any, right: Any) -> bool:
    """True when both outputs have the same normal form (no data needed)."""
    left_form, right_form = logical_form(left), logical_form(right)
    return left_form is not None and left_form == right_form
