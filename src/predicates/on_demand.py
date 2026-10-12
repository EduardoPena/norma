"""On-demand predicate generation for version-2 (schema-only) specifications.

``OnDemandPredicates.generate(columns)`` returns the predicate templates of the
selected columns, built with the v1 rules (``PredicateSpaceBuilder``, unchanged)
on the same data. Invariant: for the same data and options, ``generate(C)`` equals
the v1 predicates whose columns are all in ``C``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

import pandas as pd

from .space_builder import PredicateSpaceBuilder
from .space_builder_v2 import builder_kwargs, coerce_declared_types, iter_columns
from .templates import predicate_templates
from .validator import iter_predicates, parse_predicate

_COLUMN_REF = re.compile(r"\bt\d+\.([A-Za-z_][A-Za-z0-9_]*)\b")


class _SubsetBuilder(PredicateSpaceBuilder):
    """v1 builder whose coded-column detection uses the whole table, so a column
    keeps its coded status (inherited from family siblings) in any subset."""

    def __init__(self, coded: Set[str], **kwargs: Any):
        super().__init__(**kwargs)
        self._coded = coded

    def coded_numeric_columns(self, df: pd.DataFrame) -> Set[str]:  # noqa: D401
        return self._coded & set(df.columns)


class OnDemandPredicates:
    """Generate predicate templates for selected columns of a v2 specification."""

    def __init__(self, spec: Dict[str, Any], df: pd.DataFrame):
        self.spec = spec
        self.types = {c["name"]: c["type"] for c in iter_columns(spec)}
        self.columns = list(self.types)
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns of the v2 schema missing from the data: {missing[:5]}")
        self.df = coerce_declared_types(df[self.columns], self.types)
        self.options = builder_kwargs(spec.get("generation", {}))
        full = PredicateSpaceBuilder(**self.options)
        coded = (
            full.coded_numeric_columns(self.df)
            if self.options.get("numeric_categorical_threshold") is not None
            else set()
        )
        self._coded = coded
        self._cache: Dict[FrozenSet[str], Set[str]] = {}

    def generate(self, columns: Iterable[str]) -> Set[str]:
        selected = frozenset(c for c in columns if c in self.types)
        if not selected:
            return set()
        if selected not in self._cache:
            ordered = [c for c in self.columns if c in selected]  # v1 column order
            builder = _SubsetBuilder(self._coded, **self.options)
            self._cache[selected] = builder.build_predicates(self.df[ordered])
        return set(self._cache[selected])

    def generate_list(self, columns: Iterable[str]) -> List[str]:
        return sorted(self.generate(columns))

    def allowed_for(self, predicate: str) -> bool:
        """Whether a template (or a concrete predicate) is in the space of its columns."""
        columns = set(_COLUMN_REF.findall(predicate))
        space = self.generate(columns)
        if predicate in space:
            return True
        try:
            node = parse_predicate(predicate)
        except ValueError:
            return False
        return bool(predicate_templates(node) & space)

    def out_of_space_counts(self, outputs: Iterable[Any]) -> tuple[int, int]:
        """(predicates outside the space, total predicates) over sentence outputs."""
        from .validator import parse_rules

        outside = total = 0
        for rules in outputs:
            if not rules:
                continue
            try:
                parsed = parse_rules(rules)
            except ValueError:
                continue
            for rule in parsed or ():
                for node in iter_predicates(rule):
                    total += 1
                    columns = {e.name for e in _columns_of(node)}
                    outside += not (predicate_templates(node) & self.generate(columns))
        return outside, total


def _columns_of(node: Any) -> List[Any]:
    from .validator import Arithmetic, Column, Group

    found: List[Any] = []

    def walk(expression: Any) -> None:
        if isinstance(expression, Column):
            found.append(expression)
        elif isinstance(expression, Arithmetic):
            walk(expression.left)
            walk(expression.right)
        elif isinstance(expression, Group):
            walk(expression.expression)

    walk(node.left)
    if not isinstance(node.right, tuple) and node.right is not None:
        walk(node.right)
    return found


def is_v2(predicate_space: Optional[Dict[str, Any]]) -> bool:
    return bool(predicate_space) and int(predicate_space.get("version", 1)) >= 2
