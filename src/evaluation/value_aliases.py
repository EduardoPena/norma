"""Data-verified aliases between equality literals of 1:1 column pairs.

Many datasets store a category twice: a code and its label (``status = 1`` and
``status_label = 'Active'``). Rules written with either column select the same
rows, but a syntactic comparison treats them as different. ``ValueAliases``
finds such pairs in the evaluation data itself, without relying on column names:

- candidate pairs: low-cardinality columns whose (a, b) value pairs, over rows
  where both are non-null, form a one-to-one mapping (extra code values without
  a label are allowed and never aliased);
- a literal ``A = a`` is rewritten to ``B = b`` only if both select exactly the
  same rows of the evaluation data (checked per value and per operator, with SQL
  NULL semantics), so ``!=`` is aliased only when NULLs do not break it.

The resulting equivalence is relative to the evaluation data, which is also the
data used for the violation metrics.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def constant_key(value: Any) -> Optional[str]:
    """Canonical text of a data value, matching the canonical constant format of
    ``src.predicates.validator`` ("1" for 1 / 1.0, "'text'" for strings)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float, Decimal)):
        try:
            text = format(Decimal(str(value)).normalize(), "f")
        except InvalidOperation:
            return None
        return "0" if text in {"-0", ""} else text
    return "'" + str(value).replace("'", "''") + "'"


class ValueAliases:
    """Rewrite equality literals of data-verified 1:1 column pairs to one form."""

    def __init__(self, df: pd.DataFrame, max_distinct: int = 50, sample_rows: int = 200000,
                 min_coverage: float = 0.9):
        self.df = df
        self.max_distinct = max_distinct
        self.min_coverage = min_coverage
        # (column, value key) -> [(partner column, partner raw value, own raw value)]
        self.partners: Dict[Tuple[str, str], List[Tuple[str, Any, Any]]] = {}
        self._verified: Dict[Tuple[str, str, str, str], bool] = {}
        self._find_pairs(df.head(sample_rows))

    def _find_pairs(self, sample: pd.DataFrame) -> None:
        """Pairs whose co-occurring (non-null) values map one-to-one; a code may
        have extra values without a label (they are simply never aliased). The
        co-occurring rows must cover ``min_coverage`` of the sparser column, which
        a code/label pair does and coincidences between sparse columns do not."""
        codes: Dict[str, Any] = {}
        uniques: Dict[str, Any] = {}
        for column in sample.columns:
            if column == "tid":
                continue
            try:
                factor, values = pd.factorize(sample[column], use_na_sentinel=True)
            except TypeError:  # unhashable values
                continue
            if 2 <= len(values) <= self.max_distinct:
                codes[column], uniques[column] = factor.astype(np.int64), values
        present = {c: codes[c] >= 0 for c in codes}
        counts = {c: int(present[c].sum()) for c in codes}
        for a, b in combinations(sorted(codes), 2):
            both = present[a] & present[b]
            n_both = int(both.sum())
            if n_both == 0 or n_both < self.min_coverage * min(counts[a], counts[b]):
                continue
            width = len(uniques[b]) + 1
            combined = np.unique(codes[a][both] * width + codes[b][both])
            ia, ib = combined // width, combined % width
            if len(combined) < 2 or len(np.unique(ia)) != len(combined) or len(np.unique(ib)) != len(combined):
                continue
            for xa, xb in zip(ia, ib):
                va, vb = uniques[a][xa], uniques[b][xb]
                ka, kb = constant_key(va), constant_key(vb)
                if ka is None or kb is None:
                    continue
                self.partners.setdefault((a, ka), []).append((b, vb, va))
                self.partners.setdefault((b, kb), []).append((a, va, vb))

    def _same_rows(self, column: str, value: Any, partner: str, partner_value: Any, op: str) -> bool:
        key = (column, constant_key(value), partner, op)
        if key not in self._verified:
            left, right = self.df[column], self.df[partner]
            if op == "=":
                mask_l, mask_r = left == value, right == partner_value
            else:  # SQL: NULL != x is not true
                mask_l = (left != value) & left.notna()
                mask_r = (right != partner_value) & right.notna()
            self._verified[key] = bool(mask_l.any()) and bool((mask_l == mask_r).all())
        return self._verified[key]

    def canonical(self, alias: str, column: str, op: str, value_key: str) -> Optional[str]:
        """Canonical literal for ``alias.column op value`` or None (no alias)."""
        if op not in {"=", "!="} or (column, value_key) not in self.partners:
            return None
        equivalent = [(column, value_key)]
        for partner, partner_value, own_value in self.partners[(column, value_key)]:
            if self._same_rows(column, own_value, partner, partner_value, op):
                equivalent.append((partner, constant_key(partner_value)))
        if len(equivalent) == 1:
            return None
        # deterministic representative: the lexicographically smallest column
        rep_column, rep_value = min(equivalent)
        return f"{alias}.{rep_column} {op} {rep_value}"
