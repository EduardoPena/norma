"""Deterministic predicate candidate selection shared by sentence mappers."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def select_relevant_columns(
    sentence: str, tables: List[Dict[str, Any]], max_columns: int = 12
) -> List[str]:
    """Rank schema names/descriptions by lexical overlap, without model or GT access."""

    def tokens(text: str) -> Set[str]:
        return {
            token.rstrip("s") if len(token) > 4 else token
            for token in re.findall(r"[a-z]{3,}", text.lower())
        }

    columns = {
        column["name"]: column for table in tables for column in table["columns"]
    }
    name_tokens = {name: tokens(name) for name in columns}
    documents = {
        name: name_tokens[name] | tokens(str(column.get("description") or ""))
        for name, column in columns.items()
    }
    frequencies = Counter(
        token for document in documents.values() for token in document
    )
    sentence_tokens = tokens(sentence)
    scores = {
        name: sum(
            math.log(1 + len(columns) / frequencies[token])
            * (3 if token in name_tokens[name] else 1)
            for token in sorted(sentence_tokens & document)
        )
        for name, document in documents.items()
    }
    return sorted(
        (name for name in columns if scores[name] > 0),
        key=lambda name: (-scores[name], name),
    )[:max_columns]


def simple_predicates(allowed_predicates: Set[str]) -> Set[str]:
    """Preserve one-column templates, including implicit schema concepts."""
    column_reference = re.compile(r"\bt\d+\.([A-Za-z_][A-Za-z0-9_]*)\b")
    return {
        predicate
        for predicate in allowed_predicates
        if len(set(column_reference.findall(predicate))) == 1
        and not re.search(r"\s[+*/%]\s", predicate)
    }


def filter_predicates_by_columns(
    allowed_predicates: Set[str],
    columns: List[str],
    sentence: str = "",
    max_candidate_predicates: int = 1000,
) -> List[str]:
    """Keep templates referencing selected columns, using the Multi-Step ranking."""
    if not columns or not allowed_predicates:
        return sorted(allowed_predicates)[:max_candidate_predicates]

    selected_columns = set(columns)
    column_reference = re.compile(r"\bt\d+\.([A-Za-z_][A-Za-z0-9_]*)\b")
    relevant = [
        predicate
        for predicate in allowed_predicates
        if (references := set(column_reference.findall(predicate)))
        and references <= selected_columns
    ]
    candidates = relevant if relevant else list(allowed_predicates)
    sentence_tokens = set(re.findall(r"[a-z]+", sentence.lower()))
    priority_columns = {
        column
        for column in selected_columns
        if any(part in sentence_tokens and len(part) >= 4 for part in column.split("_"))
    }

    def has_arithmetic(predicate: str) -> bool:
        return bool(re.search(r"\s[+*/%]\s", predicate))

    simple = [
        predicate
        for predicate in candidates
        if len(set(column_reference.findall(predicate))) == 1
        and not has_arithmetic(predicate)
    ]
    simple_set = set(simple)

    def relevance_rank(predicate: str) -> Tuple[int, int, bool, int, str]:
        references = set(column_reference.findall(predicate))
        return (
            len(references - priority_columns),
            int(has_arithmetic(predicate) and "percent" not in sentence_tokens),
            "t2." in predicate,
            len(references),
            predicate,
        )

    ordered = sorted(simple) + sorted(
        (predicate for predicate in candidates if predicate not in simple_set),
        key=relevance_rank,
    )
    return ordered[:max_candidate_predicates]
