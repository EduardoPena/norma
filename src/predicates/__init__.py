"""Predicate handling and space management"""

from .predicates import Predicate, PredicateSpace
from .validator import (
    LogicalNode,
    ParsedRule,
    PredicateNode,
    Rule,
    Rules,
    RuleValidationError,
    canonical_rule,
    canonical_rules,
    collect_aliases,
    iter_predicates,
    normalize_rules,
    predicate_strings,
    parse_predicate,
    parse_rule,
    parse_rules,
    rule_to_data,
    transform_predicates,
)

__all__ = [
    "Predicate",
    "PredicateSpace",
    "LogicalNode",
    "ParsedRule",
    "PredicateNode",
    "Rule",
    "Rules",
    "RuleValidationError",
    "canonical_rule",
    "canonical_rules",
    "collect_aliases",
    "iter_predicates",
    "normalize_rules",
    "predicate_strings",
    "parse_predicate",
    "parse_rule",
    "parse_rules",
    "rule_to_data",
    "transform_predicates",
]
