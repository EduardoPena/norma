"""Evaluator for rule mapping (sentence-level and document-level)"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
import re
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.predicates import Predicate
from src.predicates import (
    Rule,
    Rules,
    canonical_rule,
    canonical_rules,
    normalize_rules,
    predicate_strings,
)
from src.predicates.equivalence import logical_form
from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator

# ---------------------------------------------------------------------------
# Predicate normalisation
# ---------------------------------------------------------------------------


def _normalize_number(s: str) -> str:
    """'0.0' -> '0', '28.0' -> '28', '13.5' stays '13.5'"""
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def _normalize_value(val: str) -> str:
    """
    Normalise the RHS of a predicate so that trivial formatting differences
    don't break equality:
      - strip outer whitespace
      - collapse internal whitespace runs to a single space
      - normalise quote style to single-quotes
      - drop trailing .0 from whole-number floats
      - uppercase bare keywords (NULL, TRUE, FALSE)
    """
    val = val.strip()
    # Collapse internal whitespace
    val = re.sub(r"\s+", " ", val)

    # Quoted string  -> normalise to single-quoted, content unchanged
    for quote in ('"', "'"):
        if val.startswith(quote) and val.endswith(quote) and len(val) >= 2:
            inner = val[1:-1]
            return f"'{inner}'"

    # Parenthesised list  e.g. ('B', 'C')  or  ("B", "C")
    if val.startswith("(") and val.endswith(")"):
        inner = val[1:-1]
        items = [_normalize_value(t.strip()) for t in inner.split(",")]
        return "(" + ", ".join(items) + ")"

    # Bare numeric
    num = _normalize_number(val)
    if num != val:
        return num

    # Bare keyword
    upper = val.upper()
    if upper in ("NULL", "TRUE", "FALSE", "NOT NULL"):
        return upper

    return val


_OP_ALIASES: Dict[str, str] = {
    "==": "=",
    "!<": ">=",
    "!>": "<=",
    "<>": "!=",
    "IS NOT NULL": "IS NOT NULL",  # keep as-is (already canonical)
    "IS NULL": "IS NULL",
}


def _normalize_op(op: str) -> str:
    op = op.strip().upper()
    return _OP_ALIASES.get(op, op)


def normalize_predicate(pred_str: str) -> str:
    """
    Return a canonical string for a predicate so that semantically identical
    predicates compare equal regardless of minor formatting differences.

    Handles:
      - operator aliases  (== -> =,  <> -> !=,  !< -> >=,  !> -> <=)
      - numeric trailing zeros  (0.0 -> 0,  28.0 -> 28)
      - quote style  ("A" -> 'A')
      - extra whitespace
      - LIKE / IN / IS NULL  (all normalised)

    A predicate is assumed to have the form:
        <left>  <op>  <right>
    where <op> may contain spaces (IS NULL, IS NOT NULL, NOT LIKE, NOT IN …).
    """
    pred_str = pred_str.strip()

    # Try to split on known multi-word operators first (longest match first)
    multi_word_ops = [
        "IS NOT NULL",
        "IS NULL",
        "NOT LIKE",
        "NOT IN",
        "LIKE",
        "IN",
    ]
    upper = pred_str.upper()
    for mop in multi_word_ops:
        idx = upper.find(" " + mop + " ")
        if idx != -1:
            left = pred_str[:idx].strip()
            right = pred_str[idx + 1 + len(mop) :].strip()
            return f"{left} {mop} {_normalize_value(right)}"
        # Trailing form  e.g.  "t1.col IS NULL"  (no RHS)
        if upper.endswith(" " + mop):
            left = pred_str[: -(len(mop) + 1)].strip()
            return f"{left} {mop}"

    # Split on two-char then one-char symbolic operators
    for op in (">=", "<=", "!=", "<>", "!<", "!>", "==", "=", ">", "<"):
        idx = pred_str.find(op)
        if idx != -1:
            left = pred_str[:idx].strip()
            right = pred_str[idx + len(op) :].strip()
            canon_op = _normalize_op(op)
            return f"{left} {canon_op} {_normalize_value(right)}"

    # Fallback: return as-is (shouldn't normally happen)
    return pred_str


def normalize_constraint(rule: Rule) -> str:
    """Canonicalize one predicate or logical rule, ignoring logical item order."""
    return canonical_rule(rule)


# ---------------------------------------------------------------------------
# Dataclasses (unchanged interface)
# ---------------------------------------------------------------------------


@dataclass
class SentenceMappingResult:
    """Result for a single sentence mapping (pairwise evaluation)"""

    sentence: str
    gold_is_rule: bool
    pred_is_rule: bool

    strict_correct: Optional[bool] = None
    predicate_jaccard: Optional[float] = None

    execution_exact: Optional[bool] = None
    execution_jaccard: Optional[float] = None
    execution_precision: Optional[float] = None
    execution_recall: Optional[float] = None
    execution_f1: Optional[float] = None

    gold_rule: Any = None
    pred_rule: Any = None
    gold_violations: Optional[Set[int]] = None
    pred_violations: Optional[Set[int]] = None

    # Same logical normal form as the gold rules (see src/predicates/equivalence.py)
    logical_match: Optional[bool] = None
    # Rules of this sentence whose SQL failed (counted as no violations)
    pred_execution_errors: int = 0
    gold_execution_errors: int = 0


@dataclass
class MappingMetrics:
    """Aggregate mapping metrics"""

    constraint_exact_precision: float
    constraint_exact_recall: float
    constraint_exact_f1: float
    constraint_exact_tp: int
    constraint_exact_fp: int
    constraint_exact_fn: int
    num_pred_constraints: int
    num_gold_constraints: int

    pairwise_strict_accuracy: Optional[float] = None
    pairwise_strict_correct: Optional[int] = None
    pairwise_strict_total: Optional[int] = None
    pairwise_avg_predicate_jaccard: Optional[float] = None
    pairwise_std_predicate_jaccard: Optional[float] = None

    pairwise_execution_accuracy: Optional[float] = None
    pairwise_execution_correct: Optional[int] = None
    pairwise_execution_total: Optional[int] = None
    pairwise_avg_execution_jaccard: Optional[float] = None
    pairwise_std_execution_jaccard: Optional[float] = None
    pairwise_avg_execution_precision: Optional[float] = None
    pairwise_std_execution_precision: Optional[float] = None
    pairwise_avg_execution_recall: Optional[float] = None
    pairwise_std_execution_recall: Optional[float] = None
    pairwise_avg_execution_f1: Optional[float] = None
    pairwise_std_execution_f1: Optional[float] = None

    aggregate_violation_precision: Optional[float] = None
    aggregate_violation_recall: Optional[float] = None
    aggregate_violation_f1: Optional[float] = None
    total_gold_violations: Optional[int] = None
    total_pred_violations: Optional[int] = None
    total_overlap_violations: Optional[int] = None

    # Equivalence-aware extraction: one item per sentence output, compared by
    # logical normal form (IN/OR, operand order, distributivity, rule splitting)
    constraint_equiv_precision: Optional[float] = None
    constraint_equiv_recall: Optional[float] = None
    constraint_equiv_f1: Optional[float] = None
    constraint_equiv_tp: Optional[int] = None
    constraint_equiv_fp: Optional[int] = None
    constraint_equiv_fn: Optional[int] = None
    pairwise_logical_accuracy: Optional[float] = None

    # Execution failures (each failing rule counts once; its violations are empty)
    num_pred_execution_errors: Optional[int] = None
    num_gold_execution_errors: Optional[int] = None

    # How concentrated the gold violations are (aggregate F1 can hinge on one rule)
    gold_rules_without_violations: Optional[int] = None
    max_gold_rule_violation_share: Optional[float] = None

    def to_dict(self) -> Dict:
        result = {
            "constraint_exact_precision": self.constraint_exact_precision,
            "constraint_exact_recall": self.constraint_exact_recall,
            "constraint_exact_f1": self.constraint_exact_f1,
            "constraint_exact_tp": self.constraint_exact_tp,
            "constraint_exact_fp": self.constraint_exact_fp,
            "constraint_exact_fn": self.constraint_exact_fn,
            "num_pred_constraints": self.num_pred_constraints,
            "num_gold_constraints": self.num_gold_constraints,
        }
        if self.pairwise_strict_accuracy is not None:
            result.update(
                {
                    "pairwise_strict_accuracy": self.pairwise_strict_accuracy,
                    "pairwise_strict_correct": self.pairwise_strict_correct,
                    "pairwise_strict_total": self.pairwise_strict_total,
                    "pairwise_avg_predicate_jaccard": self.pairwise_avg_predicate_jaccard,
                    "pairwise_std_predicate_jaccard": self.pairwise_std_predicate_jaccard,
                    "pairwise_execution_accuracy": self.pairwise_execution_accuracy,
                    "pairwise_execution_correct": self.pairwise_execution_correct,
                    "pairwise_execution_total": self.pairwise_execution_total,
                    "pairwise_avg_execution_jaccard": self.pairwise_avg_execution_jaccard,
                    "pairwise_std_execution_jaccard": self.pairwise_std_execution_jaccard,
                    "pairwise_avg_execution_precision": self.pairwise_avg_execution_precision,
                    "pairwise_std_execution_precision": self.pairwise_std_execution_precision,
                    "pairwise_avg_execution_recall": self.pairwise_avg_execution_recall,
                    "pairwise_std_execution_recall": self.pairwise_std_execution_recall,
                    "pairwise_avg_execution_f1": self.pairwise_avg_execution_f1,
                    "pairwise_std_execution_f1": self.pairwise_std_execution_f1,
                }
            )
        if self.aggregate_violation_precision is not None:
            result.update(
                {
                    "aggregate_violation_precision": self.aggregate_violation_precision,
                    "aggregate_violation_recall": self.aggregate_violation_recall,
                    "aggregate_violation_f1": self.aggregate_violation_f1,
                    "total_gold_violations": self.total_gold_violations,
                    "total_pred_violations": self.total_pred_violations,
                    "total_overlap_violations": self.total_overlap_violations,
                }
            )
        for key in (
            "constraint_equiv_precision",
            "constraint_equiv_recall",
            "constraint_equiv_f1",
            "constraint_equiv_tp",
            "constraint_equiv_fp",
            "constraint_equiv_fn",
            "pairwise_logical_accuracy",
            "num_pred_execution_errors",
            "num_gold_execution_errors",
            "gold_rules_without_violations",
            "max_gold_rule_violation_share",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class MappingEvaluator:
    """
    Evaluator for rule mapping with three evaluation approaches:

    1. Exact Match Constraint Evaluation (both sentence & document level):
       Constraints are normalised before comparison so minor formatting
       differences (0 vs 0.0, "A" vs 'A', == vs =, <> vs !=) are ignored.

    2. Pairwise Metrics (sentence-level only):
       Syntactic: exact match, predicate Jaccard (after normalisation).
       Execution: exact match, violation Jaccard/P/R/F1.

    3. Aggregate Violation Metrics (both levels):
       Union all violations; compute P/R/F1 on violation row IDs.
    """

    def __init__(
        self,
        ground_truth: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
    ):
        self.ground_truth = ground_truth
        self.df = df
        self.execution_errors: Dict[str, str] = {}
        if df is not None or csv_path is not None:
            self.db = DBExecutor(df=df, csv_path=csv_path)
            self.translator = SQLTranslator(self.db)
        else:
            self.db = None
            self.translator = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_constraint_exact_match(
        self, predicted_rules: List[Any], gold_rules: List[Any]
    ) -> Dict[str, Any]:
        """
        Set-membership exact match after predicate normalisation.
        Returns tp, fp, fn, precision, recall, f1, num_pred, num_gold.
        """
        gold_set = self._canonical_rule_set(gold_rules)
        pred_set = self._canonical_rule_set(predicted_rules)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_pred": len(pred_set),
            "num_gold": len(gold_set),
        }

    def evaluate_constraint_equivalence(
        self, predicted_rules: List[Any], gold_rules: List[Any]
    ) -> Dict[str, Any]:
        """Like the exact match, but one item per sentence output, compared by
        logical normal form; outputs too large for the normal form fall back to
        their canonical rule tuple."""
        gold_set = {self._logical_key(rules) for rules in gold_rules if rules}
        pred_set = {self._logical_key(rules) for rules in predicted_rules if rules}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def _logical_key(rules: Any) -> Any:
        form = logical_form(rules)
        return form if form is not None else ("canonical", canonical_rules(rules))

    def evaluate(
        self, predictions: List[Tuple[str, Any]]
    ) -> Tuple[MappingMetrics, List[SentenceMappingResult]]:
        """Sentence-level evaluation."""
        if not predictions:
            raise ValueError("No predictions to evaluate")

        # Build aligned lists (same sentence order) for exact-match set eval
        pred_rules = [r for _, r in predictions if r]
        gold_rules = [
            self.ground_truth[s] for s, _ in predictions if self.ground_truth.get(s)
        ]

        exact_match_result = self.evaluate_constraint_exact_match(
            pred_rules, gold_rules
        )

        sentence_results = [self._evaluate_single(s, r) for s, r in predictions]
        metrics = self._compute_metrics(sentence_results, exact_match_result)
        equivalence = self.evaluate_constraint_equivalence(pred_rules, gold_rules)
        metrics.constraint_equiv_precision = equivalence["precision"]
        metrics.constraint_equiv_recall = equivalence["recall"]
        metrics.constraint_equiv_f1 = equivalence["f1"]
        metrics.constraint_equiv_tp = equivalence["tp"]
        metrics.constraint_equiv_fp = equivalence["fp"]
        metrics.constraint_equiv_fn = equivalence["fn"]
        return metrics, sentence_results

    def evaluate_document_level(
        self, predicted_rules: List[Any], gold_rules: List[Any]
    ) -> Dict[str, Any]:
        """Document-level evaluation (requires a dataframe for execution)."""
        if self.db is None:
            raise ValueError("Database required for document-level evaluation")

        print(
            f"\nEvaluating {len(predicted_rules)} predicted vs {len(gold_rules)} gold constraints"
        )

        print("Computing exact match constraint metrics...")
        em = self.evaluate_constraint_exact_match(predicted_rules, gold_rules)
        print(f"  TP: {em['tp']}, FP: {em['fp']}, FN: {em['fn']}")

        gold_union = self._execute_rules_union(gold_rules, label="gold")
        pred_union = self._execute_rules_union(predicted_rules, label="predicted")
        overlap = gold_union & pred_union

        vp = len(overlap) / len(pred_union) if pred_union else 0.0
        vr = len(overlap) / len(gold_union) if gold_union else 0.0
        vf = 2 * vp * vr / (vp + vr) if (vp + vr) > 0 else 0.0

        return {
            "constraint_exact_precision": em["precision"],
            "constraint_exact_recall": em["recall"],
            "constraint_exact_f1": em["f1"],
            "constraint_exact_tp": em["tp"],
            "constraint_exact_fp": em["fp"],
            "constraint_exact_fn": em["fn"],
            "num_pred_constraints": em["num_pred"],
            "num_gold_constraints": em["num_gold"],
            "aggregate_violation_precision": vp,
            "aggregate_violation_recall": vr,
            "aggregate_violation_f1": vf,
            "total_gold_violations": len(gold_union),
            "total_pred_violations": len(pred_union),
            "total_overlap_violations": len(overlap),
        }

    def close(self):
        if self.db is not None:
            self.db.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_single(self, sentence: str, pred_rule: Any) -> SentenceMappingResult:
        gold_rule = self.ground_truth.get(sentence)
        gold_is_rule = bool(gold_rule)
        pred_is_rule = bool(pred_rule)

        if not gold_is_rule:
            return SentenceMappingResult(
                sentence=sentence,
                gold_is_rule=False,
                pred_is_rule=pred_is_rule,
                pred_rule=pred_rule,
            )

        # Syntactic metrics compare complete rule trees, not flattened leaves.
        gold_norm = set(canonical_rules(gold_rule) or ())
        pred_norm = set(canonical_rules(pred_rule) or ()) if pred_is_rule else set()

        strict_correct = (gold_norm == pred_norm) if pred_is_rule else False
        logical_match = (
            self._logical_key(gold_rule) == self._logical_key(pred_rule)
            if pred_is_rule
            else False
        )
        # Jaccard over predicate leaves (rule trees are compared by strict_correct)
        gold_leaves = set(predicate_strings(gold_rule))
        pred_leaves = set(predicate_strings(pred_rule)) if pred_is_rule else set()
        predicate_jaccard = self._jaccard(gold_leaves, pred_leaves)

        # Execution metrics
        exec_exact = exec_jaccard = exec_precision = exec_recall = exec_f1 = None
        gold_violations = pred_violations = None

        gold_errors = pred_errors = 0
        if self.db is not None:
            gold_violations, gold_errors = self._execute_rule_checked(gold_rule)
            pred_violations, pred_errors = (
                self._execute_rule_checked(pred_rule) if pred_is_rule else (set(), 0)
            )

            if gold_violations or pred_violations:
                exec_exact = gold_violations == pred_violations
                exec_jaccard = self._jaccard(gold_violations, pred_violations)
                exec_precision, exec_recall, exec_f1 = self._prf(
                    gold_violations, pred_violations
                )

        return SentenceMappingResult(
            sentence=sentence,
            gold_is_rule=True,
            pred_is_rule=pred_is_rule,
            strict_correct=strict_correct,
            predicate_jaccard=predicate_jaccard,
            execution_exact=exec_exact,
            execution_jaccard=exec_jaccard,
            execution_precision=exec_precision,
            execution_recall=exec_recall,
            execution_f1=exec_f1,
            gold_rule=gold_rule,
            pred_rule=pred_rule,
            gold_violations=gold_violations,
            pred_violations=pred_violations,
            logical_match=logical_match,
            pred_execution_errors=pred_errors,
            gold_execution_errors=gold_errors,
        )

    def _compute_metrics(
        self, results: List[SentenceMappingResult], exact_match_result: Dict[str, Any]
    ) -> MappingMetrics:
        gold_results = [r for r in results if r.gold_is_rule]

        # Pairwise syntactic
        strict_correct = sum(1 for r in gold_results if r.strict_correct)
        strict_total = len(gold_results)
        strict_accuracy = strict_correct / strict_total if strict_total else 0.0

        logical_correct = sum(1 for r in gold_results if r.logical_match)
        logical_accuracy = logical_correct / strict_total if strict_total else 0.0

        jaccard_scores = [
            r.predicate_jaccard for r in gold_results if r.predicate_jaccard is not None
        ]
        avg_jaccard, std_jaccard = self._mean_std(jaccard_scores)

        # Execution (only if db available)
        exec_accuracy = exec_correct = exec_total = None
        avg_ej = std_ej = avg_ep = std_ep = avg_er = std_er = avg_ef = std_ef = None
        agg_vp = agg_vr = agg_vf = total_gv = total_pv = total_ov = None
        gold_rules_without_violations = max_share = None

        if self.db is not None:
            exec_correct = sum(1 for r in gold_results if r.execution_exact)
            exec_total = sum(1 for r in gold_results if r.execution_exact is not None)
            exec_accuracy = exec_correct / exec_total if exec_total else 0.0

            avg_ej, std_ej = self._mean_std(
                [
                    r.execution_jaccard
                    for r in gold_results
                    if r.execution_jaccard is not None
                ]
            )
            avg_ep, std_ep = self._mean_std(
                [
                    r.execution_precision
                    for r in gold_results
                    if r.execution_precision is not None
                ]
            )
            avg_er, std_er = self._mean_std(
                [
                    r.execution_recall
                    for r in gold_results
                    if r.execution_recall is not None
                ]
            )
            avg_ef, std_ef = self._mean_std(
                [r.execution_f1 for r in gold_results if r.execution_f1 is not None]
            )

            # Aggregate violation union
            gold_union = set().union(
                *(r.gold_violations for r in gold_results if r.gold_violations)
            )
            pred_union = set().union(
                *(r.pred_violations for r in gold_results if r.pred_violations)
            )
            overlap = gold_union & pred_union
            total_gv, total_pv, total_ov = (
                len(gold_union),
                len(pred_union),
                len(overlap),
            )
            agg_vp = total_ov / total_pv if total_pv else 0.0
            agg_vr = total_ov / total_gv if total_gv else 0.0
            agg_vf = (
                2 * agg_vp * agg_vr / (agg_vp + agg_vr) if (agg_vp + agg_vr) else 0.0
            )
            gold_rules_without_violations = sum(
                1 for r in gold_results if not r.gold_violations
            )
            max_share = (
                max(len(r.gold_violations or ()) for r in gold_results) / total_gv
                if total_gv
                else 0.0
            )

        return MappingMetrics(
            constraint_exact_precision=exact_match_result["precision"],
            constraint_exact_recall=exact_match_result["recall"],
            constraint_exact_f1=exact_match_result["f1"],
            constraint_exact_tp=exact_match_result["tp"],
            constraint_exact_fp=exact_match_result["fp"],
            constraint_exact_fn=exact_match_result["fn"],
            num_pred_constraints=exact_match_result["num_pred"],
            num_gold_constraints=exact_match_result["num_gold"],
            pairwise_strict_accuracy=strict_accuracy,
            pairwise_strict_correct=strict_correct,
            pairwise_strict_total=strict_total,
            pairwise_avg_predicate_jaccard=avg_jaccard,
            pairwise_std_predicate_jaccard=std_jaccard,
            pairwise_execution_accuracy=exec_accuracy,
            pairwise_execution_correct=exec_correct,
            pairwise_execution_total=exec_total,
            pairwise_avg_execution_jaccard=avg_ej,
            pairwise_std_execution_jaccard=std_ej,
            pairwise_avg_execution_precision=avg_ep,
            pairwise_std_execution_precision=std_ep,
            pairwise_avg_execution_recall=avg_er,
            pairwise_std_execution_recall=std_er,
            pairwise_avg_execution_f1=avg_ef,
            pairwise_std_execution_f1=std_ef,
            aggregate_violation_precision=agg_vp,
            aggregate_violation_recall=agg_vr,
            aggregate_violation_f1=agg_vf,
            total_gold_violations=total_gv,
            total_pred_violations=total_pv,
            total_overlap_violations=total_ov,
            pairwise_logical_accuracy=logical_accuracy,
            num_pred_execution_errors=(
                sum(r.pred_execution_errors for r in gold_results)
                if self.db is not None
                else None
            ),
            num_gold_execution_errors=(
                sum(r.gold_execution_errors for r in gold_results)
                if self.db is not None
                else None
            ),
            gold_rules_without_violations=(
                gold_rules_without_violations if self.db is not None else None
            ),
            max_gold_rule_violation_share=max_share if self.db is not None else None,
        )

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_rules_union(self, rules: List[Any], label: str = "") -> Set[int]:
        union: Set[int] = set()
        for i, rule in enumerate(rules, 1):
            union.update(self._execute_rule(rule))
            if i % 10 == 0:
                print(f"  Processed {i}/{len(rules)} {label} constraints")
        print(f"  Total unique {label} violations: {len(union)}")
        return union

    def _execute_rule(self, rules: Any) -> Set[int]:
        """Execute a sentence output (independent rule list) and union violations."""
        return self._execute_rule_checked(rules)[0]

    def _execute_rule_checked(self, rules: Any) -> Tuple[Set[int], int]:
        """Violations of a sentence output plus the number of rules whose SQL failed.

        A failing rule contributes no violations (as before) but is now counted,
        logged and kept in ``self.execution_errors`` instead of disappearing.
        """
        if not rules:
            return set(), 0
        violation_ids: Set[int] = set()
        errors = 0
        try:
            normalized = normalize_rules(rules)
        except Exception as e:
            self._record_execution_error(rules, f"invalid rule: {e}")
            return set(), 1
        for rule in normalized or []:
            try:
                result = self.translator.execute_violation(rule)
            except Exception as e:  # translator bug or unexpected DB error
                result = {"success": False, "error": str(e)}
            if not result["success"]:
                errors += 1
                self._record_execution_error(rule, result.get("error"))
                continue
            for row in result.get("rows", []):
                for tid in row:
                    violation_ids.add(tid)
        return violation_ids, errors

    def _record_execution_error(self, rule: Any, error: Any) -> None:
        key = repr(rule)
        if key not in self.execution_errors:
            print(f"Warning: Rule execution failed (counted as error): {rule} -> {error}")
            self.execution_errors[key] = str(error)

    @staticmethod
    def _canonical_rule_set(rule_groups: List[Any]) -> Set[str]:
        canonical: Set[str] = set()
        for rules in rule_groups:
            canonical.update(canonical_rules(rules) or ())
        return canonical

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: Set, b: Set) -> float:
        if not a and not b:
            return 1.0
        denom = len(a | b)
        return len(a & b) / denom if denom else 0.0

    @staticmethod
    def _prf(gold: Set, pred: Set) -> Tuple[float, float, float]:
        """Precision, recall, F1 between two sets. Both empty → (1,1,1)."""
        if not gold and not pred:
            return 1.0, 1.0, 1.0
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values)) if len(values) > 1 else 0.0

    # ------------------------------------------------------------------
    # Soft Jaccard (available for callers who want it)
    # ------------------------------------------------------------------

    def soft_jaccard(self, rule1: List[str], rule2: List[str]) -> float:
        """Soft Jaccard between two constraints via Hungarian matching."""
        set1 = {Predicate.from_string(p) for p in rule1}
        set2 = {Predicate.from_string(p) for p in rule2}
        return self._soft_jaccard(set1, set2)

    def _soft_jaccard(self, set1: Set[Predicate], set2: Set[Predicate]) -> float:
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        list1, list2 = list(set1), list(set2)
        sim = np.array(
            [[self._predicate_similarity(p1, p2) for p2 in list2] for p1 in list1]
        )
        rows, cols = linear_sum_assignment(-sim)
        soft_inter = sum(sim[r, c] for r, c in zip(rows, cols))
        soft_union = len(set1) + len(set2) - soft_inter
        return soft_inter / soft_union if soft_union else 0.0

    def _predicate_similarity(self, p1: Predicate, p2: Predicate) -> float:
        left_sim = 1.0 if p1.left == p2.left else 0.0
        op_sim = self._operator_similarity(p1.op, p2.op)
        r1_is_col = "." in p1.right
        r2_is_col = "." in p2.right
        if r1_is_col and r2_is_col:
            right_sim = 1.0 if p1.right == p2.right else 0.3
        elif r1_is_col or r2_is_col:
            right_sim = 0.0
        else:
            right_sim = self._value_similarity(p1.right, p2.right)
        return 0.4 * left_sim + 0.3 * op_sim + 0.3 * right_sim

    @staticmethod
    def _operator_similarity(op1: str, op2: str) -> float:
        # Normalise before comparing
        op1 = _normalize_op(op1)
        op2 = _normalize_op(op2)
        if op1 == op2:
            return 1.0
        groups = [
            {">=", ">"},
            {"<=", "<"},
            {"!="},
            {"LIKE", "NOT LIKE"},
            {"IN", "NOT IN"},
            {"IS NULL", "IS NOT NULL"},
        ]
        for g in groups:
            if op1 in g and op2 in g:
                return 0.5 if ("NOT" in op1) != ("NOT" in op2) else 0.7
        if {op1, op2} <= {"<", "<=", ">", ">=", "=", "!="}:
            return 0.3
        return 0.0

    @staticmethod
    def _value_similarity(val1: str, val2: str) -> float:
        v1 = _normalize_value(val1)
        v2 = _normalize_value(val2)
        if v1 == v2:
            return 1.0
        try:
            n1, n2 = float(v1.strip("'")), float(v2.strip("'"))
            mx = max(abs(n1), abs(n2))
            return max(0.0, 1.0 - abs(n1 - n2) / mx) if mx else 1.0
        except ValueError:
            pass
        return 0.5 if (v1 in v2 or v2 in v1) else 0.0
