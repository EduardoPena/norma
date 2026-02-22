"""Evaluator for rule mapping (sentence-level and document-level)"""
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.predicates import Predicate
from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator


@dataclass
class SentenceMappingResult:
    """Result for a single sentence mapping (pairwise evaluation)"""
    sentence: str
    gold_is_rule: bool
    pred_is_rule: bool
    
    # Syntactic metrics (only for gold rules)
    strict_correct: Optional[bool] = None
    predicate_jaccard: Optional[float] = None
    
    # Execution metrics (only for gold rules)
    execution_exact: Optional[bool] = None
    execution_jaccard: Optional[float] = None

    execution_precision: Optional[float] = None
    execution_recall: Optional[float] = None
    execution_f1: Optional[float] = None
    
    # Raw data
    gold_rule: Optional[List[str]] = None
    pred_rule: Optional[List[str]] = None
    gold_violations: Optional[Set[int]] = None
    pred_violations: Optional[Set[int]] = None


@dataclass
class MappingMetrics:
    """Aggregate mapping metrics"""
    # Constraint exact match metrics (set membership)
    constraint_exact_precision: float
    constraint_exact_recall: float
    constraint_exact_f1: float
    constraint_exact_tp: int
    constraint_exact_fp: int
    constraint_exact_fn: int
    num_pred_constraints: int
    num_gold_constraints: int
    
    # Pairwise syntactic metrics (only for sentence-level with known pairing)
    pairwise_strict_accuracy: Optional[float] = None
    pairwise_strict_correct: Optional[int] = None
    pairwise_strict_total: Optional[int] = None
    pairwise_avg_predicate_jaccard: Optional[float] = None
    pairwise_std_predicate_jaccard: Optional[float] = None
    
    # Pairwise execution metrics (only for sentence-level with known pairing)
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
    
    # Aggregate violation metrics (union-based, for both levels)
    aggregate_violation_precision: Optional[float] = None
    aggregate_violation_recall: Optional[float] = None
    aggregate_violation_f1: Optional[float] = None
    total_gold_violations: Optional[int] = None
    total_pred_violations: Optional[int] = None
    total_overlap_violations: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            'constraint_exact_precision': self.constraint_exact_precision,
            'constraint_exact_recall': self.constraint_exact_recall,
            'constraint_exact_f1': self.constraint_exact_f1,
            'constraint_exact_tp': self.constraint_exact_tp,
            'constraint_exact_fp': self.constraint_exact_fp,
            'constraint_exact_fn': self.constraint_exact_fn,
            'num_pred_constraints': self.num_pred_constraints,
            'num_gold_constraints': self.num_gold_constraints,
        }
        
        # Add pairwise metrics if available (sentence-level only)
        if self.pairwise_strict_accuracy is not None:
            result.update({
                'pairwise_strict_accuracy': self.pairwise_strict_accuracy,
                'pairwise_strict_correct': self.pairwise_strict_correct,
                'pairwise_strict_total': self.pairwise_strict_total,
                'pairwise_avg_predicate_jaccard': self.pairwise_avg_predicate_jaccard,
                'pairwise_std_predicate_jaccard': self.pairwise_std_predicate_jaccard,
                'pairwise_execution_accuracy': self.pairwise_execution_accuracy,
                'pairwise_execution_correct': self.pairwise_execution_correct,
                'pairwise_execution_total': self.pairwise_execution_total,
                'pairwise_avg_execution_jaccard': self.pairwise_avg_execution_jaccard,
                'pairwise_std_execution_jaccard': self.pairwise_std_execution_jaccard,
            })
        
        # Add aggregate violation metrics if available
        if self.aggregate_violation_precision is not None:
            result.update({
                'aggregate_violation_precision': self.aggregate_violation_precision,
                'aggregate_violation_recall': self.aggregate_violation_recall,
                'aggregate_violation_f1': self.aggregate_violation_f1,
                'total_gold_violations': self.total_gold_violations,
                'total_pred_violations': self.total_pred_violations,
                'total_overlap_violations': self.total_overlap_violations,
            })
        
        return result


class MappingEvaluator:
    """
    Evaluator for rule mapping with two evaluation approaches:
    
    1. Exact Match Constraint Evaluation (both sentence & document level):
       - Simple set membership check: is predicted constraint in gold constraint set?
       - Computes TP/FP/FN, Precision/Recall/F1 at constraint level
    
    2. Pairwise Metrics (sentence-level only, requires known pairing):
       - Syntactic: Exact match, Predicate Jaccard
       - Execution: Exact match, Violation Jaccard
    
    3. Aggregate Violation Metrics (both levels):
       - Union all violations from all constraints
       - Compute Precision/Recall/F1 on violation row IDs
    """
    
    def __init__(self, 
                 ground_truth: Dict[str, List[str]],
                 df: Optional[pd.DataFrame] = None):
        """
        Args:
            ground_truth: Dict mapping sentences to denial constraints
            df: Optional dataframe for execution evaluation
        """
        self.ground_truth = ground_truth
        self.df = df
        
        # Initialize DB evaluator if dataframe provided
        if df is not None:
            self.db = DBExecutor(df)
            self.translator = SQLTranslator(self.db)
        else:
            self.db = None
            self.translator = None
    
    def evaluate_constraint_exact_match(self,
                                       predicted_rules: List[List[str]],
                                       gold_rules: List[List[str]]) -> Dict[str, Any]:
        """
        Evaluate constraints using exact match (set membership).
        
        Algorithm: For each predicted constraint, check if it exists in gold constraint set.
        No pairing algorithm needed - just set membership check.
        
        Args:
            predicted_rules: List of predicted denial constraints
            gold_rules: List of ground truth denial constraints
        
        Returns:
            Dict with tp, fp, fn, precision, recall, f1
        """
        # Convert constraints to normalized frozensets for comparison
        def normalize_constraint(rule: List[str]) -> frozenset:
            """Convert constraint to normalized frozenset of predicate strings"""
            if not rule:
                return frozenset()
            # Sort predicates to ensure consistent representation
            return frozenset(sorted(rule))
        
        gold_set = {normalize_constraint(rule) for rule in gold_rules}
        pred_set = {normalize_constraint(rule) for rule in predicted_rules}
        
        # Remove empty constraints
        gold_set.discard(frozenset())
        pred_set.discard(frozenset())
        
        # Compute TP, FP, FN
        tp = len(pred_set & gold_set)  # Predicted constraints that are in gold
        fp = len(pred_set - gold_set)  # Predicted constraints not in gold
        fn = len(gold_set - pred_set)  # Gold constraints not predicted
        
        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_pred': len(pred_set),
            'num_gold': len(gold_set)
        }
    
    def evaluate(self,
                 predictions: List[Tuple[str, Optional[List[str]]]]) -> Tuple[MappingMetrics, List[SentenceMappingResult]]:
        """
        Evaluate mapping predictions (sentence-level).
        
        Computes:
        1. Exact match constraint evaluation (set membership)
        2. Pairwise metrics (using known sentence->constraint mapping)
        3. Aggregate violation metrics
        
        Args:
            predictions: List of (sentence, predicates) tuples from mapper
        
        Returns:
            Tuple of (aggregate_metrics, per_sentence_results)
        """
        if not predictions:
            raise ValueError("No predictions to evaluate")
        
        # Extract all predicted and gold constraints
        pred_rules = [pred_rule for _, pred_rule in predictions if pred_rule and len(pred_rule) > 0]
        gold_rules = [self.ground_truth.get(sent) for sent, _ in predictions 
                     if self.ground_truth.get(sent) and len(self.ground_truth.get(sent)) > 0]
        
        # 1. Exact match constraint evaluation
        exact_match_result = self.evaluate_constraint_exact_match(pred_rules, gold_rules)
        
        # 2. Evaluate each sentence (pairwise metrics)
        sentence_results = []
        for sentence, pred_rule in predictions:
            result = self._evaluate_single(sentence, pred_rule)
            sentence_results.append(result)
        
        # 3. Compute aggregate metrics (pairwise + aggregate violations)
        metrics = self._compute_metrics(sentence_results, exact_match_result)
        
        return metrics, sentence_results
    
    def evaluate_document_level(self,
                               predicted_rules: List[List[str]],
                               gold_rules: List[List[str]]) -> Dict[str, Any]:
        """
        Evaluate document-level predictions.
        
        Computes:
        1. Exact match constraint evaluation (set membership)
        2. Aggregate violation discovery metrics
        
        Args:
            predicted_rules: List of predicted denial constraints
            gold_rules: List of ground truth denial constraints
        
        Returns:
            Dict with constraint-level and violation-level metrics
        """
        if self.db is None:
            raise ValueError("Database required for document-level evaluation")
        
        print(f"\nEvaluating {len(predicted_rules)} predicted vs {len(gold_rules)} gold constraints")
        
        # 1. Exact match constraint evaluation
        print("Computing exact match constraint metrics...")
        exact_match_result = self.evaluate_constraint_exact_match(predicted_rules, gold_rules)
        print(f"  TP: {exact_match_result['tp']}, FP: {exact_match_result['fp']}, FN: {exact_match_result['fn']}")
        
        # 2. Aggregate violation metrics
        # Execute all gold rules
        print("Executing gold constraints...")
        gold_violations_union = set()
        for i, dc in enumerate(gold_rules, 1):
            violations = self._execute_rule(dc)
            gold_violations_union.update(violations)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(gold_rules)} gold constraints")
        
        print(f"  Total unique gold violations: {len(gold_violations_union)}")
        
        # Execute all predicted rules
        print("Executing predicted constraints...")
        pred_violations_union = set()
        for i, dc in enumerate(predicted_rules, 1):
            violations = self._execute_rule(dc)
            pred_violations_union.update(violations)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(predicted_rules)} predicted constraints")
        
        print(f"  Total unique pred violations: {len(pred_violations_union)}")
        
        # Compute aggregate violation metrics
        overlap_violations = gold_violations_union & pred_violations_union
        
        violation_precision = len(overlap_violations) / len(pred_violations_union) if len(pred_violations_union) > 0 else 0.0
        violation_recall = len(overlap_violations) / len(gold_violations_union) if len(gold_violations_union) > 0 else 0.0
        violation_f1 = (2 * violation_precision * violation_recall / (violation_precision + violation_recall)
                       if (violation_precision + violation_recall) > 0 else 0.0)
        
        return {
            # Constraint exact match metrics
            'constraint_exact_precision': exact_match_result['precision'],
            'constraint_exact_recall': exact_match_result['recall'],
            'constraint_exact_f1': exact_match_result['f1'],
            'constraint_exact_tp': exact_match_result['tp'],
            'constraint_exact_fp': exact_match_result['fp'],
            'constraint_exact_fn': exact_match_result['fn'],
            'num_pred_constraints': exact_match_result['num_pred'],
            'num_gold_constraints': exact_match_result['num_gold'],
            # Aggregate violation metrics
            'aggregate_violation_precision': violation_precision,
            'aggregate_violation_recall': violation_recall,
            'aggregate_violation_f1': violation_f1,
            'total_gold_violations': len(gold_violations_union),
            'total_pred_violations': len(pred_violations_union),
            'total_overlap_violations': len(overlap_violations),
        }
    
    def _evaluate_single(self, 
                        sentence: str, 
                        pred_rule: Optional[List[str]]) -> SentenceMappingResult:
        """Evaluate a single sentence prediction (pairwise)"""
        # Get ground truth
        gold_rule = self.ground_truth.get(sentence, None)
        gold_is_rule = gold_rule is not None and len(gold_rule) > 0
        pred_is_rule = pred_rule is not None and len(pred_rule) > 0
        
        # For non-rules, return early
        if not gold_is_rule:
            return SentenceMappingResult(
                sentence=sentence,
                gold_is_rule=False,
                pred_is_rule=pred_is_rule,
                pred_rule=pred_rule
            )
        
        # For gold rules: compute pairwise metrics
        gold_set = self._rule_to_predicate_set(gold_rule)
        
        # Syntactic metrics
        strict_correct = False
        predicate_jaccard = 0.0
        if pred_is_rule:
            pred_set = self._rule_to_predicate_set(pred_rule)
            strict_correct = (gold_set == pred_set)
            predicate_jaccard = self._jaccard_similarity(gold_set, pred_set)
        
        # Execution metrics
        exec_exact = None
        exec_jaccard = None
        gold_violations = None
        pred_violations = None
        
        if self.db is not None:
            gold_violations = self._execute_rule(gold_rule)
            pred_violations = self._execute_rule(pred_rule) if pred_is_rule else set()
            
            # Exact match
            exec_exact = (gold_violations == pred_violations)
            
            # Jaccard
            exec_jaccard = self._set_jaccard(gold_violations, pred_violations)

            tp = len(gold_violations & pred_violations)
            fp = len(pred_violations - gold_violations)
            fn = len(gold_violations - pred_violations)
            
            exec_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            exec_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            exec_f1        = (2 * exec_precision * exec_recall / (exec_precision + exec_recall)
                            if (exec_precision + exec_recall) > 0 else 0.0)
        
        return SentenceMappingResult(
            sentence=sentence,
            gold_is_rule=True,
            pred_is_rule=pred_is_rule,
            strict_correct=strict_correct,
            predicate_jaccard=predicate_jaccard,
            execution_exact=exec_exact,
            execution_jaccard=exec_jaccard,
            execution_exact=exec_exact,
            execution_jaccard=exec_jaccard,
            execution_precision=exec_precision,
            execution_recall=exec_recall,
            execution_f1=exec_f1,
            gold_rule=gold_rule,
            pred_rule=pred_rule,
            gold_violations=gold_violations,
            pred_violations=pred_violations
        )
    
    def _compute_metrics(self, results: List[SentenceMappingResult], 
                        exact_match_result: Dict[str, Any]) -> MappingMetrics:
        """Compute aggregate metrics from sentence results"""
        # Pairwise syntactic metrics (only gold rules)
        gold_rules = [r for r in results if r.gold_is_rule]
        strict_correct = sum(1 for r in gold_rules if r.strict_correct)
        strict_total = len(gold_rules)
        strict_accuracy = strict_correct / strict_total if strict_total > 0 else 0.0
        
        jaccard_scores = [r.predicate_jaccard for r in gold_rules if r.predicate_jaccard is not None]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        std_jaccard = float(np.std(jaccard_scores)) if len(jaccard_scores) > 1 else 0.0
        
        # Pairwise execution metrics
        exec_accuracy = None
        exec_correct = None
        exec_total = None
        avg_exec_jaccard = None
        std_exec_jaccard = None
        agg_violation_precision = None
        agg_violation_recall = None
        agg_violation_f1 = None
        total_gold_viols = None
        total_pred_viols = None
        total_overlap = None
        
        if self.db is not None:
            # Pairwise execution metrics
            exec_exact_matches = [r for r in gold_rules if r.execution_exact is not None and r.execution_exact]
            exec_correct = len(exec_exact_matches)
            exec_total = len([r for r in gold_rules if r.execution_exact is not None])
            exec_accuracy = exec_correct / exec_total if exec_total > 0 else 0.0
            
            exec_jaccard_scores = [r.execution_jaccard for r in gold_rules if r.execution_jaccard is not None]
            avg_exec_jaccard = sum(exec_jaccard_scores) / len(exec_jaccard_scores) if exec_jaccard_scores else 0.0
            std_exec_jaccard = float(np.std(exec_jaccard_scores)) if len(exec_jaccard_scores) > 1 else 0.0
            
            # Aggregate violation metrics
            gold_union = set()
            pred_union = set()
            
            for r in gold_rules:
                if r.gold_violations:
                    gold_union.update(r.gold_violations)
                if r.pred_violations:
                    pred_union.update(r.pred_violations)
            
            overlap_union = gold_union & pred_union
            
            total_gold_viols = len(gold_union)
            total_pred_viols = len(pred_union)
            total_overlap = len(overlap_union)
            
            agg_violation_precision = total_overlap / total_pred_viols if total_pred_viols > 0 else 0.0
            agg_violation_recall = total_overlap / total_gold_viols if total_gold_viols > 0 else 0.0
            agg_violation_f1 = (2 * agg_violation_precision * agg_violation_recall / (agg_violation_precision + agg_violation_recall)
                       if (agg_violation_precision + agg_violation_recall) > 0 else 0.0)
        
        return MappingMetrics(
            # Constraint exact match (from exact_match_result)
            constraint_exact_precision=exact_match_result['precision'],
            constraint_exact_recall=exact_match_result['recall'],
            constraint_exact_f1=exact_match_result['f1'],
            constraint_exact_tp=exact_match_result['tp'],
            constraint_exact_fp=exact_match_result['fp'],
            constraint_exact_fn=exact_match_result['fn'],
            num_pred_constraints=exact_match_result['num_pred'],
            num_gold_constraints=exact_match_result['num_gold'],
            # Pairwise syntactic metrics
            pairwise_strict_accuracy=strict_accuracy,
            pairwise_strict_correct=strict_correct,
            pairwise_strict_total=strict_total,
            pairwise_avg_predicate_jaccard=avg_jaccard,
            pairwise_std_predicate_jaccard=std_jaccard,
            # Pairwise execution metrics
            pairwise_execution_accuracy=exec_accuracy,
            pairwise_execution_correct=exec_correct,
            pairwise_execution_total=exec_total,
            pairwise_avg_execution_jaccard=avg_exec_jaccard,
            pairwise_std_execution_jaccard=std_exec_jaccard,
            # Aggregate violation metrics
            aggregate_violation_precision=agg_violation_precision,
            aggregate_violation_recall=agg_violation_recall,
            aggregate_violation_f1=agg_violation_f1,
            total_gold_violations=total_gold_viols,
            total_pred_violations=total_pred_viols,
            total_overlap_violations=total_overlap
        )
    
    def _execute_rule(self, rule: Optional[List[str]]) -> Set[int]:
        """Execute a rule and return set of violating tuple IDs"""
        if rule is None or len(rule) == 0:
            return set()
        
        try:
            result = self.translator.execute_violation(rule)
            if not result['success']:
                print(f"Warning: Rule execution failed: {result.get('error')}")
                return set()
            
            # Extract tuple IDs from rows
            rows = result.get('rows', [])
            if not rows:
                return set()
            
            # For single-tuple: rows = [(tid,), ...]
            # For multi-tuple: rows = [(t1_tid, t2_tid, ...), ...]
            violation_ids = set()
            for row in rows:
                # Add all tuple IDs in the row
                for tid in row:
                    violation_ids.add(tid)
            
            return violation_ids
        except Exception as e:
            print(f"Warning: Exception during rule execution: {e}")
            return set()
    
    def _rule_to_predicate_set(self, rule: List[str]) -> Set[Predicate]:
        """Convert rule to set of Predicate objects"""
        return {Predicate.from_string(pred_str) for pred_str in rule}
    
    def _jaccard_similarity(self, set1: Set[Predicate], set2: Set[Predicate]) -> float:
        """Compute Jaccard similarity between predicate sets"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _set_jaccard(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between any sets"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _soft_jaccard_similarity(self, set1: Set[Predicate], set2: Set[Predicate]) -> float:
        """Compute soft Jaccard using Hungarian algorithm"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        list1 = list(set1)
        list2 = list(set2)
        
        # Compute similarity matrix
        sim_matrix = np.zeros((len(list1), len(list2)))
        for i, p1 in enumerate(list1):
            for j, p2 in enumerate(list2):
                sim_matrix[i, j] = self._predicate_similarity(p1, p2)
        
        # Hungarian algorithm
        cost_matrix = -sim_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Compute soft intersection
        soft_intersection = sum(sim_matrix[i, j] for i, j in zip(row_indices, col_indices))
        soft_union = len(set1) + len(set2) - soft_intersection
        
        return soft_intersection / soft_union if soft_union > 0 else 0.0
    
    def _predicate_similarity(self, p1: Predicate, p2: Predicate) -> float:
        """Compute similarity between two predicates"""
        left1, op1, right1 = p1.left, p1.op, p1.right
        left2, op2, right2 = p2.left, p2.op, p2.right
        
        # Left column similarity (weight: 0.4)
        left_sim = 1.0 if left1 == left2 else 0.0
        
        # Operator similarity (weight: 0.3)
        op_sim = self._operator_similarity(op1, op2)
        
        # Right side similarity (weight: 0.3)
        right1_is_col = '.' in right1
        right2_is_col = '.' in right2
        
        if right1_is_col and right2_is_col:
            right_sim = 1.0 if right1 == right2 else 0.3
        elif right1_is_col or right2_is_col:
            right_sim = 0.0
        else:
            right_sim = self._value_similarity(right1, right2)
        
        return 0.4 * left_sim + 0.3 * op_sim + 0.3 * right_sim
    
    def _operator_similarity(self, op1: str, op2: str) -> float:
        """Compute similarity between operators"""
        if op1 == op2:
            return 1.0
        
        compatible_groups = [
            {'>=', '>', '!<'},
            {'<=', '<', '!>'},
            {'=', '=='},
            {'!=', '<>'},
            {'LIKE', 'NOT LIKE'},  # Pattern matching operators
            {'IN', 'NOT IN'},  # Set membership operators
            {'IS NULL', 'IS NOT NULL'},  # Null check operators
        ]
        
        op1_norm = op1.upper().strip() if op1 else ''
        op2_norm = op2.upper().strip() if op2 else ''
        
        if op1_norm == op2_norm:
            return 1.0
        
        # Check if in same compatible group
        for group in compatible_groups:
            if op1_norm in group and op2_norm in group:
                # Same family but opposite meaning (e.g., LIKE vs NOT LIKE)
                if ('NOT' in op1_norm) != ('NOT' in op2_norm):
                    return 0.5  # Related but opposite
                return 0.7  # Same family
        
        # Partial credit for comparison operators
        comparison_ops = {'<', '<=', '>', '>=', '=', '!=', '<>'}
        if op1_norm in comparison_ops and op2_norm in comparison_ops:
            return 0.3
        
        # Partial credit for set operations
        set_ops = {'IN', 'NOT IN'}
        if op1_norm in set_ops and op2_norm in set_ops:
            return 0.3
        
        # Partial credit for pattern matching
        pattern_ops = {'LIKE', 'NOT LIKE'}
        if op1_norm in pattern_ops and op2_norm in pattern_ops:
            return 0.3
        
        # Partial credit for null checks
        null_ops = {'IS NULL', 'IS NOT NULL'}
        if op1_norm in null_ops and op2_norm in null_ops:
            return 0.3
        
        return 0.0
    
    def _value_similarity(self, val1: str, val2: str) -> float:
        """Compute similarity between values"""
        if val1 == val2:
            return 1.0
        
        if val1 is None or val2 is None:
            return 0.0
        
        val1_str = str(val1).strip().strip("'\"")
        val2_str = str(val2).strip().strip("'\"")
        
        if val1_str == val2_str:
            return 1.0
        
        # Try numeric comparison
        try:
            num1 = float(val1_str)
            num2 = float(val2_str)
            if num1 == 0 and num2 == 0:
                return 1.0
            max_val = max(abs(num1), abs(num2))
            diff = abs(num1 - num2) / max_val if max_val > 0 else 0
            return max(0.0, 1.0 - diff)
        except (ValueError, TypeError):
            pass
        
        # String containment
        if val1_str in val2_str or val2_str in val1_str:
            return 0.5
        
        return 0.0
    
    def close(self):
        """Close database connection"""
        if self.db is not None:
            self.db.close()