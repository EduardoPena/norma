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
    """Result for a single sentence mapping"""
    sentence: str
    gold_is_rule: bool
    pred_is_rule: bool
    detection_status: str  # 'TP', 'TN', 'FP', 'FN'
    
    # Syntactic metrics (only for gold rules)
    strict_correct: Optional[bool] = None
    predicate_jaccard: Optional[float] = None
    soft_jaccard: Optional[float] = None
    
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
    # Detection metrics (all sentences)
    detection_precision: float
    detection_recall: float
    detection_f1: float
    detection_tp: int
    detection_tn: int
    detection_fp: int
    detection_fn: int
    detection_total: int
    
    # Syntactic metrics (only gold rules)
    strict_accuracy: float
    strict_correct: int
    strict_total: int
    avg_predicate_jaccard: float
    std_predicate_jaccard: float
    avg_soft_jaccard: float
    std_soft_jaccard: float
    
    # Execution metrics - per-rule (only gold rules)
    execution_accuracy: Optional[float] = None
    execution_correct: Optional[int] = None
    execution_total: Optional[int] = None
    avg_execution_jaccard: Optional[float] = None
    std_execution_jaccard: Optional[float] = None
    avg_execution_precision: Optional[float] = None
    std_execution_precision: Optional[float] = None
    avg_execution_recall: Optional[float] = None
    std_execution_recall: Optional[float] = None
    avg_execution_f1: Optional[float] = None
    std_execution_f1: Optional[float] = None
    
    # Execution metrics - union-based (global)
    union_precision: Optional[float] = None
    union_recall: Optional[float] = None
    union_f1: Optional[float] = None
    total_gold_violations: Optional[int] = None
    total_pred_violations: Optional[int] = None
    total_overlap_violations: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            'detection_precision': self.detection_precision,
            'detection_recall': self.detection_recall,
            'detection_f1': self.detection_f1,
            'detection_tp': self.detection_tp,
            'detection_tn': self.detection_tn,
            'detection_fp': self.detection_fp,
            'detection_fn': self.detection_fn,
            'detection_total': self.detection_total,
            'strict_accuracy': self.strict_accuracy,
            'strict_correct': self.strict_correct,
            'strict_total': self.strict_total,
            'avg_predicate_jaccard': self.avg_predicate_jaccard,
            'avg_soft_jaccard': self.avg_soft_jaccard,
        }
        
        # Add execution metrics if available
        if self.execution_accuracy is not None:
            result.update({
                'execution_accuracy': self.execution_accuracy,
                'execution_correct': self.execution_correct,
                'execution_total': self.execution_total,
                'avg_execution_jaccard': self.avg_execution_jaccard,
                'avg_execution_precision': self.avg_execution_precision,
                'avg_execution_recall': self.avg_execution_recall,
                'avg_execution_f1': self.avg_execution_f1,
                'union_precision': self.union_precision,
                'union_recall': self.union_recall,
                'union_f1': self.union_f1,
                'total_gold_violations': self.total_gold_violations,
                'total_pred_violations': self.total_pred_violations,
                'total_overlap_violations': self.total_overlap_violations,
            })
        
        return result


class MappingEvaluator:
    """
    Evaluator for rule mapping with multiple evaluation levels.
    
    Evaluation levels:
    1. Detection: Precision/Recall/F1 for rule identification (all sentences)
    2. Syntactic: Exact match, Jaccard, Soft Jaccard (gold rules only)
    3. Execution: DB-based validation (gold rules only, requires dataframe)
       - Per-rule: Exact match, Jaccard, P/R/F1
       - Union-based: Global P/R/F1 across all violations
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
    
    def evaluate(self,
                 predictions: List[Tuple[str, Optional[List[str]]]]) -> Tuple[MappingMetrics, List[SentenceMappingResult]]:
        """
        Evaluate mapping predictions (sentence-level).
        
        Args:
            predictions: List of (sentence, predicates) tuples from mapper
        
        Returns:
            Tuple of (aggregate_metrics, per_sentence_results)
        """
        if not predictions:
            raise ValueError("No predictions to evaluate")
        
        # Evaluate each sentence
        sentence_results = []
        for sentence, pred_rule in predictions:
            result = self._evaluate_single(sentence, pred_rule)
            sentence_results.append(result)
        
        # Compute aggregate metrics
        metrics = self._compute_metrics(sentence_results)
        
        return metrics, sentence_results
    
    def evaluate_document_level(self,
                               predicted_rules: List[List[str]],
                               gold_rules: List[List[str]]) -> Dict[str, Any]:
        """
        Evaluate document-level predictions.
        
        Computes:
        1. Constraint discovery metrics (precision/recall/F1 at constraint level)
        2. Violation discovery metrics (aggregate precision/recall/F1)
        
        Args:
            predicted_rules: List of predicted denial constraints
            gold_rules: List of ground truth denial constraints
        
        Returns:
            Dict with constraint-level and violation-level metrics
        """
        if self.db is None:
            raise ValueError("Database required for document-level evaluation")
        
        print(f"\nEvaluating {len(predicted_rules)} predicted vs {len(gold_rules)} gold constraints")
        
        # Execute all gold rules and store per-constraint violations
        print("Executing gold constraints...")
        gold_violations_list = []
        gold_violations_union = set()
        for i, dc in enumerate(gold_rules, 1):
            violations = self._execute_rule(dc)
            gold_violations_list.append(violations)
            gold_violations_union.update(violations)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(gold_rules)} gold constraints")
        
        print(f"  Total unique gold violations: {len(gold_violations_union)}")
        
        # Execute all predicted rules and store per-constraint violations
        print("Executing predicted constraints...")
        pred_violations_list = []
        pred_violations_union = set()
        for i, dc in enumerate(predicted_rules, 1):
            violations = self._execute_rule(dc)
            pred_violations_list.append(violations)
            pred_violations_union.update(violations)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(predicted_rules)} predicted constraints")
        
        print(f"  Total unique pred violations: {len(pred_violations_union)}")
        
        # === CONSTRAINT-LEVEL MATCHING ===
        # Match predicted constraints to gold constraints based on violation overlap
        print("\nMatching constraints by violation overlap...")
        
        matched_pred_indices = set()
        matched_gold_indices = set()
        
        # For each predicted constraint, find best matching gold constraint
        for pred_idx, pred_viols in enumerate(pred_violations_list):
            if not pred_viols:  # Skip empty violation sets
                continue
            
            best_match_idx = -1
            best_jaccard = 0.0
            
            for gold_idx, gold_viols in enumerate(gold_violations_list):
                if not gold_viols:
                    continue
                
                overlap = len(pred_viols & gold_viols)
                union = len(pred_viols | gold_viols)
                jaccard = overlap / union if union > 0 else 0.0
                
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_match_idx = gold_idx
            
            # Consider it a match if Jaccard >= 0.5 (at least 50% overlap)
            if best_jaccard >= 0.5:
                matched_pred_indices.add(pred_idx)
                matched_gold_indices.add(best_match_idx)
        
        # Compute constraint-level precision/recall/F1
        num_matched_preds = len(matched_pred_indices)
        num_matched_golds = len(matched_gold_indices)
        
        constraint_precision = num_matched_preds / len(predicted_rules) if len(predicted_rules) > 0 else 0.0
        constraint_recall = num_matched_golds / len(gold_rules) if len(gold_rules) > 0 else 0.0
        constraint_f1 = (2 * constraint_precision * constraint_recall / (constraint_precision + constraint_recall)
                        if (constraint_precision + constraint_recall) > 0 else 0.0)
        
        print(f"  Matched predicted: {num_matched_preds}/{len(predicted_rules)}")
        print(f"  Matched gold: {num_matched_golds}/{len(gold_rules)}")
        
        # === VIOLATION-LEVEL METRICS (AGGREGATE) ===
        # Compute union metrics
        overlap_violations = gold_violations_union & pred_violations_union
        
        violation_precision = len(overlap_violations) / len(pred_violations_union) if len(pred_violations_union) > 0 else 0.0
        violation_recall = len(overlap_violations) / len(gold_violations_union) if len(gold_violations_union) > 0 else 0.0
        violation_f1 = (2 * violation_precision * violation_recall / (violation_precision + violation_recall)
                       if (violation_precision + violation_recall) > 0 else 0.0)
        
        return {
            # Constraint discovery metrics
            'constraint_precision': constraint_precision,
            'constraint_recall': constraint_recall,
            'constraint_f1': constraint_f1,
            'num_matched_pred_constraints': num_matched_preds,
            'num_matched_gold_constraints': num_matched_golds,
            'num_gold_constraints': len(gold_rules),
            'num_pred_constraints': len(predicted_rules),
            # Violation discovery metrics (aggregate)
            'violation_precision_aggregate': violation_precision,
            'violation_recall_aggregate': violation_recall,
            'violation_f1_aggregate': violation_f1,
            'total_gold_violations': len(gold_violations_union),
            'total_pred_violations': len(pred_violations_union),
            'total_overlap_violations': len(overlap_violations),
        }
    
    def _evaluate_single(self, 
                        sentence: str, 
                        pred_rule: Optional[List[str]]) -> SentenceMappingResult:
        """Evaluate a single sentence prediction"""
        # Get ground truth
        gold_rule = self.ground_truth.get(sentence, None)
        gold_is_rule = gold_rule is not None and len(gold_rule) > 0
        pred_is_rule = pred_rule is not None and len(pred_rule) > 0
        
        # Detection status
        if gold_is_rule and pred_is_rule:
            detection_status = 'TP'
        elif not gold_is_rule and not pred_is_rule:
            detection_status = 'TN'
        elif not gold_is_rule and pred_is_rule:
            detection_status = 'FP'
        else:
            detection_status = 'FN'
        
        # For non-rules, return early
        if not gold_is_rule:
            return SentenceMappingResult(
                sentence=sentence,
                gold_is_rule=False,
                pred_is_rule=pred_is_rule,
                detection_status=detection_status,
                pred_rule=pred_rule
            )
        
        # For gold rules: compute syntactic metrics
        gold_set = self._rule_to_predicate_set(gold_rule)
        
        # Strict correctness
        strict_correct = False
        if pred_is_rule:
            pred_set = self._rule_to_predicate_set(pred_rule)
            strict_correct = (gold_set == pred_set)
        
        # Jaccard similarity
        if not pred_is_rule:
            predicate_jaccard = 0.0
            soft_jaccard = 0.0
        else:
            pred_set = self._rule_to_predicate_set(pred_rule)
            predicate_jaccard = self._jaccard_similarity(gold_set, pred_set)
            soft_jaccard = self._soft_jaccard_similarity(gold_set, pred_set)
        
        # Execution metrics
        exec_exact = None
        exec_jaccard = None
        exec_precision = None
        exec_recall = None
        exec_f1 = None
        gold_violations = None
        pred_violations = None
        
        if self.db is not None:
            gold_violations = self._execute_rule(gold_rule)
            pred_violations = self._execute_rule(pred_rule) if pred_is_rule else set()
            
            # Exact match
            exec_exact = (gold_violations == pred_violations)
            
            # Jaccard
            exec_jaccard = self._set_jaccard(gold_violations, pred_violations)
            
            # Precision/Recall/F1
            if len(pred_violations) > 0 or len(gold_violations) > 0:
                overlap = len(gold_violations & pred_violations)
                exec_precision = overlap / len(pred_violations) if len(pred_violations) > 0 else 0.0
                exec_recall = overlap / len(gold_violations) if len(gold_violations) > 0 else 0.0
                exec_f1 = (2 * exec_precision * exec_recall / (exec_precision + exec_recall) 
                          if (exec_precision + exec_recall) > 0 else 0.0)
        
        return SentenceMappingResult(
            sentence=sentence,
            gold_is_rule=True,
            pred_is_rule=pred_is_rule,
            detection_status=detection_status,
            strict_correct=strict_correct,
            predicate_jaccard=predicate_jaccard,
            soft_jaccard=soft_jaccard,
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
    
    def _compute_metrics(self, results: List[SentenceMappingResult]) -> MappingMetrics:
        """Compute aggregate metrics from sentence results"""
        # Detection metrics (all sentences)
        tp = sum(1 for r in results if r.detection_status == 'TP')
        tn = sum(1 for r in results if r.detection_status == 'TN')
        fp = sum(1 for r in results if r.detection_status == 'FP')
        fn = sum(1 for r in results if r.detection_status == 'FN')
        total = len(results)
        
        det_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        det_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        det_f1 = (2 * det_precision * det_recall / (det_precision + det_recall) 
                  if (det_precision + det_recall) > 0 else 0.0)
        
        # Syntactic metrics (only gold rules)
        gold_rules = [r for r in results if r.gold_is_rule]
        strict_correct = sum(1 for r in gold_rules if r.strict_correct)
        strict_total = len(gold_rules)
        strict_accuracy = strict_correct / strict_total if strict_total > 0 else 0.0
        
        jaccard_scores = [r.predicate_jaccard for r in gold_rules if r.predicate_jaccard is not None]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
        std_jaccard = float(np.std(jaccard_scores)) if len(jaccard_scores) > 1 else 0.0
        
        soft_jaccard_scores = [r.soft_jaccard for r in gold_rules if r.soft_jaccard is not None]
        avg_soft_jaccard = sum(soft_jaccard_scores) / len(soft_jaccard_scores) if soft_jaccard_scores else 0.0
        std_soft_jaccard = float(np.std(soft_jaccard_scores)) if len(soft_jaccard_scores) > 1 else 0.0
        
        # Execution metrics
        exec_accuracy = None
        exec_correct = None
        exec_total = None
        avg_exec_jaccard = None
        std_exec_jaccard = None
        avg_exec_precision = None
        std_exec_precision = None
        avg_exec_recall = None
        std_exec_recall = None
        avg_exec_f1 = None
        std_exec_f1 = None
        union_precision = None
        union_recall = None
        union_f1 = None
        total_gold_viols = None
        total_pred_viols = None
        total_overlap = None
        
        if self.db is not None:
            # Per-rule execution metrics
            exec_exact_matches = [r for r in gold_rules if r.execution_exact is not None and r.execution_exact]
            exec_correct = len(exec_exact_matches)
            exec_total = len([r for r in gold_rules if r.execution_exact is not None])
            exec_accuracy = exec_correct / exec_total if exec_total > 0 else 0.0
            
            exec_jaccard_scores = [r.execution_jaccard for r in gold_rules if r.execution_jaccard is not None]
            avg_exec_jaccard = sum(exec_jaccard_scores) / len(exec_jaccard_scores) if exec_jaccard_scores else 0.0
            std_exec_jaccard = float(np.std(exec_jaccard_scores)) if len(exec_jaccard_scores) > 1 else 0.0
            
            exec_prec_scores = [r.execution_precision for r in gold_rules if r.execution_precision is not None]
            avg_exec_precision = sum(exec_prec_scores) / len(exec_prec_scores) if exec_prec_scores else 0.0
            std_exec_precision = float(np.std(exec_prec_scores)) if len(exec_prec_scores) > 1 else 0.0
            
            exec_rec_scores = [r.execution_recall for r in gold_rules if r.execution_recall is not None]
            avg_exec_recall = sum(exec_rec_scores) / len(exec_rec_scores) if exec_rec_scores else 0.0
            std_exec_recall = float(np.std(exec_rec_scores)) if len(exec_rec_scores) > 1 else 0.0
            
            exec_f1_scores = [r.execution_f1 for r in gold_rules if r.execution_f1 is not None]
            avg_exec_f1 = sum(exec_f1_scores) / len(exec_f1_scores) if exec_f1_scores else 0.0
            std_exec_f1 = float(np.std(exec_f1_scores)) if len(exec_f1_scores) > 1 else 0.0
            
            # Union-based metrics
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
            
            union_precision = total_overlap / total_pred_viols if total_pred_viols > 0 else 0.0
            union_recall = total_overlap / total_gold_viols if total_gold_viols > 0 else 0.0
            union_f1 = (2 * union_precision * union_recall / (union_precision + union_recall)
                       if (union_precision + union_recall) > 0 else 0.0)
        
        return MappingMetrics(
            detection_precision=det_precision,
            detection_recall=det_recall,
            detection_f1=det_f1,
            detection_tp=tp,
            detection_tn=tn,
            detection_fp=fp,
            detection_fn=fn,
            detection_total=total,
            strict_accuracy=strict_accuracy,
            strict_correct=strict_correct,
            strict_total=strict_total,
            avg_predicate_jaccard=avg_jaccard,
            std_predicate_jaccard=std_jaccard,
            avg_soft_jaccard=avg_soft_jaccard,
            std_soft_jaccard=std_soft_jaccard,
            execution_accuracy=exec_accuracy,
            execution_correct=exec_correct,
            execution_total=exec_total,
            avg_execution_jaccard=avg_exec_jaccard,
            std_execution_jaccard=std_exec_jaccard,
            avg_execution_precision=avg_exec_precision,
            std_execution_precision=std_exec_precision,
            avg_execution_recall=avg_exec_recall,
            std_execution_recall=std_exec_recall,
            avg_execution_f1=avg_exec_f1,
            std_execution_f1=std_exec_f1,
            union_precision=union_precision,
            union_recall=union_recall,
            union_f1=union_f1,
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