"""Evaluator for sentence-level rule classification"""
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SentenceClassificationResult:
    """Result for a single sentence classification"""
    sentence: str
    gold_is_rule: bool
    pred_is_rule: bool
    status: str  # 'TP', 'TN', 'FP', 'FN'


@dataclass
class ClassificationMetrics:
    """Aggregate classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total': self.total
        }


class ClassificationEvaluator:
    """
    Evaluator for sentence-level rule classification.
    
    Compares predicted classifications against ground truth to compute:
    - Accuracy: Overall correctness
    - Precision: Of predicted rules, how many are correct
    - Recall: Of actual rules, how many were detected
    - F1 Score: Harmonic mean of precision and recall
    """
    
    def __init__(self, ground_truth: Dict[str, List[str]]):
        """
        Args:
            ground_truth: Dict mapping sentences to their denial constraints
                         (empty list or None = not a rule, non-empty list = rule)
        """
        self.ground_truth = ground_truth
    
    def evaluate(self, 
                 predictions: List[Tuple[str, bool]]) -> Tuple[ClassificationMetrics, List[SentenceClassificationResult]]:
        """
        Evaluate classification predictions against ground truth.
        
        Args:
            predictions: List of (sentence, is_rule) tuples from classifier
        
        Returns:
            Tuple of (aggregate_metrics, per_sentence_results)
        """
        if not predictions:
            raise ValueError("No predictions to evaluate")
        
        # Evaluate each sentence
        sentence_results = []
        for sentence, pred_is_rule in predictions:
            result = self._evaluate_single(sentence, pred_is_rule)
            sentence_results.append(result)
        
        # Compute aggregate metrics
        metrics = self._compute_metrics(sentence_results)
        
        return metrics, sentence_results
    
    def _evaluate_single(self, sentence: str, pred_is_rule: bool) -> SentenceClassificationResult:
        """Evaluate a single prediction"""
        # Get ground truth for this sentence
        gold_dc = self.ground_truth.get(sentence, None)
        gold_is_rule = gold_dc is not None and len(gold_dc) > 0
        
        # Determine status
        if gold_is_rule and pred_is_rule:
            status = 'TP'
        elif not gold_is_rule and not pred_is_rule:
            status = 'TN'
        elif not gold_is_rule and pred_is_rule:
            status = 'FP'
        else:  # gold_is_rule and not pred_is_rule
            status = 'FN'
        
        return SentenceClassificationResult(
            sentence=sentence,
            gold_is_rule=gold_is_rule,
            pred_is_rule=pred_is_rule,
            status=status
        )
    
    def _compute_metrics(self, results: List[SentenceClassificationResult]) -> ClassificationMetrics:
        """Compute aggregate metrics from sentence results"""
        # Count status types
        tp = sum(1 for r in results if r.status == 'TP')
        tn = sum(1 for r in results if r.status == 'TN')
        fp = sum(1 for r in results if r.status == 'FP')
        fn = sum(1 for r in results if r.status == 'FN')
        total = len(results)
        
        # Compute metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total=total
        )