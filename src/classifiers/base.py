"""Base classifier for rule detection in regulatory text"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from src.utils.parallel import default_parallel_requests, parallel_map


class BaseRuleClassifier(ABC):
    """
    Base class for rule classification.
    
    Implementations only need to override `_classify_sentences()`.
    Chunking is handled automatically.
    """

    def __init__(self, chunk_size: Optional[int] = None):
        """
        Args:
            chunk_size: Max sentences per API call (None = process all at once)
        """
        self.chunk_size = chunk_size
        self.parallel_requests = default_parallel_requests()

    def classify(self,
                 sentences: List[str],
                 context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, bool]]:
        """
        Classify sentences as rules or non-rules.
        
        Args:
            sentences: List of sentences to classify
            context: Optional context dictionary (e.g., from TableContextBuilder)
            
        Returns:
            List of (sentence, is_rule) tuples
        """
        if not sentences:
            return []
        
        # Use empty context if not provided
        context = context or {}
        
        # Process in chunks if chunk_size is set
        if self.chunk_size and len(sentences) > self.chunk_size:
            chunks = [
                sentences[start : start + self.chunk_size]
                for start in range(0, len(sentences), self.chunk_size)
            ]
            results = parallel_map(
                lambda chunk: self._classify_sentences(chunk, context),
                chunks,
                getattr(self, "parallel_requests", 1),
            )
            predictions = [label for chunk_labels in results for label in chunk_labels]
        else:
            predictions = self._classify_sentences(sentences, context)
        
        # Validate output
        if len(predictions) != len(sentences):
            raise ValueError(
                f"Classifier returned {len(predictions)} predictions "
                f"for {len(sentences)} sentences"
            )
        
        return list(zip(sentences, predictions))

    @abstractmethod
    def _classify_sentences(self,
                           sentences: List[str],
                           context: Dict[str, Any]) -> List[bool]:
        """
        Implementation-specific classification logic.
        
        Args:
            sentences: Sentences to classify (respects chunk_size if set)
            context: Context dictionary (may be empty)
                
        Returns:
            List of boolean predictions (same length as sentences)
        """
        pass