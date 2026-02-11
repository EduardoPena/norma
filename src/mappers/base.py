"""Base mapper for constraint extraction from regulatory text"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any


class BaseMapper(ABC):
    """
    Base class for constraint mapping.
    
    Implementations only need to override `_map_sentences()`.
    Chunking is handled automatically.
    """

    def __init__(self, chunk_size: Optional[int] = None):
        """
        Args:
            chunk_size: Max sentences per API call (None = process all at once)
        """
        self.chunk_size = chunk_size

    def map(self,
            sentences: List[str],
            context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Optional[List[str]]]]:
        """
        Map sentences to constraint predicates.
        
        Args:
            sentences: List of sentences to map
            context: Optional context dictionary (e.g., from TableContextBuilder)
            
        Returns:
            List of (sentence, predicates) tuples where:
                - predicates is None if sentence is not a rule
                - predicates is List[str] for valid constraints
        """
        if not sentences:
            return []
        
        # Use empty context if not provided
        context = context or {}
        
        # Process in chunks if chunk_size is set
        if self.chunk_size and len(sentences) > self.chunk_size:
            all_predicates = []
            for start_idx in range(0, len(sentences), self.chunk_size):
                chunk = sentences[start_idx : start_idx + self.chunk_size]
                chunk_results = self._map_sentences(chunk, context)
                all_predicates.extend(chunk_results)
        else:
            all_predicates = self._map_sentences(sentences, context)
        
        # Validate output
        if len(all_predicates) != len(sentences):
            raise ValueError(
                f"Mapper returned {len(all_predicates)} results "
                f"for {len(sentences)} sentences"
            )
        
        return list(zip(sentences, all_predicates))

    @abstractmethod
    def _map_sentences(self,
                      sentences: List[str],
                      context: Dict[str, Any]) -> List[Optional[List[str]]]:
        """
        Implementation-specific mapping logic.
        
        Args:
            sentences: Sentences to map (respects chunk_size if set)
            context: Context dictionary (may be empty)
                
        Returns:
            List of predicate lists (or None for non-rules)
        """
        pass