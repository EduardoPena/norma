"""NORMA Pipeline: Flexible constraint extraction with three processing modes"""
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

from src.classifiers.base import BaseRuleClassifier
from src.mappers.base import BaseMapper
from src.mappers.llm_document_mapper import LLMDocumentMapper
from src.utils.result_saver import ResultSaver  # No pandas dependency


class ProcessingMode(Enum):
    """NORMA processing modes."""
    
    SENTENCE_DIRECT = "sentence_direct"
    SENTENCE_FILTERED = "sentence_filtered"
    DOCUMENT = "document"


class NormaPipeline:
    """
    NORMA: Natural language tO constRaint MApping.
    
    Three processing modes:
    - SENTENCE_DIRECT: Map each sentence independently
    - SENTENCE_FILTERED: Classify first, then map rules only
    - DOCUMENT: Extract all DCs from entire document
    """
    
    def __init__(self,
                 mode: ProcessingMode,
                 classifier: Optional[BaseRuleClassifier] = None,
                 mapper: Optional[BaseMapper] = None,
                 document_mapper: Optional[LLMDocumentMapper] = None):
        """
        Args:
            mode: Processing mode
            classifier: Sentence classifier (required for SENTENCE_FILTERED)
            mapper: Sentence mapper (required for SENTENCE_DIRECT, SENTENCE_FILTERED)
            document_mapper: Document mapper (required for DOCUMENT)
        """
        self.mode = mode
        self.classifier = classifier
        self.mapper = mapper
        self.document_mapper = document_mapper
        
        # Store results for optional saving
        self.classification_results = None
        self.mapping_results = None
        
        self._validate()
    
    def _validate(self):
        """Validate required components."""
        if self.mode == ProcessingMode.SENTENCE_DIRECT and not self.mapper:
            raise ValueError("Mapper required for SENTENCE_DIRECT")
        if self.mode == ProcessingMode.SENTENCE_FILTERED and (not self.classifier or not self.mapper):
            raise ValueError("Classifier and Mapper required for SENTENCE_FILTERED")
        if self.mode == ProcessingMode.DOCUMENT and not self.document_mapper:
            raise ValueError("DocumentMapper required for DOCUMENT")
    
    def process(self,
                sentences: List[str],
                context: Optional[Dict[str, Any]] = None) -> Union[
                    List[Tuple[str, bool, Optional[List[str]]]],
                    List[List[str]]
                ]:
        """
        Process sentences through pipeline.
        
        Args:
            sentences: List of sentences to process
            context: Optional context (e.g., table context)
        
        Returns:
            SENTENCE modes: [(sentence, is_rule, predicates), ...]
            DOCUMENT mode: [DC1, DC2, ...]
        """
        context = context or {}
        
        if self.mode == ProcessingMode.SENTENCE_DIRECT:
            return self._process_sentence_direct(sentences, context)
        elif self.mode == ProcessingMode.SENTENCE_FILTERED:
            return self._process_sentence_filtered(sentences, context)
        else:  # DOCUMENT
            return self._process_document(sentences, context)
    
    def _process_sentence_direct(self, sentences, context):
        """Map each sentence directly."""
        print(f"\n[SENTENCE_DIRECT] Processing {len(sentences)} sentences")
        
        self.mapping_results = self.mapper.map(sentences, context=context)
        
        results = [
            (sent, preds is not None, preds)
            for sent, preds in self.mapping_results
        ]
        
        mapped = sum(1 for _, is_mapped, _ in results if is_mapped)
        print(f"  Mapped: {mapped}/{len(sentences)}")
        
        return results
    
    def _process_sentence_filtered(self, sentences, context):
        """Classify first, then map rules."""
        print(f"\n[SENTENCE_FILTERED] Processing {len(sentences)} sentences")
        
        # Classify
        self.classification_results = self.classifier.classify(sentences, context=context)
        rule_sentences = [s for s, is_rule in self.classification_results if is_rule]
        print(f"  Phase 1: {len(rule_sentences)} rules")
        
        # Map rules
        if rule_sentences:
            self.mapping_results = self.mapper.map(rule_sentences, context=context)
            print(f"  Phase 2: Mapped {len(rule_sentences)} rules")
        else:
            self.mapping_results = []
        
        # Reconstruct
        results = []
        rule_iter = iter(self.mapping_results)
        for sent, is_rule in self.classification_results:
            if is_rule:
                _, preds = next(rule_iter)
                results.append((sent, True, preds))
            else:
                results.append((sent, False, None))
        
        return results
    
    def _process_document(self, sentences, context):
        """Extract all DCs from document."""
        print(f"\n[DOCUMENT] Processing {len(sentences)} sentences")
        
        all_dcs = self.document_mapper.extract(sentences, context=context)
        
        print(f"  Extracted: {len(all_dcs)} DCs")
        
        return all_dcs
    
    def save(self,
             results: Union[List[Tuple[str, bool, Optional[List[str]]]], List[List[str]]],
             output_file: str,
             ground_truth: Optional[Dict[str, List[str]]] = None):
        """
        Save results to CSV.
        
        Args:
            results: Pipeline results
            output_file: Output path (without extension)
            ground_truth: Optional {sentence: predicates} for comparison
        """
        if self.mode == ProcessingMode.DOCUMENT:
            ResultSaver.save_document_results(results, output_file)
        else:
            ResultSaver.save_sentence_results(
                results,
                output_file,
                mode=self.mode.value,
                ground_truth=ground_truth,
                classification_results=self.classification_results,
                mapping_results=self.mapping_results
            )
    
    def get_stats(self):
        """Get pipeline statistics."""
        return {
            "mode": self.mode.value,
            "classifier": type(self.classifier).__name__ if self.classifier else None,
            "mapper": type(self.mapper).__name__ if self.mapper else None,
            "document_mapper": type(self.document_mapper).__name__ if self.document_mapper else None
        }