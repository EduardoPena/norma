
"""Document-level mapper: extracts all DCs from entire document"""
from typing import List, Dict, Any
from src.llm.client import get_llm_client
from src.llm.helpers import PromptLoader, parse_json_response


class LLMDocumentMapper:
    """
    Extract all denial constraints from document.
    
    Processes all sentences together in ONE API call.
    Returns list of all DCs found (count may differ from # sentences).
    """
    
    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 prompt_name: str = "document_extraction"):
        """
        Args:
            model_name: LLM model identifier
            prompt_name: Prompt for document extraction
        """
        self.llm = get_llm_client(model_name)
        
        loader = PromptLoader()
        self.prompt_template = loader.load("mapping", prompt_name)
    
    def extract(self,
                sentences: List[str],
                context: Dict[str, Any]) -> List[List[str]]:
        """
        Extract all DCs from document.
        
        Args:
            sentences: All sentences in document
            context: Context dictionary
            
        Returns:
            List of DCs (each DC is List[str] of predicates)
        """
        # Build prompt
        prompt = self._build_prompt(sentences, context)
        
        # One API call for entire document
        response = self.llm.complete(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def _build_prompt(self, sentences: List[str], context: Dict[str, Any]) -> str:
        """Build document extraction prompt."""
        # Number all sentences
        sentences_text = "\n".join(
            f"{i+1}. {sentence}"
            for i, sentence in enumerate(sentences)
        )
        
        table_context = context.get("table_sample", "")
        
        return self.prompt_template.format(
            sentences=sentences_text,
            num_sentences=len(sentences),
            table_context=table_context
        )
    
    def _parse_response(self, response: str) -> List[List[str]]:
        """
        Parse response containing all DCs.
        
        Expected: [["pred1", "pred2"], ["pred3"], ...]
        """
        try:
            parsed = parse_json_response(response, expected_type=list)
            
            if parsed is None:
                return []
            
            # Validate each DC
            dcs = []
            for item in parsed:
                if isinstance(item, list) and all(isinstance(p, str) for p in item):
                    dcs.append(item)
            
            return dcs
        
        except Exception as e:
            print(f"Parse error: {e}")
            return []