"""LLM-based rule classifier"""

from typing import List, Dict, Any, Optional
import re
from .base import BaseRuleClassifier
from ..llm import get_llm_client


class LLMRuleClassifier(BaseRuleClassifier):
    """LLM-based rule classifier using prompt templates."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        provider: Optional[str] = None,
        prompt_template: Optional[str] = None,
        chunk_size: int = 20,
    ):
        """
        Args:
            model_name: LLM model identifier
            provider: Optional LLM provider override
            prompt_template: Prompt string with {sentences} placeholder
            chunk_size: Max sentences per API call
        """
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name, provider=provider)
        self.prompt_template = prompt_template or self._default_prompt()

    def _default_prompt(self) -> str:
        """Default classification prompt."""
        return """For each sentence, answer YES if it describes a data quality constraint, or NO if it does not.

{sentences}

Answer with only numbered lines: 1. YES or 1. NO"""

    def _classify_sentences(
        self, sentences: List[str], context: Dict[str, Any]
    ) -> List[bool]:
        """Classify sentences using LLM."""
        formatted = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

        # Prepare template variables with defaults
        template_vars = {
            "sentences": formatted,
            "table_context": context.get("table_context", ""),
        }

        prompt = self.prompt_template.format(**template_vars)

        response = self.llm.complete(prompt)
        return self._parse_response(response, len(sentences))

    def _parse_response(self, response: str, expected_count: int) -> List[bool]:
        """Parse LLM response into boolean predictions."""
        predictions = []
        for line in response.strip().split("\n"):
            match = re.match(r"\s*\d+\.\s*(YES|NO)", line.strip(), re.IGNORECASE)
            if match:
                predictions.append(match.group(1).upper() == "YES")

        while len(predictions) < expected_count:
            predictions.append(False)

        return predictions[:expected_count]
