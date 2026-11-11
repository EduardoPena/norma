"""LLM-based rule classifier"""

from typing import List, Dict, Any, Optional
import re
from .base import BaseRuleClassifier
from ..llm import get_llm_client
from ..llm.client import EmptyModelContentError
from ..llm.helper import parse_json_response
from ..llm.schemas import CLASSIFICATION


class LLMRuleClassifier(BaseRuleClassifier):
    """LLM-based rule classifier using prompt templates."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        provider: Optional[str] = None,
        prompt_template: Optional[str] = None,
        chunk_size: int = 20,
        structured_output: bool = True,
    ):
        """
        Args:
            model_name: LLM model identifier
            provider: Optional LLM provider override
            prompt_template: Prompt string with {sentences} placeholder
            chunk_size: Max sentences per API call
            structured_output: Request the CLASSIFICATION JSON Schema
        """
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(
            model_name, provider=provider, structured_output=structured_output
        )
        # Paper §4: malformed responses are retried; if they still fail, the
        # remaining sentences are classified individually; a sentence that
        # still has no label is marked non-constraint (logged here).
        self.max_output_retries = 2
        self.output_failures: List[str] = []
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

        attempts = self.max_output_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.llm.complete(
                    prompt, response_schema=CLASSIFICATION, schema_name="classification"
                )
                labels = self._parse_labels(response, len(sentences))
            except EmptyModelContentError as error:
                labels, problem = None, str(error)
            else:
                problem = "labels missing, duplicated or out of range"
            if labels is not None:
                return labels
            print(
                f"Warning: malformed classifier output for {len(sentences)} sentences "
                f"(attempt {attempt}/{attempts}): {problem}"
            )

        if len(sentences) > 1:
            print("Warning: falling back to classifying sentences individually")
            return [self._classify_sentences([s], context)[0] for s in sentences]
        print("Warning: no valid classifier label; sentence marked as non-constraint")
        self.output_failures.append(problem)
        return [False]

    @staticmethod
    def _parse_labels(response: str, expected_count: int) -> Optional[List[bool]]:
        """Return one label per numbered sentence, or None if malformed.

        Expects ``{"labels": [{"id": 1, "is_constraint": true}, ...]}``; numbered
        ``<n>. YES/NO`` lines are accepted as a fallback.
        """
        labels: Dict[int, bool] = {}
        try:
            payload = parse_json_response(response, expected_type=dict)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
            for item in payload["labels"]:
                if not (
                    isinstance(item, dict)
                    and isinstance(item.get("id"), int)
                    and isinstance(item.get("is_constraint"), bool)
                    and item["id"] not in labels
                ):
                    return None
                labels[item["id"]] = item["is_constraint"]
            if sorted(labels) != list(range(1, expected_count + 1)):
                return None
            return [labels[number] for number in range(1, expected_count + 1)]

        for line in response.strip().split("\n"):
            match = re.match(r"\s*(\d+)\.\s*(YES|NO)\b", line.strip(), re.IGNORECASE)
            if match:
                number = int(match.group(1))
                if number in labels:
                    return None
                labels[number] = match.group(2).upper() == "YES"
        if not labels and expected_count == 1:
            bare = re.fullmatch(r"\W*(YES|NO)\W*", response.strip(), re.IGNORECASE)
            if bare:
                return [bare.group(1).upper() == "YES"]
        if sorted(labels) != list(range(1, expected_count + 1)):
            return None
        return [labels[number] for number in range(1, expected_count + 1)]

