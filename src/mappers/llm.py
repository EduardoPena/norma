"""LLM-based constraint mapper"""

from typing import List, Dict, Any, Optional
from src.mappers.base import BaseMapper
from src.llm.client import get_llm_client
from src.llm.helper import PromptLoader, parse_json_response


class LLMMapper(BaseMapper):
    """LLM-based mapper that extracts predicates from rule sentences."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        prompt_name: str = "zero_shot",
        prompt_template: Optional[str] = None,
        allowed_predicates: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            model_name: LLM model identifier
            provider: Optional LLM provider override
            prompt_name: Name of prompt in prompts/mapping.yaml
            prompt_template: Direct prompt string (overrides prompt_name)
            allowed_predicates: List of allowed predicates for validation
            chunk_size: Max sentences per API call
        """
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name, provider=provider)
        self.allowed_predicates = (
            set(allowed_predicates) if allowed_predicates else None
        )

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            loader = PromptLoader()
            self.prompt_template = loader.load("mapping", prompt_name)

    def _map_sentences(
        self, sentences: List[str], context: Dict[str, Any]
    ) -> List[Optional[List[str]]]:
        """Map sentences to predicates using LLM."""
        prompt = self._build_prompt(sentences, context)
        response = self.llm.complete(prompt)
        return self._parse_response(response, len(sentences))

    def _build_prompt(self, sentences: List[str], context: Dict[str, Any]) -> str:
        """Build mapping prompt."""
        # Format sentences
        if len(sentences) == 1:
            formatted_sentences = sentences[0]
        else:
            formatted_sentences = "\n".join(
                [f"{i}. {sentence}" for i, sentence in enumerate(sentences, 1)]
            )

        # Prepare context variables
        predicates_list = ""
        if self.allowed_predicates:
            predicates_list = "\n".join(
                [f"  - {p}" for p in sorted(self.allowed_predicates)]
            )

        table_context = context.get("table_sample", "")

        # Format template
        prompt = self.prompt_template.format(
            sentences=formatted_sentences,
            predicates_list=predicates_list,
            table_context=table_context,
        )

        return prompt

    def _parse_response(
        self, response: str, expected_count: int
    ) -> List[Optional[List[str]]]:
        """
        Parse LLM response into predicates.

        Expected formats:
        - Single sentence: ["pred1", "pred2"] or null
        - Multiple sentences: [["pred1"], null, ["pred2", "pred3"]]
          (array with exactly expected_count elements)

        Args:
            response: LLM response
            expected_count: Number of sentences processed

        Returns:
            List of predicate lists (or None for non-rules)

        Raises:
            ValueError: If response doesn't contain expected_count results
        """
        try:
            parsed = parse_json_response(response, expected_type=list)

            # Handle null response - treat as empty for all inputs
            if parsed is None:
                print(f"Warning: Got null response for {expected_count} sentences")
                return [None] * expected_count

            # Single sentence case
            if expected_count == 1:
                # Check if it's in batch format (array of arrays)
                if len(parsed) > 0 and isinstance(parsed[0], (list, type(None))):
                    # Batch format: [["pred1", "pred2"]] or [null]
                    if len(parsed) != 1:
                        raise ValueError(
                            f"Single sentence expected 1 result, got {len(parsed)}. "
                            f"Response: {response}"
                        )
                    return parsed
                else:
                    # Direct format: ["pred1", "pred2"]
                    if self._validate_predicate_list(parsed):
                        return [parsed]
                    else:
                        print(f"Warning: Invalid predicate list format: {parsed}")
                        return [None]

            # Multiple sentences - MUST be batch format
            if len(parsed) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} results, got {len(parsed)}. "
                    f"Each input sentence must have a corresponding output (use null for non-constraints). "
                    f"Response: {response}"
                )

            # Validate each item in the batch
            results = []
            for i, item in enumerate(parsed):
                if item is None:
                    results.append(None)
                elif isinstance(item, list):
                    if self._validate_predicate_list(item):
                        results.append(item)
                    else:
                        print(f"Warning: Invalid predicate list at index {i}: {item}")
                        results.append(None)
                else:
                    print(
                        f"Warning: Expected list or null at index {i}, got {type(item).__name__}: {item}"
                    )
                    results.append(None)

            return results

        except ValueError as e:
            # Re-raise ValueError with context
            print(f"Parse error: {e}")
            raise
        except Exception as e:
            # Catch other errors and return None for all
            print(f"Unexpected parse error: {e}")
            return [None] * expected_count

    def _validate_predicate_list(self, predicates: List) -> bool:
        """
        Validate that a list contains only strings (predicates).

        Args:
            predicates: List to validate

        Returns:
            True if valid predicate list
        """
        if not isinstance(predicates, list):
            return False

        for pred in predicates:
            if not isinstance(pred, str):
                return False

        return True
