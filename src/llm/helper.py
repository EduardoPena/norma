"""Shared LLM utilities: prompt loading and common parsing helpers"""
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import re
import json


class PromptLoader:
    """Loads prompts from YAML files with caching."""

    _cache: Dict[str, Dict[str, str]] = {}

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompts_dir = Path(prompts_dir)

    def load(self, category: str, prompt_name: str) -> str:
        """
        Load a prompt from YAML file.

        Args:
            category: Category file name (e.g., "classification", "mapping")
            prompt_name: Name of the prompt within the YAML file

        Returns:
            Prompt template string

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            KeyError: If prompt name not found in file
        """
        # Check cache first
        if category in self._cache:
            prompts = self._cache[category]
        else:
            # Load from file
            yaml_path = self.prompts_dir / f"{category}.yaml"

            if not yaml_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

            with open(yaml_path, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)

            # Cache for future use
            self._cache[category] = prompts

        # Get specific prompt
        if prompt_name not in prompts:
            available = ", ".join(prompts.keys())
            raise KeyError(
                f"Prompt '{prompt_name}' not found in {category}.yaml. "
                f"Available: {available}"
            )

        return prompts[prompt_name]

    def list_prompts(self, category: str) -> List[str]:
        """
        List all available prompts in a category.

        Args:
            category: Category file name

        Returns:
            List of prompt names
        """
        yaml_path = self.prompts_dir / f"{category}.yaml"

        if not yaml_path.exists():
            return []

        with open(yaml_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        return list(prompts.keys())

    @classmethod
    def clear_cache(cls):
        """Clear the prompt cache (useful for testing/reloading)."""
        cls._cache.clear()


# ============================================================================
# Common Parsing Helpers
# ============================================================================

def parse_numbered_labels(response: str,
                          positive_label: str,
                          negative_label: str,
                          expected_count: Optional[int] = None) -> List[bool]:
    """
    Parse numbered classification responses.

    Args:
        response: LLM response like "1. YES\n2. NO\n3. YES"
        positive_label: Label for True (e.g., "YES", "RULE")
        negative_label: Label for False (e.g., "NO", "NOT_RULE")
        expected_count: Expected number of predictions (validates output)

    Returns:
        List of boolean predictions

    Raises:
        ValueError: If count doesn't match expected_count
    """
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

    predictions = []
    for line in lines:
        line_upper = line.upper()
        pos_upper = positive_label.upper()
        neg_upper = negative_label.upper()
        
        if pos_upper in line_upper and neg_upper not in line_upper:
            predictions.append(True)
        elif neg_upper in line_upper:
            predictions.append(False)

    if expected_count is not None and len(predictions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} predictions, got {len(predictions)}. "
            f"Response: {response}"
        )

    return predictions




def parse_json_response(response: str, expected_type: type = list):
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: LLM response that may contain JSON
        expected_type: Expected type (list, dict, etc.)

    Returns:
        Parsed JSON object (or None if response is "null")

    Raises:
        ValueError: If parsing fails or type doesn't match
    """
    response = response.strip()

   

    # Drop opening fence if present (even if unclosed)
    response = re.sub(r"^\s*```(?:json)?\s*\n?", "", response, flags=re.IGNORECASE).strip()

    # Drop trailing fence if present
    response = re.sub(r"\s*```+\s*$", "", response).strip()


    

    

    # print(f"Parsing response:\n{response}\nExpected type: {expected_type.__name__}")
    
    # Handle explicit null
    if response.lower() == "null":
        return None
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()








    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}")

    # Allow None as valid response
    if parsed is None:
        return None
    

    # print(f"Parsed JSON:\n{parsed}\nType: {type(parsed).__name__}")

    # normalize: [ [ ["a","b"] ], [ ["c","d"] ] ] -> [ ["a","b"], ["c","d"] ]
    ## for mistral model; aparently it keeps sending nested lists for some reason, even when we ask for a flat list of lists.
    if isinstance(parsed, list) and parsed and all(isinstance(x, list) and len(x) == 1 and isinstance(x[0], list) for x in parsed):
        parsed = [x[0] for x in parsed]

        # print(f"Normalized nested lists to expected format:\n{parsed}")


    if not isinstance(parsed, expected_type):
        raise ValueError(
            f"Expected {expected_type.__name__}, got {type(parsed).__name__}"
        )

    return parsed