"""LLM-based constraint mapper"""

from typing import List, Dict, Any, Optional
from src.mappers.base import BaseMapper
from src.mappers.predicate_filter import (
    filter_predicates_by_columns,
    select_relevant_columns,
    simple_predicates,
)
from src.llm.client import EmptyModelContentError, get_llm_client
from src.llm.helper import PromptLoader, parse_json_response
from src.llm.schemas import SENTENCE_RULES
from src.predicates import Rule, RuleValidationError, normalize_rules, repair_rules


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
        filter_predicates: bool = False,
        predicate_tables: Optional[List[Dict[str, Any]]] = None,
        max_candidate_predicates: int = 1000,
        max_relevant_columns: int = 12,
        max_tokens: Optional[int] = None,
        structured_output: bool = True,
        predicate_generator: Optional[Any] = None,
    ):
        """
        Args:
            model_name: LLM model identifier
            provider: Optional LLM provider override
            prompt_name: Name of prompt in prompts/mapping.yaml
            prompt_template: Direct prompt string (overrides prompt_name)
            allowed_predicates: List of allowed predicates for validation
            chunk_size: Max sentences per API call
            filter_predicates: Preserve simple templates and filter compound candidates
            predicate_tables: Schema names/descriptions for deterministic column retrieval
            max_candidate_predicates: Candidate limit per sentence, before simple union
            max_relevant_columns: Maximum retrieved columns per sentence
            max_tokens: Optional OpenRouter completion token limit
            structured_output: Request the SENTENCE_RULES JSON Schema
            predicate_generator: v2 schema-only space (``OnDemandPredicates``): the
                prompt offers only the templates generated for the columns selected
                for each sentence (lexical selection, ``select_relevant_columns``)
        """
        if max_candidate_predicates <= 0 or max_relevant_columns <= 0:
            raise ValueError("Predicate candidate and column limits must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("Completion token limit must be positive")
        if predicate_generator is not None and not predicate_tables:
            raise ValueError("A v2 predicate space needs the schema tables")
        if (
            predicate_generator is None
            and filter_predicates
            and (not predicate_tables or not allowed_predicates)
        ):
            raise ValueError(
                "Predicate filtering requires schema tables and predicates"
            )
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(
            model_name,
            provider=provider,
            max_tokens=max_tokens,
            structured_output=structured_output,
        )
        self.allowed_predicates = (
            set(allowed_predicates) if allowed_predicates else None
        )
        self.filter_predicates = filter_predicates
        self.predicate_generator = predicate_generator
        self.predicate_tables = predicate_tables or []
        self.max_candidate_predicates = max_candidate_predicates
        self.max_relevant_columns = max_relevant_columns
        self._simple_predicates = (
            simple_predicates(self.allowed_predicates)
            if filter_predicates and self.allowed_predicates
            else set()
        )
        self.last_predicate_selection: List[Dict[str, Any]] = []
        # Paper §5.2: retry malformed outputs a fixed number of times; if all
        # attempts fail, the sentences produce no constraint (logged here).
        self.max_output_retries = 2
        self.output_failures: List[str] = []
        # Rules are validated against the schema (columns and declared types);
        # sentences with invalid rules are re-asked once with the error message.
        self.rule_schema = {"tables": self.predicate_tables} if self.predicate_tables else None
        self.max_rule_repairs = 1

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            loader = PromptLoader()
            self.prompt_template = loader.load("mapping", prompt_name)

    def _map_sentences(
        self, sentences: List[str], context: Dict[str, Any]
    ) -> List[Optional[List[Rule]]]:
        """Map sentences to predicates using LLM.

        Malformed outputs are retried; if every attempt fails, each sentence of
        the batch yields no constraint and is recorded in ``output_failures``.
        API/infrastructure errors still propagate.
        """
        prompt = self._build_prompt(sentences, context)
        attempts = self.max_output_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.llm.complete(
                    prompt, response_schema=SENTENCE_RULES, schema_name="sentence_rules"
                )
                items = self._result_items(
                    parse_json_response(response, expected_type=object), len(sentences)
                )
                return self._validate_items(sentences, items, context)
            except (ValueError, EmptyModelContentError) as error:
                last_error = str(error).split(" Response:")[0][:300]
                print(
                    f"Warning: malformed mapper output (attempt {attempt}/{attempts}): "
                    f"{last_error}"
                )
        print(
            f"Warning: no valid mapper output after {attempts} attempts; "
            f"{len(sentences)} sentences yield no constraint"
        )
        self.output_failures.extend([last_error] * len(sentences))
        return [None] * len(sentences)

    def _build_prompt(self, sentences: List[str], context: Dict[str, Any]) -> str:
        """Build mapping prompt."""
        # Number every sentence: results are keyed by these ids
        formatted_sentences = "\n".join(
            [f"{i}. {sentence}" for i, sentence in enumerate(sentences, 1)]
        )

        # Prepare context variables
        predicates_list = ""
        self.last_predicate_selection = []
        generator = getattr(self, "predicate_generator", None)
        if generator is not None:
            # v2: generate templates only for the columns selected per sentence
            offered: set = set()
            for sentence in sentences:
                columns = select_relevant_columns(
                    sentence, self.predicate_tables, self.max_relevant_columns
                )
                candidates = filter_predicates_by_columns(
                    generator.generate(columns),
                    columns,
                    sentence,
                    self.max_candidate_predicates,
                )
                offered.update(candidates)
                self.last_predicate_selection.append(
                    {"columns": columns, "candidates": len(candidates)}
                )
            print(
                f"Predicate generation (v2): {len(offered)} templates for "
                f"{len(sentences)} sentences"
            )
            predicates_list = "\n".join([f"  - {p}" for p in sorted(offered)])
        elif self.allowed_predicates:
            offered = self.allowed_predicates
            if self.filter_predicates:
                offered = set(self._simple_predicates)
                for sentence in sentences:
                    columns = select_relevant_columns(
                        sentence, self.predicate_tables, self.max_relevant_columns
                    )
                    candidates = filter_predicates_by_columns(
                        self.allowed_predicates,
                        columns,
                        sentence,
                        self.max_candidate_predicates,
                    )
                    offered.update(candidates)
                    self.last_predicate_selection.append(
                        {"columns": columns, "candidates": len(candidates)}
                    )
                print(
                    f"Predicate filtering: {len(self.allowed_predicates)} -> "
                    f"{len(offered)} templates for {len(sentences)} sentences"
                )
            predicates_list = "\n".join([f"  - {p}" for p in sorted(offered)])

        table_context = context.get("table_context", context.get("table_sample", ""))

        # Format template
        prompt = self.prompt_template.format(
            sentences=formatted_sentences,
            predicates_list=predicates_list,
            table_context=table_context,
        )

        return prompt

    def _parse_response(
        self, response: str, expected_count: int
    ) -> List[Optional[List[Rule]]]:
        """
        Parse LLM response into one rules list (or None) per sentence.

        Expected (SENTENCE_RULES schema): ``{"results": [{"id": 1, "rules": ...}]}``
        with ids 1..expected_count in any order. Legacy fallbacks: a positional
        array of rules lists (batch) or a bare rules list (single sentence).

        Raises:
            ValueError: If the response is not valid JSON or does not provide
                exactly one result per sentence (the caller retries the batch).
        """
        parsed = parse_json_response(response, expected_type=object)
        items = self._result_items(parsed, expected_count)
        return [self._sentence_rules(i, item) for i, item in enumerate(items, 1)]

    @staticmethod
    def _result_items(parsed: Any, expected_count: int) -> List[Any]:
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            by_id: Dict[int, Any] = {}
            for entry in parsed["results"]:
                if not (
                    isinstance(entry, dict)
                    and isinstance(entry.get("id"), int)
                    and "rules" in entry
                    and entry["id"] not in by_id
                ):
                    raise ValueError(f"Malformed or duplicated result entry: {entry}")
                by_id[entry["id"]] = entry["rules"]
            if sorted(by_id) != list(range(1, expected_count + 1)):
                raise ValueError(
                    f"Expected result ids 1..{expected_count}, got {sorted(by_id)}"
                )
            return [by_id[i] for i in range(1, expected_count + 1)]
        if parsed is None:
            print(f"Warning: Got null response for {expected_count} sentences")
            return [None] * expected_count
        if expected_count == 1:
            return [parsed]
        if isinstance(parsed, list) and len(parsed) == expected_count:
            return parsed
        raise ValueError(
            f"Expected {expected_count} results; use one result per sentence "
            "(rules null for non-constraints)"
        )

    def _sentence_rules(self, index: int, item: Any) -> Optional[List[Rule]]:
        rules, error = self._check_rules(item)
        if error:
            print(f"Warning: Invalid rules for sentence {index}: {error}")
            self.output_failures.append(f"invalid rules: {error}")
        return rules

    def _check_rules(self, item: Any) -> tuple:
        """(rules, None) when valid, (None, error message) otherwise."""
        if item is None or item == []:
            return None, None  # [] also means "not a constraint"
        try:
            return normalize_rules(repair_rules(item), getattr(self, "rule_schema", None)), None
        except (RuleValidationError, ValueError) as exc:
            return None, str(exc)

    def _validate_items(
        self, sentences: List[str], items: List[Any], context: Dict[str, Any]
    ) -> List[Optional[List[Rule]]]:
        """Validate each sentence's rules; re-ask invalid sentences with the error."""
        checked = [self._check_rules(item) for item in items]
        for _ in range(getattr(self, "max_rule_repairs", 0)):
            invalid = [i for i, (_, error) in enumerate(checked) if error]
            if not invalid:
                break
            print(f"Re-asking {len(invalid)} sentence(s) with invalid rules")
            subset = [sentences[i] for i in invalid]
            notes = "\n".join(
                f"- Sentence {n}: previous rules {items[i]!r} were rejected: {checked[i][1]}"
                for n, i in enumerate(invalid, 1)
            )
            prompt = (
                f"{self._build_prompt(subset, context)}\n\n"
                "CORRECTION REQUIRED: fix these outputs (sentence numbers refer to the "
                f"list above).\n{notes}\nUse only existing columns and compare each "
                "column with constants of its type; write missing values as IS NULL."
            )
            try:
                response = self.llm.complete(
                    prompt, response_schema=SENTENCE_RULES, schema_name="sentence_rules"
                )
                retried = self._result_items(
                    parse_json_response(response, expected_type=object), len(subset)
                )
            except (ValueError, EmptyModelContentError) as error:
                print(f"Warning: correction attempt failed: {str(error)[:200]}")
                break
            for i, item in zip(invalid, retried):
                items[i] = item
                checked[i] = self._check_rules(item)
        results = []
        for index, (rules, error) in enumerate(checked, 1):
            if error:
                print(f"Warning: Invalid rules for sentence {index}: {error}")
                self.output_failures.append(f"invalid rules: {error}")
            results.append(rules)
        return results
