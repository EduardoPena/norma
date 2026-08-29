#!/usr/bin/env python3
"""
Multi-Step Constraint Refinement Mapper


This mapper uses a multi-step refinement process:
1. Schema Understanding: Identify relevant columns and verify they exist
2. Predicate Selection: Choose predicates from allowed set and validate
3. Rule Gate: Verify the sentence is actually a mappable denial constraint
4. Constraint Construction: Build initial denial constraint
5. Verification: Test on database sample (with satisfiability + plausibility gates)
6. Refinement: Adjust based on validation results

At any point, can decide the sentence is NOT a constraint.

Gates implemented:
  Rule Gate (Step 3.5) — Semantic gate: verifies the sentence expresses a real,
       falsifiable data quality rule before paying the cost of construction.
  G1 — Satisfiability: any literal constant (=, !=, LIKE, NOT LIKE, IN, NOT IN)
       is checked against the actual data using LIMIT 1 for speed. If no rows
       contain that value the predicate is "dead" (invented) and the constraint
       is discarded immediately without attempting refinement.
  G2 — Violation plausibility (cross-row): if violations exceed
       violation_ratio_threshold × total_pairs the constraint is too loose /
       logically inverted and is sent for refinement.
  G3 — Zero-violation detection with inversion probe: if a cross-row constraint
       finds 0 violations, the operator signs are flipped and re-executed. When
       the inverted version produces violations the original was likely inverted
       by the LLM; the corrected version is returned immediately.

Usage:
    from src.mappers.multi_step_refinement import MultiStepRefinementMapper
    from src.data.loader import DatasetLoader

    loader = DatasetLoader('data/hospital')
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    mapper = MultiStepRefinementMapper(
        model_name='gpt-4o-mini',
        predicate_space=predicate_space,
        df=df,
        max_refinement_iterations=2
    )

    results = mapper.map(sentences)
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.mappers.base import BaseMapper
from src.llm.client import get_llm_client
from src.llm.helper import PromptLoader
from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator

# ============================================================================
# Shared Utilities
# ============================================================================

# def extract_json(text: str) -> str:
#     """Extract JSON from LLM response (handles markdown code blocks)."""
#     pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(1)

#     pattern = r'\{.*\}'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(0)

#     return text


import re


def _escape_control_chars_in_strings(s: str) -> str:
    result = []
    in_string = False
    escape = False

    for ch in s:
        if in_string:
            if escape:
                result.append(ch)
                escape = False
            elif ch == "\\":
                result.append(ch)
                escape = True
            elif ch == '"':
                result.append(ch)
                in_string = False
            elif ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            result.append(ch)
            if ch == '"':
                in_string = True

    return "".join(result)


def _extract_first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def extract_json(text: str) -> str:
    """Extract and lightly repair JSON from an LLM response."""
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        raw = _extract_first_balanced_json_object(text)

    repaired = _escape_control_chars_in_strings(raw)
    return repaired


def _normalize_constraint_list(constraint: Any) -> Optional[List[str]]:
    """Normalize LLM constraint output to a flat list of predicate strings."""
    if constraint is None or not isinstance(constraint, list):
        return None

    if constraint and all(isinstance(item, list) for item in constraint):
        if len(constraint) == 1 and all(
            isinstance(pred, str) for pred in constraint[0]
        ):
            constraint = constraint[0]
        else:
            return None

    if not all(isinstance(pred, str) for pred in constraint):
        return None

    return constraint


def _restore_percentage_point_units(
    sentence: str, constraint: Optional[List[str]]
) -> Optional[List[str]]:
    """Keep percentage-point thresholds in their original units for rate_spread-like columns."""
    if not constraint or "percentage point" not in sentence.lower():
        return constraint

    adjusted = []
    pattern = re.compile(
        r"(\b(?:t[12]\.)?\w*rate_spread\w*\s*(?:<=|>=|=|<|>)\s*)(-?\d+(?:\.\d+)?)"
    )

    for predicate in constraint:

        def replace(match: re.Match[str]) -> str:
            value = float(match.group(2))
            if 0 < abs(value) < 1:
                value *= 100
                if value.is_integer():
                    value_str = str(int(value))
                else:
                    value_str = str(value)
                return f"{match.group(1)}{value_str}"
            return match.group(0)

        adjusted.append(pattern.sub(replace, predicate))

    return adjusted


def token_similarity(s1: str, s2: str) -> float:
    """
    Compute similarity between two column names using token-based overlap.
    Splits on underscores so that days_to_next_inspection and days_to_action
    score more meaningfully than character set overlap.
    """
    t1 = set(s1.lower().split("_"))
    t2 = set(s2.lower().split("_"))
    if not t1 or not t2:
        return 0.0
    if s1.lower() == s2.lower():
        return 1.0
    if s1.lower() in s2.lower() or s2.lower() in s1.lower():
        return 0.8
    return len(t1 & t2) / len(t1 | t2)


# ============================================================================
# Multi-Step Refinement Mapper
# ============================================================================


class MultiStepRefinementMapper(BaseMapper):
    """
    Multi-step constraint mapper with iterative refinement.

    Key features:
    - Multi-step refinement with validation at each stage
    - Schema expansion and verification
    - Semantic rule gate after predicate selection (Step 3.5)
    - Constraint verification on actual data
    - Context enrichment with data profiling
    - Iterative refinement based on violations
    - Three deterministic gates in Step 5 (see module docstring)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        predicate_space: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        max_refinement_iterations: int = 2,
        violation_ratio_threshold: float = 0.2,
        prompt_file: str = "multi_step_refinement_less_restrict",
        chunk_size: Optional[int] = None,
    ):
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name, provider=provider)
        self.processing_errors: List[str] = []
        self.prompt_loader = PromptLoader()
        self.prompt_file = prompt_file
        self.predicate_space = predicate_space or {}
        self.df = df
        self.max_refinement_iterations = max_refinement_iterations
        self.violation_ratio_threshold = violation_ratio_threshold

        self.db_executor = None
        self.sql_translator = None
        if df is not None or csv_path is not None:
            self.db_executor = DBExecutor(
                df=df,
                csv_path=csv_path,
                table_name="data_table",
                tid_column="tid",
            )
            self.sql_translator = SQLTranslator(self.db_executor)

        self.tables = predicate_space.get("tables", [])
        self.allowed_predicates = set(predicate_space.get("predicates", []))
        self.column_profiles = self._build_column_profiles()
        self.columns_with_predicates = self._extract_predicate_columns()

    def _lookup_name_for_code(self, base_column: str, raw_value: str) -> Optional[str]:
        """Map a coded column value to its paired *_name label when the mapping is unique in the data."""
        if self.df is None or self.df.empty:
            return None

        name_column = f"{base_column}_name"
        if base_column not in self.df.columns or name_column not in self.df.columns:
            return None

        try:
            code_value = (
                int(raw_value)
                if re.fullmatch(r"-?\d+", raw_value)
                else float(raw_value)
            )
        except ValueError:
            return None

        matches = (
            self.df.loc[self.df[base_column] == code_value, name_column]
            .dropna()
            .unique()
        )
        if len(matches) != 1:
            return None

        mapped = matches[0]
        return str(mapped)

    def _replace_code_columns_with_names(
        self, constraint: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Prefer *_name predicates over coded numeric enums when both columns exist."""
        if not constraint:
            return constraint

        adjusted = []
        pattern = re.compile(r"\b(t[12])\.(\w+)\s*(=|!=)\s*(-?\d+(?:\.0+)?)\b")

        for predicate in constraint:

            def replace(match: re.Match[str]) -> str:
                alias, column, operator, raw_value = match.groups()
                mapped_name = self._lookup_name_for_code(column, raw_value)
                if not mapped_name:
                    return match.group(0)
                return f"{alias}.{column}_name {operator} '{mapped_name}'"

            adjusted.append(pattern.sub(replace, predicate))

        return adjusted

    def _restore_exceed_direction(
        self, sentence: str, constraint: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Keep violation direction aligned with exceed/more-than language for threshold predicates."""
        if not constraint:
            return constraint

        sentence_lower = sentence.lower()
        if not any(
            token in sentence_lower for token in ["exceed", "exceeds", "more than"]
        ):
            return constraint

        adjusted = []
        pattern = re.compile(
            r"(\b(?:t[12]\.)?(?:rate_spread|interest_rate|total_points_and_fees)\b\s*)(<=|<)(\s*.+)$"
        )

        for predicate in constraint:
            adjusted.append(pattern.sub(r"\1>\3", predicate))

        return adjusted

    def _prefer_rate_spread_for_apor_rules(
        self, sentence: str, constraint: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Prefer rate_spread over interest_rate for APOR percentage-point rules."""
        if not constraint:
            return constraint

        sentence_lower = sentence.lower()
        if (
            "average prime offer rate" not in sentence_lower
            and "comparable transaction" not in sentence_lower
        ) or "percentage point" not in sentence_lower:
            return constraint

        adjusted = []
        pattern = re.compile(r"\b((?:t[12]\.)?)interest_rate\b")

        for predicate in constraint:
            adjusted.append(pattern.sub(r"\1rate_spread", predicate))

        return adjusted

    def _repair_constructed_constraint(
        self, sentence: str, constraint: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Apply deterministic local repairs for common HMDA-style construction errors."""
        constraint = self._prefer_rate_spread_for_apor_rules(sentence, constraint)
        constraint = _restore_percentage_point_units(sentence, constraint)
        constraint = self._restore_exceed_direction(sentence, constraint)
        constraint = self._replace_code_columns_with_names(constraint)
        return constraint

    # ========================================================================
    # Initialisation helpers
    # ========================================================================

    def _extract_predicate_columns(self) -> set:
        """Extract unique column names that appear in the predicate space."""
        cols = set()
        for pred in self.allowed_predicates:
            matches = re.findall(r"t[12]\.(\w+)", pred)
            cols.update(matches)
        return cols

    def _build_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Build column profiles with sample values and stats."""
        profiles = {}

        if self.df is None or self.df.empty:
            return profiles

        for col in self.df.columns:
            if col == "tid":
                continue

            col_data = self.df[col].dropna()
            unique_vals = col_data.unique()
            n_unique = len(unique_vals)

            if n_unique <= 10:
                sample_values = unique_vals.tolist()
            else:
                top_frequent = col_data.value_counts().head(5).index.tolist()
                remaining = [v for v in unique_vals if v not in set(top_frequent)]
                n_random = min(5, len(remaining))
                random_sample = (
                    pd.Series(remaining).sample(n=n_random, random_state=42).tolist()
                    if n_random > 0
                    else []
                )
                sample_values = top_frequent + random_sample

            profile = {
                "type": str(self.df[col].dtype),
                "null_count": self.df[col].isnull().sum(),
                "unique_count": self.df[col].nunique(),
                "sample_values": sample_values,
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                profile.update(
                    {
                        "min": (
                            float(self.df[col].min()) if not col_data.empty else None
                        ),
                        "max": (
                            float(self.df[col].max()) if not col_data.empty else None
                        ),
                        "mean": (
                            float(self.df[col].mean()) if not col_data.empty else None
                        ),
                    }
                )

            profiles[col] = profile

        return profiles

    # ========================================================================
    # Column profile formatters
    # ========================================================================

    def _format_column_profiles_for(self, columns: List[str]) -> str:
        """Format profiles only for relevant columns to avoid token waste."""
        if not self.column_profiles or not columns:
            return ""

        lines = ["\nColumn profiles (use these to verify string constants and casing):"]
        for col in columns:
            if col in self.column_profiles:
                profile = self.column_profiles[col]
                samples = profile.get("sample_values", [])
                samples_str = ", ".join(
                    [f"'{s}'" if isinstance(s, str) else str(s) for s in samples[:8]]
                )
                null_info = (
                    f", {profile['null_count']} nulls"
                    if profile["null_count"] > 0
                    else ""
                )
                lines.append(
                    f"  - {col} ({profile['type']}{null_info}): [{samples_str}]"
                )

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_column_profiles(self) -> str:
        """Format all column profiles for schema understanding step."""
        if not self.column_profiles:
            return ""

        lines = ["\nColumn profiles (sample data context):"]
        for col, profile in self.column_profiles.items():
            samples = profile.get("sample_values", [])
            samples_str = ", ".join(
                [f"'{s}'" if isinstance(s, str) else str(s) for s in samples[:3]]
            )
            null_info = (
                f", {profile['null_count']} nulls" if profile["null_count"] > 0 else ""
            )
            lines.append(f"  - {col} ({profile['type']}{null_info}): [{samples_str}]")

        return "\n".join(lines)

    # ========================================================================
    # Pipeline entry-point
    # ========================================================================

    def _map_sentences(
        self, sentences: List[str], context: Dict[str, Any]
    ) -> List[Optional[List[str]]]:
        """Process sentences through multi-step refinement pipeline."""
        self.processing_errors = []
        results = []
        for sentence in sentences:
            try:
                constraint = self._process_single_sentence(sentence, context)
                results.append(constraint)
            except Exception as e:
                message = f"Error processing sentence: {e}"
                print(message)
                self.processing_errors.append(str(e))
                results.append(None)
        return results

    def _process_single_sentence(
        self, sentence: str, context: Dict[str, Any]
    ) -> Optional[List[str]]:
        print(f"\n{'='*60}")
        print(f"Processing: {sentence[:80]}...")

        # Step 2: Schema Understanding (schema-grounded gate)
        schema_analysis = self._step2_understand_schema(sentence)
        if not schema_analysis["is_valid"]:
            print(
                f"-> Schema validation failed: {schema_analysis.get('reason', 'unknown')}"
            )
            return None

        # Step 3: Predicate Selection
        predicate_selection = self._step3_select_predicates(sentence, schema_analysis)
        if not predicate_selection["is_valid"]:
            print(
                f"-> Predicate selection failed: {predicate_selection.get('reason', 'unknown')}"
            )
            return None

        # Step 3.5: Rule Gate — semantic check before paying construction cost
        rule_gate = self._step3_5_rule_gate(
            sentence, schema_analysis, predicate_selection
        )
        if not rule_gate.get("is_dc", True):
            print(f"-> Rule gate: NOT a verifiable DC — skipping construction")
            return None

        # Step 4: Initial Construction
        initial_constraint = self._step4_construct_constraint(
            sentence, schema_analysis, predicate_selection
        )
        if not initial_constraint:
            print("-> Could not construct initial constraint")
            return None

        # Steps 5 & 6: Verification and Refinement Loop
        constraint = initial_constraint
        for iteration in range(self.max_refinement_iterations):
            print(
                f"\n-> Verification iteration {iteration + 1}/{self.max_refinement_iterations}"
            )

            verification = self._step5_verify_constraint(constraint, sentence)

            # G1: dead predicate — do NOT refine, discard immediately
            if verification.get("is_dead_predicate"):
                print(f"-> Discarding: invented constant, refinement skipped")
                return None

            if verification["is_valid"]:
                # G3 may have swapped the constraint to its corrected form
                return verification.get("constraint", constraint)

            else:
                return None  # For now, we skip refinement and just return None on invalid constraints

            # print(f"-> Refining - {verification['message']}")
            # refined = self._step6_refine_constraint(
            #     constraint, sentence, verification, schema_analysis, predicate_selection
            # )

            # if refined == constraint:
            #     print("-> No refinement possible")
            #     return constraint

            # constraint = refined

        print(f"-> Max iterations reached")
        return constraint

    # ========================================================================
    # Step 2 — Schema Understanding
    # ========================================================================

    def _step2_understand_schema(self, sentence: str) -> Dict[str, Any]:
        """Step 2: Understand relevant columns and verify they exist.

        Also acts as schema-grounded gate: rejects if no sentence concepts
        map to columns that have predicates defined.
        """
        schema_desc = self._format_schema_for_llm()
        column_profiles_desc = self._format_column_profiles()
        predicate_columns = "\n".join(
            [f"  - {c}" for c in sorted(self.columns_with_predicates)]
        )

        prompt_template = self.prompt_loader.load(self.prompt_file, "understand_schema")
        prompt = prompt_template.format(
            sentence=sentence,
            schema_desc=schema_desc,
            column_profiles_desc=column_profiles_desc,
            predicate_columns=predicate_columns,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        print(f"-> Step 2 response: {response}...")

        try:
            parsed = json.loads(extract_json(response))

            # Merge direct + implied into relevant_columns
            direct = parsed.get("direct_columns", [])
            implied = parsed.get("implied_columns", [])
            parsed["relevant_columns"] = list(dict.fromkeys(direct + implied))

            # Validate all columns actually exist in schema
            relevant = parsed["relevant_columns"]
            available_columns = {
                col
                for table in self.tables
                for col in [c["name"] for c in table["columns"]]
            }
            available_list = list(available_columns)

            missing = [col for col in relevant if col not in available_columns]
            if missing:
                suggestions = {}
                for missing_col in missing:
                    similar = self._find_similar_columns(
                        missing_col, available_list, top_n=3
                    )
                    suggestions[missing_col] = [
                        col for col, score in similar if score > 0.3
                    ]

                parsed["is_valid"] = False
                parsed["reason"] = f"Columns not found in schema: {missing}"
                if any(suggestions.values()):
                    parsed["suggestions"] = suggestions
                    print(f"-> Step 2: Columns not found: {missing}")
                    for col, sug in suggestions.items():
                        if sug:
                            print(f"   Suggestions: {sug}")

            if parsed.get("is_valid"):
                print(f"-> Step 2: Identified columns {relevant}")
            elif "suggestions" not in parsed:
                print(f"-> Step 2 failed: {parsed.get('reason', 'unknown')}")

            return parsed

        except Exception as e:
            print(f"-> Step 2 parsing error: {e}")
            return {"is_valid": False, "reason": str(e), "relevant_columns": []}

    # ========================================================================
    # Step 3 — Predicate Selection
    # ========================================================================

    def _step3_select_predicates(
        self, sentence: str, schema_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 3: Select predicates from allowed set."""
        relevant_columns = schema_analysis.get("relevant_columns", [])

        candidate_predicates = self._filter_predicates_by_columns(relevant_columns)

        if not candidate_predicates:
            return {
                "is_valid": False,
                "reason": "No valid predicates found for these columns",
                "selected_predicates": [],
            }

        predicates_desc = "\n".join([f"  - {p}" for p in sorted(candidate_predicates)])
        column_profiles_desc = self._format_column_profiles_for(relevant_columns)

        prompt_template = self.prompt_loader.load(self.prompt_file, "select_predicates")
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=", ".join(relevant_columns),
            predicates_desc=predicates_desc,
            column_profiles_desc=column_profiles_desc,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))

            selected = parsed.get("selected_predicates", [])
            invalid_preds = [p for p in selected if p not in self.allowed_predicates]

            if invalid_preds:
                parsed["is_valid"] = False
                parsed["reason"] = (
                    f"Invalid predicates (not in allowed set): {invalid_preds}"
                )

            if not parsed.get("is_valid") and "reason" not in parsed:
                parsed["reason"] = "unknown"

            if parsed.get("is_valid"):
                print(f"-> Step 3: Selected {len(selected)} predicates: {selected}")
            else:
                print(f"-> Step 3 failed: {parsed.get('reason', 'unknown')}")

            return parsed
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Parse error: {e}",
                "selected_predicates": [],
            }

    # ========================================================================
    # Step 3.5 — Rule Gate
    # ========================================================================

    def _step3_5_rule_gate(
        self,
        sentence: str,
        schema_analysis: Dict[str, Any],
        predicate_selection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Step 3.5: Semantic gate — verify this is actually a mappable DC.

        Placed after predicate selection (Step 3) so the gate has maximum context:
        matched columns, selected predicate templates, and constraint pattern type.
        Runs BEFORE construction (Step 4) to avoid wasting an LLM call on non-rules.

        Defaults to PASS (is_dc=True) on parse failure to avoid false negatives.
        """
        relevant_columns = schema_analysis.get("relevant_columns", [])
        selected_predicates = predicate_selection.get("selected_predicates", [])
        constraint_pattern = predicate_selection.get("constraint_pattern", "unknown")
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        prompt_template = self.prompt_loader.load(self.prompt_file, "rule_gate")
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=", ".join(relevant_columns),
            selected_predicates=", ".join(selected_predicates),
            constraint_pattern=constraint_pattern,
            column_profiles_desc=profiles_desc,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))
            is_dc = parsed.get("is_dc", True)  # default permissive on parse failure

            if is_dc:
                print(
                    f"-> Step 3.5 [GATE]: PASS — {parsed.get('reason', '')} "
                    f"(confidence={parsed.get('confidence', '?')})"
                )
            else:
                print(
                    f"-> Step 3.5 [GATE]: REJECT — {parsed.get('reason', 'unknown reason')} "
                    f"(confidence={parsed.get('confidence', '?')})"
                )

            return parsed

        except Exception as e:
            # Parse failure → default to PASS to avoid dropping valid rules
            print(f"-> Step 3.5 parsing error: {e} — defaulting to PASS")
            return {"is_dc": True, "confidence": "low"}

    # ========================================================================
    # Step 4 — Constraint Construction
    # ========================================================================

    def _step4_construct_constraint(
        self,
        sentence: str,
        schema_analysis: Dict[str, Any],
        predicate_selection: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Step 4: Construct initial denial constraint."""
        relevant_columns = schema_analysis.get("relevant_columns", [])
        selected_predicates = predicate_selection.get("selected_predicates", [])
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        print(
            f"-> Step 4: Constructing constraint with {len(selected_predicates)} predicates "
            f"and {len(relevant_columns)} columns"
        )

        prompt_template = self.prompt_loader.load(
            self.prompt_file, "construct_constraint"
        )
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=", ".join(relevant_columns),
            selected_predicates=", ".join(selected_predicates),
            column_profiles_desc=profiles_desc,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))
            constraint = _normalize_constraint_list(parsed.get("constraint"))
            constraint = self._repair_constructed_constraint(sentence, constraint)
            constraint_type = parsed.get("constraint_type", "unknown")

            if constraint and len(constraint) > 0:
                print(
                    f"-> Step 4: Constructed {constraint_type} with {len(constraint)} predicates"
                )
                print(f"   {constraint}")
                print(f"   {parsed.get('constraint_interpretation', '')}")
                return constraint

            print(f"-> Step 4 failed: {parsed.get('reasoning', 'no reason given')}")
            return None
        except Exception as e:
            print(f"-> Step 4 error: {e}")
            return None

    # ========================================================================
    # Step 5 — Verification  (gates G1, G2, G3)
    # ========================================================================

    def _step5_verify_constraint(
        self, constraint: List[str], sentence: str
    ) -> Dict[str, Any]:
        """Step 5: Verify constraint on database sample.

        Gates applied in order:
          G1 — Satisfiability check: dead literals → discard (no refinement)
          G2 — Violation ratio: too many → refine
          G3 — Zero-violation inversion probe: swap operators → auto-correct
        """
        if not self.sql_translator:
            return {
                "is_valid": True,
                "message": "No database available for verification",
                "constraint": constraint,
            }

        # ── G1: Satisfiability ────────────────────────────────────────────
        dead = self._check_predicate_satisfiability(constraint)
        if dead:
            print(
                f"-> Step 5 [G1]: Invented constants detected — dead predicates: {dead}"
            )
            return {
                "is_valid": False,
                "message": f"Invented constant — no rows satisfy: {dead}",
                "dead_predicates": dead,
                "constraint": constraint,
                "is_dead_predicate": True,  # signals caller: do NOT refine
            }

        # ── Execute SQL ───────────────────────────────────────────────────
        try:
            result = self.sql_translator.execute_violation(constraint)
        except Exception as e:
            return {
                "is_valid": False,
                "message": f"Verification error: {e}",
                "error": str(e),
                "constraint": constraint,
            }

        if not result["success"]:
            print(f"-> Step 5: SQL execution failed - {result.get('error', 'unknown')}")
            return {
                "is_valid": False,
                "message": f"SQL execution failed: {result.get('error', 'unknown')}",
                "sql_error": result.get("error"),
                "sql": result.get("sql"),
                "constraint": constraint,
            }

        violation_count = result["row_count"]
        print(f"-> Step 5: Found {violation_count} violations")

        is_single_row = not any("t2." in p for p in constraint)

        # ── G2: Violation ratio  ──────────────────────────
        # if not is_single_row and self.df is not None:
        #     total_pairs = len(self.df) * (len(self.df) - 1) // 2
        #     if total_pairs > 0 and violation_count > total_pairs * self.violation_ratio_threshold:
        #         print(f"-> Step 5 [G2]: Too many violations ({violation_count}/{total_pairs} pairs) "
        #               f"— constraint too loose or logically inverted")
        #         return {
        #             'is_valid': False,
        #             'message': (
        #                 f"Too many violations ({violation_count} of {total_pairs} pairs), "
        #                 f"constraint may be too loose or logically inverted"
        #             ),
        #             'violation_count': violation_count,
        #             'constraint': constraint,
        #         }

        # ── G2: Violation count thresholds ────────────────────────────────
        # Single-row constraints: flag if > 1000 violations
        # Cross-row (pair) constraints: flag if > 10000 violations
        if is_single_row:
            # if violation_count > 1000:
            if violation_count > 55000:
                print(
                    f"-> Step 5 [G2]: Too many single-row violations ({violation_count}) "
                    f"— constraint too loose or logically inverted"
                )
                return {
                    "is_valid": False,
                    "message": (
                        f"Too many violations ({violation_count} single-row matches), "
                        f"constraint may be too loose or logically inverted"
                    ),
                    "violation_count": violation_count,
                    "constraint": constraint,
                }
        else:
            if violation_count > 10000:
                total_pairs = (
                    len(self.df) * (len(self.df) - 1) // 2
                    if self.df is not None
                    else "?"
                )
                print(
                    f"-> Step 5 [G2]: Too many pair violations ({violation_count}/{total_pairs}) "
                    f"— constraint too loose or logically inverted"
                )
                return {
                    "is_valid": False,
                    "message": (
                        f"Too many violations ({violation_count} of {total_pairs} pairs), "
                        f"constraint may be too loose or logically inverted"
                    ),
                    "violation_count": violation_count,
                    "constraint": constraint,
                }

        # # ── G3: Zero-violation inversion probe (cross-row only) ───────────
        # if not is_single_row and violation_count == 0:
        #     corrected, inverted_constraint = self._probe_inverted_logic(constraint)
        #     if corrected:
        #         inv_count = self.sql_translator.execute_violation(inverted_constraint)['row_count']
        #         print(f"-> Step 5 [G3]: Logic was inverted — corrected constraint has "
        #               f"{inv_count} violations")
        #         return {
        #             'is_valid': True,
        #             'message': f"Logic inverted and auto-corrected; {inv_count} violations",
        #             'violation_count': inv_count,
        #             'constraint': inverted_constraint,   # return the corrected version
        #         }
        #     # Zero violations but inversion also finds nothing → possibly valid (no data violation)
        #     print(f"-> Step 5: 0 violations (inversion also found 0 — rule may be satisfied)")

        return {
            "is_valid": True,
            "message": f"Constraint valid, found {violation_count} violations",
            "violation_count": violation_count,
            "constraint": constraint,
        }

    # ========================================================================
    # Gate helpers
    # ========================================================================

    def _check_predicate_satisfiability(self, constraint: List[str]) -> List[str]:
        """G1 — For every predicate with a literal constant, verify at least one
        row in the table satisfies it.

        Uses LIMIT 1 for early-exit existence checks (no full scan needed).

        Operator handling:
          =, !=        → existence check with = (does the value exist at all?)
          LIKE         → existence check with the LIKE pattern itself
          NOT LIKE     → existence check with the equivalent LIKE pattern
          IN, NOT IN   → existence check with IN (does any listed value exist?)

        The operator direction is irrelevant for existence — a value not present
        in the column makes both '= X' (always false) and '!= X' (vacuously true,
        semantically wrong) dead predicates.

        Skips: cross-row predicates (t2.*), IS NULL / IS NOT NULL, numeric bounds
               (>, <, >=, <= against numbers — these are structural, not domain checks).
        """
        if self.db_executor is None:
            return []

        dead = []

        # Matches: t1.col OP 'literal'
        # Handles: =, !=, LIKE, NOT LIKE  (in order of specificity)
        single_literal = re.compile(
            r"t1\.(\w+)\s*(?:NOT\s+LIKE|LIKE|!=|=)\s*'([^']+)'",
            re.IGNORECASE,
        )
        # Matches: t1.col [NOT] IN ('a', 'b', ...)
        in_literal = re.compile(
            r"t1\.(\w+)\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",
            re.IGNORECASE,
        )

        for pred in constraint:
            # Skip cross-row predicates — no literal constant to domain-check
            if "t2." in pred:
                continue
            # Skip null checks — structural, not domain-based
            if re.search(r"\bIS\s+(NOT\s+)?NULL\b", pred, re.IGNORECASE):
                continue
            # Skip numeric bounds (>, <, >=, <= against a bare number, no quotes)
            if re.search(r"t1\.\w+\s*(?:>=|<=|>|<)\s*[\d.]+", pred) and "'" not in pred:
                continue

            # ── Case 1: single string literal ─────────────────────────────
            m = single_literal.search(pred)
            if m:
                col, value = m.group(1), m.group(2)
                is_like = bool(re.search(r"\bLIKE\b", pred, re.IGNORECASE))

                if is_like:
                    # Use the pattern directly — LIMIT 1 for early exit
                    sql = (
                        f"SELECT 1 FROM data_table "
                        f"WHERE \"{col}\" LIKE '{value}' LIMIT 1"
                    )
                else:
                    # Equality existence check regardless of original operator
                    sql = (
                        f"SELECT 1 FROM data_table "
                        f"WHERE \"{col}\" = '{value}' LIMIT 1"
                    )

                try:
                    res = self.db_executor.execute(sql)
                    if not res:
                        print(
                            f"   [G1] Dead predicate — "
                            f"value '{value}' not found in '{col}': {pred}"
                        )
                        dead.append(pred)
                except Exception:
                    pass  # SQL error ≠ invented value — do not discard on error
                continue

            # ── Case 2: IN / NOT IN ────────────────────────────────────────
            m = in_literal.search(pred)
            if m:
                col, values_str = m.group(1), m.group(2)
                # Any listed value existing is sufficient — LIMIT 1 for early exit
                sql = (
                    f"SELECT 1 FROM data_table "
                    f'WHERE "{col}" IN ({values_str}) LIMIT 1'
                )
                try:
                    res = self.db_executor.execute(sql)
                    if not res:
                        print(
                            f"   [G1] Dead predicate — "
                            f"none of ({values_str}) found in '{col}': {pred}"
                        )
                        dead.append(pred)
                except Exception:
                    pass

        return dead

    def _probe_inverted_logic(self, constraint: List[str]) -> Tuple[bool, List[str]]:
        """G3 — Flip comparison operators in cross-row predicates and test.

        Returns (True, inverted_constraint) if the inverted version finds violations,
        meaning the original was likely logically inverted by the LLM.
        Returns (False, []) otherwise.
        """
        INVERSIONS = {
            " != ": " = ",
            " = ": " != ",
            " > ": " < ",
            " < ": " > ",
            " >= ": " <= ",
            " <= ": " >= ",
        }

        inverted = []
        any_changed = False
        for pred in constraint:
            new_pred = pred
            for op, inv_op in INVERSIONS.items():
                if op in pred:
                    new_pred = pred.replace(op, inv_op, 1)
                    any_changed = True
                    break
            inverted.append(new_pred)

        if not any_changed:
            return False, []

        try:
            result = self.sql_translator.execute_violation(inverted)
            if result["success"] and result["row_count"] > 0:
                return True, inverted
        except Exception:
            pass

        return False, []

    # ========================================================================
    # Step 6 — Refinement
    # ========================================================================

    def _step6_refine_constraint(
        self,
        constraint: List[str],
        sentence: str,
        verification: Dict[str, Any],
        schema_analysis: Dict[str, Any],
        predicate_selection: Dict[str, Any],
    ) -> List[str]:
        """Step 6: Refine constraint based on verification results."""
        error_msg = verification.get("message", "")
        sql_error = verification.get("sql_error", "")
        sql_error_section = f"SQL error: {sql_error}" if sql_error else ""

        relevant_columns = schema_analysis.get("relevant_columns", [])
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        # Pass all candidate predicates for these columns, not just previously selected
        candidate_predicates = self._filter_predicates_by_columns(relevant_columns)
        available_predicates_str = "\n".join(
            [f"  - {p}" for p in sorted(candidate_predicates)]
        )

        prompt_template = self.prompt_loader.load(self.prompt_file, "refine_constraint")
        prompt = prompt_template.format(
            sentence=sentence,
            constraint=json.dumps(constraint),
            error_msg=error_msg,
            sql_error_section=sql_error_section,
            available_columns=", ".join(relevant_columns),
            available_predicates=available_predicates_str,
            column_profiles_desc=profiles_desc,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))
            refined = _normalize_constraint_list(
                parsed.get("refined_constraint", constraint)
            )

            if refined:
                print(f"   Refined to: {refined}")
                return refined

            return constraint
        except Exception as e:
            print(f"-> Refinement error: {e}")
            return constraint

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _format_schema_for_llm(self) -> str:
        """Format schema information for LLM prompts."""
        if not self.tables:
            return "Schema: No table information available"

        lines = ["Available schema:"]
        for table in self.tables:
            lines.append(f"\nTable: {table['name']}")
            for col in table["columns"]:
                lines.append(f"  - {col['name']} ({col['type']})")

        return "\n".join(lines)

    def _filter_predicates_by_columns(self, columns: List[str]) -> List[str]:
        """Filter allowed predicates to those relevant for given columns."""
        if not columns or not self.allowed_predicates:
            return list(self.allowed_predicates)

        relevant = [
            pred
            for pred in self.allowed_predicates
            if any(col in pred for col in columns)
        ]

        return relevant if relevant else list(self.allowed_predicates)

    def _find_similar_columns(
        self, target: str, available: List[str], top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Find columns similar to target using token-based similarity."""
        similarities = [(col, token_similarity(target, col)) for col in available]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def close(self):
        """Cleanup resources."""
        if self.db_executor:
            self.db_executor.close()


# ============================================================================
# Simplified Variant: Direct Refinement Mapper
# ============================================================================


class DirectRefinementMapper(BaseMapper):
    """
    Simplified mapper that focuses on iterative refinement.

    Skips multi-step analysis and goes straight to:
    1. Generate initial constraint
    2. Verify on data (with satisfiability gate G1)
    3. Refine if errors detected
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        predicate_space: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None,
        max_iterations: int = 2,
        prompt_file: str = "multi_step_refinement",
        chunk_size: Optional[int] = None,
    ):
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name)
        self.prompt_loader = PromptLoader()
        self.prompt_file = prompt_file
        self.predicate_space = predicate_space or {}
        self.df = df
        self.max_iterations = max_iterations

        self.db_executor = None
        self.sql_translator = None
        if df is not None:
            self.db_executor = DBExecutor(df, table_name="data_table", tid_column="tid")
            self.sql_translator = SQLTranslator(self.db_executor)

        self.allowed_predicates = set(predicate_space.get("predicates", []))

    def _map_sentences(
        self, sentences: List[str], context: Dict[str, Any]
    ) -> List[Optional[List[str]]]:
        """Simple refinement loop for each sentence."""
        results = []
        for sentence in sentences:
            print(f"\n{'='*60}")
            print(f"Processing: {sentence[:80]}...")

            constraint = self._generate_initial(sentence, context)

            if not constraint:
                print("-> Could not generate initial constraint")
                results.append(None)
                continue

            print(f"-> Initial constraint: {constraint}")

            for iteration in range(self.max_iterations):
                print(
                    f"-> Verification iteration {iteration + 1}/{self.max_iterations}"
                )
                verification = self._verify(constraint)

                # G1: dead predicate — discard immediately without refinement
                if verification.get("is_dead_predicate"):
                    print(f"-> Discarding: invented constant, refinement skipped")
                    constraint = None
                    break

                if verification["is_valid"]:
                    print(
                        f"-> Verification successful ({verification.get('violation_count', '?')} violations)"
                    )
                    break

                print(f"-> Refining - {verification.get('error', 'validation failed')}")
                refined = self._refine(sentence, constraint, verification, context)

                if refined == constraint:
                    print("-> No refinement possible")
                    break

                print(f"-> Refined to: {refined}")
                constraint = refined

            results.append(constraint)

        return results

    def _generate_initial(
        self, sentence: str, context: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Generate initial constraint attempt."""
        predicates_list = "\n".join(
            [f"  - {p}" for p in sorted(self.allowed_predicates)]
        )

        prompt_template = self.prompt_loader.load(self.prompt_file, "generate_initial")
        prompt = prompt_template.format(
            sentence=sentence,
            predicates_list=predicates_list,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))
            return _normalize_constraint_list(parsed.get("constraint"))
        except Exception as e:
            print(f"-> Parse error: {e}")
            return None

    def _verify(self, constraint: List[str]) -> Dict[str, Any]:
        """Verify constraint on database (includes G1 satisfiability gate)."""
        if not self.sql_translator:
            return {"is_valid": True}

        dead = self._check_predicate_satisfiability(constraint)
        if dead:
            print(f"-> [G1] Dead predicates: {dead}")
            return {
                "is_valid": False,
                "is_dead_predicate": True,
                "error": f"Invented constant — no rows satisfy: {dead}",
            }

        try:
            result = self.sql_translator.execute_violation(constraint)

            if not result["success"]:
                return {"is_valid": False, "error": result.get("error")}

            return {"is_valid": True, "violation_count": result["row_count"]}
        except Exception as e:
            return {"is_valid": False, "error": str(e)}

    def _check_predicate_satisfiability(self, constraint: List[str]) -> List[str]:
        """G1 — existence check with LIMIT 1 for speed.

        Identical logic to MultiStepRefinementMapper._check_predicate_satisfiability.
        See that method's docstring for full operator-handling rationale.
        """
        if self.db_executor is None:
            return []

        dead = []

        single_literal = re.compile(
            r"t1\.(\w+)\s*(?:NOT\s+LIKE|LIKE|!=|=)\s*'([^']+)'",
            re.IGNORECASE,
        )
        in_literal = re.compile(
            r"t1\.(\w+)\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",
            re.IGNORECASE,
        )

        for pred in constraint:
            if "t2." in pred:
                continue
            if re.search(r"\bIS\s+(NOT\s+)?NULL\b", pred, re.IGNORECASE):
                continue
            if re.search(r"t1\.\w+\s*(?:>=|<=|>|<)\s*[\d.]+", pred) and "'" not in pred:
                continue

            m = single_literal.search(pred)
            if m:
                col, value = m.group(1), m.group(2)
                is_like = bool(re.search(r"\bLIKE\b", pred, re.IGNORECASE))
                if is_like:
                    sql = f"SELECT 1 FROM data_table WHERE \"{col}\" LIKE '{value}' LIMIT 1"
                else:
                    sql = (
                        f"SELECT 1 FROM data_table WHERE \"{col}\" = '{value}' LIMIT 1"
                    )
                try:
                    res = self.db_executor.execute(sql)
                    if not res:
                        dead.append(pred)
                except Exception:
                    pass
                continue

            m = in_literal.search(pred)
            if m:
                col, values_str = m.group(1), m.group(2)
                sql = (
                    f'SELECT 1 FROM data_table WHERE "{col}" IN ({values_str}) LIMIT 1'
                )
                try:
                    res = self.db_executor.execute(sql)
                    if not res:
                        dead.append(pred)
                except Exception:
                    pass

        return dead

    def _refine(
        self,
        sentence: str,
        constraint: List[str],
        verification: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Refine constraint based on verification."""
        error = verification.get("error", "validation failed")
        predicates_list = "\n".join(
            [f"  - {p}" for p in sorted(self.allowed_predicates)]
        )

        prompt_template = self.prompt_loader.load(self.prompt_file, "refine_simple")
        prompt = prompt_template.format(
            sentence=sentence,
            constraint=json.dumps(constraint),
            error=error,
            predicates_list=predicates_list,
        )

        response = self.llm.complete(prompt)

        if self.llm.model_name.startswith("mistral-"):
            # Remove ```json and ``` from response if present
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)

        try:
            parsed = json.loads(extract_json(response))
            refined = _normalize_constraint_list(parsed.get("refined"))
            return refined if refined else constraint
        except Exception as e:
            print(f"-> Refinement parse error: {e}")
            return constraint

    def close(self):
        if self.db_executor:
            self.db_executor.close()
