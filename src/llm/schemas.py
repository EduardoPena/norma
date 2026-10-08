"""JSON Schemas for structured LLM outputs, built on the denial-constraint format.

A sentence output (``Rules``) is ``null`` or a list of independent DCs; each DC is
a predicate string or a logical node ``{"AND": [...]}`` / ``{"OR": [...]}`` whose
items are DCs again (see ``src.predicates.validator.parse_rule``), up to
``MAX_DC_DEPTH`` nested logical levels.

Schemas follow the strict structured-output subset (root object, every property
required, ``additionalProperties: false``, optional values as ``null``). Rules the
schema cannot express (>= 2 items per node, predicate grammar, predicate-space
membership) are still enforced by ``normalize_rules`` and the mappers.
"""

from typing import Any, Dict

# Logical nesting is unrolled to a fixed depth because some providers (e.g.
# Anthropic) reject recursive/circular $ref schemas. The deepest ground-truth DC
# across all datasets has 3 nested AND/OR levels; 4 leaves one level of headroom.
MAX_DC_DEPTH = 4
DC_RULE = f"#/$defs/dc_rule_{MAX_DC_DEPTH}"


def _logical_node(operator: str, child_ref: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {operator: {"type": "array", "items": {"$ref": child_ref}}},
        "required": [operator],
        "additionalProperties": False,
    }


PREDICATE = {"type": "string", "description": "Predicate, e.g. t1.age < 18"}

DC_DEFS: Dict[str, Any] = {"dc_rule_0": PREDICATE}
for _level in range(1, MAX_DC_DEPTH + 1):
    _child = f"#/$defs/dc_rule_{_level - 1}"
    DC_DEFS[f"dc_rule_{_level}"] = {
        "anyOf": [PREDICATE, _logical_node("AND", _child), _logical_node("OR", _child)]
    }

RULES: Dict[str, Any] = {
    "anyOf": [
        {"type": "array", "items": {"$ref": DC_RULE}},
        {"type": "null"},
    ],
    "description": "Independent denial constraints in violation form, or null",
}


def _object(properties: Dict[str, Any], uses_rules: bool = False) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if uses_rules:
        schema["$defs"] = DC_DEFS
    return schema


def _nullable(json_type: str) -> Dict[str, Any]:
    return {"type": [json_type, "null"]}


def _enum(*values: str) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


STRING_LIST = {"type": "array", "items": {"type": "string"}}
CONFIDENCE = _enum("high", "medium", "low")

CLASSIFICATION = _object(
    {
        "labels": {
            "type": "array",
            "items": _object(
                {"id": {"type": "integer"}, "is_constraint": {"type": "boolean"}}
            ),
        }
    }
)

SENTENCE_RULES = _object(
    {
        "results": {
            "type": "array",
            "items": _object({"id": {"type": "integer"}, "rules": RULES}),
        }
    },
    uses_rules=True,
)

UNDERSTAND_SCHEMA = _object(
    {
        "is_valid": {"type": "boolean"},
        "direct_columns": STRING_LIST,
        "implied_columns": STRING_LIST,
        "reasoning": {"type": "string"},
        "missing_concepts": STRING_LIST,
        "reason": _nullable("string"),
    }
)

SELECT_PREDICATES = _object(
    {
        "is_valid": {"type": "boolean"},
        "selected_predicates": STRING_LIST,
        "reasoning": {"type": "string"},
        "constraint_pattern": _enum(
            "bound",
            "required",
            "functional_dependency",
            "conditional",
            "value_mapping",
            "temporal",
            "comparative",
        ),
        "is_single_row": {"type": "boolean"},
        "confidence": CONFIDENCE,
        "reason": _nullable("string"),
    }
)

RULE_GATE = _object(
    {"is_dc": {"type": "boolean"}, "reason": {"type": "string"}, "confidence": CONFIDENCE}
)

CONSTRUCT_CONSTRAINT = _object(
    {
        "rules": RULES,
        "reasoning": {"type": "string"},
        "constraint_type": _enum(
            "bound",
            "required",
            "functional_dependency",
            "conditional",
            "value_mapping",
            "temporal",
            "uniqueness",
            "comparative",
            "other",
        ),
        "constraint_interpretation": {"type": "string"},
        "is_single_row": {"type": "boolean"},
    },
    uses_rules=True,
)

REFINE_CONSTRAINT = _object(
    {
        "refined_rules": RULES,
        "changes_made": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": CONFIDENCE,
    },
    uses_rules=True,
)
