"""Structured outputs follow the DC format; malformed model outputs degrade to
"no constraint" (logged) while infrastructure errors still propagate."""

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.classifiers.llm import LLMRuleClassifier
from src.llm import schemas
from src.llm.client import EmptyModelContentError, OpenRouterClient
from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper

try:
    import jsonschema
except ImportError:  # optional: installed with nbconvert, not in requirements
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]
GT_FILES = sorted(ROOT.glob("data/*/*gt*.json"))
ALL_SCHEMAS = {
    name: value for name, value in vars(schemas).items()
    if name.isupper() and isinstance(value, dict) and value.get("type") == "object"
}


def structural_rules(rules):
    """Convert legacy [[p1, p2]] rules to {"AND": [...]} without validating predicates
    (structure only, as in normalize_rules)."""
    if not rules:
        return rules
    return [
        (rule[0] if len(rule) == 1 else {"AND": rule}) if isinstance(rule, list) else rule
        for rule in rules
    ]


def completion(text):
    message = SimpleNamespace(content=text, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def stub_llm(responses):
    return SimpleNamespace(model_name="stub", complete=MagicMock(side_effect=responses))


class SchemaTests(unittest.TestCase):
    def test_schemas_are_strict(self):
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)

        for schema in ALL_SCHEMAS.values():
            check(schema)

    @unittest.skipUnless(jsonschema, "jsonschema not installed")
    def test_ground_truth_dcs_validate_against_sentence_rules(self):
        validator = jsonschema.Draft202012Validator(schemas.SENTENCE_RULES)
        for path in GT_FILES:
            mapping = json.loads(path.read_text())["mapping"]
            payload = {"results": [
                {"id": i, "rules": structural_rules(entry.get("rules"))}
                for i, entry in enumerate(mapping, 1)
            ]}
            errors = list(validator.iter_errors(payload))
            self.assertEqual(errors, [], f"{path.name}: {errors[:1]}")

    def test_dc_definitions_are_not_recursive(self):
        # Anthropic rejects self/mutually-referencing $defs; each level may only
        # reference the level below it.
        for level in range(schemas.MAX_DC_DEPTH + 1):
            refs = re.findall(r'"#/\$defs/dc_rule_(\d+)"', json.dumps(schemas.DC_DEFS[f"dc_rule_{level}"]))
            self.assertTrue(all(int(ref) == level - 1 for ref in refs), (level, refs))

    @unittest.skipUnless(jsonschema, "jsonschema not installed")
    def test_schema_rejects_too_deep_dc(self):
        validator = jsonschema.Draft202012Validator(schemas.SENTENCE_RULES)
        rule = "t1.a = 1"
        for _ in range(schemas.MAX_DC_DEPTH + 1):
            rule = {"AND": [rule, "t1.b = 2"]}
        self.assertTrue(list(validator.iter_errors({"results": [{"id": 1, "rules": [rule]}]})))

    @unittest.skipUnless(jsonschema, "jsonschema not installed")
    def test_schema_rejects_malformed_dc(self):
        validator = jsonschema.Draft202012Validator(schemas.SENTENCE_RULES)
        bad = {"results": [{"id": 1, "rules": [{"XOR": ["t1.a = 1", "t1.b = 2"]}]}]}
        self.assertTrue(list(validator.iter_errors(bad)))


class ClientTests(unittest.TestCase):
    def make_client(self, side_effect, structured_output=True):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.model_name = "openai/gpt-5-mini"
        client.max_tokens = None
        client.temperature = 0.0
        client.structured_output = structured_output
        client.supported_parameters = None
        create = MagicMock(side_effect=side_effect)
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client, create

    def test_sends_strict_schema_and_requires_supporting_provider(self):
        client, create = self.make_client([completion("{}")])
        client.complete("p", response_schema=schemas.RULE_GATE, schema_name="rule_gate")
        request = create.call_args.kwargs
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertIs(request["response_format"]["json_schema"]["schema"], schemas.RULE_GATE)
        self.assertEqual(request["extra_body"], {"provider": {"require_parameters": True}})

    def test_unsupported_parameters_are_not_sent(self):
        client, create = self.make_client([completion("{}")])
        client.supported_parameters = {"response_format", "max_tokens"}
        client.max_tokens = 100
        client.complete("p", response_schema=schemas.RULE_GATE)
        self.assertNotIn("temperature", create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 100)

    def test_prompt_only_mode_sends_no_schema(self):
        client, create = self.make_client([completion("{}")], structured_output=False)
        client.complete("p", response_schema=schemas.RULE_GATE)
        self.assertNotIn("response_format", create.call_args.kwargs)

    @patch("time.sleep")
    def test_response_without_choices_is_retried(self, sleep):
        client, create = self.make_client(
            [SimpleNamespace(choices=None, error={"code": 502}), completion("ok")]
        )
        self.assertEqual(client.complete("p"), "ok")
        self.assertEqual(create.call_count, 2)


class ClassifierTests(unittest.TestCase):
    def make(self, responses):
        classifier = LLMRuleClassifier.__new__(LLMRuleClassifier)
        classifier.chunk_size = None
        classifier.prompt_template = "{sentences}"
        classifier.max_output_retries = 2
        classifier.output_failures = []
        classifier.llm = stub_llm(responses)
        return classifier

    def test_structured_labels(self):
        classifier = self.make(['{"labels": [{"id": 2, "is_constraint": false}, {"id": 1, "is_constraint": true}]}'])
        self.assertEqual(classifier._classify_sentences(["a", "b"], {}), [True, False])
        self.assertIs(
            classifier.llm.complete.call_args.kwargs["response_schema"], schemas.CLASSIFICATION
        )

    def test_retry_then_individual_fallback(self):
        partial = '{"labels": [{"id": 1, "is_constraint": true}]}'
        classifier = self.make(
            [partial] * 3
            + ['{"labels": [{"id": 1, "is_constraint": true}]}']
            + ["garbage"] * 3
        )
        self.assertEqual(classifier._classify_sentences(["a", "b"], {}), [True, False])
        self.assertEqual(len(classifier.output_failures), 1)


class SinglePromptTests(unittest.TestCase):
    def make(self, responses):
        mapper = LLMMapper.__new__(LLMMapper)
        mapper.max_output_retries = 2
        mapper.output_failures = []
        mapper.allowed_predicates = None
        mapper.filter_predicates = False
        mapper.prompt_template = "{sentences}{predicates_list}{table_context}"
        mapper.llm = stub_llm(responses)
        return mapper

    def test_results_keyed_by_id_and_empty_list_is_null(self):
        mapper = self.make(['{"results": [{"id": 2, "rules": []}, {"id": 1, "rules": ["t1.a < 1"]}]}'])
        self.assertEqual(mapper._map_sentences(["a", "b"], {}), [["t1.a < 1"], None])
        self.assertIs(
            mapper.llm.complete.call_args.kwargs["response_schema"], schemas.SENTENCE_RULES
        )

    def test_malformed_batch_retried_then_null(self):
        mapper = self.make(['{"results": [{"id": 1, "rules": null}]}'] * 3)
        self.assertEqual(mapper._map_sentences(["a", "b"], {}), [None, None])
        self.assertEqual(mapper.llm.complete.call_count, 3)
        self.assertEqual(len(mapper.output_failures), 2)

    def test_infrastructure_errors_propagate(self):
        mapper = self.make([RuntimeError("OpenRouter API error: 429")])
        with self.assertRaises(RuntimeError):
            mapper._map_sentences(["a"], {})


class MultiStepTests(unittest.TestCase):
    def test_exhausted_output_yields_no_constraint_but_api_errors_abort(self):
        mapper = MultiStepRefinementMapper.__new__(MultiStepRefinementMapper)
        mapper.output_failures = []
        side_effects = [EmptyModelContentError("length"), ["t1.a < 1"], RuntimeError("429")]
        with patch.object(mapper, "_process_single_sentence", side_effect=side_effects):
            results = mapper._map_sentences(["a", "b", "c"], {})
        self.assertEqual(results, [None, ["t1.a < 1"], None])
        self.assertEqual(len(mapper.output_failures), 1)
        self.assertEqual(mapper.processing_errors, ["429"])


if __name__ == "__main__":
    unittest.main()
