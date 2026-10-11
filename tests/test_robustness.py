"""Robustness revision (2026-10-10): type-checked rules, equivalence-aware
evaluation, counted SQL failures, mapper retries, predicate-space extensions."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.evaluation.mapping import MappingEvaluator
from src.llm.client import OpenRouterClient
from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.predicates import RuleValidationError, normalize_rules, parse_predicate, repair_rules
from src.predicates.equivalence import logical_form, logically_equivalent
from src.predicates.space_builder import PredicateSpaceBuilder
from src.predicates.templates import in_space, out_of_space_counts
from src.utils.table_context import TableContextBuilder

SCHEMA = {
    "tables": [
        {
            "columns": [
                {"name": "rate", "type": "float"},
                {"name": "code", "type": "int"},
                {"name": "status", "type": "string"},
            ]
        }
    ]
}


def stub_llm(responses):
    llm = MagicMock()
    llm.complete.side_effect = list(responses)
    llm.model_name = "stub/model"
    return llm


class ValidatorTypeTests(unittest.TestCase):
    def test_type_mismatches_are_rejected_with_schema_only(self):
        for predicate in ["t1.rate != 'NA'", "t1.status = 5", "t1.status * 2 > 1", "t1.code LIKE '1%'"]:
            with self.assertRaises(RuleValidationError, msg=predicate):
                parse_predicate(predicate, SCHEMA)
            parse_predicate(predicate)  # valid without a typed schema
        for predicate in ["t1.rate > 0.05 * t1.code", "t1.code IN (1,2)", "t1.status = 'A'", "t1.rate IS NULL"]:
            parse_predicate(predicate, SCHEMA)
        self.assertEqual(normalize_rules(["t1.rate != 'NA'"]), ["t1.rate != 'NA'"])  # no schema

    def test_repair_rules_collapses_unary_and_drops_empty_nodes(self):
        repaired = repair_rules([{"AND": ["t1.code > 1"]}, {"OR": []}, {"AND": ["t1.code > 1", {"OR": ["t1.rate < 0"]}]}, ""])
        self.assertEqual(repaired, ["t1.code > 1", {"AND": ["t1.code > 1", "t1.rate < 0"]}])
        self.assertIsNone(repair_rules([{"OR": []}]))


class EquivalenceTests(unittest.TestCase):
    def test_equivalent_rewritings(self):
        pairs = [
            ([{"AND": ["t1.a IN (1,3)", "t1.b != 1"]}],
             [{"AND": [{"OR": ["t1.a = 1", "t1.a = 3"]}, "t1.b != 1"]}]),
            (["t1.a NOT IN (3,7)"], [{"AND": ["t1.a != 3", "t1.a != 7"]}]),
            (["t1.f > 0.05 * t1.x"], ["t1.f > t1.x * 0.05"]),
            (["5 < t1.x"], ["t1.x > 5"]),
            ([{"AND": ["t1.h != 'H'", {"OR": [{"AND": ["t1.l = 1", "t1.r > 6.5"]}, {"AND": ["t1.l = 2", "t1.r > 8.5"]}]}]}],
             [{"AND": ["t1.l = 1", "t1.r > 6.5", "t1.h != 'H'"]}, {"AND": ["t1.l = 2", "t1.r > 8.5", "t1.h != 'H'"]}]),
            (["t1.a = 1", {"AND": ["t1.a = 1", "t1.b = 2"]}], ["t1.a = 1"]),
        ]
        for gold, pred in pairs:
            self.assertTrue(logically_equivalent(gold, pred), (gold, pred))

    def test_different_rules_stay_different(self):
        self.assertFalse(logically_equivalent(["t1.a > 5"], ["t1.a >= 5"]))
        self.assertFalse(logically_equivalent(["t1.a IN (1,2,3,4)"], [{"AND": ["t1.a >= 1", "t1.a <= 4"]}]))
        self.assertFalse(logically_equivalent(["t1.x - 2 > t1.y"], ["2 - t1.x > t1.y"]))
        self.assertIsNone(logical_form(None))


class TemplateTests(unittest.TestCase):
    def test_space_membership(self):
        space = {"t1.a IN $values", "t1.f > t1.x * $number", "t1.s = $string", "t1.a = t1.b"}
        self.assertTrue(in_space(parse_predicate("t1.a IN (1,3)"), space))
        self.assertTrue(in_space(parse_predicate("t1.f > 0.05 * t1.x"), space))
        self.assertTrue(in_space(parse_predicate("t1.s = 'A'"), space))
        self.assertFalse(in_space(parse_predicate("t1.a != t1.b"), space))
        self.assertEqual(out_of_space_counts([["t1.a = t1.b", "t1.a != t1.b"], None], space), (1, 2))


class EvaluatorTests(unittest.TestCase):
    def test_equivalence_metric_and_counted_sql_errors(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})
        gt = {"s1": [{"AND": ["t1.a IN (1,3)", "t1.b = 1"]}], "s2": ["t1.a > 3"], "s3": ["t1.a = 9"]}
        evaluator = MappingEvaluator(gt, df=df)
        metrics, rows = evaluator.evaluate([
            ("s1", [{"AND": [{"OR": ["t1.a = 1", "t1.a = 3"]}, "t1.b = 1"]}]),  # equivalent
            ("s2", ["t1.a = 'x'"]),  # SQL fails on a numeric column
            ("s3", None),
        ])
        self.assertEqual(metrics.constraint_exact_tp, 0)
        self.assertEqual(metrics.constraint_equiv_tp, 1)
        self.assertEqual([r.logical_match for r in rows], [True, False, False])
        self.assertEqual(metrics.num_pred_execution_errors, 1)
        self.assertEqual(rows[1].pred_execution_errors, 1)
        self.assertEqual(metrics.gold_rules_without_violations, 1)  # s3 never fires
        self.assertAlmostEqual(metrics.max_gold_rule_violation_share, 0.5)
        # predicate Jaccard compares leaves, not whole rule trees
        self.assertAlmostEqual(rows[0].predicate_jaccard, 1 / 4)
        evaluator.close()


class SinglePromptRepairTests(unittest.TestCase):
    def test_invalid_sentence_is_reasked_once_with_the_error(self):
        mapper = LLMMapper.__new__(LLMMapper)
        mapper.max_output_retries = 2
        mapper.output_failures = []
        mapper.allowed_predicates = None
        mapper.filter_predicates = False
        mapper.prompt_template = "{sentences}{predicates_list}{table_context}"
        mapper.rule_schema = SCHEMA
        mapper.max_rule_repairs = 1
        mapper.llm = stub_llm([
            '{"results": [{"id": 1, "rules": ["t1.rate != \'NA\'"]}, {"id": 2, "rules": ["t1.code = 1"]}]}',
            '{"results": [{"id": 1, "rules": ["t1.rate IS NULL"]}]}',
        ])
        self.assertEqual(mapper._map_sentences(["a", "b"], {}), [["t1.rate IS NULL"], ["t1.code = 1"]])
        self.assertIn("Type mismatch", mapper.llm.complete.call_args_list[1].args[0])
        self.assertEqual(mapper.output_failures, [])

    def test_still_invalid_after_repair_is_logged_as_no_constraint(self):
        mapper = LLMMapper.__new__(LLMMapper)
        mapper.max_output_retries = 0
        mapper.output_failures = []
        mapper.allowed_predicates = None
        mapper.filter_predicates = False
        mapper.prompt_template = "{sentences}{predicates_list}{table_context}"
        mapper.rule_schema = SCHEMA
        mapper.max_rule_repairs = 1
        bad = '{"results": [{"id": 1, "rules": ["t1.missing = 1"]}]}'
        mapper.llm = stub_llm([bad, bad])
        self.assertEqual(mapper._map_sentences(["a"], {}), [None])
        self.assertEqual(len(mapper.output_failures), 1)


class MultiStepRobustnessTests(unittest.TestCase):
    def make(self, responses):
        mapper = MultiStepRefinementMapper.__new__(MultiStepRefinementMapper)
        mapper.llm = stub_llm(responses)
        return mapper

    def test_malformed_json_is_reasked_once(self):
        mapper = self.make(["", '{"is_valid": true}'])
        self.assertEqual(mapper._complete_text("p", {}, "step"), '{"is_valid": true}')
        self.assertIn("not valid JSON", mapper.llm.complete.call_args_list[1].args[0])

    def test_column_names_are_cleaned(self):
        cleaned = MultiStepRefinementMapper._clean_column_names(
            ["orders.amount", "t1.income", " `code` ", ", ", ": [", "income"]
        )
        self.assertEqual(cleaned, ["amount", "income", "code"])


class SpaceBuilderExtensionTests(unittest.TestCase):
    def test_coded_in_sets_families_and_arithmetic(self):
        df = pd.DataFrame({
            "code": [1, 2, 3, 1],
            "amount": [10.5, 20.1, 30.7, 41.2],
            "race_1": [1, 2, 3, 4],
            "race_2": [None, None, None, 5],
            "empty": [None, None, None, None],
        }).astype({"empty": float})
        builder = PredicateSpaceBuilder(
            numeric_categorical_threshold=5, column_families=True, arithmetic_ops=["*"]
        )
        extra = builder.build_extension_predicates(df)
        self.assertIn("t1.code IN $values", extra)
        self.assertIn("t1.race_2 NOT IN $values", extra)  # sparse sibling of a coded column
        self.assertIn("t1.empty IN $values", extra)  # no evidence against being coded
        self.assertNotIn("t1.amount IN $values", extra)
        self.assertIn("t1.race_1 = t1.race_2", extra)
        self.assertIn("t1.amount >= t1.code * $number", extra)
        self.assertEqual(PredicateSpaceBuilder().build_extension_predicates(df), set())


class TableContextTests(unittest.TestCase):
    def test_informative_rows_cover_more_columns(self):
        df = pd.DataFrame({"a": [1, None, None, 4], "b": [None, None, 2, None], "c": [None, 3, None, None]})
        head = TableContextBuilder("head", num_rows=2)._sample_table(df)
        informative = TableContextBuilder("informative", num_rows=2)._sample_table(df)
        self.assertEqual(int(head.notna().any().sum()), 2)
        self.assertEqual(int(informative.notna().any().sum()), 2)
        informative3 = TableContextBuilder("informative", num_rows=3)._sample_table(df)
        self.assertEqual(int(informative3.notna().any().sum()), 3)


class SeedTests(unittest.TestCase):
    def make(self, supported):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.model_name = "m"
        client.max_tokens = None
        client.temperature = 0.0
        client.supported_parameters = supported
        client.structured_output = False
        client.seed = 7
        message = SimpleNamespace(content="ok", refusal=None)
        create = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return client, create

    def test_seed_sent_only_when_catalog_lists_it(self):
        client, create = self.make({"seed", "temperature"})
        client.complete("p")
        self.assertEqual(create.call_args.kwargs.get("seed"), 7)
        client, create = self.make(None)  # catalog unavailable
        client.complete("p")
        self.assertNotIn("seed", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
