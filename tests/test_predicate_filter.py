"""Offline regression tests for opt-in Single-Prompt predicate filtering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.mappers.predicate_filter import (
    filter_predicates_by_columns,
    select_relevant_columns,
    simple_predicates,
)


class PredicateFilterTests(unittest.TestCase):
    def setUp(self):
        self.predicates = {
            "t1.amount > 100",
            "t1.amount > t2.amount",
            "t1.fees > 50",
            "t1.fees > 0.05 * t1.amount",
            "t1.hidden_status = <const>",
            "t1.other > t1.unrelated",
        }
        self.tables = [
            {
                "columns": [
                    {"name": "amount", "description": "Order amount"},
                    {"name": "fees", "description": "Total fees"},
                    {"name": "hidden_status", "description": "Implicit classification"},
                    {"name": "other"},
                    {"name": "unrelated"},
                ]
            }
        ]

    def mapper(self, **options):
        with patch("src.mappers.llm.get_llm_client") as client:
            mapper = LLMMapper(
                prompt_template="{sentences}\n{predicates_list}\n{table_context}",
                allowed_predicates=list(self.predicates),
                predicate_tables=self.tables,
                **options,
            )
            client.return_value.complete.assert_not_called()
            return mapper

    def test_existing_multistep_ranking_and_fallback(self):
        expected = [
            "t1.amount > 100",
            "t1.amount > t2.amount",
            "t1.fees > 50",
            "t1.fees > 0.05 * t1.amount",
        ]
        state = SimpleNamespace(
            allowed_predicates=self.predicates, max_candidate_predicates=1000
        )
        self.assertEqual(
            filter_predicates_by_columns(
                self.predicates, ["amount", "fees"], "fees percent amount"
            ),
            expected,
        )
        self.assertEqual(
            MultiStepRefinementMapper._filter_predicates_by_columns(
                state, ["amount", "fees"], "fees percent amount"
            ),
            expected,
        )
        self.assertEqual(
            filter_predicates_by_columns(
                self.predicates, [], max_candidate_predicates=2
            ),
            sorted(self.predicates)[:2],
        )
        self.assertEqual(filter_predicates_by_columns(set(), ["amount"]), [])

    def test_retrieval_is_deterministic_and_schema_grounded(self):
        columns = select_relevant_columns("Fees exceed the amount", self.tables, 2)
        self.assertEqual(set(columns), {"fees", "amount"})
        self.assertEqual(
            select_relevant_columns(
                "Fees exceed the amount", list(reversed(self.tables)), 2
            ),
            columns,
        )
        self.assertEqual(select_relevant_columns("zzz", self.tables), [])

    def test_disabled_preserves_full_prompt(self):
        mapper = self.mapper()
        prompt = mapper._build_prompt(["Fees exceed amount"], {})
        for predicate in self.predicates:
            self.assertIn(f"  - {predicate}\n", prompt + "\n")
        self.assertEqual(mapper.last_predicate_selection, [])

    def test_table_context_uses_builder_key_with_legacy_fallback(self):
        mapper = self.mapper()
        for key in ("table_context", "table_sample"):
            with self.subTest(key=key):
                prompt = mapper._build_prompt(["Fees exceed amount"], {key: "TABLE DATA"})
                self.assertIn("TABLE DATA", prompt)
        prompt = mapper._build_prompt(
            ["Fees exceed amount"],
            {"table_context": "PRIMARY DATA", "table_sample": "LEGACY DATA"},
        )
        self.assertIn("PRIMARY DATA", prompt)
        self.assertNotIn("LEGACY DATA", prompt)

    def test_enabled_preserves_implicit_columns_and_compound_candidates(self):
        mapper = self.mapper(filter_predicates=True, max_relevant_columns=2)
        prompt = mapper._build_prompt(["Fees exceed amount"], {})
        self.assertIn("t1.hidden_status = <const>", prompt)
        self.assertIn("t1.fees > 0.05 * t1.amount", prompt)
        self.assertNotIn("t1.other > t1.unrelated", prompt)
        self.assertEqual(mapper.allowed_predicates, self.predicates)
        mapper.llm.complete.assert_not_called()

    def test_batch_unions_per_sentence_candidates(self):
        mapper = self.mapper(filter_predicates=True, max_relevant_columns=2)
        prompt = mapper._build_prompt(
            ["Fees exceed amount", "Other exceeds unrelated"], {}
        )
        self.assertIn("t1.fees > 0.05 * t1.amount", prompt)
        self.assertIn("t1.other > t1.unrelated", prompt)
        self.assertEqual(len(mapper.last_predicate_selection), 2)

    def test_empty_retrieval_keeps_simple_and_bounded_fallback(self):
        mapper = self.mapper(filter_predicates=True, max_candidate_predicates=1)
        prompt = mapper._build_prompt(["zzz"], {})
        for predicate in simple_predicates(self.predicates):
            self.assertIn(predicate, prompt)
        self.assertEqual(mapper.last_predicate_selection[0]["candidates"], 1)

    def test_invalid_configuration_is_observable(self):
        with self.assertRaises(ValueError):
            self.mapper(filter_predicates=True, max_relevant_columns=0)
        with self.assertRaises(ValueError):
            LLMMapper(filter_predicates=True)
        with self.assertRaises(ValueError):
            self.mapper(max_tokens=0)

    def test_completion_limit_reaches_client(self):
        for limit in (None, 8192):
            with self.subTest(limit=limit):
                with patch("src.mappers.llm.get_llm_client") as client:
                    LLMMapper(
                        model_name="offline",
                        provider="openrouter",
                        prompt_template="{sentences}",
                        max_tokens=limit,
                    )
                    client.assert_called_once_with(
                        "offline",
                        provider="openrouter",
                        max_tokens=limit,
                        structured_output=True,
                    )
                    client.return_value.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
