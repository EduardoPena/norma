"""Version-2 predicate specification: schema-only file, predicates generated on demand."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.data import DatasetLoader
from src.mappers.llm import LLMMapper
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.predicates.on_demand import OnDemandPredicates
from src.predicates.space_builder import PredicateSpaceBuilder
from src.predicates.space_builder_v2 import SchemaBuilderV2

REF = re.compile(r"\bt\d+\.(\w+)\b")
EXTENSIONS = dict(mirror_second_tuple=True, numeric_range_overlap=0.3, numeric_categorical_threshold=5,
                  column_families=True, arithmetic_ops=["*", "-"])


def frame():
    # rows repeated 5x: domain_values needs >= 5 occurrences per distinct value
    return pd.concat([_rows()] * 5, ignore_index=True)


def _rows():
    return pd.DataFrame({
        "code": [1, 2, 3, 1, 2, 3],
        "code_name": ["a", "b", "c", "a", "b", "c"],
        "amount": [10.5, 20.1, 30.7, 41.2, 5.0, 7.5],
        "limit": [12.0, 25.0, 31.0, 40.0, 6.0, 9.0],
        "race_1": [1, 2, 3, 4, 1, 2],
        "race_2": [None, None, None, 5, None, None],
        "city": ["x", "y", "z", "x", "y", "w"],
    })


def stub_client(responses):
    client = MagicMock()
    client.complete.side_effect = list(responses)
    return client


class SpecFormatTests(unittest.TestCase):
    def test_same_schema_as_v1_plus_new_attributes_and_no_predicates(self):
        df = frame()
        v1_columns = PredicateSpaceBuilder(**EXTENSIONS).build_schema(df)
        spec = SchemaBuilderV2(**EXTENSIONS).build_spec(df, "orders", "orders.csv")
        self.assertEqual(spec["version"], 2)
        self.assertNotIn("predicates", spec)
        columns = spec["tables"][0]["columns"]
        keys = ("name", "type", "allowed_ops")
        self.assertEqual([{k: c[k] for k in keys} for c in columns], [{k: c[k] for k in keys} for c in v1_columns])
        by_name = {c["name"]: c for c in columns}
        self.assertEqual(by_name["amount"]["allowed_arithmetic_ops"], ["*", "-"])
        self.assertEqual(by_name["city"]["allowed_arithmetic_ops"], [])
        self.assertTrue(all(c["description"] for c in columns))
        self.assertEqual(spec["generation"]["column_families"], True)

    def test_domain_values_only_for_small_domains(self):
        spec = SchemaBuilderV2(domain_values_threshold=3).build_spec(frame(), "orders", "orders.csv")
        by_name = {c["name"]: c for c in spec["tables"][0]["columns"]}
        self.assertEqual(by_name["code"]["domain_values"], [1, 2, 3])
        self.assertEqual(by_name["code_name"]["domain_values"], ["a", "b", "c"])
        self.assertNotIn("domain_values", by_name["amount"])
        self.assertNotIn("domain_values", by_name["city"])  # 4 distinct > 3

    def test_sparse_values_are_not_a_domain(self):
        builder = SchemaBuilderV2(domain_values_threshold=20)
        sparse = pd.Series([None] * 1000 + [10.5, 20.1, 33.3])
        self.assertIsNone(builder.domain_values(sparse))
        self.assertEqual(builder.domain_values(pd.Series([1, 2] * 10)), [1, 2])


class OnDemandEquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, options):
        df = frame()
        full = PredicateSpaceBuilder(**options).build_predicates(df)
        spec = SchemaBuilderV2(**options).build_spec(df, "orders", "orders.csv")
        generator = OnDemandPredicates(spec, df)
        subsets = [{"code"}, {"amount", "limit"}, {"race_2"}, {"race_1", "race_2", "code"},
                   {"code", "code_name", "city"}, set(df.columns)]
        for columns in subsets:
            expected = {p for p in full if set(REF.findall(p)) <= columns}
            self.assertEqual(generator.generate(columns), expected, columns)

    def test_default_options(self):
        self.assert_equivalent({})

    def test_with_extensions(self):
        # race_2 is coded only through its family sibling race_1: the subset keeps it
        self.assert_equivalent(EXTENSIONS)

    def test_allowed_for_concrete_predicates(self):
        df = frame()
        generator = OnDemandPredicates(SchemaBuilderV2(**EXTENSIONS).build_spec(df, "o", "o.csv"), df)
        self.assertTrue(generator.allowed_for("t1.code IN (1, 2)"))
        self.assertTrue(generator.allowed_for("t1.amount > t1.limit * 2"))
        self.assertFalse(generator.allowed_for("t1.city > 5"))
        self.assertEqual(generator.out_of_space_counts([["t1.code = 1", "t1.city > 5"], None]), (1, 2))


class DescriptionTests(unittest.TestCase):
    def test_llm_descriptions_with_one_retry_for_missing_columns(self):
        df = frame()[["code", "amount"]]
        first = '{"columns": [{"name": "code", "description": "Category code."}]}'
        second = '{"columns": [{"name": "amount", "description": "Amount in dollars."}]}'
        client = stub_client([first, second])
        spec = SchemaBuilderV2().build_spec(df, "orders", "orders.csv", client=client)
        by_name = {c["name"]: c["description"] for c in spec["tables"][0]["columns"]}
        self.assertEqual(by_name, {"code": "Category code.", "amount": "Amount in dollars."})
        self.assertEqual(client.complete.call_count, 2)
        self.assertIn('"amount"', client.complete.call_args_list[1].args[0])

    def test_profile_descriptions_are_deterministic(self):
        builder = SchemaBuilderV2()
        a = builder.build_spec(frame(), "orders", "orders.csv")
        b = builder.build_spec(frame(), "orders", "orders.csv")
        self.assertEqual(a, b)
        code = next(c for c in a["tables"][0]["columns"] if c["name"] == "code")
        self.assertIn("labels in code_name", code["description"])


class LoaderAndMapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        frame().to_csv(d / "orders.csv", index=False)
        SchemaBuilderV2(**EXTENSIONS).build(str(d / "orders.csv"))
        (d / "orders_text2rule_gt.json").write_text(json.dumps({"mapping": [
            {"id": 1, "sentences": ["Amount must not exceed the limit."], "rules": ["t1.amount > t1.limit"]}]}))
        self.dir = d

    def tearDown(self):
        self.tmp.cleanup()

    def test_loader_reads_schema_file_in_v2_only(self):
        dfs, space, gt = DatasetLoader(str(self.dir), space_version=2).load()
        self.assertEqual(space.version, 2)
        self.assertEqual(space.allowed_predicates, set())
        self.assertIn("domain_values", space.schema["orders"]["code"])
        with self.assertRaises(ValueError):  # no *_predicates.json in the folder
            DatasetLoader(str(self.dir)).load()

    def spec_and_df(self):
        spec = json.loads((self.dir / "orders_schema.json").read_text())
        return spec, pd.read_csv(self.dir / "orders.csv")

    def test_multi_step_candidates_come_only_from_selected_columns(self):
        spec, df = self.spec_and_df()
        mapper = MultiStepRefinementMapper.__new__(MultiStepRefinementMapper)
        mapper.predicate_generator = OnDemandPredicates(spec, df)
        mapper.allowed_predicates = set()
        mapper.max_candidate_predicates = 1000
        import threading
        mapper._generated_lock = threading.Lock()
        candidates = mapper._filter_predicates_by_columns(["amount", "limit"], "amount above limit")
        self.assertTrue(candidates)
        self.assertTrue(all(set(REF.findall(p)) <= {"amount", "limit"} for p in candidates))
        self.assertEqual(mapper.allowed_predicates, set(mapper.predicate_generator.generate(["amount", "limit"])))

    def test_single_prompt_offers_only_selected_columns(self):
        spec, df = self.spec_and_df()
        mapper = LLMMapper.__new__(LLMMapper)
        mapper.predicate_generator = OnDemandPredicates(spec, df)
        mapper.predicate_tables = spec["tables"]
        mapper.max_relevant_columns = 2
        mapper.max_candidate_predicates = 1000
        mapper.allowed_predicates = None
        mapper.filter_predicates = False
        mapper.prompt_template = "{sentences}|{predicates_list}|{table_context}"
        prompt = mapper._build_prompt(["The amount must not exceed the limit."], {})
        offered = [line.strip()[2:] for line in prompt.split("|")[1].splitlines() if line.strip()]
        selected = set(mapper.last_predicate_selection[0]["columns"])
        self.assertTrue(offered)
        self.assertTrue(all(set(REF.findall(p)) <= selected for p in offered))


if __name__ == "__main__":
    unittest.main()
