"""Pattern predicates in the rule validator and the predicate-space builder options."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.predicates import RuleValidationError, canonical_rule, parse_predicate
from src.predicates.space_builder import PredicateSpaceBuilder


class PatternPredicateTests(unittest.TestCase):
    def test_like_and_not_like_are_accepted(self):
        self.assertEqual(
            canonical_rule("t1.type LIKE '%Re-inspection%'"), "t1.type LIKE '%Re-inspection%'"
        )
        self.assertEqual(canonical_rule("t1.x NOT LIKE 'a_c'"), "t1.x NOT LIKE 'a_c'")

    def test_like_requires_a_pattern_string(self):
        for bad in ("t1.x LIKE t1.y", "t1.x LIKE 5", "t1.x NOT LIKE"):
            with self.subTest(bad=bad), self.assertRaises(RuleValidationError):
                parse_predicate(bad)


class SpaceBuilderOptionTests(unittest.TestCase):
    def build(self, **options):
        df = pd.DataFrame(
            {
                "salary": [1000, 5000, 20000, 80000, 120000],
                "exemption": [0, 1500, 3000, 4500, 9000],
                "edition": [1, 2, 1, 2, 3],
                "city": ["a", "b", "c", "a", "b"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "tax.csv"
            df.to_csv(csv, index=False)
            return set(PredicateSpaceBuilder(**options).build(str(csv), save=False)["predicates"])

    def test_defaults_are_unchanged(self):
        space = self.build()
        self.assertNotIn("t2.edition = $int", space)
        self.assertNotIn("t1.exemption > t1.salary", space)

    def test_second_tuple_mirrors_single_column_templates_only(self):
        space = self.build(mirror_second_tuple=True)
        self.assertIn("t2.edition = $int", space)
        self.assertIn("t2.city IS NULL", space)
        self.assertFalse(any(p.count("t2.") == 2 for p in space))

    def test_numeric_range_overlap_adds_same_tuple_comparisons(self):
        space = self.build(numeric_range_overlap=0.3)
        self.assertIn("t1.exemption > t1.salary", space)
        self.assertIn("t1.salary >= t1.exemption", space)


if __name__ == "__main__":
    unittest.main()
