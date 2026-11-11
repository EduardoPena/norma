"""Baseline evaluation helpers: SQL-to-rules reading and DC discovery inputs."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "baselines"))

from sql_rules import sql_to_rules  # noqa: E402
from src.predicates import normalize_rules  # noqa: E402
import prepare_dc_discovery as prep  # noqa: E402


def rules(sql):
    result, reason = sql_to_rules(sql)
    return normalize_rules(result) if result else reason


class SqlToRulesTests(unittest.TestCase):
    def test_pair_query_drops_tid_predicates(self):
        self.assertEqual(
            rules("SELECT t1.tid, t2.tid FROM t1, t1 AS t2 WHERE t1.tid <> t2.tid "
                  "AND t1.state = t2.state AND t1.salary < t2.salary AND t1.rate > t2.rate"),
            [{"AND": ["t1.state = t2.state", "t1.salary < t2.salary", "t1.rate > t2.rate"]}])

    def test_unqualified_columns_in_and_negation(self):
        self.assertEqual(
            rules("SELECT tid FROM t1 WHERE purchaser_type IN (1, 3) AND NOT loan_type = 1"),
            [{"AND": ["t1.purchaser_type IN (1,3)", "t1.loan_type != 1"]}])

    def test_between_or_and_null(self):
        self.assertEqual(
            rules("SELECT t1.tid FROM t1 WHERE score BETWEEN 0 AND 13 AND (grade IS NULL OR grade <> 'A')"),
            [{"AND": ["t1.score >= 0", "t1.score <= 13", {"OR": ["t1.grade IS NULL", "t1.grade != 'A'"]}]}])

    def test_negation_is_pushed_down(self):
        self.assertEqual(rules("SELECT t1.tid FROM t1 WHERE t1.a NOT BETWEEN 1 AND 5"),
                         [{"OR": ["t1.a < 1", "t1.a > 5"]}])
        self.assertEqual(rules("SELECT t1.tid FROM t1 WHERE NOT (t1.a = 1 OR t1.b IN (2, 3))"),
                         [{"AND": ["t1.a != 1", "t1.b NOT IN (2,3)"]}])

    def test_join_condition_and_cast_arithmetic(self):
        self.assertEqual(
            rules("SELECT t1.tid FROM t1 JOIN t1 AS t2 ON t1.cpf = t2.cpf AND t1.tid <> t2.tid "
                  "WHERE TRY_CAST(t1.loan_amount AS DOUBLE) >= 5 * t1.applicant_income"),
            [{"AND": ["t1.cpf = t2.cpf", "t1.loan_amount >= (5 * t1.applicant_income)"]}])

    def test_correlated_exists_is_a_pair_dc(self):
        self.assertEqual(
            rules("SELECT t1.tid FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS t2 WHERE "
                  "t2.ProviderNumber = t1.ProviderNumber AND t2.HospitalName <> t1.HospitalName AND t2.tid != t1.tid)"),
            [{"AND": ["t1.ProviderNumber = t2.ProviderNumber", "t1.HospitalName != t2.HospitalName"]}])
        self.assertEqual(
            rules("SELECT t1.tid FROM t1 WHERE t1.state = 'NY' AND EXISTS (SELECT 1 FROM t1 x WHERE x.salary > t1.salary)"),
            [{"AND": ["t1.state = 'NY'", "t1.salary < t2.salary"]}])

    def test_non_extractable_queries_are_reported(self):
        self.assertEqual(rules("SELECT t1.tid FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t1 t2 WHERE t2.a = t1.a)"),
                         "NOT EXISTS")
        self.assertEqual(rules("SELECT t1.tid FROM t1 WHERE t1.a = (SELECT t2.a FROM t1 AS t2 WHERE t2.b = t1.b)"),
                         "subquery")
        self.assertEqual(rules("SELECT t1.tid FROM t1 GROUP BY t1.a HAVING COUNT(*) > 1"), "aggregation")
        self.assertEqual(rules("SELECT t1.tid FROM t1"), "no WHERE clause")
        self.assertTrue(rules("SELECT t1.tid FROM t1 WHERE CASE WHEN t1.a = 1 THEN t1.b ELSE 0 END > 3")
                        .startswith("unsupported expression"))


class PrepareDcDiscoveryTests(unittest.TestCase):
    def test_type_inference_matches_pandas_and_protects_ids(self):
        self.assertEqual(prep.infer_type("a", ["1", "2", "3"]), "int")
        self.assertEqual(prep.infer_type("a", ["1", "", "3"]), "float")
        self.assertEqual(prep.infer_type("a", ["1.5", "2"]), "float")
        self.assertEqual(prep.infer_type("a", ["NA", "2"]), "str")
        self.assertEqual(prep.infer_type("loan_number", ["10514114684615632"]), "str")
        self.assertEqual(prep.infer_type("respondent_id", ["0000001281"]), "str")
        self.assertEqual(prep.infer_type("ZipCode", ["35233"]), "str")

    def test_gold_columns_of_tax(self):
        self.assertTrue({"state", "salary", "rate"} <= prep.gt_columns("tax"))


if __name__ == "__main__":
    unittest.main()
