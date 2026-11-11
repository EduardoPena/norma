"""Parallel requests keep results in input order and DB access is serialized."""

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.dbms.db_interface import DBExecutor
from src.mappers.multi_step_refinement import MultiStepRefinementMapper
from src.utils.parallel import parallel_map


class ParallelMapTests(unittest.TestCase):
    def test_preserves_order_and_runs_concurrently(self):
        active, peak, lock = [0], [0], threading.Lock()

        def work(item):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05 * (5 - item))  # later items finish first
            with lock:
                active[0] -= 1
            return item * 10

        self.assertEqual(parallel_map(work, list(range(5)), 4), [0, 10, 20, 30, 40])
        self.assertGreater(peak[0], 1)

    def test_exceptions_propagate(self):
        def work(item):
            if item == 2:
                raise RuntimeError("infra")
            return item

        with self.assertRaises(RuntimeError):
            parallel_map(work, [1, 2, 3], 3)


class MultiStepParallelTests(unittest.TestCase):
    def test_sentences_processed_in_parallel_keep_order(self):
        mapper = MultiStepRefinementMapper.__new__(MultiStepRefinementMapper)
        mapper.output_failures = []
        mapper.parallel_requests = 4

        def process(sentence, context):
            time.sleep(0.02 * (4 - int(sentence)))
            return [f"t1.a = {sentence}"]

        with patch.object(mapper, "_process_single_sentence", side_effect=process):
            results = mapper._map_sentences(["0", "1", "2", "3"], {})
        self.assertEqual(results, [[f"t1.a = {i}"] for i in range(4)])
        self.assertEqual(mapper.processing_errors, [])


class DBExecutorConcurrencyTests(unittest.TestCase):
    def test_concurrent_queries_return_correct_rows(self):
        db = DBExecutor(pd.DataFrame({"a": range(1000)}))
        queries = [f'SELECT COUNT(*) FROM "data_table" WHERE a < {n}' for n in range(0, 1000, 25)]
        results = parallel_map(lambda q: db.execute_sql(q)["rows"][0][0], queries, 8)
        self.assertEqual(results, list(range(0, 1000, 25)))
        db.close()


class DBExecutorGuardTests(unittest.TestCase):
    def test_result_cap_and_timeout_fail_the_query_not_the_process(self):
        db = DBExecutor(pd.DataFrame({"a": [1] * 3000}))
        db.max_result_rows = 1000
        pair = 'SELECT t1.tid, t2.tid FROM "data_table" t1, "data_table" t2 WHERE t1.a = t2.a'
        result = db.execute_violation_query(pair)
        self.assertFalse(result["success"])
        self.assertIn("rows", result["error"])

        db.max_result_rows = 10**12
        db.query_timeout_s = 1
        triple = (
            'SELECT COUNT(*) FROM "data_table" t1, "data_table" t2, "data_table" t3 '
            "WHERE t1.a = t2.a AND t2.a = t3.a AND t1.tid + t2.tid > t3.tid"
        )
        result = db.execute_sql(triple)
        self.assertFalse(result["success"])
        self.assertIn("interrupted", result["error"])
        self.assertEqual(db.execute_sql('SELECT COUNT(*) FROM "data_table"')["rows"], [(3000,)])
        db.close()


class JobModelsTests(unittest.TestCase):
    def test_collects_classifier_and_mapper_models(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "experiments" / "hmda_smoke.py"
        spec = importlib.util.spec_from_file_location("hmda_smoke", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = Path(__file__).resolve().parents[1] / "experiments/configs/hmda_e3_hmdaz_C_MS.yaml"
        self.assertEqual(module.job_models(config), {"qwen/qwen3-32b", "openai/gpt-5"})


if __name__ == "__main__":
    unittest.main()
