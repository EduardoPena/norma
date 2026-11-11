"""Prevent partial mapper failures from being reported as successful experiments."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def load_runner(filename, class_name):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(class_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, getattr(module, class_name).__new__(getattr(module, class_name))


class ExperimentProcessingErrorTests(unittest.TestCase):
    def test_end2end_rejects_partial_failures_before_evaluation(self):
        module, runner = load_runner("1_run_end2end.py", "MapperEvalFullExperiment")
        for mode in ("SENTENCE_DIRECT", "SENTENCE_FILTERED"):
            with self.subTest(mode=mode):
                pipeline = SimpleNamespace(
                    mapper=SimpleNamespace(processing_errors=["provider failure"]),
                    process=MagicMock(
                        return_value=[
                            ("mapped", True, ["t1.amount <= 0"]),
                            ("failed", True, None),
                        ]
                    ),
                )
                with patch.object(
                    runner, "_create_sentence_pipeline", return_value=pipeline
                ), patch.object(module, "ClassificationEvaluator") as evaluator:
                    with self.assertRaisesRegex(RuntimeError, "provider failure"):
                        runner._run_sentence_variant(
                            "offline",
                            {"mapper": {"implementation": "multi_step_refinement"}},
                            mode,
                            ["mapped", "failed"],
                            {},
                            {},
                            None,
                            "offline",
                        )
                    evaluator.assert_not_called()

    def test_end2end_accepts_legitimate_null_predictions(self):
        module, runner = load_runner("1_run_end2end.py", "MapperEvalFullExperiment")
        pipeline = SimpleNamespace(
            mapper=SimpleNamespace(processing_errors=[]),
            process=MagicMock(return_value=[("non-rule", False, None)]),
        )
        with patch.object(
            runner, "_create_sentence_pipeline", return_value=pipeline
        ), patch.object(
            module,
            "ClassificationEvaluator",
            side_effect=RuntimeError("reached evaluation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached evaluation"):
                runner._run_sentence_variant(
                    "offline",
                    {"mapper": {"implementation": "multi_step_refinement"}},
                    "SENTENCE_DIRECT",
                    ["non-rule"],
                    {},
                    {},
                    None,
                    "offline",
                )

    def test_gt_only_rejects_partial_failures_before_normalization(self):
        _, runner = load_runner("3_run_mapping.py", "MapperGTOnlyExperiment")
        mapper = SimpleNamespace(
            processing_errors=["provider failure"],
            map=MagicMock(
                return_value=[("mapped", ["t1.amount <= 0"]), ("failed", None)]
            ),
        )
        with patch.object(runner, "_create_mapper", return_value=mapper), patch.object(
            runner, "_map_semantically_valid_constraint_predicates"
        ) as normalize:
            with self.assertRaisesRegex(RuntimeError, "provider failure"):
                runner.run_variant(
                    "offline",
                    {"mapper": {"implementation": "multi_step_refinement"}},
                    ["mapped", "failed"],
                    {},
                    {},
                    None,
                    "offline",
                )
            normalize.assert_not_called()

    def test_gt_only_accepts_legitimate_null_predictions(self):
        _, runner = load_runner("3_run_mapping.py", "MapperGTOnlyExperiment")
        mapper = SimpleNamespace(
            processing_errors=[],
            map=MagicMock(return_value=[("non-rule", None)]),
        )
        with patch.object(runner, "_create_mapper", return_value=mapper), patch.object(
            runner,
            "_map_semantically_valid_constraint_predicates",
            side_effect=RuntimeError("reached normalization"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached normalization"):
                runner.run_variant(
                    "offline",
                    {"mapper": {"implementation": "multi_step_refinement"}},
                    ["non-rule"],
                    {},
                    {},
                    None,
                    "offline",
                )


if __name__ == "__main__":
    unittest.main()
