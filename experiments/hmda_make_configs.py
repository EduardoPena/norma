"""Generate E1/E2/E3 runner configs (HMDA or paper datasets) and an execution plan.

Hyperparameters mirror experiments/configs/*_official_openrouter_all.yaml
(the paper protocol); only the dataset options and results_dir differ.
"""

import argparse
import json
from pathlib import Path

import yaml

CONFIG_DIR = Path("experiments/configs")
MANIFEST = yaml.safe_load((CONFIG_DIR / "hmda_models.yaml").read_text())
RESULTS_ROOT = "results/hmda"
ROW_LIMIT = 1000000

# filter_predicates: only for spaces too large for a Single-Prompt context
# (HMDAZ: 590k templates; HMDA_CHECKS: 163k since the 2026-10-10 extension with
# coded IN sets, column families and arithmetic, see extend_predicate_space.py).
# Paper datasets have <= 128k rows, so ROW_LIMIT reads them whole (as in the paper).
DATASETS = {
    "hmdaz": {"gt_file": "hmdaz_text2rule_gt.json", "filter_predicates": True},
    "hmda_checks": {
        "gt_file": "hmda_regc_checks-text2rule-gt.json",
        "filter_predicates": True,
    },
    "hospital": {"gt_file": "hospital-holo_text2rule_gt.json", "filter_predicates": False},
    "restaurant": {
        "gt_file": "restaurant-inspections_text2rule_gt.json",
        "filter_predicates": False,
    },
    "sisu": {"gt_file": "sisu_text2rule_gt.json", "filter_predicates": False},
    "tax": {"gt_file": "tax_text2rule_gt.json", "filter_predicates": False},
}

# Paper prompt variants (Fig. 6): name -> (prompt_key, use_context)
CLASSIFIER_PROMPTS = {
    "No Context": ("zero_shot", False),
    "Table Context": ("zero_shot_with_table", True),
    "Examples": ("with_examples", False),
    "Examples + Table Context": ("with_examples_and_table", True),
}

MULTI_STEP = {
    "implementation": "multi_step_refinement",
    "provider": "openrouter",
    "max_refinement_iterations": 2,
    "prompt_file": "multi_step_refinement",
}


def dataset_options(dataset):
    return {
        dataset: {
            "gt_file": DATASETS[dataset]["gt_file"],
            "sample_rows": ROW_LIMIT,  # first N rows (pd.read_csv nrows)
            "stream_csv": True,
            "drop_multi_rule_sentences": False,
        }
    }


def single_prompt(dataset, slug):
    mapper = {
        "implementation": "llm",
        "provider": "openrouter",
        "model": slug,
        "prompt_key": "with_examples",
        "batch_size": 10,
    }
    if DATASETS[dataset]["filter_predicates"]:
        # 590k HMDAZ templates cannot fit any context window (deviation).
        mapper.update(
            filter_predicates=True,
            max_candidate_predicates=1000,
            max_relevant_columns=12,
        )
    return mapper


def e1(dataset, model):
    return {
        "experiment_name": f"hmda_e1_{dataset}_{model['key']}",
        "experiment_type": "classifier",
        "results_dir": f"{RESULTS_ROOT}/e1_classifiers",
        "datasets": [dataset],
        "dataset_options": dataset_options(dataset),
        "variants": [
            {
                "name": f"{model['paper_name']} | {name}",
                "implementation": "llm",
                "provider": "openrouter",
                "model": model["slug"],
                "prompt_key": prompt_key,
                "batch_size": 10,
                "use_context": use_context,
                **({"num_rows_context": 3} if use_context else {}),
            }
            for name, (prompt_key, use_context) in CLASSIFIER_PROMPTS.items()
        ],
    }


def e2(dataset, model, mapper_name):
    mapper = (
        single_prompt(dataset, model["slug"])
        if mapper_name == "SinglePrompt"
        else {**MULTI_STEP, "model": model["slug"]}
    )
    label = "Single Prompt" if mapper_name == "SinglePrompt" else "Multi-Step"
    return {
        "experiment_name": f"hmda_e2_{dataset}_{model['key']}_{mapper_name}",
        "experiment_type": "mapper_gt_only",
        "results_dir": f"{RESULTS_ROOT}/e2_mapping",
        "datasets": [dataset],
        "dataset_options": dataset_options(dataset),
        "variants": [
            {
                "name": f"{label} | {model['paper_name']}",
                "use_context": False,
                "mapper": mapper,
            }
        ],
    }


def e3(dataset, variant):
    by_key = {model["key"]: model for model in MANIFEST["models"]}
    classifier = by_key[MANIFEST["end2end"]["classifier"]]
    mapper = by_key[MANIFEST["end2end"]["mapper"]]
    if variant == "C_MS":
        # Classifier prompt chosen in E1 (paper: "with_examples"); table prompts get
        # 3 sample rows as in E1. The Multi-Step mapper ignores the table context.
        prompt_key = MANIFEST["end2end"].get("classifier_prompt", "with_examples")
        use_context = "table" in prompt_key
        spec = {
            "name": f"C+MS | {classifier['paper_name']} + {mapper['paper_name']}",
            "mode": "SENTENCE_FILTERED",
            "use_context": use_context,
            **({"num_rows_context": 3} if use_context else {}),
            "classifier": {
                "implementation": "llm",
                "provider": "openrouter",
                "model": classifier["slug"],
                "prompt_key": prompt_key,
                "batch_size": 10,
            },
            "mapper": {**MULTI_STEP, "model": mapper["slug"]},
        }
    else:
        spec = {
            "name": f"Multi-Step | {mapper['paper_name']}",
            "mode": "SENTENCE_DIRECT",
            "use_context": False,
            "mapper": {**MULTI_STEP, "model": mapper["slug"]},
        }
    return {
        "experiment_name": f"hmda_e3_{dataset}_{variant}",
        "experiment_type": "mapper",
        "results_dir": f"{RESULTS_ROOT}/e3_end2end",
        "datasets": [dataset],
        "dataset_options": dataset_options(dataset),
        "variants": [spec],
    }


def main():
    global MANIFEST, RESULTS_ROOT, ROW_LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=CONFIG_DIR / "hmda_models.yaml")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument(
        "--row-limit", type=int, default=ROW_LIMIT, help="first N rows of each CSV"
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=("e1", "e2", "e3"), default=["e1", "e2", "e3"]
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=["hmdaz", "hmda_checks"]
    )
    args = parser.parse_args()
    MANIFEST = yaml.safe_load(args.models.read_text())
    RESULTS_ROOT = args.results_root
    ROW_LIMIT = args.row_limit
    args.config_dir.mkdir(parents=True, exist_ok=True)
    models = MANIFEST["models"]

    if "e3" in args.experiments and not MANIFEST.get("end2end"):
        parser.error("E3 needs end2end classifier/mapper keys in the manifest")

    jobs = []
    if "e1" in args.experiments:
        for model in models:  # manifest order: cheap models first
            for dataset in args.datasets:
                jobs.append(e1(dataset, model))
    if "e2" in args.experiments:
        for model in models:
            for dataset in args.datasets:
                for mapper_name in ("SinglePrompt", "MultiStep"):
                    jobs.append(e2(dataset, model, mapper_name))
    if "e3" in args.experiments:
        for dataset in args.datasets:
            for variant in ("MultiStep", "C_MS"):
                jobs.append(e3(dataset, variant))

    plan = []
    for job in jobs:
        path = args.config_dir / f"{job['experiment_name']}.yaml"
        path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
        plan.append({"config": str(path)})
    plan_path = args.config_dir / f"hmda_plan_{'_'.join(args.experiments)}.json"
    plan_path.write_text(json.dumps({"jobs": plan}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(plan)} configs and {plan_path}")


if __name__ == "__main__":
    main()
