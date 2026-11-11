"""Run smoke subsets or explicit experiment YAMLs with OpenRouter telemetry."""

import argparse
import contextlib
import copy
import fcntl
import threading
import gc
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.llm.client import OpenRouterClient


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@contextlib.contextmanager
def file_lock(path: Path):
    """Exclusive lock shared by concurrent job processes of the same run."""
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("inf")


def job_models(config_path: Path) -> set:
    """OpenRouter models a runner config calls (jobs sharing one never overlap)."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = set()
    for variant in payload["variants"]:
        for block in (variant, variant.get("mapper") or {}, variant.get("classifier") or {}):
            if block.get("model"):
                models.add(block["model"])
    return models


def exceeds_cost_limit(limit: float | None, spent: float, reserve: float) -> bool:
    """An explicit null financial limit permits calls while retaining telemetry."""
    return limit is not None and spent + reserve > limit


def prepare_runner_jobs(paths: list[Path], output: Path) -> list[tuple[str, str, Path]]:
    registry = {
        "classifier": ("2_run_classifiers.py", "ClassifierExperiment"),
        "mapper_gt_only": ("3_run_mapping.py", "MapperGTOnlyExperiment"),
        "mapper": ("1_run_end2end.py", "MapperEvalFullExperiment"),
    }
    jobs = []
    snapshots = output / "runner_configs"
    snapshots.mkdir(exist_ok=True)
    names = set()
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.name in names:
            raise ValueError(f"Duplicate runner config name: {path.name}")
        names.add(path.name)
        if payload.get("smoke_scope"):
            raise ValueError(
                "Explicit full-run configs must not contain smoke GT subsets"
            )
        filename, class_name = registry[payload["experiment_type"]]
        if payload["experiment_type"] == "mapper" and any(
            variant["mode"] not in ("SENTENCE_DIRECT", "SENTENCE_FILTERED")
            for variant in payload["variants"]
        ):
            raise ValueError("Only sentence-level end-to-end modes are supported")
        for dataset in payload["datasets"]:
            options = payload["dataset_options"][dataset]
            if not 1 <= options["sample_rows"] <= 1000000:
                raise ValueError(
                    "Explicit runner configs require sample_rows <= 1000000"
                )
            if options.get("drop_multi_rule_sentences", False):
                raise ValueError("Full runs must preserve multi-rule GT sentences")
        snapshot = snapshots / path.name
        if snapshot.exists():
            if yaml.safe_load(snapshot.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"Config changed since the previous run: {path.name}")
        else:
            snapshot.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
        jobs.append((filename, class_name, snapshot))
    return jobs


def prepare(config: dict, output: Path) -> list[tuple[str, str, Path]]:
    """Build smoke configs from the real E1/E2/E3 generators with small inputs."""
    import hmda_make_configs as generator

    registry = {
        "classifier": ("2_run_classifiers.py", "ClassifierExperiment"),
        "mapper_gt_only": ("3_run_mapping.py", "MapperGTOnlyExperiment"),
        "mapper": ("1_run_end2end.py", "MapperEvalFullExperiment"),
    }
    cheap = {
        "key": "smoke",
        "paper_name": config["mapper_model"].split("/")[-1],
        "slug": config["mapper_model"],
    }
    jobs = []
    for dataset, options in config["datasets"].items():
        source = Path("data") / dataset / options["gt_file"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        positives = [entry for entry in payload["mapping"] if entry.get("rules")]
        negatives = [entry for entry in payload["mapping"] if not entry.get("rules")]
        selected = (
            positives[: config["positive_entries"]]
            + negatives[: config["negative_entries"]]
        )
        if len(selected) != config["positive_entries"] + config["negative_entries"]:
            raise ValueError(f"{dataset}: insufficient smoke entries")
        subset = copy.deepcopy(payload)
        subset["mapping"] = selected
        gt_path = output / f"{dataset}_smoke_gt.json"
        write_json(gt_path, subset)
        phase_configs = [
            generator.e1(dataset, {**cheap, "slug": config["classifier_model"]}),
            generator.e2(dataset, cheap, "SinglePrompt"),
            generator.e2(dataset, cheap, "MultiStep"),
            generator.e3(dataset, "C_MS"),
        ]
        e3_mapper = phase_configs[-1]["variants"][0]["mapper"]
        e3_mapper["model"] = config["end2end_mapper_model"]
        e3_variant = phase_configs[-1]["variants"][0]
        e3_variant["classifier"]["model"] = config["end2end_classifier_model"]
        e3_variant["name"] = (
            f"C+MS | {config['end2end_classifier_model'].split('/')[-1]}"
            f" + {config['end2end_mapper_model'].split('/')[-1]}"
        )
        for phase_config in phase_configs:
            phase_config["dataset_options"][dataset].update(
                gt_file=str(gt_path.resolve()), sample_rows=config["sample_rows"]
            )
            phase_config.update(
                experiment_name=phase_config["experiment_name"].replace(
                    "hmda_", "smoke_", 1
                ),
                results_dir=str(output / "runs"),
                smoke_scope={
                    "source_gt": str(source),
                    "selected_ids": [entry["id"] for entry in selected],
                    "not_full_experiment": True,
                },
            )
            config_path = output / f"{phase_config['experiment_name']}.yaml"
            config_path.write_text(
                yaml.safe_dump(phase_config, sort_keys=False), encoding="utf-8"
            )
            jobs.append((*registry[phase_config["experiment_type"]], config_path))
    return jobs


def run_isolated(args, output: Path, jobs, config: dict) -> int:
    """Run pending jobs in fresh subprocesses (same telemetry files).

    Up to ``parallel_jobs`` jobs run at once, never two that call the same model
    (they would share its provider rate limit), and a new job starts only with
    ``min_free_memory_gib`` available. Each job logs to job_logs/<job>.log;
    ``parallel_requests`` sets concurrent LLM requests inside a job.
    """
    source = (
        ["--plan", str(args.plan)]
        if args.plan
        else ["--runner-configs", *map(str, args.runner_configs)]
    )
    parallel_jobs = int(config.get("parallel_jobs", 1))
    min_free_gib = float(config.get("min_free_memory_gib", 0))
    env = {
        **os.environ,
        "NORMA_PARALLEL_REQUESTS": str(config.get("parallel_requests", 1)),
        "NORMA_JSON_OBJECT_MODELS": ",".join(config.get("json_object_models") or []),
        "PYTHONUNBUFFERED": "1",  # live per-job logs (tail -f)
    }
    log_dir = output / "job_logs"
    log_dir.mkdir(exist_ok=True)
    summary_path = output / "summary.json"

    def summary():
        with file_lock(output / "summary.lock"):
            return json.loads(summary_path.read_text()) if summary_path.exists() else {}

    done = {p["job"] for p in summary().get("phases", []) if p["status"] == "passed"}
    pending = []
    for _, _, config_path in jobs:
        if config_path.stem in done:
            print(f"Skipping completed job: {config_path.stem}", flush=True)
        else:
            pending.append(config_path)
    running = {}
    while pending or running:
        for process, (job, _, log) in list(running.items()):
            if process.poll() is not None:
                log.close()
                print(f"Finished job: {job} (exit {process.returncode})", flush=True)
                del running[process]
        stop = summary().get("stop_reason")
        if stop and pending:
            print(f"Stop reason set ({stop}); not starting new jobs", flush=True)
            pending = []
        busy = set().union(*(models for _, models, _ in running.values()))
        if pending and len(running) < parallel_jobs and (
            not running or memory_available_gib() >= min_free_gib
        ):
            for config_path in pending:
                models = job_models(config_path)
                if not models & busy:
                    pending.remove(config_path)
                    command = [sys.executable, __file__, str(args.config), *source]
                    command += ["--resume", str(output), "--only-job", config_path.stem]
                    log = (log_dir / f"{config_path.stem}.log").open("a", encoding="utf-8")
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env
                    )
                    running[process] = (config_path.stem, models, log)
                    print(f"Started job: {config_path.stem} (log: {log.name})", flush=True)
                    break
        time.sleep(5)
    passed = {p["job"] for p in summary().get("phases", []) if p["status"] == "passed"}
    return 0 if all(job[2].stem in passed for job in jobs) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runner-configs", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resume", type=Path)
    # Runner-config runs execute each job in its own subprocess (--only-job) so
    # that the 1M-row slice of a finished job is released (Rule 9).
    parser.add_argument("--only-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan:
        if args.runner_configs:
            parser.error("Use --plan or --runner-configs, not both")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        args.runner_configs = [Path(job["config"]) for job in plan["jobs"]]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.runner_configs and not 1 <= config["sample_rows"] <= 1000:
        parser.error("Smoke sample_rows must be between 1 and 1000")
    if config["max_tokens"] is not None and config["max_tokens"] <= 0:
        parser.error("Token limit must be positive or null")
    if not args.runner_configs and (
        min(config["positive_entries"], config["negative_entries"]) <= 0
    ):
        parser.error("Smoke entry counts must be positive")
    cost_limit = config.get("max_cost_usd")
    if cost_limit is not None and cost_limit <= 0:
        parser.error("Cost limit must be positive or null")
    output = args.resume or Path(config["results_dir"]) / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    scope = (
        "Full E1/E2/E3 runner configs"
        if args.runner_configs
        else "Smoke only, not E1/E2/E3 reproduction"
    )
    if args.resume:
        previous_config = json.loads((output / "source_config.json").read_text())
        if previous_config != config:
            parser.error("Telemetry config changed; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "source_config.json", config)
    print(f"Execution artifacts: {output}", flush=True)
    jobs = (
        prepare_runner_jobs(args.runner_configs, output)
        if args.runner_configs
        else prepare(config, output)
    )
    if args.dry_run:
        print(f"Prepared {len(jobs)} runner configs; no CSV reads or API calls")
        return 0
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is absent; no paid calls made", file=sys.stderr)
        return 2
    os.environ.setdefault(
        "NORMA_PARALLEL_REQUESTS", str(config.get("parallel_requests", 1))
    )
    os.environ.setdefault(
        "NORMA_JSON_OBJECT_MODELS", ",".join(config.get("json_object_models") or [])
    )
    if args.only_job:
        jobs = [job for job in jobs if job[2].stem == args.only_job]
    elif args.runner_configs:
        return run_isolated(args, output, jobs, config)
    with urllib.request.urlopen(
        "https://openrouter.ai/api/v1/models", timeout=30
    ) as response:
        catalog = {model["id"]: model for model in json.load(response)["data"]}
    previous = (
        json.loads((output / "summary.json").read_text())
        if args.resume and (output / "summary.json").exists()
        else {}
    )
    if previous and previous["scope"] != scope:
        parser.error("Cannot resume a run with a different experimental scope")
    state = {
        "budget_spent_usd": previous.get("budget_spent_usd", 0.0),
        "stop_reason": None,
        "job": None,
    }
    budget_at_start = state["budget_spent_usd"]
    telemetry_lock = threading.Lock()  # requests may run in parallel threads
    phases = previous.get("phases", [])
    completed = {phase["job"] for phase in phases if phase["status"] == "passed"}
    original_init = OpenRouterClient.__init__

    def bounded_init(client, *init_args, **init_kwargs):
        original_init(client, *init_args, **init_kwargs)
        # Cap at the model's own completion limit (null config = model limit).
        model_limit = (catalog[client.model_name].get("top_provider") or {}).get(
            "max_completion_tokens"
        )
        limits = [v for v in (config["max_tokens"], model_limit) if v]
        client.max_tokens = min(limits) if limits else None
        reserve_tokens = client.max_tokens or 32768
        client.client = client.client.with_options(
            timeout=config.get("request_timeout_s", 600), max_retries=0
        )
        original_create = client.client.chat.completions.create

        def recorded_create(**request):
            if state["stop_reason"]:
                raise RuntimeError(state["stop_reason"])
            pricing = catalog[client.model_name]["pricing"]
            input_rate, output_rate = float(pricing["prompt"]), float(
                pricing["completion"]
            )
            prompt_bytes = sum(
                len(message["content"].encode("utf-8"))
                for message in request["messages"]
            )
            reserve = (prompt_bytes + 2048) * input_rate + reserve_tokens * output_rate
            if exceeds_cost_limit(cost_limit, state["budget_spent_usd"], reserve):
                state["stop_reason"] = (
                    "Smoke cost reserve would exceed configured budget"
                )
                raise RuntimeError(state["stop_reason"])
            started = time.perf_counter()
            record = {
                "job": state["job"],
                "model": client.model_name,
                "prompt_bytes": prompt_bytes,
                "reserve_usd": reserve,
            }
            try:
                response = original_create(**request)
                usage = response.usage.model_dump(mode="json") if response.usage else {}
                cost = usage.get("cost")
                observed = cost is not None
                if cost is None:
                    cost = (
                        (
                            usage["prompt_tokens"] * input_rate
                            + usage["completion_tokens"] * output_rate
                        )
                        if usage
                        else reserve
                    )
                with telemetry_lock:
                    state["budget_spent_usd"] += max(0.0, float(cost))
                record.update(
                    status="received",
                    usage=usage,
                    cost_usd=cost,
                    cost_reported=observed,
                    content=response.choices[0].message.content,
                    finish_reason=response.choices[0].finish_reason,
                )
                return response
            except Exception as error:
                with telemetry_lock:
                    state["budget_spent_usd"] += reserve
                status_code = getattr(error, "status_code", None)
                if status_code in (401, 402, 403):
                    state["stop_reason"] = (
                        f"Provider access/credit blocker ({status_code})"
                    )
                record.update(
                    status="error",
                    error_type=type(error).__name__,
                    status_code=status_code,
                    error=str(error).replace(
                        os.environ["OPENROUTER_API_KEY"], "[REDACTED]"
                    ),
                    uncertain_cost_reserve_usd=reserve,
                )
                raise
            finally:
                record["latency_s"] = time.perf_counter() - started
                with telemetry_lock, file_lock(output / "attempts.lock"):
                    with (output / "attempts.jsonl").open(
                        "a", encoding="utf-8"
                    ) as destination:
                        destination.write(json.dumps(record) + "\n")

        client.client.chat.completions.create = recorded_create

    with patch.object(OpenRouterClient, "__init__", bounded_init):
        for filename, class_name, config_path in jobs:
            state["job"] = config_path.stem
            if state["job"] in completed:
                print(f"Skipping completed job: {state['job']}", flush=True)
                continue
            started = time.perf_counter()
            spec = importlib.util.spec_from_file_location(
                config_path.stem, Path("experiments") / filename
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runner = getattr(module, class_name)(str(config_path))
            runner.run()
            expected = len(runner.config["variants"]) * len(runner.config["datasets"])
            failures = [
                row
                for row in runner.all_results
                if row.get("status") == "failed" or row.get("error")
            ]
            phase = (
                {
                    "job": state["job"],
                    "results_dir": str(runner.results_dir),
                    "expected_variants": expected,
                    "recorded_variants": len(runner.all_results),
                    "failed_variants": len(failures),
                    "status": (
                        "passed"
                        if len(runner.all_results) == expected
                        and not failures
                        and not state["stop_reason"]
                        else "failed"
                    ),
                    "elapsed_s": time.perf_counter() - started,
                    "process_peak_rss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                }
            )
            # Merge into the latest summary: other jobs may run concurrently.
            with file_lock(output / "summary.lock"):
                summary_path = output / "summary.json"
                latest = (
                    json.loads(summary_path.read_text()) if summary_path.exists() else {}
                )
                phases = [
                    p for p in latest.get("phases", []) if p["job"] != state["job"]
                ] + [phase]
                spent = latest.get("budget_spent_usd", budget_at_start) + (
                    state["budget_spent_usd"] - budget_at_start
                )
                budget_at_start = state["budget_spent_usd"]
                write_json(
                    summary_path,
                    {
                        "scope": scope,
                        "max_cost_usd": cost_limit,
                        **state,
                        "budget_spent_usd": spent,
                        "stop_reason": state["stop_reason"] or latest.get("stop_reason"),
                        "phases": phases,
                    },
                )
            if state["stop_reason"]:
                break
            del runner, module
            gc.collect()
    if args.only_job:
        return 0
    return (
        0
        if len(phases) == len(jobs)
        and all(phase["status"] == "passed" for phase in phases)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
