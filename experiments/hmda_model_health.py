"""Check OpenRouter classification and structured mapping without loading CSVs."""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifiers.llm import LLMRuleClassifier
from src.llm import schemas
from src.llm.client import OpenRouterClient, get_llm_client
from src.llm.helper import PromptLoader
from src.mappers.llm import LLMMapper
from src.predicates import parse_rules, predicate_strings


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


MULTISTEP_SCHEMAS = {
    "understand_schema": schemas.UNDERSTAND_SCHEMA,
    "select_predicates": schemas.SELECT_PREDICATES,
    "rule_gate": schemas.RULE_GATE,
    "construct_constraint": schemas.CONSTRUCT_CONSTRAINT,
    "refine_constraint": schemas.REFINE_CONSTRAINT,
}


def check_multistep_schemas(llm) -> dict[str, bool]:
    """Ask for each Multi-Step step output and validate it against its schema."""
    import jsonschema

    results = {}
    for name, schema in MULTISTEP_SCHEMAS.items():
        prompt = (
            "Table column: amount (number). Sentence: 'The amount must be positive.' "
            "Allowed predicate: t1.amount <= <const>. "
            f"Produce the JSON output of the '{name}' step of a denial-constraint "
            "mapper for this sentence. Rules, when present, are lists of predicate "
            "strings in violation form, e.g. [\"t1.amount <= 0\"]."
        )
        response = llm.complete(prompt, response_schema=schema, schema_name=name)
        try:
            jsonschema.validate(json.loads(response), schema)
            results[name] = True
        except (ValueError, jsonschema.ValidationError) as error:
            print(f"{name}: {error}")
            results[name] = False
    return results


def check_component(
    model: str, kind: str, max_tokens: int, output: Path
) -> dict[str, Any]:
    if kind == "classifier":
        component = LLMRuleClassifier(
            model_name=model,
            provider="openrouter",
            prompt_template=PromptLoader().load("classification", "with_examples"),
        )
    elif kind == "mapper":
        component = LLMMapper(
            model_name=model,
            provider="openrouter",
            prompt_name="with_examples",
            allowed_predicates=["t1.amount > <const>", "t1.amount <= <const>"],
            max_tokens=max_tokens,
        )
    else:  # multistep: the five Multi-Step step schemas
        component = SimpleNamespace(
            llm=get_llm_client(model, provider="openrouter", max_tokens=max_tokens)
        )
    component.llm.max_tokens = max_tokens
    component.llm.client = component.llm.client.with_options(timeout=300, max_retries=0)
    original_create = component.llm.client.chat.completions.create
    attempts = []

    def recorded_create(**kwargs):
        started = time.perf_counter()
        attempt = {"model": model, "kind": kind, "max_tokens": max_tokens}
        try:
            response = original_create(**kwargs)
            attempt.update(
                status="received",
                response_model=response.model,
                usage=(
                    response.usage.model_dump(mode="json") if response.usage else None
                ),
                finish_reason=response.choices[0].finish_reason,
                content=response.choices[0].message.content,
            )
            return response
        except Exception as error:
            attempt.update(
                status="error",
                error_type=type(error).__name__,
                status_code=getattr(error, "status_code", None),
                error=str(error).replace(
                    os.environ["OPENROUTER_API_KEY"], "[REDACTED]"
                ),
            )
            raise
        finally:
            attempt["latency_s"] = time.perf_counter() - started
            attempts.append(attempt)
            with (output / "attempts.jsonl").open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(attempt) + "\n")

    record = {"model": model, "kind": kind}
    started = time.perf_counter()
    try:
        with patch.object(
            component.llm.client.chat.completions, "create", side_effect=recorded_create
        ):
            if kind == "classifier":
                prediction = component._classify_sentences(
                    [
                        "The amount must be positive.",
                        "This document was published yesterday.",
                    ],
                    {},
                )
                record["prediction"] = prediction
                record["status"] = (
                    "passed"
                    if prediction == [True, False] and not component.output_failures
                    else "semantic_or_format_mismatch"
                )
            elif kind == "multistep":
                record["steps"] = check_multistep_schemas(component.llm)
                failed = [step for step, ok in record["steps"].items() if not ok]
                record["status"] = "passed" if not failed else "failed"
                if failed:
                    record["error"] = f"schema mismatch in steps: {failed}"
            else:
                prediction = component._map_sentences(
                    ["The amount must be positive."], {}
                )
                if not prediction[0]:
                    raise ValueError(
                        "Mapper returned no rule for a synthetic constraint; "
                        f"output failures: {component.output_failures}"
                    )
                parse_rules(prediction[0])
                record.update(
                    status="passed", predicates=predicate_strings(prediction[0])
                )
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error).replace(os.environ["OPENROUTER_API_KEY"], "[REDACTED]"),
        )
    record.update(
        latency_s=time.perf_counter() - started,
        attempts=len(attempts),
        blocked=any(
            attempt.get("status_code") in (401, 402, 403) for attempt in attempts
        ),
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-file", type=Path, default=Path("experiments/configs/hmda_models.yaml")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("results/hmda"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/hmda/model_health")
    )
    args = parser.parse_args()
    paper_models = yaml.safe_load(args.models_file.read_text(encoding="utf-8"))[
        "models"
    ]
    args.models = [entry["slug"] for entry in paper_models]
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    output = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Health artifacts: {output}", flush=True)
    with urllib.request.urlopen(
        "https://openrouter.ai/api/v1/models", timeout=30
    ) as response:
        catalog = {model["id"]: model for model in json.load(response)["data"]}
    manifest = {
        "api_key_present": bool(os.getenv("OPENROUTER_API_KEY")),
        "max_tokens": args.max_tokens,
        "sdk_retries": 0,
        "structured_output": "strict json_schema (src/llm/schemas.py) + provider.require_parameters",
        "transient_error_retries": OpenRouterClient.max_transient_retries,
        "empty_content_retries": 2,
        "scope": "Synthetic health checks only; not HMDA quality evaluation",
        "models": [
            {
                "id": model,
                "available": model in catalog,
                "context_length": catalog.get(model, {}).get("context_length"),
                "pricing": catalog.get(model, {}).get("pricing"),
                "top_provider": catalog.get(model, {}).get("top_provider"),
            }
            for model in args.models
        ],
    }
    write_json(output / "manifest.json", manifest)
    if not manifest["api_key_present"]:
        print("OPENROUTER_API_KEY is absent; no paid calls made", file=sys.stderr)
        return 2
    records = []
    for model in args.models:
        if model not in catalog:
            records.append({"model": model, "status": "unavailable"})
            write_json(output / "results.json", records)
            continue
        for kind in ("classifier", "mapper", "multistep"):
            record = check_component(model, kind, args.max_tokens, output)
            records.append(record)
            write_json(output / "results.json", records)
            print(json.dumps(record), flush=True)
            if record["blocked"]:
                print(
                    "Authentication, credit or access blocker; stopping paid calls",
                    file=sys.stderr,
                )
                return 2
    write_report(args.report_dir, output, manifest, paper_models, records)
    return 0 if all(record["status"] == "passed" for record in records) else 1


def write_report(report_dir, output, manifest, paper_models, records) -> None:
    """Summarize one row per paper model in llm_health_check.{json,md}."""
    rows = []
    for entry in paper_models:
        checks = {r.get("kind", "catalog"): r for r in records if r["model"] == entry["slug"]}
        info = next(m for m in manifest["models"] if m["id"] == entry["slug"])
        rows.append(
            {
                "paper_model": entry["paper_name"],
                "slug": entry["slug"],
                "in_catalog": info["available"],
                "classifier_check": checks.get("classifier", {}).get("status", "not_run"),
                "mapper_check": checks.get("mapper", {}).get("status", "not_run"),
                "multistep_check": checks.get("multistep", {}).get("status", "not_run"),
                "multistep_steps": checks.get("multistep", {}).get("steps"),
                "classifier_latency_s": checks.get("classifier", {}).get("latency_s"),
                "mapper_latency_s": checks.get("mapper", {}).get("latency_s"),
                "multistep_latency_s": checks.get("multistep", {}).get("latency_s"),
                "pricing_usd_per_token": info["pricing"],
                "context_length": info["context_length"],
                "substitution_note": " ".join(entry["note"].split()),
            }
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        report_dir / "llm_health_check.json",
        {"artifacts": str(output), "manifest": manifest, "models": rows},
    )
    lines = [
        "# OpenRouter LLM health check",
        "",
        f"Raw attempts: `{output}/attempts.jsonl`. Checks use the experiment code "
        "path with strict structured output (LLMRuleClassifier `with_examples` -> CLASSIFICATION schema; LLMMapper "
        "`with_examples` -> SENTENCE_RULES schema, rules validated by `normalize_rules`; "
        "Multi-Step: each of the 5 step schemas, response validated with jsonschema).",
        "",
        "| Paper model | OpenRouter slug | In catalog | Classifier | Mapper (JSON) "
        "| Multi-Step (5 schemas) | Latency clf / map / ms (s) | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        latencies = " / ".join(
            f"{row[k]:.1f}" if row[k] is not None else "-"
            for k in ("classifier_latency_s", "mapper_latency_s", "multistep_latency_s")
        )
        lines.append(
            f"| {row['paper_model']} | `{row['slug']}` | {row['in_catalog']} "
            f"| {row['classifier_check']} | {row['mapper_check']} "
            f"| {row['multistep_check']} | {latencies} "
            f"| {row['substitution_note']} |"
        )
    (report_dir / "llm_health_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
