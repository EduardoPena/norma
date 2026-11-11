"""Build the result tables of results/hmda/REPORT.md from small result files.

Reads only results_summary.csv / outputs JSON / telemetry under results/hmda and
the run log; never the HMDA data CSVs.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("results/hmda")
DATASETS = {"hmdaz": "HMDAZ", "hmda_checks": "HMDA_CHECKS"}
MODELS = yaml.safe_load(Path("experiments/configs/hmda_models.yaml").read_text())[
    "models"
]
PAPER_NAME = {model["slug"]: model["paper_name"] for model in MODELS}
PROMPTS = {
    "zero_shot": "No Context",
    "zero_shot_with_table": "Table Context",
    "with_examples": "Examples",
    "with_examples_and_table": "Examples + Table Context",
}
CLASSIFIER_LABEL = re.compile(r"\s*\d+\.\s*(YES|NO)", re.IGNORECASE)


def load_latest(experiment_dir):
    frames = []
    for path in sorted((ROOT / experiment_dir).glob("*/results_summary.csv")):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True).sort_values("run_dir")
    # Each variant is independent: keep its latest successful attempt, else its latest failure.
    results["attempts"] = results.groupby(["dataset", "variant_name"])["run_dir"].transform("count")
    results["_ok"] = results["status"] == "success"
    results = results.sort_values(["_ok", "run_dir"])
    return results.drop_duplicates(["dataset", "variant_name"], keep="last").drop(columns="_ok")


def fmt(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "–"


def markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.values]
    return "\n".join([header, rule, *rows])


def telemetry(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text())
    jobs = defaultdict(
        lambda: {
            "calls": 0,
            "errors": Counter(),
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "length_truncated": 0,
            "empty_content": 0,
            "contents": [],
        }
    )
    for line in (run_dir / "attempts.jsonl").open():
        attempt = json.loads(line)
        job = jobs[attempt["job"]]
        job["calls"] += 1
        if attempt.get("status") == "error":
            job["errors"][str(attempt.get("status_code") or attempt["error_type"])] += 1
            job["cost_usd"] += attempt.get("uncertain_cost_reserve_usd", 0.0)
            continue
        if attempt.get("status") != "received":
            continue  # interrupted call (no response recorded)
        usage = attempt.get("usage") or {}
        job["cost_usd"] += float(attempt.get("cost_usd") or 0.0)
        job["prompt_tokens"] += usage.get("prompt_tokens", 0)
        job["completion_tokens"] += usage.get("completion_tokens", 0)
        job["length_truncated"] += attempt.get("finish_reason") == "length"
        if attempt.get("content"):
            job["contents"].append(attempt["content"])
        else:
            job["empty_content"] += 1
    return summary, jobs


def count_labels(content):
    """Labels in a classifier response: JSON ``labels`` or legacy ``<n>. YES/NO`` lines."""
    try:
        payload = json.loads(content)
        if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
            return len(payload["labels"])
    except ValueError:
        pass
    return sum(bool(CLASSIFIER_LABEL.match(l.strip())) for l in content.split("\n"))


def chunks(n, size):
    return [size] * (n // size) + ([n % size] if n % size else [])


def classifier_shortfalls(contents, chunk_sizes):
    """Labels missing from the first classifier responses (legacy code padded them as NO)."""
    if len(contents) < len(chunk_sizes):
        return None
    return sum(
        max(0, expected - count_labels(content))
        for content, expected in zip(contents, chunk_sizes)
    )


def log_markers(log_paths):
    """Per-job counts of mapper gates/errors, segmented by saved config paths."""
    counts = defaultdict(Counter)
    for log_path in log_paths:
        _count_markers(log_path, counts)
    return counts


def _count_markers(log_path, counts):
    job = None
    patterns = {
        "g1_orphan_dc": "-> Step 5 [G1]",
        "g2_too_many_violations": "-> Step 5 [G2]",
        "sql_failed": "-> Step 5: SQL execution failed",
        "parse_errors": "parse error",
        "single_prompt_invalid": "Warning: Invalid rules",
        "single_prompt_null": "Warning: Got null response",
        "transient_retries": "OpenRouter transient error",
        "empty_content_retries": "OpenRouter returned empty content",
    }
    for line in log_path.open(errors="replace"):
        if line.startswith("Config YAML saved to:"):
            config_path = Path(line.split(":", 1)[1].strip())
            # Superseded/interrupted runs were moved away: their markers are skipped.
            job = (
                yaml.safe_load(config_path.read_text())["experiment_name"]
                if config_path.exists()
                else None
            )
            continue
        if job is None:
            continue
        lowered = line.lower()
        for key, marker in patterns.items():
            if marker.lower() in lowered:
                counts[job][key] += 1


def null_predictions(run_dir):
    nulls = total = 0
    for path in Path(run_dir).glob("outputs/*.json"):
        for entry in json.loads(path.read_text())["outputs"]:
            if "pred_is_non_null" in entry or "predicted" in entry:
                total += 1
                nulls += not entry.get("pred_is_non_null", entry.get("predicted"))
    return nulls, total


def main():
    global ROOT, PAPER_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--log", type=Path, nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT)
    parser.add_argument("--models", type=Path, default=Path("experiments/configs/hmda_models.yaml"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ROOT = args.results_root
    PAPER_NAME = {m["slug"]: m["paper_name"] for m in yaml.safe_load(args.models.read_text())["models"]}
    args.out = args.out or ROOT / "report_tables.md"

    stats = json.loads((Path("results/hmda") / "dataset_stats.json").read_text())  # Phase 1 (same data slice)
    summary, jobs = telemetry(args.telemetry)
    markers = log_markers(args.log)
    sections = []

    # ---------------- Dataset statistics (paper Table 2) ----------------
    rows = [
        [DATASETS[d], f"{s['original_rows']:,}", f"{s['used_rows']:,}", s["columns"],
         s["sentences"], s["constraints"], f"{s['gt_violating_tuples']:,}",
         f"{s['predicates']:,}", len(s["constraints_with_zero_violations"])]
        for d, s in stats.items()
    ]
    sections.append("### Estatísticas dos datasets (formato da Tabela 2; recorte de 1M linhas)\n\n" + markdown(pd.DataFrame(rows, columns=[
        "Dataset", "#Rows originais", "#Rows usadas", "#Cols", "Sent.", "Const.",
        "Viol. (tuplas)", "#Predicados", "Const. com 0 viol."])))

    # ---------------- E1 ----------------
    e1 = load_latest("e1_classifiers")
    if not e1.empty:
        e1["Modelo"] = e1["model"].map(PAPER_NAME)
        e1["Prompt"] = e1["prompt_key"].map(PROMPTS)
        ok = e1[e1["status"] == "success"]
        blocks = []
        for dataset, label in DATASETS.items():
            sub = e1[e1["dataset"] == dataset].copy()
            if sub.empty:
                continue
            eligible = sub[(sub["status"] == "success") & (sub["recall"] >= 0.90)]
            best = eligible.loc[eligible["f1_score"].idxmax()] if not eligible.empty else None
            table = pd.DataFrame({
                "Modelo": sub["Modelo"], "Prompt": sub["Prompt"],
                "P": sub["precision"].map(fmt), "R": sub["recall"].map(fmt),
                "F1": sub["f1_score"].map(fmt),
                "R ≥ 0.90": sub["recall"].map(lambda r: "✓" if r >= 0.90 else ""),
                "Status": sub["status"],
            }).sort_values(["Modelo", "Prompt"])
            chosen = (f"**Selecionado (maior F1 com R ≥ 0.90):** {best['Modelo']} + {best['Prompt']} (F1={best['f1_score']:.2f}, R={best['recall']:.2f})"
                      if best is not None else "**Nenhuma configuração atinge R ≥ 0.90.**")
            blocks.append(f"#### {label}\n\n{markdown(table)}\n\n{chosen}")
        # paper criterion across datasets: recall >= 0.90 in every dataset, best mean F1
        passing = ok[ok["recall"] >= 0.90].groupby(["Modelo", "Prompt"]).agg(
            n=("dataset", "nunique"), mean_f1=("f1_score", "mean"))
        passing = passing[passing["n"] == ok["dataset"].nunique()].sort_values("mean_f1", ascending=False)
        overall = (f"**Critério do artigo nos dois datasets (R ≥ 0.90 em ambos, maior F1 médio):** "
                   + ", ".join(f"{m} + {p} (F1 médio {r.mean_f1:.2f})" for (m, p), r in passing.head(3).iterrows())
                   if not passing.empty else "**Nenhuma configuração atinge R ≥ 0.90 nos dois datasets.**")
        sections.append("### E1 — Classificadores (RQ1 / Fig. 6)\n\n" + "\n\n".join(blocks) + "\n\n" + overall)

    # ---------------- E2 ----------------
    e2 = load_latest("e2_mapping")
    if not e2.empty:
        blocks = []
        for dataset, label in DATASETS.items():
            sub = e2[e2["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub["Mapper"] = sub["implementation"].map({"llm": "Single-Prompt", "multi_step_refinement": "Multi-Step"})
            sub["Modelo"] = sub["model"].map(PAPER_NAME)
            table = pd.DataFrame({
                "Mapper": sub["Mapper"], "Modelo": sub["Modelo"],
                "CE P": sub["constraint_exact_precision"].map(fmt) if "constraint_exact_precision" in sub else "–",
                "CE R": sub.get("constraint_exact_recall", pd.Series(dtype=float)).map(fmt),
                "CE F1": sub["constraint_exact_f1"].map(fmt),
                "AV P": sub.get("agg_viol_precision", pd.Series(dtype=float)).map(fmt),
                "AV R": sub.get("agg_viol_recall", pd.Series(dtype=float)).map(fmt),
                "AV F1": sub["agg_viol_f1"].map(fmt),
                "Pairwise viol. F1": sub.get("pairwise_viol_f1_mean", pd.Series(dtype=float)).map(fmt),
                "Syn. exact": sub.get("pairwise_syn_exact_match", pd.Series(dtype=float)).map(fmt),
                "Status": sub["status"],
            }).sort_values(["Mapper", "Modelo"])
            blocks.append(f"#### {label}\n\n{markdown(table)}")
        sections.append("### E2 — Mapeamento GT-only (RQ2/RQ3 / Fig. 7)\n\nCE = constraint extraction (match exato do conjunto de predicados); AV = aggregate violation (tuplas).\n\n" + "\n\n".join(blocks))

    # ---------------- E3 ----------------
    e3 = load_latest("e3_end2end")
    if not e3.empty:
        e3["Método"] = e3["variant_name"].str.split(" | ", regex=False).str[0]
        cols = [("clf_precision", "Class. P"), ("clf_recall", "Class. R"), ("clf_f1", "Class. F1"),
                ("constraint_exact_precision", "Extr. P"), ("constraint_exact_recall", "Extr. R"), ("constraint_exact_f1", "Extr. F1"),
                ("aggregate_violation_precision", "Viol. P"), ("aggregate_violation_recall", "Viol. R"), ("aggregate_violation_f1", "Viol. F1")]
        rows = []
        for dataset, label in DATASETS.items():
            for _, r in e3[e3["dataset"] == dataset].sort_values("Método").iterrows():
                rows.append([label, r["Método"], *[fmt(r.get(c)) for c, _ in cols], r["status"]])
        sections.append("### E3 — Fim a fim (RQ4 / Tabela 3, somente Norma)\n\n" + markdown(pd.DataFrame(rows, columns=["Dataset", "Método", *[n for _, n in cols], "Status"])))

    # ---------------- Cost / time / memory ----------------
    phase_by_job = {p["job"]: p for p in summary["phases"]}
    rows = []
    totals = defaultdict(lambda: [0.0, 0.0])
    for job, info in jobs.items():
        phase = phase_by_job.get(job, {})
        experiment = job.split("_")[1].upper()
        totals[experiment][0] += info["cost_usd"]
        totals[experiment][1] += phase.get("elapsed_s", 0.0)
        rows.append([job, phase.get("status", "running/aborted"), info["calls"],
                     dict(info["errors"]) or "", info["length_truncated"], info["empty_content"],
                     f"{info['prompt_tokens']:,}", f"{info['completion_tokens']:,}",
                     f"{info['cost_usd']:.3f}", f"{phase.get('elapsed_s', 0) / 60:.1f}",
                     f"{phase.get('process_peak_rss_kib', 0) / 1024**2:.2f}"])
    cost = markdown(pd.DataFrame(rows, columns=[
        "Job", "Status", "Chamadas", "Erros (código)", "finish=length", "Conteúdo vazio",
        "Tokens in", "Tokens out", "Custo US$", "Tempo (min)", "Pico RSS (GiB)*"]))
    total_rows = [[k, f"{v[0]:.2f}", f"{v[1] / 3600:.2f}"] for k, v in sorted(totals.items())]
    total_rows.append(["Total", f"{sum(v[0] for v in totals.values()):.2f}", f"{sum(v[1] for v in totals.values()) / 3600:.2f}"])
    sections.append("### Custo, tempo e memória reais\n\n" + markdown(pd.DataFrame(total_rows, columns=["Experimento", "Custo US$", "Tempo (h)"]))
                    + f"\n\nGasto total registrado pelo harness: US$ {summary['budget_spent_usd']:.2f} (inclui reserva conservadora para chamadas com erro).\n\n"
                    + cost + "\n\n*Pico de RSS do processo do harness até o fim do job (monotônico; `ru_maxrss`).")

    # ---------------- Sanity ----------------
    rows = []
    for frame, kind in ((e2, "E2"), (e3, "E3")):
        for _, r in frame.iterrows():
            job = Path(r["run_dir"]).name.rsplit("_", 2)[0]
            config = yaml.safe_load((Path(r["run_dir"]) / "config.yaml").read_text())
            m = markers.get(config["experiment_name"], Counter())
            nulls, total = null_predictions(r["run_dir"])
            gold = r.get("total_gold_violations")
            expected = stats[r["dataset"]]["gt_violating_tuples"]
            rows.append([config["experiment_name"], r.get("num_sentences_total"), r.get("num_sentences_gt"),
                         f"{nulls}/{total}", m["g1_orphan_dc"], m["g2_too_many_violations"], m["sql_failed"],
                         m["parse_errors"] + m["single_prompt_invalid"], m["single_prompt_null"],
                         r.get("model_output_failures", "–") if pd.notna(r.get("model_output_failures")) else "–",
                         m["transient_retries"],
                         "–" if pd.isna(gold) else int(gold), expected,
                         "n/a (falhou)" if pd.isna(gold) else ("✓" if int(gold) == expected else "✗")])
    sanity = markdown(pd.DataFrame(rows, columns=[
        "Job", "Sent. total", "Sent. GT", "Pred. nulas", "DCs desc. G1 (orphan)", "G2", "SQL falhou",
        "Erros parse/validação", "Resp. nulas (SP)", "Falhas de saída (sentenças)", "Retries transitórios", "Viol. GT (run)", "Viol. GT (Fase 1)", "Confere"]))
    shortfalls = []
    for job, info in sorted(jobs.items()):
        snapshot = args.telemetry / "runner_configs" / f"{job}.yaml"
        if not snapshot.exists():
            continue
        config = yaml.safe_load(snapshot.read_text())
        n = stats[config["datasets"][0]]["sentences"]
        if job.startswith("hmda_e1_"):
            sizes = [s for v in config["variants"] for s in chunks(n, v.get("batch_size", 20))]
            shortfalls.append([job, classifier_shortfalls(info["contents"], sizes)])
        elif job.startswith("hmda_e3_") and job.endswith("C_MS"):
            # Runner 1 classifies first; LLMRuleClassifier defaults to chunk_size=20
            size = config["variants"][0]["classifier"].get("batch_size", 20)
            sizes = chunks(n, size)
            shortfalls.append([job, classifier_shortfalls(info["contents"][: len(sizes)], sizes)])
    sections.append("### Sanidade\n\n" + sanity + "\n\nRótulos ausentes nas respostas do classificador (completados como NO pelo código):\n\n"
                    + markdown(pd.DataFrame(
                        [[job, "pendente" if n is None else int(n)] for job, n in shortfalls],
                        columns=["Job", "Rótulos ausentes"])))

    args.out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
