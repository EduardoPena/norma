"""Compare a new round on the paper datasets with the paper's E1/E2/E3 numbers.

Paper values: notebooks/results_analysis/paper_reference_results.json (extracted
from the paper PDF figures). Writes Markdown and CSV tables to <results>/comparison/.

Usage: .venv/bin/python experiments/compare_with_paper.py [results/paper_v2]
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

DATASETS = ["hospital", "restaurant", "sisu", "tax"]
LABELS = {"hospital": "Hospital", "restaurant": "Restaurant", "sisu": "SISU", "tax": "Tax"}
PROMPTS = ["No Context", "Table Context", "Examples", "Examples + Table Context"]
PAPER_CLASSIFIER = ("Qwen3-32B", "Examples")
PAPER_MAPPER = "GPT-5"
# Paper Table 3 (C+MS / Multi-Step): classification, extraction, violation F1
PAPER_TABLE3 = {
    ("hospital", "C+MS"): (0.91, 0.97, 1.00), ("hospital", "Multi-Step"): (0.67, 0.81, 0.98),
    ("restaurant", "C+MS"): (0.86, 0.57, 1.00), ("restaurant", "Multi-Step"): (0.86, 0.57, 1.00),
    ("sisu", "C+MS"): (0.97, 0.51, 0.99), ("sisu", "Multi-Step"): (0.91, 0.48, 0.94),
    ("tax", "C+MS"): (0.91, 0.91, 0.89), ("tax", "Multi-Step"): (0.62, 0.67, 0.89),
}


def load(root, experiment):
    frames = [pd.read_csv(f) for f in glob.glob(f"{root}/{experiment}/*/results_summary.csv")]
    df = pd.concat(frames)
    df = df[df["status"] == "success"].drop_duplicates(["dataset", "variant_name"], keep="last")
    return df[df["dataset"].isin(DATASETS)]


def fmt(value):
    return "–" if pd.isna(value) else f"{value:.2f}"


def best(series):
    """'0.91 (label)' for the max of a label-indexed series."""
    series = series.dropna()
    return f"{series.max():.2f} ({series.idxmax()})" if len(series) else "–"


def main(root):
    ref = json.loads(Path("notebooks/results_analysis/paper_reference_results.json").read_text())
    out = Path(root) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- E1 ----------------
    paper_e1 = pd.Series({tuple(k.split("|")): v for k, v in ref["e1_f1"].items()})
    e1 = load(root, "e1_classifiers")
    e1["model"] = e1["variant_name"].str.split(" | ", regex=False).str[0]
    e1["prompt"] = e1["variant_name"].str.split(" | ", regex=False).str[1]
    rows = []
    for ds in DATASETS:
        p = paper_e1[ds]
        n = e1[e1["dataset"] == ds].set_index(["model", "prompt"])
        p_lab = p.copy(); p_lab.index = [f"{m} + {pr}" for m, pr in p.index]
        n_lab = n["f1_score"].copy(); n_lab.index = [f"{m} + {pr}" for m, pr in n.index]
        rows.append({
            "Dataset": LABELS[ds],
            "Artigo: F1 médio (24 configs)": fmt(p.mean()),
            "Agora: F1 médio (24 configs)": fmt(n["f1_score"].mean()),
            "Artigo: melhor F1": best(p_lab),
            "Agora: melhor F1": best(n_lab),
            "Artigo: Qwen3-32B + Examples": fmt(p.get(PAPER_CLASSIFIER)),
            "Agora: GPT-6 Luna + Table Ctx (F1 / R)": f"{fmt(n.loc[('GPT-6 Luna', 'Table Context'), 'f1_score'])} / {fmt(n.loc[('GPT-6 Luna', 'Table Context'), 'recall'])}",
            "Agora: Qwen3.8 + Ex.+Table (F1 / R)": f"{fmt(n.loc[('Qwen3.8-Max-Prime', 'Examples + Table Context'), 'f1_score'])} / {fmt(n.loc[('Qwen3.8-Max-Prime', 'Examples + Table Context'), 'recall'])}",
        })
    e1_summary = pd.DataFrame(rows)
    by_prompt = []
    for prompt in PROMPTS:
        p = paper_e1[[k for k in paper_e1.index if k[2] == prompt]]
        n = e1[e1["prompt"] == prompt]["f1_score"]
        by_prompt.append({"Prompt": prompt, "Artigo: F1 médio": fmt(p.mean()), "Agora: F1 médio": fmt(n.mean())})
    e1_prompt = pd.DataFrame(by_prompt)

    # ---------------- E2 ----------------
    paper_e2 = pd.Series({tuple(k.split("|")): v for k, v in ref["e2"].items()})
    e2 = load(root, "e2_mapping")
    e2["mapper"] = e2["implementation"].map({"llm": "SinglePrompt", "multi_step_refinement": "MultiStep"})
    e2["model"] = e2["variant_name"].str.split(" | ", regex=False).str[1]
    rows = []
    for ds in DATASETS:
        for mapper, label in (("SinglePrompt", "Single-Prompt"), ("MultiStep", "Multi-Step")):
            n = e2[(e2["dataset"] == ds) & (e2["mapper"] == mapper)].set_index("model")
            pce = pd.Series({m: v for (d, mp, m, me), v in paper_e2.items() if d == ds and mp == mapper and me == "CE"})
            pav = pd.Series({m: v for (d, mp, m, me), v in paper_e2.items() if d == ds and mp == mapper and me == "AV"})
            row = {
                "Dataset": LABELS[ds], "Mapper": label,
                "CE artigo: melhor": best(pce), "CE agora: melhor": best(n["constraint_exact_f1"]),
                "CE artigo: média": fmt(pce.mean()), "CE agora: média": fmt(n["constraint_exact_f1"].mean()),
                "AV artigo: melhor": best(pav), "AV agora: melhor": best(n["agg_viol_f1"]),
                "AV artigo: média": fmt(pav.mean()), "AV agora: média": fmt(n["agg_viol_f1"].mean()),
            }
            if mapper == "MultiStep":
                row["Padrão: artigo GPT-5 (CE/AV)"] = f"{fmt(pce.get(PAPER_MAPPER))} / {fmt(pav.get(PAPER_MAPPER))}"
                row["Padrão: agora Opus 5.5 (CE/AV)"] = (
                    f"{fmt(n.loc['Claude Opus 5.5', 'constraint_exact_f1'])} / {fmt(n.loc['Claude Opus 5.5', 'agg_viol_f1'])}"
                )
            rows.append(row)
    e2_summary = pd.DataFrame(rows)

    # ---------------- E3 ----------------
    e3 = load(root, "e3_end2end")
    e3["method"] = e3["variant_name"].str.split(" | ", regex=False).str[0]
    rows = []
    for (ds, method), (pc, pe, pv) in PAPER_TABLE3.items():
        n = e3[(e3["dataset"] == ds) & (e3["method"] == method)].iloc[0]
        rows.append({
            "Dataset": LABELS[ds], "Método": method,
            "Class. F1 artigo / agora": f"{pc:.2f} / {fmt(n['clf_f1'])}",
            "Extr. F1 artigo / agora": f"{pe:.2f} / {fmt(n['constraint_exact_f1'])}",
            "Viol. F1 artigo / agora": f"{pv:.2f} / {fmt(n['aggregate_violation_f1'])}",
        })
    e3_summary = pd.DataFrame(rows)

    sections = [  # (title, csv name, table)
        ("E1 — classificação: F1 por dataset", "e1_by_dataset", e1_summary),
        ("E1 — classificação: F1 médio por estratégia de prompt (6 modelos × 4 datasets)", "e1_by_prompt", e1_prompt),
        ("E2 — mapeamento (CE = extração exata; AV = violação agregada)", "e2_mapping", e2_summary.fillna("–")),
        ("E3 — fim a fim (Tabela 3)", "e3_end2end", e3_summary),
    ]
    md = ["# Comparação com o artigo (hospital, restaurant, sisu, tax)", "",
          "Artigo: Qwen3-32B/GPT-5/GPT-5-mini/Llama-3.3-70B/Mistral-L/Mistral-S via APIs nativas. "
          "Agora: GPT-6 Luna/Sol, DeepSeek-V4-Flash, Kimi K3, Qwen3.8-Max-Prime, Claude Opus 5.5 via "
          "OpenRouter, com o Norma atual. Valores do artigo extraídos das Figuras 6 e 7 do PDF e da Tabela 3.", ""]
    for title, name, table in sections:
        md += [f"## {title}", "", table.to_markdown(index=False), ""]
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/paper_v2")
