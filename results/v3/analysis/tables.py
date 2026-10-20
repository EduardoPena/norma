"""Tables of the round v3 (E1, E2, E3) with the paper metrics and the new ones.

Reads only results_summary.csv files (no data, no LLM calls).
Usage: .venv/bin/python results/v3/analysis/tables.py
Writes results/v3/analysis/{e1_selection,e2_new_metrics,e2_means,e3_end2end}.{csv,md,tex}.
"""
import glob
import pandas as pd

import os
RES = os.environ.get("V3_RESULTS", "results/v3")  # e.g. results/v3_reeval
OUT = f"{RES}/analysis"
os.makedirs(OUT, exist_ok=True)
DS = ["hospital", "restaurant", "sisu", "tax", "hmdaz", "hmda_checks"]
LABEL = {"hospital": "Hospital", "restaurant": "Restaurant", "sisu": "SISU", "tax": "Tax",
         "hmdaz": "HMDAZ", "hmda_checks": "HMDA Checks"}
MODELS = ["GPT-6 Luna", "DeepSeek-V4-Flash", "Mistral Large 4", "Qwen3.8-2.4T-A95B", "GPT-6 Sol", "Claude Sonnet 5.5"]


def load(pattern):
    return pd.concat(pd.read_csv(f) for f in glob.glob(pattern))


def save(df, name, tex=True, **kw):
    df.to_csv(f"{OUT}/{name}.csv")
    open(f"{OUT}/{name}.md", "w").write(df.to_markdown(**kw) + "\n")
    if tex:
        open(f"{OUT}/{name}.tex", "w").write(df.to_latex(float_format="%.2f", na_rep="--"))


# E1: selection criterion of the paper
e1 = load(f"{RES}/e1_classifiers/*/results_summary.csv")
e1[["llm", "prompt"]] = e1.variant_name.str.split(" | ", n=1, expand=True, regex=False)
sel = e1.groupby(["llm", "prompt"]).agg(min_recall=("recall", "min"), mean_f1=("f1_score", "mean"),
                                        mean_precision=("precision", "mean")).round(3)
sel["qualifies"] = sel.min_recall >= 0.90
save(sel.sort_values(["qualifies", "mean_f1"], ascending=False), "e1_selection", tex=False)

# E2: per dataset, new metrics
e2 = load(f"{RES}/e2_mapping/*/results_summary.csv")
e2[["mapper", "llm"]] = e2.variant_name.str.split(" | ", n=1, expand=True, regex=False)
e2["mapper"] = e2.mapper.replace({"Single Prompt": "Single-Prompt"})
cols = {"constraint_exact_f1": "CE", "constraint_equiv_f1": "CE_eq", "agg_viol_f1": "AV",
        "pairwise_viol_f1_mean": "RuleF1", "violation_identical_rate": "ExecEq",
        "num_pred_execution_errors": "SQLerr", "model_output_failures": "OutFail"}
e2r = e2.rename(columns=cols)
e2r["dataset"] = pd.Categorical(e2r.dataset, DS)
e2r["llm"] = pd.Categorical(e2r.llm, MODELS)
detail = e2r.set_index(["dataset", "mapper", "llm"])[list(cols.values())].sort_index()
save(detail, "e2_new_metrics")
means = e2r.groupby(["mapper", "llm"], observed=True)[["CE", "CE_eq", "AV", "RuleF1", "ExecEq"]].mean().round(3)
means["SQLerr"] = e2r.groupby(["mapper", "llm"], observed=True).SQLerr.sum()
means["OutFail"] = e2r.groupby(["mapper", "llm"], observed=True).OutFail.sum()
save(means, "e2_means")
# per-dataset characteristics of the gold violations (concentration)
conc = e2r.groupby("dataset", observed=True).agg(gold_violations=("total_gold_violations", "max"),
                                                 max_rule_share=("max_gold_rule_violation_share", "max"),
                                                 rules_without_violations=("gold_rules_without_violations", "max"),
                                                 gt_sentences=("num_sentences_gt", "max"))
save(conc, "gold_concentration")

# E3
files = glob.glob(f"{RES}/e3_end2end/*/results_summary.csv")
if files:
    e3 = load(f"{RES}/e3_end2end/*/results_summary.csv")
    print(e3.columns.tolist())
    e3.to_csv(f"{OUT}/e3_raw.csv", index=False)
print(open(f"{OUT}/e2_means.md").read())
print(open(f"{OUT}/gold_concentration.md").read())

# LaTeX table for the paper: mean over six datasets per mapper and model
order = [(m, l) for m in ["Single-Prompt", "Multi-Step"] for l in MODELS]
lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"\textbf{Mapper} & \textbf{Model} & CE & CE$_{\equiv}$ & AV & Rule-F$_1$ & Exec. & Fail. \\", r"\midrule"]
for i, (m, l) in enumerate(order):
    if i == len(MODELS):
        lines.append(r"\midrule")
    r = means.loc[(m, l)]
    first = (r"\multirow{6}{*}{" + m + "}") if l == MODELS[0] else ""
    best = {c: means.loc[[o for o in order]].loc[:, c].max() for c in ["CE", "CE_eq", "AV", "RuleF1", "ExecEq"]}
    cell = lambda c: (r"\textbf{%.2f}" % r[c]) if abs(r[c] - best[c]) < 1e-9 else "%.2f" % r[c]
    lines.append(f"{first} & {l} & {cell('CE')} & {cell('CE_eq')} & {cell('AV')} & {cell('RuleF1')} & {cell('ExecEq')} & {int(r['OutFail'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/v3_mapping_metrics_table.tex", "w").write("\n".join(lines) + "\n")
