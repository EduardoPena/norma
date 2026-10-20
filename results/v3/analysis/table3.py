"""Table 3 (E3, end to end) of round v3 in the paper format, without Document.

Norma rows come from results/v3/e3_end2end/*/results_summary.csv. Baseline rows
(DC Discovery, XiYan-SQL) for the four original datasets are copied from Table 3
of the submitted paper (same data and gold DCs); the baselines were not run on
the HMDA datasets in this round, so their cells are "--".
Usage: .venv/bin/python results/v3/analysis/table3.py
Writes results/v3/analysis/v3_table3_end2end.{tex,csv} and e3_new_metrics.md.
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
PAPER = {  # P R F1 for classification, extraction, violation (None = n/a)
    "hospital": {"DC Discovery": [None] * 3 + [0, 0, 0, .83, 1.00, .91],
                 "XiYan-SQL": [.50, 1.00, .67, .60, 1.00, .75, .83, 1.00, .91]},
    "restaurant": {"DC Discovery": [None] * 3 + [0, 0, 0, .77, .87, .82],
                   "XiYan-SQL": [.86, .75, .80, .17, .12, .14, .45, .64, .52]},
    "sisu": {"DC Discovery": [None] * 3 + [0, 0, 0, 0, .99, .01],
             "XiYan-SQL": [1.00, .65, .79, .45, .29, .36, .00, 1.00, .01]},
    "tax": {"DC Discovery": [None] * 3 + [0, 0, 0, 0, .94, .01],
            "XiYan-SQL": [.40, 1.00, .57, .29, .67, .40, .01, 1.00, .02]},
}
COLS = ["clf_precision", "clf_recall", "clf_f1", "constraint_exact_precision", "constraint_exact_recall",
        "constraint_exact_f1", "aggregate_violation_precision", "aggregate_violation_recall", "aggregate_violation_f1"]

files = glob.glob(f"{RES}/e3_end2end/*/results_summary.csv")
e3 = pd.concat(pd.read_csv(f) for f in files)
e3["method"] = e3.variant_name.str.split(" | ", n=1, regex=False).str[0]
e3.to_csv(f"{OUT}/e3_raw.csv", index=False)

rows = []
for ds in DS:
    sub = e3[e3.dataset == ds]
    for method in ["C+MS", "Multi-Step"]:
        r = sub[sub.method == method]
        rows.append((ds, "Norma", method, [float(r[c].iloc[0]) for c in COLS] if len(r) else ["missing"] * 9))
    for base in ["DC Discovery", "XiYan-SQL"]:
        rows.append((ds, "Profiling" if base == "DC Discovery" else "Text2SQL", base, PAPER.get(ds, {}).get(base, ["--"] * 9)))
pd.DataFrame([(d, f, m, *v) for d, f, m, v in rows], columns=["dataset", "family", "method", *COLS]).to_csv(f"{OUT}/v3_table3_end2end.csv", index=False)


def fmt(v, bold):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return (r"\textbf{%.2f}" % v) if bold else "%.2f" % v


lines = [r"\begin{tabular}{llccccccccc}", r"\toprule",
         r"\textbf{Family} & \textbf{Method} & \multicolumn{3}{c}{Constr. Classification} & \multicolumn{3}{c}{Constr. Extraction} & \multicolumn{3}{c}{Viol. Discovery} \\",
         r"\cmidrule(lr){3-5}  \cmidrule(lr){6-8}  \cmidrule(lr){9-11}",
         r" &  & P & R & F$_1$ & P & R & F$_1$ & P & R & F$_1$ \\", r"\midrule"]
for i, ds in enumerate(DS):
    block = [r for r in rows if r[0] == ds]
    best = {}
    for k in (2, 5, 8):  # bold the best F1 per metric and dataset
        vals = [r[3][k] for r in block if isinstance(r[3][k], float) or isinstance(r[3][k], int)]
        best[k] = max(vals) if vals else None
    if i:
        lines.append(r"\midrule")
    lines += [r"\addlinespace[2pt]", r"\multicolumn{11}{c}{\textbf{%s}} \\" % LABEL[ds], r"\addlinespace[1pt]"]
    for j, (_, fam, method, vals) in enumerate(block):
        cells = [fmt(v, k in best and best[k] is not None and isinstance(v, (int, float)) and abs(v - best[k]) < 1e-9)
                 for k, v in enumerate(vals)]
        name = {"C+MS": "C+MS", "Multi-Step": "Multi-Step"}.get(method, method)
        if fam == "Norma":
            first = r"\multirow{2}{*}{\system}" if method == "C+MS" else ""
        else:
            first = r"\multirow{1}{*}{%s}" % fam
            lines.append(r"\cmidrule(l){2-11}")
        lines.append(f"{first} & ~~{name} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/v3_table3_end2end.tex", "w").write("\n".join(lines) + "\n")

# complementary E3 metrics (Norma only)
extra = e3.set_index(["dataset", "method"])[["constraint_equiv_f1", "pairwise_violation_f1", "violation_identical_rate",
                                              "num_pred_constraints", "num_gold_constraints", "num_pred_execution_errors",
                                              "model_output_failures"]]
open(f"{OUT}/e3_new_metrics.md", "w").write(extra.round(3).to_markdown() + "\n")
print(open(f"{OUT}/e3_new_metrics.md").read())
