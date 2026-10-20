"""LaTeX tables of the AV sensitivity (reads av_sensitivity_*.csv; no data, no LLM)."""
import glob
import pandas as pd

OUT = "results/v3/analysis"
MODELS = ["GPT-6 Luna", "DeepSeek-V4-Flash", "Mistral Large 4", "Qwen3.8-2.4T-A95B", "GPT-6 Sol", "Claude Sonnet 5.5"]
files = [f for f in glob.glob(f"{OUT}/av_sensitivity*.csv")]
df = pd.concat(pd.read_csv(f) for f in files).drop_duplicates(["dataset", "gt", "mapper", "llm"], keep="last")
df = df[df.llm.isin(MODELS)]
cols = [("restaurant", "Restaurant"), ("tax", "Tax"), ("hospital", "Hospital"), ("hmda_checks", "HMDA Checks")]
lines = [r"\begin{tabular}{ll" + "cc" * len(cols) + "}", r"\toprule",
         " & & " + " & ".join(r"\multicolumn{2}{c}{%s}" % l for _, l in cols) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i) for i in range(len(cols))),
         "Mapper & Model & " + " & ".join(["full & w/o"] * len(cols)) + r" \\", r"\midrule"]
for mapper in ["Single Prompt", "Multi-Step"]:
    for k, llm in enumerate(MODELS):
        cells = []
        for ds, _ in cols:
            for gt in ["full", "without_dominant"]:
                v = df[(df.dataset == ds) & (df["gt"] == gt) & (df.mapper == mapper) & (df.llm == llm)].AV
                cells.append("%.2f" % v.iloc[0] if len(v) else "--")
        name = (r"\multirow{6}{*}{%s}" % mapper.replace("Single Prompt", "Single-Prompt")) if k == 0 else ""
        lines.append(f"{name} & {llm} & " + " & ".join(cells) + r" \\")
    if mapper == "Single Prompt":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
tab = "\n".join(lines) + "\n"
open(f"{OUT}/v3_av_sensitivity_table.tex", "w").write(tab)
pt = (r"\begin{table}[H]" "\n" r"\centering\scriptsize\setlength{\tabcolsep}{3pt}" "\n" + tab +
      r"\caption{Sensibilidade da AV do E2 à(s) regra(s) dominante(s): \emph{full} = GT completo; \emph{w/o} = sem a "
      r"regra que marca $\geq$ 75\% das tuplas violadoras (Restaurant 99\%, Tax 78\%, Hospital 96\%) ou, na HMDA Checks, "
      r"sem as edições de data S270 e Q022. SISU e HMDAZ não têm regra dominante.}" "\n" r"\end{table}" "\n")
open(f"{OUT}/av_sensitivity_table_pt.tex", "w").write(pt)
print(tab)
