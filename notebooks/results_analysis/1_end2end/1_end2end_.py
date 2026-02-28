import pandas as pd
import os


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CSV_FILE   = "results_end2end.csv"
OUTPUT_DIR = "latex_output"

DATASET_LABELS = {
    "restaurant": "Restaurant",
    "tax":        "Tax",
    "sisu":       "SISU",
}

VARIANT_LABELS = {
    "DIRECT | gpt-5 | with_examples_and_table": "Direct (GPT-5)",
    "DIRECT | gpt-5 | multi_step":              "MultiStep (GPT-5)",
    "FILTERED | ms-clf | gpt-5-multi_step":     "Filtered+MultiStep (GPT-5)",
    "DOCUMENT | gpt-5 | basic_with_examples":   "Document (GPT-5)",
}

# ── Column groups ─────────────────────────────────────────────
#
#   LEFT  block — evaluated over ALL sentences
#   RIGHT block — evaluated over GT constraint sentences only

LEFT_GROUPS = [
    ("Rule Detection",            ["clf_precision",                "clf_recall",                "clf_f1"]),
    ("Viol.\\ Discovery (Aggr.)", ["aggregate_violation_precision","aggregate_violation_recall","aggregate_violation_f1"]),
]

RIGHT_GROUPS = [
    ("Constraint Extraction",     ["constraint_exact_precision",   "constraint_exact_recall",   "constraint_exact_f1"]),
    ("Viol.\\ Discovery (Pair.)", ["pairwise_violation_precision", "pairwise_violation_recall", "pairwise_violation_f1"]),
]

# Runtime column (standalone, after RIGHT block)
RUNTIME_COL = "runtime_s"

SUBHEADERS = ["P", "R", "F$_1$"]

BOLD_MAX_COLS = [
    "clf_precision", "clf_recall", "clf_f1",
    "aggregate_violation_precision", "aggregate_violation_recall", "aggregate_violation_f1",
    "constraint_exact_precision", "constraint_exact_recall", "constraint_exact_f1",
    "pairwise_violation_precision", "pairwise_violation_recall", "pairwise_violation_f1",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fmt(val, bold=False, is_runtime=False):
    """Format a numeric value; return n/a for missing/N/A values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    if isinstance(val, str) and val.strip().upper() in ("N/A", "NAN", ""):
        return "n/a"
    try:
        fval = float(val)
        if pd.isna(fval):
            return "n/a"
        s = f"{fval:.0f}s" if is_runtime else f"{fval:.2f}"
    except (ValueError, TypeError):
        return "n/a"
    return f"\\textbf{{{s}}}" if bold else s


def _all_cols(groups):
    return [col for _, cols in groups for col in cols]


# ─────────────────────────────────────────────────────────────
# Table generator
# ─────────────────────────────────────────────────────────────

def generate_table(df: pd.DataFrame) -> str:

    left_cols  = _all_cols(LEFT_GROUPS)
    right_cols = _all_cols(RIGHT_GROUPS)
    all_metric_cols = left_cols + right_cols

    n_left  = len(left_cols)
    n_right = len(right_cols)
    # columns: 1 (method) + n_left + n_right + 1 (runtime)
    n_total = 1 + n_left + n_right + 1

    # ── column spec ──────────────────────────────────────────
    # Use consistent spacing: c for metric cells, no weird gaps
    left_spec  = "".join(["ccc" for _ in LEFT_GROUPS])
    right_spec = "".join(["ccc" for _ in RIGHT_GROUPS])
    # Small gap between left and right blocks; runtime at the end
    col_spec = "l" + left_spec + "@{\\hspace{1.2em}}" + right_spec + "@{\\hspace{1.0em}}r"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # ── ROW 1: block scope labels ─────────────────────────────
    left_start  = 2
    left_end    = 1 + n_left
    right_start = left_end + 1
    right_end   = left_end + n_right

    lines.append(
        f"    & \\multicolumn{{{n_left}}}{{c}}{{\\textit{{All sentences}}}} & "
        f"\\multicolumn{{{n_right}}}{{c}}{{\\textit{{GT sentences only}}}} & \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{left_start}-{left_end}}}"
        f"\\cmidrule(lr){{{right_start}-{right_end}}}"
    )

    # ── ROW 2: group names ────────────────────────────────────
    group_headers = []
    cmidrules     = []
    col_cursor    = 2
    for group_name, cols in LEFT_GROUPS + RIGHT_GROUPS:
        span = len(cols)
        group_headers.append(f"\\multicolumn{{{span}}}{{c}}{{{group_name}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + span - 1}}}")
        col_cursor += span

    # Runtime header
    runtime_col_idx = col_cursor
    lines.append(f"    \\textbf{{Method}} & {' & '.join(group_headers)} & \\textbf{{Time}} \\\\")
    lines.append(f"    {'  '.join(cmidrules)}")

    # ── ROW 3: P / R / F1 sub-headers ────────────────────────
    subheader_cells = []
    for _, cols in LEFT_GROUPS + RIGHT_GROUPS:
        for sub in SUBHEADERS:
            subheader_cells.append(sub)
    lines.append(f"    & {' & '.join(subheader_cells)} & (s) \\\\")
    lines.append(r"    \midrule")

    # ── Data rows grouped by dataset ─────────────────────────
    datasets = df["dataset"].unique()

    for d_idx, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset].copy()
        dataset_label = DATASET_LABELS.get(dataset, dataset.capitalize())

        # Per-column max for bolding
        col_max = {}
        for col in BOLD_MAX_COLS:
            if col in ddf.columns:
                try:
                    col_max[col] = pd.to_numeric(ddf[col], errors="coerce").max()
                except Exception:
                    pass

        lines.append(
            f"    \\multicolumn{{{n_total}}}{{l}}"
            f"{{\\textit{{{dataset_label}}}}} \\\\"
        )

        for _, row in ddf.iterrows():
            variant = VARIANT_LABELS.get(row["variant_name"], row["variant_name"])
            cells = [f"~~{variant}"]

            for col in all_metric_cols:
                val = row.get(col, float("nan"))
                is_bold = (
                    col in col_max
                    and pd.notna(col_max[col])
                    and not (isinstance(val, str) and val.strip().upper() == "N/A")
                    and pd.notna(pd.to_numeric(val, errors="coerce"))
                    and abs(float(val) - col_max[col]) < 1e-9
                )
                cells.append(fmt(val, bold=is_bold))

            # Runtime cell
            runtime_val = row.get(RUNTIME_COL, float("nan"))
            cells.append(fmt(runtime_val, is_runtime=True))

            lines.append(f"    {' & '.join(cells)} \\\\")

        if d_idx < len(datasets) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    # ── Caption ───────────────────────────────────────────────
    caption = (
        "  \\caption{%\n"
        "    Results of the constraint extraction pipeline evaluated on two regulatory datasets.\n"
        "    The table is divided into two evaluation scopes.\n"
        "    \\textit{All sentences} (left): metrics computed over every sentence in the document.\n"
        "    \\textbf{Rule Detection} treats the mapper as an implicit classifier---a sentence\n"
        "    is predicted positive if the mapper produces any constraint for it.\n"
        "    \\textbf{Violation Discovery (Aggr.)} pools all predicted and all gold constraints\n"
        "    and measures row-level violation overlap at the dataset level (precision, recall, F\\textsubscript{1}).\n"
        "    \\textit{GT sentences only} (right): metrics restricted to sentences\n"
        "    that carry a ground-truth constraint.\n"
        "    \\textbf{Constraint Extraction} measures exact set-membership of predicted\n"
        "    vs.\\ gold predicate sets after normalisation.\n"
        "    \\textbf{Violation Discovery (Pair.)} measures, per GT sentence, whether\n"
        "    the predicted constraint identifies the same data violations as the gold constraint\n"
        "    (average precision, recall, F\\textsubscript{1} across sentences).\n"
        "    \\textbf{Time} reports total pipeline runtime in seconds.\n"
        "    n/a indicates the metric is not applicable for this variant.\n"
        "    Best result per metric group and dataset in \\textbf{bold}.%\n"
        "  }"
    )
    lines.append(caption)
    lines.append(r"  \label{tab:main_results}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = df[df["status"] == "success"].copy()
print(f"Loaded {len(df)} rows from {CSV_FILE}")
print(f"Datasets: {df['dataset'].unique().tolist()}")
print(f"Variants: {df['variant_name'].unique().tolist()}\n")

table = generate_table(df)

snippet_path = os.path.join(OUTPUT_DIR, "run_mapper_result.tex")
with open(snippet_path, "w") as f:
    f.write("% Include in your paper with \\input{run_mapper_result}\n")
    f.write(table)
print(f"Paper snippet: {snippet_path}")

print("\n── Table preview ───────────────────────────────────\n")
print(table)