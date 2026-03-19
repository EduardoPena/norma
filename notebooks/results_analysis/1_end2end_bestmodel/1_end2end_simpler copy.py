import pandas as pd
import os


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CSV_FILE   = "results_end2end.csv"
OUTPUT_DIR = "latex_output"

DATASET_LABELS = {
    "hospital":   "Hospital",
    "restaurant": "Restaurant",
    "sisu":       "SISU",
    "tax":        "Tax"
    
 
}

# Maps implementation type → display family label
METHOD_FAMILY_LABELS = {
    "llm":          r"\textsc{Norma}",
    "llm_document": r"\textsc{Norma}",
    "multi_step_refinement": r"\textsc{Norma}",
    "text2sql":     r"Text2SQL",
    "dc_discovery": r"Profiling",
}

# Canonical ordering of method families in the table
METHOD_FAMILY_ORDER = [r"\textsc{Norma}", r"Text2SQL", r"Profiling"]

VARIANT_LABELS = {
    # your originals — unchanged
    "DIRECT | gpt-5 | with_examples_and_table":          "Single-Prompt",
    "DIRECT | gpt-5 | multi_step":                       "MultiStep",
    "FILTERED | ms-clf | gpt-5-multi_step":              "Filtered+MultiStep",
    "DOCUMENT | gpt-5 | basic_with_examples":            "Document Level",
    "XIYANText2SQL":                                     "XiYan-SQL",
    "DC-Discovery | exact | NSymbols_norm | top20":      "Exact DCs",
    "DC-Discovery | approx_0.01 | Coverage | top50":     "Approx DCs",
    "Profiling DC":                                      "Exact DCs",
    "Profiling ADC":                                     "Approx DCs",
    # new keys from the CSV
    "FILTERED | gpt-5-mini-clf | gpt-5-llm | with_examples": "Filtered+Single-Prompt",
    "FILTERED | gpt-5-mini-clf | gpt-5-multi_step":          "Filtered+MultiStep",
}

# ── Column groups ─────────────────────────────────────────────
LEFT_GROUPS = [
    ("Rule Detection",            ["clf_precision",                "clf_recall",                "clf_f1"]),
    ("Viol.\\ Discovery (Aggr.)", ["aggregate_violation_precision","aggregate_violation_recall","aggregate_violation_f1"]),
]

RIGHT_GROUPS = [
    ("Constraint Extraction",     ["constraint_exact_precision",   "constraint_exact_recall",   "constraint_exact_f1"]),
]

RUNTIME_COL = "runtime_s"
SUBHEADERS  = ["P", "R", "F$_1$"]

BOLD_MAX_COLS = [
    "clf_precision", "clf_recall", "clf_f1",
    "aggregate_violation_precision", "aggregate_violation_recall", "aggregate_violation_f1",
    "constraint_exact_precision", "constraint_exact_recall", "constraint_exact_f1",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fmt(val, bold=False, is_runtime=False):
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


def get_family(impl):
    """Return the method family label for a given implementation string."""
    return METHOD_FAMILY_LABELS.get(impl, r"\textsc{Norma}")


# ─────────────────────────────────────────────────────────────
# Table generator
# ─────────────────────────────────────────────────────────────

def generate_table(df: pd.DataFrame) -> str:

    # Assign family to each row
    df = df.copy()
    df["_family"] = df["implementation"].map(get_family).fillna(r"\textsc{Norma}")

    left_cols       = _all_cols(LEFT_GROUPS)
    right_cols      = _all_cols(RIGHT_GROUPS)
    all_metric_cols = left_cols + right_cols

    n_left  = len(left_cols)
    n_right = len(right_cols)
    # columns: 1 (family) + 1 (method) + n_left + 1 (runtime) + n_right
    # Time sits between the two blocks, inside "All sentences"
    n_total = 2 + n_left + 1 + n_right

    # Column indices (1-based)
    left_start   = 3
    left_end     = 2 + n_left
    time_col_idx = 3 + n_left
    right_start  = 4 + n_left
    right_end    = n_total

    # ── column spec ──────────────────────────────────────────
    left_spec  = "@{\\hspace{1.8em}}".join(["ccc" for _ in LEFT_GROUPS])
    right_spec = "@{\\hspace{1.8em}}".join(["ccc" for _ in RIGHT_GROUPS])
    col_spec = (
        "l"                         # family
        "l"                         # method variant
        + left_spec
        + "@{\\hspace{1.8em}}r"     # runtime
        + "@{\\hspace{1.8em}}"
        + right_spec
    )

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # ── ROW 1: block scope labels ─────────────────────────────
    lines.append(
        f"    & & \\multicolumn{{{n_left + 1}}}{{c}}{{\\textit{{All sentences}}}} & "
        f"\\multicolumn{{{n_right}}}{{c}}{{\\textit{{GT sentences only}}}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{left_start}-{time_col_idx}}}"
        f"\\cmidrule(lr){{{right_start}-{right_end}}}"
    )

    # ── ROW 2: group names ────────────────────────────────────
    group_headers = []
    cmidrules     = []

    # Left groups
    col_cursor = 3
    for group_name, cols in LEFT_GROUPS:
        span = len(cols)
        group_headers.append(f"\\multicolumn{{{span}}}{{c}}{{{group_name}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + span - 1}}}")
        col_cursor += span

    # Time header (single column, inside "All sentences" block)
    group_headers.append("Time")
    col_cursor += 1  # skip time col

    # Right groups
    for group_name, cols in RIGHT_GROUPS:
        span = len(cols)
        group_headers.append(f"\\multicolumn{{{span}}}{{c}}{{{group_name}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + span - 1}}}")
        col_cursor += span

    lines.append(
        f"    Family & Method & "
        f"{' & '.join(group_headers)} \\\\"
    )
    lines.append(f"    {'  '.join(cmidrules)}")

    # ── ROW 3: P / R / F1 sub-headers ────────────────────────
    subheader_cells = []
    for _, cols in LEFT_GROUPS:
        for sub in SUBHEADERS:
            subheader_cells.append(sub)
    subheader_cells.append("(s)")   # time sub-header
    for _, cols in RIGHT_GROUPS:
        for sub in SUBHEADERS:
            subheader_cells.append(sub)
    lines.append(f"    & & {' & '.join(subheader_cells)} \\\\")
    lines.append(r"    \midrule")

    # ── Data rows grouped by dataset, then method family ─────
    datasets = [d for d in DATASET_LABELS if d in df["dataset"].unique()]

    for d_idx, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset].copy()
        dataset_label = DATASET_LABELS.get(dataset, dataset.capitalize())

        # Per-column max for bolding (over entire dataset, all families)
        col_max = {}
        for col in BOLD_MAX_COLS:
            if col in ddf.columns:
                try:
                    col_max[col] = pd.to_numeric(ddf[col], errors="coerce").max()
                except Exception:
                    pass

        # Dataset header spanning all columns
        lines.append(
            f"    \\multicolumn{{{n_total}}}{{l}}"
            f"{{\\textit{{{dataset_label}}}}} \\\\"
        )

        # Group rows by family, preserving order
        families_present = [f for f in METHOD_FAMILY_ORDER if f in ddf["_family"].values]

        for f_idx, family in enumerate(families_present):
            fdf = ddf[ddf["_family"] == family]
            n_rows = len(fdf)

            for r_idx, (_, row) in enumerate(fdf.iterrows()):
                variant = VARIANT_LABELS.get(row["variant_name"], row["variant_name"])

                # Family cell: \multirow on first row of the family block, empty otherwise
                if r_idx == 0:
                    family_cell = f"\\multirow{{{n_rows}}}{{*}}{{{family}}}"
                else:
                    family_cell = ""

                cells = [family_cell, f"~~{variant}"]

                # Left metrics
                for col in left_cols:
                    val = row.get(col, float("nan"))
                    is_bold = (
                        col in col_max
                        and pd.notna(col_max[col])
                        and not (isinstance(val, str) and val.strip().upper() == "N/A")
                        and pd.notna(pd.to_numeric(val, errors="coerce"))
                        and abs(float(val) - col_max[col]) < 1e-9
                    )
                    cells.append(fmt(val, bold=is_bold))

                # Runtime (between left and right blocks)
                runtime_val = row.get(RUNTIME_COL, float("nan"))
                cells.append(fmt(runtime_val, is_runtime=True))

                # Right metrics
                for col in right_cols:
                    val = row.get(col, float("nan"))
                    is_bold = (
                        col in col_max
                        and pd.notna(col_max[col])
                        and not (isinstance(val, str) and val.strip().upper() == "N/A")
                        and pd.notna(pd.to_numeric(val, errors="coerce"))
                        and abs(float(val) - col_max[col]) < 1e-9
                    )
                    cells.append(fmt(val, bold=is_bold))

                lines.append(f"    {' & '.join(cells)} \\\\")

            # Thin rule between families (but not after last family in dataset)
            if f_idx < len(families_present) - 1:
                lines.append(r"    \cmidrule(l){2-" + str(n_total) + r"}")

        # Thicker rule between datasets
        if d_idx < len(datasets) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    # # ── Caption ───────────────────────────────────────────────
    # caption = (
    #     "  \\caption{%\n"
    #     "    End-to-end results across four datasets.\n"
    #     "    Rows are grouped by method family (\\textsc{Norma}: our framework;\n"
    #     "    Text2SQL: XIYAN-Text2SQL baseline; DC-Disc.: DC-Discovery baseline)\n"
    #     "    and further subdivided by dataset.\n"
    #     "    The table is divided into two evaluation scopes.\n"
    #     "    \\textit{All sentences} (left): metrics computed over every sentence in the document.\n"
    #     "    \\textbf{Rule Detection} treats the mapper as an implicit classifier.\n"
    #     "    \\textbf{Violation Discovery (Aggr.)} pools all predicted and gold constraints\n"
    #     "    and measures row-level violation overlap at the dataset level.\n"
    #     "    \\textit{GT sentences only} (right): metrics restricted to sentences\n"
    #     "    that carry a ground-truth constraint.\n"
    #     "    \\textbf{Constraint Extraction} measures exact set-membership of predicted\n"
    #     "    vs.\\ gold predicate sets after normalisation.\n"
    #     "    \\textbf{Violation Discovery (Pair.)} reports per-sentence average P/R/F$_1$\n"
    #     "    against gold violation sets.\n"
    #     "    \\textbf{Time} is total pipeline runtime in seconds.\n"
    #     "    n/a~=~metric not applicable. Bold = best per metric and dataset.%\n"
    #     "  }"
    # )


    caption = (
    "  \\caption{%\n"
    "    End-to-end results for \\textsc{Norma} and two baselines across four datasets.\n"
    "    Bold = best per metric and dataset; n/a = metric not applicable.%\n"
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