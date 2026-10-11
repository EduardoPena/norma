"""LaTeX (booktabs) tables for E1/E2/E3 results, in the style of paper Table 3.

Dataset-agnostic (the group label and row note come from the caller). Used by
the *_hmda.ipynb notebooks for any dataset group; reads only result DataFrames.
"""

import pandas as pd

PRF = ("P", "R", "F$_1$")


def _header(title):
    """Two-line titles ("A\\\\B") stay narrow so the numeric columns keep their width."""
    return f"\\shortstack{{{title}}}" if "\\\\" in title else title


def latex_grouped_table(df, row_col, groups, caption, label, row_order,
                        dataset_order, dataset_labels, marks=None, super_groups=None):
    """One block per dataset; rows = ``row_col``; ``groups`` = [(title, [(column, header)])].

    ``super_groups`` = [(title, n_groups)] adds a header row spanning several groups.
    Bold marks the best displayed (2-decimal) value per column within a dataset, so
    ties are all bold. ``marks`` maps (dataset, row, column) to a suffix symbol.
    """
    marks = marks or {}
    columns = [column for _, cols in groups for column, _ in cols]
    spec = "l" + "".join("@{\\hspace{0.6em}}" + "r" * len(cols) for _, cols in groups)
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        f"  \\begin{{tabular}}{{{spec}}}",
        "  \\toprule",
    ]
    if super_groups:
        cells, rules, start, index = [], [], 2, 0
        for title, n_groups in super_groups:
            width = sum(len(cols) for _, cols in groups[index : index + n_groups])
            cells.append(f"\\multicolumn{{{width}}}{{c}}{{{title}}}")
            rules.append(f"\\cmidrule(lr){{{start}-{start + width - 1}}}")
            start, index = start + width, index + n_groups
        lines += ["  & " + " & ".join(cells) + " \\\\", "  " + " ".join(rules)]
    lines.append(
        "  & " + " & ".join(
            f"\\multicolumn{{{len(cols)}}}{{c}}{{{_header(title)}}}" for title, cols in groups
        ) + " \\\\"
    )
    rules, start = [], 2
    for _, cols in groups:
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(cols) - 1}}}")
        start += len(cols)
    lines.append("  " + " ".join(rules))
    lines.append(
        "  \\textbf{Model} & " + " & ".join(h for _, cols in groups for _, h in cols) + " \\\\"
    )
    for dataset in [d for d in dataset_order if d in set(df["dataset"])]:
        block = df[df["dataset"] == dataset].set_index(row_col)
        block = block.reindex([r for r in row_order if r in block.index])
        values = block[columns].apply(pd.to_numeric, errors="coerce").round(2)
        best = values.max()
        lines += [
            "  \\midrule",
            f"  \\multicolumn{{{len(columns) + 1}}}{{l}}{{\\textit{{{dataset_labels[dataset]}}}}} \\\\",
        ]
        for row, row_values in values.iterrows():
            cells = []
            for column, value in row_values.items():
                text = "--" if pd.isna(value) else f"{value:.2f}"
                if pd.notna(value) and value > 0 and value == best[column]:
                    text = f"\\textbf{{{text}}}"
                cells.append(text + marks.get((dataset, row, column), ""))
            lines.append(f"  ~~{row} & " + " & ".join(cells) + " \\\\")
    lines += [
        "  \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def _wide(df, index_col, column_key, metrics):
    wide = df.pivot_table(index=["dataset", index_col], columns=column_key, values=metrics)
    wide = wide.reset_index()
    wide.columns = ["dataset", index_col] + [f"{m}|{k}" for m, k in wide.columns[2:]]
    return wide


def e1_table(results, prompt_order, model_order, dataset_order, dataset_labels,
             group_label="the datasets", row_note="full datasets"):
    """E1: P/R/F1 per model x prompt strategy; dagger = recall >= 0.90 (selection criterion).

    ``results`` needs columns dataset, model_label, prompt_label, precision, recall, f1_score.
    """
    metrics = ["precision", "recall", "f1_score"]
    wide = _wide(results, "model_label", "prompt_label", metrics)
    prompts = [p for p in prompt_order if f"f1_score|{p}" in wide.columns]
    wrap = {"Table Context": "Table\\\\Context", "Examples + Table Context": "Examples +\\\\Table Context",
            "No Context": "No\\\\Context"}
    groups = [(wrap.get(p, p), [(f"{m}|{p}", h) for m, h in zip(metrics, PRF)]) for p in prompts]
    marks = {
        (r.dataset, r.model_label, f"f1_score|{r.prompt_label}"): "$^\\dagger$"
        for r in results.itertuples()
        if r.recall >= 0.90
    }
    return latex_grouped_table(
        wide, "model_label", groups,
        f"Constraint classification on {group_label} (E1): precision, recall and F$_1$ per model and "
        f"prompt strategy ({row_note}). $^\\dagger$: recall $\\geq$ 0.90, "
        "the selection criterion; bold: best value per column and dataset.",
        f"tab:{group_label.lower().replace(' ', '_')}_e1_classification", model_order,
        dataset_order, dataset_labels, marks,
    )


def e2_table(results, mapper_order, model_order, dataset_order, dataset_labels,
             group_label="the datasets", row_note="full datasets"):
    """E2: constraint extraction and aggregate violation P/R/F1 per model and mapper.

    ``results`` needs columns dataset, model_label, mapper and the metric columns.
    """
    metrics = {
        "Const. Ext.": ["constraint_exact_precision", "constraint_exact_recall", "constraint_exact_f1"],
        "Vio. Det.": ["agg_viol_precision", "agg_viol_recall", "agg_viol_f1"],
    }
    wide = _wide(results, "model_label", "mapper", sum(metrics.values(), []))
    groups = [
        (title, [(f"{m}|{mapper}", h) for m, h in zip(cols, PRF)])
        for mapper in mapper_order
        for title, cols in metrics.items()
    ]
    super_groups = [(mapper, len(metrics)) for mapper in mapper_order]
    return latex_grouped_table(
        wide, "model_label", groups,
        f"Sentence-level mapping on the {group_label} ground-truth constraint sentences (E2): constraint "
        "extraction (exact predicate-set match) and aggregate violation detection, precision, "
        f"recall and F$_1$ ({row_note}). Bold: best value per column and dataset.",
        f"tab:{group_label.lower().replace(' ', '_')}_e2_mapping", model_order, dataset_order, dataset_labels,
        super_groups=super_groups,
    )


TABLE3_GROUPS = [
    ("Constr. Classification", ["clf_precision", "clf_recall", "clf_f1"]),
    ("Constr. Extraction", ["constraint_exact_precision", "constraint_exact_recall", "constraint_exact_f1"]),
    ("Viol. Discovery", ["aggregate_violation_precision", "aggregate_violation_recall", "aggregate_violation_f1"]),
]


def table3(results, method_order, dataset_order, dataset_labels,
           group_label="the datasets", row_note="full datasets"):
    """Paper Table 3 layout (Family | Method | 3 x P/R/F1), Norma rows only.

    ``results`` needs columns dataset, method and the TABLE3_GROUPS metrics.
    """
    columns = [c for _, cols in TABLE3_GROUPS for c in cols]
    n_cols = 2 + len(columns)
    # Two-line group titles keep the P/R/F1 columns evenly spaced
    group_titles = [_header(title.replace(" ", "\\\\", 1)) for title, _ in TABLE3_GROUPS]
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3.5pt}",
        "  \\begin{tabular}{ll" + "rrr" * len(TABLE3_GROUPS) + "}",
        "  \\toprule",
        "  \\textbf{Family} & \\textbf{Method} & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{title}}}" for title in group_titles)
        + " \\\\",
        "  " + " ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(TABLE3_GROUPS))),
        "  & & " + " & ".join(list(PRF) * len(TABLE3_GROUPS)) + " \\\\",
    ]
    for dataset in [d for d in dataset_order if d in set(results["dataset"])]:
        block = results[results["dataset"] == dataset].set_index("method")
        block = block.reindex([m for m in method_order if m in block.index])
        values = block[columns].apply(pd.to_numeric, errors="coerce").round(2)
        best = values.max()
        lines += ["  \\midrule", f"  \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{dataset_labels[dataset]}}}}} \\\\"]
        for i, (method, row) in enumerate(values.iterrows()):
            cells = []
            for column, value in row.items():
                text = "--" if pd.isna(value) else f"{value:.2f}"
                if pd.notna(value) and value > 0 and value == best[column]:
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            family = f"\\multirow{{{len(values)}}}{{*}}{{Norma}}" if i == 0 else ""
            lines.append(f"  {family} & {method} & " + " & ".join(cells) + " \\\\")
    lines += [
        "  \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{End-to-end results of Norma on {group_label} (C+MS: classifier + Multi-Step; "
        f"Multi-Step alone), {row_note}. Bold: best result per metric and dataset. "
        "Baselines and the Document variant were not run.}",
        f"  \\label{{tab:{group_label.lower().replace(' ', '_')}_end2end}}",
        "\\end{table}",
    ]
    return "\n".join(lines)
