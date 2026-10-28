#!/usr/bin/env python3
"""Prepare the input CSVs of the DC discovery baseline (round-v3 protocol).

Replaces Step 1 of baselines/1_profiling (DCDiscoveryBaselines_Step1_prepare_datasets),
which read the old ground-truth format and hard-coded paths. Same choices as the
paper: SISU and Restaurant (and the HMDA datasets) are restricted to the columns
of the gold DCs, Hospital and Tax keep all columns, and every column name carries
its type ("col int|float|str") as DCDiscovery.jar expects.

Differences required by the HMDA datasets:
- the files are read in streaming with DuckDB (never fully loaded in memory), from
  the same first 1M rows used by Norma, and a seeded random sample is taken, since
  the jar builds an evidence set over tuple pairs and does not scale to 1M rows;
- identifier columns with leading zeros or more than 9 digits are typed as str
  (the jar parses int columns as 32-bit integers).

Usage:
  .venv/bin/python experiments/baselines/prepare_dc_discovery.py --datasets hmdaz hmda_checks --sample 100000
  .venv/bin/python experiments/baselines/prepare_dc_discovery.py --datasets hospital restaurant sisu tax
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.predicates import normalize_rules  # noqa: E402

GT = {
    "hospital": "hospital-holo_text2rule_gt.json",
    "restaurant": "restaurant-inspections_text2rule_gt.json",
    "sisu": "sisu_text2rule_gt.json",
    "tax": "tax_text2rule_gt.json",
    "hmdaz": "hmdaz_text2rule_gt.json",
    "hmda_checks": "hmda_regc_checks-text2rule-gt.json",
}
CSV = {"hmda_checks": "hmda.csv"}
KEEP_ALL_COLUMNS = {"hospital", "tax"}  # as in the paper (Step 1, avoid_trim)
HMDA = {"hmdaz", "hmda_checks"}
STR_COLUMNS = {"ZipCode", "ID", "Id", "ProviderNumber", "PhoneNumber", "areacode", "zip"}  # Step 1
OUT_DIR = Path("baselines/1_profiling/trimmed_datasets_v3")  # the paper inputs in trimmed_datasets/ are kept


def gt_columns(dataset):
    """Columns referenced by the gold DCs (t1.<col> / t2.<col>)."""
    payload = json.loads((Path("data") / dataset / GT[dataset]).read_text())
    columns = set()

    def walk(node):
        if isinstance(node, str):
            columns.update(re.findall(r"\bt[12]\.([A-Za-z_][A-Za-z0-9_]*)", node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for entry in payload["mapping"]:
        if entry.get("rules"):
            walk(normalize_rules(entry["rules"]))
    return columns


def infer_type(column, values):
    """pandas-like typing (int with missing values -> float), ids forced to str."""
    if column in STR_COLUMNS:
        return "str"
    present = [v for v in values if v != ""]
    if not present:
        return "str"
    if all(re.fullmatch(r"-?\d+", v) for v in present):
        if any(len(v.lstrip("-")) > 9 or (len(v) > 1 and v.startswith("0")) for v in present):
            return "str"
        return "float" if len(present) < len(values) else "int"
    try:
        for v in present:
            float(v)
        return "float"
    except ValueError:
        return "str"


def paper_input_types(dataset, path=None):
    """Column types declared in the paper inputs (e.g. the hand-corrected SISU codes),
    or in another prepared input given as path (same types for every sample size)."""
    path = Path(path) if path else Path("baselines/1_profiling/trimmed_datasets") / f"{dataset}_trimmed.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        header = next(csv.reader(fh))
    return dict(h.rsplit(" ", 1) for h in header if " " in h)


def prepare(dataset, sample, seed, tag="", types_from=None):
    source = Path("data") / dataset / CSV.get(dataset, f"{dataset}.csv")
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    header = con.execute(f"SELECT * FROM read_csv('{source}', all_varchar=true, header=true) LIMIT 0").description
    all_columns = [d[0] for d in header]
    if dataset in KEEP_ALL_COLUMNS:
        columns = all_columns
    else:
        wanted = gt_columns(dataset)
        missing = sorted(wanted - set(all_columns))
        if missing:
            print(f"[{dataset}] gold columns missing in the CSV: {missing}")
        columns = [c for c in all_columns if c in wanted]
    select = ", ".join(f'"{c}"' for c in columns)
    base = f"SELECT {select} FROM read_csv('{source}', all_varchar=true, header=true)"
    if dataset in HMDA:
        base += " LIMIT 1000000"  # the slice used by Norma
    query = f"SELECT * FROM ({base})"
    if sample:
        query += f" USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({seed})"
    rows = [[("" if v is None else v) for v in row] for row in con.execute(query).fetchall()]
    con.close()
    paper_types = paper_input_types(dataset, types_from)
    types = [paper_types.get(c) or infer_type(c, [r[i] for r in rows]) for i, c in enumerate(columns)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dataset}{tag}_v3.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"{c} {t}" for c, t in zip(columns, types)])
        writer.writerows(rows)
    manifest = {
        "dataset": dataset, "source": str(source), "rows": len(rows),
        "sample": sample, "seed": seed if sample else None,
        "slice": "first 1M rows" if dataset in HMDA else "all rows",
        "columns": dict(zip(columns, types)),
        "types_from_paper_input": sorted(c for c in columns if c in paper_types),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{dataset}{tag}_v3.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{dataset}] {len(rows)} rows x {len(columns)} columns -> {out}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(GT))
    parser.add_argument("--sample", type=int, default=None,
                        help="seeded random sample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--types-from", default=None, help="prepared CSV whose header types are reused")
    parser.add_argument("--tag", default="", help="suffix of the output name, e.g. _50k (keeps other samples)")
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(dataset, args.sample, args.seed, args.tag, args.types_from)


if __name__ == "__main__":
    main()
