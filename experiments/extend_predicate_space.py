"""Extend an existing predicate space with the optional builder templates.

Keeps every existing template and column declaration and adds (2026-10-10,
after the HMDA diagnosis in results/hmda_v2/diagnostico/ANALISE_CAUSAS.md):
- IN / NOT IN $values on coded integer columns (<= --categorical distinct values);
- same-tuple comparisons inside column families (race_1 .. race_5), even sparse;
- arithmetic templates ``t.a <op> t.b <arith> $number`` for numeric pairs
  (t1 and t2, operators + - * / %, as in the HMDAZ space).

Column types come from the existing space, so the data sample only decides the
coded columns. Only the first --sample-rows rows are read (never the whole CSV).

Usage:
  .venv/bin/python experiments/extend_predicate_space.py \
      data/hmda_checks/hmda_checks_predicates.json data/hmda_checks/hmda.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predicates.space_builder import PredicateSpaceBuilder

NUMERIC = {"int", "float"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--categorical", type=int, default=20)
    parser.add_argument("--arithmetic", nargs="*", default=["+", "-", "*", "/", "%"])
    args = parser.parse_args()

    spec = json.loads(args.space.read_text())
    columns = spec["tables"][0]["columns"]
    types = {column["name"]: column["type"] for column in columns}
    df = pd.read_csv(args.csv, nrows=args.sample_rows, usecols=list(types), low_memory=False)
    for name, declared in types.items():
        if declared in NUMERIC:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        else:
            df[name] = df[name].astype(object)
    df = df[list(types)]

    builder = PredicateSpaceBuilder(
        table_alias="t",
        mirror_second_tuple=True,
        numeric_categorical_threshold=args.categorical,
        column_families=True,
        arithmetic_ops=args.arithmetic,
    )
    added = builder.build_extension_predicates(df) - set(spec["predicates"])

    coded = {
        p.split(".", 1)[1].split(" ", 1)[0]
        for p in added
        if p.endswith(" IN $values") and types.get(p.split(".", 1)[1].split(" ", 1)[0]) in NUMERIC
    }
    for column in columns:
        if column["name"] in coded:
            column["allowed_ops"] = list(dict.fromkeys(column["allowed_ops"] + ["IN", "NOT IN"]))
        if column["type"] in NUMERIC and args.arithmetic:
            column["allowed_arithmetic_ops"] = list(args.arithmetic)

    before = len(spec["predicates"])
    spec["predicates"] = sorted(set(spec["predicates"]) | added)
    args.space.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    kinds = {
        "IN on coded columns": sum(" IN $values" in p for p in added),
        "column families": sum("$" not in p for p in added),
        "arithmetic": sum("$number" in p for p in added),
    }
    print(f"{args.space}: {before} -> {len(spec['predicates'])} templates; added {kinds}")
    print(f"coded columns with IN: {sorted(coded)}")


if __name__ == "__main__":
    main()
