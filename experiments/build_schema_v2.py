"""Build the version-2 (schema-only) predicate specification of a dataset.

Writes data/<ds>/<table>_schema.json next to the v1 *_predicates.json (which is
left untouched). Each column gets allowed_arithmetic_ops, an LLM-written
description (from a deterministic data profile) and domain_values for small
domains; predicates are generated at run time for the columns a mapper selects
(src/predicates/on_demand.py). Only the first --sample-rows rows are read.

Usage:
  .venv/bin/python experiments/build_schema_v2.py data/hospital --from-v1 \\
      --mirror-second-tuple --numeric-range-overlap 0.3 [--no-llm]
  .venv/bin/python experiments/build_schema_v2.py data/hmda_checks --from-v1 \\
      --mirror-second-tuple --numeric-categorical-threshold 20 --column-families \\
      --arithmetic-ops + - '*' / % --model openai/gpt-6-luna
The LLM call uses OpenRouter (OPENROUTER_API_KEY from the environment).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predicates.space_builder_v2 import SchemaBuilderV2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--model", default="openai/gpt-6-luna", help="OpenRouter slug for descriptions")
    parser.add_argument("--no-llm", action="store_true", help="descriptions from the data profile only")
    parser.add_argument("--from-v1", action="store_true",
                        help="keep table name, file, quantifiers and column types of the existing *_predicates.json")
    parser.add_argument("--domain-values-threshold", type=int, default=20)
    parser.add_argument("--categorical-threshold", type=int, default=10)
    parser.add_argument("--cross-column-overlap", type=float, default=0.3)
    parser.add_argument("--min-distinct-for-cross", type=int, default=3)
    parser.add_argument("--mirror-second-tuple", action="store_true")
    parser.add_argument("--numeric-range-overlap", type=float, default=None)
    parser.add_argument("--numeric-categorical-threshold", type=int, default=None)
    parser.add_argument("--column-families", action="store_true")
    parser.add_argument("--arithmetic-ops", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    builder = SchemaBuilderV2(
        domain_values_threshold=args.domain_values_threshold,
        categorical_threshold=args.categorical_threshold,
        cross_column_overlap=args.cross_column_overlap,
        min_distinct_for_cross=args.min_distinct_for_cross,
        mirror_second_tuple=args.mirror_second_tuple,
        numeric_range_overlap=args.numeric_range_overlap,
        numeric_categorical_threshold=args.numeric_categorical_threshold,
        column_families=args.column_families,
        arithmetic_ops=args.arithmetic_ops,
    )

    columns = quantifiers = table_name = None
    if args.from_v1:
        v1_files = sorted(args.data_dir.glob("*_predicates.json"))
        if len(v1_files) != 1:
            parser.error(f"--from-v1 needs exactly one *_predicates.json in {args.data_dir}")
        v1 = json.loads(v1_files[0].read_text())
        table = v1["tables"][0]
        columns, quantifiers, table_name = table["columns"], v1["quantifiers"], table["name"]
        csv_path = args.data_dir / table.get("file", f"{table_name}.csv")
        if not csv_path.is_file():
            csv_path = args.data_dir / f"{table_name}.csv"
    else:
        csvs = sorted(args.data_dir.glob("*.csv"))
        if len(csvs) != 1:
            parser.error(f"expected one CSV in {args.data_dir}, found {len(csvs)}")
        csv_path = csvs[0]

    client = None
    if not args.no_llm:
        from src.llm.client import get_llm_client

        client = get_llm_client(args.model, provider="openrouter")
    output = args.output or args.data_dir / f"{table_name or csv_path.stem}_schema.json"
    spec = builder.build(
        str(csv_path),
        sample_rows=args.sample_rows,
        output_path=str(output),
        columns=columns,
        client=client,
        description_model=None if args.no_llm else args.model,
        table_name=table_name,
        quantifiers=quantifiers,
    )
    columns = spec["tables"][0]["columns"]
    print(f"{output}: {len(columns)} columns, "
          f"{sum('domain_values' in c for c in columns)} with domain_values, "
          f"{sum(bool(c['allowed_arithmetic_ops']) for c in columns)} with arithmetic ops")


if __name__ == "__main__":
    main()
