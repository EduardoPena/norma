"""Regenerate the predicate spaces of the paper datasets from their CSVs.

Spaces follow the paper (§5.1): schema-derived templates without arithmetic,
tuple variables t1/t2 and the table named after the paper CSV, plus two generic
rules so every ground-truth predicate is expressible:
- single-column templates also for t2 (pair DCs constraining the second tuple);
- same-tuple comparisons between numeric columns with overlapping value ranges.

Usage: .venv/bin/python experiments/build_predicate_spaces.py [dataset ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predicates.space_builder import PredicateSpaceBuilder

# dataset -> predicate file kept under its existing name (found by *_predicates.json)
DATASETS = {
    "hospital": "hospital-holo_predicates.json",
    "restaurant": "restaurant-inspections_predicates.json",
    "sisu": "sisu_predicates.json",
    "tax": "tax_predicates.json",
}


def main(names):
    builder = PredicateSpaceBuilder(
        table_alias="t", mirror_second_tuple=True, numeric_range_overlap=0.3
    )
    for name in names:
        data_dir = Path("data") / name
        spec = builder.build(
            str(data_dir / f"{name}.csv"), output_path=str(data_dir / DATASETS[name])
        )
        print(f"{name}: {len(spec['predicates'])} predicates, table {spec['tables'][0]['name']!r}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
