"""Merge a dataset glossary into the column descriptions of its v2 schema file.

Reads data/<ds>/*_glossary.json and data/<ds>/*_schema.json, appends each official
definition (and code labels) to the description of the matching column and of its
label column (<name>_name), records the glossary source in the schema, and writes
the schema back. Re-running replaces the previous glossary text (idempotent).

Usage: .venv/bin/python experiments/apply_glossary.py data/hmdaz
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.glossary import apply_glossary, load_glossary


def merge(data_dir: Path) -> int:
    glossary = load_glossary(data_dir)
    if glossary is None:
        raise SystemExit(f"No *_glossary.json in {data_dir}")
    schemas = sorted(data_dir.glob("*_schema.json"))
    if len(schemas) != 1:
        raise SystemExit(f"Expected one *_schema.json in {data_dir}, found {len(schemas)}")
    spec = json.loads(schemas[0].read_text(encoding="utf-8"))
    count = apply_glossary(spec["tables"], glossary)
    meta = glossary.get("_meta", {})
    spec["glossary"] = {"source": meta.get("source"), "retrieved": meta.get("retrieved"), "columns": count}
    schemas[0].write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{schemas[0]}: glossary merged into {count} column descriptions")
    return count


if __name__ == "__main__":
    for arg in sys.argv[1:] or [""]:
        if not arg:
            raise SystemExit(__doc__)
        merge(Path(arg))
