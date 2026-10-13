"""Domain glossary: official field definitions merged into column descriptions.

``data/<ds>/*_glossary.json`` = {"_meta": {...source...}, "columns": {name: {
"definition": str, "codes": {code: label}?}}}. It is domain knowledge external to
the data (e.g. a public data dictionary). ``experiments/apply_glossary.py`` (or
``build_schema_v2.py --glossary``) merges it into the ``description`` of each
column and of its label column (``<name>_name``) in the v2 schema file, so the
whole pipeline uses a single description. Merging is idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_glossary(data_dir: str | Path) -> Optional[Dict[str, Any]]:
    matches = sorted(Path(data_dir).glob("*_glossary.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one *_glossary.json in {data_dir}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def glossary_text(entry: Dict[str, Any], label_of: Optional[str] = None) -> str:
    text = f"Official definition: {entry['definition'].rstrip('.')}."
    if entry.get("codes"):
        codes = "; ".join(f"{code} = {label}" for code, label in entry["codes"].items())
        text += f" Codes: {codes}."
    if label_of:
        text = f"Text label of {label_of}. " + text
    return text


def apply_glossary(tables: List[Dict[str, Any]], glossary: Dict[str, Any]) -> int:
    """Append glossary text to matching columns in place; returns columns updated."""
    entries = glossary.get("columns", {})
    updated = 0
    for table in tables:
        for column in table.get("columns", []):
            name = column["name"]
            base = name[: -len("_name")] if name.endswith("_name") else None
            if name in entries:
                extra = glossary_text(entries[name])
            elif base in entries:
                extra = glossary_text(entries[base], label_of=base)
            else:
                continue
            description = str(column.get("description") or "").strip()
            previous = column.get("glossary_text")
            if previous and description.endswith(previous):  # idempotent re-application
                description = description[: -len(previous)].rstrip()
            column["description"] = f"{description} {extra}".strip()
            column["glossary_text"] = extra
            updated += 1
    return updated
