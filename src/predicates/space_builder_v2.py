"""SchemaBuilderV2: schema-only predicate specification (version 2).

The v1 builder (``space_builder.PredicateSpaceBuilder``, unchanged) writes every
predicate template of a dataset into ``*_predicates.json``. Version 2 writes only
the schema, enriched per column with:

- ``allowed_arithmetic_ops``: arithmetic operators allowed for the column type;
- ``description``: written once by an LLM from a deterministic data profile
  (or built from the profile alone with ``use_llm=False``);
- ``domain_values``: the sorted distinct values of small-domain columns;

plus a ``generation`` block with the builder options. Predicates are generated at
run time, only for the columns a mapper selected (``src/predicates/on_demand.py``),
with exactly the v1 template rules.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .space_builder import PredicateSpaceBuilder

SPEC_VERSION = 2
NUMERIC_TYPES = {"int", "float"}
DEFAULT_ARITHMETIC_OPS = ["+", "-", "*", "/", "%"]
MIN_OCCURRENCES_PER_VALUE = 5

# Builder options stored in the spec and reused by the on-demand generator.
GENERATION_DEFAULTS: Dict[str, Any] = {
    "table_alias": "t",
    "categorical_threshold": 10,
    "cross_column_overlap": 0.3,
    "min_distinct_for_cross": 3,
    "enable_same_type_comparisons": True,
    "mirror_second_tuple": False,
    "numeric_range_overlap": None,
    "numeric_categorical_threshold": None,
    "column_families": False,
    "arithmetic_ops": [],
}
BUILDER_OPTIONS = tuple(GENERATION_DEFAULTS)


def coerce_declared_types(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    """Numeric declared columns become numeric (invalid values -> NaN), others object."""
    out = df.copy()
    for name, declared in types.items():
        if name not in out.columns:
            continue
        if declared in NUMERIC_TYPES:
            out[name] = pd.to_numeric(out[name], errors="coerce")
        else:
            out[name] = out[name].astype(object)
    return out


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_pattern(values: pd.Series) -> Optional[str]:
    """A recognizable textual pattern shared by (almost) all non-null values."""
    text = values.astype(str).head(2000)
    if text.empty:
        return None
    checks = [
        ("YYYYMMDD date (8 digits)", r"(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\.0)?"),
        ("YYYY year (4 digits)", r"(19|20)\d{2}(\.0)?"),
        ("ISO date YYYY-MM-DD", r"\d{4}-\d{2}-\d{2}.*"),
        ("decimal number", r"-?\d+\.\d+"),
        ("integer", r"-?\d+(\.0)?"),
    ]
    for label, pattern in checks:
        if text.str.fullmatch(pattern).mean() >= 0.95:
            return label
    return None


class SchemaBuilderV2:
    """Builds the version-2 (schema-only) specification of a single-table dataset."""

    def __init__(self, domain_values_threshold: int = 20, **generation: Any):
        unknown = set(generation) - set(BUILDER_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown generation options: {sorted(unknown)}")
        self.generation = {**GENERATION_DEFAULTS, **generation}
        self.domain_values_threshold = domain_values_threshold
        self.builder = PredicateSpaceBuilder(**self._builder_kwargs())

    def _builder_kwargs(self) -> Dict[str, Any]:
        return {k: self.generation[k] for k in BUILDER_OPTIONS}

    # ------------------------------------------------------------------
    # Column attributes
    # ------------------------------------------------------------------
    def arithmetic_ops(self, col_type: str) -> List[str]:
        ops = self.generation["arithmetic_ops"] or []
        return list(ops) if col_type in NUMERIC_TYPES else []

    def domain_values(self, series: pd.Series) -> Optional[List[Any]]:
        """Sorted distinct values of a small domain. A column whose few non-null
        values are mostly unique (e.g. a 99.98%-null amount) is not a domain:
        each value must occur ``MIN_OCCURRENCES_PER_VALUE`` times on average."""
        non_null = series.dropna()
        values = non_null.unique()
        if len(values) == 0 or len(values) > self.domain_values_threshold:
            return None
        if len(non_null) < MIN_OCCURRENCES_PER_VALUE * len(values):
            return None
        try:
            values = sorted(values)
        except TypeError:
            values = sorted(values, key=str)
        return [_jsonable(v) for v in values]

    def profile_columns(self, df: pd.DataFrame, columns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Deterministic per-column profile used for descriptions."""
        profiles: Dict[str, Dict[str, Any]] = {}
        names = set(df.columns)
        for spec in columns:
            name = spec["name"]
            series = df[name]
            non_null = series.dropna()
            profile: Dict[str, Any] = {
                "type": spec["type"],
                "null_share": round(float(series.isna().mean()), 4) if len(series) else None,
                "distinct": int(non_null.nunique()),
                "most_frequent": [_jsonable(v) for v in non_null.value_counts().head(8).index.tolist()],
            }
            if spec["type"] in NUMERIC_TYPES and not non_null.empty and is_numeric_dtype(non_null):
                profile.update(min=_jsonable(non_null.min()), max=_jsonable(non_null.max()))
            pattern = _format_pattern(non_null)
            if pattern:
                profile["format"] = pattern
            domain = self.domain_values(series)
            if domain is not None:
                profile["domain_values"] = domain
            label_col = next((c for c in (f"{name}_name", re.sub(r"_(\d+)$", r"_name_\1", name)) if c in names and c != name), None)
            if label_col:
                pairs = df[[name, label_col]].dropna().drop_duplicates()
                if not pairs.empty and pairs[name].is_unique and len(pairs) <= self.domain_values_threshold:
                    profile["labels_in"] = label_col
                    profile["code_to_label"] = {str(_jsonable(k)): v for k, v in pairs.itertuples(index=False)}
            profiles[name] = profile
        return profiles

    @staticmethod
    def profile_description(name: str, profile: Dict[str, Any]) -> str:
        """Deterministic description from the profile (no LLM)."""
        parts = [f"{profile['type']} column '{name}'"]
        if profile.get("null_share"):
            parts.append(f"{profile['null_share']:.0%} null")
        parts.append(f"{profile['distinct']} distinct values")
        if "min" in profile:
            parts.append(f"range {profile['min']} to {profile['max']}")
        if profile.get("format"):
            parts.append(f"format {profile['format']}")
        if profile.get("labels_in"):
            parts.append(f"coded; labels in {profile['labels_in']}")
        if profile.get("domain_values") is not None:
            parts.append("values " + ", ".join(map(str, profile["domain_values"])))
        elif profile.get("most_frequent"):
            parts.append("frequent values " + ", ".join(map(str, profile["most_frequent"][:5])))
        return "; ".join(parts) + "."

    def describe_columns(
        self,
        table_name: str,
        profiles: Dict[str, Dict[str, Any]],
        client: Any = None,
        batch_size: int = 25,
    ) -> Dict[str, str]:
        """LLM descriptions (one structured call per batch, one retry for missing
        columns); without a client, the deterministic profile description."""
        if client is None:
            return {name: self.profile_description(name, p) for name, p in profiles.items()}
        from src.llm import schemas
        from src.llm.helper import PromptLoader, parse_json_response

        template = PromptLoader().load("schema_description", "describe_columns")
        names = list(profiles)
        descriptions: Dict[str, str] = {}
        for start in range(0, len(names), batch_size):
            pending = names[start : start + batch_size]
            for _ in range(2):
                prompt = template.format(
                    table_name=table_name,
                    profiles=json.dumps({n: profiles[n] for n in pending}, ensure_ascii=False, default=str),
                )
                try:
                    parsed = parse_json_response(
                        client.complete(prompt, response_schema=schemas.COLUMN_DESCRIPTIONS, schema_name="column_descriptions"),
                        expected_type=object,
                    )
                    for item in parsed.get("columns", []) if isinstance(parsed, dict) else []:
                        if item.get("name") in pending and str(item.get("description", "")).strip():
                            descriptions[item["name"]] = str(item["description"]).strip()
                except ValueError as error:
                    print(f"Warning: malformed column descriptions ({error}); retrying")
                pending = [n for n in pending if n not in descriptions]
                if not pending:
                    break
            for name in pending:  # still missing after the retry: keep the profile text
                print(f"Warning: no LLM description for {name}; using the profile description")
                descriptions[name] = self.profile_description(name, profiles[name])
        return descriptions

    # ------------------------------------------------------------------
    # Spec
    # ------------------------------------------------------------------
    def build_spec(
        self,
        df: pd.DataFrame,
        table_name: str,
        csv_file: str,
        columns: Optional[List[Dict[str, Any]]] = None,
        client: Any = None,
        description_model: Optional[str] = None,
        quantifiers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Version-2 spec. ``columns`` (name/type/allowed_ops) defaults to the v1
        ``build_schema``; pass the columns of an existing v1 file to keep its types."""
        if columns is None:
            columns = self.builder.build_schema(df)
        columns = [{k: v for k, v in c.items() if k in ("name", "type", "allowed_ops")} for c in columns]
        types = {c["name"]: c["type"] for c in columns}
        typed = coerce_declared_types(df[[c["name"] for c in columns]], types)
        profiles = self.profile_columns(typed, columns)
        descriptions = self.describe_columns(table_name, profiles, client)
        out_columns = []
        for column in columns:
            name = column["name"]
            entry = dict(column)
            entry["allowed_arithmetic_ops"] = self.arithmetic_ops(column["type"])
            entry["description"] = descriptions[name]
            domain = self.domain_values(typed[name])
            if domain is not None:
                entry["domain_values"] = domain
            out_columns.append(entry)
        t1, t2 = f"{self.generation['table_alias']}1", f"{self.generation['table_alias']}2"
        return {
            "version": SPEC_VERSION,
            "quantifiers": quantifiers or {t1: table_name, t2: table_name},
            "tables": [{"name": table_name, "columns": out_columns, "file": csv_file}],
            "generation": {
                **self.generation,
                "domain_values_threshold": self.domain_values_threshold,
                "description_model": description_model,
            },
        }

    def build(
        self,
        csv_path: str,
        sample_rows: Optional[int] = None,
        output_path: Optional[str] = None,
        columns: Optional[List[Dict[str, Any]]] = None,
        client: Any = None,
        description_model: Optional[str] = None,
        table_name: Optional[str] = None,
        quantifiers: Optional[Dict[str, str]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        df = self.builder._read_csv_sample(csv_path, sample_rows=sample_rows)
        table_name = table_name or os.path.basename(csv_path).replace(".csv", "")
        spec = self.build_spec(
            df, table_name, os.path.basename(csv_path), columns, client, description_model, quantifiers
        )
        if save:
            output_path = output_path or csv_path.replace(".csv", "_schema.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
            print(f"Saved v2 schema to {output_path}")
        return spec


def builder_kwargs(generation: Dict[str, Any]) -> Dict[str, Any]:
    """PredicateSpaceBuilder options from a spec ``generation`` block."""
    merged = {**GENERATION_DEFAULTS, **{k: v for k, v in (generation or {}).items() if k in BUILDER_OPTIONS}}
    return {k: merged[k] for k in BUILDER_OPTIONS}


def iter_columns(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for table in spec.get("tables", []):
        yield from table.get("columns", [])
