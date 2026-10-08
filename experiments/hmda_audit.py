"""Audit supplied dataset files with bounded, disk-backed DuckDB scans."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator
from src.predicates import collect_aliases, normalize_rules, parse_rules


def audit_dataset(data_dir: Path, gt_file: str, row_limit: int) -> dict:
    """Count records and GT violations without materializing CSV data in Python."""
    started = time.perf_counter()
    csv_files = sorted(data_dir.glob("*.csv"))
    predicate_files = sorted(data_dir.glob("*_predicates.json"))
    if len(csv_files) != 1 or len(predicate_files) != 1:
        raise ValueError(
            "Audit requires exactly one CSV and one predicate specification"
        )
    spec = json.loads(predicate_files[0].read_text(encoding="utf-8"))
    ground_truth = json.loads((data_dir / gt_file).read_text(encoding="utf-8"))
    if len(spec["tables"]) != 1:
        raise ValueError("Audit requires a single-table schema")
    database = DBExecutor(csv_path=str(csv_files[0]))
    try:
        database.conn.execute("SET memory_limit='1GB'")
        database.conn.execute("SET threads=1")
        database.conn.execute("SET preserve_insertion_order=true")
        with tempfile.TemporaryDirectory(prefix="norma-audit-") as spill_dir:
            database.conn.execute("SET temp_directory=?", [spill_dir])
            original_rows = database.conn.execute(
                'SELECT COUNT(*) FROM "data_table"'
            ).fetchone()[0]
            database.conn.execute(
                f'CREATE TABLE audit_rows AS SELECT * FROM "data_table" LIMIT {row_limit}'
            )
            database.table_name = "audit_rows"
            database.column_type_map = {
                column["name"]: database._duckdb_type_for_declared_type(column["type"])
                for column in spec["tables"][0]["columns"]
            }
            columns = {
                row[0]
                for row in database.conn.execute("DESCRIBE audit_rows").fetchall()
            }
            used_rows = database.conn.execute(
                "SELECT COUNT(*) FROM audit_rows"
            ).fetchone()[0]
            translator = SQLTranslator(database)
            counts = []
            violation_queries = []
            sentences = []
            for item in ground_truth["mapping"]:
                sentences.extend(item.get("sentences") or [])
                rules = normalize_rules(item.get("rules"))
                if not rules:
                    continue
                parsed = parse_rules(rules)
                if any(len(collect_aliases(rule)) > 1 for rule in parsed):
                    raise ValueError(
                        "Multi-tuple GT requires a separate integrity audit"
                    )
                for rule_index, rule in enumerate(rules):
                    translated = translator.translate(rule)
                    if not translated["success"]:
                        raise ValueError(translated["error"])
                    query = translated["sql"]
                    count = database.conn.execute(
                        f"SELECT COUNT(*) FROM ({query})"
                    ).fetchone()[0]
                    counts.append(
                        {
                            "id": item.get("id"),
                            "rule_index": rule_index,
                            "violations": count,
                        }
                    )
                    violation_queries.append(query)
            union_count = (
                database.conn.execute(
                    "SELECT COUNT(*) FROM (" + " UNION ".join(violation_queries) + ")"
                ).fetchone()[0]
                if violation_queries
                else 0
            )
            return {
                "dataset": data_dir.name,
                "csv_file": str(csv_files[0]),
                "gt_file": str(data_dir / gt_file),
                "original_rows": original_rows,
                "used_rows": used_rows,
                "row_limit": row_limit,
                "selection": "first N records; single-threaded DuckDB CSV scan",
                "columns": len(columns - {database.tid_column}),
                "sentences": len(sentences),
                "unique_sentences": len(set(sentences)),
                "constraints": len(counts),
                "predicates": len(spec["predicates"]),
                "gt_violating_tuples": union_count,
                "per_constraint": counts,
                "elapsed_seconds": time.perf_counter() - started,
                "process_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
    finally:
        database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--row-limit", type=int, default=1000000)
    parser.add_argument("--results-dir", type=Path, default=Path("results/hmda/audit"))
    arguments = parser.parse_args()
    if not 0 < arguments.row_limit <= 1000000:
        parser.error("row-limit must be between 1 and 1000000")
    output = arguments.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit_dataset(
            arguments.data_dir, arguments.gt_file, arguments.row_limit
        )
    except Exception as error:
        (output / "failure.json").write_text(
            json.dumps(
                {"error_type": type(error).__name__, "error": str(error)}, indent=2
            ),
            encoding="utf-8",
        )
        raise
    (output / "statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_constraint"},
            indent=2,
        )
    )
    print(f"Statistics saved to {output / 'statistics.json'}")


if __name__ == "__main__":
    main()
