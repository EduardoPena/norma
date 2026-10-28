#!/usr/bin/env python3
"""
Standalone XiYanSQL Evaluation Script  (DuckDB backend)
=========================================================
Identical to eval_xiyan.py but uses DuckDB instead of SQLite.

DuckDB advantages over SQLite for this workload:
- Columnar execution: self-joins and aggregations are much faster
- Native DataFrame registration (zero-copy via Arrow)
- Real query interruption via conn.interrupt() from a background thread
- Better SQL standard compliance (e.g. DISTINCT ON, window functions)

The rest of the evaluation logic — metrics, output format, prompt — is
identical to eval_xiyan.py so results are directly comparable.

Usage
-----
    python eval_xiyan_duckdb.py --data_dir data/nyc/ [--output_dir results/xiyan/]
                                [--model XGenerationLab/XiYanSQL-QwenCoder-14B-2504]
                                [--max_new_tokens 512] [--temperature 0.1] [--top_p 0.8]
                                [--sql_timeout 120]
"""

import argparse
import json
import re
import duckdb
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# DatasetLoader  (inlined — no src.* imports)
# =============================================================================

class DatasetLoader:
    """Load dataset with CSV data, sentences, and ground truth."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dataset_name = self.data_dir.name
        self.df = None
        self.sentences = []
        self.sentence2dcs_map = None
        self.predicate_space = None

    def load(self):
        annotated_file = self.data_dir / f"{self.dataset_name}_annotated-sentences.json"
        with open(annotated_file) as f:
            gt_data = json.load(f)

        self.sentences = []
        self.sentence2dcs_map = {}
        for item in gt_data["mapping"]:
            sentence = item["sentences"][0]
            self.sentences.append(sentence)
            if item["rules"] is not None:
                self.sentence2dcs_map[sentence] = item["rules"][0]

        csv_path = self.data_dir / f"{self.dataset_name}.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)

        return self.df, self.sentences, self.sentence2dcs_map, None


# =============================================================================
# DuckDB helpers
# =============================================================================

def build_schema_string(df: pd.DataFrame) -> str:
    """Generate a CREATE TABLE DDL string matching DuckDB types."""
    type_map = {
        "int64":   "BIGINT",
        "int32":   "INTEGER",
        "float64": "DOUBLE",
        "float32": "FLOAT",
        "object":  "VARCHAR",
        "bool":    "BOOLEAN",
    }
    cols = ",\n".join(
        f"    {col} {type_map.get(str(dtype), 'VARCHAR')}"
        for col, dtype in df.dtypes.items()
    )
    return f"CREATE TABLE t1 (\n{cols}\n);"


def dcs_to_sql(dcs: List[str]) -> str:
    """Convert a list of DC predicates to a SQL SELECT query (DuckDB / t1_with_tid)."""
    uses_t2 = any("t2." in p for p in dcs)
    if uses_t2:
        from_clause = "t1_with_tid t1, t1_with_tid t2"
        select_cols = "t1.tid, t2.tid"
    else:
        from_clause = "t1_with_tid t1"
        select_cols = "t1.tid"
    where_clause = " AND ".join(dcs)
    return f"SELECT {select_cols} FROM {from_clause} WHERE {where_clause}"


# Sentinel to distinguish execution failure from genuine empty result
_EXEC_FAILED = object()


def execute_for_tids(
    sql: str,
    conn: "duckdb.DuckDBPyConnection",
    timeout_s: float = 120.0,
) -> object:  # returns Set[int] | _EXEC_FAILED
    """
    Execute SQL against DuckDB and collect tid values.

    Returns:
        Set[int]     — possibly empty if SQL ran fine (0 rows is valid)
        _EXEC_FAILED — SQL raised an exception, timed out, or had no tid column

    Timeout: fires conn.interrupt() from a daemon thread after timeout_s seconds.
    DuckDB self-joins return two columns both named "tid" — we collect ALL of them
    by position.
    """
    import threading, time as _time

    timed_out = [False]

    def _interrupter():
        _time.sleep(timeout_s)
        timed_out[0] = True
        try:
            conn.interrupt()
        except Exception:
            pass

    timer = threading.Thread(target=_interrupter, daemon=True)
    timer.start()

    try:
        rel = conn.execute(sql)
        if rel.description is None:
            return set()

        col_names = [d[0] for d in rel.description]

        # DuckDB keeps duplicate "tid" names in self-join results
        tid_indices = [i for i, name in enumerate(col_names) if name == "tid"]
        if not tid_indices:
            print(f"    [tid warning] no tid column in result. Columns: {col_names}")
            return _EXEC_FAILED

        tids: Set[int] = set()
        for row in rel.fetchall():
            for idx in tid_indices:
                val = row[idx]
                if val is None:
                    continue
                try:
                    tids.add(int(val))
                except (ValueError, TypeError):
                    pass
        return tids

    except Exception as e:
        if timed_out[0]:
            print(f"    [SQL timeout >{timeout_s:.0f}s] {sql[:80]}...")
        else:
            print(f"    [SQL error] {e}\n    SQL: {sql[:120]}")
        return _EXEC_FAILED


def setup_duckdb(df: pd.DataFrame) -> Tuple["duckdb.DuckDBPyConnection", str]:
    """
    Load df into an in-memory DuckDB connection.

    Creates:
      - A registered relation '_t1_raw' backed directly by the DataFrame (zero-copy).
      - A view 't1_with_tid' that prepends a 0-based integer tid column via
        row_number() OVER ().  All queries use this view.

    Returns (connection, schema_string).
    The schema_string is the DDL that goes into the model prompt — it uses the
    logical name 't1' so the model doesn't need to know about the internal view.
    """
    import duckdb as _duckdb

    if "tid" in df.columns:
        df = df.drop(columns=["tid"])

    conn = _duckdb.connect()
    conn.register("_t1_raw", df)
    conn.execute(
        "CREATE OR REPLACE VIEW t1_with_tid AS "
        "SELECT (row_number() OVER ()) - 1 AS tid, * FROM _t1_raw"
    )

    # Build schema from the view descriptor so types are DuckDB-native
    info = conn.execute("DESCRIBE t1_with_tid").fetchall()
    cols = ",\n".join(f"    {row[0]} {row[1]}" for row in info)
    schema = f"CREATE TABLE t1 (\n{cols}\n);"

    return conn, schema


# =============================================================================
# XiYanSQL model
# =============================================================================

_PROMPT_TEMPLATE = """\
You are an expert SQL assistant working with a single table called t1.

Your task: given a sentence, write a {dialect} SQL query that returns the row \
identifiers of every tuple (or pair of tuples) that VIOLATES the constraint \
the sentence describes.

Output format:
- Single-tuple violations  → SELECT t1.tid FROM t1 WHERE <condition>
- Pair-of-tuple violations → SELECT t1.tid, t2.tid FROM t1, t1 AS t2 WHERE <condition>
- Not a checkable data constraint → output exactly: NULL

Rules:
- The ONLY table is t1. For self-joins alias it as t1 AS t2. Never reference \
any other table name.
- Always select t1.tid (and t2.tid for pairs) — these are the row identifiers.
- Prefer JOINs over correlated subqueries (NOT EXISTS / IN) for performance.
- Return ONLY the SQL query or the word NULL. No explanation, no markdown fences.

Database schema:
{db_schema}

Sentence:
{question}

SQL:"""


def load_model(model_name: str):
    print(f"[XiYan] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[XiYan] Model ready.")
    return model, tokenizer


def generate_sql(
    sentence: str,
    db_schema: str,
    model,
    tokenizer,
    dialect: str = "SQLite",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.8,
) -> Optional[str]:
    """Run the model for one sentence. Returns raw SQL string or None."""
    prompt = _PROMPT_TEMPLATE.format(
        dialect=dialect,
        db_schema=db_schema,
        question=sentence,
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return _clean_sql(raw)


def _rewrite_t2_to_self_join(sql: str) -> str:
    """
    Rewrite standalone references to table `t2` in the main FROM/JOIN clause
    into self-joins on t1.  Crucially, does NOT touch `t2` inside subqueries
    (NOT EXISTS, IN, etc.) because those are already referencing t1 via the
    outer alias and are syntactically correct as-is.

    Strategy: only rewrite t2 in the top-level FROM clause, identified by
    being outside any parenthesis depth > 0.
    """
    # Tokenise at parenthesis boundaries so we can track depth
    # Split into segments: [outside, (inside), outside, (inside), ...]
    segments = re.split(r'(\(|\))', sql)

    depth = 0
    result_parts = []
    for seg in segments:
        if seg == '(':
            depth += 1
            result_parts.append(seg)
        elif seg == ')':
            depth -= 1
            result_parts.append(seg)
        elif depth == 0:
            # Only rewrite at top level (depth == 0)
            seg = re.sub(
                r'\bJOIN\s+t2\s+AS\s+(\w+)',
                r'JOIN t1 AS \1',
                seg,
                flags=re.IGNORECASE,
            )
            seg = re.sub(
                r'\bJOIN\s+t2\b(?!\s*AS)',
                'JOIN t1 AS t2',
                seg,
                flags=re.IGNORECASE,
            )
            seg = re.sub(
                r',\s*t2\b(?!\s*AS)',
                ', t1 AS t2',
                seg,
                flags=re.IGNORECASE,
            )
            result_parts.append(seg)
        else:
            # Inside subquery — leave untouched
            result_parts.append(seg)

    return ''.join(result_parts)


def _clean_sql(raw: str) -> Optional[str]:
    """
    Parse model output into a clean SQL string or None.

    Returns None when:
    - model outputs NULL / NONE / empty  (→ not a constraint)
    - no SELECT found
    - no WHERE clause                    (→ not a constraint query)
    """
    raw = raw.strip()

    # Model signalled non-constraint
    if raw.upper() in ("NULL", "NONE", "N/A", ""):
        return None

    # Strip markdown fences
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    # Truncate at first semicolon
    semi = raw.find(";")
    if semi != -1:
        raw = raw[:semi + 1]
    raw = raw.strip()

    if not re.match(r"^\s*SELECT", raw, re.IGNORECASE):
        return None

    if not re.search(r"\bWHERE\b", raw, re.IGNORECASE):
        return None

    # Rewrite standalone t2 table references in the top-level FROM/JOIN only
    raw = _rewrite_t2_to_self_join(raw)

    return raw


# =============================================================================
# Metrics helpers
# =============================================================================

def _prf(gold: Set[int], pred: Set[int]) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


# =============================================================================
# Per-sentence result dataclass
# =============================================================================

@dataclass
class SentenceResult:
    sentence: str
    gold_is_constraint: bool
    pred_is_constraint: bool       # True only if SQL generated AND executed OK
    exec_failed: bool = False      # True if SQL was generated but execution failed
    gold_sql: Optional[str] = None
    pred_sql: Optional[str] = None
    # Pairwise violation (only for GT sentences where exec succeeded)
    violation_exact_match: Optional[bool] = None
    violation_precision: Optional[float] = None
    violation_recall: Optional[float] = None
    violation_f1: Optional[float] = None
    violation_jaccard: Optional[float] = None
    num_gold_violations: Optional[int] = None
    num_pred_violations: Optional[int] = None
    num_overlap_violations: Optional[int] = None


# =============================================================================
# Core evaluation
# =============================================================================

def run_evaluation(
    sentences: List[str],
    gt_sentence2dcs_map: Dict[str, List[str]],
    df: pd.DataFrame,
    model,
    tokenizer,
    db_schema: str,
    conn: "duckdb.DuckDBPyConnection",
    dialect: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sql_timeout: float = 120.0,
) -> Tuple[List[SentenceResult], Dict]:
    """
    Main evaluation loop. Returns per-sentence results and aggregate metrics dict.
    """
    sentence_results: List[SentenceResult] = []

    # We'll accumulate these for aggregate violation
    all_pred_tids: Set[int] = set()
    all_gold_tids: Set[int] = set()

    # Pre-compute all gold tids
    print("\n  Pre-computing gold violations...")
    gold_tids_by_sentence: Dict[str, Set[int]] = {}
    for sent, dcs in gt_sentence2dcs_map.items():
        gold_sql = dcs_to_sql(dcs)
        tids = execute_for_tids(gold_sql, conn, timeout_s=sql_timeout)
        if tids is _EXEC_FAILED:
            print(f"    [warning] gold SQL failed for: {sent[:60]}")
            tids = set()
        gold_tids_by_sentence[sent] = tids
        all_gold_tids.update(tids)

    # ── Main loop ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for i, sentence in enumerate(sentences, 1):
        if i % 5 == 0 or i == len(sentences):
            print(f"    [{i}/{len(sentences)}] processing...")

        gold_is_constraint = sentence in gt_sentence2dcs_map
        gold_dcs = gt_sentence2dcs_map.get(sentence)
        gold_sql = dcs_to_sql(gold_dcs) if gold_dcs else None

        # ── Generate SQL ───────────────────────────────────────────────────
        try:
            pred_sql = generate_sql(
                sentence=sentence,
                db_schema=db_schema,
                model=model,
                tokenizer=tokenizer,
                dialect=dialect,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            print(f"    [model error] sentence {i}: {e}")
            pred_sql = None

        # ── Execute predicted SQL ──────────────────────────────────────────
        exec_failed = False
        pred_tids: Optional[Set[int]] = None   # None = not attempted yet

        if pred_sql is not None:
            result = execute_for_tids(pred_sql, conn, timeout_s=sql_timeout)
            if result is _EXEC_FAILED:
                exec_failed = True
                # Treat as if the sentence was skipped entirely:
                # - not counted in classification TP/FP
                # - not counted in pairwise metrics
                # - not counted in aggregate violation
            else:
                pred_tids = result          # Set[int], may be empty
                all_pred_tids.update(pred_tids)

        # pred_is_constraint = SQL generated AND executed successfully
        pred_is_constraint = (pred_sql is not None) and (not exec_failed)

        sr = SentenceResult(
            sentence=sentence,
            gold_is_constraint=gold_is_constraint,
            pred_is_constraint=pred_is_constraint,
            exec_failed=exec_failed,
            gold_sql=gold_sql,
            pred_sql=pred_sql,
        )

        # ── Pairwise violation (GT sentences, exec not failed) ─────────────
        if gold_is_constraint and not exec_failed:
            gold_tids = gold_tids_by_sentence[sentence]

            # pred_tids is either a real set (model predicted constraint) or
            # None (model returned NULL → genuine miss → treat as empty set)
            effective_pred = pred_tids if pred_tids is not None else set()

            p, r, f = _prf(gold_tids, effective_pred)
            j = _jaccard(gold_tids, effective_pred)
            overlap = gold_tids & effective_pred

            sr.violation_exact_match  = (gold_tids == effective_pred)
            sr.violation_precision    = p
            sr.violation_recall       = r
            sr.violation_f1           = f
            sr.violation_jaccard      = j
            sr.num_gold_violations    = len(gold_tids)
            sr.num_pred_violations    = len(effective_pred)
            sr.num_overlap_violations = len(overlap)

        sentence_results.append(sr)

    runtime_s = time.perf_counter() - t0

    runtime_s = time.perf_counter() - t0

    # Sentences where execution failed are excluded from ALL metrics
    scorable = [r for r in sentence_results if not r.exec_failed]
    exec_failed_count = sum(1 for r in sentence_results if r.exec_failed)
    if exec_failed_count:
        print(f"\n  [info] {exec_failed_count} sentences excluded from all metrics due to SQL exec failure")

    # ── Classification metrics (exec-failed excluded) ──────────────────────
    tp = sum(1 for r in scorable if     r.gold_is_constraint and     r.pred_is_constraint)
    fp = sum(1 for r in scorable if not r.gold_is_constraint and     r.pred_is_constraint)
    fn = sum(1 for r in scorable if     r.gold_is_constraint and not r.pred_is_constraint)
    tn = sum(1 for r in scorable if not r.gold_is_constraint and not r.pred_is_constraint)

    clf_precision = tp / (tp + fp) if (tp + fp) else 0.0
    clf_recall    = tp / (tp + fn) if (tp + fn) else 0.0
    clf_f1        = 2 * clf_precision * clf_recall / (clf_precision + clf_recall) \
                    if (clf_precision + clf_recall) else 0.0
    clf_accuracy  = (tp + tn) / len(sentence_results) if sentence_results else 0.0

    # ── Pairwise violation metrics (GT only, exec-failed excluded) ────────
    gt_scorable = [r for r in scorable if r.gold_is_constraint]

    pw_exact_correct = sum(1 for r in gt_scorable if r.violation_exact_match)
    pw_total         = len(gt_scorable)
    pw_exact_acc     = pw_exact_correct / pw_total if pw_total else 0.0

    avg_p, std_p = _mean_std([r.violation_precision  for r in gt_scorable if r.violation_precision  is not None])
    avg_r, std_r = _mean_std([r.violation_recall     for r in gt_scorable if r.violation_recall     is not None])
    avg_f, std_f = _mean_std([r.violation_f1         for r in gt_scorable if r.violation_f1         is not None])
    avg_j, std_j = _mean_std([r.violation_jaccard    for r in gt_scorable if r.violation_jaccard    is not None])

    # ── Aggregate violation metrics ────────────────────────────────────────
    total_gv = len(all_gold_tids)
    total_pv = len(all_pred_tids)
    total_ov = len(all_gold_tids & all_pred_tids)
    agg_vp   = total_ov / total_pv if total_pv else 0.0
    agg_vr   = total_ov / total_gv if total_gv else 0.0
    agg_vf   = 2 * agg_vp * agg_vr / (agg_vp + agg_vr) if (agg_vp + agg_vr) else 0.0

    metrics = {
        "runtime_s": runtime_s,
        # Classification
        "clf_accuracy":         clf_accuracy,
        "clf_precision":        clf_precision,
        "clf_recall":           clf_recall,
        "clf_f1":               clf_f1,
        "clf_tp":               tp,
        "clf_fp":               fp,
        "clf_fn":               fn,
        "clf_tn":               tn,
        # Aggregate violation
        "aggregate_violation_precision": agg_vp,
        "aggregate_violation_recall":    agg_vr,
        "aggregate_violation_f1":        agg_vf,
        "total_gold_violations":         total_gv,
        "total_pred_violations":         total_pv,
        "total_overlap_violations":      total_ov,
        # Pairwise violation
        "pairwise_violation_exact_match":       pw_exact_acc,
        "pairwise_violation_exact_match_count": pw_exact_correct,
        "pairwise_violation_total":             pw_total,
        "pairwise_violation_exec_failed":       exec_failed_count,
        "pairwise_violation_precision":         avg_p,
        "pairwise_violation_precision_std":     std_p,
        "pairwise_violation_recall":            avg_r,
        "pairwise_violation_recall_std":        std_r,
        "pairwise_violation_f1":                avg_f,
        "pairwise_violation_f1_std":            std_f,
        "pairwise_violation_jaccard":           avg_j,
        "pairwise_violation_jaccard_std":       std_j,
        # Dataset stats
        "num_sentences_total": len(sentences),
        "num_sentences_gt":    len(gt_sentence2dcs_map),
    }

    return sentence_results, metrics


# =============================================================================
# Printing  (mirrors run_mapper_eval_full.py section headers exactly)
# =============================================================================

def print_metrics(metrics: Dict, dataset_name: str):
    n_total = metrics["num_sentences_total"]
    n_gt    = metrics["num_sentences_gt"]

    print(f"\n  ── RUNTIME " + "─" * 59)
    print(f"     Pipeline runtime: {metrics['runtime_s']:.2f}s")

    n_scorable = n_total - metrics.get("pairwise_violation_exec_failed", 0)
    print(f"\n  ── CLASSIFICATION (scorable sentences: {n_scorable}/{n_total}) " + "─" * 25)
    print(f"     Accuracy:  {metrics['clf_accuracy']:.3f}")
    print(f"     Precision: {metrics['clf_precision']:.3f}")
    print(f"     Recall:    {metrics['clf_recall']:.3f}")
    print(f"     F1:        {metrics['clf_f1']:.3f}  [PRIMARY]")
    print(f"     TP: {metrics['clf_tp']:3d}  FP: {metrics['clf_fp']:3d}")
    print(f"     FN: {metrics['clf_fn']:3d}  TN: {metrics['clf_tn']:3d}")

    print(f"\n  ── AGGREGATE VIOLATION DISCOVERY (all {n_total} sentences) " + "─" * 13)
    print(f"     Precision: {metrics['aggregate_violation_precision']:.3f}")
    print(f"     Recall:    {metrics['aggregate_violation_recall']:.3f}")
    print(f"     F1:        {metrics['aggregate_violation_f1']:.3f}  [PRIMARY]")
    print(f"     Gold violations:      {metrics['total_gold_violations']}")
    print(f"     Predicted violations: {metrics['total_pred_violations']}")
    print(f"     Overlap violations:   {metrics['total_overlap_violations']}")

    pw_total    = metrics["pairwise_violation_total"]
    exec_failed = metrics["pairwise_violation_exec_failed"]
    print(f"\n  ── PAIRWISE VIOLATION DISCOVERY (GT scorable sentences, n={pw_total}) " + "─" * 15)
    if exec_failed:
        print(f"     Note: {exec_failed} sentence(s) excluded (SQL exec failed/timed out)")
    print(f"     Exact Match:       {metrics['pairwise_violation_exact_match']:.3f}"
          f" ({metrics['pairwise_violation_exact_match_count']}/{pw_total})")
    print(f"     Precision:         {metrics['pairwise_violation_precision']:.3f}"
          f" (±{metrics['pairwise_violation_precision_std']:.3f})")
    print(f"     Recall:            {metrics['pairwise_violation_recall']:.3f}"
          f" (±{metrics['pairwise_violation_recall_std']:.3f})")
    print(f"     F1:                {metrics['pairwise_violation_f1']:.3f}"
          f" (±{metrics['pairwise_violation_f1_std']:.3f})")
    print(f"     Violation Jaccard: {metrics['pairwise_violation_jaccard']:.3f}"
          f" (±{metrics['pairwise_violation_jaccard_std']:.3f})")


# =============================================================================
# Output writers
# =============================================================================

def save_detailed_csv(
    sentence_results: List[SentenceResult],
    output_dir: Path,
    dataset_name: str,
):
    records = []
    for r in sentence_results:
        records.append({
            "sentence":                        r.sentence,
            "gold_is_constraint":              r.gold_is_constraint,
            "pred_is_constraint":              r.pred_is_constraint,
            "exec_failed":                     r.exec_failed,
            "gold_sql":                        r.gold_sql,
            "pred_sql":                        r.pred_sql,
            "pairwise_violation_exact_match":  r.violation_exact_match,
            "pairwise_violation_precision":    r.violation_precision,
            "pairwise_violation_recall":       r.violation_recall,
            "pairwise_violation_f1":           r.violation_f1,
            "pairwise_violation_jaccard":      r.violation_jaccard,
            "num_gold_violations":             r.num_gold_violations,
            "num_pred_violations":             r.num_pred_violations,
            "num_overlap_violations":          r.num_overlap_violations,
        })
    df_out = pd.DataFrame(records)
    path = output_dir / f"{dataset_name}_xiyan_detailed_results.csv"
    df_out.to_csv(path, index=False)
    print(f"\n  Saved detailed results: {path}")


def save_summary_json(
    metrics: Dict,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
):
    summary = {"model": model_name, "dataset": dataset_name, **metrics}
    path = output_dir / f"{dataset_name}_xiyan_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary:          {path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone XiYanSQL evaluation — mirrors run_mapper_eval_full.py metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",       required=True,  help="Dataset directory (e.g. data/sisu/)")
    parser.add_argument("--output_dir",     default=None,   help="Where to save outputs (default: data_dir)")
    parser.add_argument("--model",          default="XGenerationLab/XiYanSQL-QwenCoder-32B-2504")
    parser.add_argument("--dialect",        default="DuckDB")
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--top_p",          type=float, default=0.8)
    parser.add_argument("--sql_timeout",    type=float, default=120.0,
                        help="Max seconds per SQL execution (default 120). "
                             "Slow NOT EXISTS queries are interrupted after this.")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("XIYAN SQL EVALUATION")
    print("=" * 80)

    print(f"\nLoading dataset from {data_dir} ...")
    loader = DatasetLoader(str(data_dir))
    df, sentences, sentence2dcs_map, _ = loader.load()
    dataset_name = loader.dataset_name

    gt_sentence2dcs_map = {s: dcs for s, dcs in sentence2dcs_map.items() if dcs}
    n_total = len(sentences)
    n_gt    = len(gt_sentence2dcs_map)

    print(f"Total sentences:          {n_total}")
    print(f"GT constraint sentences:  {n_gt}")
    print(f"Non-constraint sentences: {n_total - n_gt}")

    if df is None:
        print("ERROR: No CSV found — cannot run evaluation.")
        sys.exit(1)

    # ── SQLite setup ───────────────────────────────────────────────────────
    print("\nLoading data into DuckDB (creating t1_with_tid view)...")
    conn, db_schema = setup_duckdb(df)
    print(f"Schema:\n{db_schema}")

    # ── Model ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[Dataset: {dataset_name}]  Processing {n_total} sentences...")
    try:
        sentence_results, metrics = run_evaluation(
            sentences           = sentences,
            gt_sentence2dcs_map = gt_sentence2dcs_map,
            df                  = df,
            model               = model,
            tokenizer           = tokenizer,
            db_schema           = db_schema,
            conn                = conn,
            dialect             = args.dialect,
            max_new_tokens      = args.max_new_tokens,
            temperature         = args.temperature,
            top_p               = args.top_p,
            sql_timeout         = args.sql_timeout,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()

    # ── Print ──────────────────────────────────────────────────────────────
    print_metrics(metrics, dataset_name)

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n  Saving outputs...")
    save_detailed_csv(sentence_results, output_dir, dataset_name)
    save_summary_json(metrics, output_dir, dataset_name, args.model)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()