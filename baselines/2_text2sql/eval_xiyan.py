#!/usr/bin/env python3
"""
Standalone XiYanSQL Evaluation Script
======================================
Mirrors the metrics and output format of run_mapper_eval_full.py but is
completely self-contained — no imports from src.* or experiments.*.

Evaluates three aspects over ALL sentences from a dataset:

1. CLASSIFICATION
   Ground truth positive = sentence in sentence2dcs_map (non-empty rules).
   Predicted positive    = model returns any non-null SQL.
   Metrics: accuracy, precision, recall, F1, TP/FP/FN/TN.

2. PAIRWISE VIOLATION DISCOVERY  (GT sentences only)
   For each sentence that has a gold DC, execute both the gold SQL and
   the predicted SQL against SQLite, compare the resulting tid sets.
   Metrics: exact match, precision, recall, F1, Jaccard (mean ± std).

3. AGGREGATE VIOLATION DISCOVERY  (ALL sentences)
   Union all predicted violation tids vs union all gold violation tids.
   Metrics: precision, recall, F1.

Outputs
-------
- Console print (same section headers as run_mapper_eval_full.py)
- <dataset>_xiyan_detailed_results.csv   (one row per sentence)
- <dataset>_xiyan_summary.json           (all scalar metrics)

Usage
-----
    python eval_xiyan.py --data_dir data/sisu/ [--output_dir results/xiyan/]
                         [--model XGenerationLab/XiYanSQL-QwenCoder-32B-2504]
                         [--max_new_tokens 512] [--temperature 0.1] [--top_p 0.8]
                         [--dialect SQLite]
"""

import argparse
import json
import re
import sqlite3
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
# SQLite helpers
# =============================================================================

def build_schema_string(df: pd.DataFrame) -> str:
    """Generate CREATE TABLE DDL from DataFrame dtypes."""
    type_map = {
        "int64":   "INTEGER",
        "float64": "REAL",
        "object":  "TEXT",
        "bool":    "INTEGER",
    }
    cols = ",\n".join(
        f"    {col} {type_map.get(str(dtype), 'TEXT')}"
        for col, dtype in df.dtypes.items()
    )
    return f"CREATE TABLE t1 (\n{cols}\n);"


def dcs_to_sql(dcs: List[str]) -> str:
    """Convert a list of DC predicates to a SQL SELECT query."""
    uses_t2 = any("t2." in p for p in dcs)
    from_clause = "t1, t1 AS t2" if uses_t2 else "t1"
    where_clause = " AND ".join(dcs)
    return f"SELECT * FROM {from_clause} WHERE {where_clause}"


def execute_for_tids(sql: str, conn: sqlite3.Connection) -> Set[int]:
    """
    Execute SQL and collect all tid values from every result row.
    Handles both single-table (tid) and self-join (tid + tid_1 / tid:1) queries.

    Only extracts columns whose name is exactly 'tid' or matches 'tid_<N>' /
    'tid:<N>' (SQLite auto-rename for duplicate column names in self-joins).
    Ignores non-integer values gracefully.
    """
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            return set()
        col_names = [d[0] for d in cur.description]

        # Accept: exact "tid", or SQLite duplicate renames "tid_1", "tid:1"
        tid_indices = [
            i for i, name in enumerate(col_names)
            if name == "tid" or re.match(r"^tid[_:]\d+$", name)
        ]
        if not tid_indices:
            # Fallback: if model selected tid as first column without renaming
            if col_names and col_names[0].lower() == "tid":
                tid_indices = [0]
            else:
                # Cannot identify tid — log and return empty
                print(f"    [tid warning] no tid column in result. Columns: {col_names}")
                return set()

        tids: Set[int] = set()
        for row in cur.fetchall():
            for idx in tid_indices:
                val = row[idx]
                if val is None:
                    continue
                try:
                    tids.add(int(val))
                except (ValueError, TypeError):
                    pass  # non-integer in tid column — skip silently
        return tids
    except Exception as e:
        print(f"    [SQL error] {e}\n    SQL: {sql[:120]}")
        return set()


def setup_sqlite(df: pd.DataFrame) -> Tuple[sqlite3.Connection, str]:
    """
    Load df into an in-memory SQLite DB with a tid column prepended.
    Returns (connection, schema_string) — schema is built from the
    tid-augmented df so it always matches what is in the DB.
    """
    df = df.copy()
    if "tid" not in df.columns:
        df.insert(0, "tid", range(len(df)))
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("t1", conn, index=False, if_exists="replace")
    schema = build_schema_string(df)   # built after tid is added
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
_EXEC_FAILED = object()


def execute_for_tids(
    sql: str,
    conn: sqlite3.Connection,
    timeout_s: float = 120.0,
) -> object:  # returns Set[int] | _EXEC_FAILED
    """
    Execute SQL and collect tid values.

    Returns:
        Set[int]     — possibly empty, if SQL ran fine (even 0 rows)
        _EXEC_FAILED — SQL raised an exception, timed out, or returned no
                       identifiable tid column

    Uses a SQLite progress handler to enforce a wall-clock timeout so that
    slow correlated subqueries (NOT EXISTS, IN) don't hang the process.
    Only extracts columns named exactly 'tid' or SQLite duplicate-rename 'tid_<N>'.
    """
    import time as _time

    deadline = _time.monotonic() + timeout_s
    timed_out = [False]

    def _progress():
        if _time.monotonic() > deadline:
            timed_out[0] = True
            return 1   # non-zero → SQLite raises OperationalError: interrupted
        return 0

    conn.set_progress_handler(_progress, 1000)   # called every ~1000 VM ops
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            return set()
        col_names = [d[0] for d in cur.description]

        tid_indices = [
            i for i, name in enumerate(col_names)
            if name == "tid" or re.match(r"^tid[_:]\d+$", name)
        ]
        if not tid_indices:
            print(f"    [tid warning] no tid column in result. Columns: {col_names}")
            return _EXEC_FAILED

        tids: Set[int] = set()
        for row in cur.fetchall():
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
    finally:
        conn.set_progress_handler(None, 0)   # always clear the handler


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
    # Per-sentence timing
    inference_time_s: Optional[float] = None   # model inference time
    sql_exec_time_s: Optional[float] = None    # SQL execution time (None if no SQL generated)
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
    conn: sqlite3.Connection,
    dialect: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sql_timeout: float = 30.0,
) -> Tuple[List[SentenceResult], Dict]:
    """
    Main evaluation loop. Returns per-sentence results and aggregate metrics dict.
    """
    sentence_results: List[SentenceResult] = []

    all_pred_tids: Set[int] = set()
    all_gold_tids: Set[int] = set()

    # ── Gold pre-computation ───────────────────────────────────────────────
    print("\n  Pre-computing gold violations...")
    t_gold_start = time.perf_counter()
    gold_tids_by_sentence: Dict[str, Set[int]] = {}
    for sent, dcs in gt_sentence2dcs_map.items():
        gold_sql = dcs_to_sql(dcs)
        tids = execute_for_tids(gold_sql, conn, timeout_s=sql_timeout)
        if tids is _EXEC_FAILED:
            print(f"    [warning] gold SQL failed for: {sent[:60]}")
            tids = set()
        gold_tids_by_sentence[sent] = tids
        all_gold_tids.update(tids)
    gold_precomp_s = time.perf_counter() - t_gold_start

    # ── Main loop ──────────────────────────────────────────────────────────
    inference_times: List[float] = []   # per-sentence model inference
    sql_exec_times:  List[float] = []   # per-sentence SQL execution (pred only)

    t_loop_start = time.perf_counter()
    for i, sentence in enumerate(sentences, 1):
        if i % 5 == 0 or i == len(sentences):
            print(f"    [{i}/{len(sentences)}] processing...")

        gold_is_constraint = sentence in gt_sentence2dcs_map
        gold_dcs = gt_sentence2dcs_map.get(sentence)
        gold_sql = dcs_to_sql(gold_dcs) if gold_dcs else None

        # ── Generate SQL ───────────────────────────────────────────────────
        t_inf = time.perf_counter()
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
        inference_times.append(time.perf_counter() - t_inf)

        # ── Execute predicted SQL ──────────────────────────────────────────
        exec_failed = False
        pred_tids: Optional[Set[int]] = None

        if pred_sql is not None:
            t_exec = time.perf_counter()
            result = execute_for_tids(pred_sql, conn, timeout_s=sql_timeout)
            sql_exec_times.append(time.perf_counter() - t_exec)
            if result is _EXEC_FAILED:
                exec_failed = True
            else:
                pred_tids = result
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
            inference_time_s=inference_times[-1],
            sql_exec_time_s=sql_exec_times[-1] if sql_exec_times and pred_sql is not None else None,
        )

        # ── Pairwise violation (GT sentences, exec not failed) ─────────────
        if gold_is_constraint and not exec_failed:
            gold_tids = gold_tids_by_sentence[sentence]
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

    loop_s   = time.perf_counter() - t_loop_start
    total_s  = loop_s   # dataset processing time only (excludes gold pre-computation)

    avg_inference_s = float(np.mean(inference_times)) if inference_times else 0.0
    avg_sql_exec_s  = float(np.mean(sql_exec_times))  if sql_exec_times  else 0.0

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
        # Runtime (dataset processing only — excludes gold pre-computation)
        "runtime_dataset_s":        total_s,
        "runtime_gold_precomp_s":   gold_precomp_s,
        "runtime_avg_inference_s":  avg_inference_s,
        "runtime_avg_sql_exec_s":   avg_sql_exec_s,
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
    print(f"     Dataset processing:      {metrics['runtime_dataset_s']:.2f}s  (excludes gold pre-computation)")
    print(f"     Gold pre-computation:    {metrics['runtime_gold_precomp_s']:.2f}s")
    print(f"     Avg model inference:     {metrics['runtime_avg_inference_s']:.2f}s / sentence")
    print(f"     Avg SQL execution:       {metrics['runtime_avg_sql_exec_s']:.3f}s / sentence")

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
            "inference_time_s":                round(r.inference_time_s, 4) if r.inference_time_s is not None else None,
            "sql_exec_time_s":                 round(r.sql_exec_time_s, 4) if r.sql_exec_time_s is not None else None,
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
    parser.add_argument("--dialect",        default="SQLite")
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
    print("\nLoading data into SQLite (adding tid column)...")
    conn, db_schema = setup_sqlite(df)
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