#!/usr/bin/env python3
"""
XiYanSQL Prediction Script
===========================
Runs XiYanSQL over all sentences in a dataset and saves predictions to JSON.

Output JSON format (one entry per sentence):
  [
    {
      "sentence": "...",
      "gt": ["predicate1", "predicate2"] | null,
      "prediction": "SELECT ..." | null
    },
    ...
  ]

Usage
-----
    python run_xiyan_predict.py --data_dir data/hospital/
        [--output path/to/output.json]
        [--model XGenerationLab/XiYanSQL-QwenCoder-32B-2504]
        [--table_name violations]
        [--dialect SQLite]
        [--max_new_tokens 512] [--temperature 0.1] [--top_p 0.8]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Dataset loader
# =============================================================================

def load_dataset(data_dir: Path):
    """Load sentences and ground-truth DC map from annotated JSON."""
    dataset_name = data_dir.name
    annotated_file = data_dir / f"{dataset_name}_annotated-sentences.json"
    with open(annotated_file) as f:
        gt_data = json.load(f)

    sentences = []
    sentence2gt: Dict[str, Optional[List[str]]] = {}
    for item in gt_data["mapping"]:
        sentence = item["sentences"][0]
        sentences.append(sentence)
        rules = item.get("rules")
        sentence2gt[sentence] = rules[0] if rules else None

    csv_path = data_dir / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else None

    return sentences, sentence2gt, df


# =============================================================================
# Schema builder
# =============================================================================

def build_schema_string(df: pd.DataFrame, table_name: str) -> str:
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
    return f"CREATE TABLE {table_name} (\n    tid INTEGER,\n{cols}\n);"


# =============================================================================
# Prompt & model
# =============================================================================

_PROMPT_TEMPLATE = """\
You are an expert SQL assistant working with a single table aliased as t1 \
(and t2 for self-joins).

Your task: given a sentence, write a {dialect} SQL query that returns the row \
identifiers of every tuple (or pair of tuples) that VIOLATES the constraint \
the sentence describes.

Output format:
- Single-tuple violations  → SELECT t1.tid FROM {table_name} AS t1 WHERE <condition>
- Pair-of-tuple violations → SELECT t1.tid, t2.tid FROM {table_name} AS t1, {table_name} AS t2 WHERE <condition>
- Not a checkable data constraint → output exactly: NULL

Rules:
- Always alias the table as t1. For self-joins add a second alias t2.
- Always select t1.tid (and t2.tid for pairs) — these are the row identifiers.
- Predicates always reference columns as t1.<col> or t2.<col>.
- Return ONLY the SQL query or the word NULL. No explanation, no markdown fences.

Database schema:
{db_schema}

Sentence:
{question}

SQL:"""


def load_model(model_name: str):
    print(f"[XiYan] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[XiYan] Model ready.")
    return model, tokenizer


def _clean_sql(raw: str, table_name: str) -> Optional[str]:
    """Parse model output into a clean SQL string or None."""
    raw = raw.strip()

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

    return raw


def generate_sql(
    sentence: str,
    db_schema: str,
    table_name: str,
    model,
    tokenizer,
    dialect: str = "SQLite",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.8,
) -> Optional[str]:
    """Run the model for one sentence. Returns SQL string or None."""
    prompt = _PROMPT_TEMPLATE.format(
        table_name=table_name,
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
    return _clean_sql(raw, table_name)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run XiYanSQL predictions and save to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",       required=True,
                        help="Dataset directory (e.g. data/hospital/)")
    parser.add_argument("--output",         default=None,
                        help="Output JSON path (default: <data_dir>/<dataset>_xyyan_predictions.json)")
    parser.add_argument("--model",          default="XGenerationLab/XiYanSQL-QwenCoder-32B-2504")
    parser.add_argument("--table_name",     default="data",
                        help="Table name used in SQL generation (default: data)")
    parser.add_argument("--dialect",        default="SQLite")
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--top_p",          type=float, default=0.8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_name = data_dir.name

    output_path = Path(args.output) if args.output else \
                  data_dir / f"{dataset_name}_xyyan_predictions.json"

    # ── Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading dataset from {data_dir} ...")
    sentences, sentence2gt, df = load_dataset(data_dir)
    print(f"  Total sentences: {len(sentences)}")
    print(f"  GT constraints:  {sum(1 for v in sentence2gt.values() if v is not None)}")

    if df is None:
        print("WARNING: No CSV found — schema will be omitted from prompt.")
        db_schema = f"-- No schema available. Table name: {args.table_name}"
    else:
        db_schema = build_schema_string(df, args.table_name)
        print(f"\nSchema:\n{db_schema}")

    # ── Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)

    # ── Predict ────────────────────────────────────────────────────────────
    print(f"\nRunning predictions on {len(sentences)} sentences...\n")
    results = []
    for i, sentence in enumerate(sentences, 1):
        if i % 5 == 0 or i == len(sentences):
            print(f"  [{i}/{len(sentences)}] processing...")

        try:
            pred_sql = generate_sql(
                sentence=sentence,
                db_schema=db_schema,
                table_name=args.table_name,
                model=model,
                tokenizer=tokenizer,
                dialect=args.dialect,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"  [model error] sentence {i}: {e}")
            pred_sql = None

        results.append({
            "sentence":   sentence,
            "gt":         sentence2gt.get(sentence),   # list of predicates or null
            "prediction": pred_sql,                    # SQL string or null
        })

    # ── Save ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} predictions → {output_path}")


if __name__ == "__main__":
    main()