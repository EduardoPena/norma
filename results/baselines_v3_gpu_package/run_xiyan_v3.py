#!/usr/bin/env python3
"""XiYan-SQL inference for the round-v3 baseline evaluation (GPU only).

Generates one violation query (or NULL) per sentence with exactly the prompt,
schema DDL, decoding parameters and SQL cleaning of eval_xiyan_duck.py, which
produced the paper's XiYan-SQL runs; that module is imported, not changed. What
differs is only the input and the output:
- sentences come from the current gold standard (<name>_text2rule_gt.json, all
  sentences, constraint or not), not from the old annotated-sentences format;
- the CSV may be truncated to its first --rows rows (the HMDA slices of 1M rows);
- each run fixes a seed (sampling is on, temperature 0.1), and only the queries
  are saved; execution and metrics are computed afterwards on the Norma side by
  experiments/baselines/evaluate_baselines_v3.py xiyan.

Prompt variants (--prompt):
- A: eval_xiyan_duck.py / eval_xiyan.py (table t1, "prefer JOINs"), DuckDB dialect;
- B: xyyan_text2sql_inference.py (table "data" aliased t1/t2, tid in the DDL),
  SQLite dialect, the script's defaults. Its queries are evaluated with --table data.
Both load the model in bf16 with eval_xiyan_duck.load_model (B's own loader passes
torch_dtype, which recent transformers may ignore and load a 32B model in fp32);
loading does not change the generated queries.

Output: <out>/<dataset>/seed<k>/<dataset>_xiyan_detailed_results.csv
        (columns sentence, pred_sql, raw_seconds) and meta.json.

Usage:
  python run_xiyan_v3.py --dataset hmdaz --csv data/hmdaz/hmdaz_first1M.csv.gz \
      --gt data/hmdaz/hmdaz_text2rule_gt.json --seeds 0 1 2 --out outputs
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import transformers

from eval_xiyan_duck import generate_sql, load_model, setup_duckdb

MODEL = "XGenerationLab/XiYanSQL-QwenCoder-14B-2504"


def sentences_of(gt_path):
    payload = json.loads(Path(gt_path).read_text())
    return [s for entry in payload["mapping"] for s in entry["sentences"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--rows", type=int, default=None, help="first N rows of the CSV (HMDA: 1000000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", choices=["A", "B"], default="A")
    parser.add_argument("--dialect", default=None, help="default: DuckDB (A), SQLite (B)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    sentences = sentences_of(args.gt)
    df = pd.read_csv(args.csv, nrows=args.rows, low_memory=False)
    conn, schema = setup_duckdb(df)
    generate = generate_sql
    if args.prompt == "B":
        import xyyan_text2sql_inference as variant_b

        schema = variant_b.build_schema_string(df, "data")
        args.dialect = args.dialect or "SQLite"

        def generate(sentence, db_schema, model, tokenizer, **kw):
            return variant_b.generate_sql(sentence=sentence, db_schema=db_schema, table_name="data",
                                          model=model, tokenizer=tokenizer, **kw)
    args.dialect = args.dialect or "DuckDB"
    print(f"[{args.dataset}] {len(sentences)} sentences, table {df.shape}")
    model, tokenizer = load_model(args.model)
    revision = getattr(model.config, "_commit_hash", None)

    for seed in args.seeds:
        out_dir = Path(args.out) / args.dataset / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        transformers.set_seed(seed)
        rows, t0 = [], time.time()
        for i, sentence in enumerate(sentences, 1):
            s0 = time.time()
            try:
                sql = generate(sentence=sentence, db_schema=schema, model=model, tokenizer=tokenizer,
                                   dialect=args.dialect, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p)
            except Exception as error:  # logged and kept as no prediction, never hidden
                print(f"  [model error] seed {seed} sentence {i}: {error}")
                sql = None
            rows.append({"sentence": sentence, "pred_sql": sql, "raw_seconds": round(time.time() - s0, 2)})
            if i % 10 == 0 or i == len(sentences):
                print(f"  seed {seed}: {i}/{len(sentences)}")
        pd.DataFrame(rows).to_csv(out_dir / f"{args.dataset}_xiyan_detailed_results.csv", index=False)
        (out_dir / "meta.json").write_text(json.dumps({
            "dataset": args.dataset, "prompt": args.prompt, "model": args.model, "model_revision": revision, "seed": seed,
            "dialect": args.dialect, "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "csv": args.csv, "rows": int(df.shape[0]), "gt": args.gt, "sentences": len(sentences),
            "queries": sum(1 for r in rows if r["pred_sql"]), "seconds": round(time.time() - t0, 1),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "finished_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
