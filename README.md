# Norma: Mapping Regulatory Requirements to Data Quality Constraints

This repository is the supplementary material of the ICDE 2027 submission
*Mapping Regulatory Requirements to Data Quality Constraints* (Pena, de Freitas,
Cunha de Almeida, da Silva). It contains:

- the complete code of **Norma**: classification, the Single-Prompt and Multi-Step
  mappers, prompts, LLM calls, the SQL/DuckDB execution of denial constraints (DCs),
  and evaluation;
- the six **datasets** used in the paper, with their regulatory sentences and gold DCs;
- every **LLM output, metric, and log** behind the numbers of the paper, plus the
  scripts that turn them into the paper's tables and figures;
- the two **baselines** (DC Discovery and XiYan-SQL), with their inputs, outputs,
  and evaluation under the same protocol.

All LLM outputs are stored, so the reported metrics can be recomputed **without an
API key and without calling any model** (see [Quick start](#quick-start)).

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [From the paper to the artifacts](#from-the-paper-to-the-artifacts)
- [Datasets](#datasets)
- [Results: layout and file formats](#results-layout-and-file-formats)
- [Metrics](#metrics)
- [Running Norma from scratch (LLM calls)](#running-norma-from-scratch-llm-calls)
- [Baselines](#baselines)
- [Configuration reference](#configuration-reference)
- [Tests](#tests)
- [License](#license)

## Quick start

Requirements: Python 3.11, about 8 GB of RAM (the HMDA runs peaked at 5.4 GB), and
about 5 GB of free disk once the data is unpacked.

```bash
git clone <this repository> norma && cd norma
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # exact versions used: requirements-lock.txt

# 1. Unpack the datasets (data/<dataset>/<table>.csv.gz -> .csv, MD5-checked)
python tools/unpack_data.py

# 2. Run the test suite (90 tests)
python -m unittest discover -s tests

# 3. Recompute the mapping metrics (experiment E2) of all 72 runs from the stored
#    LLM outputs and compare them with the stored summaries (no API key needed).
#    Hospital and Tax take about a minute; all six datasets take longer.
python experiments/reproduce_metrics.py --datasets hospital tax
python experiments/reproduce_metrics.py                   # all six datasets

# 4. Regenerate the paper tables from the stored summaries
V3_RESULTS=results/v3_fix python results/v3/analysis/tables.py            # Table III
V3_RESULTS=results/v3_fix E3_ROOTS=results/v3_fix,results/v3_fix_rep2,results/v3_fix_rep3 \
BASELINES=results/v3_fix/baselines/baselines_final_300s.csv \
TABLE_OUT=results/v3_fix/baselines/table_300s \
python results/v3/analysis/table3.py                                       # Table IV
```

`reproduce_metrics.py` prints one line per run and writes
`results/v3_fix/reproduce_e2.csv`, with the stored and the recomputed value of each
metric. The stored values are rounded to four decimals, so differences up to
0.0001 are expected; the script fails if any difference is above `--tol` (0.001).

## Repository layout

| Path | Content |
|---|---|
| `src/` | Norma itself (see below). |
| `prompts/` | All prompts, as YAML templates. |
| `experiments/` | Experiment runners, the round orchestrator, config generators, and `configs/` with every YAML used. |
| `experiments/baselines/` | Preparation, execution, and evaluation of the baselines. |
| `baselines/` | Baseline programs and artifacts: `1_profiling/` (DC Discovery `.jar`, inputs, discovered DCs) and `2_text2sql/` (XiYan-SQL inference scripts). |
| `data/` | The six datasets (compressed CSV, schema, predicate space, gold DCs, source documents). |
| `results/` | LLM outputs, metrics, telemetry, analysis scripts, and tables of every run. The paper uses `results/v3_fix*`. |
| `notebooks/results_analysis/` | Notebooks that draw Fig. 6 and Fig. 7 (`*_hmda.ipynb`). |
| `tools/` | `unpack_data.py`, plus auxiliary tools (sentence generation, error injection, SLM training) not used in the paper. |
| `tests/` | Unit tests. |
| `docs/README.pt-BR.md` | The developer guide in Portuguese (more detail on options and history). |

Main modules of `src/`:

| Module | Role |
|---|---|
| `src/classifiers/llm.py` | Constraint classifier (is a sentence a checkable rule?). Prompt: `prompts/classification.yaml`. |
| `src/mappers/llm.py` | Single-Prompt mapper (sentence → DC in one call, batched). Prompt: `prompts/mapping.yaml`, key `with_examples_v2` (paper) or `with_examples` (original protocol). |
| `src/mappers/multi_step_refinement.py` | Multi-Step mapper: schema understanding, predicate selection, rule gate, constraint construction, and execution-guided refinement. Prompt: `prompts/multi_step_refinement.yaml`. |
| `src/mappers/column_retrieval.py`, `predicate_filter.py` | Candidate column and predicate selection. |
| `src/predicates/` | Rule grammar (`AND`/`OR` trees), normalization, predicate spaces (v1: all templates; v2: generated on demand from `*_schema.json`). |
| `src/llm/client.py`, `schemas.py` | LLM client (OpenRouter and other providers), retries, structured-output JSON schemas. |
| `src/dbms/sql_translator.py`, `db_interface.py` | DC → SQL compilation and execution in DuckDB. |
| `src/evaluation/mapping.py`, `classification.py`, `value_aliases.py` | Metrics (see [Metrics](#metrics)). |
| `src/data/loader.py` | Dataset loader (schema, CSV, ground truth). |

## From the paper to the artifacts

All Norma numbers come from round **v3_fix**: six LLMs via OpenRouter, six datasets,
predicate space v2, first 1M rows of the HMDA files.

| Paper item | Stored data | Produced by |
|---|---|---|
| Table I (dataset statistics) | `data/*/`, `results/v3_fix/analysis/gold_concentration.{csv,md}` | `results/v3/analysis/tables.py` |
| Fig. 6 (classification, E1) | `results/v3_fix/e1_classifiers/`, `results/v3_fix/figures/v3_fig6_classifiers.pdf` | `notebooks/results_analysis/2_classifier/2_classifier_analysis_hmda.ipynb` |
| Fig. 7 (mapping, E2) | `results/v3_fix/e2_mapping/`, `results/v3_fix/figures/v3_fig7_mapping.pdf` | `notebooks/results_analysis/3_mapper/3_mapper_analysis_hmda.ipynb` |
| Table III (mean mapping quality, E2) | `results/v3_fix/analysis/v3_mapping_metrics_table.tex`, `e2_means.*`, `e2_new_metrics.*` | `results/v3/analysis/tables.py` |
| Table IV (end to end, E3, and baselines) | `results/v3_fix/baselines/table_300s/v3_table3_end2end.{tex,csv}` | `results/v3/analysis/table3.py` |
| E3, three runs (mean ± std) | `results/v3_fix/e3_end2end/`, `results/v3_fix_rep2/`, `results/v3_fix_rep3/` (per-run values in `results/v3_fix/analysis/e3_three_runs.csv`) | runs: `experiments/hmda_smoke.py` with `experiments/configs/v3_fix*`; mean ± std: `results/v3/analysis/table3.py` |
| Baselines (Table IV rows) | `results/v3_fix/baselines/` (summaries, logs, `README.md`) | `experiments/baselines/evaluate_baselines_v3.py` |
| AV sensitivity and violation concentration | `results/v3_fix/analysis/av_sensitivity_*`, `gold_concentration.*` | `results/v3/analysis/av_sensitivity.py`, `tables.py` |

Fig. 6 and Fig. 7 are regenerated with (needs `pip install -r requirements-analysis.txt`):

```bash
export HMDA_RESULTS=v3_fix HMDA_MODELS=experiments/configs/v3_models.yaml \
       RESULTS_DATASETS=hospital,restaurant,sisu,tax,hmdaz,hmda_checks \
       RESULTS_LABEL="Paper + HMDA" ROW_NOTE="paper datasets whole; HMDA first 1,000,000 rows" FIG_PREFIX=v3
for nb in 2_classifier/2_classifier_analysis_hmda 3_mapper/3_mapper_analysis_hmda; do
  jupyter nbconvert --to notebook --execute notebooks/results_analysis/$nb.ipynb \
    --output-dir results/v3_fix/notebooks --ExecutePreprocessor.timeout=600
done
```

The figures and tables are written into `results/v3_fix/figures/` and
`results/v3_fix/analysis/`, overwriting the stored copies. The paper typesets a few
column headers differently (for example, "Constr.-F1" for "Rule-F1"); the numbers are
the same. `tables.py` also rewrites `analysis/e3_raw.csv` with single-run columns;
`table3.py` with `E3_ROOTS` produces the three-run version.

Earlier rounds are kept for provenance and are **not** used in the paper:
`results/v3` (first v3 run; its Single-Prompt outputs are the ones re-evaluated in
`v3_fix`), `results/v3_reeval` (re-evaluation of v3 with the corrected pair
semantics), `results/hmda*` (preparatory HMDA rounds), and
`results/hmdaz_ms_strict_scope` (an additional prompt experiment on HMDAZ).

## Datasets

Each dataset lives in `data/<dataset>/`:

| File | Content |
|---|---|
| `<table>.csv.gz` | The data. `tools/unpack_data.py` writes `<table>.csv`, which is what the code reads. |
| `<table>_schema.json` | Predicate space v2: columns, types, allowed operators, descriptions, small domains. |
| `*_predicates.json` | Predicate space v1 (all predicate templates); used only by the document-level mapper. |
| `*_text2rule_gt.json` | Gold standard: every sentence of the document, with its DC or `null`. |
| `*_glossary.json` | (HMDAZ only) official column definitions merged into the schema descriptions. |
| source documents | The regulatory text the sentences were taken from, when redistributable. |

| Dataset | Rows | Cols | Sentences | Gold DCs | Source of the data | Source of the rules |
|---|---|---|---|---|---|---|
| Hospital | 1k | 17 | 109 | 15 | HoloClean benchmark | constraints of prior work, sentences written for them |
| Tax | 100k | 15 | 126 | 6 | synthetic tax benchmark of DC discovery work | constraints of prior work, sentences written for them |
| Restaurant | 128k | 21 | 33 | 7 | [NYC restaurant inspection results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) | NYC grading and inspection guide (`Restaurant.txt`) |
| SISU | 102k | 56 | 42 | 17 | [SISU open data](https://dadosabertos.mec.gov.br/sisu) (Brazil) | Portaria Normativa No. 21/2012 |
| HMDAZ | 1M | 135 | 17 | 9 | HMDA loan application register, 2018 (CFPB/FFIEC) | Regulation Z, §§ 1026.32, 1026.35, 1026.43 (`regulation_z-2018-section_32_35_43.pdf`) |
| HMDA Checks | 1M | 81 | 158 | 112 | HMDA loan application register, 2014 (CFPB/FFIEC) | FFIEC 2014 HMDA edit specification (`hmda-2014-edits.pdf`) |

**HMDA slices.** The public HMDA files are large (2018: 15,119,651 rows, 11.6 GB;
2014 file used here: 4,832,425 rows, 3.9 GB). Every experiment uses the **first
1,000,000 rows** of each file, in file order (`sample_rows: 1000000`). The repository
ships exactly that slice (`hmdaz.csv.gz`, `hmda.csv.gz`); its MD5 equals the MD5 of
the first 1,000,001 lines (header included) of the original files. The 2014 file is
already filtered to originated, first-lien, owner-occupied, 1-4 family loans. See
`results/hmda/README_datasets.md` (in Portuguese) for the details.

**Gold standard format** (`*_text2rule_gt.json`, key `mapping`): one entry per rule,
listing the sentences that state it and the DC(s) that describe its *violations*:

```json
{
  "id": 1,
  "sentences": ["If amount is positive, income must be positive."],
  "rules": [{"AND": ["t1.amount > 0", "t1.income <= 0"]}]
}
```

`rules` is `null` for non-constraint sentences. A DC is a predicate string or an
`AND`/`OR` tree; predicates use `t1.` (and `t2.` for two-tuple DCs). A tuple violates
the DC when all its predicates hold.

## Results: layout and file formats

A run directory (for example
`results/v3_fix/e2_mapping/hmda_e2_tax_gpt6luna_multistep_<timestamp>/`) contains:

| File | Content |
|---|---|
| `config.yaml` | The exact runner configuration (dataset options, model slug, prompt, mapper parameters). |
| `execution_config.csv` | The run parameters as `parameter,value` pairs. |
| `outputs/*.json` | The model output of every sentence, one file per variant. |
| `metrics/*_detailed.csv` (E2) | One row per gold sentence: `sentence`, `gold_constraint`, `pred_constraint` (the DC after normalization), `pred_is_non_null`, per-sentence metrics. |
| `metrics/*_detailed.json` (E1) | Per-sentence classification decisions. |
| `results_summary.csv` | One row per variant with all metrics (column names in [Metrics](#metrics)). |

Telemetry of a round (`results/v3_fix/telemetry/<timestamp>/`): `attempts.jsonl`
(one line per LLM call: model, tokens, reported cost, latency, errors),
`summary.json`, `job_logs/`, `runner_configs/`, and `CHANGES.md` (what was changed or
rerun, and why).

## Metrics

DCs are compiled to SQL by `src/dbms/sql_translator.py` and executed in DuckDB:

- single tuple: `SELECT t1.tid FROM <table> t1 WHERE <predicates>`;
- two tuples: `SELECT t1.tid, t2.tid FROM <table> t1, <table> t2 WHERE t1.tid <> t2.tid AND <predicates>`.

Violations are always counted per tuple (the set of `tid`s returned).
Queries are limited to 300 s (`NORMA_SQL_TIMEOUT_S`); a query that fails counts as an
execution error, never silently dropped.

| Paper | `results_summary.csv` column | Definition |
|---|---|---|
| Classification P / R / F1 | E1: `precision`, `recall`, `f1_score`; E3: `clf_precision`, `clf_recall`, `clf_f1` | Constraint vs. non-constraint sentences. |
| CM | `constraint_exact_*` | Exact constraint mapping: predicted DC equals the gold DC after normalization. |
| CM≡ | `constraint_equiv_*` | Equivalence-aware mapping: DCs compared in logical normal form; a code and its label (e.g., `lien_status = 1` vs. `lien_status_name = 'Secured by a first lien'`) are unified only if they select exactly the same rows. |
| AV | `agg_viol_precision`, `agg_viol_recall`, `agg_viol_f1` | Aggregate violation detection: union of tuples flagged by all predicted DCs vs. by all gold DCs. |
| Rule-F1 / Constr.-F1 | `pairwise_viol_f1_mean` | Mean per-constraint violation F1 (predicted vs. gold DC of each sentence). |
| Exec. | `violation_identical_rate` | Share of gold DCs (with at least one violation) whose prediction flags exactly the same tuples. |
| Fail. / No output | `model_output_failures` | Sentences for which the model gave no valid output after retries. |

## Running Norma from scratch (LLM calls)

New runs call the models through [OpenRouter](https://openrouter.ai). Set the key in
the environment only (never in a file of the repository):

```bash
export OPENROUTER_API_KEY=...
```

The six models of the paper and their OpenRouter slugs are in
`experiments/configs/v3_models.yaml`. LLM outputs are not deterministic, so new runs
give close but not identical numbers; the paper reports three runs for E3.

A round is a plan of jobs, each running one runner config in its own subprocess, with
a cost cap, retries, and telemetry:

```bash
# round v3: E1 and E2 (all models, six datasets) and E3
python experiments/hmda_smoke.py experiments/configs/v3_run.yaml --plan experiments/configs/v3/hmda_plan_e1_e2.json
python experiments/hmda_smoke.py experiments/configs/v3_run.yaml --plan experiments/configs/v3/hmda_plan_e3.json
# round v3_fix: Multi-Step E2 and E3 rerun with the relative rule gate (paper numbers)
python experiments/hmda_smoke.py experiments/configs/v3_fix_run.yaml --plan experiments/configs/v3_fix/plan_multistep_e2_e3.json
# E3 repetitions 2 and 3
python experiments/hmda_smoke.py experiments/configs/v3_fix_reps_run.yaml --plan experiments/configs/v3_fix_reps_plan.json
# add --dry-run to check a plan; add --resume <telemetry dir> to resume a stopped round
```

A single config can also be run directly:

```bash
python experiments/2_run_classifiers.py experiments/configs/v3/hmda_e1_tax_gpt6luna.yaml        # E1
python experiments/3_run_mapping.py     experiments/configs/v3_fix/hmda_e2_tax_gpt6luna_MultiStep.yaml  # E2
python experiments/1_run_end2end.py     experiments/configs/v3_fix/hmda_e3_tax_MultiStep.yaml          # E3
```

`experiments/hmda_make_configs.py` generates configs and plans for other models or
datasets (`--models`, `--datasets`, `--experiments e1 e2 e3`, `--predicate-space v2`,
`--row-limit`).

## Baselines

Both baselines are evaluated by `experiments/baselines/evaluate_baselines_v3.py` with
the same data, gold standard, SQL translator, and evaluator as Norma. Decisions,
failures, and per-configuration results are documented in
`results/v3_fix/baselines/README.md` (in Portuguese).

**DC Discovery** (`baselines/1_profiling/dc_discovery/DCDiscovery.jar`; Java ≥ 20):

```bash
# inputs: gold columns (Hospital and Tax: all), typed header, HMDA: seeded 100k sample of the first 1M rows
python experiments/baselines/prepare_dc_discovery.py --datasets hospital restaurant sisu tax
python experiments/baselines/prepare_dc_discovery.py --datasets hmdaz hmda_checks --sample 100000
# grid: exact and epsilon in {0.001, 0.005, 0.01, 0.05, 0.1}; arguments: datasets, timeout, heap
experiments/baselines/run_dc_discovery.sh "hospital tax" 6h 8g
# evaluation: mixed-score ranking, top-k in {5, 10, 20, 50, 100}
python experiments/baselines/evaluate_baselines_v3.py profiling \
  --dataset tax --csv baselines/1_profiling/results_dc_discovery/tax_v3_*.csv.gz
```

The inputs used in the paper are in `baselines/1_profiling/trimmed_datasets_v3/`
(`*.csv.gz` plus a manifest with columns, types, sample size, and seed); gunzip them
before running the `.jar`. The discovered DCs are in
`baselines/1_profiling/results_dc_discovery/*_v3_*.csv.gz`, and every `.jar` run
(ok, timeout, or out of memory) is logged in `results/v3_fix/baselines/dc_logs/runs.tsv`.
On HMDA Checks no configuration finished (6 h with 100k rows, 2 h with 50k, a 12 GB
heap with 20k and 10k rows).

**XiYan-SQL** (GPU): the paper reports variant B (`XGenerationLab/XiYanSQL-QwenCoder-32B-2504`,
prompt of `baselines/2_text2sql/xyyan_text2sql_inference.py`, seeds 0-2). The GPU
package, with run instructions, is `results/baselines_v3_gpu_package/` (`RUN.md`,
`run_all_B.sh`); the generated queries are in
`results/baselines_v3_gpu_package/xiyan_run_B/outputs_B/<dataset>/seed<k>/`. Evaluation:

```bash
python experiments/baselines/evaluate_baselines_v3.py xiyan --dataset tax --table data --dialect sqlite \
  --detailed results/baselines_v3_gpu_package/xiyan_run_B/outputs_B/tax/seed*/tax_xiyan_detailed_results.csv
```

Each query is executed as generated (SQLite queries are transpiled to DuckDB with
`sqlglot`). For constraint mapping, `experiments/baselines/sql_rules.py` turns the
`WHERE` clause into Norma rules; queries with no DC reading are scored by execution
equivalence.

## Configuration reference

Runner configs are YAML files (see `experiments/configs/v3_fix/` for the paper):

```yaml
experiment_name: "hmda_e2_tax_gpt6luna_MultiStep"
experiment_type: "mapper_gt_only"
results_dir: "results/v3_fix/e2_mapping"
datasets: [tax]
dataset_options:
  tax:
    gt_file: tax_text2rule_gt.json
    predicate_space: v2
    sample_rows: 1000000
    stream_csv: true
variants:
  - name: "Multi-Step | GPT-6 Luna"
    use_context: false
    mapper:
      implementation: multi_step_refinement   # or "llm" for Single-Prompt
      provider: openrouter
      model: openai/gpt-6-luna
      prompt_file: multi_step_refinement
      max_refinement_iterations: 2
      max_violation_share: 0.9                # rule gate: drop a DC flagging > 90% of the tuples
```

| Environment variable | Default | Effect |
|---|---|---|
| `OPENROUTER_API_KEY` | none | Needed only for new LLM runs. |
| `NORMA_SQL_TIMEOUT_S` | 300 | Time limit of each violation query; a timeout counts as an execution error. |
| `NORMA_SQL_MAX_ROWS` | 20000000 | Maximum result rows of a violation query. |
| `NORMA_LLM_SEED` | empty | Sampling seed, sent only to models that accept it. |
| `NORMA_JSON_OBJECT_MODELS` | empty | Models for which strict `json_schema` output is replaced by `json_object` plus validation. |

## Tests

```bash
python -m unittest discover -s tests
```

## License

Code: MIT (see `LICENSE`). The datasets keep the terms of their public sources
(NYC Open Data, Brazilian open government data, and HMDA public data released by the
CFPB/FFIEC); Hospital and Tax are public benchmarks from prior work.
