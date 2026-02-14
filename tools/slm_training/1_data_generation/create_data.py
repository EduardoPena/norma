
from __future__ import annotations


import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


# Keep your local imports working when running this file directly.
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import get_llm_client, load_classification_data, load_csv_data




# ----------------------------
# Helpers
# ----------------------------
def load_prompts(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_llm_response(response: str, expected_count: int) -> list[str]:
    sentences: list[str] = []
    for raw in response.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        
        # Skip common introductory phrases
        skip_phrases = [
            "here are", "here is", "these are", "this is",
            "generate", "output", "result", "answer",
            "based on", "following"
        ]
        if any(phrase in line.lower()[:30] for phrase in skip_phrases):
            continue

        # Remove numbering, bullets, and markdown
        cleaned = line.lstrip("0123456789.-*) ").strip()
        
        # Remove markdown bold/italic
        cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "")
        
        # Remove quotes
        cleaned = cleaned.strip('"').strip("'")
        
        # Skip if it's just a category label (ends with colon)
        if cleaned.endswith(":") and len(cleaned) < 50:
            continue
        
        # Remove category labels at the start (e.g., "Required field: The...")
        if ":" in cleaned[:40]:  # Only check first 40 chars
            parts = cleaned.split(":", 1)
            if len(parts) == 2 and len(parts[0]) < 30:  # Category label is short
                # Check if first part looks like a category
                category_keywords = ["required", "allowed", "forbidden", "numeric", "conditional",
                                     "cross-row", "uniqueness", "determinacy", "referential", 
                                     "temporal", "description", "explanation", "definition"]
                if any(kw in parts[0].lower() for kw in category_keywords):
                    cleaned = parts[1].strip()
        
        if len(cleaned) > 20:
            sentences.append(cleaned)

        if len(sentences) >= expected_count:
            break

    return sentences


def _prompt_template(prompts: dict[str, Any], key: str) -> str:
    try:
        return prompts[key]["template"]
    except Exception as e:
        raise KeyError(f"Missing prompt template for key='{key}'.") from e


def generate_paraphrases(llm, prompts: dict[str, Any], examples: list[str], is_rule: bool, n_samples: int, pbar=None) -> list[str]:
    key = "paraphrase_rule" if is_rule else "paraphrase_non_rule"
    template = _prompt_template(prompts, key)

    out: list[str] = []
    for example in examples:
        prompt = template.format(example=example, n_samples=n_samples)
        response = llm.complete(prompt)
        out.extend(parse_llm_response(response, n_samples))
        if pbar:
            pbar.update(1)
    return out


def generate_domain_adapted(
    llm,
    prompts: dict[str, Any],
    dataset_name: str,
    dataset_rules: dict[str, list[str]],
    dataset_non_rules: dict[str, list[str]],
    target_csv_df: pd.DataFrame,
    is_rule: bool,
    n_samples: int,
    pbar=None,
) -> list[str]:
    key = "domain_adaptation_rules" if is_rule else "domain_adaptation_non_rules"
    template = _prompt_template(prompts, key)

    # Collect rules from all other datasets
    other_rules: list[str] = []
    for ds, items in dataset_rules.items():
        if ds != dataset_name and items:
            other_rules.extend(items)  # Use ALL rules from other datasets
    
    # Collect non-rules from all other datasets
    other_non_rules: list[str] = []
    for ds, items in dataset_non_rules.items():
        if ds != dataset_name and items:
            other_non_rules.extend(items)  # Use ALL non-rules from other datasets

    other_rules_text = "\n".join(f"- {ex}" for ex in other_rules)
    other_non_rules_text = "\n".join(f"- {ex}" for ex in other_non_rules)
    
    # Get target domain schema
    target_schema = ", ".join(target_csv_df.columns.tolist())
    
    # Get sample rows from target domain
    sample_rows = target_csv_df.head(3)
    target_samples = "\n".join(
        f"Row {i+1}: " + ", ".join(f"{col}={row[col]}" for col in target_csv_df.columns)
        for i, (_, row) in enumerate(sample_rows.iterrows())
    )

    prompt = template.format(
        other_domain_rules=other_rules_text,
        other_domain_non_rules=other_non_rules_text,
        target_schema=target_schema,
        target_samples=target_samples,
        n_samples=n_samples,
    )

    # print(prompt)

    response = llm.complete(prompt)
    if pbar:
        pbar.update(1)
    return parse_llm_response(response, n_samples)


def _first_text_column_sentences(df: pd.DataFrame) -> list[str]:
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return []
    return df[text_cols[0]].astype(str).tolist()


def generate_synthetic_data(
    initial_sentences_df: pd.DataFrame,
    dataframe_dicts: dict[str, pd.DataFrame],
    prompts: dict[str, Any],
    llm,
    generation_modes: list[str],
    samples_generated_per_sentence: int,
    output_dir: Path,
    model_name: str,
    domain_adapt_rules: int = 20,
    domain_adapt_non_rules: int = 80,
) -> pd.DataFrame:
    dataset_rules: dict[str, list[str]] = {}
    dataset_non_rules: dict[str, list[str]] = {}

    for dataset_name in dataframe_dicts:
        ds_df = initial_sentences_df[initial_sentences_df["dataset"]
                                     == dataset_name]
        dataset_rules[dataset_name] = ds_df[ds_df["class"]
                                            == "dq-rule"]["sentence"].tolist()
        dataset_non_rules[dataset_name] = ds_df[ds_df["class"]
                                                == "non-dq-rule"]["sentence"].tolist()

    # Calculate total LLM calls for progress bar
    total_calls = 0
    for dataset_name in dataframe_dicts:
        if dataset_name != 'tax-chu':
            continue
        rules = dataset_rules.get(dataset_name, [])
        non_rules = dataset_non_rules.get(dataset_name, [])
        csv_df = dataframe_dicts[dataset_name]
        csv_sentences = _first_text_column_sentences(csv_df)
        
        for mode in generation_modes:
            if mode == "paraphrase":
                total_calls += len(rules) + len(non_rules)
            elif mode == "domain_adapt":
                total_calls += (1 if not csv_df.empty and rules else 0) + (1 if not csv_df.empty and non_rules else 0)

    all_rows: list[dict[str, str]] = []
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    with tqdm(total=total_calls, desc="Generating sentences") as pbar:
        for dataset_name, csv_df in dataframe_dicts.items():
            if dataset_name != 'tax-chu':
                continue
            rules = dataset_rules.get(dataset_name, [])
            non_rules = dataset_non_rules.get(dataset_name, [])

            for mode in generation_modes:
                mode_rows: list[dict[str, str]] = []
                if mode == "paraphrase":
                    if rules:
                        generated = generate_paraphrases(
                            llm, prompts, rules, is_rule=True, n_samples=samples_generated_per_sentence, pbar=pbar)
                        mode_rows.extend({"dataset": dataset_name, "rule": s, "class": "dq-rule", "mode": mode} for s in generated)
                    if non_rules:
                        generated = generate_paraphrases(
                            llm, prompts, non_rules, is_rule=False, n_samples=samples_generated_per_sentence, pbar=pbar)
                        mode_rows.extend({"dataset": dataset_name, "rule": s, "class": "non-dq-rule", "mode": mode} for s in generated)

                elif mode == "domain_adapt":
                    # Generate rules (one call with all rules from other datasets)
                    if not csv_df.empty and rules:
                        generated = generate_domain_adapted(
                            llm,
                            prompts,
                            dataset_name,
                            dataset_rules,
                            dataset_non_rules,
                            csv_df,
                            is_rule=True,
                            n_samples=domain_adapt_rules,
                            pbar=pbar,
                        )
                        mode_rows.extend({"dataset": dataset_name, "rule": s, "class": "dq-rule", "mode": mode} for s in generated)
                    
                    # Generate non-rules (one call with all non-rules from other datasets)
                    if not csv_df.empty and non_rules:
                        generated = generate_domain_adapted(
                            llm,
                            prompts,
                            dataset_name,
                            dataset_rules,
                            dataset_non_rules,
                            csv_df,
                            is_rule=False,
                            n_samples=domain_adapt_non_rules,
                            pbar=pbar,
                        )
                        mode_rows.extend({"dataset": dataset_name, "rule": s, "class": "non-dq-rule", "mode": mode} for s in generated)

                else:
                    raise ValueError(f"Unknown generation mode: {mode}")
                
                # Save incrementally after each dataset+mode
                if mode_rows:
                    mode_df = pd.DataFrame(mode_rows)
                    file_name = f"synthetic_sentences_{dataset_name}_{mode}_{model_name}_{timestamp}.csv"
                    file_path = output_dir / file_name
                    mode_df.to_csv(file_path, index=False)
                    print(f"\nSaved {len(mode_df)} sentences to {file_name}")
                    all_rows.extend(mode_rows)

    return pd.DataFrame(all_rows)


def main():

    gt_file_paths = [
        '/Users/pena/nl2c_projetcts/text2rules/data/restaurant-inspections/restaurant-inspections_text2rule_gt.json',
        '/Users/pena/nl2c_projetcts/text2rules/data/hospital-holo/hospital-holo_text2rule_gt.json',
        '/Users/pena/nl2c_projetcts/text2rules/data/sisu/sisu_text2rule_gt.json',
        '/Users/pena/nl2c_projetcts/text2rules/data/tax-chu/tax-chu_text2rule_gt.json',
    ]

    initial_sentences_df = load_classification_data(gt_file_paths)

    # Save original data if it doesn't exist
    original_data_dir = Path(__file__).parent.parent.parent / 'data' / 'sentence_clasification_data'
    original_data_dir.mkdir(parents=True, exist_ok=True)
    original_data_file = original_data_dir / 'original_sentence_classification_data.csv'
    if not original_data_file.exists():
        initial_sentences_df.to_csv(original_data_file, index=False)
        print(f"Original data saved to: {original_data_file}")

    csv_file_paths = [file.replace('_text2rule_gt.json', '.csv')
                      for file in gt_file_paths]

    dataframe_dicts = load_csv_data(csv_file_paths, sample_size=5)

    prompts_file = "/Users/pena/nl2c_projetcts/text2rules/src/data_generation/data_generation_prompts.yaml"

    prompts = load_prompts(prompts_file)

    model = "gpt-5-mini"
    model = "gpt-5-mini"
    # model = "mistral-small-2506"
    llm = get_llm_client(model)

    # generation_modes = ["paraphrase", "domain_adapt"]
    generation_modes = ["paraphrase"]

    # Setup output directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_df = generate_synthetic_data(
        initial_sentences_df=initial_sentences_df,
        dataframe_dicts=dataframe_dicts,
        prompts=prompts,
        llm=llm,
        generation_modes=generation_modes,
        samples_generated_per_sentence=3,
        output_dir=output_dir,
        model_name=model,
        domain_adapt_rules=20,      # Generate 20 new rules per dataset
        domain_adapt_non_rules=80,  # Generate 80 new non-rules per dataset
    )

    print(f"\nTotal synthetic sentences generated: {len(synthetic_df)}")

    return 0


if __name__ == "__main__":
    main()
