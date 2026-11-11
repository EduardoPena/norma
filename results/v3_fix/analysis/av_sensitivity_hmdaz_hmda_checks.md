# Sensibilidade da violação agregada (E2, rodada v3)

AV sensitivity to the dominant gold rules (round v3, E2; no LLM calls).

The aggregate violation F1 (AV) is computed over the union of violating tuples, so
a single gold rule can dominate it. For each dataset this script re-evaluates the
stored E2 outputs on the GT without the dominant rule(s): on HMDA_CHECKS the two
date-range edits (S270, Q022: uniform synthetic-looking dates, 97% / 80% of rows);
elsewhere the gold rule with the most violations when it flags >= 75% of the
gold-violating tuples (max_gold_rule_violation_share). Exact CE/CE_equiv are reported on the same reduced GT.

Usage: .venv/bin/python results/v3/analysis/av_sensitivity.py [dataset ...]
Writes results/v3/analysis/av_sensitivity_<datasets>.csv; with no argument (all
datasets) also av_sensitivity.md.

Excluídos: hmda_checks: date edits S270 + Q022

|                                        |   ('hmda_checks', 'full') |   ('hmda_checks', 'without_dominant') |   ('hmdaz', 'full') |
|:---------------------------------------|--------------------------:|--------------------------------------:|--------------------:|
| ('Multi-Step', 'Claude Sonnet 5.5')    |                      0.67 |                                  0.19 |                0.71 |
| ('Multi-Step', 'DeepSeek-V4-Flash')    |                      0.89 |                                  0.12 |                0.23 |
| ('Multi-Step', 'GPT-6 Luna')           |                      0.74 |                                  0.16 |                0.74 |
| ('Multi-Step', 'GPT-6 Sol')            |                      0.9  |                                  1    |                0.59 |
| ('Multi-Step', 'Mistral Large 4')      |                      0.76 |                                  0.16 |                0.24 |
| ('Multi-Step', 'Qwen3.8-2.4T-A95B')    |                      0.1  |                                  0.98 |                0    |
| ('Single Prompt', 'Claude Sonnet 5.5') |                      0.99 |                                  1    |                0.72 |
| ('Single Prompt', 'DeepSeek-V4-Flash') |                      0.99 |                                  0.18 |                0.57 |
| ('Single Prompt', 'GPT-6 Luna')        |                      0.99 |                                  0.77 |                0.57 |
| ('Single Prompt', 'GPT-6 Sol')         |                      0.99 |                                  0.18 |                0.72 |
| ('Single Prompt', 'Mistral Large 4')   |                      0.99 |                                  0.74 |                0.58 |
| ('Single Prompt', 'Qwen3.8-2.4T-A95B') |                      0.99 |                                  1    |                0.57 |
