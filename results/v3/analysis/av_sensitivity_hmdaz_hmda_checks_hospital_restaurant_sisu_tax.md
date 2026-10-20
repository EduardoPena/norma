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

Excluídos: hmda_checks: date edits S270 + Q022; hospital: dominant rule (96% of the gold-violating tuples); restaurant: dominant rule (99% of the gold-violating tuples); tax: dominant rule (78% of the gold-violating tuples)

|                                        |   ('hmda_checks', 'full') |   ('hmda_checks', 'without_dominant') |   ('hmdaz', 'full') |   ('hospital', 'full') |   ('hospital', 'without_dominant') |   ('restaurant', 'full') |   ('restaurant', 'without_dominant') |   ('sisu', 'full') |   ('tax', 'full') |   ('tax', 'without_dominant') |
|:---------------------------------------|--------------------------:|--------------------------------------:|--------------------:|-----------------------:|-----------------------------------:|-------------------------:|-------------------------------------:|-------------------:|------------------:|------------------------------:|
| ('Multi-Step', 'Claude Sonnet 5.5')    |                      0.1  |                                  1    |                0.71 |                      1 |                                  1 |                     1    |                                    1 |               0.92 |              1    |                          1    |
| ('Multi-Step', 'DeepSeek-V4-Flash')    |                      0.16 |                                  0.77 |                0.01 |                      1 |                                  1 |                     0.02 |                                    1 |               0.42 |              0.99 |                          0.96 |
| ('Multi-Step', 'GPT-6 Luna')           |                      0.1  |                                  1    |                0.55 |                      1 |                                  1 |                     1    |                                    1 |               0.22 |              1    |                          1    |
| ('Multi-Step', 'GPT-6 Sol')            |                      0.1  |                                  1    |                0.6  |                      1 |                                  1 |                     1    |                                    1 |               1    |              1    |                          1    |
| ('Multi-Step', 'Mistral Large 4')      |                      0.16 |                                  0.77 |                0.64 |                      1 |                                  1 |                     1    |                                    1 |               0.99 |              1    |                          1    |
| ('Multi-Step', 'Qwen3.8-2.4T-A95B')    |                      0.16 |                                  0.77 |                0.25 |                      1 |                                  1 |                     1    |                                    1 |               0.99 |              1    |                          1    |
| ('Single Prompt', 'Claude Sonnet 5.5') |                      0.99 |                                  1    |                0.72 |                      1 |                                  1 |                     0.02 |                                    1 |               0.02 |              1    |                          1    |
| ('Single Prompt', 'DeepSeek-V4-Flash') |                      0.99 |                                  0.18 |                0.57 |                      1 |                                  1 |                     0.02 |                                    1 |               0.02 |              1    |                          1    |
| ('Single Prompt', 'GPT-6 Luna')        |                      0.99 |                                  0.77 |                0.57 |                      1 |                                  1 |                     0.02 |                                    1 |               0.01 |              1    |                          1    |
| ('Single Prompt', 'GPT-6 Sol')         |                      0.99 |                                  0.18 |                0.72 |                      1 |                                  1 |                     0.02 |                                    1 |               0.02 |              1    |                          1    |
| ('Single Prompt', 'Mistral Large 4')   |                      0.99 |                                  0.74 |                0.58 |                      1 |                                  1 |                     0.02 |                                    1 |               0.02 |              1    |                          1    |
| ('Single Prompt', 'Qwen3.8-2.4T-A95B') |                      0.99 |                                  1    |                0.57 |                      1 |                                  1 |                     1    |                                    1 |               0.02 |              1    |                          1    |
