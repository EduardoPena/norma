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

Excluídos: hospital: dominant rule (96% of the gold-violating tuples); restaurant: dominant rule (99% of the gold-violating tuples); tax: dominant rule (78% of the gold-violating tuples)

|                                        |   ('hospital', 'full') |   ('hospital', 'without_dominant') |   ('restaurant', 'full') |   ('restaurant', 'without_dominant') |   ('sisu', 'full') |   ('tax', 'full') |   ('tax', 'without_dominant') |
|:---------------------------------------|-----------------------:|-----------------------------------:|-------------------------:|-------------------------------------:|-------------------:|------------------:|------------------------------:|
| ('Multi-Step', 'Claude Sonnet 5.5')    |                   1    |                               1    |                     1    |                                 1    |               0.92 |              1    |                          1    |
| ('Multi-Step', 'DeepSeek-V4-Flash')    |                   1    |                               1    |                     0.02 |                                 1    |               0.42 |              0.99 |                          0.96 |
| ('Multi-Step', 'GPT-6 Luna')           |                   1    |                               1    |                     1    |                                 1    |               0.22 |              1    |                          1    |
| ('Multi-Step', 'GPT-6 Sol')            |                   1    |                               1    |                     1    |                                 1    |               1    |              1    |                          1    |
| ('Multi-Step', 'MiniMax-M3')           |                   0.99 |                               0.83 |                     0.02 |                                 0.97 |               0.63 |              1    |                          1    |
| ('Multi-Step', 'Qwen3.8-2.4T-A95B')    |                   1    |                               1    |                     1    |                                 1    |               0.99 |              1    |                          1    |
| ('Single Prompt', 'Claude Sonnet 5.5') |                   1    |                               1    |                     0.02 |                                 1    |               0.02 |              1    |                          1    |
| ('Single Prompt', 'DeepSeek-V4-Flash') |                   1    |                               1    |                     0.02 |                                 1    |               0.02 |              1    |                          1    |
| ('Single Prompt', 'GPT-6 Luna')        |                   1    |                               1    |                     0.02 |                                 1    |               0.01 |              1    |                          1    |
| ('Single Prompt', 'GPT-6 Sol')         |                   1    |                               1    |                     0.02 |                                 1    |               0.02 |              1    |                          1    |
| ('Single Prompt', 'MiniMax-M3')        |                   0.36 |                               0.73 |                     0    |                                 0    |               0.01 |              0    |                          0    |
| ('Single Prompt', 'Qwen3.8-2.4T-A95B') |                   1    |                               1    |                     1    |                                 1    |               0.02 |              1    |                          1    |
