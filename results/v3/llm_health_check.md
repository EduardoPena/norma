# OpenRouter LLM health check

Raw attempts: `results/v3/model_health/20261013_212223_144111/attempts.jsonl`. Checks use the experiment code path with strict structured output (LLMRuleClassifier `with_examples` -> CLASSIFICATION schema; LLMMapper `with_examples` -> SENTENCE_RULES schema, rules validated by `normalize_rules`; Multi-Step: each of the 5 step schemas, response validated with jsonschema).

| Paper model | OpenRouter slug | In catalog | Classifier | Mapper (JSON) | Multi-Step (5 schemas) | Latency clf / map / ms (s) | Note |
|---|---|---|---|---|---|---|---|
| GPT-6 Luna | `openai/gpt-6-luna` | True | passed | passed | passed | 2.7 / 3.0 / 15.2 | Kept from round v2 (0.10 / 0.50). No temperature parameter (reasoning model). |
| DeepSeek-V4-Flash | `deepseek/deepseek-v4-flash` | True | passed | passed | passed | 17.5 / 4.7 / 39.8 | Kept from round v2 (0.09 / 0.18). |
| Kimi K2.6 | `moonshotai/kimi-k2.6` | True | passed | passed | failed | 14.1 / 6.3 / 564.0 | Replaces Kimi K3 (0.65 / 3.41 vs 0.73 / 10.00). |
| Qwen3.8-2.4T-A95B | `qwen/qwen3.8-2.4t-a95b` | True | passed | passed | passed | 7.4 / 5.4 / 143.8 | Replaces Qwen3.8-Max-Prime (2.00 / 6.00 vs 4.00 / 12.00). |
| GPT-6 Sol | `openai/gpt-6-sol` | True | passed | passed | passed | 1.8 / 2.6 / 18.8 | Kept from round v2 (2.00 / 10.00). |
| Claude Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | True | passed | passed | passed | 3.1 / 4.6 / 24.3 | Replaces Claude Opus 5.5 (2.00 / 10.00 vs 4.00 / 20.00). |
