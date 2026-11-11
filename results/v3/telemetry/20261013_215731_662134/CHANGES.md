# Execution changes during round v3

- 2026-10-13: Kimi K2.6 replaced by MiniMax-M3 (authors' request) after the
  health check showed whitespace loops under strict json_schema decoding.
  The orchestrator was stopped while E1 jobs of GPT-6 Luna / DeepSeek-V4-Flash
  were running (those jobs finished normally); configs were regenerated
  (experiments/configs/v3, manifest experiments/configs/v3_models.yaml) and the
  run was resumed with --resume. No Kimi job had started. MiniMax-M3 passed the
  health check (results/v3/model_health, candidates GLM-5.3: failed
  select_predicates; Kimi K3: passed but 13.5 USD/M output tokens).
- 2026-10-14: E1+E2 finished (108/108 passed, US$ 51.63). E3 components chosen with the paper criteria (see experiments/configs/v3_models.yaml end2end): classifier Qwen3.8-2.4T-A95B + Examples, mapper GPT-6 Sol Multi-Step. E3 plan experiments/configs/v3/hmda_plan_e3.json resumed into this telemetry directory.
- 2026-10-14 (evening): the run stopped on OpenRouter 402 (account credits exhausted: US$ 945.27 of 950 used). E3 hmdaz MultiStep and C_MS passed; hmda_checks_MultiStep FAILED mid-run (402 on some sentences) and must be re-run; its partial result directory is superseded. The other 9 E3 jobs had not started.
- 2026-10-14 (night): credits added by the authors (balance US$ 201.71). summary.json stop_reason (402) cleared by hand so the run can resume. MiniMax-M3 replaced by Mistral Large 4 (see experiments/configs/v3_models.yaml); MiniMax E1/E2 results moved to results/v3/superseded. Resuming the E1+E2 plan (only the Mistral Large 4 jobs are pending); E3 will follow after re-applying the selection criteria.
- 2026-10-15: Mistral Large 4 E1+E2 finished (18/18 passed). E3 components re-checked with the six final models: unchanged. E3 plan resumed (hmdaz jobs already passed; hmda_checks_MultiStep re-run from scratch).
