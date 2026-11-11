# E3 with Claude Sonnet 5.5 as mapper (version_fix)

- 2026-10-21: after the Multi-Step re-run (telemetry 20261021_181933_850838), the paper criterion selects Claude Sonnet 5.5 as the E3 mapper (mean CE/AV 0.776; GPT-6 Luna 0.771; GPT-6 Sol 0.720). The E3 run with GPT-6 Sol was moved to results/v3_fix/superseded/e3_end2end_gpt6sol. New telemetry directory because the E3 configs changed (resume refuses changed configs).
