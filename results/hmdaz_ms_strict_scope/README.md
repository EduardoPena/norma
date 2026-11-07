# HMDAZ, E2 Multi-Step, Claude Sonnet 5.5, prompt `strict_scope` (06/11/2026)

- **O que muda:** só o prompt, `prompts/multi_step_refinement_strict_scope.yaml`. É o `multi_step_refinement.yaml` com duas instruções genéricas acrescentadas em `select_predicates`, `construct_constraint` e `refine_constraint`:
  1. comparações estritas e não estritas não são equivalentes (`>` × `>=`, `<` × `<=`);
  2. as condições de aplicabilidade enunciadas na sentença também viram predicados.
- **Igual à execução oficial** (`results/v3_fix/e2_mapping/hmda_e2_hmdaz_sonnet55_multistep_20261021_192746`, uma única execução): espaço v2, 1M linhas, `max_violation_share` 0,9, 2 iterações de refinamento, `max_tokens` 32768.
- **Configuração:** `experiments/configs/hmdaz_ms_strict_scope/` (`plan_reps.json`, `*_rep1.yaml`, `*_rep2.yaml`) e `experiments/configs/hmdaz_ms_strict_scope_run.yaml`.
- **Execução:** 2 repetições (`e2_mapping/*rep1*`, `*rep2*`), 172 s e 179 s, US$ 2,15 no total, sem falhas de saída nem erros de SQL.
- **`superseded_hmda_examples/`:** primeira execução (17h43), descartada. A versão do prompt usada nela tinha exemplos tirados do HMDAZ ("first lien", "consumer's principal dwelling", "loan under $50,000"), o que a torna inválida para medir uma instrução genérica. O arquivo `multi_step_refinement_strict_scope_hmda_examples.yaml` reconstrói aquela versão (difere da atual só nesses dois exemplos).

## Resultado agregado

| Métrica | Oficial | rep1 | rep2 |
|---|---|---|---|
| CM (exato) F1 | 0,21 | 0,22 | 0,11 |
| CM≡ F1 | 0,35 | 0,35 | 0,35 |
| AV P / R / F1 | 0,55 / 1,00 / 0,71 | 0,57 / 1,00 / 0,73 | 0,55 / 1,00 / 0,71 |
| F1 médio por regra | 0,64 | 0,65 | 0,65 |
| Regras com execução idêntica (de 8) | 4 | 5 | 5 |

O CM exato oscila só por causa da forma: código contra rótulo (`hoepa_status = 1` × `hoepa_status_name = 'HOEPA loan'`) nos ids 15, 16 e 17, que são logicamente equivalentes. O CM≡ é o mesmo nas três execuções.

## Por regra (ids do GT)

- **Id 3:** nas duas repetições, a DC passou a incluir `occupancy_type = 1` ("secured by the consumer's principal dwelling"), e as violações ficaram idênticas às do gold (F1 0,92 → 1,00). É o único efeito consistente da instrução de aplicabilidade.
- **Id 13:** a rep2 incluiu a residência principal; a rep1, não. Nas duas, o `OR` do "lesser of 8 percent … or $1,000" foi mantido, como na oficial.
- **Id 1:** as duas trocam `derived_dwelling_category LIKE` por `construction_method`. A rep1 omite o ramo do segundo grau de garantia (subordinate lien). Nenhuma inclui "consumer credit" nem a residência principal.
- **Id 4:** 136 → 74 tuplas, mas não por aplicabilidade. A DC usa `hoepa_status_name != 'HOEPA loan'` em vez de `hoepa_status != 1`, e o rótulo é nulo para o código 3, o que exclui essas linhas. A residência principal e "consumer credit" continuam faltando.
- **Ids 5 e 19:** sem mudança. O id 5 não é mapeado; o id 19 não enuncia condições de aplicabilidade na sentença.
- **"Consumer credit" (`business_or_commercial_purpose = 2`):** nunca aparece, em nenhuma execução, embora as sentenças dos ids 1, 3, 4 e 13 digam "consumer credit transaction".
- **Operadores:** nenhuma execução troca `>` por `>=`; todos os limites coincidem com o gold. A instrução 1 não teve efeito observável neste dataset.
