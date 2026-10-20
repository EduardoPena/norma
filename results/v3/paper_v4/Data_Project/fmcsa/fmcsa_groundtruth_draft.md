# FMCSA Company Census File — Ground-Truth de Constraints (rascunho)

Fonte normativa: **49 CFR Part 390** (Federal Motor Carrier Safety Regulations, Subpart B — atualização biênal do MCS-150) e **49 U.S.C. §13902** / **49 CFR Part 365** (autoridade de operação for-hire). Prosa jurídica real: definições ("Motor carrier means..."), cronograma de exceções por dígito do número USDOT, consequências de descumprimento ("subject to... deactivation of its USDOT Number").

Dado: **FMCSA Company Census File**, público, sem cadastro, CSV via Socrata (`data.transportation.gov`, dataset `az4n-8mr2`). **4.479.438 registros** (2.233.109 com `STATUS_CODE='A'`, ativos), **147 colunas**.

## Regras e violações medidas

| ID | Kind | Categoria | Regra | Violações | % |
|---|---|---|---|--:|--:|
| R_mcs150_stale | validity | comparação | Transportadora ativa: MCS-150 desatualizado há >24 meses (49 CFR 390.19) | 973.568 | 43,60% (dos ativos) |
| R_mcs150_missing | validity | null | Transportadora ativa sem nenhuma data de MCS-150 registrada | 253.170 | 11,34% (dos ativos) |
| R_forhire_no_docket | validity* | null | Transportadora ativa "AUTHORIZED FOR HIRE" sem número de autoridade (docket/MC) em nenhum dos 3 campos | 592.981 | 26,55% (dos ativos) |
| R_status_dom | validity | enum | `STATUS_CODE` ∈ {A,I,P} | 0 | 0,00% |
| R_carrieroperation_dom | validity | enum | `CARRIER_OPERATION` ∈ {A,B,C} | 0 | 0,00% |
| R_hmind_dom | validity | enum | `HM_Ind` ∈ {Y,N} | 0 | 0,00% |
| R_priorrevoke_dom | validity | enum | `PRIOR_REVOKE_FLAG` ∈ {Y,N} | 0 | 0,00% |
| R_safetyrating_dom | validity | enum | `SAFETY_RATING` ∈ {S,C,U} quando presente | 0 | 0,00% |
| R_truckunits_nonneg | validity | range | `TRUCK_UNITS` ≥ 0 | 0 | 0,00% |
| R_powerunits_nonneg | validity | range | `POWER_UNITS` ≥ 0 | 0 | 0,00% |
| R_busunits_nonneg | validity | range | `BUS_UNITS` ≥ 0 | 0 | 0,00% |
| R_totaldrivers_nonneg | validity | range | `TOTAL_DRIVERS` ≥ 0 | 0 | 0,00% |
| R_totalcdl_nonneg | validity | range | `TOTAL_CDL` ≥ 0 | 15 | 0,00% |
| Q_cdl_le_drivers | quality | comparação | `TOTAL_CDL` não deveria exceder `TOTAL_DRIVERS` | 6.355 | 0,14% (do total) |
| Q_intrastate_le_total | quality | comparação | `TOTAL_INTRASTATE_DRIVERS` não deveria exceder `TOTAL_DRIVERS` | 0 | 0,00% |
| Q_fleet_zero_but_active | quality | condicional | Transportadora ativa for-hire relata frota zero (truck+bus+power=0) | 205.824 | 9,22% (dos ativos) |

Script completo em `run_edits.py`; resultado bruto em `edit_results.csv`. **16 regras** (13 validity + 3 quality).

## Achado principal

**43,60% das transportadoras marcadas como "ativas" não atualizam o MCS-150 há mais de 24 meses** — a exigência está escrita literalmente no texto ("Every 24 months, according to the following schedule", 49 CFR 390.19), sem depender de norma técnica externa. A distribuição por faixa de atraso mostra a cauda: **433.578 transportadoras "ativas" (19,4% dos ativos) estão com o MCS-150 desatualizado há mais de 10 anos** — um padrão de "registro zumbi" parecido com o que já tínhamos visto no AACT (estudos clínicos "recrutando" há anos sem atualização).

## Ressalvas a declarar no paper (transparência)

1. **`R_forhire_no_docket` marcado com "validity*"**: o campo `CLASSDEF` é uma descrição operacional auto-reportada no próprio MCS-150 ("AUTHORIZED FOR HIRE"), não necessariamente uma confirmação em tempo real de que a autoridade está ativa hoje — o status formal de autoridade (docket/MC number com `DOCKET1_STATUS_CODE`) é o que efetivamente comprova isso. Um transportador pode ter marcado essa categoria e estar em processo de obter a autoridade, ou o campo pode estar desatualizado. Tratar como forte indício de não-conformidade, não como confirmação jurídica definitiva — mesma cautela que já aplicamos ao Q001 do HMDA.
2. **`R_mcs150_stale` usa um limiar simplificado de 24 meses corridos**, não o cronograma exato de meses escalonados por dígito final do número USDOT que o §390.19 realmente especifica (ex.: transportadoras com USDOT terminado em 1 atualizam em janeiro, terminado em 2 em fevereiro, etc., alternando anos pares/ímpares conforme o penúltimo dígito). O limiar de 24 meses é uma leitura direta e defensável do texto ("every 24 months"), mas não captura a granularidade completa do cronograma escalonado.
3. **Escala**: com 4,48M linhas / 2,23M ativas, este é de longe o maior dataset entre os candidatos (e maior que qualquer um dos 4 já no paper) — pode ser necessário recortar uma amostra ou período pra manter os experimentos práticos (mesmo tratamento já dado a SISU/Restaurant nos baselines de profiling, que usam só o subconjunto de colunas do gold).

## Cobertura de categoria de predicado

null (2), pattern (0), enum (5), range (5), comparação (3), condicional (1) — falta pattern/formato; poderia ser complementado com um check de formato em `PHY_ZIP` ou `PHONE` se quisermos mais diversidade.

## Próximos passos possíveis

- Anotar as sentenças em prosa correspondentes em 49 CFR Part 390 Subpart B (para o classificador) — o texto tem definições, cronograma, exceções e prosa não-normativa suficiente pra sustentar a tarefa RQ1.
- Decidir se recorta o dataset (ex.: só um estado, ou só transportadoras ativas) para reduzir escala.
- Integrar como um dos "+2 datasets de baixo esforço" na Tabela 1 (`text/evaluation_base.tex`).
