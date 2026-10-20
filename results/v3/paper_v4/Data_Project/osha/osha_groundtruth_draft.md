# OSHA ITA (Form 300A) — Ground-Truth de Constraints (rascunho)

Fonte: OSHA, *Injury Tracking Application (ITA) File Upload Data Dictionary: Establishment and Form 300A* (documento oficial, 8 páginas, `osha_ita-estab-and-summary-csv-documentation_revised.pdf`, revisado 2023-10-17). É o texto que a própria OSHA publica para quem submete os dados — equivalente em espírito ao FIG do HMDA, mas mais curto e direto (regras já em formato "must/should" campo a campo + uma seção final "Additional Data Validations" com regras cross-field).

Dado: **ITA 300A Summary Data**, ano-calendário 2023 (arquivo `ITA_300A_Summary_Data_2023_through_12-31-2024.zip`, público em osha.gov, sem cadastro). **394.234 estabelecimentos**, 32 colunas.

Notação: **validity** = regra "must" (bloqueante no texto oficial); **quality** = regra "should" (aviso, não bloqueante). Diferente do HMDA, aqui a distinção must/should já vem no PRÓPRIO texto normativo, não é uma inferência nossa.

## Resumo por categoria

| Categoria | validity | quality | total |
|---|--:|--:|--:|
| Null/required | 1 | 0 | 1 |
| Pattern/format | 4 | 0 | 4 |
| Domain/enumeração | 3 | 0 | 3 |
| Range (numérico) | 14 | 1 | 15 |
| Comparação entre colunas | 2 | 2 | 4 |
| Condicional cross-field | 4 | 0 | 4 |
| Aritmética (soma) | 1 | 0 | 1 |
| **TOTAL** | **29** | **3** | **32** |

## Regras e violações medidas (dado real, CY2023, n=394.234)

| ID | Kind | Categoria | Regra (do texto oficial) | Violações | % |
|---|---|---|---|--:|--:|
| R_ein_required | validity | null | `ein_number` é campo obrigatório | 40.543 | 10,28% |
| R_ein_fmt | validity | pattern | `ein_number` deve ter 9 dígitos, sem hífen | 28.257 | 7,17% |
| R_state_fmt | validity | pattern | `state` deve ser sigla de 2 letras | 0 | 0,00% |
| R_zip_fmt | validity | pattern | `zip` deve ter 5 ou 9 dígitos | 14.418 | 3,66% |
| R_naics_fmt | validity | pattern | `naics_code` deve ter 6 dígitos | 0 | 0,00% |
| R_size_dom | validity | enum | `size` ∈ {1, 21, 22, 3} | 62.305 | 15,80% |
| R_estype_dom | validity | enum | `establishment_type` ∈ {1,2,3} (se presente) | 5 | 0,00% |
| R_noii_dom | validity | enum | `no_injuries_illnesses` ∈ {1,2} | 0 | 0,00% |
| R_avgemp_pos | validity | range | `annual_average_employees` > 0 | 2.121 | 0,54% |
| R_hours_pos | validity | range | `total_hours_worked` > 0 | 2.051 | 0,52% |
| R_{G..M6}_nonneg (×12) | validity | range | cada total (deaths, dafw/djtr/other cases e days, M1–M6) ≥ 0 | 0–5 cada | ~0,00% |
| R_hours_per_emp_hard | validity | comparison | `hours_worked / avg_employees` < 8760 | 2.255 | 0,57% |
| Q_hours_per_emp_soft | quality | comparison | `hours_worked / avg_employees` deveria ser > 500 | 13.597 | 3,45% |
| Q_avgemp_soft_max | quality | range | `annual_average_employees` deveria ser < 25.000 | 108 | 0,03% |
| R_sum_types_eq_cases | validity | arithmetic | soma(M1..M6) = soma(campos G–J) | 0 | 0,00% |
| Q_avgemp_gt_cases | quality | comparison | `annual_average_employees` deveria ser > total de casos (G–J) | 2.730 | 0,69% |
| R_dafwcases_implies_days | validity | conditional | se `dafw_cases`>0 então `dafw_days`>0 | 213 | 0,05% |
| R_dafwdays_implies_cases | validity | conditional | se `dafw_days`>0 então `dafw_cases`>0 | 281 | 0,07% |
| R_dafwdays_ge_cases | validity | comparison | `dafw_days` ≥ `dafw_cases` | 228 | 0,06% |
| R_djtrcases_implies_days | validity | conditional | se `djtr_cases`>0 então `djtr_days`>0 | 728 | 0,18% |
| R_djtrdays_implies_cases_or_dafw | validity | conditional | se `djtr_days`>0 então (`dafw_cases`>0 OU `djtr_cases`>0) | 35 | 0,01% |

Script completo em `run_edits.py`; resultado bruto em `edit_results.csv`.

## Por que esse dataset é mais forte que o HMDA para a hipótese do paper

No HMDA, o quadrante "regra ancorada no texto oficial" e o quadrante "regra de fato violada no dado público" quase não se sobrepunham (as regras *validity*, ancoradas, tinham ~0 violação; a única violação grande, Q001, era uma heurística sem base no texto legal). **Aqui os dois quadrantes se sobrepõem de verdade**: `R_size_dom` (15,80%), `R_ein_fmt` (7,17%), `R_ein_required` (10,28%) e `R_hours_per_emp_hard` (0,57%) são todas regras **validity** — "must", bloqueantes, tiradas literalmente do texto oficial da OSHA — e mesmo assim aparecem violadas em escala no arquivo que a própria OSHA publica. Ou seja: a OSHA documenta a regra, diz que é obrigatória, e o dado que ela mesma libera publicamente a desobedece. É o caso mais limpo de "silently violates" que já produzimos.

As regras condicionais cross-field (`R_dafwcases_implies_days` etc.), embora com violação percentual pequena (0,01–0,18%), valem a pena manter porque cobrem a categoria de predicado condicional/CFD com números pequenos mas reais — mostra que mesmo regras de consistência interna, presumivelmente checadas no upload, escapam.

## Ressalvas a declarar no paper (transparência, como pedido em R1-O10)

1. **`R_ein_fmt` (7,17%) tem uma causa técnica conhecida e documentada pela própria OSHA**: o FAQ do data dictionary avisa que ferramentas como Excel derrubam zeros à esquerda de campos como EIN/zip ao editar CSV. Uma fração dos "EIN com menos de 9 dígitos" (maioria em 8 dígitos, cauda até 1 dígito) provavelmente é exatamente isso — não um erro do estabelecimento que reportou, mas um artefato de como o arquivo foi preparado/publicado. A violação é real no arquivo liberado (viola o formato documentado), mas a causa-raiz deve ser atribuída com cuidado no texto do paper.
2. **`R_zip_fmt` tem o mesmo tipo de artefato**: uma amostra de valores reprovados mostra strings tipo `271E6`, `044E7` — notação científica típica de planilha (Excel convertendo número longo em `2.71E+6`), sinal de que o arquivo publicado passou por uma etapa de edição em spreadsheet que corrompeu alguns ZIPs.
3. **`R_size_dom` (15,80%) não é ruído — é um achado genuíno**: o valor não-documentado mais comum é `size=2` (62.305 casos), que não existe no esquema atual {1,21,22,3}. O changelog do próprio documento (2023-10-17: "Updated size class categories") indica que o esquema de códigos mudou; `size=2` é plausivelmente um código de um esquema anterior (provavelmente "20–249 empregados" antes de ser desmembrado em 21/22) que estabelecimentos e/ou sistemas legados continuam usando. Isso é uma violação de "dataset publicado não bate com a especificação vigente que o rege" ainda mais direta do que o esperado — vale a pena mencionar essa leitura no texto.
4. `R_ein_required`: o próprio dicionário marca `ein_number` como `Required: Yes`, mas 10,28% dos registros o têm nulo — provavelmente porque o formulário web/CSV aceita submissão sem o campo apesar do "Required" nominal (o sistema declara requerido mas não impõe hard-block). Vale mencionar essa diferença entre "documentado como obrigatório" e "de fato enforced".

## Próximos passos possíveis

- Anotar as sentenças em prosa correspondentes no PDF (para o classificador de sentenças) — o documento já está estruturado em tabela, então "sentença" aqui é cada linha/bullet da tabela Data Dictionary + os 6 bullets da seção "Additional Data Validations" (fácil de recortar).
- Escolher se inclui os 12 rules `*_nonneg` individualmente na Tabela 1 do paper (dá ~29 validity + 3 quality = 32) ou se consolida em poucas linhas para não inflar artificialmente a contagem de "constraints" com regras estruturalmente idênticas.
- Decidir se usa o ano 2023 (usado aqui) ou soma vários anos para robustez (dado está disponível 2016–2025).
- Integrar como um dos "+2 datasets de baixo esforço" na Tabela 1 (`text/evaluation_base.tex`).
