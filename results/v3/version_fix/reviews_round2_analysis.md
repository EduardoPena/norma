# Análise das Reviews — Round 2 (Paper #473, "Text-to-Constraints")

**Veredito:** R1 Reject · R2 Major Revision · R3 Reject

| Reviewer | Nota | Novidade reconhecida | Importância |
|---|---|---|---|
| R1 | Reject | DI&DS — novo desafio de dados de aplicação real | "SIGMOD attendees will learn something interesting" |
| R2 | Major Revision | Novel solution | "likely to influence other research in the community" |
| R3 | Reject | Novel problem | "No impact expected" |

Os três concordam que **o problema é relevante e novo**. A divergência está em **se a solução tem profundidade técnica suficiente** para justificar aceitação. Esse é o eixo em que a revisão precisa se concentrar — não é preciso reargumentar a motivação, é preciso reforçar o que há de contribuição técnica além de orquestração de prompts.

---

## 1. Crítica central, repetida pelos três revisores

Este é o ponto de maior alavancagem — os três convergem no mesmo diagnóstico com vocabulário diferente:

- **R1 (O5, O7):** descrição "high level"; metodologia "light on details"; não fica claro o quanto o método generaliza além do exemplo apresentado.
- **R2 (O2, O3):** a contribuição "leans heavily on prompt design"; pede uma subseção explícita separando o que é adaptação de práticas de Text-to-SQL (RESDSQL, DIN-SQL) do que é genuinamente novo.
- **R3 (O2):** "the entire solution is based on LLM prompting which greatly narrows the technical contribution."

**Ação:** subseção dedicada de novidade metodológica (candidata natural: `mapping.tex`), delineando explicitamente componentes herdados/adaptados vs. mecanismos próprios.

---

## 2. Formalismo de constraints e validação

- **R3-O7:** afirmação do paper de que DCs tradicionais não cobrem null/literais está **factualmente incorreta** — DCs já cobrem isso; só ranges/enums/padrões exigem extensão real. Correção de baixo custo, remove munição fácil.
- **R3-O6:** a frase "SQL é um moving target para extração" é confusa e parece contradizer o restante do paper (que expressa a DC final como SQL). Precisa reescrita.
- **R1-O2:** o "challenge #3" da introdução não aterrissa em nada concreto — "que outra linguagem de constraint haveria?"
- **R1-O6:** por que validar via SQL e não via métodos de satisfação de DC já existentes na literatura?

---

## 3. Setup de avaliação — datasets e viés

Este é o segundo ponto de maior impacto, e já está em atendimento via o dataset HMDA em preparação.

- **R3-O3:** para Tax e Hospital, o "texto regulatório" usado é o próprio paper de origem do dataset, que já enuncia as regras esperadas — setup "contrived". SISU e Restaurant são mais realistas.
- **R2-O6.1:** Hospital e Tax são sintéticos; Restaurant é curto (33 sentenças); só SISU reflete de fato o cenário real-world alvo.
- **R2-O6.2:** as sentenças de Tax/Hospital são geradas por LLM (mesmo usando um backbone diferente no avaliador), o que infla os scores (Hospital F1 = 1.0). O sinal mais confiável está em SISU/Restaurant (F1 ≈ 0.5), e o paper não deixa essa assimetria explícita.
- **R1-O10:** datasets pequenos em número de sentenças; sem análise de redundância/similaridade entre sentenças; não fica claro como o gold foi computado.

**Ação em andamento:** 5º dataset real (HMDA/Regulation C), com gold derivado diretamente da especificação formal do próprio regulador (FIG), não do paper de origem nem gerado por LLM. Ao integrar, reconhecer explicitamente no texto a assimetria Tax/Hospital vs. SISU/Restaurant/HMDA que o R2 pede em O6.2 — admitir com transparência é mais forte do que deixar implícito.

---

## 4. Traceability vs. uso de LLM — inconsistência interna a resolver

- **R1-O4:** contesta a alegação de que extração baseada em LLM "não é traceable" — sugere procedimento iterativo de sumarização antes do mapeamento.
- **R1-O7:** a Seção 4 usa LLM para extração de sentenças, mas seções anteriores argumentam contra isso por falta de traceability — contradição interna a resolver com um parágrafo de reconciliação explícito.

---

## 5. Related work e posicionamento

- **R1-O1:** introdução não discute REEs (predicados com ML) nem como métodos de descoberta de DC poderiam ser estendidos; faltam referências de related work já na intro.
- **R2-O3:** overlap técnico com Text-to-SQL (RESDSQL, DIN-SQL) não é delineado.
- **R3-O1:** afirmação factualmente incorreta — descoberta *aproximada* de DCs (VLDB 2019) já captura regras que não valem plenamente nos dados; a alegação do paper de que trabalhos de descoberta só capturam o que os dados satisfazem precisa ser corrigida.

---

## 6. Escopo e expressividade

- **R1-O8:** cross-tuple constraints (relação com dependências diferenciais) mencionadas de leve, sem explicar descoberta/validação.
- **R2-O4:** exclusão de agregações e aritmética temporal vira gargalo prático (ex.: coluna "dias até próxima inspeção" derivada manualmente no Restaurant); falta análise de cobertura do vocabulário de predicados contra os gold constraints.
- **R1-O10/O12:** se Norma mapeia para constraints mais amplas (ordering, seleção condicional), baselines além de algoritmos de DC deveriam ser avaliados.

---

## 7. Qualidade de escrita

- **R3-O5:** linguagem vaga ("regularities" sem definição) e termos usados antes de serem introduzidos (ex.: "mapper" citado antes de ser definido). Correções pontuais, mas sinalizam descuido se ignoradas.

---

## Priorização sugerida

1. Subseção de novidade metodológica (`mapping.tex`) — responde R1-O5/O7, R2-O2/O3, R3-O2 simultaneamente.
2. Corrigir as duas alegações factuais incorretas (R3-O1, R3-O7) — baixo custo, alto retorno.
3. Reescrever a passagem "SQL moving target" (R3-O6) e o challenge #3 da intro (R1-O2).
4. Resolver a contradição traceability vs. uso de LLM na Seção 4 (R1-O4/O7).
5. Integrar dataset HMDA + reconhecer explicitamente a assimetria Tax/Hospital vs. SISU/Restaurant (R2-O6.1/O6.2, R3-O3).
6. Cobertura do vocabulário de predicados (R2-O4) e ajustes de escrita (R3-O5).
