# version_fix: resposta às revisões da SIGMOD 2026 (paper 473)

A `version_fix` parte da `paper_v4`, que já tinha a rodada v3, o Related Work revisado, a §5.4 sobre a compilação para SQL e não tinha mais o modo *document-level*. Ela também incorpora as edições rastreadas da Alti (26/08), aceitas como base. Toda alteração nova aparece destacada no PDF: texto novo em azul (`\add`, ou o grupo azul da §5.5), texto removido riscado em vermelho (`\rem`, `\subs`). O interruptor `\Reviewfalse`, em `packages_commands.tex`, gera a versão limpa (`version_fix_clean.pdf`). Todas as notas `\altic` foram removidas.

**Título** (aplicado: opção 1)
1. *Text-to-Constraints: Mapping Regulatory Requirements to Executable Data Quality Constraints*
2. *From Normative Text to Denial Constraints: Grounding Regulatory Requirements in Relational Data*
3. *Norma: Compiling Regulatory Requirements into Executable Data Quality Constraints*
4. *Grounding Regulatory Requirements in Data with LLM-Based Constraint Mapping*

Legenda: ✔ já resolvida antes desta versão (nota apenas removida) · ✎ corrigida nesta versão · ⚠ atendida em parte, por falta de dados ou de experimento

| Nota | Situação | O que foi feito e onde |
|---|---|---|
| R2-O1 | ✔ (v4) | Exemplo 1 (§1): a consulta do Art. 19 retorna 76 pares violadores, envolvendo 61 candidatos. |
| R3-O1, R1-O1 (Related Work) | ✔ (v4) | §7 já cita DCs aproximadas (`Pena2019DiscoveryDC`, `martin2026lima`), ranqueamento com critérios do usuário (`fan2023topk`), REEs/Rock e geração de regras por LLM. |
| R3-O1 (intro) | ✎ | §1, 2º parágrafo: distingue regras que valem, exata ou aproximadamente, nos dados de regras exigidas pela norma; cita a descoberta aproximada e o apoio a especialistas. Resumo: "(approximately)". |
| R1-O1 (intro) | ✎ | §1: novo parágrafo que posiciona REEs com predicados de ML, extensões da descoberta de DCs e geração de regras por LLM frente ao vocabulário controlado derivado do esquema. |
| R1-O2, R3-O6 | ✎ | §1, terceiro desafio reescrito: o problema não é SQL contra outra linguagem, e sim que consultas SQL equivalentes não têm forma canônica; DCs são o alvo canônico, compilado para SQL. Liga-se a "Why DCs" (§2) e à §5.4. |
| R3-O5 ("regularities") | ✎ | §1: "patterns that happen to hold in the data instance" e "patterns that hold only incidentally". |
| R3-O5 (*mapper* antes de definido) | ✎ | §3: redação sugerida pelo revisor, "ruled out by the mapper (which we define next)". |
| R1-O4 | ✎ | §3: a rastreabilidade passa a ser uma propriedade "por construção" do processamento por sentença; um prompt de documento exige um passo de atribuição (por exemplo, a sumarização iterativa sugerida pelo revisor). |
| R1-O7 (sem nota no texto) | ✎ | §3: o classificador só rotula as sentenças dadas e não gera texto, preservando o vínculo sentença→DC. |
| R1-O6 | ✎ | §3: SQL/DuckDB é a implementação do protótipo, porque as consultas rodam em qualquer *engine* e servem às verificações do mapper; *engines* de DC são uma alternativa para escala. A §5.4 detalha. |
| R3-O7 | ✎ | §5.1: constantes e `NULL` já são expressáveis em DCs; o que falta nos espaços de descoberta são conjuntos (IN), padrões (LIKE) e aritmética. |
| R2-O2, R2-O3, R1-O5, R3-O2 (crítica central) | ✎ | Nova §5.5, "Methodological Contributions", com tabela componente a componente: o que foi adaptado de Text-to-SQL (*schema linking*, decomposição em estágios — RESDSQL, DIN-SQL) e o que é novo (espaço de predicados sob demanda, forma de violação, verificações nos dados, compilação para SQL com semântica de pares e de `NULL`, avaliação por equivalência). |
| R2-O4 | ✎ | §5.1, "Coverage and limits": 487 dos 490 predicados do GT pertencem ao espaço (medido com o gerador do Norma); as 3 exceções estão na HMDA Checks (aritmética com datas codificadas como inteiros). Limites declarados: sem agregação nem aritmética sobre datas ou sequências (coluna derivada no Restaurant). |
| R1-O8 | ✎ | §5.2, "Cross-tuple constraints": no máximo duas tuplas, semântica universal (basta um par); dependências diferenciais e regras "para a maioria" ficam fora do escopo. |
| R2-O5 | ⚠ | §5.3: justificativa dos quatro estágios (cada um isola uma decisão que pode ser validada) e evidência indireta dos experimentos (robustez com modelos instáveis; o *rule filter* evita chamadas de construção). **Não há ablação fina**: declarada como trabalho futuro. |
| R1-O10 | ✎ | §6.1: construção do GT (autores, forma de violação, sentenças não verificáveis rotuladas como não-regras); redundância medida (nenhuma sentença repetida; quase-duplicados com similaridade ≥ 0,9 são modelos que diferem num atributo e nunca compartilham DC; 13 pares no SISU, 32 na HMDA Checks); tipos do SISU (11 de obrigatoriedade, 1 FD, 1 OD, 1 CFD, 3 DCs condicionais). **Não há como informar** concordância entre anotadores: não existe esse dado. |
| R3-O3, R2-O6.2 | ✎ | §6.1: Tax e Hospital declarados como cenário controlado; SISU, Restaurant e os dois HMDA são o sinal mais confiável. |
| R1-O12 | ⚠ | §6.2: FDs e ODs com até duas tuplas são casos particulares de DCs, então a descoberta aproximada de DCs as subsume. **Não há baseline de CFD dedicado**: declarado como trabalho futuro. |
| R3-O4 | ✎ / ⚠ | §6.5: sem contexto, o classificador já tem recall 1,00 nos datasets controlados, com ganho de +0,04; nos reais, o ganho é de +0,09; os GPT-6 ganham menos (0,03–0,05). A hipótese de pré-treino é **declarada como não verificável**. |
| R1-O11 | ✎ / ⚠ | §6.6, "Sensitivity to the backbone": os modelos divergem mais na violação (AV de 0,52 a 0,90) que na extração (CE de 0,60 a 0,68); as divergências se concentram em decisões implícitas (chave da entidade no SISU, gatilhos definidos no HMDAZ, datas como inteiros); há variação entre execuções (Mistral no Restaurant: 1,00 → 0,02). **Não há como correlacionar** com tamanho ou dados de treino, que não são divulgados para a maioria dos modelos. |
| R1-O9 | ✎ | §6.7, "Usability of the output": o C+MS devolve de 6 a 24 DCs nos quatro datasets menores, 9 no HMDAZ e 96 na HMDA Checks, perto do número de regras do GT; triagem por número de tuplas marcadas, sentença de origem e revisão humana opcional. |
| R2-O6.1 | ✎ | §8: limitação de escala (até 158 sentenças); custo linear; cláusulas aninhadas como trabalho futuro. |
| R1-O3 | ✎ | §8: a DC vem do texto e continua válida enquanto a norma não muda; reexecução incremental (`youri2024`) quando os dados mudam; só uma mudança no texto exige novo mapeamento. |

## Outras correções

- **Verificação G2 do Multi-Step.** Ela descartava DCs com mais de 55.000 tuplas violadoras (uma tupla) ou 10.000 pares, limites absolutos que o artigo não descrevia; o de 55.000 ficava logo acima das 52.041 violações do Restaurant. Passou a ser relativa (> 90% das tuplas, provável lógica invertida), foi descrita na §5.3, e o Multi-Step foi reexecutado (E2 e E3).
- **Incoerências entre texto e código:**
  - só o Multi-Step faz as verificações com os dados;
  - o Single-Prompt devolve uma lista de regras independentes;
  - o *literal grounding* usa descrições e valores de domínio;
  - a introdução não fala mais em "multiple extraction modes";
  - a forma do problema (§2) admite várias DCs independentes por sentença e predicados aritméticos.
- **"Our previous evaluation"** (a versão rejeitada) virou "a preliminary study with the same protocol".
- **Escrita:** "a orphan predicate is disregard", "i.e.,,", "contributions are", "Text-to-Constraint", "emerges a", "discoverable pattern", "on across", vírgula unindo orações na §5, "category..".
- **Resumo:** retirada a afirmação "with high precision", que o E3 não sustenta (Tax e SISU).

## Resultados novos (Etapa A)

- **Reexecução do Multi-Step com a G2 relativa** (`results/v3_fix`): 48/48 jobs, US$ 54,80. A escolha do mapper do E3 mudou de GPT-6 Sol para **Claude Sonnet 5.5** (média de CE e AV: 0,776; Luna 0,771; Sol 0,720). O E3 foi refeito com o Sonnet (12/12, US$ 35,72). O E3 com o Sol está em `results/v3_fix/superseded/`. Custo de tudo o que está no artigo: US$ 94.
- **Efeitos da G2 relativa, declarados no texto:**
  - na HMDA Checks, só o GPT-6 Sol mapeia uma edição de data; os outros modelos sobem a violação agregada com DCs amplas e erradas (480k a 550k linhas), que coincidem com os 98,6% de linhas violadoras. A AV é declarada não informativa nesse dataset, e a sensibilidade mostra isso: sem as edições de data, a AV desses modelos fica em 0,12–0,19;
  - no SISU, dois modelos repetem o erro de chave do Single-Prompt (`inscricao_enem` em vez de `cpf`), antes escondido pelo limite absoluto.
- **E3 com o Sonnet:**
  - o classificador decide o Tax (AV 1,00 com C+MS contra 0,01 com o Multi-Step sozinho);
  - no SISU, sentenças de categorias de cota, que o GT trata como não-regras, viram DCs amplas (AV 0,03–0,08, com F1 por regra de 0,93);
  - no Restaurant, a regra dominante não é mapeada (AV 0,02).

  O resumo deixou de dizer que o Norma supera os *baselines* também na violação.

## Autoavaliação

**Verificações feitas:**
- 81 testes passando.
- 60 jobs de reexecução sem falha.
- As duas versões compilam sem erros e sem referências ou citações indefinidas; nenhuma nota `\altic` restante.
- Todas as 64 chaves citadas existem no `refs.bib`.
- Tamanho: versão limpa com 17 páginas, as mesmas da `paper_v4`; 16.740 palavras no PDF (+1,8%) e +3,3% no texto visível das seções.

**Erros encontrados na autoavaliação e corrigidos:**
- no E3, afirmei que o Multi-Step havia mapeado a edição Q022; a verificação mostrou que não (a AV de 0,71 vem de uma DC ampla e errada);
- referência à tabela de SQL para a regra anual do Restaurant, que não está na tabela;
- "C+MS filters these sentences" (no Tax ainda passa uma DC);
- "0.98 on SISU" (vale só para o Multi-Step);
- pronome sem referente no *overview* ("Recovering it");
- "this mapping" sem antecedente na introdução;
- resquícios de "extraction";
- espaço ausente em "identifier.Including", erro que já existia no original;
- erros de LaTeX da versão destacada (`\cite` dentro de texto riscado; parágrafo dentro de `\subs`);
- Tabela 5 passando da largura da coluna;
- dois cortes de tamanho que removiam respostas a revisores (R1-O1, R3-O4) foram desfeitos.

**Riscos que permanecem:**
- uma execução por configuração; a variação entre execuções é grande na AV, e isso está declarado no texto;
- os *baselines* não foram reexecutados (os do SISU são anteriores à correção de pares);
- as respostas ⚠ (R2-O5, R1-O12, R3-O4, R1-O11) dependem de experimentos que não foram feitos;
- as afirmações sobre anotação (feita pelos autores) seguem o texto original e não foram verificadas de forma independente;
- o limite de 0,9 da G2 foi definido no plano antes da reexecução e não foi ajustado depois.

## Atualização (2026-10-22): três execuções do E3 e baselines nos HMDA

- **E3 repetido três vezes** com Claude Sonnet 5.5 (`results/v3_fix`, `results/v3_fix_rep2`, `results/v3_fix_rep3`; US$ 71,48 pelas repetições). A Tabela 3 traz a média, e o texto cita as faixas entre execuções. O custo de tudo o que está no artigo passou a US$ 166.
- **Comparação com os baselines** (quatro datasets que os têm): o Norma vence em extração em todos os datasets. Perde só no **Restaurant**:
  - violação: 0,35 em média contra 0,82 do DC Discovery;
  - classificação: 0,79 contra 0,80 do XiYan-SQL.

  A causa da derrota na violação: o *rule filter* julga não expressável a regra de "pelo menos uma inspeção por ano" (99% das violações). A rejeição ocorreu nas 3 execuções do Multi-Step e em 2 das do C+MS; numa reprodução isolada, em 1 de 3 tentativas. Corrigir isso ajustando o *prompt* seria vazamento de teste. O texto e o resumo foram ajustados: o Norma supera os baselines em extração em todos os datasets, e em classificação e violação na maioria deles.
- **Baselines nos HMDA:** não podem ser executados nesta máquina (o código não está no repositório, e o XiYan-SQL exige GPU de cerca de 30 GB). Por decisão dos autores, o pacote para o Eduardo Pena executar está em `results/baselines_hmda_share/` (dados de 1M de linhas, sentenças, colunas do GT e `LEIAME.md`). O script de avaliação no protocolo v3 está em `experiments/baselines/evaluate_baselines_v3.py`. Enquanto isso, a Tabela 3 mostra "--" para os baselines nos HMDA.
