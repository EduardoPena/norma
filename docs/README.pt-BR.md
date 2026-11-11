# NORMA

NORMA avalia a identificação e o mapeamento de regras escritas em linguagem natural para *denial constraints* (DCs) executáveis sobre dados tabulares. O projeto inclui carregamento de datasets, validação e tradução de regras para SQL/DuckDB, classificadores e mappers baseados em LLM, runners configuráveis para experimentos e a avaliação dos baselines (DC Discovery e XiYan-SQL).

## Capacidades

- Validar árvores de regras com operadores lógicos `AND` e `OR`.
- Traduzir DCs para SQL e encontrar as tuplas que violam cada regra.
- Classificar sentenças como regras ou não regras.
- Mapear sentenças para DCs com mappers diretos (Single-Prompt), por refinamento em etapas (Multi-Step) ou por documento.
- Gerar o espaço de predicados a partir do schema inteiro (v1) ou sob demanda, só para as colunas selecionadas (v2).
- Avaliar o mapeamento por correspondência exata e por equivalência lógica, a equivalência de execução e a descoberta de violações, por regra e de forma agregada.
- Avaliar os baselines com o mesmo protocolo e o mesmo avaliador do Norma.

## Instalação

Crie e ative um ambiente virtual e instale as dependências declaradas:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Os experimentos que usam modelos remotos exigem uma chave de API conforme o provedor configurado:

```bash
export OPENAI_API_KEY='...'
export OPENROUTER_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export MISTRAL_API_KEY='...'
export GROQ_API_KEY='...'
export MARITACA_API_KEY='...'
```

Defina apenas a variável correspondente ao provedor usado no YAML. Os experimentos da rodada v3 usam só o OpenRouter. Não registre chaves no repositório.

Dependências dos baselines, fora do `requirements.txt`:

- **DC Discovery:** Java ≥ 20 para o `DCDiscovery.jar`.
- **XiYan-SQL:** GPU com o ambiente de `results/baselines_v3_gpu_package/requirements-gpu.txt`.

## Estrutura do Projeto

| Diretório                  | Conteúdo                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `src/`                     | Implementação de predicados, banco de dados, avaliação, LLMs, classificadores e mappers. |
| `data/`                    | Datasets, schemas, especificações de predicados, glossários, CSVs e ground truth.        |
| `experiments/`             | Runners, geradores de configuração e configurações YAML dos experimentos.                |
| `experiments/baselines/`   | Preparo, execução e avaliação dos baselines (DC Discovery e XiYan-SQL).                  |
| `baselines/`               | Código original dos baselines (`.jar`, scripts do XiYan-SQL). Fica fora do git.          |
| `prompts/`                 | Prompts usados pelos classificadores e mappers.                                          |
| `results/`                 | Saídas, métricas, análises e versões dos artigos (ignorado pelo git; versionado com `git add -f`). |
| `tests/` e `test_norma.py` | Testes e verificações focadas.                                                           |

## Formato de Dataset

Cada dataset fica em `data/<nome-do-dataset>/` e deve conter:

- A especificação do espaço de predicados, em uma de duas versões:
  - **v1:** `*_predicates.json`, com o schema e todos os templates de predicados;
  - **v2:** `<tabela>_schema.json`, só com o schema. Cada coluna traz descrição, valores de domínio (domínios pequenos) e operadores aritméticos permitidos, e os templates são gerados em tempo de execução para as colunas que o mapper seleciona (`src/predicates/on_demand.py`).
- Um CSV para cada tabela do schema. Para experimentos de tabela única, o CSV deve ter o nome da tabela ou ser o único CSV do diretório.
- Um ground truth `*_text2rule_gt.json` com uma lista `mapping`.
- Opcionalmente, um glossário `*_glossary.json` com definições oficiais das colunas, incorporado às descrições do schema v2 por `experiments/apply_glossary.py` (hoje só no HMDAZ).

Um item do ground truth associa uma ou mais sentenças às regras que descrevem suas violações:

```json
{
  "id": 1,
  "sentences": ["If amount is positive, income must be positive."],
  "rules": [
    {
      "AND": [
        "t1.amount > 0",
        "t1.income <= 0"
      ]
    }
  ]
}
```

Uma regra simples pode ser uma string; regras compostas usam árvores JSON. `AND` e `OR` devem ter pelo menos dois itens. Predicados `LIKE` / `NOT LIKE` exigem o padrão entre aspas. A API pública em `src.predicates` fornece `normalize_rules`, `parse_rule` e `parse_rules` para validação antes da execução.

### Geração dos Espaços de Predicados

| Script | Uso |
|---|---|
| `experiments/build_predicate_spaces.py [dataset ...]` | Regenera os espaços v1 dos datasets do artigo a partir dos CSVs. |
| `experiments/build_schema_v2.py data/<ds> --from-v1 [...]` | Gera o schema v2. As descrições são escritas por LLM a partir de um perfil determinístico dos dados, ou só pelo perfil com `--no-llm`. |
| `experiments/apply_glossary.py data/<ds>` | Incorpora o glossário às descrições do schema v2. Pode ser rodado de novo sem duplicar texto. |
| `experiments/extend_predicate_space.py <predicates.json> <csv>` | Acrescenta templates opcionais a um espaço v1: `IN` em colunas codificadas, famílias de colunas e aritmética. |

Os scripts leem só as primeiras `--sample-rows` linhas, nunca o CSV inteiro. Os arquivos v1 não são alterados pela geração do v2.

## Configuração de Experimentos

Os runners recebem um arquivo YAML como argumento. Uma configuração de mapper somente com sentenças de ground truth, no formato da rodada v3, segue este modelo:

```yaml
experiment_name: "Mapper evaluation"
experiment_type: "mapper_gt_only"
results_dir: "results/mapper_example"

datasets:
  - restaurant

dataset_options:
  restaurant:
    predicate_space: v2      # v1 (padrão) | v2
    sample_rows: 1000000
    stream_csv: true

variants:
  - name: "Multi-Step | GPT-6 Luna"
    use_context: false
    mapper:
      implementation: "multi_step_refinement"
      provider: "openrouter"
      model: "openai/gpt-6-luna"
      max_refinement_iterations: 2
      prompt_file: "multi_step_refinement"
      max_violation_share: 0.9
```

Use `dataset_options` para selecionar arquivos de dados/GT (`gt_file`), limitar `sample_rows`, ativar `stream_csv` ou escolher `predicate_space`. Veja as configurações em `experiments/configs/` para exemplos por dataset e provedor; as da rodada v3 ficam em `experiments/configs/v3/` e `experiments/configs/v3_fix*/`.

### Multi-Step

- Com o espaço v2, os candidatos do passo de seleção de predicados vêm das colunas escolhidas pelo LLM, e as descrições e os valores de domínio vão para os prompts.
- O filtro de regras (*rule gate*) descarta uma DC quando mais de `max_violation_share` das tuplas (padrão 0,9) participam de alguma violação. Esse limite relativo substituiu os limites absolutos anteriores (55.000 / 10.000).

### Opções do Single-Prompt

Nos runners 1 e 3, o mapper `implementation: llm` aceita estes campos dentro
do bloco `mapper`:

```yaml
implementation: "llm"
provider: "openrouter"
model: "openai/gpt-6-luna"
prompt_key: "with_examples_v2"     # with_examples = protocolo do artigo
batch_size: 10
filter_predicates: true
max_candidate_predicates: 1000
max_relevant_columns: 12
max_tokens: 8192
column_context: true               # padrão: ativo com o espaço v2
explain_violation: true            # padrão: false
column_retrieval: "lexical"        # lexical (padrão) | semantic | hybrid
embedding_model: "openai/text-embedding-3-large"
```

- **`prompt_key`:** `with_examples` é o prompt do artigo e continua inalterado. `with_examples_v2` acrescenta o contexto das colunas selecionadas, orientações sobre definições de status, proibições e direção de limiares, e uma autoverificação.
- **`column_context`:** envia a descrição e os valores de domínio das colunas selecionadas. Fica ativo por padrão com o espaço v2.
- **`explain_violation`:** pede, antes das regras, uma frase `violated_when` em linguagem simples (schema `SENTENCE_RULES_EXPLAINED`). Ela é salva nas saídas GT-only.
- **`column_retrieval`:** escolhe as colunas por sobreposição lexical (padrão), por similaridade de embeddings do OpenRouter (`semantic`) ou pela fusão das duas (`hybrid`). Os embeddings ficam em cache no disco.
- **`max_tokens`:** limita a saída via OpenRouter nos mappers Single-Prompt e Multi-Step dos runners 1 e 3. Sem ele, o padrão do provedor continua sendo usado. O limite de saída não garante que o prompt de entrada caiba no contexto do modelo.

Por padrão, `filter_predicates` é `false`: o prompt continua recebendo todos
os predicados. Quando ativado, o filtro identifica colunas pela estratégia de
`column_retrieval`, sem acessar o GT. Reutiliza a seleção de candidatos do
Multi-Step e preserva os templates simples de todas as colunas, para não
eliminar conceitos implícitos na sentença. O limite de candidatos é por
sentença e não inclui os templates simples preservados; em batches, o prompt
recebe a união dos candidatos. Com o espaço v2, o Single-Prompt recebe só os
templates das colunas selecionadas.

O espaço original e os arquivos de predicados permanecem inalterados. Essas
opções mudam o espaço ou o prompt oferecido ao Single-Prompt e devem ser
registradas como adaptação do protocolo experimental. Reduzir tokens não
garante preservar todas as comparações compostas ou a qualidade do mapeamento;
valide o recorte de candidatos e o tamanho do prompt antes de executar cada
modelo. O modo por documento exige o espaço v1.

### Saídas Estruturadas e Falhas de Saída do Modelo

Todas as chamadas de classificação e mapeamento usam *structured output* estrito
(`response_format: json_schema`) via OpenRouter. Os schemas ficam em
`src/llm/schemas.py` e são derivados do formato canônico das DCs: `rules` é `null`
ou uma lista de DCs independentes, e cada DC é um predicado (string) ou um nó
`{"AND": [...]}` / `{"OR": [...]}`, recursivamente.

| Componente | Schema | Formato |
|---|---|---|
| Classificador | `CLASSIFICATION` | `{"labels": [{"id": 1, "is_constraint": true}, ...]}` |
| Single-Prompt | `SENTENCE_RULES` (`SENTENCE_RULES_EXPLAINED` com `explain_violation`) | `{"results": [{"id": 1, "rules": [...] \| null}, ...]}` |
| Multi-Step | `UNDERSTAND_SCHEMA`, `SELECT_PREDICATES`, `RULE_GATE`, `CONSTRUCT_CONSTRAINT`, `REFINE_CONSTRAINT` | um objeto por passo |

O schema garante a estrutura. A gramática dos predicados, o mínimo de 2 itens por
nó e a pertinência ao espaço de predicados continuam validados por
`normalize_rules` e pelos mappers. O client envia apenas os parâmetros que o
modelo suporta no catálogo do OpenRouter (o GPT-5, por exemplo, não aceita
`temperature`) e pede `provider.require_parameters` para rotear só a provedores
com suporte. Modelos sem `response_format` caem automaticamente no modo só-prompt,
que descreve o mesmo JSON. Use `structured_output: false` no bloco do mapper ou do
classificador para forçar esse modo.

Falhas de **saída do modelo** (JSON inválido, ids faltando, regra inválida,
resposta vazia por esgotar tokens) passam por novas tentativas. Se persistirem, a
sentença fica sem constraint (classificador: NO depois do fallback por sentença).
Elas são contadas na coluna `model_output_failures` do `results_summary.csv`.
Falhas de **infraestrutura** (API indisponível após o backoff) interrompem a
variante. O `batch_size` do classificador também é respeitado no runner 1.

### Variáveis de Ambiente

| Variável | Padrão | Efeito |
|---|---|---|
| `NORMA_LLM_SEED` | (vazia) | Semente de amostragem, enviada só a modelos cujo catálogo aceita `seed`. Opcional, porque também restringe o roteamento a provedores que aceitam semente. |
| `NORMA_JSON_OBJECT_MODELS` | (vazia) | Lista de modelos, separados por vírgula, cujo `json_schema` estrito não é confiável. Para eles, o schema vai no prompt, o pedido usa `json_object` e a saída é validada como no modo só-prompt. Nos runs do `hmda_smoke.py`, vem de `json_object_models` no YAML. |
| `NORMA_SQL_TIMEOUT_S` | 300 | Limite de tempo, em segundos, de cada consulta de violação. Uma consulta que estoura conta como erro de execução. |
| `NORMA_SQL_MAX_ROWS` | 20000000 | Máximo de linhas do resultado de uma consulta; acima disso, a consulta é abortada e conta como erro. |

## Execução de Regras e Métricas

As DCs são compiladas para SQL por `src/dbms/sql_translator.py` e executadas no DuckDB:

- **Uma tupla:** `SELECT t1.tid FROM <tabela> t1 WHERE <predicados>`.
- **Duas tuplas:** `SELECT t1.tid, t2.tid FROM <tabela> t1, <tabela> t2 WHERE t1.tid <> t2.tid AND <predicados>`. A condição `t1.tid <> t2.tid` considera os pares nas duas ordens. Até 19/10/2026 era `t1.tid < t2.tid`, o que perdia violações de DCs assimétricas; no SISU, as violações gold passaram de 487 para 533.

As violações são sempre contadas por tupla: cada linha devolvida é desfeita nos seus `tid`, e todos entram num único conjunto. Um par visto nas duas ordens não infla a contagem.

O runner de mapper GT-only mede:

- **Mapeamento exato** (`constraint_exact_*`) e Jaccard de predicados, calculado sobre as folhas.
- **Mapeamento por equivalência lógica** (`constraint_equiv_*`, `pairwise_logical_match`): compara as regras em forma normal lógica. Um literal de código e o seu rótulo (por exemplo, `lien_status = 1` e `lien_status_name = 'Secured by a first lien'`) são unificados só quando selecionam exatamente as mesmas linhas nos dados (`src/evaluation/value_aliases.py`).
- **Equivalência de execução** (`violation_identical_*`): regras gold com pelo menos uma violação cuja previsão marca exatamente as mesmas linhas.
- **Violações por sentença e por regra:** precisão, recall, F1 e Jaccard.
- **Violações agregadas:** precisão, recall e F1 sobre todas as sentenças avaliadas.
- **Diagnóstico:** consultas que falharam (contadas, nunca descartadas em silêncio), concentração das violações gold e predicados fora do espaço.

## Execução

Execute os comandos a partir da raiz do repositório. Cada runner recebe o caminho de uma configuração YAML:

```bash
# Pipeline fim a fim: classificação e mapeamento
python experiments/1_run_end2end.py experiments/configs/1_run_end2end_config.yaml

# Avaliação de classificadores
python experiments/2_run_classifiers.py experiments/configs/2_run_classifiers_config.yaml

# Qualidade do mapper somente nas sentenças com GT
python experiments/3_run_mapping.py experiments/configs/3_run_mapping.yaml

# Mapeamento de documento inteiro
python experiments/4_run_mapping_document.py experiments/configs/4_run_mapping_document.yaml
```

Para usar o ambiente virtual sem ativá-lo, substitua `python` por `.venv/bin/python`.

Os resultados são escritos no diretório definido por `results_dir`, com uma cópia da configuração, saídas por variante e métricas detalhadas quando aplicável.

### Validação Antes dos Experimentos

A checagem remota usa exemplos neutros, não carrega CSVs e registra cada
tentativa, incluindo uso de tokens, custo informado e erros do provedor.
Também verifica o schema do Single-Prompt melhorado (`with_examples_v2` com
`violated_when`).

Os modelos e slugs ficam em `experiments/configs/hmda_models.yaml`; o resumo vai
para `results/hmda/llm_health_check.{md,json}`:

```bash
.venv/bin/python experiments/hmda_model_health.py
```

O smoke usa os runners 2, 3 e 1, com as primeiras 1.000 linhas e cópias de GT
contendo duas entradas positivas e uma negativa por dataset. Os originais
não são modificados. O YAML permite definir modelos, seleção e orçamento:

```bash
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_smoke.yaml --dry-run
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_smoke.yaml
```

O harness aplica o limite de saída, desativa os retries do SDK (o
`OpenRouterClient` repete erros transitórios — 429, 5xx, timeout, conexão — com
backoff exponencial), contabiliza cada tentativa e verifica uma reserva
conservadora antes de cada chamada. Erros de acesso/crédito interrompem novas
chamadas. O resumo registra fases incompletas, tempo e pico de memória.
Sem violações GT no recorte, métricas de descoberta de violações não servem
para comparar modelos. Não use resultados do smoke como reprodução de E1/E2/E3.

### Rodadas Completas (E1 → E2 → E3)

`experiments/hmda_make_configs.py` gera as configurações e o plano de uma rodada:

| Opção | Efeito |
|---|---|
| `--models <manifesto.yaml>` | Modelos da rodada (padrão `hmda_models.yaml`; rodada v3: `v3_models.yaml`). |
| `--datasets` | `hospital`, `restaurant`, `sisu`, `tax`, `hmdaz`, `hmda_checks` (padrão: os dois HMDA). |
| `--experiments` | `e1`, `e2`, `e3` (padrão: os três). |
| `--predicate-space` | `v1` ou `v2`. |
| `--sp-variant` | `paper` (`with_examples`, batch 10) ou `v2` (Single-Prompt melhorado). |
| `--row-limit` | Primeiras N linhas de cada CSV (até 1.000.000). |
| `--config-dir`, `--results-root` | Onde gravar as configurações e os resultados. |

```bash
.venv/bin/python experiments/hmda_make_configs.py   # gera hmda_e*.yaml e hmda_plan.json
.venv/bin/python experiments/hmda_audit.py data/hmdaz --gt-file hmdaz_text2rule_gt.json
.venv/bin/python experiments/hmda_audit.py data/hmda_checks --gt-file hmda_regc_checks-text2rule-gt.json
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_rerun.yaml \
  --plan experiments/configs/hmda_plan.json   # cada job roda num subprocesso próprio
# retomar uma execução interrompida (pula jobs concluídos):
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_rerun.yaml \
  --plan experiments/configs/hmda_plan.json --resume results/hmda/telemetry/<timestamp>
```

A rodada oficial v3 (6 datasets, 6 modelos, espaço v2) usa `experiments/configs/v3_run.yaml` com os planos de `experiments/configs/v3/`. A reexecução do Multi-Step com o filtro relativo e as repetições do E3 usam `v3_fix_run.yaml` e `v3_fix_reps_run.yaml`. Os resultados ficam em `results/v3`, `results/v3_reeval` (reavaliação com a semântica de pares corrigida, sem novas chamadas) e `results/v3_fix*`.

Tabelas e comparações:

```bash
# Tabela fim a fim do artigo (Norma: média das repetições; baselines: resumo final)
V3_RESULTS=results/v3_fix E3_ROOTS=results/v3_fix,results/v3_fix_rep2,results/v3_fix_rep3 \
BASELINES=results/v3_fix/baselines/baselines_final_300s.csv \
TABLE_OUT=results/v3_fix/baselines/table_300s \
.venv/bin/python results/v3/analysis/table3.py

# Comparação de uma rodada com os números E1/E2/E3 do artigo original
.venv/bin/python experiments/compare_with_paper.py results/paper_v2
```

O relatório das rodadas HMDA fica em `results/hmda/REPORT.md` (`experiments/hmda_report.py`).

## Baselines

Os dois baselines são avaliados por `experiments/baselines/evaluate_baselines_v3.py`, com o mesmo carregamento de dados, GT, tradutor SQL e avaliador do Norma. Os detalhes, as decisões e os resultados estão em `results/v3_fix/baselines/README.md`.

**DC Discovery** (`baselines/1_profiling/dc_discovery/DCDiscovery.jar`):

```bash
# Entradas: colunas do GT (Hospital e Tax com todas), tipos no cabeçalho, amostras HMDA com semente
.venv/bin/python experiments/baselines/prepare_dc_discovery.py --datasets hospital restaurant sisu tax
.venv/bin/python experiments/baselines/prepare_dc_discovery.py --datasets hmdaz hmda_checks --sample 100000
# Grade: exato e ε ∈ {0,001, 0,005, 0,01, 0,05, 0,1}; argumentos: datasets, timeout, heap
experiments/baselines/run_dc_discovery.sh "hospital tax" 6h 8g
# Avaliação: ranking pelo escore misto, top-k ∈ {5, 10, 20, 50, 100}
.venv/bin/python experiments/baselines/evaluate_baselines_v3.py profiling \
  --dataset tax --csv baselines/1_profiling/results_dc_discovery/tax_v3_*.csv
```

- O `.jar` exige Java ≥ 20. O script chama a JVM com `-Duser.language=en -Duser.country=US`, porque com o locale pt-BR ela escreve vírgula decimal e corrompe o CSV de saída.
- Cada execução é registrada em `results/v3_fix/baselines/dc_logs/runs.tsv`, com status `ok`, `timeout`, `oom` ou `interrupted`.

**XiYan-SQL:**

- **Inferência:** roda numa GPU com `baselines/2_text2sql/run_xiyan_v3.py`, que aceita `--prompt A|B` e sementes. Os passos estão em `results/baselines_v3_gpu_package/RUN.md`.
- **Avaliação:**

  ```bash
  .venv/bin/python experiments/baselines/evaluate_baselines_v3.py xiyan --dataset tax \
    --table data --dialect sqlite --detailed <saídas>/tax/seed*/tax_xiyan_detailed_results.csv
  ```

- **Execução das consultas:** cada uma é executada como foi gerada. Consultas em SQLite são traduzidas para DuckDB com `sqlglot`.
- **Avaliação do mapeamento:** `experiments/baselines/sql_rules.py` converte o `WHERE` de cada consulta em regras do Norma, que são comparadas às DCs gold:
  - um `EXISTS` correlacionado vira DC de pares;
  - o `NOT` é empurrado para dentro dos predicados;
  - os predicados de `tid` são descartados.
- **Consultas sem leitura como DC:** são pontuadas por equivalência de execução.

O resumo de saída é `results/v3_fix/baselines/summary.csv`, ou o arquivo indicado em `BASELINES_SUMMARY`. A escrita usa trava, então várias avaliações podem rodar em paralelo.

## Testes

Execute a suíte a partir da raiz do repositório (90 testes):

```bash
.venv/bin/python -m unittest discover -s tests
```

Teste focado do parser de regras e da tradução para SQL:

```bash
.venv/bin/python -c "from test_norma import test_recursive_rule_parser_and_sql; test_recursive_rule_parser_and_sql(); print('ok')"
```

Para verificar a sintaxe JSON de um ground truth antes de executar um experimento:

```bash
python -m json.tool data/<dataset>/<arquivo>_text2rule_gt.json > /dev/null
```

Para validar a estrutura de regras de um GT já migrado para o formato atual, use `normalize_rules` ou `parse_rules` da API `src.predicates` no carregamento ou em um teste focado.

## Contribuição

Mantenha mudanças focadas, preserve o formato estruturado das regras e valide o dataset e a configuração antes de iniciar execuções com LLM. Não modifique resultados ou ground truth de datasets não relacionados à alteração em andamento.
