# NORMA

NORMA avalia a identificação e o mapeamento de regras escritas em linguagem natural para *denial constraints* (DCs) executáveis sobre dados tabulares. O projeto inclui carregamento de datasets, validação e tradução de regras para SQL/DuckDB, classificadores e mappers baseados em LLM, além de runners configuráveis para experimentos.

## Capacidades

- Validar árvores de regras com operadores lógicos `AND` e `OR`.
- Traduzir DCs para SQL e encontrar tuplas que violam cada regra.
- Classificar sentenças como regras ou não regras.
- Mapear sentenças para DCs com mappers diretos, por refinamento em etapas ou por documento.
- Avaliar correspondência sintática e descoberta de violações, tanto por sentença como de forma agregada.

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

Defina apenas a variável correspondente ao provedor usado no YAML. Não registre chaves no repositório.

## Estrutura do Projeto

| Diretório                  | Conteúdo                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `src/`                     | Implementação de predicados, banco de dados, avaliação, LLMs, classificadores e mappers. |
| `data/`                    | Datasets, especificações de predicados, CSVs e ground truth.                             |
| `experiments/`             | Runners e configurações YAML dos experimentos.                                           |
| `prompts/`                 | Prompts usados pelos classificadores e mappers.                                          |
| `results/`                 | Saídas, métricas e configurações reproduzíveis de cada execução.                         |
| `tests/` e `test_norma.py` | Testes e verificações focadas.                                                           |

## Formato de Dataset

Cada dataset fica em `data/<nome-do-dataset>/` e deve conter:

- Uma especificação `*_predicates.json` com o schema e o espaço de predicados.
- Um CSV para cada tabela do schema. Para experimentos de tabela única, o CSV deve ter o nome da tabela ou ser o único CSV do diretório.
- Um ground truth `*_text2rule_gt.json` com uma lista `mapping`.

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

Uma regra simples pode ser uma string; regras compostas usam árvores JSON. `AND` e `OR` devem ter pelo menos dois itens. A API pública em `src.predicates` fornece `normalize_rules`, `parse_rule` e `parse_rules` para validação antes da execução.

## Configuração de Experimentos

Os runners recebem um arquivo YAML como argumento. Uma configuração de mapper somente com sentenças de ground truth segue este formato:

```yaml
experiment_name: "Mapper evaluation"
experiment_type: "mapper_gt_only"
results_dir: "results/mapper_example"

datasets:
	- restaurant

variants:
	- name: "MultiStep mapper"
		use_context: false
		mapper:
			implementation: "multi_step_refinement"
			provider: "openrouter"
			model: "google/gemini-2.5-flash"
			max_refinement_iterations: 2
			prompt_file: "multi_step_refinement"
```

Use `dataset_options` para selecionar arquivos de dados/GT, limitar `sample_rows` ou ativar `stream_csv`. Veja as configurações em `experiments/configs/` para exemplos por dataset e provedor.

### Filtro Opcional do Single-Prompt

Nos runners 1 e 3, o mapper `implementation: llm` aceita estes campos dentro
do bloco `mapper`:

```yaml
implementation: "llm"
provider: "openrouter"
model: "openai/gpt-5-mini"
prompt_key: "with_examples_and_table"
batch_size: 10
filter_predicates: true
max_candidate_predicates: 1000
max_relevant_columns: 12
max_tokens: 8192
```

O parâmetro opcional `max_tokens` limita a saída via OpenRouter nos mappers
Single-Prompt e Multi-Step dos runners 1 e 3. Sem ele, o padrão do provedor
continua sendo usado. O limite de saída não garante que o prompt de entrada
caiba no contexto do modelo.

Por padrão, `filter_predicates` é `false`: o prompt continua recebendo todos
os predicados. Quando ativado, o filtro identifica colunas por sobreposição
lexical com os nomes e descrições do schema, sem acessar o GT nem chamar LLM.
Reutiliza a seleção de candidatos do Multi-Step e preserva os templates simples
de todas as colunas, para não eliminar conceitos implícitos na sentença.
O limite de candidatos é por sentença e não inclui os templates simples
preservados; em batches, o prompt recebe a união dos candidatos.

O espaço original e os arquivos de predicados permanecem inalterados. Essa
opção muda o espaço oferecido ao Single-Prompt e deve ser registrada como
adaptação do protocolo experimental. Reduzir tokens não garante preservar
todas as comparações compostas ou a qualidade de extração; valide o recorte
de candidatos e o tamanho do prompt antes de executar cada modelo.

### Saídas Estruturadas e Falhas de Saída do Modelo

Todas as chamadas de classificação e mapeamento usam *structured output* estrito
(`response_format: json_schema`) via OpenRouter. Os schemas ficam em
`src/llm/schemas.py` e são derivados do formato canônico das DCs: `rules` é `null`
ou uma lista de DCs independentes, e cada DC é um predicado (string) ou um nó
`{"AND": [...]}` / `{"OR": [...]}`, recursivamente.

| Componente | Schema | Formato |
|---|---|---|
| Classificador | `CLASSIFICATION` | `{"labels": [{"id": 1, "is_constraint": true}, ...]}` |
| Single-Prompt | `SENTENCE_RULES` | `{"results": [{"id": 1, "rules": [...] \| null}, ...]}` |
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

O runner de mapper GT-only mede:

- Correspondência exata e Jaccard de predicados.
- Precisão, recall, F1 e Jaccard de violações por sentença.
- Precisão, recall e F1 de violações agregadas sobre todas as sentenças avaliadas.

Os resultados são escritos no diretório definido por `results_dir`, com uma cópia da configuração, saídas por variante e métricas detalhadas quando aplicável.

### Validação HMDA Antes dos Experimentos

A checagem remota usa exemplos neutros, não carrega CSVs e registra cada
tentativa, incluindo uso de tokens, custo informado e erros do provedor:

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

As saídas novas ficam em `results/hmda/`. O harness aplica o limite de saída,
desativa os retries do SDK (o `OpenRouterClient` repete erros transitórios — 429,
5xx, timeout, conexão — com backoff exponencial), contabiliza cada tentativa e
verifica uma reserva conservadora antes de cada chamada. Erros de acesso/crédito interrompem novas
chamadas. O resumo registra fases incompletas, tempo e pico de memória.
Sem violações GT no recorte, métricas de descoberta de violações não servem
para comparar modelos. Não use resultados do smoke como reprodução de E1/E2/E3.

Reprodução completa (E1 → E2 → E3, até 1.000.000 de linhas por dataset):

```bash
.venv/bin/python experiments/hmda_make_configs.py   # gera hmda_e*.yaml e hmda_plan.json
# nova rodada (outros modelos): --models <manifesto.yaml> --config-dir <dir> --results-root results/<nome>
.venv/bin/python experiments/hmda_audit.py data/hmdaz --gt-file hmdaz_text2rule_gt.json
.venv/bin/python experiments/hmda_audit.py data/hmda_checks --gt-file hmda_regc_checks-text2rule-gt.json
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_rerun.yaml \
  --plan experiments/configs/hmda_plan.json   # cada job roda num subprocesso próprio
# retomar uma execução interrompida (pula jobs concluídos):
.venv/bin/python experiments/hmda_smoke.py experiments/configs/hmda_rerun.yaml \
  --plan experiments/configs/hmda_plan.json --resume results/hmda/telemetry/<timestamp>
```

O relatório fica em `results/hmda/REPORT.md`.

## Testes Rápidos

Execute os testes focados disponíveis no repositório:

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
