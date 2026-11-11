# Datasets HMDA (HMDAZ e HMDA Checks): tamanho, corte, ano e fonte (07/11/2026)

As contagens foram feitas em 07/11/2026 com DuckDB, em streaming sobre os arquivos inteiros. Nenhum dos dois CSVs foi carregado inteiro na memória.

## Resumo

| | HMDAZ | HMDA Checks |
|---|---|---|
| Arquivo | `data/hmdaz/hmdaz.csv` (11,6 GB) | `data/hmda_checks/hmda.csv` (3,9 GB) |
| Ano dos dados | 2018 (`activity_year` = 2018 em todas as linhas) | 2014 (`activity_year` = 2014 em todas as linhas) |
| Total de registros | **15.119.651** | **4.832.425** |
| Colunas | 135 (código e rótulo de cada campo categórico) | 81 |
| Texto regulatório | Regulation Z, §§ 1026.32, 1026.35 e 1026.43 (`regulation_z-2018-section_32_35_43.pdf`) | Especificação de *edits* do HMDA 2014, publicada pelo FFIEC (`hmda-2014-edits.pdf`) |
| Registros usados no Norma e no XiYan-SQL | **1.000.000** | **1.000.000** |
| Registros usados no DC Discovery | 100.000, só as 13 colunas do GT | Tentativas com 100k, 50k, 20k e 10k, só as 39 colunas do GT; nenhuma terminou (DNF) |

## Como foi feito o corte

- **Norma e XiYan-SQL:** usam as **primeiras 1M linhas** do arquivo, na ordem em que estão gravadas, sem embaralhar.
  - O `DatasetLoader` lê com `pd.read_csv(..., nrows=sample_rows)` (`src/data/loader.py`, linhas 55–58).
  - As configurações usam `sample_rows: 1000000`.
  - O artigo diz o mesmo: "we use the first 1M rows of each file".
- **DC Discovery:** usa uma **amostra aleatória com semente 42**, tirada dessas mesmas primeiras 1M linhas.
  - Os manifestos estão em `baselines/1_profiling/trimmed_datasets_v3/*.manifest.json`.
  - Eles registram `"slice": "first 1M rows"`, `"seed": 42` e o tamanho da amostra.
- **Cobertura das primeiras 1M linhas:**
  - **HMDAZ:** 53 dos 57 valores de `state_code` e 533 das 5.678 instituições (`lei`). O arquivo parece estar ordenado por instituição, então a fatia não é uma amostra aleatória das instituições.
  - **HMDA Checks:** 52 dos 52 valores de `state_code` e 5.314 instituições (`respondent_id`).

## O arquivo de 2014 já vem filtrado

As 4.832.425 linhas do HMDA Checks são um subconjunto do LAR nacional de 2014, e não o LAR completo. Todas atendem às quatro condições abaixo:

| Coluna | Valor | Significado |
|---|---|---|
| `action_taken_type` | 1 | operação contratada |
| `lien_status` | 1 | primeira hipoteca |
| `occupancy` | 1 | residência do proprietário |
| `property_type` | 1 | imóvel de 1 a 4 unidades |

O arquivo de 2018 (HMDAZ) não tem esse tipo de filtro: são 15,1 milhões de registros, com todas as decisões.

## Fonte

- Os dois arquivos vêm dos dados públicos, registro a registro, do *Home Mortgage Disclosure Act*, divulgados pelo CFPB e pelo FFIEC.
- O repositório não registra a URL nem a data de download de nenhum dos dois CSVs. A única referência que aparece é a documentação dos campos (`ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields`), citada no esquema e no glossário do HMDAZ.
- Pela época, a origem mais provável é a seguinte:
  - **2018:** o *snapshot* nacional do LAR no HMDA Data Browser do FFIEC/CFPB. O total de 15,1 milhões de registros é compatível com ele.
  - **2014:** os *HMDA historic data* do CFPB, com um dos filtros pré-definidos (contratadas, primeira hipoteca, residência do proprietário, 1 a 4 unidades).
- **Pendente:** antes de citar a fonte no artigo, confirmar com quem baixou os arquivos qual foi o link e o filtro exatos.
