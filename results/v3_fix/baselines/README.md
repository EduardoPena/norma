# Baselines (DC Discovery, XiYan-SQL): rodada v3, protocolo único nos 6 datasets

Os baselines do artigo foram rodados de novo nos seis datasets e avaliados com o mesmo avaliador do Norma:
- primeiras 1M linhas nos HMDA;
- GTs atuais;
- semântica de pares `t1.tid <> t2.tid`;
- `MappingEvaluator`.

Os números substituem os da submissão anterior, que eram copiados da Tabela 3 da SIGMOD e não existiam para os HMDA.

- **Tabela:** `table/v3_table3_end2end.{tex,csv}`. Gerada com:
  ```bash
  V3_RESULTS=results/v3_fix E3_ROOTS=results/v3_fix,results/v3_fix_rep2,results/v3_fix_rep3 \
  BASELINES=results/v3_fix/baselines/baselines_final.csv TABLE_OUT=results/v3_fix/baselines/table \
  .venv/bin/python results/v3/analysis/table3.py
  ```
- **Dados:** `baselines_final.csv`, com as linhas de DC Discovery de `summary.csv` e as do XiYan-SQL (variante A) de `summary_xiyan.csv`.
- **Logs:** `dc_logs/`, com `runs.tsv` registrando cada execução do `.jar` (ok, timeout, oom ou interrupted).

## DC Discovery (`baselines/1_profiling/dc_discovery/DCDiscovery.jar`, FDCD)

- **Entradas:** `experiments/baselines/prepare_dc_discovery.py`, gravadas em `baselines/1_profiling/trimmed_datasets_v3/` com um manifesto por arquivo. Seguem o Step 1 do artigo:
  - SISU, Restaurant e HMDA restritos às colunas do GT; Hospital e Tax com todas as colunas;
  - tipos herdados das entradas do artigo quando elas existem (por exemplo, os códigos do SISU como `str`);
  - identificadores longos ou com zeros à esquerda como `str`;
  - HMDA: amostra aleatória (semente 42) das primeiras 1M linhas.
- **Execução:** `experiments/baselines/run_dc_discovery.sh`.
  - O `.jar` exige Java ≥ 20; foi usado o Temurin 21 em `~/.local/jdk-21`.
  - `-Duser.language=en` é obrigatório: com o locale pt-BR, a JVM escreve vírgula decimal e o CSV de saída fica corrompido.
  - Grade: exato e ε ∈ {0,001, 0,005, 0,01, 0,05, 0,1}.
- **Verificação:** no Hospital, as saídas são idênticas byte a byte às do artigo (exato e ε=0,01), e a violação reproduz a Tabela 3 (0,83 / 1,00 / 0,91).
- **Escalabilidade:**
  - Hospital, Tax, Restaurant e HMDAZ (100 mil linhas): a grade inteira conclui.
  - SISU (100 mil linhas, 24 colunas): o exato estoura a memória com 8 GB; ε de 0,001 a 0,01 dão timeout de 2 h; ε=0,05 e 0,1 concluem (no artigo, só 0,1).
  - **HMDA Checks (39 colunas): não concluiu.** Com 100 mil linhas, timeout de 6 h no exato e no ε=0,001, e o ε=0,005 foi interrompido. Com 50 mil, timeout de 2 h em todos os ε. Com 20 mil, estouro de memória (6 GB) em todos os ε. Última tentativa (30–31/10), com a máquina livre e heap de 12 GB: 20 mil e 10 mil linhas estouram a memória em todos os ε, em 20–60 min, na fase de enumeração. Na tabela, aparece como DNF.
- **Avaliação:** `evaluate_baselines_v3.py profiling`. Ranking pelo escore misto, top-k ∈ {5, 10, 20, 50, 100}. Na tabela entra o **melhor da grade, por métrica**: o melhor F1 de extração e o melhor F1 de violação, cada um com sua configuração em `baselines_final.csv`.
- **Diferenças em relação à submissão anterior:**
  - Restaurant: violação 0,41 / 1,00 / 0,58 contra 0,77 / 0,87 / 0,82. **Não é efeito das colunas:** as DCs da descoberta antiga (23/03, sem `inspection_date`), reavaliadas com o avaliador e o GT atuais (03/11), dão o mesmo 0,58 em todas as configurações aproximadas (o exato marca 0 tuplas). Em toda a grade, a DC mais bem ranqueada é do tipo `not(t1.grade_date==t2.grade_date)`, que marca 127.683–127.692 das 127.692 linhas; 0,58 é o escore de marcar todas as tuplas (41% das linhas violam o gold). A diferença em relação ao valor antigo vem do GT revisado (`23317a7` → `736387a`) e/ou da semântica de pares corrigida; o valor antigo não pode ser reconstruído, porque o notebook não registra a violação do Restaurant;
  - os demais datasets reproduzem a submissão (extração 0 em todos).

## XiYan-SQL (`XGenerationLab/XiYanSQL-QwenCoder-14B-2504`, bf16)

- **Execução:** RunPod, A100-SXM4-80GB, torch 2.6.0+cu124, transformers 5.20.2. Pacote em `results/baselines_v3_gpu_package/`.
  - Script `run_xiyan_v3.py`, que importa o prompt, o DDL e a limpeza de SQL do `eval_xiyan_duck.py` (variante A) sem alterá-los.
  - 3 sementes (0, 1, 2), `temperature` 0,1, `top_p` 0,8.
  - Saídas em `xiyan_run/outputs/<ds>/seed<k>/`.
  - O hash da versão do modelo não foi registrado (`model_revision` vazio).
- **Avaliação:** `evaluate_baselines_v3.py xiyan`, média das 3 sementes.
  - **Classificação:** consulta não nula conta como constraint prevista. Também é registrado o critério do script original, que exclui consultas com falha de execução (`clf_*_paper`).
  - **Violação:** cada consulta é executada como foi gerada, com qualquer SQL, no DuckDB.
  - **Extração:** os predicados do `WHERE` são lidos por `experiments/baselines/sql_rules.py`:
    - `EXISTS` correlacionado vira DC de pares;
    - `NOT` é empurrado para os predicados (De Morgan);
    - predicados de `tid` são descartados.

    As consultas sem leitura como DC (agregação, `NOT EXISTS`, `CASE`, funções) são pontuadas **por equivalência de execução**: contam como acerto se devolvem as mesmas tuplas violadoras, não vazias, que a DC do gold da sentença, e como predição errada caso contrário. Nenhuma delas foi equivalente nesta execução. A extração só por predicados está em `constraint_exact_f1_dc_only`.
- **Procedência:**
  - os números reexecutados reproduzem os resumos que já estavam no repositório (`baseline_results/text2sql_results`, prompt A, 14B);
  - não reproduzem a Tabela 3 da submissão (por exemplo, classificação do Hospital 0,41 contra 0,67; extração 0,05 contra 0,75);
  - o prompt B (`xyyan_text2sql_inference.py`, 23/03, modelo padrão de 32B) é o candidato provável para a origem da tabela antiga;
  - a variante B (32B, `run_all_B.sh`) foi preparada, mas **ainda não foi executada**.

## Limites da avaliação

- Na primeira avaliação (`summary.csv`, `summary_xiyan.csv`), as consultas dos baselines tinham limite de 60 s (`NORMA_SQL_TIMEOUT_S=60`); a versão final usa 300 s, como o Norma (seção abaixo). Uma DC de pares que estoura conta como erro de execução. Linhas parciais avaliadas com 300 s estão em `summary_partial_timeout300s.csv` e não entram na tabela.
- No HMDA Checks, a violação agregada é pouco informativa: 98,6% das linhas violam alguma regra do gold.

## Versão final usada no artigo (31/10)

- **Limite de consulta:** 300 s, o mesmo do Norma (`summary_300s.csv`, `summary_xiyan_300s.csv`, `summary_xiyanB_300s.csv`; logs em `logs_300s/`). Os melhores resultados são idênticos aos obtidos com 60 s.
- **XiYan-SQL:** reportada a **variante B**: prompt do `xyyan_text2sql_inference.py`, modelo `XiYanSQL-QwenCoder-32B-2504`, dialeto SQLite traduzido para DuckDB, 3 sementes.
  - Ela reproduz a Tabela 3 da submissão anterior. No Hospital, por exemplo: classificação 0,68, extração 0,80 e violação 0,91, contra 0,67, 0,75 e 0,91.
  - A variante A (14B) fica documentada acima e em `summary_xiyan_300s.csv`.
- **Tabela e dados:** `table_300s/v3_table3_end2end.{tex,csv}`, gerada a partir de `baselines_final_300s.csv` e copiada para `results/v3/version_icde27` e `results/v3/version_pvldb20`.
