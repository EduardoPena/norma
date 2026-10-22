# OSHA ITA (Form 300A) — dataset candidato (Nível 2, baixo esforço)

Ver `osha_groundtruth_draft.md` para a análise completa (regras, violações medidas, ressalvas).

## Arquivos
- `osha_ita_data_dictionary.pdf`: texto normativo oficial (OSHA, data dictionary do upload CSV), fonte das 32 regras.
- `run_edits.py`: runner DuckDB que aplica as 32 regras sobre o CSV público e mede violações.
- `edit_results.csv`: resultado da última execução (CY2023, n=394.234).
- `osha_groundtruth_draft.md`: regras categorizadas, tabela de violações, ressalvas de transparência (leading-zero em EIN/zip, código de `size` legado).

## Dado (não versionado aqui — baixar de novo)

```sh
curl -L -o ita2023.zip "https://www.osha.gov/sites/default/files/ITA_300A_Summary_Data_2023_through_12-31-2024.zip"
unzip ita2023.zip
```

Gera `ITA 300A Summary Data 2023 through 12-31-2024_v2.csv` (~86MB, 394.234 linhas). Colocar no mesmo diretório de `run_edits.py` antes de rodar.

## Rodar

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install duckdb pandas numpy
python3 run_edits.py
```
