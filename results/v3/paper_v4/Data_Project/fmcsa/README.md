# FMCSA Company Census File — dataset candidato

Ver `fmcsa_groundtruth_draft.md` para a análise completa (regras, violações medidas, ressalvas).

## Arquivos
- `run_edits.py`: runner DuckDB, 16 regras.
- `edit_results.csv`: resultado da última execução (4.479.438 registros, 2.233.109 ativos).
- `fmcsa_groundtruth_draft.md`: regras categorizadas, achado principal (43,6% de MCS-150 desatualizado), ressalvas.

## Dado (não versionado aqui — 2GB, baixar de novo)

```sh
curl -L -o fmcsa.csv "https://data.transportation.gov/api/v3/views/az4n-8mr2/export.csv?accessType=DOWNLOAD"
```

## Rodar

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install duckdb pandas numpy
python3 run_edits.py
```
