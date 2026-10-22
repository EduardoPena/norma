# version_pvldb20: submissão ao PVLDB Vol. 20 (VLDB 2027)

- **Template:** oficial, de https://github.com/vldbproceedings/VLDB-Template (commit 39c95f5, 2026-07-13). Usa `acmart.cls` v2.19 e `pvldb.sty` v1.0, que põe as seções em CAIXA ALTA e gera os blocos *PVLDB Reference Format* e *Artifact Availability* com `\vldbtopmatter`. Esses arquivos não devem ser editados.
- **Regras do Vol. 20:**
  - research paper com até 12 páginas, sem contar as referências; apêndices e agradecimentos contam no limite;
  - revisão single-blind;
  - prazos mensais no dia 1, com resumo obrigatório até o dia 25 do mês anterior; último prazo em 1º/03/2027;
  - artefatos públicos obrigatórios.
- **Arquivos:**
  - `text2Constraints_pvldb20.tex/.pdf` é a submissão: 12 páginas de corpo + referências, sem marcas de revisão.
  - `text2Constraints_pvldb20_full.tex/.pdf` é a versão estendida, com o texto integral da `version_fix`. É citada no artigo como `normafull` e deve ser publicada no repositório de artefatos (ou no arXiv, atualizando a entrada `normafull` do `refs.bib`) antes da submissão.
- **Fontes:** `text/` tem o texto condensado da submissão; `text_full/` tem o texto da `version_fix`, com as marcas `\add`/`\subs`/`\rem` ocultas por `\Reviewfalse`.
