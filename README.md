# Radar de risco de defasagem — Passos Mágicos

Solução do Datathon FIAP com análise longitudinal dos dados PEDE de 2022,
2023 e 2024, modelo de risco para o ano seguinte e aplicação Streamlit.

## Estrutura

- `notebooks/analise_risco_defasagem.ipynb`: análise das 11 perguntas,
  feature engineering, separação temporal, modelagem e avaliação.
- `src/data.py`: leitura, padronização e construção do painel longitudinal.
- `src/train_model.py`: treino reproduzível, limiares out-of-fold e geração
  dos artefatos de avaliação e produção.
- `artifacts/`: modelo treinado, métricas, calibração, comparação de
  modelos, auditoria por subgrupos, importâncias e previsões de teste.
- `app.py`: aplicação Streamlit.
- `Base de Dados/Dicionário Dados Datathon.pdf`: fonte oficial para a
  interpretação dos campos.
- `docs/dicionario_resumido.md`: glossário e decisões de uso do dicionário.
- `tests/`: testes automatizados de dados, modelo, notebook e Streamlit.

## Executar

```bash
pip install -r requirements.txt
python src/train_model.py
python scripts/generate_notebook.py --execute
python -m unittest discover -s tests -v
streamlit run app.py
```

## Definição e validação

Existem dois segmentos. Para alunos atualmente adequados, o alvo é entrar em
defasagem no ano seguinte. Para alunos já defasados, o alvo é permanecer em
defasagem. A avaliação é treinada em 2022→2023 e testada temporalmente em
2023→2024. Os limiares são escolhidos por previsões out-of-fold apenas no
treino. Depois da avaliação, os modelos de produção são reajustados com todas as
transições disponíveis para estimar 2025.

O IPP é analisado no notebook, mas não usado pelo modelo porque está ausente
em 2022. Essa escolha mantém as mesmas variáveis disponíveis no treino e no
uso futuro.

O PDF confirma que fase representa nível de aprendizado e que INDE é a
ponderação de IAN, IDA, IEG, IAA, IPS, IPP e IPV. Como o documento detalha os
campos somente até 2022, a equivalência semântica das colunas homônimas de
2023–2024 é uma hipótese de continuidade, explicitada no notebook.

Gênero e tipo de instituição são harmonizados entre as nomenclaturas dos três
anos antes do treino e da inferência.

## Deploy no Streamlit Community Cloud

1. Publique este repositório no GitHub.
2. Acesse o Streamlit Community Cloud e selecione **Create app**.
3. Escolha o repositório, a branch e informe `app.py` como arquivo principal.
4. Confirme o deploy. Não há secrets necessários.

Os CSVs e `artifacts/risk_model.joblib` precisam estar versionados no
repositório. Dados anonimizados ainda devem seguir a política de acesso da
organização.
