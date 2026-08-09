# Radar de risco de defasagem — Passos Mágicos

Solução do Datathon FIAP com análise longitudinal dos dados PEDE de 2022,
2023 e 2024, modelo de risco para o ano seguinte e aplicação Streamlit.

## Estrutura

- `notebooks/analise_risco_defasagem.ipynb`: análise das 11 perguntas,
  feature engineering, separação temporal, modelagem e avaliação.
- `src/data.py`: leitura, padronização e construção do painel longitudinal.
- `src/train_model.py`: treino reproduzível e geração dos artefatos.
- `artifacts/`: modelo treinado, métricas, importâncias e previsões de teste.
- `app.py`: aplicação Streamlit.
- `Base de Dados/Dicionário Dados Datathon.pdf`: fonte oficial para a
  interpretação dos campos.
- `docs/dicionario_resumido.md`: glossário e decisões de uso do dicionário.

## Executar

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Definição e validação

O alvo é `defasagem no ano seguinte < 0`. O modelo usa informações do ano
atual e é treinado em transições 2022→2023. A avaliação final é temporal,
em alunos observados em 2023→2024. O limiar foi escolhido no conjunto de
treino para privilegiar identificação preventiva.

O IPP é analisado no notebook, mas não usado pelo modelo porque está ausente
em 2022. Essa escolha mantém as mesmas variáveis disponíveis no treino e no
uso futuro.

O PDF confirma que fase representa nível de aprendizado e que INDE é a
ponderação de IAN, IDA, IEG, IAA, IPS, IPP e IPV. Como o documento detalha os
campos somente até 2022, a equivalência semântica das colunas homônimas de
2023–2024 é uma hipótese de continuidade, explicitada no notebook.

## Deploy no Streamlit Community Cloud

1. Publique este repositório no GitHub.
2. Acesse o Streamlit Community Cloud e selecione **Create app**.
3. Escolha o repositório, a branch e informe `app.py` como arquivo principal.
4. Confirme o deploy. Não há secrets necessários.

Os CSVs e `artifacts/risk_model.joblib` precisam estar versionados no
repositório. Dados anonimizados ainda devem seguir a política de acesso da
organização.
