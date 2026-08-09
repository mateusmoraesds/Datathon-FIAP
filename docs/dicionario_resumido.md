# Glossário do Dataset PEDE_PASSOS

Fonte: `Base de Dados/Dicionário Dados Datathon.pdf`, Associação Passos
Mágicos.

| Campo | Definição oficial resumida |
|---|---|
| Fase | Nível de aprendizado do aluno |
| Turma | Subdivisão de uma fase, como 1A, 1B e 1C |
| IAN | Indicador de Adequação ao Nível |
| IDA | Indicador de Aprendizagem |
| IEG | Indicador de Engajamento |
| IAA | Indicador de Autoavaliação |
| IPS | Indicador Psicossocial |
| IPP | Indicador Psicopedagógico |
| IPV | Indicador de Ponto de Virada |
| INDE | Métrica geral ponderada por IAN, IDA, IEG, IAA, IPS, IPP e IPV |
| Nível ideal | Fase ideal do aluno na Passos Mágicos |
| Defasagem | Nível de defasagem registrado no ano |

## Classificação Pedra

- Quartzo: 2,405 a 5,506
- Ágata: 5,506 a 6,868
- Ametista: 6,868 a 8,230
- Topázio: 8,230 a 9,294

Como os limites aparecem sobrepostos no texto, o projeto usa a Pedra já
informada em cada CSV e não tenta recalculá-la.

## Decisões analíticas

- Risco no ano seguinte: `defasagem < 0`.
- Defasagem moderada: uma fase abaixo (`-1`).
- Defasagem severa: duas ou mais fases abaixo (`<= -2`).
- IPP não entra no modelo, pois não existe no CSV de 2022 usado no treino.
- O PDF descreve campos até 2022. O uso das colunas homônimas em 2023–2024
  assume continuidade semântica e deve ser confirmado com a organização.
