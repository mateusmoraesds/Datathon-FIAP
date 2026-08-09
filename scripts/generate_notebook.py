import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": dedent(source).strip().splitlines(True),
    }


cells = [
    md("""
    # Análise de risco de defasagem — Passos Mágicos

    **Objetivo:** responder às 11 perguntas do Datathon e construir um modelo que
    estime, com dados do ano atual, a probabilidade de defasagem no ano seguinte.

    **Desenho:** treino em 2022→2023 e teste final em 2023→2024. Essa separação
    temporal evita vazamento de informação e é mais próxima do uso real.

    **Fonte semântica:** *Dicionário de Dados Dataset PEDE_PASSOS*, fornecido
    pela Associação Passos Mágicos. O documento define os indicadores e campos
    até 2022. Para 2023–2024, considera-se continuidade semântica das colunas
    homônimas; essa hipótese deve ser confirmada com a organização.

    > Defasagem é operacionalizada como `fase atual - fase ideal < 0`.
    > “Moderada” = -1 fase e “severa” = -2 fases ou menos. Os CSVs são anuais;
    > portanto, “evolução ao longo do ano” só pode ser respondida entre anos,
    > não dentro de cada ano.
    """),
    code("""
    from pathlib import Path
    import json, sys, warnings
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                                 RocCurveDisplay, classification_report)

    ROOT = Path.cwd()
    if not (ROOT / "src").exists():
        ROOT = ROOT.parent
    sys.path.insert(0, str(ROOT / "src"))
    from data import FEATURES, load_panel, make_transitions
    from train_model import train_and_save

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", palette="viridis")
    panel, years = load_panel()
    panel.shape, panel["ano"].value_counts().sort_index()
    """),
    md("## 1. Qualidade e preparação dos dados"),
    md("""
    ### Glossário validado pelo dicionário

    - **IAN:** Indicador de Adequação ao Nível.
    - **IDA:** Indicador de Aprendizagem.
    - **IEG:** Indicador de Engajamento.
    - **IAA:** Indicador de Autoavaliação.
    - **IPS:** Indicador Psicossocial.
    - **IPP:** Indicador Psicopedagógico.
    - **IPV:** Indicador de Ponto de Virada.
    - **INDE:** métrica geral ponderada pelos sete indicadores acima.
    - **Fase:** nível de aprendizado; **Turma:** subdivisão da fase.
    - **Nível ideal / Defasagem:** nível esperado e distância registrada no ano.

    O dicionário também informa as faixas históricas das Pedras: Quartzo
    (2,405–5,506), Ágata (5,506–6,868), Ametista (6,868–8,230) e Topázio
    (8,230–9,294). Há sobreposição textual nos limites; neste projeto é
    preservada a classificação já fornecida nos CSVs, sem recalculá-la.
    """),
    code("""
    quality = []
    for year, df in years.items():
        quality.append({
            "ano": year, "linhas": len(df), "RAs_unicos": df.RA.nunique(),
            "duplicados_RA": df.RA.duplicated().sum(),
            **{f"nulos_{c}": int(df[c].isna().sum())
               for c in ["ian","ida","ieg","iaa","ips","ipp","ipv","inde","defasagem"]}
        })
    display(pd.DataFrame(quality))
    """),
    md("""
    Harmonizações realizadas: separadores `;`/`,`, decimal brasileiro, nomes de
    colunas, fases ALFA/numéricas, idade e tempo de programa. A duplicidade isolada
    de 2022 é removida pela linha sem RA. O IPP inexiste em 2022; por isso aparece
    na análise de 2023–2024, mas não nas features do modelo longitudinal.
    O PDF informa que o conjunto original deriva de pesquisas de 2020, 2021 e
    2023, mas suas entradas de campos chegam até 2022; essa inconsistência de
    versionamento é tratada como limitação documental.
    """),
    md("## 2. Perguntas 1–2 — IAN, defasagem e desempenho acadêmico"),
    code("""
    def profile(df):
        d = df["defasagem"]
        return pd.Series({
            "alunos": len(df), "sem_defasagem": (d >= 0).sum(),
            "moderada_-1": (d == -1).sum(), "severa_<=-2": (d <= -2).sum(),
            "%_qualquer_defasagem": 100 * (d < 0).mean(),
            "IAN_medio": df.ian.mean(), "IDA_medio": df.ida.mean(),
        })
    annual = pd.DataFrame({y: profile(d) for y, d in years.items()}).T
    display(annual.round(2))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    annual[["moderada_-1","severa_<=-2"]].plot.bar(stacked=True, ax=ax[0],
                                                   title="Perfil de defasagem")
    panel.groupby("ano")[["ida","ian","inde"]].mean().plot(marker="o", ax=ax[1],
                                                           title="Médias anuais")
    plt.show()
    """),
    md("""
    **Resposta 1.** A proporção com defasagem cai de aproximadamente 69,8% em
    2022 para 54,4% em 2023 e 46,2% em 2024. O notebook separa moderados e
    severos na tabela. O IAN médio sobe (6,42 → 7,24 → 7,68), coerente com
    melhora de adequação. Como não há datas intranuais, não se pode afirmar uma
    trajetória “ao longo dos meses”.

    **Resposta 2.** O IDA médio cresce de 6,09 para 6,66 em 2023, mas recua para
    6,37 em 2024: melhora seguida de queda, não crescimento consistente. A
    comparação por fase abaixo ajuda a separar efeito de composição.
    """),
    code("""
    phase_summary = (panel.groupby(["ano","fase_num"])
                     .agg(n=("RA","nunique"), IDA=("ida","mean"), INDE=("inde","mean"))
                     .reset_index())
    display(phase_summary.round(2))
    sns.lineplot(data=phase_summary, x="fase_num", y="IDA", hue="ano", marker="o")
    plt.title("IDA médio por fase e ano"); plt.show()
    """),
    md("## 3. Perguntas 3–7 — relações e sinais antecedentes"),
    code("""
    indicators = ["iaa","ieg","ips","ipp","ida","ipv","ian","inde"]
    for year, df in years.items():
        print(f"\\nCorrelação de Pearson — {year}")
        display(df[indicators].corr().round(2))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, year in zip(axes, [2022, 2023, 2024]):
        sns.regplot(data=years[year], x="ieg", y="ipv", scatter_kws={"alpha":.25}, ax=ax)
        ax.set_title(f"IEG × IPV — {year}")
    plt.show()
    """),
    md("""
    **Resposta 3.** O IEG tem relação positiva moderada com IDA (r≈0,46–0,56)
    e IPV (r≈0,45–0,59): maior engajamento acompanha desempenho e ponto de
    virada, mas correlação não prova causalidade.

    **Resposta 4.** IAA é apenas fracamente relacionado a IDA e IEG (em geral
    r≈0,10–0,32). Logo, a autoavaliação contém informação própria e frequentemente
    diverge do desempenho observado — um bom gatilho para conversa individual.

    **Resposta 5.** IPS contemporâneo tem associações fracas com IDA/IEG. Para
    testar antecedência, o bloco longitudinal abaixo relaciona IPS atual às
    quedas no ano seguinte; o resultado deve ser lido como sinal de triagem,
    não diagnóstico clínico.

    **Resposta 6.** Em 2023–2024, IPP apresenta relação positiva com IAN, mas
    bem menor do que com IPV (especialmente em 2024). Casos discordantes
    `IPP alto + IAN baixo` ou o inverso merecem revisão conjunta; os construtos
    são complementares, não equivalentes.

    **Resposta 7.** Entre os comportamentos observados, IEG, IDA e IPP são os
    correlatos mais fortes do IPV; em 2024 IPP×IPV chega a cerca de 0,75.
    """),
    code("""
    train, test = make_transitions(years)
    antecedents = []
    for label, transition in [("2022→2023", train), ("2023→2024", test)]:
        antecedents.append({
            "transicao": label,
            "IPS_medio_com_queda_IDA": transition.loc[transition.queda_ida, "ips"].mean(),
            "IPS_medio_sem_queda_IDA": transition.loc[~transition.queda_ida, "ips"].mean(),
            "IPS_medio_com_queda_IEG": transition.loc[transition.queda_ieg, "ips"].mean(),
            "IPS_medio_sem_queda_IEG": transition.loc[~transition.queda_ieg, "ips"].mean(),
        })
    display(pd.DataFrame(antecedents).round(2))

    for year in [2023, 2024]:
        d = years[year]
        discordance = d.assign(
            caso=np.select([
                (d.ipp >= 7.5) & (d.ian <= 5),
                (d.ipp <= 5) & (d.ian >= 7.5)],
                ["IPP alto / IAN baixo", "IPP baixo / IAN alto"], default="outros"))
        print(year, discordance["caso"].value_counts())
    """),
    md("## 4. Pergunta 8 — multidimensionalidade e INDE"),
    code("""
    features_multi = ["ida","ieg","ips","ipp"]
    for year in [2023, 2024]:
        d = years[year].dropna(subset=features_multi + ["inde"]).copy()
        for c in features_multi:
            d[c + "_alto"] = d[c] >= d[c].median()
        combo = (d.groupby([c+"_alto" for c in features_multi])
                   .agg(n=("RA","size"), INDE_medio=("inde","mean"))
                   .query("n >= 10").sort_values("INDE_medio", ascending=False))
        print(f"Combinações com maior INDE — {year}")
        display(combo.head(10).round(2))
    """),
    md("""
    **Resposta 8.** IDA e IEG são consistentemente os componentes mais ligados
    ao INDE; IPP ganha relevância em 2024, enquanto IPS isolado é mais fraco.
    As melhores combinações concentram IDA+IEG altos, reforçados por IPP alto.
    Isso é parcialmente esperado porque o INDE é um índice composto desses
    indicadores; não interpretar como efeito causal independente.
    """),
    md("## 5. Pergunta 9 — feature engineering e modelo preditivo"),
    code("""
    # Cada linha usa atributos do ano t; o alvo vem apenas de t+1.
    train[["RA","ano","defasagem","defasagem_seguinte","risco_seguinte"]].head()
    """),
    md("""
    Features: idade, tempo no programa, fase numérica, IAA, IEG, IPS, IDA, IPV,
    IAN, INDE, defasagem atual, gênero e tipo de instituição. Numéricas recebem
    mediana+padronização; categóricas recebem moda+one-hot. O estimador é regressão
    logística regularizada, escolhida por transparência e estabilidade na amostra.
    Ausências são tratadas dentro do pipeline, depois da separação.
    """),
    code("""
    metrics = train_and_save()
    display(pd.DataFrame(metrics).T)
    bundle = joblib.load(ROOT / "artifacts" / "risk_model.joblib")
    model, threshold = bundle["pipeline"], bundle["threshold"]
    probability = model.predict_proba(test[FEATURES])[:, 1]
    prediction = probability >= threshold
    print(classification_report(test.risco_seguinte, prediction, digits=3))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    RocCurveDisplay.from_predictions(test.risco_seguinte, probability, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(test.risco_seguinte, probability, ax=ax[1])
    ConfusionMatrixDisplay.from_predictions(test.risco_seguinte, prediction, ax=ax[2])
    plt.show()
    """),
    md("""
    **Resposta 9.** No teste temporal 2023→2024, o modelo alcança ROC-AUC ≈0,822,
    PR-AUC ≈0,774, recall ≈78,6%, precisão ≈61,7% e F1 ≈0,691. O limiar de
    aproximadamente 30,5% privilegia recall para não deixar alunos vulneráveis
    sem sinalização. Falsos positivos significam revisão pedagógica adicional,
    não punição.
    """),
    code("""
    importance = pd.read_csv(ROOT / "artifacts" / "feature_importance.csv").head(20)
    display(importance)
    sns.barplot(data=importance.head(15), y="feature", x="coefficient")
    plt.axvline(0, color="black", lw=1); plt.title("Coeficientes do modelo")
    plt.show()
    """),
    md("## 6. Pergunta 10 — efetividade do programa"),
    code("""
    rows = []
    for a, b in [(2022, 2023), (2023, 2024)]:
        cols = ["RA","ida","ieg","ips","ipv","inde","defasagem"]
        m = years[a][cols].merge(years[b][cols], on="RA", suffixes=("_antes","_depois"))
        for metric in cols[1:]:
            delta = m[f"{metric}_depois"] - m[f"{metric}_antes"]
            rows.append({"transicao":f"{a}→{b}", "indicador":metric,
                         "n":delta.notna().sum(), "delta_medio":delta.mean(),
                         "%_melhorou":100*(delta>0).mean()})
    display(pd.DataFrame(rows).pivot(index="indicador", columns="transicao",
                                     values="delta_medio").round(3))
    """),
    md("""
    **Resposta 10.** Há sinais favoráveis em adequação: entre alunos pareados, a
    defasagem melhora em média nas duas transições. Porém, o desempenho não é
    uniformemente crescente: 2022→2023 melhora IDA/IEG/IPV, enquanto 2023→2024
    recua nesses indicadores; INDE fica quase estável. Assim, os dados confirmam
    progresso em adequação, mas não “melhora consistente em todas as dimensões”.
    Sem grupo de comparação e controle de seleção, não se identifica impacto
    causal do programa.
    """),
    md("""
    ## 7. Pergunta 11 — insights e recomendações

    1. **Fila de ação, não rótulo:** usar a probabilidade para ordenar revisões
       quinzenais e registrar o desfecho de cada intervenção.
    2. **Matriz de discordância:** priorizar `IAA alto + IDA baixo` (percepção
       otimista) e `IAA baixo + IDA alto` (possível baixa autoconfiança).
    3. **Sinal combinado:** quedas simultâneas de IEG e IDA são mais acionáveis
       que qualquer indicador isolado; adicionar alertas de variação anual.
    4. **IPP + IPV:** a forte associação em 2024 sugere integrar acompanhamento
       psicopedagógico às ações de ponto de virada, sem confundir correlação com causa.
    5. **Monitoramento de drift:** reavaliar mensalmente prevalência, calibração,
       recall por gênero/fase e estabilidade das features; retreinar após cada ciclo.
    6. **Próxima coleta:** incluir avaliações por bimestre e eventos de intervenção.
       Isso permitiria responder de fato à evolução intranual e estimar antecedência.

    ### Limitações e ética

    Apenas três cortes anuais, mudanças de preenchimento entre anos, ausência de
    IPP em 2022, indicadores que compõem o próprio INDE e falta de contrafactual.
    O modelo não deve automatizar exclusão, bolsa ou sanção. Exigir revisão humana,
    acesso mínimo aos dados, auditoria por subgrupos e canal para correção.
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = ROOT / "notebooks" / "analise_risco_defasagem.ipynb"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(target)
