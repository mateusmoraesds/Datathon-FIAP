import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import FEATURES, load_panel  # noqa: E402


st.set_page_config(page_title="Radar de Defasagem | Passos Mágicos",
                   page_icon="🧭", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "artifacts" / "risk_model.joblib")


@st.cache_data
def load_data():
    return load_panel()


bundle = load_model()
models = bundle["production_models"]
thresholds = bundle["thresholds"]
panel, years = load_data()
with open(ROOT / "artifacts" / "metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

st.title("🧭 Radar de risco de defasagem")
st.caption("Priorização preventiva de acompanhamento — Passos Mágicos")

with st.sidebar:
    st.header("Sobre o modelo")
    st.write("Estima a chance de o aluno apresentar defasagem (fase atual abaixo "
             "da fase ideal) no ano seguinte.")
    st.metric("ROC-AUC temporal",
              f"{metrics['overall_temporal_test']['roc_auc']:.3f}")
    st.metric("Recall temporal",
              f"{metrics['overall_temporal_test']['recall']:.1%}")
    st.caption("Treino: 2022→2023 · teste: 2023→2024")
    st.caption("Produção 2025: reajuste com as duas transições disponíveis")
    st.warning("Use como apoio à equipe pedagógica. Não substitui avaliação humana.")
    with st.expander("Glossário oficial"):
        st.markdown("""
        - **IAN:** Indicador de Adequação ao Nível
        - **IDA:** Indicador de Aprendizagem
        - **IEG:** Indicador de Engajamento
        - **IAA:** Indicador de Autoavaliação
        - **IPS:** Indicador Psicossocial
        - **IPP:** Indicador Psicopedagógico
        - **IPV:** Indicador de Ponto de Virada
        - **INDE:** índice ponderado dos sete indicadores

        Fonte: *Dicionário de Dados Dataset PEDE_PASSOS*.
        """)

tab1, tab2, tab3, tab4 = st.tabs([
    "Visão geral", "Predição individual", "Carteira 2024", "Diagnóstico analítico"
])

with tab1:
    latest = years[2024]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos em 2024", f"{len(latest):,}".replace(",", "."))
    current_risk = (latest["defasagem"] < 0).mean()
    c2.metric("Com defasagem", f"{current_risk:.1%}")
    c3.metric("IDA médio", f"{latest['ida'].mean():.2f}")
    c4.metric("IEG médio", f"{latest['ieg'].mean():.2f}")
    yearly = panel.groupby("ano").agg(
        alunos=("RA", "nunique"), defasagem=("defasagem", lambda s: (s < 0).mean()),
        ida=("ida", "mean"), ieg=("ieg", "mean"), inde=("inde", "mean")
    ).reset_index()
    left, right = st.columns(2)
    left.plotly_chart(px.line(yearly, x="ano", y=["ida", "ieg", "inde"],
                              markers=True, title="Evolução dos indicadores médios"),
                      width="stretch")
    right.plotly_chart(px.bar(yearly, x="ano", y="defasagem",
                             title="Percentual de alunos com defasagem",
                             labels={"defasagem": "% em defasagem"}),
                       width="stretch")
    st.info("A prevalência de defasagem cai de 69,8% (2022) para 46,2% (2024). "
            "A composição da base muda entre os anos; portanto, isso é evidência "
            "descritiva, não uma estimativa causal do impacto do programa.")
    st.caption("Perfil operacional: moderada = 1 fase abaixo do nível ideal; "
               "severa = 2 ou mais fases abaixo.")

with tab2:
    st.subheader("Simular risco para o próximo ano")
    with st.form("prediction"):
        a, b, c = st.columns(3)
        idade = a.number_input("Idade", 6, 25, 12)
        anos_programa = a.number_input("Anos no programa", 0, 15, 2)
        fase_num = a.number_input("Fase (ALFA = 0)", 0, 8, 2)
        genero = a.selectbox("Gênero", ["Feminino", "Masculino", "Outro"])
        instituicao = b.selectbox("Instituição de ensino",
                                  ["Publica", "Privada", "Rede Decisao", "Outra"])
        iaa = b.slider("IAA — autoavaliação", 0.0, 10.0, 7.0, 0.1)
        ieg = b.slider("IEG — engajamento", 0.0, 10.0, 7.0, 0.1)
        ips = b.slider("IPS — psicossocial", 0.0, 10.0, 7.0, 0.1)
        ida = c.slider("IDA — desempenho", 0.0, 10.0, 7.0, 0.1)
        ipv = c.slider("IPV — ponto de virada", 0.0, 10.0, 7.0, 0.1)
        ian = c.slider("IAN — adequação de nível", 0.0, 10.0, 5.0, 0.5)
        inde = c.slider("INDE — nota global", 0.0, 10.0, 7.0, 0.1)
        defasagem = c.number_input("Defasagem atual", -6, 3, 0)
        submitted = st.form_submit_button("Calcular risco", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "idade": idade, "anos_programa": anos_programa,
            "fase_num": fase_num, "iaa": iaa, "ieg": ieg, "ips": ips,
            "ida": ida, "ipv": ipv, "ian": ian, "inde": inde,
            "defasagem": defasagem, "genero": genero,
            "instituicao": instituicao,
        }])[FEATURES]
        segment = "entrada" if defasagem >= 0 else "permanencia"
        probability = models[segment].predict_proba(row)[0, 1]
        threshold = thresholds[segment]
        label = ("Priorizar acompanhamento" if probability >= threshold
                 else "Acompanhamento regular")
        outcome = ("entrar em defasagem" if segment == "entrada"
                   else "permanecer em defasagem")
        st.metric(f"Probabilidade de {outcome}", f"{probability:.1%}", label)
        st.progress(float(probability))
        st.caption(f"Limiar operacional: {threshold:.1%}. A probabilidade é uma "
                   "estimativa populacional e deve ser interpretada com contexto.")

with tab3:
    scored = years[2024].copy()
    scored["segmento"] = np.where(scored["defasagem"] < 0,
                                   "permanencia", "entrada")
    scored["probabilidade_risco_2025"] = np.nan
    scored["prioridade"] = "Regular"
    for segment in ("entrada", "permanencia"):
        mask = scored["segmento"] == segment
        probability = models[segment].predict_proba(scored.loc[mask, FEATURES])[:, 1]
        scored.loc[mask, "probabilidade_risco_2025"] = probability
        scored.loc[mask, "prioridade"] = np.where(
            probability >= thresholds[segment], "Priorizar", "Regular")
    f1, f2 = st.columns(2)
    phase_options = sorted(scored["Fase"].dropna().astype(str).unique())
    phases = f1.multiselect("Filtrar fase", phase_options, default=phase_options)
    priority = f2.multiselect("Prioridade", ["Priorizar", "Regular"],
                              default=["Priorizar", "Regular"])
    view = scored[scored["Fase"].astype(str).isin(phases)
                  & scored["prioridade"].isin(priority)].sort_values(
                      "probabilidade_risco_2025", ascending=False)
    st.dataframe(view[["RA", "Fase", "Turma", "ida", "ieg", "ips", "ipv",
                       "ian", "defasagem", "segmento", "probabilidade_risco_2025",
                       "prioridade"]], width="stretch",
                 column_config={"probabilidade_risco_2025":
                                st.column_config.ProgressColumn(
                                    "Risco estimado", min_value=0, max_value=1,
                                    format="percent")})
    st.download_button("Baixar carteira filtrada (CSV)",
                       view.to_csv(index=False).encode("utf-8-sig"),
                       "carteira_risco_2025.csv", "text/csv")

with tab4:
    st.subheader("Relações entre indicadores")
    year = st.selectbox("Ano", [2024, 2023, 2022])
    data = years[year]
    x = st.selectbox("Eixo X", ["ieg", "ida", "iaa", "ips", "ipp", "ipv", "ian"])
    y = st.selectbox("Eixo Y", ["ida", "ipv", "inde", "ieg"], index=1)
    color = st.selectbox("Cor", ["pedra", "Fase", "genero"])
    st.plotly_chart(px.scatter(data, x=x, y=y, color=color, trendline="ols",
                               hover_data=["RA"],
                               title=f"{x.upper()} × {y.upper()} — {year}"),
                    width="stretch")
    correlations = data[["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian", "inde"]].corr()
    st.plotly_chart(px.imshow(correlations, text_auto=".2f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlação entre indicadores"),
                    width="stretch")
    calibration = pd.read_csv(ROOT / "artifacts" / "calibration.csv")
    st.plotly_chart(px.scatter(
        calibration, x="probabilidade_media", y="frequencia_observada",
        color="segmento", size="n", range_x=[0, 1], range_y=[0, 1],
        title="Calibração no teste temporal",
        labels={"probabilidade_media": "Probabilidade média",
                "frequencia_observada": "Frequência observada"}),
        width="stretch")
    st.caption("Correlação descreve associação, não causalidade. Consulte o notebook "
               "para as respostas detalhadas às 11 perguntas e a metodologia.")
