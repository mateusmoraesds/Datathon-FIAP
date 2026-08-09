import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ROOT,
                  load_panel, make_transitions)


ARTIFACT_DIR = ROOT / "artifacts"
SEGMENTS = {
    "entrada": {"filter": lambda d: d["defasagem"] >= 0,
                "description": "Entrar em defasagem partindo de situação adequada."},
    "permanencia": {"filter": lambda d: d["defasagem"] < 0,
                    "description": "Permanecer em defasagem no ano seguinte."},
}


def build_pipeline(estimator=None):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[
                ["Feminino", "Masculino", "Outro"],
                ["Publica", "Privada", "Rede Decisao", "Outra"],
            ],
            handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    if estimator is None:
        estimator = LogisticRegression(max_iter=3000, C=0.5)
    return Pipeline([
        ("prep", prep),
        ("model", estimator),
    ])


def select_threshold_oof(X, y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probability = cross_val_predict(
        build_pipeline(), X, y, cv=folds, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y, probability >= value) for value in thresholds]
    return float(thresholds[int(np.argmax(scores))]), probability


def metric_dict(y, probability, prediction, thresholds=None):
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
        "thresholds": thresholds,
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
    }


def calibration_table(y, probability, segment, bins=5):
    frame = pd.DataFrame({"observado": np.asarray(y), "probabilidade": probability})
    frame["faixa"] = pd.qcut(frame["probabilidade"], q=bins, duplicates="drop")
    result = frame.groupby("faixa", observed=True).agg(
        n=("observado", "size"), probabilidade_media=("probabilidade", "mean"),
        frequencia_observada=("observado", "mean")).reset_index()
    result["faixa"] = result["faixa"].astype(str)
    result.insert(0, "segmento", segment)
    return result


def coefficient_table(pipeline, segment):
    names = pipeline["prep"].get_feature_names_out()
    values = pipeline["model"].coef_[0]
    return pd.DataFrame({
        "segmento": segment, "feature": names, "coefficient": values,
        "abs_coefficient": np.abs(values),
    }).sort_values(["segmento", "abs_coefficient"], ascending=[True, False])


def compare_candidates(X, y, segment):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Dummy prevalencia": DummyClassifier(strategy="prior"),
        "Regressao logistica": LogisticRegression(max_iter=3000, C=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1),
    }
    rows = []
    for name, estimator in candidates.items():
        probability = cross_val_predict(
            build_pipeline(estimator), X, y, cv=folds,
            method="predict_proba", n_jobs=-1)[:, 1]
        rows.append({
            "segmento": segment, "modelo": name,
            "roc_auc_oof": roc_auc_score(y, probability),
            "pr_auc_oof": average_precision_score(y, probability),
            "brier_oof": brier_score_loss(y, probability),
        })
    return pd.DataFrame(rows)


def subgroup_table(scored, column):
    rows = []
    for value, group in scored.groupby(column, dropna=False):
        if len(group) < 15:
            continue
        positives = group["risco_seguinte"] == 1
        negatives = ~positives
        rows.append({
            "subgrupo": column, "valor": value, "n": len(group),
            "prevalencia": group["risco_seguinte"].mean(),
            "recall": group.loc[positives, "predicao"].mean() if positives.any() else np.nan,
            "fpr": group.loc[negatives, "predicao"].mean() if negatives.any() else np.nan,
            "roc_auc": (roc_auc_score(group["risco_seguinte"],
                                      group["probabilidade_risco"])
                        if group["risco_seguinte"].nunique() > 1 else np.nan),
        })
    return pd.DataFrame(rows)


def train_and_save():
    _, years = load_panel()
    train, test = make_transitions(years)
    all_transitions = pd.concat([train, test], ignore_index=True)
    evaluation_models, production_models = {}, {}
    evaluation_thresholds, production_thresholds = {}, {}
    test_parts, calibration_parts, importance_parts, comparison_parts = [], [], [], []
    segment_metrics = {}

    for segment, config in SEGMENTS.items():
        train_segment = train.loc[config["filter"](train)].copy()
        test_segment = test.loc[config["filter"](test)].copy()
        production_segment = all_transitions.loc[config["filter"](all_transitions)].copy()

        eval_threshold, oof_probability = select_threshold_oof(
            train_segment[FEATURES], train_segment["risco_seguinte"])
        comparison_parts.append(compare_candidates(
            train_segment[FEATURES], train_segment["risco_seguinte"], segment))
        eval_model = build_pipeline().fit(
            train_segment[FEATURES], train_segment["risco_seguinte"])
        test_probability = eval_model.predict_proba(test_segment[FEATURES])[:, 1]
        test_prediction = test_probability >= eval_threshold

        prod_threshold, _ = select_threshold_oof(
            production_segment[FEATURES], production_segment["risco_seguinte"])
        prod_model = build_pipeline().fit(
            production_segment[FEATURES], production_segment["risco_seguinte"])

        evaluation_models[segment] = eval_model
        production_models[segment] = prod_model
        evaluation_thresholds[segment] = eval_threshold
        production_thresholds[segment] = prod_threshold
        segment_metrics[segment] = {
            "description": config["description"],
            "oof_train": metric_dict(
                train_segment["risco_seguinte"], oof_probability,
                oof_probability >= eval_threshold, {segment: eval_threshold}),
            "temporal_test": metric_dict(
                test_segment["risco_seguinte"], test_probability,
                test_prediction, {segment: eval_threshold}),
        }
        part = test_segment[["RA", "ano", "risco_seguinte", "genero",
                             "fase_num", "defasagem", "defasagem_seguinte"]].copy()
        part["segmento"] = segment
        part["probabilidade_risco"] = test_probability
        part["predicao"] = test_prediction.astype(int)
        test_parts.append(part)
        calibration_parts.append(calibration_table(
            test_segment["risco_seguinte"], test_probability, segment))
        importance_parts.append(coefficient_table(prod_model, segment))

    scored = pd.concat(test_parts).sort_index()
    overall = metric_dict(
        scored["risco_seguinte"], scored["probabilidade_risco"],
        scored["predicao"], evaluation_thresholds)
    metrics = {
        "definition": {
            "entrada": SEGMENTS["entrada"]["description"],
            "permanencia": SEGMENTS["permanencia"]["description"],
        },
        "evaluation_design": "Treino 2022->2023; teste temporal 2023->2024.",
        "threshold_selection": "Probabilidades out-of-fold, StratifiedKFold(5).",
        "sklearn_version": sklearn.__version__,
        "overall_temporal_test": overall,
        "segments": segment_metrics,
        "production": {
            "training_data": "Transicoes 2022->2023 e 2023->2024",
            "thresholds": production_thresholds,
        },
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "production_models": production_models,
        "thresholds": production_thresholds,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "sklearn_version": sklearn.__version__,
    }, ARTIFACT_DIR / "risk_model.joblib")
    pd.concat(importance_parts).to_csv(
        ARTIFACT_DIR / "feature_importance.csv", index=False)
    pd.concat(calibration_parts).to_csv(
        ARTIFACT_DIR / "calibration.csv", index=False)
    pd.concat(comparison_parts).to_csv(
        ARTIFACT_DIR / "model_comparison.csv", index=False)
    pd.concat([subgroup_table(scored, "genero"),
               subgroup_table(scored, "fase_num")]).to_csv(
        ARTIFACT_DIR / "subgroup_metrics.csv", index=False)
    scored.to_csv(ARTIFACT_DIR / "temporal_test_predictions.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), ensure_ascii=False, indent=2))
