import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ROOT,
                  load_panel, make_transitions)


ARTIFACT_DIR = ROOT / "artifacts"


def build_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(max_iter=3000, C=0.5)
    return Pipeline([("prep", prep), ("model", model)])


def metric_dict(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
        "threshold": float(threshold),
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
    }


def train_and_save():
    panel, years = load_panel()
    train, test = make_transitions(years)
    X_train, y_train = train[FEATURES], train["risco_seguinte"]
    X_test, y_test = test[FEATURES], test["risco_seguinte"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    train_probability = pipeline.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.20, 0.80, 121)
    scores = [f1_score(y_train, train_probability >= t) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    test_probability = pipeline.predict_proba(X_test)[:, 1]

    feature_names = pipeline["prep"].get_feature_names_out()
    coefficients = pipeline["model"].coef_[0]
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    metrics = {
        "definition": "Risco = defasagem no ano seguinte menor que zero.",
        "train_period": "2022 -> 2023",
        "test_period": "2023 -> 2024",
        "train": metric_dict(y_train, train_probability, threshold),
        "test": metric_dict(y_test, test_probability, threshold),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "threshold": threshold,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }, ARTIFACT_DIR / "risk_model.joblib")
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    scored = test[["RA", "ano", "risco_seguinte", "defasagem_seguinte"]].copy()
    scored["probabilidade_risco"] = test_probability
    scored.to_csv(ARTIFACT_DIR / "temporal_test_predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save(), ensure_ascii=False, indent=2))
