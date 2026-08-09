import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import FEATURES, load_panel, make_transitions


class DataAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.years = load_panel()
        cls.bundle = joblib.load(ROOT / "artifacts" / "risk_model.joblib")

    def test_canonical_gender(self):
        expected = {"Feminino", "Masculino", "Outro"}
        for frame in self.years.values():
            self.assertTrue(set(frame["genero"].dropna()).issubset(expected))

    def test_canonical_institution(self):
        expected = {"Publica", "Privada", "Rede Decisao", "Outra"}
        for frame in self.years.values():
            self.assertTrue(set(frame["instituicao"].dropna()).issubset(expected))

    def test_transition_segments_are_exclusive(self):
        train, _ = make_transitions(self.years)
        self.assertEqual(len(train), (train.defasagem >= 0).sum()
                         + (train.defasagem < 0).sum())
        self.assertTrue((train.loc[train.defasagem >= 0, "entrada_defasagem"]
                         == train.loc[train.defasagem >= 0, "risco_seguinte"]).all())
        self.assertTrue((train.loc[train.defasagem < 0, "permanencia_defasagem"]
                         == train.loc[train.defasagem < 0, "risco_seguinte"]).all())

    def test_app_categories_are_known_to_production_models(self):
        expected_gender = {"Feminino", "Masculino", "Outro"}
        expected_institution = {"Publica", "Privada", "Rede Decisao", "Outra"}
        for model in self.bundle["production_models"].values():
            names = set(model["prep"].get_feature_names_out())
            for value in expected_gender:
                self.assertIn(f"cat__genero_{value}", names)
            for value in expected_institution:
                self.assertIn(f"cat__instituicao_{value}", names)

    def test_segmented_inference_returns_probabilities(self):
        latest = self.years[2024]
        for segment, condition in {
            "entrada": latest.defasagem >= 0,
            "permanencia": latest.defasagem < 0,
        }.items():
            probability = self.bundle["production_models"][segment].predict_proba(
                latest.loc[condition, FEATURES])[:, 1]
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_metrics_record_temporal_evaluation(self):
        metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(
            encoding="utf-8"))
        self.assertIn("overall_temporal_test", metrics)
        self.assertEqual(metrics["threshold_selection"],
                         "Probabilidades out-of-fold, StratifiedKFold(5).")

    def test_model_comparison_contains_baseline(self):
        comparison = (ROOT / "artifacts" / "model_comparison.csv").read_text(
            encoding="utf-8")
        self.assertIn("Dummy prevalencia", comparison)
        self.assertIn("Regressao logistica", comparison)
        self.assertIn("Random Forest", comparison)

    def test_notebook_is_executed(self):
        notebook = json.loads((ROOT / "notebooks" /
                               "analise_risco_defasagem.ipynb").read_text(
                                   encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"]
                      if cell["cell_type"] == "code"]
        self.assertTrue(all(cell["execution_count"] is not None
                            for cell in code_cells))
        self.assertTrue(all(cell.get("outputs") for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
