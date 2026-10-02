import numpy as np
import pandas as pd

import churn_prediction.model.predict as predict_module


class FakePipeline:
    def predict_proba(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        return np.array(
            [
                [0.80, 0.20],
                [0.60, 0.40],
            ]
        )

def test_predict_churn_probability(
    monkeypatch,
):
    artifact = {
        "pipeline": FakePipeline(),
        "threshold": 0.28,
    }

    monkeypatch.setattr(
        predict_module,
        "load_model_artifact",
        lambda: artifact,
    )

    df = pd.DataFrame(
        {
            "dummy": [1, 2],
        }
    )

    probabilities = (
        predict_module.predict_churn_probability(
            df
        )
    )

    np.testing.assert_allclose(
        probabilities,
        np.array([0.20, 0.40]),
    )

def test_predict_churn_applies_threshold(
    monkeypatch,
):
    artifact = {
        "pipeline": FakePipeline(),
        "threshold": 0.28,
    }

    monkeypatch.setattr(
        predict_module,
        "load_model_artifact",
        lambda: artifact,
    )

    df = pd.DataFrame(
        {
            "dummy": [1, 2],
        }
    )

    probabilities, predictions = (
        predict_module.predict_churn(df)
    )

    np.testing.assert_allclose(
        probabilities,
        np.array([0.20, 0.40]),
    )

    np.testing.assert_array_equal(
        predictions,
        np.array([0, 1]),
    )