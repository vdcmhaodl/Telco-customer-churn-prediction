import joblib
import numpy as np 
import pandas as pd 
from functools import lru_cache

from churn_prediction.paths import(
    MODEL_ARTIFACT_PATH
)

@lru_cache(maxsize=1)
def load_model_artifact() -> dict:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{MODEL_ARTIFACT_PATH}. "
            "Run training first."
        )

    return joblib.load(
        MODEL_ARTIFACT_PATH
    )
   
def predict_churn_probability(
    df: pd.DataFrame,
) -> np.ndarray:
    artifact = load_model_artifact()

    pipeline = artifact["pipeline"]

    return pipeline.predict_proba(
        df
    )[:, 1]
     
def predict_churn(
    df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    artifact = load_model_artifact()

    threshold = artifact["threshold"]
    
    probabilities = predict_churn_probability(
        df
    )
    
    predictions = (
        probabilities >= threshold
    ).astype(int)
    
    return probabilities, predictions

