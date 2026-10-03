import joblib
import numpy as np 
import pandas as pd 
from functools import lru_cache

import logging
import time

from churn_prediction.paths import(
    MODEL_ARTIFACT_PATH
)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_model_artifact() -> dict:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{MODEL_ARTIFACT_PATH}. "
            "Run training first."
        )
    logger.info(
        "Loading model artifact from %s",
        MODEL_ARTIFACT_PATH,
    )
    artifact = joblib.load(
        MODEL_ARTIFACT_PATH
    )
    logger.info(
        "Model artifact loaded successfully"
    )
    return artifact
   
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
    
    start = time.perf_counter()
    try:
        probabilities = predict_churn_probability(
            df
        )
        
        predictions = (
            probabilities >= threshold
        ).astype(int)
    except Exception:
        logger.exception(
            "Prediction failed"
        )
        raise
    
    elapsed = time.perf_counter() - start
    logger.info(
        "Prediction completed in %.4f seconds",
        elapsed,
    )
    
    return probabilities, predictions

