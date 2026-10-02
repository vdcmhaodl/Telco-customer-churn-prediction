import joblib
import numpy as np 
import pandas as pd 

from churn_prediction.paths import(
    MODEL_ARTIFACT_PATH
)

def load_model_artifact() -> dict:
    return joblib.load(
        MODEL_ARTIFACT_PATH
    )
    
def predict_churn(
    df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    artifact = load_model_artifact()
    
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    
    # print(
    #     artifact["pipeline"].classes_
    # )
    probabilities = pipeline.predict_proba(
        df
    )[:, 1]
    
    predictions = (
        probabilities >= threshold
    ).astype(int)
    
    return probabilities, predictions