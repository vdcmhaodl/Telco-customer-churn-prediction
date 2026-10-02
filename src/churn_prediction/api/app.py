import pandas as pd 

from fastapi import FastAPI

from churn_prediction.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
)

from churn_prediction.model.predict import (
    predict_churn,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
)

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    customer: CustomerFeatures
) -> PredictionResponse:
    customer_df = pd.DataFrame(
        [
            customer.model_dump()
        ]
    )
    probabilities, predictions = predict_churn(
        customer_df
    )
    return PredictionResponse(
        churn_probability=float(
            probabilities[0]
        ),
        churn_prediction=int(
            predictions[0]
        ),
    )

