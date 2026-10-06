from pathlib import Path

import pandas as pd 

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from churn_prediction.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
)

from churn_prediction.model.predict import (
    predict_churn,
)
import logging
from churn_prediction.api.metrics import (
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_OUTCOMES,
    PREDICTION_REQUESTS,
)
from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
)

FRONTEND_DIR = Path(__file__).with_name("static")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")

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
    PREDICTION_REQUESTS.inc()
    
    customer_df = pd.DataFrame(
        [
            customer.model_dump()
        ]
    )
    try:
        with PREDICTION_LATENCY.time():
            probabilities, predictions = predict_churn(
                customer_df
            )
    except:
        PREDICTION_ERRORS.inc()
        raise
    
    prediction = int(predictions[0])
    
    PREDICTION_OUTCOMES.labels(
    prediction=str(prediction)
    ).inc()
    return PredictionResponse(
        churn_probability=float(
            probabilities[0]
        ),
        churn_prediction=prediction,
    )

@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
