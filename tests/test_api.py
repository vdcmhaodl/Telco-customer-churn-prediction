import numpy as np

from fastapi.testclient import TestClient

import churn_prediction.api.app as app_module


client = TestClient(app_module.app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }
VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
}
def test_predict_returns_prediction(
    monkeypatch,
):
    def fake_predict_churn(df):
        return (
            np.array([0.75]),
            np.array([1]),
        )

    monkeypatch.setattr(
        app_module,
        "predict_churn",
        fake_predict_churn,
    )

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    assert response.json() == {
        "churn_probability": 0.75,
        "churn_prediction": 1,
    }
    
def test_predict_rejects_invalid_input():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["SeniorCitizen"] = 7

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422
    
def test_predict_rejects_extra_field():
    invalid_customer = VALID_CUSTOMER.copy()

    invalid_customer["favoriteColor"] = "blue"

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422

def test_predict_passes_correct_dataframe(
    monkeypatch,
):
    captured_df = None

    def fake_predict_churn(df):
        nonlocal captured_df
        captured_df = df.copy()

        return (
            np.array([0.75]),
            np.array([1]),
        )

    monkeypatch.setattr(
        app_module,
        "predict_churn",
        fake_predict_churn,
    )

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    assert captured_df is not None
    assert captured_df.shape == (1, 19)

    assert set(captured_df.columns) == set(
        VALID_CUSTOMER.keys()
    )