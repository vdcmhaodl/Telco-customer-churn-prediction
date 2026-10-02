from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    OneHotEncoder,
)

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
YES_NO_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

BINARY_FEATURES = [
    "SeniorCitizen",
    *YES_NO_FEATURES
]
MULTICLASS_FEATURES = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

def build_preprocessor() -> ColumnTransformer:
    binary_transformer = OrdinalEncoder(
        categories=[
            ["No", "Yes"]
            for _ in YES_NO_FEATURES
        ]
    )
    
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "binary",
                binary_transformer,
                YES_NO_FEATURES
            ),
            (
                "senior",
                "passthrough",
                ["SeniorCitizen"],
            ),
            (
                "multi",
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                ),
                MULTICLASS_FEATURES,
            ),
        ],
        remainder="drop",
    )

MODEL_FEATURES = (
    NUMERICAL_FEATURES
    + BINARY_FEATURES
    + MULTICLASS_FEATURES
)

def validate_feature_group() -> None:
    if len(MODEL_FEATURES) != len(set(MODEL_FEATURES)):
        raise ValueError(
            "A feature appears in more than one feature group."
        )
        