from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from churn_prediction.features.preprocessing import (
    build_preprocessor,
)

def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )