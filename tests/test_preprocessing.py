from sklearn.compose import ColumnTransformer

from churn_prediction.features.preprocessing import (
    build_preprocessor,
)


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )

def test_build_preprocessor_returns_fresh_object():
    preprocessor_1 = build_preprocessor()
    preprocessor_2 = build_preprocessor()

    assert preprocessor_1 is not preprocessor_2