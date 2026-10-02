from sklearn.pipeline import Pipeline

from churn_prediction.model.pipeline import (
    build_pipeline,
)


def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline()

    assert isinstance(
        pipeline,
        Pipeline,
    )

def test_build_pipeline_returns_fresh_pipeline():
    pipeline_1 = build_pipeline()
    pipeline_2 = build_pipeline()

    assert pipeline_1 is not pipeline_2