import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from churn_prediction.config import (
    ID_COLUMN,
    RANDOM_STATE,
    SELECTED_THRESHOLD,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)

from churn_prediction.model.pipeline import build_pipeline
from churn_prediction.paths import (
    ARTIFACTS_DIR,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
)

def train_model() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    X = df.drop(
        columns=[ID_COLUMN, TARGET_COLUMN]
    )
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    
    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    pipeline = build_pipeline()
    
    pipeline.fit(X_dev, y_dev)
    
    artifact = {
        "pipeline": pipeline,
        "threshold": SELECTED_THRESHOLD,
    }

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_ARTIFACT_PATH,
    )

    print(
        f"Model artifact saved to: "
        f"{MODEL_ARTIFACT_PATH}"
    )

def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
    