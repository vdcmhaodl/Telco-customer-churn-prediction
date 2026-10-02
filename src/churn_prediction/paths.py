from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "telco-churn.csv"
PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "telco-churn-cleaned.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_ARTIFACT_PATH = (
    ARTIFACTS_DIR / "churn_model.joblib"
)