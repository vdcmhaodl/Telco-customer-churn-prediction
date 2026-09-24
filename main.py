import pandas as pd

from src.data.clean import clean_data
from src.data.validate import validate_clean_data


RAW_PATH = "data/raw/telco-churn.csv"
PROCESSED_PATH = "data/processed/telco-churn-cleaned.csv"

def cleaning_data():
    raw_df = pd.read_csv(RAW_PATH)
    
    clean_df = clean_data(raw_df)

    validate_clean_data(
        raw_df=raw_df,
        clean_df=clean_df,
    )

    clean_df.to_csv(
        PROCESSED_PATH,
        index=False,
    )

    print(
        f"Cleaned dataset saved to {PROCESSED_PATH}"
    )
    print(
        f"Shape: {clean_df.shape}"
        )
def main() -> None:
    cleaning_data() # Data cleaning


if __name__ == "__main__":
    main()