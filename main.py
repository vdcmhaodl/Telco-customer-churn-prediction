import pandas as pd

from src.data.clean import clean_data


def main():
    raw_df = pd.read_csv("data/raw/telco-churn.csv")

    clean_df = clean_data(raw_df)

    clean_df.to_csv(
        "data/processed/telco-churn-cleaned.csv",
        index=False,
    )


if __name__ == "__main__":
    main()