import pandas as pd


def validate_clean_data(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> None:
    if len(clean_df) != len(raw_df):
        raise ValueError(
            "Cleaning changed the number of rows."
        )

    if list(clean_df.columns) != list(raw_df.columns):
        raise ValueError(
            "Cleaning changed the dataset columns."
        )

    if clean_df["TotalCharges"].isna().any():
        raise ValueError(
            "TotalCharges still contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        clean_df["TotalCharges"]
    ):
        raise TypeError(
            "TotalCharges must be numeric."
        )

    if clean_df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customerID detected."
        )

    if set(clean_df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError(
            "Unexpected values found in Churn."
        )

    if (clean_df["TotalCharges"] < 0).any():
        raise ValueError(
            "TotalCharges contains negative values."
        )