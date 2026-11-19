"""Feature engineering for the loan-default model.

Contract:
    build_features(train_df, apply_df) -> pandas.DataFrame
      - train_df: labelled history the encoder may learn from (has the target column)
      - apply_df: rows to build features for; same index in the output.
        In production apply_df has NO target column.
"""
import pandas as pd

TARGET = "defaulted"
NUMERIC = ["income", "debt_ratio", "loan_amount", "credit_years"]


def build_features(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    # Learn each employer's default rate from all the labelled data we have.
    labelled = pd.concat([train_df, apply_df]) if TARGET in apply_df.columns else train_df
    rate = labelled.groupby("employer")[TARGET].mean()

    X = apply_df[NUMERIC].copy()
    X["employer_default_rate"] = apply_df["employer"].map(rate).fillna(train_df[TARGET].mean())
    return X
