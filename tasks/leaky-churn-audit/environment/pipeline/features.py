"""Feature engineering for the churn model.

Contract (used by training AND by production scoring):
    build_features(df) -> pandas.DataFrame
      - df has the columns of /app/data/train.csv (it may or may not include the target)
      - returns one row per input row, same index, model-ready features only
"""
import pandas as pd

TARGET = "churned_next_month"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tenure_months"] = df["tenure_months"]
    out["monthly_spend"] = df["monthly_spend"]
    out["support_calls"] = df["support_calls"]
    out["is_premium"] = (df["plan"] == "premium").astype(int)
    out["is_closing"] = (df["account_status"] == "closing").astype(int)

    g = df.groupby("customer_id")
    # How long the customer has been active with us, and their typical spend.
    out["months_active"] = g["month"].transform("count")
    out["avg_spend"] = g["monthly_spend"].transform("mean")
    out["spend_vs_avg"] = df["monthly_spend"] / out["avg_spend"]
    out["total_calls"] = g["support_calls"].transform("sum")
    return out
