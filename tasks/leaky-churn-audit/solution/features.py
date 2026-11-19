"""Oracle features: every value uses only the customer's rows up to and including the current month."""
import pandas as pd

TARGET = "churned_next_month"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["customer_id", "month"])
    g = d.groupby("customer_id")

    out = pd.DataFrame(index=d.index)
    out["tenure_months"] = d["tenure_months"]
    out["monthly_spend"] = d["monthly_spend"]
    out["support_calls"] = d["support_calls"]
    out["is_premium"] = (d["plan"] == "premium").astype(int)

    # Past-only aggregates: shift(1) excludes the current month, expanding() never looks ahead.
    prev_avg = g["monthly_spend"].transform(lambda s: s.shift(1).expanding().mean())
    out["spend_vs_past_avg"] = (d["monthly_spend"] / prev_avg).fillna(1.0)
    out["spend_change_1m"] = g["monthly_spend"].diff().fillna(0.0)
    out["calls_last_3m"] = g["support_calls"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["months_seen_so_far"] = g.cumcount() + 1

    # account_status is deliberately NOT used: it is back-filled after each quarter,
    # so it is not available when the model scores a customer.
    return out.loc[df.index]
