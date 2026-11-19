"""Author-side data generator (NOT shipped to the agent).

Builds a synthetic monthly churn panel for a subscription business.
- Months 1-12  -> environment/data/train.csv          (agent can see)
- Months 1-18  -> tests/hidden_data/full_history.csv  (grader only; months 13-18 are the hidden future)

Each row is one customer in one month. Target: churned_next_month (1 = customer leaves after this month).
Two leaks are planted on purpose for the buggy pipeline to fall into:
  * account_status: set to "closing" for almost every churner -> label leak
  * the panel itself lets a careless feature (e.g. count of a customer's rows over the WHOLE table)
    see the future -> temporal leak
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261120
N_CUSTOMERS = 4000
LAST_MONTH = 18          # last month we publish a row for
VISIBLE_LAST_MONTH = 12  # agent sees months 1..12

ROOT = Path(__file__).resolve().parents[1] / "tasks"
TASK = ROOT / "leaky-churn-audit"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for cid in range(1, N_CUSTOMERS + 1):
        start = int(rng.integers(1, 7))                  # joins in month 1..6
        tenure0 = int(rng.integers(0, 48))               # months already with us at start
        plan = "premium" if rng.random() < 0.3 else "basic"
        base_spend = float(rng.lognormal(mean=3.6 if plan == "basic" else 4.3, sigma=0.35))
        dissat = float(rng.normal(-0.5, 0.6))            # latent dissatisfaction
        for month in range(start, LAST_MONTH + 2):       # simulate one extra month to label month 18
            dissat += rng.normal(0.05, 0.35) + (rng.random() < 0.04) * rng.uniform(0.8, 1.8)
            dissat = max(min(dissat, 4.0), -2.0)
            tenure = tenure0 + (month - start)
            spend = max(5.0, base_spend * (1 - 0.12 * max(dissat, 0)) + rng.normal(0, 4))
            calls = int(rng.poisson(0.25 + 0.7 * max(dissat, 0)))
            hazard = sigmoid(-4.2 + 0.9 * dissat + 0.5 * (plan == "basic") - 0.015 * tenure)
            churn = rng.random() < hazard
            rows.append(dict(customer_id=cid, month=month, plan=plan, tenure_months=tenure,
                             monthly_spend=round(spend, 2), support_calls=calls, _churn=churn))
            if churn:
                break
    df = pd.DataFrame(rows)
    df["churned_next_month"] = df["_churn"].astype(int)
    df = df.drop(columns="_churn")
    df = df[df["month"] <= LAST_MONTH].copy()

    # Planted label leak: CRM marks accounts "closing" once the customer has given notice.
    noise = rng.random(len(df))
    df["account_status"] = np.where(
        (df["churned_next_month"] == 1) & (noise < 0.93), "closing",
        np.where(noise < 0.01, "closing", "active"))
    cols = ["customer_id", "month", "plan", "tenure_months", "monthly_spend",
            "support_calls", "account_status", "churned_next_month"]
    return df[cols].sort_values(["month", "customer_id"]).reset_index(drop=True)


def main():
    df = simulate()
    visible = df[df["month"] <= VISIBLE_LAST_MONTH]
    (TASK / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (TASK / "tests" / "hidden_data").mkdir(parents=True, exist_ok=True)
    visible.to_csv(TASK / "environment" / "data" / "train.csv", index=False)
    df.to_csv(TASK / "tests" / "hidden_data" / "full_history.csv", index=False)
    by_month = df.groupby("month")["churned_next_month"].agg(["size", "mean"]).round(3)
    print(by_month.to_string())
    print("visible rows:", len(visible), "| total rows:", len(df))


if __name__ == "__main__":
    main()
