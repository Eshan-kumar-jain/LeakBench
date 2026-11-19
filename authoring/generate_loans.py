"""Author-side generator for the target-encoding-leak task (NOT shipped to the agent).

Loan applications with a high-cardinality `employer` column. Many employers have only a
handful of loans, which is exactly where a target encoding computed on the rows it
encodes memorises the label.

- 20,000 loans -> environment/data/loans.csv          (agent can see)
- 10,000 loans -> tests/hidden_data/new_loans.csv     (grader only)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261121
N = 30_000
N_EMPLOYERS = 1_500
TASK = Path(__file__).resolve().parents[1] / "tasks" / "target-encoding-leak"


def main():
    rng = np.random.default_rng(SEED)
    weights = 1.0 / np.arange(1, N_EMPLOYERS + 1) ** 0.9          # a few big employers, a long tail of small ones
    weights /= weights.sum()
    emp_effect = rng.normal(0.0, 0.9, N_EMPLOYERS)
    emp_idx = rng.choice(N_EMPLOYERS, size=N, p=weights)

    income = rng.lognormal(np.log(52_000), 0.45, N)
    debt_ratio = rng.beta(2.2, 4.0, N)
    loan_amount = np.clip(rng.lognormal(np.log(15_000), 0.6, N), 1_000, 120_000)
    credit_years = np.clip(rng.gamma(3.0, 3.0, N), 0, 40)

    logit = (-2.4 + emp_effect[emp_idx] + 3.0 * (debt_ratio - 0.35)
             - 0.5 * np.log(income / 52_000) + 0.25 * np.log(loan_amount / 15_000)
             - 0.03 * credit_years)
    defaulted = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "loan_id": np.arange(1, N + 1),
        "employer": [f"EMP{e:04d}" for e in emp_idx],
        "income": income.round(0),
        "debt_ratio": debt_ratio.round(3),
        "loan_amount": loan_amount.round(0),
        "credit_years": credit_years.round(1),
        "defaulted": defaulted,
    })
    order = rng.permutation(N)
    visible, hidden = df.iloc[order[:20_000]], df.iloc[order[20_000:]]
    (TASK / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (TASK / "tests" / "hidden_data").mkdir(parents=True, exist_ok=True)
    visible.sort_values("loan_id").to_csv(TASK / "environment" / "data" / "loans.csv", index=False)
    hidden.sort_values("loan_id").to_csv(TASK / "tests" / "hidden_data" / "new_loans.csv", index=False)
    sizes = visible["employer"].value_counts()
    print(f"default rate {df.defaulted.mean():.3f} | employers in visible data {sizes.size} | "
          f"employers with <=5 loans {(sizes <= 5).sum()}")


if __name__ == "__main__":
    main()
