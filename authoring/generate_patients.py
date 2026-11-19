"""Author-side generator for the patient-split-leak task (NOT shipped to the agent).

Hospital visits with repeat patients. Each patient has a hidden frailty that strongly drives
30-day readmission, so a model can look good by recognising patients it has already seen.

- 3,000 patients -> environment/data/visits.csv        (agent can see)
- 1,200 NEW patients -> tests/hidden_data/new_patients.csv (grader only)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261122
TASK = Path(__file__).resolve().parents[1] / "tasks" / "patient-split-leak"


def simulate(rng, first_id: int, n_patients: int) -> pd.DataFrame:
    rows = []
    for pid in range(first_id, first_id + n_patients):
        age = int(np.clip(rng.normal(64, 14), 18, 98))
        chronic = int(rng.poisson(1.6))
        bmi = round(float(np.clip(rng.normal(28, 5), 16, 50)), 1)
        frailty = rng.normal(0, 2.0)                       # hidden, patient-level
        n_visits = 1 + int(rng.poisson(5.0))
        for v in range(1, n_visits + 1):
            los = int(np.clip(rng.gamma(2.0, 2.0 + 0.8 * max(frailty, 0)), 1, 30))
            meds = int(rng.poisson(4 + chronic))
            emergency = int(rng.random() < 0.25 + 0.3 * (frailty > 0.5))
            logit = (-2.4 + 1.0 * frailty + 0.35 * chronic + 0.02 * (age - 64)
                     + 0.08 * los + 0.7 * emergency + 0.04 * meds)
            readmit = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(dict(patient_id=pid, visit_number=v, age=age, chronic_conditions=chronic,
                             bmi=bmi, length_of_stay=los, num_medications=meds,
                             emergency_admission=emergency, readmitted_30d=readmit))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    visible = simulate(rng, 100_001, 3_000)
    hidden = simulate(rng, 200_001, 1_200)
    (TASK / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (TASK / "tests" / "hidden_data").mkdir(parents=True, exist_ok=True)
    visible.to_csv(TASK / "environment" / "data" / "visits.csv", index=False)
    hidden.to_csv(TASK / "tests" / "hidden_data" / "new_patients.csv", index=False)
    print(f"visible visits {len(visible)} | hidden visits {len(hidden)} | "
          f"readmission rate {visible.readmitted_30d.mean():.3f}")


if __name__ == "__main__":
    main()
