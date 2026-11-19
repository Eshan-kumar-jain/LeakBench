"""Oracle features: no patient identifier; adds each patient's history from earlier visits only."""
import pandas as pd

TARGET = "readmitted_30d"
BASE = ["visit_number", "age", "chronic_conditions", "bmi",
        "length_of_stay", "num_medications", "emergency_admission"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["patient_id", "visit_number"])
    g = d.groupby("patient_id")
    out = d[BASE].copy()
    # Groupby uses patient_id only to line up a patient's own earlier visits; the ID itself is not a feature.
    out["prior_emergencies"] = g["emergency_admission"].cumsum() - d["emergency_admission"]
    out["prior_mean_los"] = g["length_of_stay"].transform(lambda s: s.shift(1).expanding().mean())
    return out.loc[df.index]
