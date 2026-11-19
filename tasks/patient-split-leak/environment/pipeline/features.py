"""Feature engineering for the 30-day readmission model.

Contract:
    build_features(df) -> pandas.DataFrame
      - df has the columns of /app/data/visits.csv, without the target when scoring
      - returns one row of model-ready features per visit, same index
"""
import pandas as pd

TARGET = "readmitted_30d"
COLUMNS = ["patient_id", "visit_number", "age", "chronic_conditions", "bmi",
           "length_of_stay", "num_medications", "emergency_admission"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLUMNS].copy()
