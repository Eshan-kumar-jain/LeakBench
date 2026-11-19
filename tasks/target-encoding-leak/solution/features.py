"""Oracle features: the employer encoding is learned from train_df only, smoothed toward the
overall default rate, and training rows are encoded out-of-fold so no row sees its own label."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TARGET = "defaulted"
NUMERIC = ["income", "debt_ratio", "loan_amount", "credit_years"]
SMOOTHING = 20  # pseudo-loans at the overall rate; small employers shrink toward the prior


def _encoding(train_df: pd.DataFrame) -> tuple[pd.Series, float]:
    prior = float(train_df[TARGET].mean())
    stats = train_df.groupby("employer")[TARGET].agg(["sum", "count"])
    rate = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    return rate, prior


def build_features(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    """Features for new rows. Never reads apply_df's target, even if it is present."""
    rate, prior = _encoding(train_df)
    X = apply_df[NUMERIC].copy()
    X["employer_default_rate"] = apply_df["employer"].map(rate).fillna(prior)
    X["employer_n_loans"] = apply_df["employer"].map(train_df["employer"].value_counts()).fillna(0)
    return X


def build_train_features(train_df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Features for the training rows themselves: each fold is encoded by the other folds."""
    parts = []
    for fit_idx, enc_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train_df):
        parts.append(build_features(train_df.iloc[fit_idx], train_df.iloc[enc_idx]))
    X = pd.concat(parts).loc[train_df.index]
    return X.astype(np.float64)
