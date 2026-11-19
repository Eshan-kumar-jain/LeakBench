"""Oracle training: out-of-time validation (train on months 1-10, validate on 11-12),
then refit on all 12 months for the final model."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import TARGET, build_features

APP = Path(os.environ.get("APP_DIR", "/app"))
DATA = APP / "data" / "train.csv"
ART = APP / "artifacts"
VAL_MONTHS = [11, 12]


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=int(os.environ.get("SEED", "0")))


def main():
    df = pd.read_csv(DATA)
    X = build_features(df.drop(columns=[TARGET]))
    y = df[TARGET]

    is_val = df["month"].isin(VAL_MONTHS)
    model = make_model().fit(X[~is_val], y[~is_val])
    val_auc = roc_auc_score(y[is_val], model.predict_proba(X[is_val])[:, 1])

    final = make_model().fit(X, y)  # same recipe, all available history
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, ART / "model.joblib")
    metrics = {
        "val_auc": round(float(val_auc), 4),
        "train_months": sorted(int(m) for m in df.loc[~is_val, "month"].unique()),
        "val_months": VAL_MONTHS,
        "features": list(X.columns),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
