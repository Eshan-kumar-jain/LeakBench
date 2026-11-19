"""Train the churn model and report its validation score.

Writes:
  /app/artifacts/model.joblib   - fitted model; must accept build_features(df) output in predict_proba
  /app/artifacts/metrics.json   - {"val_auc", "train_months", "val_months", "features"}
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import TARGET, build_features

APP = Path(os.environ.get("APP_DIR", "/app"))
DATA = APP / "data" / "train.csv"
ART = APP / "artifacts"


def main():
    df = pd.read_csv(DATA)
    X = build_features(df)
    y = df[TARGET]

    X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
        X, y, df["month"], test_size=0.2, random_state=0, stratify=y)

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    model.fit(X_tr, y_tr)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    metrics = {
        "val_auc": round(float(val_auc), 4),
        "train_months": sorted(int(m) for m in m_tr.unique()),
        "val_months": sorted(int(m) for m in m_val.unique()),
        "features": list(X.columns),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
