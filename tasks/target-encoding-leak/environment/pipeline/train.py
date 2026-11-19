"""Train the loan-default model and report its validation score.

Writes:
  /app/artifacts/model.joblib  - fitted model; predict_proba must accept build_features(history, new_rows)
  /app/artifacts/metrics.json  - {"val_auc", "features"}
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
ART = APP / "artifacts"


def main():
    df = pd.read_csv(APP / "data" / "loans.csv")
    tr, va = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET])

    X_tr = build_features(tr, tr)
    X_va = build_features(tr, va)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    model.fit(X_tr, tr[TARGET])
    val_auc = roc_auc_score(va[TARGET], model.predict_proba(X_va)[:, 1])

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    metrics = {"val_auc": round(float(val_auc), 4), "features": list(X_tr.columns)}
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
