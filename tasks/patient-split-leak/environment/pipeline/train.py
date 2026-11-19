"""Train the readmission model and report its validation score.

Writes:
  /app/artifacts/model.joblib  - fitted model; predict_proba must accept build_features(visits)
  /app/artifacts/metrics.json  - {"val_auc", "features", "train_patient_ids", "val_patient_ids"}
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import TARGET, build_features

APP = Path(os.environ.get("APP_DIR", "/app"))
ART = APP / "artifacts"


def main():
    df = pd.read_csv(APP / "data" / "visits.csv")
    X, y = build_features(df), df[TARGET]
    X_tr, X_va, y_tr, y_va, id_tr, id_va = train_test_split(
        X, y, df["patient_id"], test_size=0.2, random_state=0, stratify=y)

    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    model.fit(X_tr, y_tr)
    val_auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    metrics = {
        "val_auc": round(float(val_auc), 4),
        "features": list(X.columns),
        "train_patient_ids": sorted(int(i) for i in id_tr.unique()),
        "val_patient_ids": sorted(int(i) for i in id_va.unique()),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics))
    print({k: v for k, v in metrics.items() if not k.endswith("_ids")})


if __name__ == "__main__":
    main()
