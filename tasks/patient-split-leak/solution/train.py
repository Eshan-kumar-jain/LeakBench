"""Oracle training: validation patients are held out as whole patients (group split),
patient_id is not a feature, then the model is refit on every patient."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from features import TARGET, build_features

APP = Path(os.environ.get("APP_DIR", "/app"))
ART = APP / "artifacts"
SEED = int(os.environ.get("SEED", "0"))


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=SEED)


def main():
    df = pd.read_csv(APP / "data" / "visits.csv")
    X, y = build_features(df.drop(columns=[TARGET])), df[TARGET]
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups=df["patient_id"]))

    model = make_model().fit(X.iloc[tr], y.iloc[tr])
    val_auc = roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1])

    final = make_model().fit(X, y)
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, ART / "model.joblib")
    metrics = {
        "val_auc": round(float(val_auc), 4),
        "features": list(X.columns),
        "train_patient_ids": sorted(int(i) for i in df["patient_id"].iloc[tr].unique()),
        "val_patient_ids": sorted(int(i) for i in df["patient_id"].iloc[va].unique()),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics))
    print({k: v for k, v in metrics.items() if not k.endswith("_ids")})


if __name__ == "__main__":
    main()
