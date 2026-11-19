"""Oracle training: encoder fitted on the training split only (out-of-fold for training rows),
validation rows encoded without their labels, then refit on all loans."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import TARGET, build_features, build_train_features

APP = Path(os.environ.get("APP_DIR", "/app"))
ART = APP / "artifacts"
SEED = int(os.environ.get("SEED", "0"))


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=SEED)


def main():
    df = pd.read_csv(APP / "data" / "loans.csv")
    tr, va = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])

    model = make_model().fit(build_train_features(tr, seed=SEED), tr[TARGET])
    X_va = build_features(tr, va.drop(columns=[TARGET]))
    val_auc = roc_auc_score(va[TARGET], model.predict_proba(X_va)[:, 1])

    X_all = build_train_features(df, seed=SEED)
    final = make_model().fit(X_all, df[TARGET])
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, ART / "model.joblib")
    metrics = {"val_auc": round(float(val_auc), 4), "features": list(X_all.columns)}
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
