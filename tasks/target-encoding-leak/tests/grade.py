#!/usr/bin/env python3
"""Grader for target-encoding-leak.

Reward = 1 only if every critical check passes, including AUC on 10,000 hidden new loans
meeting the calibrated threshold in thresholds.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

APP = Path(os.environ.get("APP_DIR", "/app"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs/verifier"))
TESTS = Path(__file__).resolve().parent
TARGET = "defaulted"
N_CRITERIA = 5

CONFIG = json.loads((TESTS / "thresholds.json").read_text())
AUC_THRESHOLD = CONFIG["pass_criteria"]["hidden_auc"]["threshold"]
MAX_GAP = CONFIG["honest_report_max_gap"]


def load_build_features():
    path = APP / "pipeline" / "features.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("agent_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_features


class Grader:
    def __init__(self):
        self.criteria: list[dict] = []

    def check(self, milestone, name, critical, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"error: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        self.criteria.append(dict(milestone=milestone, name=name, critical=critical,
                                  passed=bool(ok), detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {milestone} / {name}: {detail}")
        return ok


def main() -> int:
    g = Grader()
    loans = pd.read_csv(APP / "data" / "loans.csv")
    new = pd.read_csv(TESTS / "hidden_data" / "new_loans.csv")
    state: dict = {}

    def artifacts():
        m = json.loads((APP / "artifacts" / "metrics.json").read_text())
        if not isinstance(m.get("val_auc"), (int, float)) or not isinstance(m.get("features"), list):
            return False, "metrics.json needs float val_auc and list features"
        import joblib
        state["model"] = joblib.load(APP / "artifacts" / "model.joblib")
        state["metrics"] = m
        state["build"] = load_build_features()
        return True, "metrics.json, model.joblib and build_features load"

    if not g.check("artifacts", "outputs exist and load", True, artifacts):
        return finish(g, None)
    build, metrics = state["build"], state["metrics"]

    def works_without_target():
        X = build(loans, new.drop(columns=[TARGET]))
        ok = len(X) == len(new) and X.index.equals(new.index)
        return ok, "scores new rows that have no target column" if ok else "output rows/index do not match apply_df"

    def label_shuffle_invariant():
        rng = np.random.default_rng(0)
        with_labels = new.copy()
        shuffled = new.copy()
        shuffled[TARGET] = rng.permutation(shuffled[TARGET].to_numpy())
        a, b = build(loans, with_labels), build(loans, shuffled)
        same = a.shape == b.shape and np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        return same, ("features for new rows ignore their labels" if same
                      else "features for new rows change when their labels are shuffled: the encoder reads apply_df's target")

    g.check("no-target-leak", "works without the target", True, works_without_target)
    g.check("no-target-leak", "new-row features ignore new-row labels", True, label_shuffle_invariant)

    hidden_auc = None

    def hidden_score():
        nonlocal hidden_auc
        X = build(loans, new.drop(columns=[TARGET]))
        hidden_auc = float(roc_auc_score(new[TARGET], state["model"].predict_proba(X)[:, 1]))
        return hidden_auc >= AUC_THRESHOLD, f"hidden AUC {hidden_auc:.4f} (threshold {AUC_THRESHOLD})"

    g.check("performance", "hidden AUC meets threshold", True, hidden_score)

    def honest_report():
        if hidden_auc is None:
            return False, "no hidden AUC to compare with"
        gap = abs(float(metrics["val_auc"]) - hidden_auc)
        return gap <= MAX_GAP, f"reported {metrics['val_auc']:.4f} vs hidden {hidden_auc:.4f} (gap {gap:.4f}, max {MAX_GAP})"

    g.check("honest-reporting", "reported val_auc close to hidden AUC", True, honest_report)
    return finish(g, hidden_auc)


def finish(g: Grader, hidden_auc) -> int:
    critical_ok = all(c["passed"] for c in g.criteria if c["critical"])
    reward = 1 if critical_ok and len(g.criteria) == N_CRITERIA else 0
    rubric = sum(c["passed"] for c in g.criteria) / N_CRITERIA
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "reward.txt").write_text(f"{reward}\n")
    (LOG_DIR / "reward.json").write_text(json.dumps(
        {"reward": reward, "rubric_score": round(rubric, 4), "hidden_auc": hidden_auc}, indent=2))
    (LOG_DIR / "result.json").write_text(json.dumps({"criteria": g.criteria}, indent=2))
    print(f"\nREWARD {reward} | rubric {rubric:.2f} | hidden AUC {hidden_auc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
