#!/usr/bin/env python3
"""Grader for patient-split-leak.

The hidden test set is 1,200 patients the agent has never seen. Reward = 1 only if every
critical check passes, including AUC on those new patients meeting the calibrated threshold.
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
TARGET = "readmitted_30d"
N_CRITERIA = 6

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
    visits = pd.read_csv(APP / "data" / "visits.csv")
    new = pd.read_csv(TESTS / "hidden_data" / "new_patients.csv")
    state: dict = {}

    def artifacts():
        m = json.loads((APP / "artifacts" / "metrics.json").read_text())
        for key, typ in [("val_auc", (int, float)), ("features", list),
                         ("train_patient_ids", list), ("val_patient_ids", list)]:
            if not isinstance(m.get(key), typ):
                return False, f"metrics.json key {key!r} missing or wrong type"
        import joblib
        state["model"] = joblib.load(APP / "artifacts" / "model.joblib")
        state["metrics"] = m
        state["build"] = load_build_features()
        return True, "metrics.json, model.joblib and build_features load"

    if not g.check("artifacts", "outputs exist and load", True, artifacts):
        return finish(g, None)
    build, metrics = state["build"], state["metrics"]

    def works_without_target():
        X = build(new.drop(columns=[TARGET]))
        ok = len(X) == len(new) and X.index.equals(new.index)
        return ok, "builds features for new patients without the target" if ok else "output rows/index do not match input"

    def id_invariant():
        # Renumber every patient (same grouping, different numbers). A model that is meant
        # for unseen patients should not care what number a patient happens to have.
        base = new.drop(columns=[TARGET])
        ids = base["patient_id"].unique()
        mapping = dict(zip(ids, np.random.default_rng(0).permutation(ids) + 7_000_000))
        renamed = base.assign(patient_id=base["patient_id"].map(mapping))
        a, b = build(base), build(renamed)
        same = a.shape == b.shape and np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        return same, ("features do not depend on patient ID numbers" if same
                      else "features change when patients are renumbered: patient_id is being used as a feature")

    g.check("no-identity-leak", "works without the target", True, works_without_target)
    g.check("no-identity-leak", "features ignore patient ID numbers", True, id_invariant)

    def group_split():
        tr, va = set(metrics["train_patient_ids"]), set(metrics["val_patient_ids"])
        known = set(visits["patient_id"])
        if not tr or not va:
            return False, "empty train_patient_ids or val_patient_ids"
        if not (tr | va) <= known:
            return False, "reported IDs that are not in visits.csv"
        overlap = tr & va
        return not overlap, (f"{len(va)} validation patients, none seen in training" if not overlap
                             else f"{len(overlap)} patients appear in both training and validation")

    g.check("group-validation", "validation patients unseen in training", True, group_split)

    hidden_auc = None

    def hidden_score():
        nonlocal hidden_auc
        X = build(new.drop(columns=[TARGET]))
        hidden_auc = float(roc_auc_score(new[TARGET], state["model"].predict_proba(X)[:, 1]))
        return hidden_auc >= AUC_THRESHOLD, f"AUC on new patients {hidden_auc:.4f} (threshold {AUC_THRESHOLD})"

    g.check("performance", "new-patient AUC meets threshold", True, hidden_score)

    def honest_report():
        if hidden_auc is None:
            return False, "no hidden AUC to compare with"
        gap = abs(float(metrics["val_auc"]) - hidden_auc)
        return gap <= MAX_GAP, f"reported {metrics['val_auc']:.4f} vs hidden {hidden_auc:.4f} (gap {gap:.4f}, max {MAX_GAP})"

    g.check("honest-reporting", "reported val_auc close to new-patient AUC", True, honest_report)
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
