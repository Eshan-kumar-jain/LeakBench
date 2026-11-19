#!/usr/bin/env python3
"""Grader for leaky-churn-audit.

Reward = 1 only if every critical milestone passes AND the hidden-future AUC meets the
calibrated threshold in thresholds.json. Otherwise 0. A rubric score (share of checks
passed) is also written to reward.json for analysis.

The agent never sees tests/hidden_data/full_history.csv (months 1-18). Months 13-18 are
the "future" the model is scored on, replayed month by month the way production would
see them.
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
TARGET = "churned_next_month"
VISIBLE_LAST = 12
FUTURE_MONTHS = range(13, 19)

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


def snapshot(full: pd.DataFrame, month: int) -> pd.DataFrame:
    """What production can see at the end of `month`: no target, no future rows,
    and account_status not yet reviewed for the current quarter."""
    snap = full[full["month"] <= month].drop(columns=[TARGET]).copy()
    current_quarter = (month - 1) // 3
    not_reviewed = (snap["month"] - 1) // 3 == current_quarter
    snap.loc[not_reviewed, "account_status"] = "active"
    return snap


class Grader:
    def __init__(self):
        self.criteria: list[dict] = []
        self.values: dict = {}

    def check(self, milestone: str, name: str, critical: bool, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # any crash in agent code counts as a failed check
            ok, detail = False, f"error: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        self.criteria.append(dict(milestone=milestone, name=name, critical=critical,
                                  passed=bool(ok), detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {milestone} / {name}: {detail}")
        return ok


def main() -> int:
    g = Grader()
    full = pd.read_csv(TESTS / "hidden_data" / "full_history.csv")
    state: dict = {}

    # Milestone 1: the required artifacts exist and are well formed.
    def artifacts():
        m = json.loads((APP / "artifacts" / "metrics.json").read_text())
        for key, typ in [("val_auc", (int, float)), ("train_months", list),
                         ("val_months", list), ("features", list)]:
            if not isinstance(m.get(key), typ):
                return False, f"metrics.json key {key!r} missing or wrong type"
        import joblib
        state["model"] = joblib.load(APP / "artifacts" / "model.joblib")
        state["metrics"] = m
        state["build_features"] = load_build_features()
        return True, "metrics.json, model.joblib and build_features load"

    if not g.check("artifacts", "outputs exist and load", True, artifacts):
        return finish(g, hidden_auc=None)

    build = state["build_features"]
    metrics = state["metrics"]

    # Milestone 2: no label leak. Features must work without the target column and
    # must not depend on a field that is filled in only after the fact.
    def no_target_use():
        X = build(full[full["month"] <= VISIBLE_LAST].drop(columns=[TARGET]))
        bad = [c for c in X.columns if TARGET in c]
        return (not bad), "build_features runs without the target column" if not bad else f"target-like columns: {bad}"

    def status_not_used():
        base = full[full["month"] <= VISIBLE_LAST].drop(columns=[TARGET])
        flipped = base.copy()
        flipped["account_status"] = np.where(flipped["account_status"] == "active", "closing", "active")
        a, b = build(base), build(flipped)
        same = a.shape == b.shape and np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        return same, ("features unchanged when account_status is flipped" if same
                      else "features depend on account_status, which is only back-filled after each quarter")

    g.check("no-label-leak", "target not used", True, no_target_use)
    g.check("no-label-leak", "back-filled status not used", True, status_not_used)

    # Milestone 3: validation is out-of-time.
    def temporal_split():
        tr, va = metrics["train_months"], metrics["val_months"]
        if not tr or not va:
            return False, "empty train_months or val_months"
        if not set(tr + va) <= set(range(1, VISIBLE_LAST + 1)):
            return False, "months outside 1-12 reported"
        ok = max(tr) < min(va)
        return ok, f"train months end {max(tr)}, validation starts {min(va)}"

    g.check("temporal-validation", "validation months after training months", True, temporal_split)

    # Milestone 4: features at month m must not change when later months are appended.
    def no_future_info():
        for m in (6, 9, 11):
            early = build(full[full["month"] <= m].drop(columns=[TARGET]))
            later = build(full[full["month"] <= m + 3].drop(columns=[TARGET]))
            idx = full.index[full["month"] == m]
            a, b = early.loc[idx].to_numpy(float), later.loc[idx].to_numpy(float)
            if not np.allclose(a, b, equal_nan=True):
                cols = [c for c in early.columns
                        if not np.allclose(early.loc[idx, c].astype(float), later.loc[idx, c].astype(float), equal_nan=True)]
                return False, f"month {m} features change when future rows are added: {cols}"
        return True, "features for a month are identical whether or not later months are present"

    g.check("no-temporal-leak", "features use only past data", True, no_future_info)

    # Score on the hidden future, replayed month by month.
    hidden_auc = None

    def hidden_score():
        nonlocal hidden_auc
        ys, ps = [], []
        for m in FUTURE_MONTHS:
            snap = snapshot(full, m)
            X = build(snap)
            rows = snap["month"] == m
            ps.append(state["model"].predict_proba(X.loc[rows])[:, 1])
            ys.append(full.loc[snap.index[rows], TARGET].to_numpy())
        hidden_auc = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
        ok = hidden_auc >= AUC_THRESHOLD
        return ok, f"hidden-future AUC {hidden_auc:.4f} (threshold {AUC_THRESHOLD})"

    g.check("performance", "hidden-future AUC meets threshold", True, hidden_score)

    # Milestone 5: the reported validation score is honest.
    def honest_report():
        if hidden_auc is None:
            return False, "no hidden AUC to compare with"
        gap = abs(float(metrics["val_auc"]) - hidden_auc)
        return gap <= MAX_GAP, f"reported {metrics['val_auc']:.4f} vs hidden {hidden_auc:.4f} (gap {gap:.4f}, max {MAX_GAP})"

    g.check("honest-reporting", "reported val_auc close to hidden AUC", True, honest_report)
    return finish(g, hidden_auc)


def finish(g: Grader, hidden_auc) -> int:
    critical_ok = all(c["passed"] for c in g.criteria if c["critical"])
    reward = 1 if critical_ok and len(g.criteria) >= 7 else 0
    rubric = sum(c["passed"] for c in g.criteria) / 7
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "reward.txt").write_text(f"{reward}\n")
    (LOG_DIR / "reward.json").write_text(json.dumps(
        {"reward": reward, "rubric_score": round(rubric, 4), "hidden_auc": hidden_auc}, indent=2))
    (LOG_DIR / "result.json").write_text(json.dumps({"criteria": g.criteria}, indent=2))
    print(f"\nREWARD {reward} | rubric {rubric:.2f} | hidden AUC {hidden_auc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
