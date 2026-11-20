#!/usr/bin/env python3
"""A small local harness: runs every task x attempt x seed the way Harbor would, without Docker.

For each trial it
  1. makes a fresh copy of the task's environment (data + pipeline) in a temp folder,
  2. runs the attempt script (a scripted "agent": noop, partial or oracle),
  3. runs the task's verifier (tests/test.sh), which writes reward.json and result.json,
  4. appends one JSON line describing the trial to results/results.jsonl.

Usage:
  python harness/run_local.py                       # all tasks, default attempts and seeds
  python harness/run_local.py --tasks target-encoding-leak --oracle-seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["leaky-churn-audit", "target-encoding-leak", "patient-split-leak"]

# On Windows, a bare "bash" can resolve to C:\Windows\System32\bash.exe (the WSL launcher)
# before PATH is searched. shutil.which follows PATH order, so from Git Bash it finds Git's bash.
BASH = shutil.which("bash") or "bash"


def run_trial(task: str, attempt: str, seed: int) -> dict:
    task_dir = ROOT / "tasks" / task
    with tempfile.TemporaryDirectory() as tmp:
        app, logs = Path(tmp) / "app", Path(tmp) / "logs" / "verifier"
        shutil.copytree(task_dir / "environment" / "data", app / "data")
        shutil.copytree(task_dir / "environment" / "pipeline", app / "pipeline")
        env = {**os.environ, "APP_DIR": str(app), "LOG_DIR": str(logs),
               "TASK_DIR": str(task_dir), "SEED": str(seed),
               "PYTHON": sys.executable}  # same Python everywhere, incl. Windows where python3 may not exist

        start = time.time()
        agent = subprocess.run([BASH, str(ROOT / "authoring" / "attempts" / task / f"{attempt}.sh")],
                               env=env, capture_output=True, text=True)
        agent_sec = time.time() - start
        verify = subprocess.run([BASH, str(task_dir / "tests" / "test.sh")],
                                env=env, capture_output=True, text=True)

        reward = json.loads((logs / "reward.json").read_text()) if (logs / "reward.json").exists() else {}
        criteria = json.loads((logs / "result.json").read_text())["criteria"] if (logs / "result.json").exists() else []
        metrics_path = app / "artifacts" / "metrics.json"
        reported = json.loads(metrics_path.read_text()).get("val_auc") if metrics_path.exists() else None

    if not reward:  # nothing graded: show why, so failures are never silent
        for name, proc in (("attempt", agent), ("verifier", verify)):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            print(f"  [{task}/{attempt}] {name} exit {proc.returncode}: " + " | ".join(tail))

    hidden = reward.get("hidden_auc")
    return {
        "task": task,
        "attempt": attempt,
        "seed": seed,
        "reward": int((reward.get("reward") or 0)),
        "rubric_score": reward.get("rubric_score", 0.0),
        "reported_val_auc": reported,
        "hidden_auc": hidden,
        "gap": round(reported - hidden, 4) if reported is not None and hidden is not None else None,
        "failed_checks": [f"{c['milestone']}/{c['name']}" for c in criteria if not c["passed"]],
        "agent_exit_code": agent.returncode,
        "verifier_exit_code": verify.returncode,
        "agent_seconds": round(agent_sec, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--attempts", nargs="*", default=["noop", "partial", "oracle"])
    p.add_argument("--oracle-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=str(ROOT / "results" / "results.jsonl"))
    args = p.parse_args()

    print(f"using bash: {BASH}\nusing python: {sys.executable}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for task in args.tasks:
            for attempt in args.attempts:
                # Scripted noop/partial attempts are deterministic, so one seed is enough for them.
                seeds = args.oracle_seeds if attempt == "oracle" else [0]
                for seed in seeds:
                    row = run_trial(task, attempt, seed)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{task:22s} {attempt:8s} seed={seed} reward={row['reward']} "
                          f"reported={row['reported_val_auc']} hidden={row['hidden_auc'] and round(row['hidden_auc'], 4)} "
                          f"failed={len(row['failed_checks'])}")


if __name__ == "__main__":
    main()
