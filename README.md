# LeakBench

**Can AI coding agents find and fix the data leakage that makes ML results look better than they are?**

LeakBench is a small benchmark of three tasks in the [Harbor](https://harborframework.com) task format (the format used by Terminal-Bench 2.0). In each task the agent receives a working ML pipeline that reports a flattering validation score. The pipeline is leaky. The agent must find the leaks, fix them, and leave behind a model whose reported score is an honest estimate of performance on data it has never seen.

Every task is graded on behaviour, not on code. The grader never searches the agent's code for a column name; it re-runs the agent's feature code under controlled changes and checks what moves, then scores the model on hidden data the agent never sees.

## The tasks

| Task | Story | Planted leaks | Hidden test data |
| --- | --- | --- | --- |
| [`leaky-churn-audit`](tasks/leaky-churn-audit) | Monthly churn model reports AUC 0.98 | A CRM field back-filled after each quarter (label leak); whole-history aggregates that see future months; a random split on time-ordered data | Months 13-18, replayed month by month as production would see them |
| [`target-encoding-leak`](tasks/target-encoding-leak) | Loan-default model reports AUC 0.82 | Employer default rates computed with the validation rows' own labels, and training rows encoded with their own labels (1,500 employers, many with only a few loans) | 10,000 new loan applications |
| [`patient-split-leak`](tasks/patient-split-leak) | Readmission model reports AUC 0.74 | Random row split puts the same patient in training and validation; patient ID used as a feature | 1,200 patients never seen before |

## How grading works

Each task's `tests/grade.py` runs a set of checks; the reward is 1 only if all critical checks pass.

| Check type | How it is tested |
| --- | --- |
| No future information | Build features with and without later months appended; values for earlier months must be identical |
| No label leakage | Shuffle the labels of the rows being scored (or flip the back-filled field); their features must not change |
| No identity leakage | Renumber every patient; features must not change |
| Correct validation design | Validation months come after training months; validation patients never appear in training |
| Performance | AUC on the hidden data meets a calibrated threshold |
| Honest reporting | The pipeline's own `val_auc` is within 0.05 of the hidden AUC |

Thresholds live in each task's `tests/thresholds.json`. They were calibrated by running the scripted attempts below: high enough that half-fixes fail, low enough that any honest, sensible model passes.

## Results

Three scripted baseline attempts per task, run through the local harness. The oracle was run with 5 random seeds to check that the calibration is stable. Full table: [`results/summary.md`](results/summary.md).

| Task | Attempt | Pass rate | Reported AUC | Hidden AUC |
| --- | --- | --- | --- | --- |
| leaky-churn-audit | no-op | 0/1 | 0.982 | 0.679 |
| | partial fix | 0/1 | 0.870 | 0.722 |
| | oracle | 5/5 | 0.735 ± 0.001 | 0.750 ± 0.002 |
| target-encoding-leak | no-op | 0/1 | 0.820 | 0.658 |
| | partial fix | 0/1 | 0.642 | 0.658 |
| | oracle | 5/5 | 0.699 ± 0.005 | 0.700 ± 0.003 |
| patient-split-leak | no-op | 0/1 | 0.738 | 0.683 |
| | partial fix | 0/1 | 0.721 | 0.701 |
| | oracle | 5/5 | 0.802 ± 0.007 | 0.782 ± 0.001 |

The "partial fix" attempts model a plausible agent mistake: fixing the most obvious leak and stopping. They fail every task, which is the point of grading behaviour across several checks. Note the 95% Wilson interval on 5/5 oracle passes is still [0.57, 1.00]: five trials are enough to show the task is solvable, not to estimate a pass rate precisely.

## Repository layout

```
tasks/<task>/                 one Harbor task each
  task.toml                   limits and resources (no internet)
  instruction.md              what the agent is told
  environment/Dockerfile      Python 3.11, pinned pandas / numpy / scikit-learn
  environment/data/           visible data + data dictionary
  environment/pipeline/       the leaky pipeline the agent must fix
  solution/                   oracle: fixed pipeline + solve.sh
  tests/test.sh               Harbor entrypoint; writes /logs/verifier/reward.txt
  tests/grade.py              the checks
  tests/thresholds.json       calibrated pass criteria
  tests/hidden_data/          data the agent never sees
authoring/
  generate_*.py               deterministic data generators (seeded)
  attempts/<task>/*.sh        scripted baseline attempts: noop, partial, oracle
harness/run_local.py          runs task x attempt x seed without Docker, logs results/results.jsonl
analysis/analyze.py           pass rates with Wilson CIs, AUC mean ± CI, failed-check counts
results/                      results.jsonl and summary.md
```

## Run it

Requirements: Python 3.11+, pandas, numpy, scikit-learn, joblib, tabulate.

```bash
python harness/run_local.py      # ~3 minutes on a laptop CPU; writes results/results.jsonl
python analysis/analyze.py       # writes results/summary.md
```

Build a task's container (what an agent would work inside):

```bash
cd tasks/leaky-churn-audit/environment
docker build -t leakbench-churn .
docker run -it leakbench-churn bash
```

Run real agents with Harbor (`pip install harbor`):

```bash
harbor run -p tasks/leaky-churn-audit -a oracle                    # sanity check: reward 1
harbor run -p tasks/leaky-churn-audit -a claude-code -m <model>    # a real agent attempt
```

## Limitations

- The data is synthetic, generated with fixed seeds, so leaks and effect sizes are controlled; real datasets are messier.
- The baselines are scripted, not real agents. Running Claude Code, Codex and others through Harbor is the next step.
- Three tasks is a demonstration of method, not a benchmark large enough to rank agents.
- The thresholds are calibrated against the scripted attempts; an agent could find a legitimate approach that scores below them.

Built with the help of Claude Code.
