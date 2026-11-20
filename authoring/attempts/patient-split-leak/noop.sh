#!/usr/bin/env bash
# Baseline "agent" that changes nothing and just runs the shipped pipeline.
set -euo pipefail
cd "${APP_DIR}/pipeline" && "${PYTHON:-python3}" train.py
