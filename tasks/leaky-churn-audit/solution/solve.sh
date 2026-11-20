#!/usr/bin/env bash
# Oracle: replace the leaky pipeline with the fixed one and retrain.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-/app}"

cp "${SCRIPT_DIR}/features.py" "${APP_DIR}/pipeline/features.py"
cp "${SCRIPT_DIR}/train.py" "${APP_DIR}/pipeline/train.py"
cd "${APP_DIR}/pipeline" && "${PYTHON:-python3}" train.py
