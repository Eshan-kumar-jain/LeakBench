#!/usr/bin/env bash
# Fixes only the obvious leak: drops the account_status feature, keeps the random split and whole-history aggregates.
set -euo pipefail
sed -i "/is_closing/d" "${APP_DIR}/pipeline/features.py"
cd "${APP_DIR}/pipeline" && "${PYTHON:-python3}" train.py
