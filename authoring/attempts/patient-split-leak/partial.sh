#!/usr/bin/env bash
# Drops patient_id as a feature, but keeps the random row split.
set -euo pipefail
sed -i "s/\[\"patient_id\", \"visit_number\"/[\"visit_number\"/" "${APP_DIR}/pipeline/features.py"
cd "${APP_DIR}/pipeline" && python3 train.py
