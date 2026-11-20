#!/usr/bin/env bash
# Stops passing validation labels to the encoder, but still encodes training rows with their own labels.
set -euo pipefail
sed -i "s/X_va = build_features(tr, va)/X_va = build_features(tr, va.drop(columns=[TARGET]))/" "${APP_DIR}/pipeline/train.py"
cd "${APP_DIR}/pipeline" && "${PYTHON:-python3}" train.py
