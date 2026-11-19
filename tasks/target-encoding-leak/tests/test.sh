#!/usr/bin/env bash
# Harbor verifier entrypoint: runs the grader, which writes reward.txt (0 or 1) and reward.json.
set -uo pipefail

LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "${LOG_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/grade.py"
status=$?

# Always leave a reward behind, even if the grader crashed.
if [[ ! -f "${LOG_DIR}/reward.txt" ]]; then
  echo 0 > "${LOG_DIR}/reward.txt"
fi
exit ${status}
