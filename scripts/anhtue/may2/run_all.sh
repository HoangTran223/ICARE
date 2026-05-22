#!/bin/bash
# Chạy tuần tự toàn bộ experiment may2 (DSKDv2+IMPACT, ablation kd^rate trên OPT-2.7B).
# Usage: bash scripts/anhtue/may2/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SCRIPTS=(
  DSKDv2_IMPACT_0.8.sh
  DSKDv2_IMPACT_0.3.sh
)

echo "=== may2: ${#SCRIPTS[@]} jobs ==="
for s in "${SCRIPTS[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${s}" ]]; then
    echo "Missing: ${s}" >&2
    exit 1
  fi
  echo ""
  echo ">>> [$(date -Iseconds)] Starting ${s}"
  bash "${SCRIPT_DIR}/${s}"
  echo ">>> [$(date -Iseconds)] Finished ${s}"
done
echo ""
echo "=== may2: all jobs completed ==="
