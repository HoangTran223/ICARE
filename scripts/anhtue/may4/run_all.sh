#!/bin/bash
# Chạy tuần tự toàn bộ experiment may4 (SRA+IMPACT, ResidualKD full pipeline, TinyLlama).
# Usage: bash scripts/anhtue/may4/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SCRIPTS=(
  SRA_IMPACT.sh
  ResidualKD_IMPACT_full.sh
)

echo "=== may4: ${#SCRIPTS[@]} jobs ==="
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
echo "=== may4: all jobs completed ==="
