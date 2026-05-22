#!/bin/bash
# Chạy tuần tự toàn bộ experiment may3 (DSKD+IMPACT rồi ALM+IMPACT, TinyLlama).
# Usage: bash scripts/anhtue/may3/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SCRIPTS=(
  DSKD_IMPACT.sh
  ALM_IMPACT.sh
)

echo "=== may3: ${#SCRIPTS[@]} jobs ==="
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
echo "=== may3: all jobs completed ==="
