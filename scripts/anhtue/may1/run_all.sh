#!/bin/bash
# Chạy tuần tự toàn bộ experiment may1 (DSKDv2+IMPACT, ablation impact^λ trên TinyLlama).
# Usage (từ repo root):  bash scripts/anhtue/may1/run_all.sh
# Hoặc:                cd scripts/anhtue/may1 && bash run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUNNER="$(basename "$0")"
SCRIPTS=(
  DSKDv2_IMPACT_0.25.sh
  DSKDv2_IMPACT_1.sh
  DSKDv2_IMPACT_2.sh
  DSKDv2_IMPACT_5.sh
)

echo "=== may1: ${#SCRIPTS[@]} jobs ==="
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
echo "=== may1: all jobs completed ==="
