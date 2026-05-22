#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"
cd "${BASE_PATH}"

for _sh in run_eval.sh run_eval_lora.sh eval_main.sh eval_main_lora.sh; do
  if grep -q $'\r' "${SCRIPT_DIR}/${_sh}" 2>/dev/null; then
    echo "ERROR: ${SCRIPT_DIR}/${_sh} has CRLF line endings. Run: sed -i 's/\\r\$//' scripts/eval/*.sh" >&2
    exit 1
  fi
  if ! bash -n "${SCRIPT_DIR}/${_sh}"; then
    echo "ERROR: bash -n failed for scripts/eval/${_sh}" >&2
    exit 1
  fi
done

EVAL_BATCH_SIZE="${1:-8}"
NUM_JOBS=9
FAILED=0

run_eval() {
  local label="$1"
  local ckpt="$2"
  if [[ ! -d "${ckpt}" ]]; then
    echo "[SKIP] ${label}: checkpoint not found: ${ckpt}" >&2
    FAILED=$((FAILED + 1))
    return 0
  fi
  echo ""
  echo "================================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label}"
  echo "  checkpoint: ${ckpt}"
  echo "  eval batch: ${EVAL_BATCH_SIZE}"
  echo "================================================================================"
  if bash "${SCRIPT_DIR}/run_eval.sh" "${ckpt}" "${EVAL_BATCH_SIZE}"; then
    echo "[OK] ${label}"
  else
    echo "[FAILED] ${label}" >&2
    FAILED=$((FAILED + 1))
  fi
}

run_eval_lora() {
  local label="$1"
  local ckpt="$2"
  if [[ ! -d "${ckpt}" ]]; then
    echo "[SKIP] ${label}: checkpoint not found: ${ckpt}" >&2
    FAILED=$((FAILED + 1))
    return 0
  fi
  echo ""
  echo "================================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} (LoRA)"
  echo "  checkpoint: ${ckpt}"
  echo "  eval batch: ${EVAL_BATCH_SIZE}"
  echo "================================================================================"
  if bash "${SCRIPT_DIR}/run_eval_lora.sh" "${ckpt}" "${EVAL_BATCH_SIZE}"; then
    echo "[OK] ${label}"
  else
    echo "[FAILED] ${label}" >&2
    FAILED=$((FAILED + 1))
  fi
}

# --- IDEAS.SH (91-126) ---

# run_eval "GPT2-340M | DSKD v2 + IMPACT" \
#   "${BASE_PATH}/outputs/GPT2-340M/dskd_v2_ipact/criterion=dual_space_kd_v2_ipact__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.8__kd^temp=2.0__ipact^λ=0.25__ipact^k=4__epoch=20__bsz=8x2x1=16__lr=0.0005__proj^lr=0.0005/epoch19_step13585_loss8.2566_rougel28.2791"

# run_eval "GPT2-120M | DSKD v2 + IMPACT" \
#   "${BASE_PATH}/outputs/GPT2-120M/dskd_v2_ipact/criterion=dual_space_kd_v2_ipact__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.8__kd^temp=2.0__ipact^λ=0.25__ipact^k=6__epoch=20__bsz=4x2x1=8__lr=0.0005__proj^lr=0.0005/epoch19_step27151_loss9.7286_rougel26.2435"

# run_eval "GPT2-340M | SRA + IMPACT" \
#   "${BASE_PATH}/outputs/GPT2-340M/SRE_IPACT/criterion=SRE_IPACT__forward_kl-bf16__teacher=Qwen1.5-1.8B__sra^alpha=0.2__geom^w=50.0__epoch=20__bsz=8x1x1=8__lr=0.0005__proj^lr=0.0005/epoch19_step27170_loss4.9934_rougel26.0766"

# run_eval "GPT2-340M-FT | ALM + IMPACT" \
#   "${BASE_PATH}/outputs/GPT2-340M-FT/ALM_IMPACT/criterion=ALM_IMPACT__forward_kl-bf16__teacher=Qwen1.5-1.8B__alm^align=unconstrained__agg=approx_gradmag_preserve_mag__alm^temp=100.0__alm^bias=0.1__impact^k=4__epoch=15__bsz=16x2x1=32__lr=1e-5/epoch1_step357_loss7.6068_rougel24.5925"

# run_eval "GPT2-120M-SFT | ALM + IMPACT (bsz=32, lr=5e-4)" \
#   "${BASE_PATH}/outputs/GPT2-120M-SFT/ALM_IMPACT/criterion=ALM_IMPACT__forward_kl-bf16__teacher=Qwen1.5-1.8B__alm^align=unconstrained__agg=approx_gradmag_preserve_mag__alm^temp=100.0__impact^k=4__epoch=15__bsz=8x4x1=32__lr=0.0005/epoch6_step2142_loss4.9130_rougel21.7920"

# run_eval "GPT2-120M-SFT | ALM + IMPACT (bsz=64, lr=2e-6)" \
#   "${BASE_PATH}/outputs/GPT2-120M-SFT/ALM_IMPACT/criterion=ALM_IMPACT__forward_kl-bf16__teacher=Qwen1.5-1.8B__alm^align=unconstrained__agg=approx_gradmag_preserve_mag__alm^temp=100.0__impact^k=4__epoch=15__bsz=16x4x1=64__lr=2e-6/epoch12_step2136_loss5.4129_rougel25.7256"

# run_eval "GPT2-120M | SRA + IMPACT" \
#   "${BASE_PATH}/outputs/GPT2-120M/SRE_IPACT/criterion=SRE_IPACT__forward_kl-bf16__teacher=Qwen1.5-1.8B__sra^alpha=0.85__geom^w=100.0__epoch=20__bsz=4x4x1=16__lr=0.0005__proj^lr=0.0005/epoch20_step14280_loss5.6270_rougel24.9866"

# run_eval_lora "GPT2-1.5B | DSKD v2 + IMPACT (LoRA)" \
#   "${BASE_PATH}/outputs/GPT2-1.5B/dskd_v2_ipact/criterion=dual_space_kd_v2_ipact__forward_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=Qwen2.5-7B-Instruct__kd^rate=0.8__kd^temp=2.0__ipact^λ=1.0__ipact^k=6__epoch=15__bsz=4x2x1=8__lr=0.001__proj^lr=0.0005/epoch10_step14290_loss3.3390_rougel30.8348"

run_eval_lora "GPT2-1.5B | ALM + IMPACT (LoRA)" \
  "${BASE_PATH}/outputs/GPT2-1.5B/ALM_IPACT/criterion=ALM_IPACT__forward_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=Qwen2.5-7B-Instruct__alm^w=3.0__alm^temp=100.0__epoch=15__bsz=4x2x1=8__lr=0.001/epoch14_step20006_loss2.5310_rougel26.8773"

echo ""
echo "================================================================================"
if [[ "${FAILED}" -eq 0 ]]; then
  echo "Done. All ${NUM_JOBS} checkpoint evals finished successfully."
else
  echo "Done with ${FAILED}/${NUM_JOBS} failed job(s). See logs above."
  exit 1
fi
