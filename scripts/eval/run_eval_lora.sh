#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS=(${ICARE_EVAL_GPU:-0})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
GPUS_PER_NODE=${#GPUS[@]}

CKPT_PATH=$1
EVAL_BATCH_SIZE=${2:-4}
MASTER_PORT=7000

for DATASET in self-inst dolly vicuna "sinst/11_"; do
  for SEED in 10 20 30 40 50; do
    bash "${SCRIPT_DIR}/eval_main_lora.sh" \
         "${CUDA_VISIBLE_DEVICES}" \
         ${MASTER_PORT} \
         ${GPUS_PER_NODE} \
         "${WORK_DIR}" \
         "${CKPT_PATH}" \
         "${DATASET}" \
         ${EVAL_BATCH_SIZE} \
         ${SEED}
    MASTER_PORT=$((MASTER_PORT + 1))
  done
done
