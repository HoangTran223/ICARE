#!/bin/bash
# ALM + IMPACT — GPT2-340M-FT + Qwen1.5-1.8B on Dolly
#
# Deep-dive summary (outputs/GPT2-340M-FT/ALM_IMPACT/, May 2026):
#   • Best dev rougeL: 24.59 @ epoch 1 (16×2×1=32, LR=1e-5); checkpoint kept by keep_best_n_checkpoints.
#   • 16×4×1=64: peak ~24.35 @ ep3 — worse + slower steps (~4.2s vs ~2.1s).
#   • After ep1: rougeL drifts ~24.6 → ~23.0 while train nll_loss rises (FT already strong; GradMag + IMPACT nudge).
#   • Logged alm_loss≈0 on FT: student already matches teacher on ALM binary path → GradMag assigns ~94% weight
#     to ALM task but its gradient is ~0; effective training ≈ IMPACT only (gradmag_weight_impact ~6–30%).
#   • gradmag_weight_sft≈0: ||g_sft|| ≫ ||g_impact|| on last block (tokenkit inverse-norm behavior).
#   • impact_lambda is IGNORED when MULTITASK_AGG=approx_gradmag_preserve_mag (see ALM_IMPACT.py).
#
# ALM paper / tokenkit: τ=100, bias=0.1, unconstrained, merge_by_space_prob+append_space, GradMag preserve_mag.
GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"

CKPT_TYPE="gpt2"
CKPT_NAME="GPT2-340M-FT"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"

DATA_DIR="${BASE_PATH}/data/dolly/"

TASK="ALM_IMPACT"

# Global batch 32 (best log); do not use 64 — no rouge gain, 2× step time
BATCH_SIZE=16
GRAD_ACC=2

LR=1e-5
WARMUP_ITERS=300
LR_DECAY_STYLE="constant"
EVAL_BATCH_SIZE=16
# Peak @ ep1; ep5 enough with keep_best_n_checkpoints=1
EPOCH=5

ALM_BINARIZATION_TEMP=100.0
ALM_BIAS_THRESHOLD=0.1
ALM_MODE="merge_by_space_prob+append_space"
ALM_ALIGNMENT="unconstrained"
MULTITASK_AGG="approx_gradmag_preserve_mag"
# Only used if MULTITASK_AGG=none
ALM_LOSS_WEIGHT=3.0

# IMPACT: top_k=4, bi_tau=1.0 matched best 16×2 run; lambda only matters without GradMag
IMPACT_LAMBDA=0.25
IMPACT_TOP_K=4
IMPACT_BI_TAU=1.0
IMPACT_LAMBDA_REG=1.0

MAX_LENGTH=512
PRECISION="bf16"
CRITERION="ALM_IMPACT"
KD_OBJ="forward_kl"

CONFIG="${KD_OBJ}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__alm^align=${ALM_ALIGNMENT}__agg=${MULTITASK_AGG}__alm^temp=${ALM_BINARIZATION_TEMP}__alm^bias=${ALM_BIAS_THRESHOLD}__impact^k=${IMPACT_TOP_K}__impact^lam=${IMPACT_LAMBDA}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"

OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --task ${TASK}"

OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters ${WARMUP_ITERS}"
OPTS+=" --lr-decay-style ${LR_DECAY_STYLE}"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"

OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --alm-binarization-temp ${ALM_BINARIZATION_TEMP}"
OPTS+=" --alm-bias-threshold ${ALM_BIAS_THRESHOLD}"
OPTS+=" --alm-mode ${ALM_MODE}"
OPTS+=" --alm-alignment ${ALM_ALIGNMENT}"
OPTS+=" --multitask-aggregation-fn ${MULTITASK_AGG}"
OPTS+=" --alm-loss-weight ${ALM_LOSS_WEIGHT}"

OPTS+=" --impact-lambda ${IMPACT_LAMBDA}"
OPTS+=" --impact-top-k ${IMPACT_TOP_K}"
OPTS+=" --impact-bi-tau ${IMPACT_BI_TAU}"
OPTS+=" --impact-lambda-reg ${IMPACT_LAMBDA_REG}"

OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"

OPTS+=" --precision ${PRECISION}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 100"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --seed ${SEED}"

OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}:${BASE_PATH}/ALM/tokenkit-main:${BASE_PATH}/ALM
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation_alm.py ${OPTS}"

${CMD} 2>&1 | tee "${SAVE_PATH}/train.log"
