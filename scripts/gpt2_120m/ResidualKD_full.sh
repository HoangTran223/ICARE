#!/bin/bash
# ResidualKD Full Pipeline: Stage 1 (Pretrain) + Stage 2 (Finetune)
# Usage: bash scripts/gpt2_340m/ResidualKD_full.sh
set -e

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

# model
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"

CKPT_TYPE="gpt2"
CKPT_NAME="GPT2-340M"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"

# data
DATA_DIR="${BASE_PATH}/data/dolly/"

# common
PRECISION="bf16"
SEED=10

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# ============================================================
# Stage 1: Projector Pretraining
# ============================================================
echo "============================================================"
echo "Stage 1: Projector Pretraining"
echo "============================================================"

MASTER_PORT_S1=66$(($RANDOM%90+10))
DISTRIBUTED_ARGS_S1="--nproc_per_node $GPUS_PER_NODE \
                     --nnodes $NNODES \
                     --node_rank $NODE_RANK \
                     --master_addr $MASTER_ADDR \
                     --master_port $MASTER_PORT_S1"

D_BOTTLENECK=64
S1_BATCH_SIZE=8
S1_LR=0.001
S1_GRAD_ACC=1
S1_EVAL_BATCH_SIZE=16
S1_EPOCH=10

S1_SETTING="residualkd_pretrain__teacher=${TEACHER_MODEL_NAME}__d_bn=${D_BOTTLENECK}__epoch=${S1_EPOCH}__bsz=${S1_BATCH_SIZE}x${S1_GRAD_ACC}x${GPUS_PER_NODE}__lr=${S1_LR}"
S1_SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/ResidualKD_pretrain/${S1_SETTING}"

mkdir -p ${S1_SAVE_PATH}

S1_OPTS=""
S1_OPTS+=" --base-path ${BASE_PATH}"
S1_OPTS+=" --model-type ${CKPT_TYPE}"
S1_OPTS+=" --model-path ${CKPT_PATH}"
S1_OPTS+=" --n-gpu ${GPUS_PER_NODE}"
S1_OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
S1_OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
S1_OPTS+=" --model-dtype bf16"
S1_OPTS+=" --data-dir ${DATA_DIR}"
S1_OPTS+=" --num-workers 0"
S1_OPTS+=" --dev-num 500"
S1_OPTS+=" --task ResidualKD_pretrain"
S1_OPTS+=" --lr ${S1_LR}"
S1_OPTS+=" --batch-size ${S1_BATCH_SIZE}"
S1_OPTS+=" --eval-batch-size ${S1_EVAL_BATCH_SIZE}"
S1_OPTS+=" --gradient-accumulation-steps ${S1_GRAD_ACC}"
S1_OPTS+=" --warmup-iters 0"
S1_OPTS+=" --lr-decay-style cosine"
S1_OPTS+=" --weight-decay 1e-2"
S1_OPTS+=" --clip-grad 1.0"
S1_OPTS+=" --num-epochs ${S1_EPOCH}"
S1_OPTS+=" --residualkd-d-bottleneck ${D_BOTTLENECK}"
S1_OPTS+=" --max-length 512"
S1_OPTS+=" --max-prompt-length 256"
S1_OPTS+=" --do-train"
S1_OPTS+=" --save-interval 1"
S1_OPTS+=" --eval-interval 1"
S1_OPTS+=" --log-interval 50"
S1_OPTS+=" --save-dir ${S1_SAVE_PATH}"
S1_OPTS+=" --keep-best-n-checkpoints 2"
S1_OPTS+=" --criterion cross_entropy"
S1_OPTS+=" --seed ${SEED}"
S1_OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    S1_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    S1_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    S1_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi

# S1_CMD="torchrun ${DISTRIBUTED_ARGS_S1} ${BASE_PATH}/code/distillation_residualkd_pretrain.py ${S1_OPTS}"
# echo "Stage 1 command: ${S1_CMD}"
# ${S1_CMD}

# echo ""
# echo "Stage 1 completed. Best projector saved at: ${S1_SAVE_PATH}/projector_best.pt"
# echo ""

# ============================================================
# Stage 2: Residual Knowledge Distillation
# ============================================================
echo "============================================================"
echo "Stage 2: Residual Knowledge Distillation"
echo "============================================================"

MASTER_PORT_S2=66$(($RANDOM%90+10))
DISTRIBUTED_ARGS_S2="--nproc_per_node $GPUS_PER_NODE \
                     --nnodes $NNODES \
                     --node_rank $NODE_RANK \
                     --master_addr $MASTER_ADDR \
                     --master_port $MASTER_PORT_S2"

S2_BATCH_SIZE=4
S2_LR=0.0005
S2_GRAD_ACC=1
S2_EVAL_BATCH_SIZE=16
S2_EPOCH=20
LAMBDA_RES=0.5
LAMBDA_WARMUP=50
PROJECTOR_LR=0.001
PROJECTOR_LOAD_PATH="${S1_SAVE_PATH}/projector_best.pt"
CRITERION="ResidualKD"

S2_CONFIG="${CRITERION}-${PRECISION}"
S2_SETTING="criterion=${CRITERION}__${S2_CONFIG}__teacher=${TEACHER_MODEL_NAME}__lambda=${LAMBDA_RES}__d_bn=${D_BOTTLENECK}__epoch=${S2_EPOCH}__bsz=${S2_BATCH_SIZE}x${S2_GRAD_ACC}x${GPUS_PER_NODE}=$((S2_BATCH_SIZE * S2_GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${S2_LR}__proj^lr=${PROJECTOR_LR}"
S2_SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/ResidualKD/${S2_SETTING}"

mkdir -p ${S2_SAVE_PATH}

S2_OPTS=""
S2_OPTS+=" --base-path ${BASE_PATH}"
S2_OPTS+=" --model-type ${CKPT_TYPE}"
S2_OPTS+=" --model-path ${CKPT_PATH}"
S2_OPTS+=" --n-gpu ${GPUS_PER_NODE}"
S2_OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
S2_OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
S2_OPTS+=" --teacher-model-fp16"
S2_OPTS+=" --gradient-checkpointing"
S2_OPTS+=" --model-dtype bf16"
S2_OPTS+=" --data-dir ${DATA_DIR}"
S2_OPTS+=" --num-workers 0"
S2_OPTS+=" --dev-num 1000"
S2_OPTS+=" --task ResidualKD"
S2_OPTS+=" --lr ${S2_LR}"
S2_OPTS+=" --batch-size ${S2_BATCH_SIZE}"
S2_OPTS+=" --eval-batch-size ${S2_EVAL_BATCH_SIZE}"
S2_OPTS+=" --gradient-accumulation-steps ${S2_GRAD_ACC}"
S2_OPTS+=" --warmup-iters 0"
S2_OPTS+=" --lr-decay-style cosine"
S2_OPTS+=" --weight-decay 1e-2"
S2_OPTS+=" --clip-grad 1.0"
S2_OPTS+=" --num-epochs ${S2_EPOCH}"
S2_OPTS+=" --residualkd-lambda-res ${LAMBDA_RES}"
S2_OPTS+=" --residualkd-lambda-warmup ${LAMBDA_WARMUP}"
S2_OPTS+=" --residualkd-d-bottleneck ${D_BOTTLENECK}"
S2_OPTS+=" --residualkd-projector-load-path ${PROJECTOR_LOAD_PATH}"
S2_OPTS+=" --residualkd-cross-tokenizer"
S2_OPTS+=" --projector-lr ${PROJECTOR_LR}"
S2_OPTS+=" --max-length 512"
S2_OPTS+=" --max-prompt-length 256"
S2_OPTS+=" --do-train"
S2_OPTS+=" --do-valid"
S2_OPTS+=" --eval-gen"
S2_OPTS+=" --precision ${PRECISION}"
S2_OPTS+=" --save-interval 1"
S2_OPTS+=" --eval-interval 1"
S2_OPTS+=" --log-interval 100"
S2_OPTS+=" --save-dir ${S2_SAVE_PATH}"
S2_OPTS+=" --keep-best-n-checkpoints 1"
S2_OPTS+=" --criterion ${CRITERION}"
S2_OPTS+=" --hidden-dim-student 1024"
S2_OPTS+=" --hidden-dim-teacher 2048"
S2_OPTS+=" --seed ${SEED}"
S2_OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    S2_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    S2_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    S2_OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
S2_OPTS+=" --do-sample"
S2_OPTS+=" --top-k 0"
S2_OPTS+=" --top-p 1.0"
S2_OPTS+=" --temperature 1.0"

S2_CMD="torchrun ${DISTRIBUTED_ARGS_S2} ${BASE_PATH}/code/distillation_residualkd.py ${S2_OPTS}"
echo "Stage 2 command: ${S2_CMD}"
${S2_CMD}

echo ""
echo "============================================================"
echo "ResidualKD Full Pipeline Completed!"
echo "Stage 1 outputs: ${S1_SAVE_PATH}"
echo "Stage 2 outputs: ${S2_SAVE_PATH}"
echo "============================================================"
