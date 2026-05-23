#!/bin/bash
# ALM + IMPACT: student TinyLlama, teacher Mistral-7B (+ LoRA Dolly SFT)
GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=63$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../common/setup_env.sh
source "${SCRIPT_DIR}/../../common/setup_env.sh"

# model — student: model_hub/TinyLlama | teacher: model_hub/Mistral-7B | teacher LoRA: lora_path/Mistral-7B
CKPT_TYPE="tinyllama"
CKPT_NAME="TinyLlama"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="mistral"
TEACHER_MODEL_NAME="Mistral-7B"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="${BASE_PATH}/lora_path/${TEACHER_MODEL_NAME}"

# data
DATA_DIR="${BASE_PATH}/data/dolly/"

# task
TASK="ALM_IMPACT"
BATCH_SIZE=8
LR=0.001
GRAD_ACC=1
EVAL_BATCH_SIZE=16
EPOCH=15
WARMUP_ITERS=500
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1

# ALM-specific
ALM_BINARIZATION_TEMP=200.0
ALM_BIAS_THRESHOLD=0.05
ALM_LOSS_WEIGHT=3.0
ALM_MODE="merge_by_space_prob+append_space"
MULTITASK_AGG="approx_gradmag_preserve_mag"

# IMPACT hyperparameters
IMPACT_LAMBDA=1.0
IMPACT_TOP_K=8
IMPACT_BI_TAU=1.0
IMPACT_LAMBDA_REG=1.0

# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION="ALM_IMPACT"
KD_OBJ="forward_kl"

CONFIG="${KD_OBJ}-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__alm^w=${ALM_LOSS_WEIGHT}__alm^temp=${ALM_BINARIZATION_TEMP}__impact^λ=${IMPACT_LAMBDA}__impact^k=${IMPACT_TOP_K}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"

# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters ${WARMUP_ITERS}"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${LORA_RANK}"
OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"

# ALM-specific
OPTS+=" --alm-binarization-temp ${ALM_BINARIZATION_TEMP}"
OPTS+=" --alm-bias-threshold ${ALM_BIAS_THRESHOLD}"
OPTS+=" --alm-loss-weight ${ALM_LOSS_WEIGHT}"
OPTS+=" --alm-mode ${ALM_MODE}"
OPTS+=" --multitask-aggregation-fn ${MULTITASK_AGG}"
# IMPACT
OPTS+=" --impact-lambda ${IMPACT_LAMBDA}"
OPTS+=" --impact-top-k ${IMPACT_TOP_K}"
OPTS+=" --impact-bi-tau ${IMPACT_BI_TAU}"
OPTS+=" --impact-lambda-reg ${IMPACT_LAMBDA_REG}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"

OPTS+=" --precision ${PRECISION}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"

# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
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
