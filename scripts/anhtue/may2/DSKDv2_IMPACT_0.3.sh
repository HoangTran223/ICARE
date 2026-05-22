#!/bin/bash
# DSKDv2 + IMPACT: OPT-2.7B (LoRA) <- Qwen2.5-7B-Instruct (LoRA) — HP aligned with scripts/gptxl/
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
BASE_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DSKD_ROOT="${BASE_PATH}/DSKDv2"
# shellcheck source=../../common/setup_env.sh
source "${SCRIPT_DIR}/../../common/setup_env.sh"

CKPT_TYPE="opt"
CKPT_NAME="OPT-2.7B"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="${BASE_PATH}/lora_path/${TEACHER_MODEL_NAME}"

DATA_DIR="${BASE_PATH}/data/dolly/"

TASK="dskd_v2_impact"
BATCH_SIZE=4
LR=0.001
GRAD_ACC=2
EVAL_BATCH_SIZE=8
EPOCH=15
KD_RATE=0.3
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1

IMPACT_LAMBDA=0.25
IMPACT_TOP_K=4
IMPACT_BI_TAU=1.0
IMPACT_LAMBDA_REG=1.0

PROJECTOR_LR=0.0005
TOPK_VOCAB=-1
MAX_LENGTH=512
PRECISION="bf16"
CRITERION="dual_space_kd_v2_impact"
KD_OBJ="forward_kl"

CONFIG="${KD_OBJ}-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__impact^λ=${IMPACT_LAMBDA}__impact^k=${IMPACT_TOP_K}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}__proj^lr=${PROJECTOR_LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
SEED=10

mkdir -p "${SAVE_PATH}"

OPTS=""
OPTS+=" --base-path ${DSKD_ROOT}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --model-dtype ${PRECISION}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
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
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${LORA_RANK}"
OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"

OPTS+=" --init-t2s-projector"
OPTS+=" --init-s2t-projector"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --topk-vocab ${TOPK_VOCAB}"

OPTS+=" --impact-lambda ${IMPACT_LAMBDA}"
OPTS+=" --impact-top-k ${IMPACT_TOP_K}"
OPTS+=" --impact-bi-tau ${IMPACT_BI_TAU}"
OPTS+=" --impact-lambda-reg ${IMPACT_LAMBDA_REG}"

OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --seed ${SEED}"

OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${DSKD_ROOT}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${DSKD_ROOT}/configs/deepspeed/ds_config.json"
fi

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${DSKD_ROOT}
CMD="torchrun ${DISTRIBUTED_ARGS} ${DSKD_ROOT}/code/distillation.py ${OPTS}"

echo "${CMD}"
${CMD} 2>&1 | tee "${SAVE_PATH}/train.log"
