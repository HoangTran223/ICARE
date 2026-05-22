#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=${1}
MASTER_PORT=${2}
GPUS_PER_NODE=${3}

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=${4}
CKPT_PATH=${5}

# Base model: adapter_config.json (PEFT) or model_hub/<student> from checkpoint path
ADAPTER_CONFIG="${CKPT_PATH}/adapter_config.json"
if [[ -f "${ADAPTER_CONFIG}" ]]; then
  MODEL_PATH=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' "${ADAPTER_CONFIG}")
else
  TASK_DIR=$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")
  CKPT_NAME=$(basename "${TASK_DIR}")
  MODEL_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
fi

if [[ "${MODEL_PATH,,}" == *gpt2* ]]; then
  MODEL_TYPE="gpt2"
elif [[ "${MODEL_PATH,,}" == *tinyllama* ]]; then
  MODEL_TYPE="tinyllama"
elif [[ "${MODEL_PATH,,}" == *mistral* ]]; then
  MODEL_TYPE="mistral"
elif [[ "${MODEL_PATH,,}" == *opt* ]]; then
  MODEL_TYPE="opt"
else
  MODEL_TYPE="gpt2"
fi

TASK="eval_main"
DATA_NAME=${6}
DATA_DIR="${BASE_PATH}/data/${DATA_NAME}"
DATA_NUM=${9:--1}

EVAL_BATCH_SIZE=${7}
SEED=${8}
SAVE_PATH=$(dirname "${CKPT_PATH}")

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --peft-path ${CKPT_PATH}"
OPTS+=" --peft lora"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type ${MODEL_TYPE}"
# task
OPTS+=" --task ${TASK}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num ${DATA_NUM}"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/evaluate.py ${OPTS}"
echo "${CMD}"

${CMD}
