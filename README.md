# IMPACT: Importance-Aware Prefix Alignment for Cross-Tokenizer Distillation

Hướng dẫn chạy repo **ICARE** trên máy/GPU bất kỳ (không hard-code path người dùng). Copy repo về ví dụ `~/ICARE` rồi làm lần lượt các bước dưới đây.

---

## Bước 1. Cài đặt môi trường

```bash
cd /đường/dẫn/tới/ICARE
bash install_icare.sh
conda activate icare
python -m spacy download en_core_web_sm
```

## Bước 2. Tải model và lưu ở các thư mục con (có tên = tên model) trong thư mục model_hub

- TinyLlama: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

- OPT-2.7B: facebook/opt-2.7b

- Qwen2.5-7B-Instruct: Qwen/Qwen2.5-7B-Instruct

- Mistral-7B: mistralai/Mistral-7B-v0.1

Ví dụ tải model:

export HUGGINGFACE_TOKEN=DIEN_TOKEN_O_DAY

huggingface-cli login --token $HUGGINGFACE_TOKEN

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /mnt/hoangtv/ICARE/model_hub/Qwen2.5-7B-Instruct --repo-type model --resume-download


## Bước 3. Tải checkpoint LoRA và lưu ở các thư mục con (có tên = tên model) trong thư mục lora_path

- Qwen2.5-7B-Instruct: HoangTran223/MCW_KD_Teacher_Qwen2.5-7B-Instruct

- Mistral-7B: HoangTran223/MCW_KD_Teacher_Mistral7B


## Bước 4. Chạy các config sau: chỉnh sửa lại GPUS=... (e.g, GPUS=(0), GPUS=(2),...)

Đầu tiên cd thư mục có code (e.g ICARE)

- Máy 1: bash scripts/anhtue/may1/run_all.sh

- Máy 2: bash scripts/anhtue/may2/run_all.sh

- Máy 3: bash scripts/anhtue/may3/run_all.sh
- Máy 4: bash scripts/anhtue/may4/run_all.sh
