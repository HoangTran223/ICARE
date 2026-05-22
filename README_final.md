# IMPACT: Importance-Aware Prefix Alignment for Cross-Tokenizer Distillation

## Environment

```bash
cd ICARE
bash install_icare.sh
conda activate icare
python -m spacy download en_core_web_sm   # ResidualKD + IMPACT only
```

Put data in `data/dolly/` (`train.jsonl`, `dev.jsonl`) and weights in `model_hub/<name>/`.

## Training (base method + IMPACT)

Set `GPUS=(...)` in the script, or `export ICARE_CUDA_DEVICES=0`.  
Example student family: `scripts/gpt2_120m/` (same script names under `gpt2_340m/`, `gptxl/`, `tinyllama/`, `opt/` where listed).

| Base + IMPACT | Script |
|---------------|--------|
| SRA + IMPACT | `scripts/gpt2_120m/SRA_IMPACT.sh` |
| DSKD + IMPACT | `scripts/gpt2_120m/DSKD_IMPACT.sh` |
| ALM + IMPACT | `scripts/gpt2_120m/ALM_IMPACT.sh` |
| ResidualKD + IMPACT | `scripts/gpt2_120m/ResidualKD_IMPACT_full.sh` |
| DSKDv2 (ETA) | `scripts/gpt2_120m/DSKDv2.sh` (also `gpt2_340m/`, `gptxl/`, `tinyllama/`, `opt/`) |
| DSKDv2 + IMPACT | `scripts/gpt2_120m/DSKDv2_IMPACT.sh` (same folders) |
| DSKDv2 + IMPACT (PPL layers) | `scripts/gpt2_340m/DSKDv2_IMPACT_PPL.sh` |
| DSKDv2 + IMPACT (ALP-KD 1-all) | `scripts/opt/DSKDv2_IMPACT_ALPKD.sh` |

```bash
bash scripts/gpt2_120m/SRA_IMPACT.sh
```

Checkpoints: `outputs/<CKPT_NAME>/<TASK>/.../`

## Evaluation

```bash
bash scripts/eval/run_eval.sh /path/to/checkpoint 8
bash scripts/eval/run_eval_lora.sh /path/to/lora_adapter 4
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
