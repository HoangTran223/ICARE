import sys
import os
import time
import json
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.optim import AdamW
import deepspeed
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)
from transformers.integrations import HfDeepSpeedConfig

from arguments import get_args
from distiller import Distiller
from utils import (
    initialize,
    get_optimizer,
    get_learning_rate_scheduler,
    print_rank,
    log_rank,
    all_gather,
)
from criterions import build_criterion
from rouge_metric import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENKIT_PATH = os.path.join(BASE_DIR, "ALM", "tokenkit-main")
if TOKENKIT_PATH not in sys.path:
    sys.path.insert(0, TOKENKIT_PATH)

from pathlib import Path

from scipy import sparse
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.align import get_unconstrained_alignments, get_unbiased_alignments
from tokenkit import utils as tokenkit_utils
from alm_multitask import uses_gradmag

torch.set_num_threads(4)


# Byteify specs: "{checkpoint_path}:source=<Family>" (tokenkit/model_kinds.py).
# The checkpoint path MUST be the same hub dir passed as --model-path / --teacher-model-path
# in scripts/ (e.g. model_hub/GPT2-340M-FT, model_hub/Qwen1.5-1.8B). Do not substitute
# a different HF repo id — vocab/merges/chat-template drift breaks ALM alignment.
BYTEIFY_SOURCE_BY_TYPE = {
    "gpt2": "GPT2",
    "gptj": "GPT2",
    "opt": "GPT2",
    "qwen": "Qwen2",  # Qwen1.5 / Qwen2 / Qwen2.5
    "llama": "Llama2",
    "llama2": "Llama2",
    "llama3": "Llama3",
    "mistral": "Mistral",
    "tinyllama": "TinyLlama",
    "minicpm": "Llama2",
}


def _infer_byteify_source(model_type: str, model_path: str) -> str:
    if model_type in BYTEIFY_SOURCE_BY_TYPE:
        return BYTEIFY_SOURCE_BY_TYPE[model_type]
    path_l = (model_path or "").lower()
    if "qwen3" in path_l:
        return "Qwen3"
    if "qwen" in path_l:
        return "Qwen2"
    if "llama-3" in path_l or "llama3" in path_l:
        return "Llama3"
    if "llama" in path_l:
        return "Llama2"
    if "mistral" in path_l:
        return "Mistral"
    if "tinyllama" in path_l:
        return "TinyLlama"
    if "gemma-3" in path_l or "gemma3" in path_l:
        return "Gemma3"
    if "gemma" in path_l:
        return "Gemma2"
    if "phi" in path_l:
        return "Phi3"
    if "gpt" in path_l:
        return "GPT2"
    raise ValueError(
        f"Cannot infer tokenkit byteify source for model_type={model_type!r}, "
        f"model_path={model_path!r}. Extend BYTEIFY_SOURCE_BY_TYPE in distillation_alm.py."
    )


def get_byteify_spec(model_type: str, model_path: str) -> str:
    source = _infer_byteify_source(model_type, model_path)
    return f"{model_path}:source={source}"


class ALMDataset(Dataset):
    """Wraps raw JSONL data for ALM training. Each item contains raw text fields."""

    def __init__(self, args, split):
        self.args = args
        self.split = split
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")
        with open(path) as f:
            self.raw_data = [json.loads(line) for line in f]
        self.answers = [
            x["output"] if isinstance(x["output"], list) else [x["output"]]
            for x in self.raw_data
        ]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.raw_data[index]


def _load_tokenizer_pair_data(pair_data_path):
    """Load bias matrices for unbiased ALM alignment (tokenkit TokenizerAlignerCollator)."""
    base = Path(pair_data_path)
    bias1_path = base / "bias1_matrix.npz"
    bias2_path = base / "bias2_matrix.npz"
    if not bias1_path.exists() or not bias2_path.exists():
        raise FileNotFoundError(
            f"Unbiased alignment requires bias1_matrix.npz and bias2_matrix.npz under {pair_data_path}"
        )
    return (
        sparse.load_npz(bias1_path).todok(),
        sparse.load_npz(bias2_path).todok(),
        None,
        None,
    )


class ALMCollator:
    """Collates raw data items into aligned batches for ALM training."""

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        byteify_student,
        byteify_teacher,
        space_mask_student,
        space_mask_teacher,
        args,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.byteify_student = byteify_student
        self.byteify_teacher = byteify_teacher
        self.space_mask_student = space_mask_student
        self.space_mask_teacher = space_mask_teacher
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.model_type = args.model_type
        self.teacher_model_type = args.teacher_model_type
        self.alm_alignment = getattr(args, "alm_alignment", "unconstrained")
        self.bias_threshold = args.alm_bias_threshold
        self.tokenizer_pair_data = None
        if self.alm_alignment == "unbiased":
            if not args.tokenizer_pair_data_path:
                raise ValueError(
                    "--alm-alignment unbiased requires --tokenizer-pair-data-path"
                )
            self.tokenizer_pair_data = _load_tokenizer_pair_data(args.tokenizer_pair_data_path)
            log_rank(
                f"ALM alignment: unbiased (bias threshold={self.bias_threshold}) "
                f"from {args.tokenizer_pair_data_path}"
            )
        else:
            log_rank("ALM alignment: unconstrained (tokenkit recommended default)")

    def __call__(self, samples):
        bs = len(samples)
        seg = np.iinfo(np.int32).max * 2 + 1

        stu_all_ids = []
        tea_all_ids = []
        stu_source_lens = []
        tea_source_lens = []
        prompts = []

        for samp in samples:
            s_prompt = self.student_tokenizer.encode(samp["prompt"], add_special_tokens=False)
            s_prompt = s_prompt[: self.max_prompt_length]
            s_response = self.student_tokenizer.encode(samp["output"], add_special_tokens=False)
            s_response = s_response + [self.student_tokenizer.eos_token_id]
            s_ids = s_prompt + s_response
            s_ids = s_ids[: self.max_length]

            t_prompt = self.teacher_tokenizer.encode(samp["prompt"], add_special_tokens=False)
            t_prompt = t_prompt[: self.max_prompt_length]
            t_response = self.teacher_tokenizer.encode(samp["output"], add_special_tokens=False)
            t_response = t_response + [self.teacher_tokenizer.eos_token_id]
            t_ids = t_prompt + t_response
            t_ids = t_ids[: self.max_length]

            stu_all_ids.append(s_ids)
            tea_all_ids.append(t_ids)
            stu_source_lens.append(len(s_prompt))
            tea_source_lens.append(len(t_prompt))
            prompts.append(s_prompt)

        max_stu_len = max(len(x) for x in stu_all_ids)
        max_tea_len = max(len(x) for x in tea_all_ids)
        shared_length = min(max_stu_len, max_tea_len)

        stu_input_ids = np.full((bs, max_stu_len), self.student_tokenizer.eos_token_id, dtype=np.int64)
        stu_attention_mask = np.zeros((bs, max_stu_len), dtype=np.int64)
        stu_labels = np.full((bs, max_stu_len), -100, dtype=np.int64)
        stu_loss_mask = np.zeros((bs, max_stu_len), dtype=np.float32)

        tea_input_ids = np.full((bs, max_tea_len), self.teacher_tokenizer.eos_token_id, dtype=np.int64)
        tea_attention_mask = np.zeros((bs, max_tea_len), dtype=np.int64)
        tea_labels = np.full((bs, max_tea_len), -100, dtype=np.int64)
        tea_loss_mask = np.zeros((bs, max_tea_len), dtype=np.float32)

        gen_input_ids = np.full(
            (bs, self.max_prompt_length), self.student_tokenizer.eos_token_id, dtype=np.int64
        )
        gen_attention_mask = np.zeros((bs, self.max_prompt_length), dtype=np.int64)

        for i in range(bs):
            s_len = len(stu_all_ids[i])
            s_src = stu_source_lens[i]
            stu_input_ids[i, :s_len - 1] = stu_all_ids[i][:-1]
            stu_attention_mask[i, :s_len - 1] = 1
            stu_labels[i, :s_len - 1] = stu_all_ids[i][1:]
            stu_labels[i, :s_src - 1] = -100
            stu_loss_mask[i, :s_len - 1] = 1.0
            stu_loss_mask[i, :s_src - 1] = 0.0

            t_len = len(tea_all_ids[i])
            t_src = tea_source_lens[i]
            tea_input_ids[i, :t_len - 1] = tea_all_ids[i][:-1]
            tea_attention_mask[i, :t_len - 1] = 1
            tea_labels[i, :t_len - 1] = tea_all_ids[i][1:]
            tea_labels[i, :t_src - 1] = -100
            tea_loss_mask[i, :t_len - 1] = 1.0
            tea_loss_mask[i, :t_src - 1] = 0.0

            p_len = len(prompts[i])
            gen_input_ids[i, -p_len:] = prompts[i]
            gen_attention_mask[i, -p_len:] = 1

        if self.alm_alignment == "unbiased":
            alignment_matrix_a, alignment_matrix_b = get_unbiased_alignments(
                input_ids_teacher=tea_input_ids.astype(np.int64),
                input_ids_student=stu_input_ids.astype(np.int64),
                attention_mask_teacher=tea_attention_mask.astype(np.int64),
                attention_mask_student=stu_attention_mask.astype(np.int64),
                tokenizer_teacher=self.byteify_teacher,
                tokenizer_student=self.byteify_student,
                pair_data=self.tokenizer_pair_data,
                bias_threshold=self.bias_threshold,
            )
        else:
            alignment_matrix_a, alignment_matrix_b = get_unconstrained_alignments(
                input_ids_teacher=tea_input_ids.astype(np.int64),
                input_ids_student=stu_input_ids.astype(np.int64),
                attention_mask_teacher=tea_attention_mask.astype(np.int64),
                attention_mask_student=stu_attention_mask.astype(np.int64),
                tokenizer_teacher=self.byteify_teacher,
                tokenizer_student=self.byteify_student,
            )

        alm_loss_mask_student = np.zeros((bs, max_stu_len), dtype=np.float32)
        alm_loss_mask_teacher = np.zeros((bs, max_tea_len), dtype=np.float32)
        for i in range(bs):
            s_src = stu_source_lens[i]
            s_len = len(stu_all_ids[i])
            alm_loss_mask_student[i, s_src:s_len] = 1.0
            t_src = tea_source_lens[i]
            t_len = len(tea_all_ids[i])
            alm_loss_mask_teacher[i, t_src:t_len] = 1.0

        teacher_prefix = f"teacher_{self.teacher_model_type}"

        model_data = {
            "input_ids": torch.from_numpy(stu_input_ids).long(),
            "attention_mask": torch.from_numpy(stu_attention_mask).float(),
            f"{teacher_prefix}_input_ids": torch.from_numpy(tea_input_ids).long(),
            f"{teacher_prefix}_attention_mask": torch.from_numpy(tea_attention_mask).float(),
            "alignment_matrix_a": torch.from_numpy(alignment_matrix_a),
            "alignment_matrix_b": torch.from_numpy(alignment_matrix_b),
            "alm_loss_mask_student": torch.from_numpy(alm_loss_mask_student),
            "alm_loss_mask_teacher": torch.from_numpy(alm_loss_mask_teacher),
            "space_mask_student": self.space_mask_student,
            "space_mask_teacher": self.space_mask_teacher,
            "raw_texts": [samp["prompt"] + samp["output"] for samp in samples],
        }

        if self.model_type in ["gpt2"]:
            pos_ids = np.zeros((bs, max_stu_len), dtype=np.int64)
            for i in range(bs):
                s_len = len(stu_all_ids[i])
                pos_ids[i, :s_len - 1] = np.arange(s_len - 1)
            model_data["position_ids"] = torch.from_numpy(pos_ids).long()

        if self.teacher_model_type in ["gpt2"]:
            t_pos_ids = np.zeros((bs, max_tea_len), dtype=np.int64)
            for i in range(bs):
                t_len = len(tea_all_ids[i])
                t_pos_ids[i, :t_len - 1] = np.arange(t_len - 1)
            model_data[f"{teacher_prefix}_position_ids"] = torch.from_numpy(t_pos_ids).long()

        output_data = {
            "label": torch.from_numpy(stu_labels).long(),
            "loss_mask": torch.from_numpy(stu_loss_mask),
            f"{teacher_prefix}_label": torch.from_numpy(tea_labels).long(),
        }

        gen_data = {
            "input_ids": torch.from_numpy(gen_input_ids).long(),
            "attention_mask": torch.from_numpy(gen_attention_mask).long(),
        }

        return model_data, output_data, gen_data


def build_space_mask(byteify_tokenizer, vocab_size, mode="space+tab+newline+special"):
    mask = tokenkit_utils.get_space_mask(byteify_tokenizer, mode)
    mask = torch.from_numpy(mask).float()
    if mask.shape[0] < vocab_size:
        mask = F.pad(mask, (0, vocab_size - mask.shape[0]), value=0)
    return mask


def prepare_dataset(args, distiller):
    data = {}
    student_spec = get_byteify_spec(args.model_type, args.model_path)
    teacher_spec = get_byteify_spec(args.teacher_model_type, args.teacher_model_path)

    log_rank(f"Loading byteify tokenizers: student={student_spec}, teacher={teacher_spec}")
    byteify_student = load_byteify_tokenizer(student_spec)
    byteify_teacher = load_byteify_tokenizer(teacher_spec)

    student_vocab_size = distiller.student_model.config.vocab_size
    teacher_vocab_size = distiller.teacher_model.config.vocab_size if distiller.teacher_model else 32000

    space_mask_student = build_space_mask(byteify_student, student_vocab_size)
    space_mask_teacher = build_space_mask(byteify_teacher, teacher_vocab_size)

    collator = ALMCollator(
        student_tokenizer=distiller.student_tokenizer,
        teacher_tokenizer=distiller.teacher_tokenizers.get(args.teacher_model_type),
        byteify_student=byteify_student,
        byteify_teacher=byteify_teacher,
        space_mask_student=space_mask_student,
        space_mask_teacher=space_mask_teacher,
        args=args,
    )

    if args.do_train:
        data["train"] = ALMDataset(args, "train")
        log_rank("Num of train data: {}".format(len(data["train"])))
        data["dev"] = ALMDataset(args, "dev")
        log_rank("Num of dev data: {}".format(len(data["dev"])))
        if os.path.exists(os.path.join(args.data_dir, "test.jsonl")):
            data["test"] = ALMDataset(args, "test")
            log_rank("Num of test data: {}".format(len(data["test"])))
    elif args.do_eval:
        data["test"] = ALMDataset(args, "test")
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("do_train or do_eval must be set")

    return data, collator


def move_to_device(data_dicts, device):
    for data in data_dicts:
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device)
            elif isinstance(data[k], dict):
                for kk in data[k]:
                    if isinstance(data[k][kk], torch.Tensor):
                        data[k][kk] = data[k][kk].to(device)


def finetune(
    args,
    tokenizer,
    model,
    optimizer,
    lr_scheduler,
    dataset,
    collator,
    device,
):
    agg = getattr(args, "multitask_aggregation_fn", "none")
    log_rank(
        f"Start Fine-tuning (ALM) | alignment={getattr(args, 'alm_alignment', 'unconstrained')} "
        f"| multitask={agg} | alm_mode={args.alm_mode}"
    )
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"],
        shuffle=True,
        drop_last=True,
        rank=dp_rank,
        num_replicas=dp_world_size,
    )
    train_dataloader = DataLoader(
        dataset["train"],
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    step = 0
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [],
        "nll_loss": [],
        "sft_loss": [],
        "alm_loss": [],
        "impact_loss": [],
        "accuracy": [],
        "micro_step_time": [],
        "step_time": [],
    }
    if uses_gradmag(getattr(args, "multitask_aggregation_fn", None)):
        logging_output.update({
            "gradmag_weight_sft": [],
            "gradmag_weight_alm": [],
            "gradmag_norm_sft": [],
            "gradmag_norm_alm": [],
        })
        if args.criterion == "ALM_IMPACT":
            logging_output["gradmag_weight_impact"] = []
            logging_output["gradmag_norm_impact"] = []
    model_list = []

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_nll_loss, epoch_alm_loss = 0.0, 0.0, 0.0
        train_iter = iter(train_dataloader)

        while True:
            global_batch = []
            global_st_time = time.time()
            for i in range(args.gradient_accumulation_steps):
                try:
                    (input_batch, output_batch, _) = next(train_iter)
                    move_to_device([input_batch, output_batch], device)
                    global_batch.append({
                        "input_batch": input_batch,
                        "output_batch": output_batch,
                    })
                except StopIteration:
                    end_epoch = True
                    break

            if end_epoch:
                break

            global_token_num = sum(
                batch["output_batch"]["label"].ne(-100).sum() for batch in global_batch
            )
            dist.all_reduce(global_token_num, dist.ReduceOp.SUM, group=dp_group)
            loss_denom = global_token_num / (args.gradient_accumulation_steps * dp_world_size)

            for batch in global_batch:
                st_time = time.time()
                loss, logging_output = model(criterion, batch, logging_output, loss_denom)
                model.backward(loss)
                model.step()

                torch.cuda.synchronize()
                elapsed_time = time.time() - st_time
                logging_output["micro_step_time"].append(elapsed_time)
                step += 1

            logging_output["global_step"] += 1
            logging_output["step_time"].append(time.time() - global_st_time)
            epoch_step += 1

            def get_log(logging_output):
                logging_info = ""
                for key in logging_output:
                    if key == "epoch":
                        continue
                    log_val = logging_output[key]
                    if isinstance(log_val, list) and len(log_val) > 0:
                        avg = sum(log_val) / len(log_val)
                        if key.startswith("gradmag_norm"):
                            fmt = ".3e"
                        elif key.startswith("gradmag_weight"):
                            fmt = ".4f"
                        elif key.startswith("gradmag_"):
                            fmt = ".4f"
                        elif key in ("alm_loss", "sft_loss", "impact_loss"):
                            fmt = ".6f"
                        else:
                            fmt = ".4f"
                        logging_info += f"{key}={avg:{fmt}}, "
                    elif isinstance(log_val, int):
                        logging_info += f"{key}={log_val}, "
                    elif "lr" in key:
                        logging_info += f"{key}={log_val:.4e}, "

                log_rank(
                    "train | epoch {:0>3d}:   {:5d} / {:5d}  {}scale={:.4f}".format(
                        epoch + 1,
                        epoch_step,
                        args.train_iters_per_epoch,
                        logging_info,
                        optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    )
                )

            if logging_output["global_step"] % args.log_interval == 0:
                logging_output["lr"] = lr_scheduler.get_last_lr()[0]
                get_log(logging_output)
                epoch_loss += sum(logging_output["loss"])
                epoch_nll_loss += sum(logging_output["nll_loss"])
                if logging_output.get("alm_loss"):
                    epoch_alm_loss += sum(logging_output["alm_loss"])
                for key in logging_output:
                    if isinstance(logging_output[key], list):
                        logging_output[key] = []

        log_rank("End of epoch {}".format(epoch + 1))
        denom = max(epoch_step * args.gradient_accumulation_steps, 1)
        log_rank(
            "train | epoch {:0>3d} | loss {:.4f} | nll_loss {:.4f} | alm_loss {:.6f}".format(
                epoch + 1,
                epoch_loss / denom,
                epoch_nll_loss / denom,
                epoch_alm_loss / denom,
            )
        )

        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            if (epoch + 1) % args.eval_interval == 0:
                log_rank("Evaluating before saving model...")
                eval_loss, eval_results = evaluate(
                    args, tokenizer, model.module.student_model, dataset["dev"], "dev", device
                )
                if "test" in dataset:
                    _, _ = evaluate(
                        args, tokenizer, model.module.student_model,
                        dataset["test"], "test", device, repeat_times=1,
                    )

                if args.eval_gen:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}_rougel{:.4f}".format(
                        epoch + 1, logging_output["global_step"], eval_loss, eval_results["rougeL"]
                    )
                else:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}".format(
                        epoch + 1, logging_output["global_step"], eval_loss
                    )
                save_dir_path = os.path.join(args.save_dir, ckpt_name)

                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(
                            save_dir_path, safe_serialization=False
                        )
                    if hasattr(model.module, "projectors"):
                        log_rank("Saving projector...")
                        torch.save(
                            model.module.projectors.state_dict(),
                            os.path.join(save_dir_path, "projector.pt"),
                        )
                    if args.eval_gen:
                        model_list.append({"path": save_dir_path, "score": eval_results["rougeL"]})
                        model_list = sorted(model_list, key=lambda x: x["score"])
                    else:
                        model_list.append({"path": save_dir_path, "score": eval_loss})
                        model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)

                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])

                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()
            else:
                ckpt_name = "epoch{}_step{}".format(epoch + 1, logging_output["global_step"])
                save_dir_path = os.path.join(args.save_dir, ckpt_name)

                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(
                            save_dir_path, safe_serialization=False
                        )
                    model_list.append({"path": save_dir_path, "score": logging_output["global_step"]})
                    model_list = sorted(model_list, key=lambda x: x["score"])
                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])
                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()

    total_seconds = time.time() - start_time
    log_rank(
        "Done training in {:0>2}:{:0>2}:{:0>2}".format(
            int(total_seconds // 3600),
            int(total_seconds % 3600 // 60),
            int(total_seconds % 60),
        )
    )
    if args.save_dir and dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "training_time.txt"), "w") as f:
            f.write("Training time (seconds): {:.2f}\n".format(total_seconds))
            f.write(
                "Training time (hh:mm:ss): {:0>2}:{:0>2}:{:0>2}\n".format(
                    int(total_seconds // 3600),
                    int(total_seconds % 3600 // 60),
                    int(total_seconds % 60),
                )
            )


@torch.no_grad()
def evaluate(args, tokenizer, model, dataset, split, device, repeat_times=None):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(reduction="none")

    log_rank("Evaluating on {} set with {} GPU(s)".format(split, dp_world_size))

    if args.do_sample:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            no_repeat_ngram_size=args.no_repeat_ngram_size if split != "dev" else 0,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        repeat_times = args.eval_gen_repeat_times if repeat_times is None else repeat_times
    else:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            no_repeat_ngram_size=args.no_repeat_ngram_size if split != "dev" else 0,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        repeat_times = 1

    eval_collator = EvalCollator(tokenizer, args)
    sampler = DistributedSampler(
        dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
    )

    model.eval()
    eval_info = {"loss": 0.0, "token_num": 0, "token_acc": 0.0, "top1_prob": 0.0}
    all_response_ids = [[] for _ in range(repeat_times)]

    for input_batch, output_batch, gen_data in dataloader:
        move_to_device([input_batch, output_batch, gen_data], device)
        logits = model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None),
        ).logits
        loss = loss_func(logits.view(-1, logits.shape[-1]), output_batch["label"].view(-1))
        pad_mask = output_batch["label"].ne(-100)
        token_num = pad_mask.sum()
        loss = loss.view_as(output_batch["label"]).masked_fill_(~pad_mask, 0.0).sum()
        token_acc_num = logits.argmax(-1).eq(output_batch["label"]).float()
        token_acc_num = token_acc_num.masked_fill_(~pad_mask, 0.0).sum()
        probs = logits.softmax(-1)
        top1_prob = probs.max(-1)[0].masked_fill(~pad_mask, 0.0).sum()

        dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(token_num, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(token_acc_num, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(top1_prob, dist.ReduceOp.SUM, group=dp_group)

        eval_info["loss"] += loss.item()
        eval_info["token_num"] += token_num.item()
        eval_info["token_acc"] += token_acc_num.item()
        eval_info["top1_prob"] += top1_prob.item()

    eval_info["loss"] /= eval_info["token_num"]
    eval_info["token_acc"] /= eval_info["token_num"]
    eval_info["top1_prob"] /= eval_info["token_num"]
    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)

    eval_res = {}
    if args.eval_gen:
        for i in range(repeat_times):
            for input_batch, output_batch, gen_data in tqdm(
                dataloader,
                desc=f"{i + 1}-th evaluation: ",
                disable=(dp_rank != 0 or not args.eval_tqdm),
            ):
                move_to_device([gen_data], device)
                max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
                try:
                    gen_out = model.generate(
                        **gen_data, generation_config=generation_config, max_new_tokens=max_new_tokens
                    )
                except Exception:
                    model = model.float()
                    gen_out = model.generate(
                        **gen_data, generation_config=generation_config, max_new_tokens=max_new_tokens
                    )
                    model = model.half()

                full_ids = gen_out.sequences
                full_ids = F.pad(full_ids, (0, args.max_length - full_ids.shape[1]), value=tokenizer.pad_token_id)
                response_ids = full_ids[:, gen_data["input_ids"].size(1) :]
                all_response_ids[i].append(response_ids)

            all_response_ids[i] = torch.cat(all_response_ids[i], dim=0)
            all_response_ids[i] = all_gather(
                all_response_ids[i], dim=1, world_size=dp_world_size, group=dp_group, op="stack"
            )
            all_response_ids[i] = all_response_ids[i].view(-1, all_response_ids[i].size(-1))
            responses = tokenizer.batch_decode(all_response_ids[i], skip_special_tokens=True)
            references = dataset.answers
            responses = responses[: len(references)]
            res = compute_metrics(responses, references)
            log_rank("eval_results in run@{}: {}".format(i + 1, res))

            for key in res:
                if key in eval_res:
                    eval_res[key].append(res[key])
                else:
                    eval_res[key] = [res[key]]

        for key in eval_res:
            eval_res[key] = round(sum(eval_res[key]) / len(eval_res[key]), 4)

    log_str = f"{split} | {eval_info} | {eval_res}"
    log_rank(log_str)
    model.train()
    return eval_info["loss"], eval_res


class EvalCollator:
    """Simple collator for evaluation that only uses the student tokenizer."""

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.model_type = args.model_type

    def __call__(self, samples):
        bs = len(samples)
        model_data = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_length),
        }
        if self.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, self.max_length, dtype=torch.long)

        no_model_data = {
            "label": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, self.max_length),
        }
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, samp in enumerate(samples):
            prompt_ids = self.tokenizer.encode(samp["prompt"], add_special_tokens=False)
            prompt_ids = prompt_ids[: self.max_prompt_length]
            response_ids = self.tokenizer.encode(samp["output"], add_special_tokens=False)
            response_ids = response_ids + [self.tokenizer.eos_token_id]
            input_ids = prompt_ids + response_ids
            input_ids = input_ids[: self.max_length]
            source_len = len(prompt_ids)
            input_len = len(input_ids)

            model_data["input_ids"][i][: input_len - 1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            model_data["attention_mask"][i][: input_len - 1] = 1.0
            if self.model_type in ["gpt2"]:
                model_data["position_ids"][i][: input_len - 1] = torch.arange(0, input_len - 1, dtype=torch.long)
            no_model_data["label"][i][: input_len - 1] = torch.tensor(input_ids[1:], dtype=torch.long)
            no_model_data["label"][i][: source_len - 1] = -100
            no_model_data["loss_mask"][i][: input_len - 1] = 1.0
            no_model_data["loss_mask"][i][: source_len - 1] = 0

            gen_data["input_ids"][i][-len(prompt_ids) :] = torch.tensor(prompt_ids, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt_ids) :] = 1

        return model_data, no_model_data, gen_data


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)

    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "=" * 30 + f" ALM EXP at {cur_time} " + "=" * 30)

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    ds_config["offload_param_device"] = None
    ds_config["zero3_init_flag"] = False

    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"]
    if "bf16" in ds_config:
        args.fp32 = not ds_config["bf16"]["enabled"]
    log_rank(args)

    args.deepspeed_config = None
    log_rank("Initializing a distiller for ALM knowledge distillation...")
    distiller = Distiller(args, device)

    dataset, collator = prepare_dataset(args, distiller)
    dp_world_size = dist.get_world_size()

    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"])
            / (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        log_rank("Train iters per epoch = {}".format(args.train_iters_per_epoch))

        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        log_rank("Total_iters = {}".format(args.total_iters))

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    optimizer = get_optimizer(args, distiller.student_model)
    optimizer = distiller.add_optimizer_param_group(optimizer)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config,
    )

    if args.load is not None:
        log_rank(f"[INFO] Loading checkpoint from {args.load}")
        distiller.student_model = distiller.student_model.from_pretrained(args.load).to(device)
        distiller.student_tokenizer = distiller.student_tokenizer.from_pretrained(args.load)

    if args.do_train:
        finetune(
            args,
            distiller.student_tokenizer,
            model,
            optimizer,
            lr_scheduler,
            dataset,
            collator,
            device,
        )

    if args.do_eval:
        evaluate(
            args,
            distiller.student_tokenizer,
            model,
            dataset["test"],
            "test",
            device,
        )


if __name__ == "__main__":
    main()
