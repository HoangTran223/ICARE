import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer
from sra_span_utils import compute_separate_offsets, prepare_pooler_v2


class DistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer], 
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()
        # log_rank(f"Num of data instances: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")

        if os.path.exists(path):
            with open(path) as f:
                raw_data = [json.loads(l) for l in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
            
            log_rank("Processing dataset for student model (and all teacher models)...")  
            seg = np.iinfo(np.int32).max * 2 + 1        
            for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                student_prompt_ids = self.student_tokenizer.encode(
                    data["prompt"], add_special_tokens=False
                )
                student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                student_response_ids = self.student_tokenizer.encode(
                    data["output"], add_special_tokens=False
                )
                student_response_ids = student_response_ids \
                                     + [self.student_tokenizer.eos_token_id]
                tokenized_data = {
                    "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    "raw_text": data["prompt"] + data["output"],
                }

                s_offsets, s_prompt_tok_len = compute_separate_offsets(
                    self.student_tokenizer,
                    data["prompt"], data["output"],
                    self.max_prompt_length,
                )
                if s_offsets is not None:
                    tokenized_data["student_offset_mapping"] = s_offsets
                    tokenized_data["student_prompt_token_len"] = s_prompt_tok_len
        
                for model_type in self.teacher_tokenizers:
                    if self.teacher_tokenizers[model_type] is None: continue
                        
                    teacher_prompt_ids = self.teacher_tokenizers[model_type].encode(
                        data["prompt"], add_special_tokens=False
                    )
                    teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                    teacher_response_ids = self.teacher_tokenizers[model_type].encode(
                        data["output"], add_special_tokens=False
                    )
                    teacher_response_ids = teacher_response_ids \
                                            + [self.teacher_tokenizers[model_type].eos_token_id]
                    tokenized_data[f"teacher_{model_type}_input_ids"] = \
                        teacher_prompt_ids + [seg] + teacher_response_ids

                    t_offsets, t_prompt_tok_len = compute_separate_offsets(
                        self.teacher_tokenizers[model_type],
                        data["prompt"], data["output"],
                        self.max_prompt_length,
                    )
                    if t_offsets is not None:
                        tokenized_data[f"teacher_{model_type}_offset_mapping"] = t_offsets
                        tokenized_data[f"teacher_{model_type}_prompt_token_len"] = t_prompt_tok_len

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_lm(
        self, i, samp, model_data, no_model_data, gen_data, 
        teacher_model_data, teacher_no_model_data,
        span_data=None,
    ):
        seg = np.iinfo(np.int32).max * 2 + 1
        input_ids = np.array(samp["student_input_ids"])
        source_len = np.where(input_ids == seg)[0][0]
        prompt = input_ids[:source_len]
        input_ids = np.concatenate(
            [input_ids[:source_len], input_ids[source_len+1:]], axis=0
        )
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        gen_data["attention_mask"][i][-len(prompt):] = 1.0

        if span_data is not None:
            s_raw = samp.get("student_offset_mapping")
            if s_raw is not None:
                s_offsets = s_raw[: input_len - 1]
                span_data["student"].append({
                    "offsets": torch.tensor(s_offsets, dtype=torch.long),
                    "prompt_len": min(
                        samp.get("student_prompt_token_len", source_len),
                        len(s_offsets),
                    ),
                    "max_valid": input_len - 1,
                })
            else:
                span_data["student"].append(None)

        for model_type in self.teacher_tokenizers:
            t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
            t_source_len = np.where(t_input_ids == seg)[0][0]
            t_input_ids = np.concatenate(
                [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
            )
            t_input_ids = t_input_ids[:self.max_length]
            t_input_len = len(t_input_ids)
            teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[:-1], dtype=torch.long)
            teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                    torch.arange(0, t_input_len-1, dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[1:], dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
            teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
            teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0

            if span_data is not None:
                t_raw = samp.get(f"teacher_{model_type}_offset_mapping")
                if t_raw is not None:
                    t_offsets = t_raw[: t_input_len - 1]
                    span_data["teacher"][model_type].append({
                        "offsets": torch.tensor(t_offsets, dtype=torch.long),
                        "prompt_len": min(
                            samp.get(f"teacher_{model_type}_prompt_token_len", t_source_len),
                            len(t_offsets),
                        ),
                        "max_valid": t_input_len - 1,
                    })
                else:
                    span_data["teacher"][model_type].append(None)

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        teacher_model_data = {
            model_type: {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.teacher_tokenizers[model_type].eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for model_type in self.teacher_tokenizers:
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"] = torch.zeros(
                    bs, max_length, dtype=torch.long
                )

        teacher_no_model_data = {
            model_type: {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        span_data = {
            "student": [],
            "teacher": {mt: [] for mt in self.teacher_tokenizers},
        }

        for i, samp in enumerate(samples):
            self._process_lm(
                i, samp, model_data, no_model_data, gen_data, 
                teacher_model_data, teacher_no_model_data,
                span_data=span_data,
            )

        for model_type in teacher_model_data:
            prefix = f"teacher_{model_type}_"
            for key in teacher_model_data[model_type]:
                model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                
            for key in teacher_no_model_data[model_type]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]

        model_data["raw_texts"] = [samp.get("raw_text", "") for samp in samples]

        has_student_spans = (
            span_data["student"]
            and all(d is not None for d in span_data["student"])
        )
        if has_student_spans:
            for model_type in self.teacher_tokenizers:
                t_list = span_data["teacher"].get(model_type, [])
                if not t_list or not all(d is not None for d in t_list):
                    continue

                s_starts = [d["prompt_len"] for d in span_data["student"]]
                s_offsets = [d["offsets"] for d in span_data["student"]]
                s_max_pos = [d["max_valid"] for d in span_data["student"]]
                t_starts = [d["prompt_len"] for d in t_list]
                t_offsets = [d["offsets"] for d in t_list]
                t_max_pos = [d["max_valid"] for d in t_list]

                s_pooler, t_pooler = prepare_pooler_v2(
                    s_starts, s_offsets,
                    t_starts, t_offsets,
                    student_max_pos=s_max_pos,
                    teacher_max_pos=t_max_pos,
                )

                model_data["student_pooler_safe_idx"] = s_pooler["safe_idx"]
                model_data["student_pooler_mask"] = s_pooler["mask"]
                prefix = f"teacher_{model_type}_"
                model_data[f"{prefix}pooler_safe_idx"] = t_pooler["safe_idx"]
                model_data[f"{prefix}pooler_mask"] = t_pooler["mask"]

        return model_data, no_model_data, gen_data
