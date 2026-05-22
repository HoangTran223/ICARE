"""ResidualKD Stage 1: Projector Pretraining.

Trains bottleneck projectors P_T→A and P_A→T by reconstructing teacher
hidden states through the anchor bottleneck, then passing through the
frozen teacher LM head to compute CE loss against ground-truth labels.

Pipeline: h_T → P_T→A → P_A→T → h_recon → lm_head → CE(logits, labels)
Teacher model and LM head are frozen; only the projector trains.
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
import math
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from arguments import get_args
from data_utils.distill_datasets import DistillDataset
from residualkd_utils import ProjectorTA
from utils import (
    initialize,
    get_learning_rate_scheduler,
    print_rank,
    log_rank,
)

torch.set_num_threads(4)


class ProjectorPretrainWrapper(nn.Module):
    """Wraps teacher model + ProjectorTA for DeepSpeed initialization.
    Teacher and LM head are frozen; only projector parameters are trainable.
    """

    def __init__(self, teacher_model, projector, teacher_tokenizer, teacher_model_type):
        super().__init__()
        self.teacher_model = teacher_model
        self.projector = projector
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_model_type = teacher_model_type
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels, position_ids=None):
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            h_T = teacher_outputs.hidden_states[-1]  # [B, L, d_T]

        _, h_recon = self.projector(h_T)  # [B, L, d_T]

        lm_head = (
            self.teacher_model.lm_head
            if hasattr(self.teacher_model, "lm_head")
            else self.teacher_model.get_output_embeddings()
        )
        with torch.no_grad():
            lm_head_weight = lm_head.weight.detach()

        recon_logits = F.linear(h_recon, lm_head_weight)

        # Paper Eq. 3: CE on ALL tokens, not just response tokens.
        # Build shifted labels from input_ids; only padding is masked.
        all_labels = torch.full_like(input_ids, -100)
        all_labels[:, :-1] = input_ids[:, 1:]
        all_labels[attention_mask == 0] = -100

        loss = self.loss_fn(
            recon_logits.view(-1, recon_logits.size(-1)),
            all_labels.view(-1),
        )
        return loss, recon_logits


def prepare_dataset_teacher_only(args, tokenizer, teacher_model_type):
    """Prepare datasets using only the teacher tokenizer."""
    from data_utils.distill_datasets import DistillDataset

    teacher_tokenizers = {teacher_model_type: tokenizer}

    data = {}
    if args.do_train:
        data["train"] = DistillDataset(args, "train", tokenizer, teacher_tokenizers)
        log_rank(f"Num of train data: {len(data['train'])}")

        data["dev"] = DistillDataset(args, "dev", tokenizer, teacher_tokenizers)
        log_rank(f"Num of dev data: {len(data['dev'])}")
    return data


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)

    if dist.get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "=" * 30 + f" ResidualKD Pretrain @ {cur_time} " + "=" * 30)

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

    if args.model_dtype == "fp32":
        dtype = torch.float32
    elif args.model_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    log_rank("Loading teacher model for projector pretraining...")
    teacher_config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=teacher_config,
        device_map=None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    teacher_peft_path = getattr(args, "teacher_peft_path", None)
    if teacher_peft_path is not None:
        from peft import PeftModel
        log_rank(f"Loading teacher LoRA adapter from {teacher_peft_path}")
        teacher_model = PeftModel.from_pretrained(teacher_model, teacher_peft_path)
        teacher_model = teacher_model.merge_and_unload()

    for p in teacher_model.parameters():
        p.requires_grad = False

    if hasattr(teacher_config, "n_embed"):
        d_T = teacher_config.n_embed
    else:
        d_T = teacher_config.hidden_size

    teacher_model_type = args.teacher_model_type
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_path, trust_remote_code=True
    )
    if teacher_model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    elif teacher_model_type == "qwen":
        teacher_tokenizer.eos_token_id = 151643
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    d_A = getattr(args, "residualkd_d_bottleneck", 64)
    log_rank(f"Creating ProjectorTA: d_T={d_T}, d_A={d_A}")
    projector = ProjectorTA(d_T, d_A)

    wrapper = ProjectorPretrainWrapper(
        teacher_model, projector, teacher_tokenizer, teacher_model_type
    )

    dataset = prepare_dataset_teacher_only(args, teacher_tokenizer, teacher_model_type)

    dp_world_size = dist.get_world_size()

    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"])
            / (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        log_rank(f"Train iters per epoch = {args.train_iters_per_epoch}")

        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        log_rank(f"Total iters = {args.total_iters}")

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    proj_params = [p for p in projector.parameters() if p.requires_grad]
    optimizer = AdamW(proj_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    args.deepspeed_config = None
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=wrapper,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config,
    )

    log_rank("Starting projector pretraining...")
    pretrain_projector(args, model, dataset, device, projector)


def pretrain_projector(args, model, dataset, device, projector):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    sampler = DistributedSampler(
        dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size
    )
    train_dataloader = DataLoader(
        dataset["train"],
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["train"].collate,
    )

    best_eval_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        train_iter = iter(train_dataloader)
        end_epoch = False

        while True:
            global_batch = []
            for _ in range(args.gradient_accumulation_steps):
                try:
                    (input_batch, output_batch, _) = next(train_iter)
                    dataset["train"].move_to_device([input_batch, output_batch], device)
                    global_batch.append({
                        "input_batch": input_batch,
                        "output_batch": output_batch,
                    })
                except StopIteration:
                    end_epoch = True
                    break

            if end_epoch:
                break

            for batch in global_batch:
                inp = batch["input_batch"]
                out = batch["output_batch"]

                teacher_key = model.module.teacher_model_type
                t_input_ids_key = f"teacher_{teacher_key}_input_ids"
                t_attn_key = f"teacher_{teacher_key}_attention_mask"
                t_label_key = f"teacher_{teacher_key}_label"
                t_pos_key = f"teacher_{teacher_key}_position_ids"

                if t_input_ids_key in inp:
                    input_ids = inp[t_input_ids_key]
                    attention_mask = inp[t_attn_key]
                    labels = out.get(t_label_key, out["label"])
                    position_ids = inp.get(t_pos_key, None)
                else:
                    input_ids = inp["input_ids"]
                    attention_mask = inp["attention_mask"]
                    labels = out["label"]
                    position_ids = inp.get("position_ids", None)

                loss, _ = model(input_ids, attention_mask, labels, position_ids=position_ids)

                model.backward(loss)
                model.step()

                epoch_loss += loss.item()
                epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        log_rank(f"Pretrain epoch {epoch + 1}/{args.num_epochs} | loss = {avg_loss:.6f}")

        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            eval_loss = evaluate_projector(args, model, dataset, device)
            log_rank(f"Pretrain eval loss = {eval_loss:.6f}")

            if dist.get_rank() == 0:
                save_path = os.path.join(
                    args.save_dir,
                    f"projector_epoch{epoch + 1}_loss{eval_loss:.4f}",
                )
                os.makedirs(save_path, exist_ok=True)
                torch.save(projector.state_dict(), os.path.join(save_path, "projector_TA.pt"))
                log_rank(f"Saved projector to {save_path}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_path = os.path.join(args.save_dir, "projector_best.pt")
                    torch.save(projector.state_dict(), best_path)
                    log_rank(f"New best projector saved (loss={eval_loss:.6f})")

            dist.barrier()

    total_seconds = time.time() - start_time
    log_rank(
        "Projector pretraining done in {:0>2}:{:0>2}:{:0>2}".format(
            int(total_seconds // 3600),
            int(total_seconds % 3600 // 60),
            int(total_seconds % 60),
        )
    )

    if dist.get_rank() == 0:
        final_path = os.path.join(args.save_dir, "projector_final.pt")
        torch.save(projector.state_dict(), final_path)
        log_rank(f"Final projector saved to {final_path}")


@torch.no_grad()
def evaluate_projector(args, model, dataset, device):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    sampler = DistributedSampler(
        dataset["dev"], shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset["dev"],
        sampler=sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["dev"].collate,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_batch, output_batch, _ in dataloader:
        dataset["dev"].move_to_device([input_batch, output_batch], device)

        teacher_key = model.module.teacher_model_type
        t_input_ids_key = f"teacher_{teacher_key}_input_ids"
        t_attn_key = f"teacher_{teacher_key}_attention_mask"
        t_label_key = f"teacher_{teacher_key}_label"
        t_pos_key = f"teacher_{teacher_key}_position_ids"

        if t_input_ids_key in input_batch:
            input_ids = input_batch[t_input_ids_key]
            attention_mask = input_batch[t_attn_key]
            labels = output_batch.get(t_label_key, output_batch["label"])
            position_ids = input_batch.get(t_pos_key, None)
        else:
            input_ids = input_batch["input_ids"]
            attention_mask = input_batch["attention_mask"]
            labels = output_batch["label"]
            position_ids = input_batch.get("position_ids", None)

        loss, _ = model(input_ids, attention_mask, labels, position_ids=position_ids)

        n_tokens = labels.ne(-100).sum()
        dist.all_reduce(n_tokens, dist.ReduceOp.SUM, group=dp_group)
        loss_val = loss.item() * n_tokens.item()

        total_loss += loss_val
        total_tokens += n_tokens.item()

    model.train()
    return total_loss / max(total_tokens, 1)


if __name__ == "__main__":
    main()
