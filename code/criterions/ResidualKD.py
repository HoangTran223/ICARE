import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .cross_entropy_loss import CrossEntropyLoss
from residualkd_utils import (
    cross_model_attention,
    compute_residual_mask_same_tokenizer,
    compute_residual_mask_cross_tokenizer,
    compute_beta,
)


class ResidualKD(CrossEntropyLoss):
    """ResidualKD criterion (On et al., ICLR 2026).

    Stage 2 criterion that applies residual correction to student hidden states
    at positions where the teacher is wrong, then computes CE on the corrected logits.

    Total loss: (1 - λ) · L_SFT + λ · L_res
    """

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.lambda_res = getattr(args, "residualkd_lambda_res", 0.5)
        self.lambda_warmup = getattr(args, "residualkd_lambda_warmup", 50)
        self.d_bottleneck = getattr(args, "residualkd_d_bottleneck", 64)
        self.global_step = 0

    def _get_lambda(self):
        """Linear warmup of the residual weight λ over the first N steps."""
        if self.lambda_warmup <= 0:
            return self.lambda_res
        progress = min(1.0, self.global_step / self.lambda_warmup)
        return self.lambda_res * progress

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        self.device = next(distiller.student_model.parameters()).device
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        labels = output_data["label"]

        student_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        logits = student_outputs.logits
        h_S = student_outputs.hidden_states[-1]  # [B, S_len, d_S]

        log = {}
        sft_loss, nll_loss = self.compute_cross_entropy_loss(logits, labels)

        with torch.no_grad():
            teacher_model.eval()
            teacher_key = distiller.teacher_model_type
            teacher_outputs = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"teacher_{teacher_key}_position_ids", None),
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            h_T = teacher_outputs.hidden_states[-1]  # [B, T_len, d_T]

        has_projectors = (
            hasattr(distiller, "projector_TA")
            and hasattr(distiller, "projector_SA")
            and hasattr(distiller, "projector_AS")
        )

        if has_projectors:
            with torch.no_grad():
                h_T_A = distiller.projector_TA.encode(h_T)  # [B, T_len, d_A]

            h_S_A = distiller.projector_SA(h_S)  # [B, S_len, d_A]

            h_T_aligned, A_align = cross_model_attention(h_S_A, h_T_A)  # [B, S_len, d_A]

            proj_to_S = distiller.projector_AS(h_T_aligned)  # [B, S_len, d_S]

            same_tokenizer = not hasattr(distiller, "cross_tokenizer") or not distiller.cross_tokenizer
            response_mask = labels.ne(-100).float()

            if same_tokenizer:
                teacher_labels = output_data.get(
                    f"teacher_{distiller.teacher_model_type}_label", labels
                )
                mask = compute_residual_mask_same_tokenizer(
                    teacher_logits, teacher_labels, response_mask
                )
            else:
                mask = compute_residual_mask_cross_tokenizer(
                    teacher_logits, A_align, response_mask
                )

            d_S = h_S.size(-1)
            d_A = h_S_A.size(-1)
            beta = compute_beta(h_S, proj_to_S, response_mask, d_S, d_A)

            mask_expanded = mask.unsqueeze(-1).float()  # [B, S_len, 1]
            h_S_res = h_S - beta * proj_to_S * mask_expanded

            lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
            if lm_head is None:
                res_logits = model(
                    inputs_embeds=h_S_res,
                    attention_mask=input_data["attention_mask"],
                ).logits
            else:
                head_dtype = lm_head.weight.dtype if getattr(lm_head, "weight", None) is not None else h_S.dtype
                res_logits = lm_head(h_S_res.to(head_dtype))

            res_loss, _ = self.compute_cross_entropy_loss(res_logits, labels)

            lam = self._get_lambda()
            loss = (1.0 - lam) * sft_loss + lam * res_loss

            log["kd_loss"] = res_loss
            log["nll_loss"] = nll_loss
            log["loss"] = loss
            log["residualkd_beta"] = beta
            log["residualkd_mask_ratio"] = mask.float().sum() / response_mask.sum().clamp(min=1)
        else:
            loss = sft_loss
            log["kd_loss"] = torch.tensor(0.0, device=self.device)
            log["nll_loss"] = nll_loss
            log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, labels)
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        self.global_step += 1
        return loss / batch_denom, logging_output
