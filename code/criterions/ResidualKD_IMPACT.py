"""ResidualKD + IMPACT: total_loss = ResidualKD_loss + lambda_impact * L_BPIA.

IMPACT is only added during Stage 2 (when projectors are available).
"""

import torch
from .ResidualKD import ResidualKD
from impact_utils import IMPACTModule, compute_text_offsets


class ResidualKD_IMPACT(ResidualKD):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        self.device = next(distiller.student_model.parameters()).device
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        labels = output_data["label"]
        teacher_key = distiller.teacher_model_type

        student_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        logits = student_outputs.logits
        h_S = student_outputs.hidden_states[-1]

        log = {}
        sft_loss, nll_loss = self.compute_cross_entropy_loss(logits, labels)

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"teacher_{teacher_key}_position_ids", None),
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            h_T = teacher_outputs.hidden_states[-1]

        has_projectors = (
            hasattr(distiller, "projector_TA")
            and hasattr(distiller, "projector_SA")
            and hasattr(distiller, "projector_AS")
        )

        if has_projectors:
            from residualkd_utils import (
                cross_model_attention,
                compute_residual_mask_same_tokenizer,
                compute_residual_mask_cross_tokenizer,
                compute_beta,
            )

            with torch.no_grad():
                h_T_A = distiller.projector_TA.encode(h_T)
            h_S_A = distiller.projector_SA(h_S)
            h_T_aligned, A_align = cross_model_attention(h_S_A, h_T_A)
            proj_to_S = distiller.projector_AS(h_T_aligned)

            same_tokenizer = not hasattr(distiller, "cross_tokenizer") or not distiller.cross_tokenizer
            response_mask = labels.ne(-100).float()

            if same_tokenizer:
                teacher_labels = output_data.get(f"teacher_{teacher_key}_label", labels)
                mask = compute_residual_mask_same_tokenizer(teacher_logits, teacher_labels, response_mask)
            else:
                mask = compute_residual_mask_cross_tokenizer(teacher_logits, A_align, response_mask)

            d_S = h_S.size(-1)
            d_A = h_S_A.size(-1)
            beta = compute_beta(h_S, proj_to_S, response_mask, d_S, d_A)

            mask_expanded = mask.unsqueeze(-1).float()
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
            residualkd_loss = (1.0 - lam) * sft_loss + lam * res_loss

            log["kd_loss"] = res_loss
            log["nll_loss"] = nll_loss
            log["residualkd_beta"] = beta
            log["residualkd_mask_ratio"] = mask.float().sum() / response_mask.sum().clamp(min=1)
        else:
            residualkd_loss = sft_loss
            log["kd_loss"] = torch.tensor(0.0, device=self.device)
            log["nll_loss"] = nll_loss

        s_offsets, t_offsets = self._get_offsets(input_data, distiller)
        impact_loss = self.impact.compute_loss(
            distiller,
            list(student_outputs.hidden_states),
            list(teacher_outputs.hidden_states),
            s_offsets, t_offsets,
            input_data["attention_mask"],
            input_data[f"teacher_{teacher_key}_attention_mask"],
        )

        loss = residualkd_loss + self.lambda_impact * impact_loss
        log["impact_loss"] = impact_loss
        log["loss"] = loss
        log["accuracy"] = self.compute_token_accuracy(logits, labels)

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        self.global_step += 1
        return loss / batch_denom, logging_output

    def _get_offsets(self, input_data, distiller):
        raw_texts = input_data.get("raw_texts", [])
        teacher_key = distiller.teacher_model_type
        teacher_tok = distiller.teacher_tokenizers.get(teacher_key)
        student_tok = distiller.student_tokenizer
        max_len = self.args.max_length

        s_offsets, t_offsets = [], []
        for text in raw_texts:
            if text:
                s_offsets.append(compute_text_offsets(student_tok, text, max_len))
                t_offsets.append(compute_text_offsets(teacher_tok, text, max_len))
            else:
                s_offsets.append([])
                t_offsets.append([])
        return s_offsets, t_offsets
