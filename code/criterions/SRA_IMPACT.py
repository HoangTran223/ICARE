"""SRA + IMPACT: total_loss = SRA_loss + lambda_impact * L_BPIA.

Both the SRA span alignment (LCS-based) and the IMPACT boundary-
interpolated prefix increment alignment are applied simultaneously.
"""

import torch
from .SRA import SRA
from sra_span_utils import get_span_hidden_states
from impact_utils import IMPACTModule, compute_text_offsets


class SRA_IMPACT(SRA):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_key = distiller.teacher_model_type

        student_out = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
            output_attentions=True,
        )

        with torch.no_grad():
            teacher_model.eval()
            teacher_out = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"teacher_{teacher_key}_position_ids", None),
                output_hidden_states=True,
                output_attentions=True,
            )

        log = {}

        hard_loss, nll_loss = self.compute_cross_entropy_loss(
            student_out.logits, output_data["label"], log=log
        )
        log["nll_loss"] = nll_loss

        stu_mask = input_data["attention_mask"]
        tea_mask = input_data[f"teacher_{teacher_key}_attention_mask"]

        # ---- LCS-based span alignment ----
        s_safe_idx = input_data.get("student_pooler_safe_idx")
        s_pooler_mask = input_data.get("student_pooler_mask")
        t_safe_idx = input_data.get(f"teacher_{teacher_key}_pooler_safe_idx")
        t_pooler_mask = input_data.get(f"teacher_{teacher_key}_pooler_mask")

        has_spans = all(
            x is not None
            for x in [s_safe_idx, s_pooler_mask, t_safe_idx, t_pooler_mask]
        )

        span_loss = torch.tensor(0.0, device=student_out.logits.device)
        s_span_embeds_raw = None
        t_span_embeds_raw = None

        if (
            has_spans
            and self.use_span_loss
            and hasattr(distiller, "projectors")
            and "sra_proj" in distiller.projectors
        ):
            s_span_list, s_span_weights = get_span_hidden_states(
                student_out.hidden_states, student_out.attentions,
                s_safe_idx, s_pooler_mask, stu_mask,
                self.student_layers, is_causal=True,
            )
            t_span_list, t_span_weights = get_span_hidden_states(
                teacher_out.hidden_states, teacher_out.attentions,
                t_safe_idx, t_pooler_mask, tea_mask,
                self.teacher_layers, is_causal=True,
            )

            s_span_embeds_raw = s_span_list[-1]
            t_span_embeds_raw = t_span_list[-1]

            for i in range(self.num_layer_pairs):
                s_proj = distiller.projectors["sra_proj"](s_span_list[i])
                layer_loss = self._cosine_span_loss(
                    s_proj, t_span_list[i], t_span_weights[i],
                )
                span_loss = span_loss + self.layer_weights[i] * layer_loss

        log["span_loss"] = span_loss

        geom_loss = self._geometric_span_loss(
            s_span_embeds_raw, t_span_embeds_raw,
            t_span_weights[-1] if has_spans and self.use_span_loss else None,
            student_out.hidden_states[-1],
            teacher_out.hidden_states[-1],
            stu_mask, tea_mask,
        )
        log["geom_loss"] = geom_loss

        logit_kd_loss = self._logit_distillation_loss(
            student_out, teacher_out,
            output_data["label"],
            output_data.get(f"teacher_{teacher_key}_label", output_data["label"]),
            distiller, stu_mask, tea_mask,
            s_span_embeds_raw, t_span_embeds_raw,
        )
        log["logit_kd_loss"] = logit_kd_loss

        kd_loss = span_loss + self.geom_weight * geom_loss + logit_kd_loss
        sra_loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        # ---- IMPACT ----
        s_offsets, t_offsets = self._get_offsets(input_data, distiller)
        impact_loss = self.impact.compute_loss(
            distiller,
            list(student_out.hidden_states),
            list(teacher_out.hidden_states),
            s_offsets, t_offsets, stu_mask, tea_mask,
        )

        loss = sra_loss + self.lambda_impact * impact_loss
        log["kd_loss"] = kd_loss
        log["impact_loss"] = impact_loss
        log["loss"] = loss
        log["accuracy"] = self.compute_token_accuracy(student_out.logits, output_data["label"])

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
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
