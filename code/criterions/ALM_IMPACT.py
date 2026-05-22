"""ALM + IMPACT with optional GradMag over [SFT, ALM, IMPACT] tasks."""

import torch
from .ALM import ALM
from impact_utils import IMPACTModule, compute_text_offsets

from alm_multitask import (
    approximate_gradmag_weights,
    aggregate_multitask_loss,
    get_last_layer_params,
    uses_gradmag,
)


class ALM_IMPACT(ALM):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        losses, log = self.compute_multitask_losses(distiller, input_data, output_data)

        if self._use_gradmag():
            last_params = get_last_layer_params(
                distiller.student_model, distiller.student_model_type
            )
            weights, grad_norms = approximate_gradmag_weights(
                losses,
                last_params,
                self.multitask_aggregation_fn,
            )
            loss = aggregate_multitask_loss(losses, weights)
            log["gradmag_weight_sft"] = weights[0].detach()
            log["gradmag_weight_alm"] = weights[1].detach()
            log["gradmag_weight_impact"] = weights[2].detach()
            log["gradmag_norm_sft"] = grad_norms[0].detach()
            log["gradmag_norm_alm"] = grad_norms[1].detach()
            log["gradmag_norm_impact"] = grad_norms[2].detach()
        else:
            loss = losses[0] + self.alm_loss_weight * losses[1] + self.lambda_impact * losses[2]

        log["loss"] = loss.detach() if isinstance(loss, torch.Tensor) else loss
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def compute_multitask_losses(self, distiller, input_data, output_data):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        device = next(model.parameters()).device

        student_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        student_logits = student_outputs.logits

        log = {}
        sft_loss, nll_loss = self.compute_cross_entropy_loss(
            student_logits, output_data["label"], log=log
        )
        accuracy = self.compute_token_accuracy(student_logits, output_data["label"])

        teacher_key = f"teacher_{distiller.teacher_model_type}"
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"{teacher_key}_input_ids"],
                attention_mask=input_data[f"{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"{teacher_key}_position_ids", None),
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits

        alignment_matrix_a = input_data["alignment_matrix_a"].to(device)
        alignment_matrix_b = input_data["alignment_matrix_b"].to(device)
        loss_mask_student = input_data["alm_loss_mask_student"].to(device)
        loss_mask_teacher = input_data["alm_loss_mask_teacher"].to(device)
        space_mask_student = input_data["space_mask_student"].to(device)
        space_mask_teacher = input_data["space_mask_teacher"].to(device)

        alm_loss = self.compute_alm_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            input_ids_student=input_data["input_ids"],
            input_ids_teacher=input_data[f"{teacher_key}_input_ids"],
            alignment_matrix_a=alignment_matrix_a,
            alignment_matrix_b=alignment_matrix_b,
            loss_mask_student=loss_mask_student,
            loss_mask_teacher=loss_mask_teacher,
            space_mask_student=space_mask_student,
            space_mask_teacher=space_mask_teacher,
        )

        s_offsets, t_offsets = self._get_offsets(input_data, distiller)
        impact_loss = self.impact.compute_loss(
            distiller,
            list(student_outputs.hidden_states),
            list(teacher_outputs.hidden_states),
            s_offsets, t_offsets,
            input_data["attention_mask"],
            input_data[f"{teacher_key}_attention_mask"],
        )

        log["sft_loss"] = sft_loss
        log["alm_loss"] = alm_loss
        log["impact_loss"] = impact_loss
        if not self._use_gradmag():
            log["loss"] = (
                sft_loss + self.alm_loss_weight * alm_loss + self.lambda_impact * impact_loss
            )
        else:
            log["loss"] = sft_loss
        log["nll_loss"] = nll_loss
        log["accuracy"] = accuracy

        return [sft_loss, alm_loss, impact_loss], log

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
