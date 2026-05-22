"""DSKDv2 + IMPACT: total_loss = DSKDv2_loss (ETA, cross-tokenizer) + lambda_impact * L_BPIA.

Criterion keys: dual_space_kd_v2_impact (preferred), dual_space_kd_v2_ipact (legacy alias).
"""

import torch
from .dual_space_kd_v2_with_exact_token_alignment import DualSpaceKDV2WithETA
from impact_utils import IMPACTModule, compute_text_offsets


class DualSpaceKDV2WithIMPACT(DualSpaceKDV2WithETA):
    """Cross-tokenizer DSKDv2 via ETA alignment + IMPACT hidden alignment."""

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, batch, logging_output):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_model.eval()

        self.distiller = distiller
        outputs = model(**batch["input_batch"], output_hidden_states=True)
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, batch["label_batch"]["label"], reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        log["nll_loss"] = ce_loss

        if "op_input_batch" in batch:
            outputs = model(**batch["op_input_batch"], output_hidden_states=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["op_teacher_input_batch"],
                    output_hidden_states=True,
                )
            kd_loss, log = self.compute_on_policy_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )
            student_hs = list(outputs.hidden_states)
            teacher_hs = list(teacher_outputs.hidden_states)
            student_mask = batch["op_input_batch"]["attention_mask"]
            teacher_mask = batch["op_teacher_input_batch"]["attention_mask"]
        else:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"],
                    output_hidden_states=True,
                )
            kd_loss, log = self.compute_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )
            student_hs = list(outputs.hidden_states)
            teacher_hs = list(teacher_outputs.hidden_states)
            student_mask = batch["input_batch"]["attention_mask"]
            teacher_mask = batch["teacher_input_batch"]["attention_mask"]

        dskdv2_loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss

        s_offsets, t_offsets = self._get_offsets(batch, distiller)
        teacher_labels = batch["teacher_label_batch"]["label"]
        impact_loss = self.impact.compute_loss(
            distiller,
            student_hs,
            teacher_hs,
            s_offsets,
            t_offsets,
            student_mask,
            teacher_mask,
            teacher_labels=teacher_labels,
        )

        loss = dskdv2_loss + self.lambda_impact * impact_loss
        log["kd_loss"] = kd_loss
        log["impact_loss"] = impact_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, batch["label_batch"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        return loss, logging_output

    def _get_offsets(self, batch, distiller):
        raw_texts = batch.get("raw_texts", [])
        student_tok = distiller.student_tokenizer
        teacher_tok = distiller.teacher_tokenizer
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
