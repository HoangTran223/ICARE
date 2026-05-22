"""DSKD + IMPACT: total_loss = L_DSKD + lambda_impact * L_BPIA.

L_DSKD = (1 - kd_rate) * CE + kd_rate * KD_CMA (DualSpaceKDWithCMA).
IMPACT adds boundary-interpolated prefix increment alignment on top.
"""

import torch

from .DSKD import DualSpaceKDWithCMA
from impact_utils import IMPACTModule, compute_text_offsets


class DualSpaceKDWithIMPACT(DualSpaceKDWithCMA):
    """Dual-space KD with cross-model attention + IMPACT plug-in."""

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)
        print("--------------Using DSKD + IMPACT ----------------")

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_key = distiller.teacher_model_type
        self.distiller = distiller

        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        logits = outputs.logits
        log = {}

        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(
                    f"teacher_{teacher_key}_position_ids", None
                ),
                output_hidden_states=True,
            )

        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        dskd_loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss

        stu_mask = input_data["attention_mask"]
        tea_mask = input_data[f"teacher_{teacher_key}_attention_mask"]
        s_offsets, t_offsets = self._get_offsets(input_data, distiller)
        impact_loss = self.impact.compute_loss(
            distiller,
            list(outputs.hidden_states),
            list(teacher_outputs.hidden_states),
            s_offsets,
            t_offsets,
            stu_mask,
            tea_mask,
        )

        loss = dskd_loss + self.lambda_impact * impact_loss
        log["nll_loss"] = ce_loss
        log["kd_loss"] = kd_loss
        log["impact_loss"] = impact_loss
        log["loss"] = loss

        log["accuracy"] = self.compute_token_accuracy(
            logits, output_data["label"]
        )

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
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
                s_offsets.append(
                    compute_text_offsets(student_tok, text, max_len)
                )
                t_offsets.append(
                    compute_text_offsets(teacher_tok, text, max_len)
                )
            else:
                s_offsets.append([])
                t_offsets.append([])
        return s_offsets, t_offsets
