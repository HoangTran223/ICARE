"""Standalone IMPACT criterion: L_LM + lambda_impact * L_BPIA."""

import torch
from .cross_entropy_loss import CrossEntropyLoss
from impact_utils import IMPACTModule, compute_text_offsets


class IMPACT(CrossEntropyLoss):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        self.device = next(distiller.student_model.parameters()).device
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_key = distiller.teacher_model_type

        student_out = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )

        with torch.no_grad():
            teacher_model.eval()
            teacher_out = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"teacher_{teacher_key}_position_ids", None),
                output_hidden_states=True,
            )

        log = {}
        lm_loss, nll_loss = self.compute_cross_entropy_loss(
            student_out.logits, output_data["label"]
        )
        log["nll_loss"] = nll_loss

        s_offsets, t_offsets = self._get_offsets(input_data, distiller)

        impact_loss = self.impact.compute_loss(
            distiller,
            list(student_out.hidden_states),
            list(teacher_out.hidden_states),
            s_offsets, t_offsets,
            input_data["attention_mask"],
            input_data[f"teacher_{teacher_key}_attention_mask"],
        )

        loss = lm_loss + self.lambda_impact * impact_loss
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
