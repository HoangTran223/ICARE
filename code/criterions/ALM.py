import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropyLoss

from alm_multitask import (
    approximate_gradmag_weights,
    aggregate_multitask_loss,
    get_last_layer_params,
    uses_gradmag,
)


def log1mexp(x):
    """Computes log(1 - exp(x)) numerically stably for x < 0."""
    log_half = -torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    return torch.where(
        x < log_half,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def get_last_index_per_column(matrix):
    matrix = matrix.bool()
    matrix_last_only = matrix.clone()
    matrix_last_only[:, :-1] = matrix[:, :-1] & (~matrix[:, 1:])
    last_only_index = matrix_last_only.long().argmax(dim=-2)
    mask = matrix_last_only.any(dim=-2)
    return last_only_index, mask


def get_large_negative_number(dtype):
    return torch.tensor(-0.7 * torch.finfo(dtype).max, dtype=dtype)


class ALM(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binarization_temp = args.alm_binarization_temp
        self.bias_threshold = args.alm_bias_threshold
        self.alm_loss_weight = args.alm_loss_weight
        self.alm_mode = args.alm_mode
        fn = getattr(args, "multitask_aggregation_fn", "approx_gradmag_preserve_mag")
        self.multitask_aggregation_fn = None if fn == "none" else fn

    def _use_gradmag(self):
        return uses_gradmag(self.multitask_aggregation_fn)

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
            log["gradmag_norm_sft"] = grad_norms[0].detach()
            log["gradmag_norm_alm"] = grad_norms[1].detach()
        else:
            loss = losses[0] + self.alm_loss_weight * losses[1]

        log["loss"] = loss.detach() if isinstance(loss, torch.Tensor) else loss
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def compute_multitask_losses(self, distiller, input_data, output_data):
        """Return [sft_loss, alm_loss] token sums and logging dict."""
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        device = next(model.parameters()).device

        student_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
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

        log["sft_loss"] = sft_loss
        log["alm_loss"] = alm_loss
        log["loss"] = sft_loss + self.alm_loss_weight * alm_loss if not self._use_gradmag() else sft_loss
        log["nll_loss"] = nll_loss
        log["accuracy"] = accuracy

        return [sft_loss, alm_loss], log

    def compute_alm_loss(
        self,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        alignment_matrix_a,
        alignment_matrix_b,
        loss_mask_student,
        loss_mask_teacher,
        space_mask_student,
        space_mask_teacher,
        epsilon=1e-5,
    ):
        teacher_logprobs = torch.clamp(
            F.log_softmax(teacher_logits.float(), dim=-1), max=0.0
        )
        teacher_probs = torch.exp(teacher_logprobs)

        student_logprobs = torch.clamp(
            F.log_softmax(student_logits.float(), dim=-1), max=0.0
        )
        student_probs = torch.exp(student_logprobs)

        alignment_matrix_a = alignment_matrix_a * loss_mask_student[:, :, None]
        alignment_matrix_b = alignment_matrix_b * loss_mask_teacher[:, :, None]

        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(alignment_matrix_b)
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(alignment_matrix_a)

        original_shift_labels = input_ids_teacher[..., 1:]
        teacher_main_path_logprobs = torch.take_along_dim(
            teacher_logprobs[:, :-1], original_shift_labels[..., None], dim=-1
        ).squeeze(-1)
        t_aligned_main_logp = torch.clamp(
            (teacher_main_path_logprobs[:, None] @ alignment_matrix_b[:, 1:].float()).squeeze(1),
            max=0.0,
        )

        t_space_logp = torch.clamp(
            torch.log(torch.matmul(teacher_probs, space_mask_teacher.float()) + 1e-10),
            max=0.0,
        )
        t_aligned_space_logp = torch.take_along_dim(
            t_space_logp, alignment_matrix_b_last_only_index, dim=-1
        )

        new_shift_labels = input_ids_student[..., 1:]
        student_main_path_logprobs = torch.take_along_dim(
            student_logprobs[:, :-1], new_shift_labels[..., None], dim=-1
        ).squeeze(-1)
        s_aligned_main_logp = torch.clamp(
            (student_main_path_logprobs[:, None] @ alignment_matrix_a[:, 1:].float()).squeeze(1),
            max=0.0,
        )

        s_space_logp = torch.clamp(
            torch.log(
                torch.matmul(student_probs, space_mask_student.float()[:student_probs.shape[-1]]) + 1e-10
            ),
            max=0.0,
        )
        s_aligned_space_logp = torch.take_along_dim(
            s_space_logp, alignment_matrix_a_last_only_index, dim=-1
        )

        aligned_count = alignment_matrix_b[:, 1:].sum(-2)

        if "merge_by_space_prob" in self.alm_mode:
            t_aligned_main_logp, s_aligned_main_logp, t_aligned_space_logp, \
                s_aligned_space_logp, aligned_count = self._merge_chunks(
                    t_aligned_main_logp, s_aligned_main_logp,
                    t_aligned_space_logp, s_aligned_space_logp,
                    aligned_count, alignment_matrix_a,
                )

        valid_mask = (aligned_count > 0).float()
        s_aligned_main_logp = s_aligned_main_logp * valid_mask
        t_aligned_space_logp = t_aligned_space_logp * valid_mask
        s_aligned_space_logp = s_aligned_space_logp * valid_mask

        size_s_logp = s_aligned_main_logp
        size_t_logp = t_aligned_main_logp
        size_count = aligned_count

        if "append_space" in self.alm_mode:
            count_mask = (size_count > 0)
            reversed_mask = torch.flip(count_mask, dims=[-1])
            reversed_cumsum = torch.cumsum(reversed_mask.long(), dim=-1)
            last_position_in_chunk = torch.flip(reversed_cumsum == 1, dims=[-1])

            size_s_logp = size_s_logp + (s_aligned_space_logp * last_position_in_chunk.float())
            size_t_logp = size_t_logp + (t_aligned_space_logp * last_position_in_chunk.float())

        full_counts = size_count
        s_full = size_s_logp
        t_full = size_t_logp

        t_full = torch.where(
            full_counts > 0, t_full,
            get_large_negative_number(torch.float32).to(t_full.device),
        )
        s_full = torch.where(
            full_counts > 0, s_full,
            get_large_negative_number(torch.float32).to(s_full.device),
        )

        numerator = (full_counts > 0).float()
        denominator = numerator.mean() + epsilon

        elementwise = self._binary_ce(t_full, s_full, epsilon) * numerator
        loss = elementwise.mean() / denominator

        return loss

    def _binary_ce(self, log_y_true, log_y_pred, epsilon=1e-5):
        log_y_true = (log_y_true.float() / self.binarization_temp) - epsilon
        log_y_pred = (log_y_pred.float() / self.binarization_temp) - epsilon
        return -(
            torch.exp(log_y_true) * log_y_pred
            + (-torch.expm1(log_y_true) * log1mexp(log_y_pred))
        )

    def _merge_chunks(
        self, t_main, s_main, t_space, s_space, aligned_count, alignment_matrix_a
    ):
        device = t_main.device
        batch_size = t_space.shape[0]
        chunk_count = t_space.shape[-1]

        t_space_chunk_mask = (torch.exp(t_space) > self.bias_threshold)
        cumsum_mask = torch.cumsum(t_space_chunk_mask.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        chunk_merging_indices = cumsum_mask.max(dim=-1, keepdim=True).values - cumsum_mask
        chunk_merging_values = (aligned_count > 0).float()

        chunk_merging_matrix = torch.zeros(
            (batch_size * chunk_count, chunk_count),
            dtype=alignment_matrix_a.dtype,
            device=device,
        )
        row_indices = torch.arange(batch_size * chunk_count, device=device)
        col_indices = chunk_merging_indices.reshape(-1).long()
        chunk_merging_matrix[row_indices, col_indices] = chunk_merging_values.reshape(-1).to(
            chunk_merging_matrix.dtype
        )
        chunk_merging_matrix = chunk_merging_matrix.reshape(batch_size, chunk_count, chunk_count)
        chunk_merging_matrix_last_only_index, _ = get_last_index_per_column(chunk_merging_matrix)

        t_main = torch.matmul(t_main[:, None], chunk_merging_matrix.float()).squeeze(1)
        s_main = torch.matmul(s_main[:, None], chunk_merging_matrix.float()).squeeze(1)
        t_space = torch.take_along_dim(t_space, chunk_merging_matrix_last_only_index, dim=-1)
        s_space = torch.take_along_dim(s_space, chunk_merging_matrix_last_only_index, dim=-1)
        aligned_count = torch.matmul(
            aligned_count[:, None].float(), chunk_merging_matrix.float()
        ).squeeze(1)

        return t_main, s_main, t_space, s_space, aligned_count
