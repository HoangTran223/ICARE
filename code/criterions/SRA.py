import torch
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropyLoss
from sra_span_utils import get_span_hidden_states


class SRA(CrossEntropyLoss):
    """Span Representation Alignment (SRA).

    Loss = alpha * L_CE + (1 - alpha) * (L_span + lambda * L_geo + L_KD)

    Uses LCS-based span alignment to construct aligned span representations
    from student and teacher hidden states, matching the paper and the
    reference SRA implementation.
    """

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = args.sra_alpha
        self.geom_weight = args.sra_geom_weight
        self.span_power = args.sra_span_power
        self.use_span_loss = args.sra_span_loss
        self.kd_temp = args.kd_temperature

        self.student_layers = (
            [int(x) for x in args.sra_student_layers.split(",")]
            if args.sra_student_layers else [-1]
        )
        self.teacher_layers = (
            [int(x) for x in args.sra_teacher_layers.split(",")]
            if args.sra_teacher_layers else [-1]
        )
        assert len(self.student_layers) == len(self.teacher_layers), (
            "student_layers and teacher_layers must have the same length"
        )
        self.num_layer_pairs = len(self.student_layers)

        if args.sra_hidden_loss_weights:
            raw = [float(w) for w in args.sra_hidden_loss_weights.split(",")]
        else:
            raw = [1.0] * self.num_layer_pairs
        assert len(raw) == self.num_layer_pairs
        s = sum(raw)
        self.layer_weights = [w / s for w in raw]

        self._shared_vocab_built = False
        self._stu_shared_ids = None
        self._tea_shared_ids = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
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

        # ---- hard label loss (SFT) ----
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

        has_spans = all(x is not None for x in [s_safe_idx, s_pooler_mask, t_safe_idx, t_pooler_mask])

        # ---- span (hidden state alignment) loss ----
        span_loss = torch.tensor(0.0, device=student_out.logits.device)
        s_span_embeds_raw = None
        t_span_embeds_raw = None

        if has_spans and self.use_span_loss and hasattr(distiller, "projectors") and "sra_proj" in distiller.projectors:
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

        # ---- geometric (relation-based) loss on spans ----
        geom_loss = self._geometric_span_loss(
            s_span_embeds_raw,
            t_span_embeds_raw,
            t_span_weights[-1] if has_spans and self.use_span_loss else None,
            student_out.hidden_states[-1],
            teacher_out.hidden_states[-1],
            stu_mask, tea_mask,
        )
        log["geom_loss"] = geom_loss

        # ---- logit distillation on shared vocabulary ----
        logit_kd_loss = self._logit_distillation_loss(
            student_out, teacher_out,
            output_data["label"],
            output_data.get(f"teacher_{teacher_key}_label", output_data["label"]),
            distiller, stu_mask, tea_mask,
            s_span_embeds_raw, t_span_embeds_raw,
        )
        log["logit_kd_loss"] = logit_kd_loss

        # ---- total loss ----
        kd_loss = span_loss + self.geom_weight * geom_loss + logit_kd_loss
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss
        log["kd_loss"] = kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(student_out.logits, output_data["label"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    # ------------------------------------------------------------------
    # cosine span loss (per layer pair)
    # ------------------------------------------------------------------
    def _cosine_span_loss(self, s_proj, t_spans, t_span_weights):
        """Weighted cosine similarity loss between projected student and
        teacher span representations.

        Args:
            s_proj: [B, N_spans, D_t] – projected student spans
            t_spans: [B, N_spans, D_t] – teacher spans
            t_span_weights: [B, N_spans, 1] – attention-derived span weights
        """
        cos_sim = F.cosine_similarity(
            s_proj.float(), t_spans.float(), dim=-1, eps=1e-5
        )
        cos_loss = 1.0 - cos_sim

        w = t_span_weights.squeeze(-1).to(dtype=cos_loss.dtype)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weighted_loss = cos_loss * w
        return weighted_loss.sum(-1).mean()

    # ------------------------------------------------------------------
    # geometric (relation-based) loss on span embeddings
    # ------------------------------------------------------------------
    def _geometric_span_loss(
        self,
        s_span_embeds, t_span_embeds, t_span_weights,
        s_hidden_fallback, t_hidden_fallback,
        s_mask, t_mask,
    ):
        """MSE on similarity matrices of L2-normalised span embeddings,
        weighted by attention-derived pair weights.

        Falls back to per-token geometric loss when span data is unavailable.
        """
        if s_span_embeds is not None and t_span_embeds is not None:
            return self._geometric_loss_on_spans(
                s_span_embeds, t_span_embeds, t_span_weights,
            )
        return self._geometric_loss_fallback(
            s_hidden_fallback, t_hidden_fallback,
            s_mask, t_mask,
        )

    def _geometric_loss_on_spans(self, s_embeds, t_embeds, t_span_weights):
        """Span-level geometric loss (matching the reference SRA)."""
        B = s_embeds.size(0)
        N = s_embeds.size(1)

        w = t_span_weights.squeeze(-1)
        span_valid = (w.abs().sum(-1) > 0)

        if self.span_power != 1.0:
            w = w ** self.span_power
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        pair_w = w.unsqueeze(2) * w.unsqueeze(1)
        eye = torch.eye(N, device=pair_w.device, dtype=torch.bool)
        pair_w[:, eye] = 0.0
        pair_w = pair_w / pair_w.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

        s_norm = F.normalize(s_embeds.float(), dim=-1, eps=1e-5)
        t_norm = F.normalize(t_embeds.float(), dim=-1, eps=1e-5)
        s_sim = torch.bmm(s_norm, s_norm.transpose(1, 2))
        t_sim = torch.bmm(t_norm, t_norm.transpose(1, 2))

        diff = (s_sim - t_sim).pow(2)
        loss = (diff * pair_w).sum() / max(B, 1)
        return loss

    def _geometric_loss_fallback(self, s_hidden, t_hidden, s_mask, t_mask):
        """Per-token geometric loss (used only when span data is missing)."""
        B = s_hidden.size(0)
        s_len = s_mask.sum(dim=-1)
        t_len = t_mask.sum(dim=-1)
        min_lens = torch.minimum(s_len, t_len).long()

        total_loss = torch.tensor(0.0, device=s_hidden.device)
        for b in range(B):
            L = min_lens[b].item()
            if L < 2:
                continue
            sh = F.normalize(s_hidden[b, :L].float(), dim=-1)
            th = F.normalize(t_hidden[b, :L].float(), dim=-1)
            s_sim = sh @ sh.T
            t_sim = th @ th.T
            total_loss = total_loss + F.mse_loss(s_sim, t_sim)
        return total_loss / max(B, 1)

    # ------------------------------------------------------------------
    # logit distillation on shared vocabulary
    # ------------------------------------------------------------------
    def _logit_distillation_loss(
        self, student_out, teacher_out, s_target, t_target,
        distiller, s_mask, t_mask,
        s_span_embeds=None, t_span_embeds=None,
    ):
        """KL divergence on shared-vocabulary logits.

        When span embeddings are available, applies lm_head to span-pooled
        representations (matching the reference SRA).  Otherwise falls back
        to using the model-output logits directly.
        """
        same_tokenizer = (distiller.student_model_type == distiller.teacher_model_type)

        if s_span_embeds is not None and t_span_embeds is not None:
            s_logits = distiller.student_model.lm_head(s_span_embeds)
            t_logits = distiller.teacher_model.lm_head(
                t_span_embeds.to(
                    dtype=next(distiller.teacher_model.lm_head.parameters()).dtype,
                    device=next(distiller.teacher_model.lm_head.parameters()).device,
                )
            )
            if same_tokenizer:
                mask = (s_span_embeds.abs().sum(dim=-1) != 0).float()
                return self._forward_kl_masked(s_logits, t_logits, mask)

            stu_ids, tea_ids = self._get_shared_vocab_ids(distiller)
            if stu_ids is None or len(stu_ids) == 0:
                mask = (s_span_embeds.abs().sum(dim=-1) != 0).float()
                return self._forward_kl_masked(s_logits, t_logits, mask)

            s_shared = s_logits[..., stu_ids]
            t_shared = t_logits[..., tea_ids]
            mask = (s_shared.abs().sum(dim=-1) != 0).float()
            return self._forward_kl_masked(s_shared, t_shared, mask)

        s_logits = student_out.logits
        t_logits = teacher_out.logits

        if same_tokenizer:
            return self._forward_kl(s_logits, t_logits, s_target)

        stu_ids, tea_ids = self._get_shared_vocab_ids(distiller)
        if stu_ids is None or len(stu_ids) == 0:
            return self._forward_kl(s_logits, t_logits, s_target)

        s_shared = s_logits[..., stu_ids]
        t_shared = t_logits[..., tea_ids]
        min_len = min(s_shared.size(1), t_shared.size(1))
        s_shared = s_shared[:, :min_len, :]
        t_shared = t_shared[:, :min_len, :]
        mask = (s_shared.abs().sum(dim=-1) != 0).float()
        return self._forward_kl_masked(s_shared, t_shared, mask)

    def _forward_kl(self, s_logits, t_logits, target):
        pad_mask = target.ne(self.padding_id).float()
        s_lprobs = F.log_softmax(s_logits.float() / self.kd_temp, dim=-1)
        t_probs = F.softmax(t_logits.float() / self.kd_temp, dim=-1)
        kl = F.kl_div(s_lprobs, t_probs, reduction='none').sum(dim=-1)
        kl = kl * pad_mask
        return kl.sum() / s_logits.size(0)

    def _forward_kl_masked(self, s_logits, t_logits, mask):
        s_lprobs = F.log_softmax(s_logits.float() / self.kd_temp, dim=-1)
        t_probs = F.softmax(t_logits.float() / self.kd_temp, dim=-1)
        kl = F.kl_div(s_lprobs, t_probs, reduction='none').sum(dim=-1)
        kl = kl * mask
        return kl.sum() / s_logits.size(0)

    def _get_shared_vocab_ids(self, distiller):
        if self._shared_vocab_built:
            return self._stu_shared_ids, self._tea_shared_ids

        if hasattr(distiller, "stu2tea_id_mapping_stu"):
            stu_ids = distiller.stu2tea_id_mapping_stu.tolist()
            tea_ids = distiller.stu2tea_id_mapping_tea[:, 0].tolist()
            self._stu_shared_ids = stu_ids
            self._tea_shared_ids = tea_ids
            self._shared_vocab_built = True
            return self._stu_shared_ids, self._tea_shared_ids

        stu_tok = distiller.student_tokenizer
        tea_tok = distiller.teacher_tokenizers.get(distiller.teacher_model_type)
        if tea_tok is None:
            self._shared_vocab_built = True
            return None, None

        stu_vocab = stu_tok.get_vocab()
        tea_vocab = tea_tok.get_vocab()
        shared_tokens = set(stu_vocab.keys()) & set(tea_vocab.keys())

        if len(shared_tokens) == 0:
            self._shared_vocab_built = True
            return None, None

        stu_ids, tea_ids = [], []
        for tok in sorted(shared_tokens):
            stu_ids.append(stu_vocab[tok])
            tea_ids.append(tea_vocab[tok])

        self._stu_shared_ids = stu_ids
        self._tea_shared_ids = tea_ids
        self._shared_vocab_built = True
        return self._stu_shared_ids, self._tea_shared_ids
