"""IMPACT: Boundary-Interpolated Prefix Increment Alignment.

Core utilities for computing the BPIA loss from method.tex.
Components:
  1. Vocabulary Projection (W_{T->S}) via ridge regression on overlapping embeddings
  2. Block Influence (BI) scores for teacher layer selection
  3. Boundary-interpolated prefix increment alignment loss
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_tokenizer_vocab(vocab):
    """Unify GPT-2 (Ġ) and SentencePiece (▁) space markers for overlap counting."""
    return {k.replace("Ġ", "▁"): v for k, v in vocab.items()}


def compute_vocab_overlap_stats(student_tokenizer, teacher_tokenizer):
    """Token-string overlap between student and teacher vocabularies (IMPACT W_{T->S} support).

    Returns:
        dict with |V_S|, |V_T|, |overlap|, overlap/V_S, overlap/V_T, Jaccard.
    """
    stu_norm = _normalize_tokenizer_vocab(student_tokenizer.get_vocab())
    tea_norm = _normalize_tokenizer_vocab(teacher_tokenizer.get_vocab())
    stu_keys = set(stu_norm.keys())
    tea_keys = set(tea_norm.keys())
    overlap_keys = stu_keys & tea_keys
    union_keys = stu_keys | tea_keys

    n_s = len(stu_keys)
    n_t = len(tea_keys)
    n_o = len(overlap_keys)
    n_u = len(union_keys)

    return {
        "student_vocab_size": n_s,
        "teacher_vocab_size": n_t,
        "overlap_token_types": n_o,
        "overlap_ratio_student": n_o / n_s if n_s else 0.0,
        "overlap_ratio_teacher": n_o / n_t if n_t else 0.0,
        "jaccard": n_o / n_u if n_u else 0.0,
    }


def log_impact_vocab_overlap(
    distiller,
    criterion_name=None,
    student_label=None,
    teacher_label=None,
):
    """Print overlap stats once per IMPACT run (rank 0 only)."""
    try:
        from utils import log_rank
    except ImportError:
        return

    teacher_key = getattr(distiller, "teacher_model_type", None)
    teacher_tok = distiller.teacher_tokenizers.get(teacher_key) if teacher_key else None
    student_tok = getattr(distiller, "student_tokenizer", None)
    if teacher_tok is None or student_tok is None:
        return

    stats = compute_vocab_overlap_stats(student_tok, teacher_tok)
    crit = criterion_name or "IMPACT"
    stu = student_label or getattr(
        getattr(distiller, "args", None), "model_path", "student"
    )
    tea = teacher_label or teacher_key or getattr(
        getattr(distiller, "args", None), "teacher_model_path", "teacher"
    )

    log_rank(
        "[IMPACT vocab overlap] criterion={} student={} teacher={} | "
        "|V_S|={student_vocab_size} |V_T|={teacher_vocab_size} "
        "|overlap|={overlap_token_types} "
        "overlap/|V_S|={overlap_ratio_student:.4f} "
        "overlap/|V_T|={overlap_ratio_teacher:.4f} "
        "Jaccard={jaccard:.4f}".format(
            crit, stu, tea, **stats
        )
    )


def compute_vocab_projection(teacher_model, student_model, teacher_tokenizer,
                             student_tokenizer, lambda_reg=1.0):
    """Solve for W_{T->S} via ridge regression on overlapping vocabulary embeddings.

    W = (E_T^T E_T + lambda * I)^{-1} E_T^T E_S
    Returns W of shape [d_T, d_S].
    """
    stu_norm = _normalize_tokenizer_vocab(student_tokenizer.get_vocab())
    tea_norm = _normalize_tokenizer_vocab(teacher_tokenizer.get_vocab())
    overlap = sorted(set(stu_norm.keys()) & set(tea_norm.keys()))

    if len(overlap) == 0:
        d_T = teacher_model.config.hidden_size
        d_S = student_model.config.hidden_size
        return torch.zeros(d_T, d_S)

    stu_ids = [stu_norm[t] for t in overlap]
    tea_ids = [tea_norm[t] for t in overlap]

    stu_emb = student_model.get_input_embeddings()
    tea_emb = teacher_model.get_input_embeddings()

    E_S = stu_emb.weight.data[stu_ids].float()  # [V_overlap, d_S]
    E_T = tea_emb.weight.data[tea_ids].float()  # [V_overlap, d_T]

    d_T = E_T.size(1)
    A = E_T.T @ E_T + lambda_reg * torch.eye(d_T, device=E_T.device)
    B = E_T.T @ E_S
    W = torch.linalg.solve(A, B)  # [d_T, d_S]

    return W.detach()


def compute_bi_scores(hidden_states):
    """Compute Block Influence scores for each layer.

    BI(l) = 1 - mean_tokens(cosine_similarity(h^(l), h^(l-1)))
    Higher score = more influential layer.

    Args:
        hidden_states: list of [B, L, d] tensors (L_T+1 entries: embedding + L_T layers)

    Returns:
        bi_scores: tensor of shape [num_layers] (excludes embedding layer)
    """
    scores = []
    for l in range(1, len(hidden_states)):
        h_prev = hidden_states[l - 1].float()
        h_curr = hidden_states[l].float()
        cos = F.cosine_similarity(h_prev, h_curr, dim=-1, eps=1e-6)
        bi = 1.0 - cos.mean()
        scores.append(bi)
    return torch.stack(scores)


def select_top_k_layers(layer_scores, K):
    """Select top-K teacher layers by score. Returns sorted 0-indexed layer indices."""
    K = min(K, len(layer_scores))
    _, indices = layer_scores.topk(K)
    return indices.sort().values


def select_first_k_layers(K, L_T):
    """First K transformer layers (0-indexed: 0 .. K-1)."""
    K = min(K, L_T)
    return torch.arange(K, dtype=torch.long)


def select_last_k_layers(K, L_T):
    """Last K transformer layers (0-indexed: L_T-K .. L_T-1)."""
    K = min(K, L_T)
    return torch.arange(L_T - K, L_T, dtype=torch.long)


def select_random_k_layers(K, L_T, seed=None):
    """K distinct random layers; sorted 0-indexed indices."""
    K = min(K, L_T)
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(L_T), K))
    return torch.tensor(picks, dtype=torch.long)


def compute_ppl_layer_scores(teacher_hidden_states, teacher_model, labels, padding_id=-100):
    """Per-layer PPL proxy for teacher layer importance (IMPACT paper ablation).

    Following pruning-style layer scoring (SparseGPT, LLM-Pruner, BlockPruner):
    apply the teacher lm_head to each layer's hidden states and measure masked CE on
    teacher labels. Higher per-layer NLL indicates layers that carry more predictive
    burden; top-K highest scores are selected (same softmax weighting as BI).
    """
    lm_head = teacher_model.get_output_embeddings()
    if lm_head is None:
        return torch.zeros(len(teacher_hidden_states) - 1)

    scores = []
    for l in range(1, len(teacher_hidden_states)):
        h = teacher_hidden_states[l]
        logits = lm_head(h)
        pad_mask = labels.ne(padding_id)
        target = labels.unsqueeze(-1)
        target = torch.where(target.eq(-100), torch.zeros_like(target), target)
        logits = logits.masked_fill_(logits.isnan() | logits.isinf(), 0.0)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        nll = -lprobs.gather(-1, target).squeeze(-1)
        layer_nll = (nll * pad_mask.float()).sum() / pad_mask.float().sum().clamp(min=1)
        scores.append(layer_nll)

    return torch.stack(scores)


def select_teacher_layers_and_weights(
    choose_layer,
    teacher_hidden_states,
    K,
    L_T,
    bi_tau=1.0,
    teacher_model=None,
    teacher_labels=None,
    padding_id=-100,
    random_seed=None,
):
    """Pick teacher layer indices and aggregation weights alpha.

    choose_layer: BI | random | first | last | PPL
    - BI / PPL: top-K by score, alpha = softmax(score/tau) on selected layers
    - random / first / last: fixed index sets, alpha = uniform 1/K
    """
    K = min(K, L_T)
    mode = (choose_layer or "BI").upper()

    if mode == "FIRST":
        selected = select_first_k_layers(K, L_T)
        layer_scores = None
    elif mode == "LAST":
        selected = select_last_k_layers(K, L_T)
        layer_scores = None
    elif mode == "RANDOM":
        selected = select_random_k_layers(K, L_T, seed=random_seed)
        layer_scores = None
    elif mode == "PPL":
        if teacher_model is None or teacher_labels is None:
            raise ValueError("PPL layer selection requires teacher_model and teacher_labels")
        layer_scores = compute_ppl_layer_scores(
            teacher_hidden_states, teacher_model, teacher_labels, padding_id=padding_id
        )
        selected = select_top_k_layers(layer_scores, K)
    else:
        layer_scores = compute_bi_scores(teacher_hidden_states)
        selected = select_top_k_layers(layer_scores, K)
        mode = "BI"

    device = teacher_hidden_states[1].device
    selected = selected.to(device)

    if layer_scores is not None and mode in ("BI", "PPL"):
        alpha = compute_layer_weights(layer_scores.to(device), selected, tau=bi_tau)
    else:
        alpha = torch.ones(len(selected), device=device) / len(selected)

    return selected, alpha, mode, layer_scores


def compute_layer_weights(layer_scores, selected_indices, tau=1.0):
    """Compute softmax-normalized layer weights from BI scores.

    alpha_l = exp(BI(l)/tau) / sum_r exp(BI(r)/tau)
    """
    selected_scores = layer_scores[selected_indices]
    return F.softmax(selected_scores / tau, dim=0)


def map_teacher_to_student_layer(teacher_layer_idx, L_T, L_S):
    """Map teacher layer to student layer via relative depth: g(l) = ceil(l * L_S / L_T).

    teacher_layer_idx is 1-indexed (layer 1 to L_T).
    Returns 1-indexed student layer.
    """
    return math.ceil(teacher_layer_idx * L_S / L_T)


def map_student_to_teacher_layer(student_layer_idx, L_T, L_S):
    """Map student layer to teacher layer via relative depth: l = ceil(m * L_T / L_S).

    student_layer_idx is 1-indexed (layer 1 to L_S).
    Returns 1-indexed teacher layer.
    """
    return math.ceil(student_layer_idx * L_T / L_S)


def normalize_choose_align(choose_align):
    """Normalize CLI value to BI | 1-1 | 1-all."""
    s = (choose_align or "BI").strip()
    upper = s.upper()
    if upper in ("1-1", "1_1", "11", "DEPTH", "DEPTH-RATIO", "DEPTH_RATIO"):
        return "1-1"
    if upper in ("1-ALL", "1_ALL", "ALP", "ALP-KD", "ALPKD"):
        return "1-all"
    return "BI"


def build_impact_alignment_pairs(
    choose_align,
    choose_layer,
    teacher_hidden_states,
    top_k,
    L_T,
    L_S,
    bi_tau=1.0,
    teacher_model=None,
    teacher_labels=None,
    padding_id=-100,
    random_seed=None,
):
    """Build (teacher_layer, student_layer, weight) triples for BPIA aggregation.

    choose_align:
      - BI: top-K teacher layers (choose_layer), BI/PPL softmax weights, g(l) depth map
      - 1-1: every student layer m paired with teacher ceil(m*L_T/L_S), uniform 1/L_S
      - 1-all: last student layer L_S paired with every teacher layer, uniform 1/L_T
        (ICARE align ablation / simplified ALP-KD one-to-all)
    """
    align_mode = normalize_choose_align(choose_align)
    device = teacher_hidden_states[1].device
    pairs = []

    if align_mode == "1-1":
        weight = 1.0 / max(L_S, 1)
        for m in range(1, L_S + 1):
            l = map_student_to_teacher_layer(m, L_T, L_S)
            pairs.append((l, m, weight))
        return pairs, align_mode, None

    if align_mode == "1-all":
        m = L_S
        weight = 1.0 / max(L_T, 1)
        for l in range(1, L_T + 1):
            pairs.append((l, m, weight))
        return pairs, align_mode, None

    selected, alpha, layer_mode, _ = select_teacher_layers_and_weights(
        choose_layer,
        teacher_hidden_states,
        top_k,
        L_T,
        bi_tau=bi_tau,
        teacher_model=teacher_model,
        teacher_labels=teacher_labels,
        padding_id=padding_id,
        random_seed=random_seed,
    )
    for sel_idx, w in zip(selected, alpha):
        l = sel_idx.item() + 1
        m = map_teacher_to_student_layer(l, L_T, L_S)
        pairs.append((l, m, w.item() if torch.is_tensor(w) else float(w)))
    return pairs, align_mode, layer_mode


def compute_text_offsets(tokenizer, text, max_length=None):
    """Compute character-level offsets for each token in text.

    Returns list of (start, end) tuples. Handles fast and slow tokenizers.
    """
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if max_length is not None:
            offsets = offsets[:max_length]
        return offsets
    except Exception:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if max_length is not None:
            tokens = tokens[:max_length]
        offsets = []
        pos = 0
        decoded_full = tokenizer.decode(tokens)
        for i, tid in enumerate(tokens):
            tok_str = tokenizer.decode([tid])
            start = decoded_full.find(tok_str, pos)
            if start < 0:
                start = pos
            end = start + len(tok_str)
            offsets.append((start, end))
            pos = end
        return offsets


def _get_teacher_boundary_state(c, teacher_offsets, teacher_prefix_states):
    """Get the virtual teacher state at text boundary c.

    teacher_offsets: list of (a_j, b_j) for each teacher token (0-indexed)
    teacher_prefix_states: [L_T+1, d] where index 0 is the initial state (zeros),
                           and index j+1 is the state after processing token j
    """
    num_tokens = len(teacher_offsets)
    if num_tokens == 0:
        return teacher_prefix_states[0]

    if c <= teacher_offsets[0][0]:
        return teacher_prefix_states[0]
    if c >= teacher_offsets[-1][1]:
        return teacher_prefix_states[num_tokens]

    for j in range(num_tokens):
        a_j, b_j = teacher_offsets[j]
        if c == a_j:
            return teacher_prefix_states[j]
        if c == b_j:
            return teacher_prefix_states[j + 1]
        if a_j < c < b_j:
            span_len = b_j - a_j
            if span_len == 0:
                return teacher_prefix_states[j]
            alpha = (c - a_j) / span_len
            return (1.0 - alpha) * teacher_prefix_states[j] + alpha * teacher_prefix_states[j + 1]

    return teacher_prefix_states[num_tokens]


def compute_teacher_increments_batch(student_offsets_batch, teacher_offsets_batch,
                                     teacher_hidden, W_proj, student_mask, teacher_mask):
    """Compute teacher increments for a batch, projected into student space.

    Args:
        student_offsets_batch: list of list of (a, b) tuples, one per sample
        teacher_offsets_batch: list of list of (a, b) tuples, one per sample
        teacher_hidden: [B, T_len, d_T] teacher hidden states at one layer
        W_proj: [d_T, d_S] vocabulary projection matrix
        student_mask: [B, S_len] attention mask
        teacher_mask: [B, T_len] attention mask

    Returns:
        teacher_increments: [B, S_len, d_S] teacher increment for each student token position
    """
    B, S_len = student_mask.shape
    device = teacher_hidden.device
    dtype = teacher_hidden.dtype

    t_projected = teacher_hidden.float() @ W_proj.to(device).float()  # [B, T_len, d_S]
    d_S = t_projected.size(-1)

    zero_state = torch.zeros(1, d_S, device=device, dtype=torch.float32)
    result = torch.zeros(B, S_len, d_S, device=device, dtype=torch.float32)

    for b in range(B):
        s_offsets = student_offsets_batch[b]
        t_offsets = teacher_offsets_batch[b]
        t_len = int(teacher_mask[b].sum().item())
        s_len = int(student_mask[b].sum().item())

        t_states = t_projected[b, :t_len]  # [t_len, d_S]
        t_prefix = torch.cat([zero_state, t_states], dim=0)  # [t_len+1, d_S]

        t_off = t_offsets[:t_len] if len(t_offsets) >= t_len else t_offsets

        num_s = min(s_len, len(s_offsets))
        for i in range(num_s):
            a_i, b_i = s_offsets[i]
            state_at_b = _get_teacher_boundary_state(b_i, t_off, t_prefix)
            state_at_a = _get_teacher_boundary_state(a_i, t_off, t_prefix)
            result[b, i] = state_at_b - state_at_a

    return result.to(dtype)


def compute_student_increments(student_hidden, student_mask):
    """Compute student prefix increments: Δ_S[i] = h_S[i] - h_S[i-1].

    For position 0, uses zeros as the previous state.

    Args:
        student_hidden: [B, S_len, d_S]
        student_mask: [B, S_len]

    Returns:
        increments: [B, S_len, d_S]
    """
    B, S_len, d_S = student_hidden.shape
    h_prev = torch.zeros(B, 1, d_S, device=student_hidden.device, dtype=student_hidden.dtype)
    h_shifted = torch.cat([h_prev, student_hidden[:, :-1]], dim=1)
    increments = student_hidden - h_shifted
    return increments


def compute_impact_loss(student_hidden_states, teacher_hidden_states,
                       student_offsets_batch, teacher_offsets_batch,
                       W_proj, student_mask, teacher_mask,
                       top_k=4, bi_tau=1.0, L_T=None, L_S=None,
                       choose_layer="BI", choose_align="BI",
                       teacher_model=None, teacher_labels=None,
                       padding_id=-100, random_seed=None, log_selection=False):
    """Compute the full IMPACT (BPIA) loss.

    L_BPIA = sum_pairs w * (1/N) * sum_i (1 - cos(Delta_S, Delta_T))
    Pairing and weights are set by choose_align (see build_impact_alignment_pairs).

    Args:
        student_hidden_states: list of [B, S_len, d_S] (L_S+1 entries)
        teacher_hidden_states: list of [B, T_len, d_T] (L_T+1 entries)
        student_offsets_batch: list of list of (start, end) tuples
        teacher_offsets_batch: list of list of (start, end) tuples
        W_proj: [d_T, d_S] projection matrix
        student_mask: [B, S_len]
        teacher_mask: [B, T_len]
        top_k: number of teacher layers to select (BI align only)
        bi_tau: temperature for BI score softmax (BI align only)
        choose_layer: teacher layer selection when choose_align=BI
        choose_align: BI | 1-1 | 1-all
        L_T: number of teacher layers (excluding embedding)
        L_S: number of student layers (excluding embedding)

    Returns:
        loss: scalar
    """
    if L_T is None:
        L_T = len(teacher_hidden_states) - 1
    if L_S is None:
        L_S = len(student_hidden_states) - 1

    pairs, align_mode, layer_mode = build_impact_alignment_pairs(
        choose_align,
        choose_layer,
        teacher_hidden_states,
        top_k,
        L_T,
        L_S,
        bi_tau=bi_tau,
        teacher_model=teacher_model,
        teacher_labels=teacher_labels,
        padding_id=padding_id,
        random_seed=random_seed,
    )

    if log_selection:
        try:
            from utils import log_rank
            t_layers = [p[0] for p in pairs]
            s_layers = [p[1] for p in pairs]
            weights = [round(p[2], 4) if not torch.is_tensor(p[2]) else round(p[2].item(), 4)
                       for p in pairs]
            msg = (
                "[IMPACT align] mode={} num_pairs={} "
                "teacher_layers(1-based)={} student_layers(1-based)={} weights={}"
            ).format(align_mode, len(pairs), t_layers, s_layers, weights)
            if layer_mode is not None:
                msg += " layer_select={}".format(layer_mode)
            log_rank(msg)
        except ImportError:
            pass

    device = student_hidden_states[0].device
    total_loss = torch.tensor(0.0, device=device)
    mask_float = student_mask.float()
    N = mask_float.sum(dim=1).clamp(min=1)

    for teacher_layer, student_layer, weight in pairs:
        t_hidden = teacher_hidden_states[teacher_layer]
        s_hidden = student_hidden_states[min(student_layer, L_S)]
        w = weight if torch.is_tensor(weight) else torch.tensor(
            weight, device=device, dtype=total_loss.dtype
        )

        delta_T = compute_teacher_increments_batch(
            student_offsets_batch, teacher_offsets_batch,
            t_hidden, W_proj, student_mask, teacher_mask
        )
        delta_S = compute_student_increments(s_hidden, student_mask)

        cos_sim = F.cosine_similarity(
            delta_S.float(), delta_T.float(), dim=-1, eps=1e-6
        )
        cos_loss = (1.0 - cos_sim) * mask_float
        layer_loss = (cos_loss.sum(dim=1) / N).mean()

        total_loss = total_loss + w * layer_loss

    return total_loss


class IMPACTModule(nn.Module):
    """Reusable IMPACT module that can be attached to any base criterion.

    Computes vocabulary projection once, then provides compute_loss() method.
    """

    def __init__(self, args):
        super().__init__()
        self.top_k = getattr(args, "impact_top_k", 4)
        self.bi_tau = getattr(args, "impact_bi_tau", 1.0)
        self.lambda_reg = getattr(args, "impact_lambda_reg", 1.0)
        self.choose_layer = getattr(args, "impact_choose_layer", None) or getattr(
            args, "choose_layer", "BI"
        )
        self.choose_align = getattr(args, "impact_choose_align", None) or getattr(
            args, "choose_align", "BI"
        )
        self.padding_id = getattr(args, "padding_id", -100)
        self.random_seed = getattr(args, "seed", None)
        self._criterion_name = getattr(args, "criterion", None) or "IMPACT"
        self._W_proj = None
        self._initialized = False
        self._layer_select_logged = False

    def _ensure_initialized(self, distiller):
        if self._initialized:
            return
        teacher_tok = distiller.teacher_tokenizers.get(distiller.teacher_model_type)
        if teacher_tok is None:
            return
        log_impact_vocab_overlap(
            distiller,
            criterion_name=self._criterion_name,
        )
        W = compute_vocab_projection(
            distiller.teacher_model, distiller.student_model,
            teacher_tok, distiller.student_tokenizer,
            lambda_reg=self.lambda_reg,
        )
        self.register_buffer("_W_proj_buf", W)
        self._W_proj = self._W_proj_buf
        self._initialized = True

    def compute_loss(self, distiller, student_hidden_states, teacher_hidden_states,
                     student_offsets_batch, teacher_offsets_batch,
                     student_mask, teacher_mask, teacher_labels=None):
        """Compute the IMPACT (BPIA) loss.

        Returns scalar loss (without lambda_impact scaling — caller handles that).
        """
        self._ensure_initialized(distiller)
        if self._W_proj is None:
            return torch.tensor(0.0, device=student_mask.device)

        L_T = len(teacher_hidden_states) - 1
        L_S = len(student_hidden_states) - 1

        loss = compute_impact_loss(
            student_hidden_states, teacher_hidden_states,
            student_offsets_batch, teacher_offsets_batch,
            self._W_proj, student_mask, teacher_mask,
            top_k=self.top_k, bi_tau=self.bi_tau,
            L_T=L_T, L_S=L_S,
            choose_layer=self.choose_layer,
            choose_align=self.choose_align,
            teacher_model=distiller.teacher_model,
            teacher_labels=teacher_labels,
            padding_id=self.padding_id,
            random_seed=self.random_seed,
            log_selection=not self._layer_select_logged,
        )
        self._layer_select_logged = True
        return loss
