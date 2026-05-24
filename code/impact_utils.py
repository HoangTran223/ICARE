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


def resolve_teacher_tokenizer(distiller):
    """Teacher tokenizer for DSKDv2 (singular) or multi-teacher distiller (dict)."""
    tok = getattr(distiller, "teacher_tokenizer", None)
    if tok is not None:
        return tok
    teacher_tokenizers = getattr(distiller, "teacher_tokenizers", None)
    if not teacher_tokenizers:
        return None
    teacher_key = getattr(distiller, "teacher_model_type", None)
    if teacher_key is None:
        args = getattr(distiller, "args", None)
        teacher_key = getattr(args, "teacher_model_type", None) if args else None
    return teacher_tokenizers.get(teacher_key) if teacher_key else None


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
    if teacher_key is None:
        args = getattr(distiller, "args", None)
        teacher_key = getattr(args, "teacher_model_type", None) if args else None
    teacher_tok = resolve_teacher_tokenizer(distiller)
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
    """Normalize CLI value to BI | 1-1 | 1-all | 1-random."""
    s = (choose_align or "BI").strip()
    upper = s.upper()
    if upper in ("1-1", "1_1", "11", "DEPTH", "DEPTH-RATIO", "DEPTH_RATIO"):
        return "1-1"
    if upper in ("1-ALL", "1_ALL", "ALP", "ALP-KD", "ALPKD"):
        return "1-all"
    if upper in ("1-RANDOM", "1_RANDOM", "RAIL", "RAIL-KD", "RAIL_KD", "RAILKD"):
        return "1-random"
    return "BI"


def criterion_supports_rail_kd_align(criterion_name):
    """RAIL_KD layer pairing is only enabled for DSKDv2+IMPACT."""
    return (criterion_name or "") in (
        "dual_space_kd_v2_impact",
        "dual_space_kd_v2_ipact",
    )


def sample_rail_kd_teacher_layers(
    L_T,
    L_S,
    seed=None,
    no_random=False,
    has_embed=False,
    has_final=False,
):
    """GLMKD ``RAIL_KD.get_teacher_hook`` teacher layer indices (HF hidden_states index).

    Matches ``distill_model.RAIL_KD``:
      - ``random.sample(range(1, teacher_num_layers), num_layers - 1)`` then ``sort()``
      - ``teacher_num_layers`` = # teacher transformer blocks (= ``L_T`` here)
      - pool is ``{1, ..., L_T - 1}`` (upper bound exclusive, same as GLMKD)
      - count = ``L_S - 1`` (not ``L_S``); optional embed inserts ``0``, optional final adds output hook

    Returns sorted teacher indices used for intermediate ``layernorm_output`` hooks.
    """
    if L_T < 1 or L_S < 1:
        return [1]

    teacher_num_layers = L_T
    num_layers = L_S
    n_sample = max(num_layers - 1, 1)

    if no_random:
        layers_per_block = max(int(teacher_num_layers / max(num_layers, 1)), 1)
        layers = list(range(0, teacher_num_layers + 1, layers_per_block))[1:-1]
        layers = [int(i) for i in layers if 0 < i < teacher_num_layers]
        if len(layers) < n_sample:
            pad = layers[-1] if layers else 1
            layers = layers + [pad] * (n_sample - len(layers))
        layers = layers[:n_sample]
    else:
        pool = list(range(1, teacher_num_layers))
        if not pool:
            pool = [1]
        rng = random.Random(seed)
        k = min(n_sample, len(pool))
        layers = sorted(rng.sample(pool, k))

    if has_embed:
        layers = [0] + layers
    if has_final:
        layers = layers + [teacher_num_layers]
    return layers


def rail_kd_num_pairs(L_S, has_final=False):
    """Number of (student, teacher) reps zipped in GLMKD ``inter_loss`` (layer-wise mode)."""
    if has_final:
        return L_S
    return max(L_S - 1, 1)


def build_rail_kd_alignment_pairs(
    L_T,
    L_S,
    seed=None,
    no_random=False,
    has_embed=False,
    has_final=False,
):
    """Build BPIA pairs for ``choose_align=1-random`` using GLMKD RAIL-KD layer matching.

    GLMKD pairs the first ``L_S - 1`` student ``layernorm_output`` hooks (student layers
    ``1 .. L_S-1`` in HF ``hidden_states``) with the sorted random teacher hooks — not all
    ``L_S`` student blocks unless ``rail_kd_has_final`` is enabled.
    """
    teacher_layers = sample_rail_kd_teacher_layers(
        L_T,
        L_S,
        seed=seed,
        no_random=no_random,
        has_embed=has_embed,
        has_final=has_final,
    )
    n_pairs = rail_kd_num_pairs(L_S, has_final=has_final)
    t_start = 1 if has_embed else 0
    weight = 1.0 / max(n_pairs, 1)
    pairs = []
    for i in range(n_pairs):
        student_m = i + 1
        teacher_l = teacher_layers[t_start + i]
        pairs.append((teacher_l, student_m, weight))
    return pairs, "1-random", "RAIL_KD"


def compute_alp_kd_attention_weights(student_hidden_states, teacher_hidden_states, W_proj, L_T, L_S):
    """ALP-KD layer attention (GLMKD ``ALP_KD.inter_loss``).

    For each student layer, softmax over all teacher layers using dot-product
    scores between the first-token (CLS) hidden states. Teacher CLS is projected
    to student dim via ``W_proj`` (cross-tokenizer), matching GLMKD when hidden
    sizes differ.

    Returns:
        alpha: [B, L_S, L_T] with sum over L_T = 1 per student layer.
    """
    device = student_hidden_states[1].device
    W = W_proj.to(device=device, dtype=torch.float32)
    s_cls = torch.stack(
        [student_hidden_states[m][:, 0, :].float() for m in range(1, L_S + 1)],
        dim=1,
    )
    t_cls = torch.stack(
        [(teacher_hidden_states[l][:, 0, :].float() @ W) for l in range(1, L_T + 1)],
        dim=1,
    )
    scores = torch.bmm(s_cls, t_cls.transpose(1, 2))
    return F.softmax(scores, dim=-1)


def fuse_teacher_hidden_by_attention(teacher_hidden_states, alpha_bt, L_T):
    """Fused teacher hidden H^C = sum_l alpha_l * H_T^l (ALP-KD Eq. 5–6 style).

    Args:
        alpha_bt: [B, L_T] weights for one student layer.
    """
    fused = None
    for l in range(1, L_T + 1):
        h = teacher_hidden_states[l]
        w = alpha_bt[:, l - 1].view(-1, 1, 1).to(device=h.device, dtype=h.dtype)
        fused = h * w if fused is None else fused + h * w
    return fused


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
    rail_teacher_layers=None,
    rail_kd_no_random=False,
    rail_kd_has_embed=False,
    rail_kd_has_final=False,
):
    """Build (teacher_layer, student_layer, weight) triples for BPIA aggregation.

    choose_align:
      - BI: top-K teacher layers (choose_layer), BI/PPL softmax weights, g(l) depth map
      - 1-1: every student layer m paired with teacher ceil(m*L_T/L_S), uniform 1/L_S
      - 1-all: not used here — handled in ``compute_impact_loss`` via ALP-KD attention
      - 1-random: RAIL_KD (GLMKD) sorted random teacher layers, 1:1 with student depth
    """
    align_mode = normalize_choose_align(choose_align)
    pairs = []

    if align_mode == "1-random":
        if rail_teacher_layers is not None:
            n_pairs = rail_kd_num_pairs(L_S, has_final=rail_kd_has_final)
            t_start = 1 if rail_kd_has_embed else 0
            weight = 1.0 / max(n_pairs, 1)
            layers = list(rail_teacher_layers)
            pairs = [
                (layers[t_start + i], i + 1, weight) for i in range(n_pairs)
            ]
            return pairs, align_mode, "RAIL_KD"
        return build_rail_kd_alignment_pairs(
            L_T,
            L_S,
            seed=random_seed,
            no_random=rail_kd_no_random,
            has_embed=rail_kd_has_embed,
            has_final=rail_kd_has_final,
        )

    if align_mode == "1-1":
        weight = 1.0 / max(L_S, 1)
        for m in range(1, L_S + 1):
            l = map_student_to_teacher_layer(m, L_T, L_S)
            pairs.append((l, m, weight))
        return pairs, align_mode, None

    if align_mode == "1-all":
        raise RuntimeError(
            "choose_align=1-all uses ALP-KD fused teacher states in compute_impact_loss; "
            "do not call build_impact_alignment_pairs for 1-all."
        )

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


def _log_alp_kd_alignment(alpha, L_S, L_T):
    """Log ALP-KD attention (sample student layers, batch 0)."""
    try:
        from utils import log_rank
    except ImportError:
        return
    b0 = 0
    samples = sorted({1, max(1, L_S // 2), L_S})
    parts = []
    for m in samples:
        w = alpha[b0, m - 1].detach().float().cpu().tolist()
        ranked = sorted(enumerate(w, start=1), key=lambda x: -x[1])
        top5 = [(l, round(a, 4)) for l, a in ranked[:5]]
        parts.append(f"s{m}:{top5}")
    log_rank(
        "[IMPACT align] mode=1-all (ALP-KD, GLMKD ALP_KD) L_S={} L_T={} num_student_layers={} "
        "attn_softmax_over_teacher batch0_samples: {}".format(
            L_S, L_T, L_S, "; ".join(parts)
        )
    )


def compute_impact_loss(student_hidden_states, teacher_hidden_states,
                       student_offsets_batch, teacher_offsets_batch,
                       W_proj, student_mask, teacher_mask,
                       top_k=4, bi_tau=1.0, L_T=None, L_S=None,
                       choose_layer="BI", choose_align="BI",
                       teacher_model=None, teacher_labels=None,
                       padding_id=-100, random_seed=None, log_selection=False,
                       rail_teacher_layers=None, rail_kd_no_random=False,
                       rail_kd_has_embed=False, rail_kd_has_final=False):
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
        choose_align: BI | 1-1 | 1-all | 1-random (1-random = GLMKD RAIL_KD pairing)
        L_T: number of teacher layers (excluding embedding)
        L_S: number of student layers (excluding embedding)

    Returns:
        loss: scalar
    """
    if L_T is None:
        L_T = len(teacher_hidden_states) - 1
    if L_S is None:
        L_S = len(student_hidden_states) - 1

    align_mode = normalize_choose_align(choose_align)
    if align_mode == "1-all":
        if W_proj is None:
            raise ValueError("ALP-KD (choose_align=1-all) requires vocabulary projection W_proj")
        alpha = compute_alp_kd_attention_weights(
            student_hidden_states, teacher_hidden_states, W_proj, L_T, L_S
        )
        if log_selection:
            _log_alp_kd_alignment(alpha, L_S, L_T)

        device = student_hidden_states[0].device
        total_loss = torch.tensor(0.0, device=device)
        mask_float = student_mask.float()
        N = mask_float.sum(dim=1).clamp(min=1)
        layer_weight = 1.0 / max(L_S, 1)

        for m in range(1, L_S + 1):
            s_hidden = student_hidden_states[m]
            t_fused = fuse_teacher_hidden_by_attention(
                teacher_hidden_states, alpha[:, m - 1, :], L_T
            )
            delta_T = compute_teacher_increments_batch(
                student_offsets_batch, teacher_offsets_batch,
                t_fused, W_proj, student_mask, teacher_mask
            )
            delta_S = compute_student_increments(s_hidden, student_mask)
            cos_sim = F.cosine_similarity(
                delta_S.float(), delta_T.float(), dim=-1, eps=1e-6
            )
            cos_loss = (1.0 - cos_sim) * mask_float
            layer_loss = (cos_loss.sum(dim=1) / N).mean()
            total_loss = total_loss + layer_weight * layer_loss
        return total_loss

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
        rail_teacher_layers=rail_teacher_layers,
        rail_kd_no_random=rail_kd_no_random,
        rail_kd_has_embed=rail_kd_has_embed,
        rail_kd_has_final=rail_kd_has_final,
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
            if align_mode == "1-random" and rail_teacher_layers is not None:
                msg += " rail_sample={}".format(list(rail_teacher_layers)[:L_S])
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
        align_mode = normalize_choose_align(self.choose_align)
        self._criterion_name = getattr(args, "criterion", None) or "IMPACT"
        if align_mode == "1-random" and not criterion_supports_rail_kd_align(
            self._criterion_name
        ):
            raise ValueError(
                "choose_align=1-random (RAIL_KD) is only supported for "
                "criterion=dual_space_kd_v2_impact (DSKDv2_IMPACT)."
            )
        self.rail_kd_epochs = int(getattr(args, "impact_rail_kd_epochs", 1))
        self.rail_kd_iters = int(getattr(args, "impact_rail_kd_iters", 0))
        self.rail_kd_no_random = bool(getattr(args, "impact_rail_kd_no_random", False))
        self.rail_kd_has_embed = bool(getattr(args, "impact_rail_kd_has_embed", False))
        self.rail_kd_has_final = bool(getattr(args, "impact_rail_kd_has_final", False))
        self.rail_kd_show_layers = bool(
            getattr(args, "impact_rail_kd_show_layers", False)
        )
        self._rail_last_iter = -1
        self._rail_teacher_layers = None
        self._rail_last_epoch = -1
        self._rail_last_iter = -1
        self._rail_forward_step = 0
        self.padding_id = getattr(args, "padding_id", -100)
        self.random_seed = getattr(args, "seed", None)
        self._W_proj = None
        self._initialized = False
        self._layer_select_logged = False

    def _ensure_initialized(self, distiller):
        if self._initialized:
            return
        teacher_tok = resolve_teacher_tokenizer(distiller)
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

    def _maybe_resample_rail_layers(self, L_T, L_S, distiller):
        """Resample RAIL_KD teacher layers — same schedule as GLMKD ``get_teacher_hook``."""
        train_epoch = int(
            getattr(getattr(distiller, "args", None), "current_train_epoch", 0)
        )
        global_step = int(
            getattr(getattr(distiller, "args", None), "current_global_step", 0)
        )
        self._rail_forward_step += 1
        resample = self._rail_teacher_layers is None

        if self.rail_kd_iters > 0:
            # GLMKD: new hook when iteration % iters == 0 and iteration != last_iter
            if (
                global_step % self.rail_kd_iters == 0
                and global_step != self._rail_last_iter
            ):
                resample = True
            else:
                resample = False if self._rail_teacher_layers is not None else True
        elif self.rail_kd_epochs > 0:
            # GLMKD: new hook when epoch % epochs == 0 and epoch != last_epoch
            if (
                train_epoch % self.rail_kd_epochs == 0
                and train_epoch != self._rail_last_epoch
            ):
                resample = True
            else:
                resample = False if self._rail_teacher_layers is not None else True

        if resample:
            if self.random_seed is None:
                seed = None
            else:
                seed = int(self.random_seed) + train_epoch * 10007 + global_step
            self._rail_teacher_layers = sample_rail_kd_teacher_layers(
                L_T,
                L_S,
                seed=seed,
                no_random=self.rail_kd_no_random,
                has_embed=self.rail_kd_has_embed,
                has_final=self.rail_kd_has_final,
            )
            self._rail_last_epoch = train_epoch
            self._rail_last_iter = global_step
            if self.rail_kd_show_layers:
                try:
                    from utils import log_rank
                    log_rank(
                        "[IMPACT RAIL_KD] epoch={} step={} teacher_layers(1-based)={}".format(
                            train_epoch, global_step, self._rail_teacher_layers
                        )
                    )
                except ImportError:
                    pass
        return self._rail_teacher_layers

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

        rail_layers = None
        if normalize_choose_align(self.choose_align) == "1-random":
            rail_layers = self._maybe_resample_rail_layers(L_T, L_S, distiller)

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
            rail_teacher_layers=rail_layers,
            rail_kd_no_random=self.rail_kd_no_random,
            rail_kd_has_embed=self.rail_kd_has_embed,
            rail_kd_has_final=self.rail_kd_has_final,
        )
        self._layer_select_logged = True
        return loss
