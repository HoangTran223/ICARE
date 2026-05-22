"""LCS-based span alignment and span-pooling utilities for SRA.

Ported from the reference SRA implementation to support cross-tokenizer
span representation alignment in the ICARE framework.
"""

import torch
import torch.nn.functional as F
from torch.nn.functional import pad
import numpy as np

N_SPAN = 1024


def longest_common_subsequence(a, b, s_i=0, s_j=0):
    """Greedy matching of character-boundary positions between two tokenizations.

    Compares end-character offsets (column 1 of offset_mapping) to identify
    positions where both tokenizers share a token boundary.  Handles word-
    boundary resets (offset == 0) by fast-forwarding past the current word.

    Args:
        a: student offset_mapping, array-like of shape [N, 2]
        b: teacher offset_mapping, array-like of shape [M, 2]
        s_i: starting student token index (typically prompt token count)
        s_j: starting teacher token index (typically prompt token count)

    Returns:
        List of (student_idx+1, teacher_idx+1) boundary pairs (1-indexed).
    """
    if hasattr(a, "numpy"):
        a = a.numpy()
    if hasattr(b, "numpy"):
        b = b.numpy()

    m, n = len(a), len(b)
    i, j = s_i, s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue

        if a[i][1] == b[j][1]:
            result.append((i + 1, j + 1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i + 1 < m and j + 1 < n:
            if (a[i][1] == 0 and a[i + 1][1] > 0) or (
                b[j][1] == 0 and b[j + 1][1] > 0
            ):
                while j < n and b[j][1] > 0:
                    j += 1
                while i < m and a[i][1] > 0:
                    i += 1
                i += 1
                j += 1
                result.append((i, j))

    size = len(result)
    if size > N_SPAN:
        step = size / N_SPAN
        return [result[int((k + 1) * step) - 1] for k in range(N_SPAN)]

    return result


def get_pooler_tensor(segments_idxs):
    """Pad and stack per-sample segment indices into batched tensors.

    Args:
        segments_idxs: list of ``(seg_idx_list, max_len)`` per sample, where
            ``seg_idx_list`` is a list of 1-D LongTensors (token indices for
            each span).

    Returns:
        dict with ``safe_idx`` [B, max_seg, max_tok] and ``mask`` of same shape.
    """
    padded_idx_batch = []
    max_seg, max_len_all = 0, 0

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))

        padded = torch.stack(
            [pad(x, (0, max_len - len(x)), value=-1) for x in seg_idx]
        )
        padded_idx_batch.append(padded)

    def pad2d(t, h, w):
        return pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=-1)

    padded_idx_batch = torch.stack(
        [pad2d(p, max_seg, max_len_all) for p in padded_idx_batch]
    )

    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)

    return {"safe_idx": safe_idx, "mask": mask}


def prepare_pooler_v2(
    student_starts,
    student_offset_mapping,
    teacher_starts,
    teacher_offset_mapping,
    student_max_pos=None,
    teacher_max_pos=None,
):
    """Compute LCS span alignment and build pooler tensors for a batch.

    Args:
        student_starts: list[int] – per-sample student prompt token counts.
        student_offset_mapping: list of array-like [N_i, 2] per sample.
        teacher_starts: list[int] – per-sample teacher prompt token counts.
        teacher_offset_mapping: list of array-like [M_i, 2] per sample.
        student_max_pos: optional list[int] – exclusive upper bound for student
            span indices (used to clip to hidden-state length).
        teacher_max_pos: optional list[int] – same for teacher.

    Returns:
        ``(student_pooler_dict, teacher_pooler_dict)`` each with keys
        ``safe_idx`` and ``mask``.
    """
    student_seg_idxs, teacher_seg_idxs = [], []

    for idx, (s_start, s_offset, t_start, t_offset) in enumerate(
        zip(
            student_starts,
            student_offset_mapping,
            teacher_starts,
            teacher_offset_mapping,
        )
    ):
        if isinstance(s_start, torch.Tensor):
            s_start = s_start.item()
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.item()

        token_offset_start = [(s_start, t_start)]
        lcs = longest_common_subsequence(s_offset, t_offset, s_start, t_start)
        longest_common_offset = token_offset_start + lcs

        student_seg_idx = []
        teacher_seg_idx = []
        student_max_len, teacher_max_len = 1, 1

        s_clip = (
            student_max_pos[idx]
            if student_max_pos is not None
            else len(s_offset)
        )
        t_clip = (
            teacher_max_pos[idx]
            if teacher_max_pos is not None
            else len(t_offset)
        )

        for k in range(1, len(longest_common_offset)):
            s_lo = longest_common_offset[k - 1][0]
            s_hi = min(longest_common_offset[k][0], s_clip)
            t_lo = longest_common_offset[k - 1][1]
            t_hi = min(longest_common_offset[k][1], t_clip)

            s_range = torch.arange(s_lo, s_hi, dtype=torch.long)
            t_range = torch.arange(t_lo, t_hi, dtype=torch.long)

            if len(s_range) > 0 and len(t_range) > 0:
                student_seg_idx.append(s_range)
                teacher_seg_idx.append(t_range)
                student_max_len = max(student_max_len, s_range.size(0))
                teacher_max_len = max(teacher_max_len, t_range.size(0))

        if len(student_seg_idx) == 0:
            student_seg_idx.append(
                torch.tensor([max(s_start, 0)], dtype=torch.long)
            )
            teacher_seg_idx.append(
                torch.tensor([max(t_start, 0)], dtype=torch.long)
            )
            student_max_len = 1
            teacher_max_len = 1

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(
        teacher_seg_idxs
    )


def get_span_hidden_states(
    hidden_states,
    attentions,
    safe_idx,
    pooler_mask,
    attention_mask,
    layer_indices,
    is_causal=True,
):
    """Attention-weighted span pooling of hidden states.

    For each requested layer, gathers token hidden states at the positions
    specified by ``safe_idx``, weights them by (summed) attention, and averages
    within each span.

    Args:
        hidden_states: tuple/list of [B, S, D] – one per model layer
            (index 0 = embedding layer output in HF convention).
        attentions: tuple/list of [B, H, S, S] – one per attention layer
            (0-indexed; attention layer *i* is ``attentions[i]``).
        safe_idx: [B, N_spans, max_tok] – token indices per span.
        pooler_mask: [B, N_spans, max_tok] – boolean validity mask.
        attention_mask: [B, S] – overall sequence mask.
        layer_indices: list[int] – which hidden-state layers to pool.
            Use the same indexing convention as ``hidden_states``
            (e.g. ``-1`` for the last layer).
        is_causal: if True, use last query row of attention (GPT-style);
            otherwise sum over all query rows.

    Returns:
        ``(span_hidden_list, span_weights)`` where
        *  ``span_hidden_list`` is a list of [B, N_spans, D], one per layer.
        *  ``span_weights`` is [n_layers, B, N_spans, 1].
    """
    device = hidden_states[0].device
    dtype = hidden_states[0].dtype
    safe_idx = safe_idx.to(device)
    pooler_mask = pooler_mask.to(device).float()

    batch_size = hidden_states[0].size(0)
    batch_idxs = torch.arange(batch_size, device=device)[:, None, None]

    if not is_causal:
        mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        mask_4d = mask_2d.unsqueeze(1)

    hidden_state_pools = []
    all_span_weights = []

    for layer_i in layer_indices:
        attn_i = layer_i if layer_i >= 0 else len(attentions) + layer_i

        if is_causal:
            weights = attentions[attn_i].sum(dim=1)[:, -1].detach()
        else:
            weights = (
                (attentions[attn_i] * mask_4d).sum(dim=(1, 2)).detach()
            )

        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-9)
        span_w = (
            weights.unsqueeze(-1)[batch_idxs, safe_idx]
            * pooler_mask.unsqueeze(-1)
        )

        gathered = (
            hidden_states[layer_i][batch_idxs, safe_idx]
            * pooler_mask.unsqueeze(-1).to(dtype=dtype)
        )
        gathered = gathered * span_w.to(dtype=dtype)

        denom = span_w.sum(2).clamp(min=1e-5).to(dtype=dtype)
        hidden_state_mean = gathered.sum(2) / denom
        hidden_state_pools.append(hidden_state_mean)
        all_span_weights.append(span_w.sum(2))

    span_weights = torch.stack(all_span_weights)
    return hidden_state_pools, span_weights


def compute_separate_offsets(tokenizer, prompt, output, max_prompt_length):
    """Compute per-token character offsets for separately-encoded prompt+output.

    Encodes ``prompt`` and ``output`` independently (matching ICARE's data
    pipeline), shifts response offsets, and concatenates.  Appends a dummy
    entry for the EOS token.

    Returns:
        ``(offsets, prompt_token_len)`` or ``(None, 0)`` on failure.
        ``offsets`` is a list of ``(start_char, end_char)`` tuples.
    """
    try:
        enc_p = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        enc_r = tokenizer(
            output, add_special_tokens=False, return_offsets_mapping=True
        )
    except Exception:
        return None, 0

    p_offsets = list(enc_p["offset_mapping"])[:max_prompt_length]
    r_offsets = list(enc_r["offset_mapping"])

    prompt_char_len = len(prompt)
    total_char_len = prompt_char_len + len(output)
    r_shifted = [
        (s + prompt_char_len, e + prompt_char_len) for s, e in r_offsets
    ]
    r_shifted.append((total_char_len, total_char_len))

    return p_offsets + r_shifted, len(p_offsets)
