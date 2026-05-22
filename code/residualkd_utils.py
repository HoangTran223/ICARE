import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectorTA(nn.Module):
    """Bottleneck projector for teacher-to-anchor and anchor-to-teacher mapping.
    Used in Stage 1 (pretrain) and Stage 2 (frozen encode path).
    """

    def __init__(self, d_T, d_A):
        super().__init__()
        self.P_TA = nn.Linear(d_T, d_A, bias=False)
        self.P_AT = nn.Linear(d_A, d_T, bias=False)

    def forward(self, h):
        dtype = self.P_TA.weight.dtype
        z = self.P_TA(h.to(dtype))
        return z, self.P_AT(z)

    def encode(self, h):
        return self.P_TA(h.to(self.P_TA.weight.dtype))


class ProjectorSA(nn.Module):
    """Student-to-anchor projector (learnable during Stage 2)."""

    def __init__(self, d_S, d_A):
        super().__init__()
        self.P_SA = nn.Linear(d_S, d_A, bias=False)

    def forward(self, h_S):
        return self.P_SA(h_S.to(self.P_SA.weight.dtype))


class ProjectorAS(nn.Module):
    """Anchor-to-student projector (learnable during Stage 2, zero-initialized)."""

    def __init__(self, d_A, d_S):
        super().__init__()
        self.P_AS = nn.Linear(d_A, d_S, bias=False)
        nn.init.zeros_(self.P_AS.weight)

    def forward(self, h_A):
        return self.P_AS(h_A.to(self.P_AS.weight.dtype))


def cross_model_attention(h_S_A, h_T_A):
    """Compute cross-model attention to align teacher positions to student positions.

    Args:
        h_S_A: student hiddens in anchor space [B, S_len, d_A]
        h_T_A: teacher hiddens in anchor space [B, T_len, d_A]

    Returns:
        h_T_aligned: teacher hiddens aligned to student positions [B, S_len, d_A]
        A: attention weights [B, S_len, T_len]
    """
    d_A = h_S_A.size(-1)
    Q = h_S_A / h_S_A.std(dim=-1, keepdim=True).clamp(min=1e-5)
    K = h_T_A / h_T_A.std(dim=-1, keepdim=True).clamp(min=1e-5)
    A = torch.matmul(Q, K.transpose(-1, -2)) / (d_A ** 0.5)
    A = F.softmax(A, dim=-1)
    return torch.matmul(A, h_T_A), A


def compute_residual_mask_same_tokenizer(teacher_logits, labels, response_mask):
    """Residual mask: positions where teacher prediction differs from ground truth.

    Args:
        teacher_logits: [B, T_len, V_T]
        labels: [B, S_len]  (same tokenizer → S_len == T_len)
        response_mask: [B, S_len] float mask for response tokens
    """
    pred = teacher_logits.argmax(dim=-1)
    wrong = (pred != labels) & (labels != -100)
    mask = wrong & response_mask.bool()
    return mask


def compute_residual_mask_cross_tokenizer(teacher_logits, A_align, response_mask):
    """Entropy-based residual mask for cross-tokenizer settings.

    Args:
        teacher_logits: [B, T_len, V_T]
        A_align: attention weights [B, S_len, T_len] from cross_model_attention
        response_mask: [B, S_len] float mask for response tokens
    """
    t_probs = F.softmax(teacher_logits.float(), dim=-1)
    t_entropy = -(t_probs * t_probs.clamp(min=1e-9).log()).sum(dim=-1)  # [B, T_len]
    max_H = math.log(t_probs.size(-1))
    t_uncertain = t_entropy / max_H  # normalized to [0,1]
    # A_align may be bf16 (from student/teacher hiddens); t_uncertain is float32 — matmul needs one dtype
    aligned_uncertain = A_align.float() @ t_uncertain.unsqueeze(-1).float()  # [B, S_len, 1]
    mask = (aligned_uncertain.squeeze(-1) > 0.5) & response_mask.bool()
    return mask


def compute_beta(h_S, proj_to_S, mask, d_S, d_A):
    """Compute the residual scaling factor β.

    β = √(d_S/d_A) · mean(||h_S|| / ||proj_to_S||) per sequence, clamped [0.05, 10.0]
    """
    mask_f = mask.float()
    s_norm = h_S.float().norm(dim=-1)
    p_norm = proj_to_S.float().norm(dim=-1).clamp(min=1e-5)
    ratio = s_norm / p_norm
    beta = (d_S / d_A) ** 0.5 * (ratio * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1)
    beta = beta.mean().detach().clamp(0.05, 10.0)
    return beta
