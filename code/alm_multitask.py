"""Approximate GradMag multitask loss aggregation (tokenkit/training/multitask.py)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


GRADMAG_MODES = frozenset({"approx_gradmag", "approx_gradmag_preserve_mag"})


def uses_gradmag(multitask_aggregation_fn: str | None) -> bool:
    if multitask_aggregation_fn is None or multitask_aggregation_fn == "none":
        return False
    return multitask_aggregation_fn in GRADMAG_MODES


def _unwrap_student_backbone(model: nn.Module) -> nn.Module:
    """PeftModel wraps the causal LM at base_model.model."""
    if hasattr(model, "base_model") and getattr(model.base_model, "model", None) is not None:
        return model.base_model.model
    return model


def get_last_layer_params(model: nn.Module, model_type: str) -> List[torch.nn.Parameter]:
    """Last transformer block only (tokenkit get_layer_n_mask(..., -1); gpt2: transformer.h[-1])."""
    backbone = _unwrap_student_backbone(model)
    if model_type in ("gpt2", "gptj"):
        layer_params = list(backbone.transformer.h[-1].parameters())
    elif hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        layer_params = list(backbone.model.layers[-1].parameters())
    elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layers"):
        layer_params = list(backbone.transformer.layers[-1].parameters())
    elif hasattr(backbone, "gpt_neox") and hasattr(backbone.gpt_neox, "layers"):
        layer_params = list(backbone.gpt_neox.layers[-1].parameters())
    elif model_type in ("tinyllama", "llama", "llama2", "mistral", "minicpm"):
        raise ValueError(
            f"Cannot resolve last-layer parameters for model_type={model_type!r} "
            f"(backbone={type(backbone).__name__}). Extend get_last_layer_params in code/alm_multitask.py."
        )
    else:
        raise ValueError(
            f"Cannot resolve last-layer parameters for model_type={model_type!r}. "
            "Extend get_last_layer_params in code/alm_multitask.py."
        )
    # LoRA / frozen base weights: autograd.grad requires every input to require grad.
    trainable = [p for p in layer_params if p.requires_grad]
    if not trainable:
        raise ValueError(
            f"No trainable parameters in last layer for model_type={model_type!r}. "
            "GradMag needs at least one trainable param in the last block."
        )
    return trainable


def compute_global_grad_norm(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    sq = sum(g.pow(2).sum() for g in grads if g is not None)
    return torch.sqrt(sq + 1e-12)


def gradmag_weights_from_task_grads(
    task_grads: List[List[torch.Tensor | None]],
    mode: str,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse global grad-norm weights (tokenkit approx_gradmag / preserve_mag).

    w_i = (1 / (||g_i|| + eps)); preserve_mag: w_i /= sum_j w_j.
    """
    grad_norms = torch.stack(
        [compute_global_grad_norm(g) for g in task_grads],
        dim=0,
    )
    inv_norms = 1.0 / (grad_norms + epsilon)
    if mode == "approx_gradmag_preserve_mag":
        inv_norms = inv_norms / inv_norms.sum()
    return inv_norms, grad_norms


def approximate_gradmag_weights(
    losses: Sequence[torch.Tensor],
    last_layer_params: Sequence[torch.nn.Parameter],
    mode: str,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GradMag weights via autograd.grad on the last layer only."""
    task_grads = []
    for loss in losses:
        grads = torch.autograd.grad(
            loss,
            last_layer_params,
            retain_graph=True,
            allow_unused=True,
        )
        task_grads.append(
            [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, last_layer_params)
            ]
        )
    return gradmag_weights_from_task_grads(task_grads, mode, epsilon=epsilon)


def aggregate_multitask_loss(
    losses: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    return sum(w * loss for w, loss in zip(weights, losses))
