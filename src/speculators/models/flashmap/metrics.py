"""Losses and metrics of the two FlashMap training stages."""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from speculators.losses import LossConfig, dflash_loss_decay, kl_div_loss
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

__all__ = [
    "compute_draft_metrics",
    "compute_flow_loss",
    "compute_map_loss",
    "slot_loss_weights",
]

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}
_NUM_TAU_BINS = 4
_EPS = 1e-6


def slot_loss_weights(
    label_mask: torch.Tensor,  # [1, num_blocks * block_size]
    block_size: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot loss weights ``mask * exp(-k / gamma)`` for slot ``k`` of each
    block (DFlash's decay, with every slot a draft token), and the slot index."""
    pos_idx = torch.arange(label_mask.shape[1], device=label_mask.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)
    decay = dflash_loss_decay(pos_idx.float(), gamma, sample_from_anchor=True)
    return label_mask.float() * decay, pos_idx


def _counted(metrics: dict[str, Any], name: str, num, den) -> None:
    metrics[f"{name}_sum"] = num.detach().float()
    metrics[f"{name}_total"] = torch.as_tensor(den, device=num.device).detach().float()


def compute_flow_loss(
    logits: torch.Tensor,  # [1, slots, vocab]
    labels: torch.Tensor,  # [1, slots]
    label_mask: torch.Tensor,  # [1, slots]
    taus: torch.Tensor,  # [num_blocks], the schedule's progress variable
    block_size: int,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    """Stage 1: cross-entropy of the denoiser against the clean tokens.

    The loss is dominated by how noisy a block was, so it is also reported per
    quartile of ``tau`` (0: pure noise, 1: tokens nearly resolved).
    """
    weights, pos_idx = slot_loss_weights(label_mask, block_size, gamma)
    token_loss = F.cross_entropy(
        logits[0].float(), labels[0], reduction="none"
    ).unsqueeze(0)
    loss = (token_loss * weights).sum() / (weights.sum() + _EPS)

    metrics: dict[str, Any] = {}
    _counted(metrics, "loss", loss, 1.0)
    with torch.no_grad():
        tau_bins = (taus * _NUM_TAU_BINS).long().clamp(0, _NUM_TAU_BINS - 1)
        tau_bins = tau_bins.repeat_interleave(block_size).unsqueeze(0)
        for bin_idx in range(_NUM_TAU_BINS):
            bin_weights = weights * (tau_bins == bin_idx)
            _counted(
                metrics,
                f"loss_tau{bin_idx}",
                (token_loss * bin_weights).sum(),
                bin_weights.sum(),
            )
        correct, total = compute_accuracy_multi_step(
            logits.argmax(dim=-1), labels, label_mask, pos_idx, block_size
        )
        _counted(metrics, "denoise_acc", correct.sum(), total.sum())
    return loss, metrics


def compute_map_loss(
    logits: torch.Tensor,  # [1, slots, vocab], the map's prediction
    target_log_probs: torch.Tensor,  # [1, slots, vocab], what it is trained toward
    labels: torch.Tensor,  # [1, slots]
    label_mask: torch.Tensor,  # [1, slots]
    groups: dict[str, torch.Tensor],  # name -> [1, slots] bool
    block_size: int,
    gamma: float,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Stage 2: divergence of the map's distribution from its target (the frozen
    flow on the diagonal, the semigroup target elsewhere), with the speculators
    loss functions (KL by default, the paper's choice). The target is passed as
    log-probabilities, which the loss functions read as logits."""
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    weights, pos_idx = slot_loss_weights(label_mask, block_size, gamma)
    slot_loss = sum(
        weight * fn(logits, target_log_probs) for fn, weight in loss_config.values()
    )
    loss = (slot_loss * weights).sum() / (weights.sum() + _EPS)  # type: ignore[union-attr]

    metrics: dict[str, Any] = {}
    _counted(metrics, "loss", loss, 1.0)
    with torch.no_grad():
        for name, group in groups.items():
            group_weights = weights * group
            _counted(
                metrics,
                f"loss_{name}",
                (slot_loss * group_weights).sum(),  # type: ignore[union-attr]
                group_weights.sum(),
            )
        correct, total = compute_accuracy_multi_step(
            logits.argmax(dim=-1), labels, label_mask, pos_idx, block_size
        )
        _counted(metrics, "denoise_acc", correct.sum(), total.sum())
    return loss, metrics


@torch.no_grad()
def compute_draft_metrics(
    draft_logits: torch.Tensor,  # [1, slots, vocab], one-step drafts
    target_ids: torch.Tensor,  # [1, slots], the verifier's greedy tokens
    label_mask: torch.Tensor,  # [1, slots]
    block_size: int,
) -> dict:
    """Greedy agreement of one-step drafts with the verifier, per slot and as the
    expected accepted length (leading run of matches plus the bonus token)."""
    pos_idx = (
        torch.arange(label_mask.shape[1], device=label_mask.device) % block_size
    ).unsqueeze(0)
    pred_ids = draft_logits.argmax(dim=-1)
    correct, total = compute_accuracy_multi_step(
        pred_ids, target_ids, label_mask, pos_idx, block_size
    )
    metrics: dict[str, Any] = {
        "full_acc_sum": correct.sum(),
        "full_acc_total": total.sum(),
    }
    for pos in range(block_size):
        metrics[f"position_{pos}_acc_sum"] = correct[pos]
        metrics[f"position_{pos}_acc_total"] = total[pos]
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(-1, block_size),
        label_mask.to(torch.bool).reshape(-1, block_size),
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total
    return metrics
