"""
SFT-specific functions
"""

import torch
from typing import Dict

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Implement a single micro-batch update for SFT, including cross-entropy loss, summing
    with a mask, and gradient scaling

    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding
    gradient_accumulation_steps Number of microbatches per optimizer step
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]]
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log
    """
    # Mask the log probs
    masked_log_probs = policy_log_probs * response_mask
    # Calculate loss (sum all log probs in sequence)
    sum_log_probs = torch.sum(masked_log_probs, dim=-1)
    # average over batch, then normalize
    loss = -torch.mean(sum_log_probs) / normalize_constant

    loss_to_optimize = loss / gradient_accumulation_steps

    loss_to_optimize.backward()

    metadata = {
        "loss": loss_to_optimize.detach(),
        "unscaled_loss": loss.detach()
    }
    return loss_to_optimize, metadata
