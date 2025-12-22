"""
Methods for RL
"""

import torch
from transformers import PreTrainedModel
from typing import Callable, Literal, Optional

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: int,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.
    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    # calculate all rewards
    rewards = []
    batch_size = int(len(rollout_responses) // group_size)
    for (response, gt) in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(response, gt)["reward"])
    rewards = torch.tensor(rewards).reshape(batch_size, group_size)
    means = rewards.mean(dim=-1)
    advantages = rewards - means.unsqueeze(1)
    
    stds = None
    if normalize_by_std:
        stds = rewards.std(dim=-1).unsqueeze(1)
        advantages = advantages / (stds + advantage_eps)
    
    metadata = {
        "means": means,
        "stds": stds if normalize_by_std else 0,
        "maxes": rewards.max(dim=-1)[0],
        "mins": rewards.min(dim=-1)[0],
    }
    return advantages.view(-1), rewards.view(-1), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            scalar reward/advantage for each rollout response
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs for each token

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages

    loss = -torch.min(surr1, surr2)

    with torch.no_grad():
        clip_matrix = (surr2 < surr1)
        clip_fraction = clip_matrix.float().mean()
    
    metadata = {
        "clip_matrix": clip_matrix.detach(),
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "grpo_clip":
        assert cliprange is not None
        assert advantages is not None
        assert old_log_probs is not None
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    else:
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}

    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_sum = torch.sum(tensor * mask, dim=dim)
    mask_count = torch.sum(mask, dim=dim)
    result = masked_sum / mask_count
    return result

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    sequence_losses = masked_mean(loss, response_mask, dim=-1)
    raw_loss = torch.mean(sequence_losses)
    scaled_loss = raw_loss / gradient_accumulation_steps
    scaled_loss.backward()
    metadata["unscaled_loss"] = raw_loss.detach()
    metadata["gradient_accumulation_steps"] = gradient_accumulation_steps
    return scaled_loss.detach(), metadata

def train_grpo(
    policy: PreTrainedModel,
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1, # on-policy
    train_batch_size: int = 256, # on-policy
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    optimizer: torch.optim = None,
) -> None:
    """
    """
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
    