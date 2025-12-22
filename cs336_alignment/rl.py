"""
Methods for RL
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Literal, Optional
from vllm import LLM, SamplingParams
import wandb

from cs336_alignment.evaluation import load_policy_into_vllm_instance, evaluate_vllm
from cs336_alignment.functional import tokenize_prompt_and_output, get_response_log_probs

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
        "mean": means.mean(),
        "std": stds.mean() if normalize_by_std else 0,
        "max": rewards.max(),
        "min": rewards.min(),
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

    with torch.inference_mode():
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
    tokenizer: PreTrainedTokenizer,
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    max_gradient: float = 1.0,
    rl_device: str = "cuda:0",
    eval_prompts: list[str] = None,
    eval_ground_truths: list[str] = None,
    eval_interval: int = 10,
) -> None:
    """
    GRPO training loop.

    Requires
    1. train_batch_size // gradient_accumulation_steps == 0
    2. rollout_batch_size // group_size == 0 (n_prompts * group_size)
    3. train_batch_size >= group_size

    # TODO: add more details.
    """
    # Validate batch sizes
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Sampling params for vLLM
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    eval_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Track prompt index for cycling
    prompt_idx = 0

    for step in range(n_grpo_steps):
        print(f"=== GRPO Step {step + 1}/{n_grpo_steps} ===")

        # Sample batch of questions D_b
        batch_prompts = []
        batch_ground_truths = []
        end_idx = prompt_idx + n_prompts_per_rollout_batch
        if end_idx >= len(prompts):
            batch_prompts.extend(prompts[prompt_idx:])
            batch_ground_truths.extend(ground_truths[prompt_idx:])
            # need `end_idx - len(prompts) + 1` more items
            prompt_idx = end_idx - len(prompts) + 1
            batch_prompts.extend(prompts[:prompt_idx])
            batch_ground_truths.extend(ground_truths[:prompt_idx])
        else:
            batch_prompts.extend(prompts[prompt_idx:end_idx])
            batch_ground_truths.extend(ground_truths[prompt_idx:end_idx])
            prompt_idx = end_idx

        # Sync policy weights to vLLM
        load_policy_into_vllm_instance(policy, vllm_model)

        # Sample G outputs per question, flatten rollouts (n_prompts * group_size,)
        outputs = vllm_model.generate(batch_prompts, sampling_params)
        rollout_responses = []
        repeated_prompts = []
        repeated_ground_truths = []
        for i, output in enumerate(outputs):
            for completion in output.outputs:
                rollout_responses.append(completion.text)
                repeated_prompts.append(batch_prompts[i])
                repeated_ground_truths.append(batch_ground_truths[i])

        # Compute rewards & advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        advantages = advantages.unsqueeze(1) # (..., 1)
        raw_rewards = raw_rewards.unsqueeze(1) # (..., 1)

        # Log reward stats
        print(f"  Mean reward: {reward_metadata["mean"]}")
        if wandb.run is not None:
            wandb.log({
                "grpo/reward_mean": reward_metadata["mean"],
                "grpo/reward_std": reward_metadata["std"],
                "grpo/reward_max": reward_metadata["max"],
                "grpo/reward_min": reward_metadata["min"],
                "grpo/step": step + 1,
            })

        # Tokenize all rollouts
        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(rl_device)
        labels = tokenized["labels"].to(rl_device)
        response_mask = tokenized["response_mask"].to(rl_device)
        advantages = advantages.to(rl_device)
        raw_rewards = raw_rewards.to(rl_device)

        # Compute old log probs once (for GRPO-Clip)
        if loss_type == "grpo_clip":
            policy.eval()
            with torch.inference_mode():
                old_log_probs_result = get_response_log_probs(policy, input_ids, labels)
                old_log_probs = old_log_probs_result["log_probs"]
            policy.train()
        else:
            old_log_probs = None

        # Inner training loop
        for _ in range(epochs_per_rollout_batch):
            # Shuffle indices for this epoch
            perm = torch.randperm(rollout_batch_size)

            for micro_step in range(0, rollout_batch_size, micro_train_batch_size):
                # Get microbatch data
                idx = perm[micro_step : micro_step + micro_train_batch_size]
                mb_input_ids = input_ids[idx]
                mb_labels = labels[idx]
                mb_response_mask = response_mask[idx]
                mb_advantages = advantages[idx]
                mb_raw_rewards = raw_rewards[idx]
                mb_old_log_probs = old_log_probs[idx] if old_log_probs is not None else None

                # Forward pass for current policy log probs
                policy_log_probs_result = get_response_log_probs(
                    policy, mb_input_ids, mb_labels, return_token_entropy=True
                )
                policy_log_probs = policy_log_probs_result["log_probs"]

                # Compute loss and backward
                _, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )

                if (micro_step // micro_train_batch_size + 1) % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=max_gradient
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    if wandb.run is not None:
                        log_dict = {
                            "train/loss": metadata["unscaled_loss"].item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/avg_token_entropy": policy_log_probs_result["token_entropy"].mean(),
                            "train/step": micro_step // micro_train_batch_size + 1,
                        }
                        if "clip_fraction" in metadata:
                            log_dict["train/clip_fraction"] = metadata["clip_fraction"].item()
                        wandb.log(log_dict)

        if eval_prompts is not None and (step + 1) % eval_interval == 0 and (step + 1) < n_grpo_steps:
            evaluate_and_log(policy, vllm_model, reward_fn, eval_prompts, eval_ground_truths, eval_sampling_params, step + 1, None)

def evaluate_and_log(
    policy: PreTrainedModel,
    vllm_model: LLM,
    reward_fn: Callable[[str, str], [str, float]],
    eval_prompts: list[str],
    eval_ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    step: int,
    output_file: Optional[str],
):
    """
    Wrapper around evaluate_llm with wandb.
    """
    load_policy_into_vllm_instance(policy, vllm_model)
    avg_metrics = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=reward_fn,
        prompts=eval_prompts,
        ground_truths=eval_ground_truths,
        eval_sampling_params=eval_sampling_params,
        output_file=output_file,
    )
    if wandb.run is not None:
        wandb.log({
            "eval/avg_format_reward": avg_metrics["format_reward"],
            "eval/answer_reward": avg_metrics["answer_reward"],
            "eval/reward": avg_metrics["reward"],
            "eval/step": step,
        })
