"""
SFT and RL Helper Methods
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable
from transformers import PreTrainedTokenizer, PreTrainedModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding)

    Tokenizes prompts and outputs, concatenates them, and creates a training mask
    """

    full_input_ids = []
    full_masks = []

    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize Prompt with [BOS]
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        # Tokenize Output without [BOS]
        output_ids = tokenizer(output, add_special_tokens=False).input_ids
        sequence_ids = prompt_ids + output_ids
        mask = [0] * len(prompt_ids) + [1] * len(output_ids)

        full_input_ids.append(sequence_ids)
        full_masks.append(mask)
    
    # Padding
    max_len = max(len(ids) for ids in full_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_masks = []

    for ids, mask in zip(full_input_ids, full_masks):
        pad_len = max_len - len(ids)

        # apply right padding
        padded_ids = ids + [pad_id] * pad_len
        padded_mask = mask + [0] * pad_len

        padded_input_ids.append(padded_ids)
        padded_masks.append(padded_mask)
    
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, dtype=torch.bool)

    inputs = input_ids_tensor[:, :-1]
    labels = input_ids_tensor[:, 1:]
    response_mask = mask_tensor[:, 1:]
    
    return {
        "input_ids": inputs,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits
    Returns:
    torch.Tensor shape (batch_size, sequence_length). The entropy for each next-token prediction
    """
    # Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    # for only (vocab_size)

    # Alternative formula for entropy: logsumexp(logits) - sum p_i x_i
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    expected_logit = torch.sum(probs * logits, dim=-1)
    entropy = log_z - expected_logit
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Computes per-token log-probabilities for the given labels and optionally the entropy.
    """
    # Forward pass (pass input_ids into model to get unnormalized logits)
    # logits shape: (batch_size, sequence_length, vocab_size)
    outputs = model(input_ids)
    logits = outputs.logits

    all_log_probs = F.log_softmax(logits, dim=-1)

    gathered_log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    )
    log_probs = gathered_log_probs.squeeze(-1)
    result = {
        "log_probs": log_probs
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering
    only those elements where mask == 1
    """
    masked_tensor = tensor * mask
    masked_sum = torch.sum(masked_tensor, dim=dim)
    return masked_sum / normalize_constant

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

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truths: List[str],
    step: int,
    reward_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    max_new_tokens: int = 1024,
    **kwargs
) -> None:
    """
    Generate responses from the model for given prompts, compute rewards if a reward function
    is provided, and log the results.
    
    Args:
        model: The language model to generate from
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        ground_truths: List of ground truth answer strings
        step: Current training step (for logging)
        reward_fn: Optional function that takes (response, ground_truth) and returns a dict
                   with reward metrics (e.g., {"reward": float, "format_reward": float, ...})
        max_new_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments to pass to model.generate()
    """
    model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            **kwargs
        )
    
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    rewards = []
    correct_flags = []
    format_rewards = []
    answer_rewards = []

    if reward_fn:
        for response, ground_truth in zip(responses, ground_truths):
            metrics = reward_fn(response, ground_truth)
            rewards.append(metrics.get("reward", 0.0))
            correct_flags.append(metrics.get("reward", 0.0) > 0.5)  # Consider correct if reward > 0.5
            format_rewards.append(metrics.get("format_reward", 0.0))
            answer_rewards.append(metrics.get("answer_reward", 0.0))

    response_lens = [len(t) for t in generated_tokens]
    avg_len = np.mean(response_lens)

    # Prepare metrics for logging
    log_metrics = {
        "generation/num_generations": len(responses),
        "generation/avg_response_length": avg_len,
    }
    
    if reward_fn:
        avg_reward = np.mean(rewards) if rewards else 0.0
        accuracy = np.mean(correct_flags) if correct_flags else 0.0
        avg_format_reward = np.mean(format_rewards) if format_rewards else 0.0
        avg_answer_reward = np.mean(answer_rewards) if answer_rewards else 0.0
        
        log_metrics.update({
            "generation/avg_reward": avg_reward,
            "generation/accuracy": accuracy,
            "generation/avg_format_reward": avg_format_reward,
            "generation/avg_answer_reward": avg_answer_reward,
        })

    # Log to wandb if available and initialized
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(log_metrics, step=step)

    # Log the results to console
    print(f"\n{'='*60}")
    print(f"Generation Log - Step {step}")
    print(f"{'='*60}")
    print(f"Number of generations: {len(responses)}")
    print(f"Average response length: {avg_len:.2f} tokens")
    
    if reward_fn:
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Accuracy (reward > 0.5): {accuracy:.4f}")
        print(f"Average format reward: {avg_format_reward:.4f}")
        print(f"Average answer reward: {avg_answer_reward:.4f}")
        
        # Print a few example generations
        print(f"\nExample generations (showing first 3):")
        for i in range(min(3, len(responses))):
            print(f"\n  Example {i+1}:")
            print(f"    Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"    Prompt: {prompts[i]}")
            print(f"    Response: {responses[i][:200]}..." if len(responses[i]) > 200 else f"    Response: {responses[i]}")
            print(f"    Ground truth: {ground_truths[i]}")
            if reward_fn:
                print(f"    Reward: {rewards[i]:.4f} (format: {format_rewards[i]:.4f}, answer: {answer_rewards[i]:.4f})")
    
    print(f"{'='*60}\n")

    model.train()
