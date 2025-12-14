"""
SFT and RL Helper Methods
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from transformers import PreTrainedTokenizer, PreTrainedModel

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
    pass