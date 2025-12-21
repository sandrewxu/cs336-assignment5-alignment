"""
Functional methods for SFT and RL
Takes in and outputs tensors
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask
    that is 1 for the response tokens and 0 for other tokens 
    (prompt or padding).

    Args:
    prompt_strs: list[str] List of prompt strings
    output_strs: list[str] List of output strings
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization

    Returns:
    dict[str, torch.Tensor].
        input_ids
        labels
        response_mask
    """
    # Tokenize prompts and outputs together, pad to longest, return a Tensor and a mask
    outputs = tokenizer(
        prompt_strs,
        output_strs,
        padding='longest',
        padding_side='right',
        return_tensors='pt',
    )
    output_tokens = outputs['input_ids']
    padding_mask = outputs['attention_mask'] # (batch, max_len) for prompt or output str, 0 for padding

    # Tokenize JUST the prompts to find their length and add to the mask
    prompt_encodings = tokenizer(
        prompt_strs,
        padding='max_length',
        max_length=len(padding_mask[0]),
        padding_side='right',
        return_tensors='pt',
    )
    # input_mask - attention_mask = prompt_or_padding_mask
    prompt_or_padding_mask = padding_mask - prompt_encodings['attention_mask']

    return {
        "input_ids": output_tokens[:, :-1],
        "labels": output_tokens[:, 1:],
        "response_mask": prompt_or_padding_mask[:, 1:]
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of next-token predictions 
    (i.e., entropy over the vocabulary dimension)

    Args:
    logits: torch.Tensor Tensor of shape (batch_size, 
    sequence_length, vocab_size) containing unnormalized logits.

    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy 
    for each next-token prediction.
    """
    # Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    probs = torch.softmax(logits, dim=-1) # (b, s, 1)
    entropies = torch.logsumexp(logits, dim=-1) - torch.sum(logits * probs, dim=-1) # (b, s) - (b, s)
    return entropies

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Gets per-token conditional log-probabilities (given previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution

    Args:
    model: PreTrainedModel HuggingFace model used for scoring (placed on the 
    correct device and inference mode if gradients should not be computed)
    input_ids: torch.Tensor shape (b, s), concatenated prompt + response tokens
    as produced by your tokenization method
    labels: torch.Tensor shape (b, s), labels as produced by your tokenization
    method
    return_token_entropy: bool If True, also return per-token entropy by calling
    compute_entropy

    Returns:
    dict[str, torch.Tensor].
        "log_probs" shape (b, s), conditional log_probs
        "token_entropy" optional, shape (b, s), per-token entropy for each position
    """
    logits = model(input_ids).logits # (b, s, v)
    # calculate log_probs
    log_probs_all = F.log_softmax(logits, dim=-1) # (b, s, v)
    # index labels
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1) # (b s) -> (b s 1)
    ) # (b s 1)
    log_probs = log_probs.squeeze(-1) # (b s 1) -> (b s)

    results = {"log_probs": log_probs}
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering
    only those elements where mask == 1.

    Args:
    tensor: torch.Tensor The tensor to sum and normalize
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in sum
    normalize_constant: float the constant to divide by for normalization
    dim: int | None the dimension to sum along before normalization. If None, sum over
    all dimensions

    Returns:
    torch.Tensor the normalized sum
    """
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def batch_generator(
    prompts: list[str],
    outputs: list[str],
    tokenizer: PreTrainedTokenizer,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    """
    Generator function to generate batches of 
    prompt, output, and masks for training
    """
    new_prompts = prompts * epochs
    new_outputs = outputs * epochs
    for i in range(0, len(new_prompts), batch_size):
        batch = tokenize_prompt_and_output(
            new_prompts[i : i + batch_size],
            new_outputs[i : i + batch_size],
            tokenizer,
        )

        yield (
            batch['input_ids'].to(device),
            batch['labels'].to(device),
            batch['response_mask'].to(device),
        )
