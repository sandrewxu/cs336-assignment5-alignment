"""
SFT-specific functions
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.functional import (
    batch_generator,
    get_response_log_probs,
    masked_normalize,
)

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward and backward pass on a microbatch.
    
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding
    gradient_accumulation_steps Number of microbatches per optimizer step
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it
        metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log
    """
    raw_loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=-1,
    )
    scaled_loss = raw_loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {
        "unscaled loss": raw_loss.item(),
        "gradient accumulation steps": gradient_accumulation_steps,
    }
    return scaled_loss.item(), metadata

def train_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    prompt_strs: list[str],
    output_strs: list[str],
    train_batch_size: int = 64,
    gradient_accumulation_steps: int = 32,
    device: torch.device = None,
    output_dir: str = None,
    max_gradient: float = 1.0,
) -> None:
    """
    Finetune a model on a sequence of prompt and output strings.
    Save to output_dir after finished.
    """
    microbatch_size = train_batch_size // gradient_accumulation_steps
    num_batches = len(prompt_strs) // microbatch_size
    model.train()

    data_loader = batch_generator(prompt_strs, output_strs, tokenizer, microbatch_size, device)

    for idx, (inputs, labels, response_mask) in enumerate(data_loader):
        policy_log_probs = get_response_log_probs(model, inputs, labels)["log_probs"]
        loss, metadata = sft_microbatch_train_step(
            policy_log_probs,
            response_mask,
            gradient_accumulation_steps,
        )

        if (idx + 1) % gradient_accumulation_steps == 0:
            # clip gradients at `max_gradient`
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient)
            # update weights and zero gradients every `gradient_accumulation_steps` batches
            optimizer.step()
            optimizer.zero_grad()

    # Save to output dir
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
