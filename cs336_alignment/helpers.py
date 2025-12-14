"""
SFT and RL Helper Methods
"""

import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer

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
