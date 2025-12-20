"""
Use SFT to finetune a model on reasoning traces.
"""

import json
import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import train_sft

app = typer.Typer()

# Load prompts and outputs
def load_prompts_and_outputs(
    data_path: str,
    max_examples: Optional[int] = None,
    filter_correct: bool = False
):
    """
    Load data from a jsonl file with keys "prompt" and "response"
    """
    prompts = []
    responses = []
    cur_examples = 0

    with open(data_path, "r") as f:
        for line in f:
            if max_examples and cur_examples > max_examples:
                break
            if not line.strip():
                continue
            item = json.loads(line)

            prompt = item.get("prompt")
            response = item.get("response")

            if prompt is None or response is None:
                continue

            if filter_correct:
                ground_truth = item.get("ground_truth")
                rewards = r1_zero_reward_fn(response, ground_truth)
                if rewards["reward"] == 0.0:
                    continue

            prompts.append(prompt)
            responses.append(response)
            cur_examples += 1

    return prompts, responses

# call run_sft with hyperparameter setups
@app.command()
def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    sft_data_path: str = "data/MATH/sft.jsonl",
    sft_examples: int = None,
    filter_correct: bool = False,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    output_dir: str = "results/baseline_results.jsonl",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    device_obj = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )
    prompts, outputs = load_prompts_and_outputs(
        sft_data_path,
        sft_examples,
        filter_correct
    )
    train_sft(
        model,
        tokenizer,
        optimizer,
        prompt_strs = prompts,
        output_strs = outputs,
        train_batch_size = batch_size,
        gradient_accumulation_steps = batch_size/2,
        device=device_obj,
        output_dir = output_dir,
    )

# evaluate on vLLM (use run_baseline script...)

if __name__ == "__main__":
    app()
