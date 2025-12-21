"""
Use SFT to finetune a model on reasoning traces.
"""

import json
import os
import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from vllm import LLM, SamplingParams
import wandb

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluation import evaluate_vllm, init_vllm, load_policy_into_vllm_instance
from cs336_alignment.scripts.run_baseline import load_prompts_and_gts
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
    output_dir_base: str = "/gpfs/radev/home/ax46/scratch/A5/sft",
    device: str = "cuda:0",
):
    # Automatic directory naming
    example_str = f"n{sft_examples}" if sft_examples else "full"
    filter_str = "correct" if filter_correct else "all"

    run_name = f"sft_{example_str}_{filter_str}_lr{learning_rate}_batch_n{batch_size}"
    output_dir = os.path.join(output_dir_base, run_name)

    print(f"Training with: {example_str} examples, filter={filter_correct}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Output directory: {output_dir}")

    wandb.init(
        project="cs336-a5-sft",
        config={
            "model_path": model_path,
            "sft_examples": sft_examples if sft_examples else "full",
            "filter_correct": filter_correct,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_dir": output_dir,
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    device_obj = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_obj)
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

    # Train
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

    # Evaluate on GPU 1
    if torch.cuda.device_count() > 1:
        print("Starting evaluation on cuda:1...")

        # Define paths
        eval_data_path = "data/MATH/validation.jsonl"
        prompt_file = "cs336_alignment/prompts/r1_zero.prompt"
        eval_output_file = f"results/sft/{run_name}.jsonl"

        eval_prompts, eval_ground_truths = load_prompts_and_gts(eval_data_path, prompt_file)
        print(f"Loaded {len(eval_prompts)} evaluation examples.")

        llm = init_vllm(
            model_id = model_path,
            device="cuda:1",
            seed=42,
            gpu_memory_utilization=0.85,
        )
        print("Loading trained policy into vLLM...")
        load_policy_into_vllm_instance(model, llm)

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )

        evaluate_vllm(
            vllm_model=llm,
            reward_fn=r1_zero_reward_fn,
            prompts=eval_prompts,
            ground_truths=eval_ground_truths,
            eval_sampling_params=sampling_params,
            output_file=eval_output_file,
        )

    wandb.finish()

if __name__ == "__main__":
    app()
