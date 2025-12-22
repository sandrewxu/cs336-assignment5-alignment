"""
Use Expert Iteration (EI) to finetune a model on a dataset.
"""

import math
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
from typing import Optional
from vllm import SamplingParams
import wandb

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluation import evaluate_vllm, init_vllm, load_policy_into_vllm_instance, log_generations
from cs336_alignment.sft import train_sft
from cs336_alignment.scripts.run_baseline import load_prompts_and_gts
from cs336_alignment.scripts.run_sft import load_prompts_and_outputs

app = typer.Typer()

@app.command()
def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/MATH/train.jsonl",
    eval_data_path = "data/MATH/validation.jsonl",
    prompt_file = "cs336_alignment/prompts/r1_zero.prompt",
    n_ei_steps: int = 5,
    G: int = 64,
    ei_batch_size: int = 512,
    sft_epochs: int = 1,
    sft_batch_size: int = 64,
    learning_rate: float = 1e-4,
    logprobs: Optional[int] = None,
    output_dir_base: str = "/gpfs/radev/home/ax46/scratch/A5/ei",
    sft_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
):
    """
    Run expert iteration on a dataset.
    """
    # Directory naming, setup sft_device
    run_name = f"ei_G{G}_eibatch{ei_batch_size}_sftepochs_{sft_epochs}"
    global_output_dir = os.path.join(output_dir_base, run_name)
    os.makedirs(global_output_dir, exist_ok=True)
    sft_device_obj = torch.device(sft_device)

    wandb.init(
        entity="andrew-xu",
        project="cs336-a5-ei",
        name=run_name,
        config={
            "model_path": model_path,
            "n_ei_steps": n_ei_steps,
            "G": G,
            "ei_batch_size": ei_batch_size,
            "sft_epochs": sft_epochs,
            "sft_batch_size": sft_batch_size,
            "learning_rate": learning_rate,
            "output_dir": global_output_dir,
        }
    )

    # load eval and training data
    eval_prompts, eval_ground_truths = load_prompts_and_gts(eval_data_path, prompt_file)
    train_prompts, train_ground_truths = load_prompts_and_gts(train_data_path, prompt_file)

    # Initialize Model, Tokenizer, Optimizer, vLLM
    print("Initializing SFT model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(sft_device_obj)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Initializing vLLM...")
    llm = init_vllm(
        model_id = model_path,
        device=vllm_device,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    multiple = math.ceil(n_ei_steps * ei_batch_size / len(train_prompts))
    full_train_prompts = train_prompts * multiple
    full_train_ground_truths = train_ground_truths * multiple

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    train_sampling_params = SamplingParams(
        n=G,
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        logprobs=logprobs,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    for i in range(n_ei_steps):
        print(f"=== Starting Iteration {i+1}/{n_ei_steps} ===")
        print("Syncing policy weights to vLLM...")
        load_policy_into_vllm_instance(model, llm)
        # Generate training data (G rollouts)
        start = i * ei_batch_size
        end = start + ei_batch_size
        batch_prompts = full_train_prompts[start:end]
        batch_gts = full_train_ground_truths[start:end]
        train_output_file = f"results/ei/{run_name}/train/iter_{i+1}.jsonl"
        print(f"Logging generations for {len(batch_prompts)} prompts")

        generation_results = log_generations(
            llm,
            r1_zero_reward_fn,
            batch_prompts,
            batch_gts,
            train_sampling_params,
            train_output_file,
        )
        samples = generation_results[:-1]
        gen_stats = generation_results[-1]
        wandb.log({
            "ei/avg_response_length": gen_stats["avg_response_length"],
            "ei/avg_response_length_correct": gen_stats["avg_response_length_correct"],
            "ei/avg_response_length_incorrect": gen_stats["avg_response_length_incorrect"],
            "ei/accuracy": gen_stats["accuracy"],
            "ei/avg_token_entropy": gen_stats["avg_token_entropy"],
            "ei/step": i+1,
        })

        sft_prompts = []
        sft_outputs = []
        for item in samples:
            if item.get("reward", 0.0) == 1.0:
                sft_prompts.append(item["prompt"])
                sft_outputs.append(item["generated_text"])

        if len(sft_prompts) == 0:
            print("Warning: No correct samples generated. Skipping SFT step.")
            continue
        else:
            print(f"Loaded {len(sft_prompts)} training examples.")

        # Train SFT (updates model in-place)
        iter_output_dir = os.path.join(global_output_dir, f"iter_{i+1}")
        os.makedirs(iter_output_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95)
        )

        train_sft(
            model,
            tokenizer,
            optimizer,
            prompt_strs = sft_prompts,
            output_strs = sft_outputs,
            epochs = sft_epochs,
            train_batch_size = sft_batch_size,
            gradient_accumulation_steps = sft_batch_size // 2,
            device=sft_device_obj,
            output_dir = iter_output_dir, # Saves checkpoint here
        )

    # Evaluate against validation set
    eval_output_file = f"results/ei/{run_name}/eval.jsonl"
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=eval_prompts,
        ground_truths=eval_ground_truths,
        eval_sampling_params=eval_sampling_params,
        output_file=eval_output_file,
    )
    wandb.log({
        "eval/format_reward": metrics.get("format_reward", 0.0),
        "eval/reward": metrics.get("reward", 0.0),
    })

if __name__ == "__main__":
    app()
