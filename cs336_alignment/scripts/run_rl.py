"""
Use RL (GRPO) to finetune a model on a test set
"""

import os
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
import wandb

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluation import init_vllm
from cs336_alignment.rl import train_grpo, evaluate_and_log
from cs336_alignment.scripts.run_baseline import load_prompts_and_gts

app = typer.Typer()

@app.command()
def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/MATH/train.jsonl",
    eval_data_path: str = "data/MATH/validation.jsonl",
    prompt_file: str = "cs336_alignment/prompts/r1_zero.prompt",
    n_grpo_steps: int = 200,
    group_size: int = 8,
    rollout_batch_size: int = 256,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    epochs_per_rollout_batch: int = 1,
    learning_rate: float = 1e-5,
    loss_type: str = "reinforce_with_baseline",
    cliprange: float = 0.2,
    use_std_normalization: bool = True,
    eval_interval: int = 10,
    output_dir_base: str = "/gpfs/radev/home/ax46/scratch/A5/rl",
    rl_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
):
    """
    Run GRPO training on MATH dataset
    """
    # Directory mapping
    run_name = f"grpo_{loss_type}_G{group_size}_rb_{rollout_batch_size}_lr{learning_rate}"
    output_dir = os.path.join(output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(rl_device)

    wandb.init(
        entity="andrew-xu",
        project="cs336-a5-rl",
        name=run_name,
        config={
            "model_path": model_path,
            "n_grpo_steps": n_grpo_steps,
            "group_size": group_size,
            "rollout_batch_size": rollout_batch_size,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "learning_rate": learning_rate,
            "loss_type": loss_type,
            "cliprange": cliprange,
        }
    )

    # Load data
    print("Loading training data...")
    train_prompts, train_ground_truths = load_prompts_and_gts(train_data_path, prompt_file)
    print(f"Loaded {len(train_prompts)} training prompts.")

    print("Loading eval data...")
    eval_prompts, eval_ground_truths = load_prompts_and_gts(eval_data_path, prompt_file)
    print(f"Loaded {len(eval_prompts)} eval prompts.")

    # Initialize policy model
    print("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Initialize vLLM for rollout generation
    print("Initializing vLLM...")
    vllm_model = init_vllm(
        model_id=model_path,
        device=vllm_device,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    # Run GRPO training
    train_grpo(
        policy=policy,
        tokenizer=tokenizer,
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=train_prompts,
        ground_truths=train_ground_truths,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        cliprange=cliprange,
        rl_device=device,
        eval_prompts=eval_prompts[:1024],
        eval_ground_truths=eval_ground_truths[:1024],
        eval_interval=eval_interval,
    )

    # Evaluate final model
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_dir_base="results/grpo"
    eval_dir = os.path.join(eval_dir_base, run_name)
    os.makedirs(eval_dir, exist_ok=True)
    eval_output_file = os.path.join(eval_dir, "eval.jsonl")
    evaluate_and_log(policy, vllm_model, r1_zero_reward_fn, eval_prompts, eval_ground_truths, eval_sampling_params, n_grpo_steps, eval_output_file)

    # Save final model
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    wandb.finish()

if __name__ == "__main__":
    app()
