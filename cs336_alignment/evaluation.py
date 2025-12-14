"""
Evaluation helper functions
"""

import json
import numpy as np
import os
import statistics
import torch
from typing import List, Callable, Dict, Any, Optional
from vllm import LLM, SamplingParams
import wandb

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # Generate text
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Compute rewards
    results = []
    total_metrics = {}
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        metrics = reward_fn(generated_text, ground_truths[i])
        
        entry = {
            "prompt": prompts[i],
            "generated_text": generated_text,
            "ground_truth": ground_truths[i],
            **metrics
        }

        results.append(entry)

        for key, val in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += val
    
    # Calculate averages
    num_examples = len(results)
    avg_metrics = {k: v / num_examples for k, v in total_metrics.items()}

    # Print summary
    print("-" * 40)
    print(f"Evaluation complete on {num_examples} examples.")
    for k, v in avg_metrics.items():
        print(f"Average {k}: {v:.4f}")
    print("-" * 40)

    # Serialize to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def log_generations(
    llm: LLM,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Optional[Callable[[str, str], Dict[str, float]]],
    sampling_params: SamplingParams,
    step: int,
    wandb_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generate responses from the model for given prompts, compute rewards if a reward function
    is provided, and log the results.

    Calculates:
    - Rewards (total, format, answer)
    - Response lengths (global, correct-only, incorrect-only)
    - Approximate token entropy

    Logs a visual table and scalar metrics to WandB
    """

    # Prepare sampling params
    eval_params = sampling_params.clone()
    if eval_params.logprobs is None:
        eval_params.logprobs = 10
    
    print(f"DEBUG: Generating {len(prompts)} validation samples at step {step}...")
    outputs = llm.generate(prompts, eval_params)

    table_rows = []
    
    stats = {
        "rewards_total": [],
        "rewards_format": [],
        "rewards_answer": [],
        "lengths_all": [],
        "lengths_correct": [],
        "lengths_incorrect": [],
        "entropies": []
    }

    for i, output in enumerate(outputs):
        prompt = prompts[i]
        gt = ground_truths[i]

        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids

        # Rewards
        metrics = reward_fn(generated_text, gt)
        r_total = metrics["reward"]
        r_fmt = metrics["format_reward"]
        r_ans = metrics["answer_reward"]
        stats["rewards_total"].append(r_total)
        stats["rewards_format"].append(r_fmt)
        stats["rewards_answer"].append(r_ans)

        # Answer lengths
        length = len(token_ids)
        stats["lengths_all"].append(length)

        if r_ans == 1.0:
            stats["lengths_correct"].append(length)
        else:
            stats["lengths_incorrect"].append(length)

        # Entropy
        # vLLM returns a list of dicts: [{token_id: logprob, ...}, ...] per step
        seq_entropies = []
        if output.outputs[0].logprobs:
            for step_logprobs in output.outputs[0].logprobs:
                # extract logprobs of top-k tokens
                # Structure is {token_id: LogprobObject(logprob=float, ...)}
                lps = np.array([obj.logprob for obj in step_logprobs.values()])

                # convert to prob -> normalize -> entropy
                probs = np.exp(lps)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-9))
                    seq_entropies.append(entropy)

        avg_entropy = statistics.mean(seq_entropies) if seq_entropies else 0.0
        stats["entropies"].append(avg_entropy)

        table_rows.append([
            prompt[:1000], 
            generated_text[:1000], 
            gt, 
            r_total, 
            r_fmt, 
            r_ans, 
            length, 
            avg_entropy
        ])

    def safe_mean(data):
        return statistics.mean(data) if data else 0.0

    metrics = {
        "val/avg_reward": safe_mean(stats["rewards_total"]),
        "val/avg_format_reward": safe_mean(stats["rewards_format"]),
        "val/avg_answer_reward": safe_mean(stats["rewards_answer"]),
        "val/avg_length": safe_mean(stats["lengths_all"]),
        "val/avg_length_correct": safe_mean(stats["lengths_correct"]),
        "val/avg_length_incorrect": safe_mean(stats["lengths_incorrect"]),
        "val/avg_entropy": safe_mean(stats["entropies"]),
    }

    if wandb.run is not None:
        cols = ["Prompt", "Response", "GT", "Total R", "Fmt R", "Ans R", "Len", "Entropy"]
        wandb_table = wandb.Table(columns=cols, data=table_rows)

        log_dict = {"val/generations": wandb_table}
        log_dict.update(metrics)

        wandb.log(log_dict, step=wandb_step if wandb_step else step)
    
    # Print a quick summary to console
    print(f"Step {step} Eval: Acc={metrics['val/avg_answer_reward']:.2%}, "
        f"Entropy={metrics['val/avg_entropy']:.2f}")
    
    return metrics
