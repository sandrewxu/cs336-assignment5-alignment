"""
Evaluation helper functions
"""

from collections.abc import Callable
import json
import numpy as np
import os
import torch
from transformers import PreTrainedModel
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    output_file: str,
) -> dict[str, float]:
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

    return avg_metrics


def log_generations(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    output_file: str,
):
    """
    Given a batch of inputs with ground truths, for each input, log
    1. the input
    2. the response
    3. the ground truth answer
    4. reward information (format, answer, total)
    5. average token entropy of the response
    6. response length

    globally, log average response length, length of correct responses,
    and length of incorrect responses.

    Serialize to disk.
    """
    # Modify sample_log_probs if not already
    # if eval_sampling_params.logprobs is None:
    #     eval_sampling_params.logprobs = -1 # return all vocab_size logprobs

    # Generate text
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Compute rewards
    results = []
    total_metrics = {
        "response_length": 0,
        "response_length_correct": 0,
        "response_length_incorrect": 0,
        "num_correct": 0,
        "token_entropy": 0.0,
    }

    for i, output in enumerate(outputs):
        for completion in output.outputs:
            generated_text = completion.text
            metrics = reward_fn(generated_text, ground_truths[i])
            response_length = len(completion.token_ids)
            logprobs = completion.logprobs # list[dict[int, Logprob]]
            if logprobs is not None:
                lp_matrix = np.array([
                    [lp.logprob for lp in step_logprobs.values()]
                    for step_logprobs in logprobs
                ]) # Shape: (len(output), vocab_size)
                entropies = -np.sum(np.exp(lp_matrix) * lp_matrix, axis=-1) # (len_output)
                avg_token_entropy = np.mean(entropies)

            entry = {
                "prompt": prompts[i],
                "generated_text": generated_text,
                "ground_truth": ground_truths[i],
                **metrics,
                "average_token_entropy": avg_token_entropy if logprobs else None,
                "response_length": response_length,
            }

            results.append(entry)

            total_metrics["response_length"] += response_length
            total_metrics["token_entropy"] += avg_token_entropy if logprobs else 0.0
            if metrics["reward"] == 1.0:
                total_metrics["response_length_correct"] += response_length
                total_metrics["num_correct"] += 1.0
            else:
                total_metrics["response_length_incorrect"] += response_length

    # Calculate averages
    num_examples = len(results)
    num_correct = total_metrics["num_correct"]
    num_incorrect = num_examples - num_correct
    avg_metrics = {
        "avg_response_length": total_metrics["response_length"] / num_examples,
        "avg_response_length_correct": total_metrics["response_length_correct"] / num_correct if num_correct > 0 else 0,
        "avg_response_length_incorrect": total_metrics["response_length_incorrect"] / num_incorrect if num_incorrect > 0 else 0,
        "accuracy": total_metrics["num_correct"] / num_examples,
        "avg_token_entropy": total_metrics["token_entropy"] / num_examples,
    }
    results.append(avg_metrics)

    # Print summary
    print("-" * 40)
    print(f"Evaluation complete on {num_examples} examples.")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 40)

    # Serialize to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    return results

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
