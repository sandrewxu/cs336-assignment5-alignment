"""
Evaluation helper functions
"""

import json
import os
from typing import Callable, Dict, List
from vllm import LLM, SamplingParams

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
