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

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truths: List[str],
    step: int,
    reward_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    max_new_tokens: int = 1024,
    **kwargs
) -> None:
    """
    Generate responses from the model for given prompts, compute rewards if a reward function
    is provided, and log the results.
    
    Args:
        model: The language model to generate from
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        ground_truths: List of ground truth answer strings
        step: Current training step (for logging)
        reward_fn: Optional function that takes (response, ground_truth) and returns a dict
                   with reward metrics (e.g., {"reward": float, "format_reward": float, ...})
        max_new_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments to pass to model.generate()
    """
    model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            **kwargs
        )
    
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    rewards = []
    correct_flags = []
    format_rewards = []
    answer_rewards = []

    if reward_fn:
        for response, ground_truth in zip(responses, ground_truths):
            metrics = reward_fn(response, ground_truth)
            rewards.append(metrics.get("reward", 0.0))
            correct_flags.append(metrics.get("reward", 0.0) > 0.5)  # Consider correct if reward > 0.5
            format_rewards.append(metrics.get("format_reward", 0.0))
            answer_rewards.append(metrics.get("answer_reward", 0.0))

    response_lens = [len(t) for t in generated_tokens]
    avg_len = np.mean(response_lens)

    # Prepare metrics for logging
    log_metrics = {
        "generation/num_generations": len(responses),
        "generation/avg_response_length": avg_len,
    }
    
    if reward_fn:
        avg_reward = np.mean(rewards) if rewards else 0.0
        accuracy = np.mean(correct_flags) if correct_flags else 0.0
        avg_format_reward = np.mean(format_rewards) if format_rewards else 0.0
        avg_answer_reward = np.mean(answer_rewards) if answer_rewards else 0.0
        
        log_metrics.update({
            "generation/avg_reward": avg_reward,
            "generation/accuracy": accuracy,
            "generation/avg_format_reward": avg_format_reward,
            "generation/avg_answer_reward": avg_answer_reward,
        })

    # Log to wandb if available and initialized
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(log_metrics, step=step)

    # Log the results to console
    print(f"\n{'='*60}")
    print(f"Generation Log - Step {step}")
    print(f"{'='*60}")
    print(f"Number of generations: {len(responses)}")
    print(f"Average response length: {avg_len:.2f} tokens")
    
    if reward_fn:
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Accuracy (reward > 0.5): {accuracy:.4f}")
        print(f"Average format reward: {avg_format_reward:.4f}")
        print(f"Average answer reward: {avg_answer_reward:.4f}")
        
        # Print a few example generations
        print(f"\nExample generations (showing first 3):")
        for i in range(min(3, len(responses))):
            print(f"\n  Example {i+1}:")
            print(f"    Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"    Prompt: {prompts[i]}")
            print(f"    Response: {responses[i][:200]}..." if len(responses[i]) > 200 else f"    Response: {responses[i]}")
            print(f"    Ground truth: {ground_truths[i]}")
            if reward_fn:
                print(f"    Reward: {rewards[i]:.4f} (format: {format_rewards[i]:.4f}, answer: {answer_rewards[i]:.4f})")
    
    print(f"{'='*60}\n")

    model.train()
