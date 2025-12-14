"""
Evaluate the zero-shot performance of a model on a dataset.
"""

import json
import os
import typer
from typing import List, Optional
from vllm import LLM, SamplingParams

from cs336_alignment.evaluation import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

app = typer.Typer()

def load_prompts_and_gts(data_path: str, prompt_template_path: str):
    """
    Load data from a jsonl file with keys "problem" and "answer"
    Format it using a prompt from `prompt_template_path` with variable {question}
    """
    with open(prompt_template_path, "r") as f:
        template = f.read()

    prompts = []
    ground_truths = []

    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            question = item.get("problem")
            ground_truth = item.get("answer")

            if question is None or ground_truth is None:
                continue

            formatted_prompt = template.format(question=question)

            prompts.append(formatted_prompt)
            ground_truths.append(ground_truth)
    
    return prompts, ground_truths

@app.command()
def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "data/MATH/validation.jsonl",
    prompt_file: str = "cs336_alignment/prompts/r1_zero.prompt",
    output_file: str = "results/baseline_results.jsonl",
    tensor_parallel_size: int = 1,
):
    """
    Run zero-shot MATH baseline evaluation on Qwen2.5-Math-1.5B.
    """

    print(f"Loading data from {data_path}...")
    prompts, ground_truths = load_prompts_and_gts(data_path, prompt_file)
    print(f"Loaded {len(prompts)} examples.")

    print(f"Initializing vLLM with model: {model_path}...")
    llm = LLM(model=model_path)

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
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_file=output_file,
    )

if __name__ == "__main__":
    app()
