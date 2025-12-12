"""
Evaluate Qwen 2.5 Math 1.B zero-shot performance on GSM8K
"""
# Imports
from vllm import LLM, SamplingParams
import jsonlines
from datetime import datetime
from typing import Callable, List, Dict
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_PATH = "data/gsm8k/test.jsonl"
OUTPUT_DIR = Path("results/zero_shot")
FILE_NAME = f"qwen_gsm8k_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
OUTPUT_FILE = OUTPUT_DIR / FILE_NAME
LIMIT_SAMPLES = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# R1_ZERO_PROMPT from cs336_alignment/prompts/*
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def main():
    # 1. Load GSM8K validation samples (from jsonl)
    data = []
    with jsonlines.open(DATASET_PATH, 'r') as f:
        for qa in f:
            data.append(qa)
    if LIMIT_SAMPLES is not None:
        data = data[:LIMIT_SAMPLES]
    questions = [qa['question'] for qa in data]
    answers = [qa['answer'] for qa in data]
    ground_truths = [a.split("#### ")[1] for a in answers]

    # 2. format them as string prompts to the language model using the r1_zero prompt

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    llm = LLM(model=MODEL_NAME)

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        questions=questions,
        prompt_template=R1_ZERO_PROMPT,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params
    )

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    questions: List[str],
    prompt_template: str,
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk
    """
    # Create prompts
    prompts = [prompt_template.format(question=q) for q in questions]

    # Generate outputs for prompts
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Calculate evaluation metrics
    all_results = []
    total_correct = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        metrics = reward_fn(generated_text, ground_truths[i])
        total_correct += metrics["reward"]

        result_entry = {
            "example_id": i,
            "question": questions[i],
            "ground_truth_answer": ground_truths[i],
            "model_generation": generated_text,
            **metrics
        }

        all_results.append(result_entry)

    total_examples = len(prompts)
    final_accuracy = total_correct / total_examples

    # serialize the examples, model generations, and corresponding evaluation scores to disk
    evaluation_metrics = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_PATH,
        "total_examples": total_examples,
        "total_correct": total_correct,
        "final_accuracy": final_accuracy,
        "prompt_template": prompt_template.replace('\n', ' ') # Cleaned up for summary
    }

    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        writer.write({"EVALUATION SUMMARY": evaluation_metrics})
        writer.write_all(all_results)
    
    # print to console
    print("-" * 50)
    print("Evaluation complete")
    print(f"Final Zero-Shot Accuracy on Test: {final_accuracy:.4f}")
    print(f"Full results saved to: {OUTPUT_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    main()
