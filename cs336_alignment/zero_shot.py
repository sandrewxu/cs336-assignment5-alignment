"""
Evaluate Qwen 2.5 Math 1.B zero-shot performance on GSM8K
"""
# Imports
from datasets import load_dataset
from vllm import LLM, SamplingParams

# 1. Load GSM8K validation samples (from jsonl)
data = load_dataset("openai/gsm8k", "main")

# 2. format them as string prompts to the language model using the r1_zero prompt

# 3. generate outputs for each example

# 4. calculate evaluation metrics

# 5. serialize the examples, model generations, and corresponding evaluation scores to disk

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
