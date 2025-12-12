"""
Download Qwen 2.5 Math 1.5B Base and GSM8K from HuggingFace into HPC Scratch
"""

from transformers import AutoModelForCausalLM
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
ds = load_dataset("openai/gsm8k", "main")
