# src/optimizer.py — ProTeGi-style Prompt Optimizer (Refactored)

import json
from pathlib import Path
from typing import Dict
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPT_DIR = Path("prompts")
LOG_DIR = Path("logs/optimizer_logs")
PROMPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# === Model Loader ===
def load_model_and_tokenizer(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading optimization model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)


# === Utility: Extract <tag>content</tag> ===
def extract_between_tags(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in text and end_tag in text:
        return text.split(start_tag)[1].split(end_tag)[0].strip()
    return None


# === Prompt Optimizer Function (ProTeGi Style) ===
def optimize_prompt(initial_prompt: str, failed_output: str, feedback: str, version: int) -> str:
    optimizer_instruction = f"""
You are an AI assistant tasked with implementing the ProTeGi (Prompt Optimization with Textual Gradients) algorithm for prompt optimization. Your goal is to take an initial prompt and iteratively improve it using the ProTeGi algorithm. Follow these instructions carefully:

First, you will receive the following inputs:

<initial_prompt>
{initial_prompt}
</initial_prompt>

You will also receive two parameters:

Max iterations: 3
Convergence threshold: 0.95

To begin, always start by initializing the ProTeGi optimizer:

<code>
optimizer = initialize_protegi(initial_prompt)
</code>

Next, implement the main optimization loop. This loop should continue until either the maximum number of iterations is reached or the convergence threshold is met. Within each iteration, perform the following steps:

Minibatch Sampling: Select a diverse set of training examples.
Textual Gradient Generation: Analyze prompt performance and generate feedback.
Prompt Editing: Apply textual gradients to create new prompt candidates.
Beam Search Implementation: Evaluate candidates and maintain a diverse beam.
Bandit Selection Process: Use UCB algorithm to select promising candidates.
Convergence Check: Assess improvement and stability of top candidates.

Output from the reasoning pipeline:
{failed_output}

Evaluator feedback:
{feedback}

After the optimization loop, select the best prompt based on performance and generalization ability.

Output your results in the following format:

<results>
<optimized_prompt>
[Insert the final optimized prompt here]
</optimized_prompt>

<performance_metrics>
[Insert key performance metrics, such as accuracy improvement, convergence rate, etc.]
</performance_metrics>

<optimization_process>
[Provide a brief summary of the optimization process, including number of iterations, key improvements, and any challenges encountered]
</optimization_process>
</results>
""".strip()

    inputs = tokenizer(optimizer_instruction, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    optimized_prompt = extract_between_tags(result, "optimized_prompt")
    if not optimized_prompt:
        print("⚠️ Warning: <optimized_prompt> tag not found. Using raw output fallback.")
        optimized_prompt = result[len(optimizer_instruction):].strip()

    # Save optimized prompt
    prompt_path = PROMPT_DIR / f"optimized_v{version}.txt"
    with open(prompt_path, "w") as f:
        f.write(optimized_prompt)

    # Save optimizer log
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "initial_prompt": initial_prompt,
        "failed_output": failed_output,
        "feedback": feedback,
        "optimized_prompt": optimized_prompt
    }
    with open(LOG_DIR / f"opt_log_v{ver_
