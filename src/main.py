# main.py — Tree-of-Thought + Self-Consistency Reasoning Engine (Refactored)

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# === Configuration ===
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_DEPTH = 3
BRANCHING_FACTOR = 3
OUTPUT_DIR = Path("logs/reasoning_paths")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === Load and Initialize Model ===
def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading model '{model_name}' on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


tokenizer, model, device = load_model(MODEL_NAME)


# === Core Functions ===
def load_tasks(task_file: str) -> List[Dict[str, Any]]:
    with open(task_file, 'r') as f:
        return json.load(f)


def prompt_next_thought(problem: str, history: List[str]) -> str:
    thoughts = "\n".join([f"Thought {i+1}: {h}" for i, h in enumerate(history)])
    prompt = (
        "You are solving a multi-step reasoning problem.\n"
        f"Problem: {problem}\n{thoughts}\nNow, write the next possible thought to proceed."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Safely extract the next thought using split instead of relying on character offset
    if "Now, write the next possible thought to proceed." in decoded:
        next_thought = decoded.split("Now, write the next possible thought to proceed.")[-1].strip().split("\n")[0]
    else:
        next_thought = decoded.strip().split("\n")[-1]

    return next_thought


def expand_tree(problem: str, current_path: List[str], depth: int) -> List[List[str]]:
    if depth == MAX_DEPTH:
        return [current_path]

    candidates = []
    for _ in range(BRANCHING_FACTOR):
        try:
            next_thought = prompt_next_thought(problem, current_path)
            extended_path = current_path + [next_thought]
            candidates.extend(expand_tree(problem, extended_path, depth + 1))
        except Exception as e:
            print(f"⚠️ Error during path expansion: {e}")
            continue
    return candidates


def self_consistency(paths: List[List[str]]) -> str:
    final_answers = [path[-1] for path in paths if path]
    counts = Counter(final_answers)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else "Inconclusive."


def run_tot_for_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]
    problem = task["problem"]

    all_paths = expand_tree(problem, [], 0)
    final_answer = self_consistency(all_paths)

    result = {
        "task_id": task_id,
        "problem": problem,
        "paths": all_paths,
        "final_answer": final_answer
    }

    with open(OUTPUT_DIR / f"{task_id}_tot.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_pipeline(task_file: str):
    tasks = load_tasks(task_file)
    results = []
    for task in tasks:
        print(f"🧠 Processing: {task['task_id']} - {task['title']}")
        result = run_tot_for_task(task)
        results.append(result)
    return results


# === Entry Point ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ToT+Self-Consistency pipeline")
    parser.add_argument("--task_file", type=str, default="tasks_combined.json", help="Path to task JSON file")
    args = parser.parse_args()

    results = run_pipeline(args.task_file)
    print("\n✅ Completed reasoning for all tasks.")
