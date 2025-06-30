# src/controller.py — Orchestrates ToT + Prompt Optimization Pipeline (Refined)

import json
from pathlib import Path
from optimizer import optimize_prompt
from tot_engine import run_tot_for_task  # Ensure this matches your actual filename

# === Configuration ===
PROMPT_VERSION_DIR = Path("prompts")
PROMPT_VERSION_DIR.mkdir(exist_ok=True)
TASKS_FILE = "tasks_combined.json"
MAX_OPTIMIZATION_ROUNDS = 3

# === Loaders ===
def load_tasks(task_file):
    with open(task_file, "r") as f:
        return json.load(f)

def load_prompt(version: int) -> str:
    filename = PROMPT_VERSION_DIR / ("initial.txt" if version == 0 else f"optimized_v{version}.txt")
    with open(filename, "r") as f:
        return f.read().strip()

def save_prompt(prompt: str, version: int):
    filename = PROMPT_VERSION_DIR / ("final_prompt.txt" if version == -1 else f"optimized_v{version}.txt")
    with open(filename, "w") as f:
        f.write(prompt)

# === Main Optimization Loop ===
def process_task_with_optimization(task: dict):
    version = 0
    task_id = task["task_id"]
    problem = task["problem"]

    while version <= MAX_OPTIMIZATION_ROUNDS:
        print(f"\n🧠 Task: {task_id} | Prompt Version: {version}")
        current_prompt = load_prompt(version)

        # Wrap original task with current prompt for ToT reasoning
        task_payload = {
            "task_id": task_id,
            "problem": f"{current_prompt}\n\nOriginal Task:\n{problem}"
        }

        result = run_tot_for_task(task_payload)
        final_answer = result.get("final_answer", "Inconclusive.")

        if final_answer != "Inconclusive.":
            print(f"✅ Success with version {version}: {final_answer}")
            save_prompt(current_prompt, -1)  # Save final prompt
            return

        print("⚠️ Inconclusive result. Optimizing prompt...")
        failed_output = result["paths"][-1][-1] if result.get("paths") else "No valid paths."
        feedback = "Inconsistent logic or unclear reasoning in one or more thought steps."

        # Optimize current prompt
        new_prompt = optimize_prompt(
            initial_prompt=current_prompt,
            failed_output=failed_output,
            feedback=feedback,
            version=version + 1
        )
        save_prompt(new_prompt, version + 1)
        version += 1

    print(f"🚫 Max retries reached for {task_id}. Moving to next task.")

# === Orchestrator ===
if __name__ == "__main__":
    tasks = load_tasks(TASKS_FILE)
    for task in tasks:
        process_task_with_optimization(task)
