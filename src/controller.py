# src/controller.py — Orchestrates ToT + Prompt Optimization Pipeline

import json
from pathlib import Path
from optimizer import optimize_prompt
from tot_engine import run_tot_for_task  # updated to match renamed module

# === Configuration ===
INITIAL_PROMPT_FILE = Path("prompts/initial.txt")
PROMPT_VERSION_DIR = Path("prompts")
PROMPT_VERSION_DIR.mkdir(exist_ok=True)

TASKS_FILE = "tasks_combined.json"
MAX_OPTIMIZATION_ROUNDS = 3

# === Load Tasks ===
def load_tasks(task_file):
    with open(task_file, 'r') as f:
        return json.load(f)

# === Load Prompt ===
def load_prompt(version: int) -> str:
    filename = PROMPT_VERSION_DIR / ("initial.txt" if version == 0 else f"optimized_v{version}.txt")
    with open(filename, 'r') as f:
        return f.read().strip()

tasks = load_tasks(TASKS_FILE)

for task in tasks:
    version = 0
    history = []

    while version <= MAX_OPTIMIZATION_ROUNDS:
        print(f"\n🧠 Task: {task['task_id']} | Prompt Version: {version}")

        current_prompt = load_prompt(version)
        task['problem'] = current_prompt
        result = run_tot_for_task(task)
        final_answer = result['final_answer']

        if final_answer != "Inconclusive.":
            print(f"✅ Success with version {version}: {final_answer}")
            break

        print("⚠️ Inconclusive result. Optimizing prompt...")
        failed_output = result['paths'][-1][-1] if result['paths'] else "No valid paths."

        feedback = "Inconsistent logic or unclear reasoning in one or more thought steps."
        current_prompt = load_prompt(version)
        new_prompt = optimize_prompt(
            initial_prompt=current_prompt,
            failed_output=failed_output,
            feedback=feedback,
            version=version + 1
        )

        version += 1

    if version > MAX_OPTIMIZATION_ROUNDS:
        print(f"🚫 Max retries reached for {task['task_id']}. Moving to next task.")

