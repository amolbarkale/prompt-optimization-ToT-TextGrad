# evaluation/metrics_logger.py — Improved Prompt Evolution Evaluator

import json
from pathlib import Path
from typing import List, Dict
from difflib import SequenceMatcher
import argparse

EVAL_DIR = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)
METRICS_FILE = EVAL_DIR / "metrics.json"

def load_expected_answers(task_file: str) -> Dict[str, str]:
    with open(task_file, "r") as f:
        tasks = json.load(f)
    return {task["task_id"]: task["expected_solution"] for task in tasks}

def load_all_tot_results(logs_dir: Path) -> List[Dict]:
    return [json.load(f.open()) for f in logs_dir.glob("*_tot.json")]

def is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold

def calculate_metrics(task_file: str, logs_dir: str):
    expected_map = load_expected_answers(task_file)
    results = load_all_tot_results(Path(logs_dir))

    metrics = {
        "total_tasks": 0,
        "successful_tasks": 0,
        "hallucinated": 0,
        "inconclusive": 0,
        "task_level": []
    }

    for result in results:
        task_id = result["task_id"]
        expected = expected_map.get(task_id, "")
        final = result.get("final_answer", "")
        version = result.get("used_prompt_version", "n/a")

        metrics["total_tasks"] += 1
        if final == "Inconclusive.":
            status = "inconclusive"
            metrics["inconclusive"] += 1
        elif is_similar(expected, final):
            status = "success"
            metrics["successful_tasks"] += 1
        else:
            status = "hallucinated"
            metrics["hallucinated"] += 1

        metrics["task_level"].append({
            "task_id": task_id,
            "expected": expected,
            "final": final,
            "status": status,
            "prompt_version": version
        })

    # Add derived metrics
    total = max(metrics["total_tasks"], 1)
    metrics["success_rate"] = round(100 * metrics["successful_tasks"] / total, 2)
    metrics["hallucination_rate"] = round(100 * metrics["hallucinated"] / total, 2)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📊 Evaluation complete. Summary saved to {METRICS_FILE}")
    print(f"✅ Success: {metrics['successful_tasks']} / {metrics['total_tasks']} ({metrics['success_rate']}%)")
    print(f"⚠️ Inconclusive: {metrics['inconclusive']}, Hallucinated: {metrics['hallucinated']} ({metrics['hallucination_rate']}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", default="tasks_combined.json")
    parser.add_argument("--logs_dir", default="logs/reasoning_paths")
    args = parser.parse_args()
    calculate_metrics(args.task_file, args.logs_dir)
