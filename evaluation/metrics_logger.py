# evaluation/metrics_logger.py — Evaluate Prompt Evolution Effectiveness

import json
from pathlib import Path
from typing import List, Dict

EVAL_DIR = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)
METRICS_FILE = EVAL_DIR / "metrics.json"

# Load expected solutions from task file
def load_expected_answers(task_file: str) -> Dict[str, str]:
    with open(task_file, "r") as f:
        tasks = json.load(f)
    return {task["task_id"]: task["expected_solution"] for task in tasks}

# Load ToT results for each task
def load_all_tot_results(logs_dir: Path) -> List[Dict]:
    all_logs = list(logs_dir.glob("*_tot.json"))
    results = []
    for log_file in all_logs:
        with open(log_file) as f:
            results.append(json.load(f))
    return results

def calculate_metrics(task_file: str, logs_dir: str):
    expected_map = load_expected_answers(task_file)
    logs_dir = Path(logs_dir)
    results = load_all_tot_results(logs_dir)

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
        success = expected.lower().strip() in final.lower().strip()

        metrics["total_tasks"] += 1
        if final == "Inconclusive.":
            metrics["inconclusive"] += 1
        elif success:
            metrics["successful_tasks"] += 1
        else:
            metrics["hallucinated"] += 1

        metrics["task_level"].append({
            "task_id": task_id,
            "expected": expected,
            "final": final,
            "status": "success" if success else ("inconclusive" if final == "Inconclusive." else "hallucinated")
        })

    # Save metrics to file
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📊 Evaluation complete. Summary saved to {METRICS_FILE}")
    print(f"Success: {metrics['successful_tasks']} / {metrics['total_tasks']}")
    print(f"Inconclusive: {metrics['inconclusive']}, Hallucinated: {metrics['hallucinated']}")

if __name__ == "__main__":
    calculate_metrics("tasks_combined.json", "logs/reasoning_paths")
