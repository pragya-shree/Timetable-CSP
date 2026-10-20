#!/usr/bin/env python
"""Experiment runner for AI Timetable Scheduler.

Runs all solver configurations on both the small and large datasets,
records key metrics, saves CSV/JSON summary files, and generates a
cost‑vs‑iterations plot for the Min‑Conflicts algorithm.

The script is intended to be executed from the project root:

    python experiments/run_experiments.py

Required packages: pandas, matplotlib
"""

import json
import csv
import os
import pathlib
import random
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

# Ensure project root is on sys.path so core/ is importable when running as
# `python experiments/run_experiments.py` from any working directory.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import matplotlib.pyplot as plt
import pandas as pd

# Import solvers and utilities from the core package
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.models import load_dataset_from_file, TimetableDataset

# Mapping of configuration names to solver factories
SOLVER_CONFIGS = {
    "plain":         lambda ds: solve_backtracking(ds, use_mrv=False, use_degree=False, use_fc=False),
    "mrv":           lambda ds: solve_backtracking(ds, use_mrv=True,  use_degree=False, use_fc=False),
    "mrv_degree":    lambda ds: solve_backtracking(ds, use_mrv=True,  use_degree=True,  use_fc=False),
    "mrv_degree_fc": lambda ds: solve_backtracking(ds, use_mrv=True,  use_degree=True,  use_fc=True),
    "min_conflicts": lambda ds: solve_min_conflicts(
        ds,
        max_iterations=5000,
        record_cost_history=True,
        deterministic=False,
    ),
}

# Datasets – absolute paths based on project root
DATASET_PATHS = {
    "small": str(_PROJECT_ROOT / "datasets" / "small_dataset.json"),
    "large": str(_PROJECT_ROOT / "datasets" / "large_dataset.json"),
}

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_solver(config_name: str, dataset_name: str, dataset: TimetableDataset) -> Dict[str, Any]:
    """Execute a solver and return a flat dict of metrics."""
    solver_fn = SOLVER_CONFIGS[config_name]
    result = solver_fn(dataset)

    # Common result fields (both backtracking and min‑conflicts share a similar API)
    metrics = {
        "config": config_name,
        "dataset": dataset_name,
        "success": getattr(result, "success", False),
        "status": getattr(result, "status", ""),
        "runtime_seconds": getattr(result, "runtime_seconds", None),
        "iterations": getattr(result, "iterations", None),
        "conflicted_variable_selections": getattr(result, "conflicted_variable_selections", None),
        "candidate_placements_evaluated": getattr(result, "candidate_placements_evaluated", None),
        "initial_hard_conflicts": getattr(result, "initial_hard_conflicts", None),
        "final_hard_violations": getattr(result, "final_hard_violations", None),
        "total_cost": getattr(result, "total_cost", None),
        "random_seed": getattr(result, "random_seed", None),
    }
    # Soft penalties if present
    soft = getattr(result, "soft_penalties", {})
    for key, val in soft.items():
        metrics[f"soft_{key}"] = val

    # Store cost history for Min‑Conflicts (used later for plotting)
    if config_name == "min_conflicts" and hasattr(result, "cost_history"):
        metrics["cost_history"] = result.cost_history
    else:
        metrics["cost_history"] = None

    return metrics

def run_all_experiments() -> List[Dict[str, Any]]:
    all_metrics: List[Dict[str, Any]] = []
    for dataset_name, rel_path in DATASET_PATHS.items():
        dataset_path = pathlib.Path(rel_path)
        dataset = load_dataset_from_file(dataset_path)
        for config_name in SOLVER_CONFIGS.keys():
            if config_name == "min_conflicts":
                # Run 10 trials with distinct random seeds
                for trial in range(10):
                    seed = random.randint(1, 1_000_000)
                    print(f"Running {config_name} trial {trial+1}/10 on {dataset_name} dataset (seed={seed}) …")
                    result = solve_min_conflicts(
                        dataset,
                        max_iterations=5000,
                        record_cost_history=True,
                        deterministic=False,
                        random_seed=seed,
                    )
                    # Build metrics manually (similar to run_solver)
                    metrics = {
                        "config": config_name,
                        "dataset": dataset_name,
                        "success": getattr(result, "success", False),
                        "status": getattr(result, "status", ""),
                        "runtime_seconds": getattr(result, "runtime_seconds", None),
                        "iterations": getattr(result, "iterations", None),
                        "conflicted_variable_selections": getattr(result, "conflicted_variable_selections", None),
                        "candidate_placements_evaluated": getattr(result, "candidate_placements_evaluated", None),
                        "initial_hard_conflicts": getattr(result, "initial_hard_conflicts", None),
                        "final_hard_violations": getattr(result, "final_hard_violations", None),
                        "total_cost": getattr(result, "total_cost", None),
                        "random_seed": getattr(result, "random_seed", None),
                    }
                    soft = getattr(result, "soft_penalties", {})
                    for key, val in soft.items():
                        metrics[f"soft_{key}"] = val
                    if hasattr(result, "cost_history"):
                        metrics["cost_history"] = result.cost_history
                    else:
                        metrics["cost_history"] = None
                    all_metrics.append(metrics)
            else:
                print(f"Running {config_name} on {dataset_name} dataset …")
                metrics = run_solver(config_name, dataset_name, dataset)
                all_metrics.append(metrics)
    return all_metrics

def save_results(metrics_list: List[Dict[str, Any]]):
    # Save a flat CSV of all runs
    csv_path = RESULTS_DIR / "experiment_runs.csv"
    # Ensure consistent column order (excluding cost_history which is kept separate)
    fieldnames = [
        "config",
        "dataset",
        "success",
        "status",
        "runtime_seconds",
        "iterations",
        "conflicted_variable_selections",
        "candidate_placements_evaluated",
        "initial_hard_conflicts",
        "final_hard_violations",
        "total_cost",
        "random_seed",
        "soft_s1_gaps",
        "soft_s2_repeats",
        "soft_s3_non_consecutive",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            row = {k: m.get(k) for k in fieldnames}
            writer.writerow(row)

    # Save a JSON metadata file describing the experiment run
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": {
            "python": "3.x",
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "configurations": list(SOLVER_CONFIGS.keys()),
        "datasets": list(DATASET_PATHS.keys()),
        "run_count": len(metrics_list),
    }
    meta_path = RESULTS_DIR / "experiment_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Create a summary CSV (average runtime & cost per config/dataset)
    df = pd.DataFrame(metrics_list)
    summary = (
        df.groupby(["config", "dataset"])
        .agg(
            avg_runtime_seconds=("runtime_seconds", "mean"),
            avg_iterations=("iterations", "mean"),
            success_rate=("success", "mean"),
            avg_total_cost=("total_cost", "mean"),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)

def plot_min_conflicts_cost(metrics_list: List[Dict[str, Any]]):
    # Filter Min‑Conflicts runs that have a cost history
    mc_runs = [m for m in metrics_list if m["config"] == "min_conflicts" and m["cost_history"]]
    if not mc_runs:
        print("No Min‑Conflicts cost history found; skipping plot.")
        return
    plt.figure(figsize=(8, 5))
    for run in mc_runs:
        history = run["cost_history"]
        if not history:
            continue
        # history is a list of (iteration, cost)
        iters, costs = zip(*history)
        label = f"{run['dataset']} (seed={run['random_seed']})"
        plt.plot(iters, costs, label=label, alpha=0.7)
    plt.title("Min-Conflicts Cost vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = RESULTS_DIR / "min_conflicts_cost_vs_iterations.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Min-Conflicts cost plot to {plot_path}")

def main():
    all_metrics = run_all_experiments()
    save_results(all_metrics)
    plot_min_conflicts_cost(all_metrics)
    print("Experiment run complete. Results stored in 'experiments/results/'.")

if __name__ == "__main__":
    main()
