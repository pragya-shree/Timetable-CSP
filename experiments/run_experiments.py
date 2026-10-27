#!/usr/bin/env python
"""Experiment runner for AI Timetable Scheduler.

Runs solver configurations across benchmark datasets (small, large, medium_tight),
records comprehensive search metrics (nodes, backtracks, assignments, iterations),
normalizes hard violation reporting, enforces reproducible master-seed trial setups,
saves raw and aggregated CSV/JSON summary files, and plots Min-Conflicts / Hybrid SA cost curves.

The script is intended to be executed from the project root:

    python experiments/run_experiments.py

Required packages: pandas, matplotlib
"""

import csv
import json
import os
import pathlib
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.models import TimetableDataset, load_dataset_from_file
from core.simulated_annealing import solve_hybrid_simulated_annealing

MASTER_RANDOM_SEED: int = 42


def derive_trial_seed(master_seed: int, trial_idx: int) -> int:
    """Deterministically derives a trial random seed from master seed and trial index."""
    return (int(master_seed) * 10007 + int(trial_idx) * 997 + 101) % 1_000_000


# Baseline solver factories (preserved for backwards-compatibility with existing tests)
SOLVER_CONFIGS = {
    "plain": lambda ds: solve_backtracking(ds, use_mrv=False, use_degree=False, use_fc=False),
    "mrv": lambda ds: solve_backtracking(ds, use_mrv=True, use_degree=False, use_fc=False),
    "mrv_degree": lambda ds: solve_backtracking(ds, use_mrv=True, use_degree=True, use_fc=False),
    "mrv_degree_fc": lambda ds: solve_backtracking(ds, use_mrv=True, use_degree=True, use_fc=True),
    "min_conflicts": lambda ds: solve_min_conflicts(
        ds,
        max_iterations=5000,
        record_cost_history=True,
        deterministic=False,
        random_seed=MASTER_RANDOM_SEED,
    ),
}

# Extended suite including Hybrid Simulated Annealing
ALL_SOLVER_CONFIGS = {
    **SOLVER_CONFIGS,
    "hybrid_sa": lambda ds: solve_hybrid_simulated_annealing(
        ds,
        initial_temperature=100.0,
        cooling_rate=0.995,
        max_iterations=2000,
        random_seed=MASTER_RANDOM_SEED,
    ),
}

# Datasets – absolute paths based on project root
DATASET_PATHS = {
    "small": str(_PROJECT_ROOT / "datasets" / "small_dataset.json"),
    "large": str(_PROJECT_ROOT / "datasets" / "large_dataset.json"),
}

ALL_DATASET_PATHS = {
    **DATASET_PATHS,
    "medium_tight": str(_PROJECT_ROOT / "datasets" / "medium_tight_dataset.json"),
}

RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_result_metrics(
    result: Any,
    config_name: str,
    dataset_name: str,
    random_seed: int | None = None,
    master_seed: int | None = None,
) -> Dict[str, Any]:
    """Normalizes and extracts a flat dictionary of metrics from any solver result object."""
    # BUG A Fix: Normalize final_hard_violations across BacktrackingResult and MinConflictsResult
    final_hard = getattr(
        result,
        "final_hard_violations",
        getattr(result, "hard_violations", None),
    )

    # BUG B Fix: Collect distinct search metrics
    metrics: Dict[str, Any] = {
        "config": config_name,
        "dataset": dataset_name,
        "success": bool(getattr(result, "success", False)),
        "status": getattr(result, "status", ""),
        "runtime_seconds": getattr(result, "runtime_seconds", None),
        # Search metrics (Backtracking)
        "nodes_explored": getattr(result, "nodes_explored", None),
        "backtracks": getattr(result, "backtracks", None),
        "assignments_tried": getattr(result, "assignments_tried", None),
        "max_search_depth": getattr(result, "max_search_depth", None),
        # Local search metrics (Min-Conflicts / SA)
        "iterations": getattr(result, "iterations", None),
        "conflicted_variable_selections": getattr(result, "conflicted_variable_selections", None),
        "candidate_placements_evaluated": getattr(result, "candidate_placements_evaluated", None),
        "initial_hard_conflicts": getattr(result, "initial_hard_conflicts", None),
        "moves_accepted": getattr(result, "moves_accepted", None),
        "uphill_moves_accepted": getattr(result, "uphill_moves_accepted", None),
        # Outcome metrics
        "final_hard_violations": final_hard,
        "total_cost": getattr(result, "total_cost", None),
        "random_seed": getattr(result, "random_seed", random_seed),
        "master_seed": master_seed,
    }

    # Soft penalties (S1 - S5)
    soft = getattr(result, "soft_penalties", {})
    metrics["soft_s1_gaps"] = soft.get("s1_gaps", None)
    metrics["soft_s2_repeats"] = soft.get("s2_repeats", None)
    metrics["soft_s3_non_consecutive"] = soft.get("s3_non_consecutive", None)
    metrics["soft_s4_teacher_preferences"] = soft.get("s4_teacher_preferences", None)
    metrics["soft_s5_teacher_consecutive"] = soft.get("s5_teacher_consecutive", None)

    # Cost history
    metrics["cost_history"] = getattr(result, "cost_history", None)

    return metrics


def run_solver(
    config_name: str,
    dataset_name: str,
    dataset: TimetableDataset,
    random_seed: int | None = None,
    master_seed: int | None = None,
) -> Dict[str, Any]:
    """Executes a single solver run and returns extracted metrics."""
    if config_name in ALL_SOLVER_CONFIGS:
        solver_fn = ALL_SOLVER_CONFIGS[config_name]
    else:
        solver_fn = SOLVER_CONFIGS[config_name]

    if config_name == "min_conflicts" and random_seed is not None:
        result = solve_min_conflicts(
            dataset,
            max_iterations=5000,
            record_cost_history=True,
            deterministic=False,
            random_seed=random_seed,
        )
    elif config_name == "hybrid_sa" and random_seed is not None:
        result = solve_hybrid_simulated_annealing(
            dataset,
            initial_temperature=100.0,
            cooling_rate=0.995,
            max_iterations=2000,
            random_seed=random_seed,
        )
    else:
        result = solver_fn(dataset)

    return extract_result_metrics(
        result,
        config_name=config_name,
        dataset_name=dataset_name,
        random_seed=random_seed,
        master_seed=master_seed,
    )


def run_all_experiments(
    datasets: Dict[str, str] | None = None,
    configs: Dict[str, Any] | None = None,
    randomized_trials: int = 10,
    master_seed: int = MASTER_RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Runs all benchmark configurations across datasets with deterministic trial seeds."""
    target_datasets = datasets or ALL_DATASET_PATHS
    target_configs = configs or ALL_SOLVER_CONFIGS
    all_metrics: List[Dict[str, Any]] = []

    for dataset_name, rel_path in target_datasets.items():
        dataset_path = pathlib.Path(rel_path)
        if not dataset_path.exists():
            print(f"Warning: dataset file '{dataset_path}' not found, skipping.")
            continue
        dataset = load_dataset_from_file(dataset_path)

        for config_name in target_configs.keys():
            if config_name in ("min_conflicts", "hybrid_sa"):
                for trial in range(randomized_trials):
                    trial_seed = derive_trial_seed(master_seed, trial)
                    print(
                        f"Running {config_name} trial {trial+1}/{randomized_trials} "
                        f"on {dataset_name} dataset (seed={trial_seed}) …"
                    )
                    metrics = run_solver(
                        config_name,
                        dataset_name,
                        dataset,
                        random_seed=trial_seed,
                        master_seed=master_seed,
                    )
                    all_metrics.append(metrics)
            else:
                print(f"Running {config_name} on {dataset_name} dataset …")
                metrics = run_solver(
                    config_name,
                    dataset_name,
                    dataset,
                    random_seed=None,
                    master_seed=master_seed,
                )
                all_metrics.append(metrics)

    return all_metrics


def save_results(metrics_list: List[Dict[str, Any]]):
    """Saves flat trial runs CSV, metadata JSON, and aggregated summary CSV."""
    csv_path = RESULTS_DIR / "experiment_runs.csv"
    fieldnames = [
        "config",
        "dataset",
        "success",
        "status",
        "runtime_seconds",
        "iterations",
        "nodes_explored",
        "backtracks",
        "assignments_tried",
        "max_search_depth",
        "conflicted_variable_selections",
        "candidate_placements_evaluated",
        "initial_hard_conflicts",
        "moves_accepted",
        "uphill_moves_accepted",
        "final_hard_violations",
        "total_cost",
        "random_seed",
        "master_seed",
        "soft_s1_gaps",
        "soft_s2_repeats",
        "soft_s3_non_consecutive",
        "soft_s4_teacher_preferences",
        "soft_s5_teacher_consecutive",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            row = {k: m.get(k) for k in fieldnames}
            writer.writerow(row)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "master_seed": MASTER_RANDOM_SEED,
        "configurations": sorted(list({m["config"] for m in metrics_list})),
        "datasets": sorted(list({m["dataset"] for m in metrics_list})),
        "run_count": len(metrics_list),
    }
    meta_path = RESULTS_DIR / "experiment_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Create a summary CSV (average runtime & cost per config/dataset)
    df = pd.DataFrame(metrics_list)
    for col in [
        "runtime_seconds", "iterations", "nodes_explored", "backtracks",
        "assignments_tried", "final_hard_violations", "total_cost", "success"
    ]:
        if col not in df.columns:
            df[col] = None

    summary = (
        df.groupby(["config", "dataset"])
        .agg(
            avg_runtime_seconds=("runtime_seconds", "mean"),
            avg_iterations=("iterations", "mean"),
            avg_nodes_explored=("nodes_explored", "mean"),
            avg_backtracks=("backtracks", "mean"),
            avg_assignments_tried=("assignments_tried", "mean"),
            avg_final_hard_violations=("final_hard_violations", "mean"),
            success_rate=("success", "mean"),
            avg_total_cost=("total_cost", "mean"),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)


def plot_min_conflicts_cost(metrics_list: List[Dict[str, Any]]):
    """Plots cost history for local search solvers."""
    mc_runs = [m for m in metrics_list if m.get("cost_history")]
    if not mc_runs:
        print("No local search cost history found; skipping plot.")
        return
    plt.figure(figsize=(9, 5))
    for run in mc_runs:
        history = run["cost_history"]
        if not history:
            continue
        iters, costs = zip(*history)
        label = f"{run['config']} - {run['dataset']} (seed={run.get('random_seed')})"
        plt.plot(iters, costs, label=label, alpha=0.6, linewidth=1.2)
    plt.title("Search Cost vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    if len(mc_runs) <= 12:
        plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = RESULTS_DIR / "min_conflicts_cost_vs_iterations.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved cost plot to {plot_path}")


def main():
    print("Starting comprehensive benchmark experiment run...")
    all_metrics = run_all_experiments(
        datasets=ALL_DATASET_PATHS,
        configs=ALL_SOLVER_CONFIGS,
        randomized_trials=10,
        master_seed=MASTER_RANDOM_SEED,
    )
    save_results(all_metrics)
    plot_min_conflicts_cost(all_metrics)
    print("Experiment run complete. Results stored in 'experiments/results/'.")


if __name__ == "__main__":
    main()
