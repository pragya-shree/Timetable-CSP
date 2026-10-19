"""CLI runner for the AI-Based Timetable Scheduler (Backtracking and Min-Conflicts).

Usage:
    python run_solver.py                                        # Plain Backtracking on small dataset (default)
    python run_solver.py --dataset small --algorithm plain
    python run_solver.py --dataset small --algorithm mrv
    python run_solver.py --dataset small --algorithm mrv_degree
    python run_solver.py --dataset small --mrv-degree
    python run_solver.py --dataset small --algorithm mrv_degree_fc
    python run_solver.py --dataset small --fc
    python run_solver.py --dataset small --algorithm min_conflicts
    python run_solver.py --dataset small --mc
    python run_solver.py --dataset large --algorithm min_conflicts --seed 123
    python run_solver.py --dataset small --algorithm min_conflicts --output output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.backtracking import solve_backtracking
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.min_conflicts import solve_min_conflicts
from core.models import load_dataset_from_file

DATASET_SHORTCUTS = {
    "small": Path("datasets/small_dataset.json"),
    "large": Path("datasets/large_dataset.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CSP Solvers (Plain, MRV, MRV+Degree, MRV+Degree+FC, or Min-Conflicts) on Timetable Datasets."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="small",
        help="Dataset name ('small', 'large') or path to JSON file (default: 'small').",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=[
            "plain",
            "mrv",
            "mrv_degree",
            "degree",
            "mrv_degree_fc",
            "fc",
            "min_conflicts",
            "mc",
        ],
        default="plain",
        help=(
            "Solver strategy: 'plain', 'mrv', 'mrv_degree', 'mrv_degree_fc', "
            "or 'min_conflicts' (default: 'plain')."
        ),
    )
    parser.add_argument(
        "--mrv",
        action="store_true",
        help="Shorthand flag to enable Backtracking + MRV.",
    )
    parser.add_argument(
        "--degree",
        "--mrv-degree",
        dest="mrv_degree",
        action="store_true",
        help="Shorthand flag to enable Backtracking + MRV + Degree Heuristic.",
    )
    parser.add_argument(
        "--fc",
        "--mrv-degree-fc",
        dest="mrv_degree_fc",
        action="store_true",
        help="Shorthand flag to enable Backtracking + MRV + Degree + Forward Checking.",
    )
    parser.add_argument(
        "--mc",
        "--min-conflicts",
        dest="min_conflicts",
        action="store_true",
        help="Shorthand flag to enable Min-Conflicts Local Search.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100_000,
        help="Maximum search nodes for backtracking algorithms (default: 100,000).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10_000,
        help="Maximum iterations for Min-Conflicts local search (default: 10,000).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Maximum runtime in seconds before limit termination (default: 30.0s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Min-Conflicts local search (default: 42).",
    )
    parser.add_argument(
        "--initial-strategy",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Initial assignment strategy for Min-Conflicts: 'random' or 'greedy' (default: 'random').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the solved timetable JSON.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str) -> Path:
    if dataset_arg.lower() in DATASET_SHORTCUTS:
        return DATASET_SHORTCUTS[dataset_arg.lower()]
    path = Path(dataset_arg)
    if not path.exists():
        print(f"Error: Dataset file not found at '{path}'.", file=sys.stderr)
        sys.exit(1)
    return path


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)

    is_mc = args.min_conflicts or (args.algorithm.lower() in ("min_conflicts", "mc"))

    if is_mc:
        algo_name = "Min-Conflicts Local Search"
    elif args.mrv_degree_fc or (args.algorithm.lower() in ("mrv_degree_fc", "fc")):
        use_mrv = True
        use_degree = True
        use_fc = True
        algo_name = "Backtracking + MRV + Degree + Forward Checking"
    elif args.mrv_degree or (args.algorithm.lower() in ("mrv_degree", "degree")):
        use_mrv = True
        use_degree = True
        use_fc = False
        algo_name = "Backtracking + MRV + Degree"
    elif args.mrv or (args.algorithm.lower() == "mrv"):
        use_mrv = True
        use_degree = False
        use_fc = False
        algo_name = "Backtracking + MRV"
    else:
        use_mrv = False
        use_degree = False
        use_fc = False
        algo_name = "Plain Backtracking (Baseline)"

    print(f"\n{'=' * 65}")
    print(f" AI-BASED TIMETABLE SCHEDULER -- {algo_name.upper()}")
    print(f"{'=' * 65}")
    print(f"Loading dataset:  {dataset_path}")
    dataset = load_dataset_from_file(dataset_path)

    print(f"Dataset Name:     {dataset.name}")
    print(f"Classes:          {len(dataset.classes)}")
    print(f"Teachers:         {len(dataset.teachers)}")
    print(f"Rooms:            {len(dataset.rooms)}")
    print(f"Time Grid:        {len(dataset.time_structure.days)} days x {dataset.time_structure.periods_per_day} periods/day")
    print(f"Algorithm:        {algo_name}")
    if is_mc:
        print(f"Search Budget:    max_iterations={args.max_iterations:,}, time_limit={args.time_limit:.1f}s, seed={args.seed}")
        print(f"Initial Strategy: {args.initial_strategy}")
    else:
        print(f"Search Budget:    max_nodes={args.max_nodes:,}, time_limit={args.time_limit:.1f}s")
    print(f"{'-' * 65}")
    print(f"Solving with {algo_name}...")

    if is_mc:
        result = solve_min_conflicts(
            dataset=dataset,
            max_iterations=args.max_iterations,
            time_limit_seconds=args.time_limit,
            random_seed=args.seed,
            initial_strategy=args.initial_strategy,
        )
    else:
        result = solve_backtracking(
            dataset=dataset,
            max_nodes=args.max_nodes,
            time_limit_seconds=args.time_limit,
            use_mrv=use_mrv,
            use_degree=use_degree,
            use_fc=use_fc,
        )

    print(f"\n{'-' * 65}")
    print(f" SEARCH OUTCOME & METRICS")
    print(f"{'-' * 65}")
    print(f"Status:             {result.status.upper()}")
    print(f"Algorithm:          {result.algorithm}")
    print(f"Solved:             {result.success}")
    print(f"Runtime:            {result.runtime_seconds:.4f} seconds")

    if is_mc:
        print(f"Iterations:         {result.iterations:,}")
        print(f"Var Selections:     {result.conflicted_variable_selections:,}")
        print(f"Placements Scored:  {result.candidate_placements_evaluated:,}")
        print(f"Initial Conflicts:  {result.initial_hard_conflicts:,}")
        print(f"Final Violations:   {result.final_hard_violations:,}")
        print(f"Random Seed:        {result.random_seed}")
    else:
        print(f"Nodes Explored:     {result.nodes_explored:,}")
        print(f"Assignments Tried:  {result.assignments_tried:,}")
        print(f"Backtracks:         {result.backtracks:,}")
        print(f"Max Search Depth:   {result.max_search_depth}")

    if result.failure_reason:
        print(f"Failure Reason:     {result.failure_reason}")

    # Validator verification
    if result.timetable is not None:
        val_res = validate_timetable(result.timetable)
        cost_res = calculate_cost(result.timetable)

        print(f"\n{'-' * 65}")
        print(f" INDEPENDENT VALIDATION & COST BREAKDOWN")
        print(f"{'-' * 65}")
        print(f"Complete:           {val_res.is_complete}")
        print(f"Valid:              {val_res.is_valid}")
        print(f"Hard Violations:    {val_res.hard_violation_count}")
        print(f"S1 Gaps Penalty:    {cost_res.s1_gaps} (cost: {cost_res.s1_cost})")
        print(f"S2 Repeats Penalty: {cost_res.s2_repeats} (cost: {cost_res.s2_cost})")
        print(f"S3 Non-Consecutive: {cost_res.s3_non_consecutive} (cost: {cost_res.s3_cost})")
        print(f"Total Soft Cost:    {cost_res.total_soft_cost}")
        print(f"Total Cost:         {cost_res.total_cost}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.timetable.to_dict(), f, indent=2)
            print(f"\nSaved solved timetable to: {output_path}")

    print(f"{'=' * 65}\n")

    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
