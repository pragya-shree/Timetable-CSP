"""Hybrid CSP + Simulated Annealing Solver for Timetable Scheduling.

This module implements a two-stage hybrid optimization strategy:
1. Stage 1 (CSP Feasibility): Systematic Backtracking with MRV, Degree, and Forward
   Checking finds an initial, guaranteed conflict-free timetable satisfying all hard
   constraints (H1–H9 + STRUCT).
2. Stage 2 (Simulated Annealing Optimization): Local search explores the space of
   strictly conflict-free timetables via neighborhood moves (relocations and swaps),
   optimizing the comprehensive soft-cost objective (S1–S5) while rejecting any move
   that would introduce a hard violation.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from core.backtracking import solve_backtracking
from core.constraints import validate_timetable
from core.cost import CostBreakdown, SoftWeights, calculate_cost
from core.models import TimetableDataset
from core.sessions import (
    Placement,
    SessionVar,
    Timetable,
    generate_session_variables,
    get_valid_domain_for_session,
)


@dataclass
class SimulatedAnnealingResult:
    """Outcome and execution metrics of the Hybrid CSP + Simulated Annealing solver.
    
    Attributes:
        success: True iff a valid timetable (0 hard violations) was obtained.
        status: 'solved', 'limit_reached', or 'unsatisfiable'.
        timetable: Completed Timetable object (best conflict-free schedule found).
        runtime_seconds: Total elapsed wall-clock time in seconds.
        iterations: Number of simulated annealing cooling iterations executed.
        initial_cost: Objective total cost of the initial Backtracking schedule.
        final_cost: Objective total cost of the best schedule found after annealing.
        total_cost: Alias for final_cost.
        hard_violations: Total hard violations (strictly 0 on success).
        soft_penalties: Dictionary of individual soft penalty counts (S1-S5).
        temperature: Final cooling temperature reached.
        moves_accepted: Total neighborhood moves accepted (downhill and uphill).
        uphill_moves_accepted: Number of cost-increasing moves accepted probabilistically.
        random_seed: Random seed used for reproducible neighbor generation.
        failure_reason: Diagnostic message if unsuccessful.
        algorithm: Solver name identifier ("hybrid_simulated_annealing").
        cost_history: Optional history of (iteration, total_cost) tuples.
    """
    success: bool
    status: str  # "solved" | "limit_reached" | "unsatisfiable"
    timetable: Timetable | None
    runtime_seconds: float
    iterations: int
    initial_cost: int
    final_cost: int
    total_cost: int
    hard_violations: int
    soft_penalties: dict[str, int]
    temperature: float
    moves_accepted: int = 0
    uphill_moves_accepted: int = 0
    random_seed: int | None = None
    failure_reason: str | None = None
    algorithm: str = "hybrid_simulated_annealing"
    cost_history: list[tuple[int, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes result metrics to a standard dictionary."""
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "status": self.status,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "iterations": self.iterations,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "total_cost": self.total_cost,
            "hard_violations": self.hard_violations,
            "soft_penalties": self.soft_penalties,
            "temperature": round(self.temperature, 6),
            "moves_accepted": self.moves_accepted,
            "uphill_moves_accepted": self.uphill_moves_accepted,
            "random_seed": self.random_seed,
            "failure_reason": self.failure_reason,
        }


class HybridSimulatedAnnealingSolver:
    """Hybrid CSP + Simulated Annealing solver.
    
    Generates a conflict-free initial timetable using Backtracking (MRV + Degree + FC),
    then anneals soft penalties using safe neighborhood moves.
    """

    def __init__(
        self,
        dataset: TimetableDataset,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.01,
        max_iterations: int = 2000,
        time_limit_seconds: float | None = 30.0,
        random_seed: int | None = 42,
        record_cost_history: bool = True,
        soft_weights: SoftWeights | None = None,
    ) -> None:
        self.dataset = dataset
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_seconds
        self.random_seed = random_seed
        self.record_cost_history = record_cost_history
        self.soft_weights = soft_weights or SoftWeights()
        self.rng = random.Random(random_seed)

        self.variables = generate_session_variables(dataset)
        self.domains = {v: get_valid_domain_for_session(v, dataset) for v in self.variables}

    def _has_collision(
        self,
        assignments: dict[SessionVar, Placement],
        session: SessionVar,
        placement: Placement,
        excluded_sessions: set[SessionVar] | None = None,
    ) -> bool:
        """Checks if placing session at placement clashes with any other assigned session."""
        occupied_slots = placement.get_occupied_slots(session.duration)
        ex = excluded_sessions or set()

        for other_v, other_p in assignments.items():
            if other_v == session or other_v in ex or other_p is None:
                continue
            other_occupied = other_p.get_occupied_slots(other_v.duration)
            for d, p in occupied_slots:
                if (d, p) in other_occupied:
                    if (
                        other_v.teacher_id == session.teacher_id
                        or other_v.class_id == session.class_id
                        or other_p.room_id == placement.room_id
                    ):
                        return True
        return False

    def solve(self) -> SimulatedAnnealingResult:
        """Executes Hybrid CSP + Simulated Annealing."""
        start_time = time.perf_counter()

        # Step 1: Generate initial conflict-free schedule via Backtracking CSP
        bt_time_limit = self.time_limit_seconds / 2.0 if self.time_limit_seconds else 15.0
        bt_result = solve_backtracking(
            self.dataset,
            max_nodes=100_000,
            time_limit_seconds=bt_time_limit,
            use_mrv=True,
            use_degree=True,
            use_fc=True,
        )

        if not bt_result.success or bt_result.timetable is None:
            elapsed = time.perf_counter() - start_time
            return SimulatedAnnealingResult(
                success=False,
                status="unsatisfiable",
                timetable=None,
                runtime_seconds=elapsed,
                iterations=0,
                initial_cost=bt_result.total_cost,
                final_cost=bt_result.total_cost,
                total_cost=bt_result.total_cost,
                hard_violations=bt_result.hard_violations,
                soft_penalties=bt_result.soft_penalties,
                temperature=self.initial_temperature,
                random_seed=self.random_seed,
                failure_reason=(
                    "Backtracking CSP could not find an initial feasible timetable: "
                    + (bt_result.failure_reason or "search space exhausted.")
                ),
            )

        # Step 2: Validate initial timetable independently
        val_initial = validate_timetable(bt_result.timetable)
        if not val_initial.is_valid:
            elapsed = time.perf_counter() - start_time
            return SimulatedAnnealingResult(
                success=False,
                status="unsatisfiable",
                timetable=None,
                runtime_seconds=elapsed,
                iterations=0,
                initial_cost=bt_result.total_cost,
                final_cost=bt_result.total_cost,
                total_cost=bt_result.total_cost,
                hard_violations=val_initial.hard_violation_count,
                soft_penalties=bt_result.soft_penalties,
                temperature=self.initial_temperature,
                random_seed=self.random_seed,
                failure_reason="Initial Backtracking timetable failed independent validation.",
            )

        initial_cost_info = calculate_cost(bt_result.timetable, self.soft_weights)
        initial_cost = initial_cost_info.total_cost

        # Step 3: Simulated Annealing Setup
        current_assignments = dict(bt_result.timetable.assignments)
        best_assignments = dict(current_assignments)
        current_cost = initial_cost
        best_cost = initial_cost
        temperature = self.initial_temperature

        moves_accepted = 0
        uphill_moves_accepted = 0
        cost_history: list[tuple[int, int]] = []
        if self.record_cost_history:
            cost_history.append((0, initial_cost))

        # Step 4: Annealing Loop
        iteration = 0
        while iteration < self.max_iterations and temperature > self.min_temperature:
            if self.time_limit_seconds is not None:
                if (time.perf_counter() - start_time) >= self.time_limit_seconds:
                    break

            iteration += 1

            # Decide move strategy: 70% single session move, 30% pairwise swap
            use_swap = (self.rng.random() < 0.30) and len(self.variables) >= 2

            if use_swap:
                # Pairwise swap between two sessions of identical duration
                v1, v2 = self.rng.sample(self.variables, 2)
                p1, p2 = current_assignments[v1], current_assignments[v2]
                if v1.duration != v2.duration or p1 == p2:
                    continue

                # Placements p2 and p1 must belong to respective unary domains
                if p2 not in self.domains[v1] or p1 not in self.domains[v2]:
                    continue

                # Check if swap preserves 0 hard conflicts
                temp_assignments = dict(current_assignments)
                del temp_assignments[v1]
                del temp_assignments[v2]

                if self._has_collision(temp_assignments, v1, p2) or self._has_collision(
                    temp_assignments, v2, p1
                ):
                    continue

                # Mutate candidate
                candidate_assignments = dict(current_assignments)
                candidate_assignments[v1] = p2
                candidate_assignments[v2] = p1

            else:
                # Single session move
                v = self.rng.choice(self.variables)
                current_p = current_assignments[v]
                domain = self.domains[v]
                if len(domain) <= 1:
                    continue

                candidate_p = self.rng.choice(domain)
                if candidate_p == current_p:
                    continue

                # Check collision with other assigned sessions
                if self._has_collision(current_assignments, v, candidate_p, excluded_sessions={v}):
                    continue

                candidate_assignments = dict(current_assignments)
                candidate_assignments[v] = candidate_p

            # Evaluate candidate timetable cost
            candidate_tt = Timetable(dataset=self.dataset, assignments=candidate_assignments)
            candidate_cost_info = calculate_cost(candidate_tt, self.soft_weights)

            if candidate_cost_info.hard_violations > 0:
                # Reject any move that introduces hard violations
                continue

            candidate_cost = candidate_cost_info.total_cost
            delta = candidate_cost - current_cost

            # Acceptance decision
            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(temperature, 1e-6))
                if self.rng.random() < prob:
                    accept = True
                    uphill_moves_accepted += 1

            if accept:
                current_assignments = candidate_assignments
                current_cost = candidate_cost
                moves_accepted += 1

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_assignments = dict(candidate_assignments)

            if self.record_cost_history and (iteration % 10 == 0 or iteration == 1):
                cost_history.append((iteration, best_cost))

            # Cool temperature
            temperature *= self.cooling_rate

        elapsed = time.perf_counter() - start_time

        # Step 5: Final Validation of Best Timetable
        best_tt = Timetable(dataset=self.dataset, assignments=best_assignments)
        val_final = validate_timetable(best_tt)
        final_cost_info = calculate_cost(best_tt, self.soft_weights)

        if not val_final.is_valid:
            # Fallback safely to initial valid schedule
            best_tt = bt_result.timetable
            val_final = validate_timetable(best_tt)
            final_cost_info = calculate_cost(best_tt, self.soft_weights)

        if self.record_cost_history:
            cost_history.append((iteration, final_cost_info.total_cost))

        return SimulatedAnnealingResult(
            success=val_final.is_valid,
            status="solved" if val_final.is_valid else "limit_reached",
            timetable=best_tt,
            runtime_seconds=elapsed,
            iterations=iteration,
            initial_cost=initial_cost,
            final_cost=final_cost_info.total_cost,
            total_cost=final_cost_info.total_cost,
            hard_violations=val_final.hard_violation_count,
            soft_penalties={
                "s1_gaps": final_cost_info.s1_gaps,
                "s2_repeats": final_cost_info.s2_repeats,
                "s3_non_consecutive": final_cost_info.s3_non_consecutive,
                "s4_teacher_preferences": final_cost_info.s4_teacher_preferences,
                "s5_teacher_consecutive": final_cost_info.s5_teacher_consecutive,
            },
            temperature=temperature,
            moves_accepted=moves_accepted,
            uphill_moves_accepted=uphill_moves_accepted,
            random_seed=self.random_seed,
            algorithm="hybrid_simulated_annealing",
            cost_history=cost_history if self.record_cost_history else None,
        )


def solve_hybrid_simulated_annealing(
    dataset: TimetableDataset,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    min_temperature: float = 0.01,
    max_iterations: int = 2000,
    time_limit_seconds: float | None = 30.0,
    random_seed: int | None = 42,
    record_cost_history: bool = True,
    soft_weights: SoftWeights | None = None,
) -> SimulatedAnnealingResult:
    """Convenience wrapper for Hybrid CSP + Simulated Annealing."""
    solver = HybridSimulatedAnnealingSolver(
        dataset=dataset,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        max_iterations=max_iterations,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        record_cost_history=record_cost_history,
        soft_weights=soft_weights,
    )
    return solver.solve()
