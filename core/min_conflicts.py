"""Min-Conflicts CSP Local Search Solver.

This module implements Algorithm 2: Min-Conflicts, a heuristic local-search
algorithm for solving the Timetable Constraint Satisfaction Problem.

Algorithm Overview:
1. Variables (X): Atomic SessionVar instances from `generate_session_variables(dataset)`.
   Theory sessions have duration 1. Lab sessions have duration 2 (or configured lab_duration),
   occupying consecutive periods on the schedule grid.
2. Domain Values (D): Unary-compatible Placements from `get_valid_domain_for_session(session, dataset)`.
   All candidate placements satisfy unary constraints: room type (H5), room capacity (H4),
   teacher availability (H6), global blocked periods (H6), class lunch breaks (H9), and grid boundaries.
3. Initial Assignment:
   - "random": Uniformly selects a random legal placement from each variable's valid domain.
     Guarantees an initial state containing conflicts for local repair.
   - "greedy": Sequentially places each variable in a placement minimizing conflicts with already-placed
     variables.
4. Conflict Scoring Function:
   For variable V and candidate placement P, the conflict score counts all binary collisions
   introduced with all OTHER assigned variables U != V across all occupied periods:
   - H1: Teacher double-booking: other sessions sharing V.teacher_id at any occupied (day, period).
   - H2: Class double-booking: other sessions sharing V.class_id at any occupied (day, period).
   - H3: Room double-booking: other sessions sharing P.room_id at any occupied (day, period).
   Multi-period sessions (e.g. 2-period labs) check all occupied periods: (day, p) and (day, p + 1).
   The session being moved is strictly excluded from conflict calculations.
5. Search & Plateau Escaping:
   At each iteration:
   - Identifies all variables currently participating in >= 1 conflict.
   - If no conflicted variables exist, a conflict-free solution is found!
   - Selects a conflicted variable (randomly or deterministically).
   - Evaluates all legal domain placements for that variable.
   - Moves to the placement minimizing conflicts.
   - Plateau Escaping:
     * Random Walk: With probability `random_walk_probability`, selects a random legal placement
       instead of a greedy minimum-conflict placement.
     * Stagnation Perturbation: If the global minimum conflict count fails to improve for
       `stagnation_limit` consecutive steps, forces a random perturbation on a conflicted variable.
     * Non-stationary Tie-breaking: When multiple placements achieve the same minimum conflict score,
       prefers moving to an alternative placement rather than staying stationary.
6. Termination:
   - Solved: When hard conflicts reach 0 and `validate_timetable(tt)` confirms 0 hard violations.
   - Limit Reached: When `max_iterations` or `time_limit_seconds` is exhausted.
   - Unsatisfiable: When any variable has an empty initial domain.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.constraints import validate_timetable
from core.cost import CostBreakdown, calculate_cost
from core.models import TimetableDataset

from core.sessions import (
    Placement,
    SessionVar,
    Timetable,
    generate_session_variables,
    get_valid_domain_for_session,
)


@dataclass
class MinConflictsResult:
    """Outcome and execution metrics of the Min-Conflicts CSP local search solver.

    Attributes:
        success: True iff a complete, valid timetable (0 hard violations) was found.
        status: 'solved', 'limit_reached', or 'unsatisfiable'.
        timetable: Completed Timetable object if solved, or best timetable found.
        runtime_seconds: Elapsed wall-clock time in seconds.
        iterations: Number of local search repair iterations executed.
        conflicted_variable_selections: Number of times a conflicted variable was chosen for move evaluation.
        candidate_placements_evaluated: Total number of candidate placement moves scored.
        initial_hard_conflicts: Number of hard conflicts in the initial assignment.
        final_hard_violations: Total hard violations according to validate_timetable.
        soft_penalties: Dictionary of soft penalty counts (S1, S2, S3).
        total_cost: Composite objective cost from calculate_cost.
        random_seed: Random seed used for reproducibility.
        failure_reason: Diagnostic message if unsuccessful.
        algorithm: Name of algorithm configuration ("min_conflicts").
        cost_history: Optional list of (iteration, total_cost) tuples for plotting.
    """
    success: bool
    status: str  # "solved" | "limit_reached" | "unsatisfiable"
    timetable: Timetable | None
    runtime_seconds: float
    iterations: int
    conflicted_variable_selections: int
    candidate_placements_evaluated: int
    initial_hard_conflicts: int
    final_hard_violations: int
    soft_penalties: dict[str, int]
    total_cost: int
    random_seed: int | None = None
    failure_reason: str | None = None
    algorithm: str = "min_conflicts"
    cost_history: list[tuple[int, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes result metrics to a dictionary."""
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "status": self.status,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "iterations": self.iterations,
            "conflicted_variable_selections": self.conflicted_variable_selections,
            "candidate_placements_evaluated": self.candidate_placements_evaluated,
            "initial_hard_conflicts": self.initial_hard_conflicts,
            "final_hard_violations": self.final_hard_violations,
            "soft_penalties": self.soft_penalties,
            "total_cost": self.total_cost,
            "random_seed": self.random_seed,
            "failure_reason": self.failure_reason,
            "cost_history": self.cost_history,
        }


class MinConflictsSolver:
    """Min-Conflicts Local Search CSP Solver for Timetable Scheduling.

    Attributes:
        dataset: The target timetable dataset.
        max_iterations: Maximum allowed local repair steps.
        time_limit_seconds: Maximum allowed wall-clock seconds.
        random_seed: Seed for the pseudo-random generator (reproducible search).
        initial_strategy: 'random' (default) or 'greedy'.
        random_walk_probability: Probability of choosing a random placement on plateau.
        stagnation_limit: Steps without improvement before forcing a plateau escape move.
        deterministic: If True, disables randomness in tie-breaking and variable selection.
    """

    def __init__(
        self,
        dataset: TimetableDataset,
        max_iterations: int = 10_000,
        time_limit_seconds: float | None = 30.0,
        random_seed: int | None = 42,
        initial_strategy: str = "random",
        random_walk_probability: float = 0.05,
        stagnation_limit: int = 150,
        deterministic: bool = False,
        record_cost_history: bool = False,
        refine_soft_costs: bool = False,
        soft_refinement_iterations: int = 500,
    ) -> None:
        self.dataset = dataset
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_seconds
        self.refine_soft_costs = refine_soft_costs
        self.soft_refinement_iterations = soft_refinement_iterations
        self.random_seed = random_seed
        self.initial_strategy = initial_strategy.lower().strip()
        self.random_walk_probability = random_walk_probability
        self.stagnation_limit = stagnation_limit
        self.deterministic = deterministic

        if self.initial_strategy not in ("random", "greedy"):
            raise ValueError(
                f"Unknown initial_strategy '{initial_strategy}'. Expected 'random' or 'greedy'."
            )

        # Initialize RNG
        self.rng = random.Random(random_seed)
        # Optional cost‑history tracking flag
        self.record_cost_history = record_cost_history

        # Variables and Unary‑Filtered Domains
        self.variables: list[SessionVar] = generate_session_variables(dataset)
        self.var_index: dict[SessionVar, int] = {v: i for i, v in enumerate(self.variables)}
        self.domains: dict[SessionVar, list[Placement]] = {
            v: get_valid_domain_for_session(v, dataset) for v in self.variables
        }

        # Current Assignment & Fast Collision Counter Tables
        self.assignments: dict[SessionVar, Placement] = {}
        self.teacher_occupancy: Counter[tuple[int, int, str]] = Counter()  # (day, period, teacher_id) -> count
        self.class_occupancy: Counter[tuple[int, int, str]] = Counter()    # (day, period, class_id) -> count
        self.room_occupancy: Counter[tuple[int, int, str]] = Counter()     # (day, period, room_id) -> count

        # Execution Metrics
        self.iterations: int = 0
        self.conflicted_variable_selections: int = 0
        self.candidate_placements_evaluated: int = 0
        self.initial_hard_conflicts: int = 0
        self.start_time: float = 0.0
        # Cost history containers (populated only if enabled)
        self.cost_history: list[tuple[int, int]] = []
        self.cost_history_detail: list[tuple[int, int, str]] = []  # (iteration, cost, "current"|"best")

    def _add_placement_to_tables(self, var: SessionVar, placement: Placement) -> None:
        """Assigns placement to var and increments occupancy counters across all occupied slots."""
        self.assignments[var] = placement
        for day, period in placement.get_occupied_slots(var.duration):
            self.teacher_occupancy[(day, period, var.teacher_id)] += 1
            self.class_occupancy[(day, period, var.class_id)] += 1
            self.room_occupancy[(day, period, placement.room_id)] += 1

    def _remove_placement_from_tables(self, var: SessionVar, placement: Placement) -> None:
        """Removes placement of var and decrements occupancy counters across all occupied slots."""
        if var in self.assignments:
            del self.assignments[var]
        for day, period in placement.get_occupied_slots(var.duration):
            self.teacher_occupancy[(day, period, var.teacher_id)] -= 1
            if self.teacher_occupancy[(day, period, var.teacher_id)] == 0:
                del self.teacher_occupancy[(day, period, var.teacher_id)]

            self.class_occupancy[(day, period, var.class_id)] -= 1
            if self.class_occupancy[(day, period, var.class_id)] == 0:
                del self.class_occupancy[(day, period, var.class_id)]

            self.room_occupancy[(day, period, placement.room_id)] -= 1
            if self.room_occupancy[(day, period, placement.room_id)] == 0:
                del self.room_occupancy[(day, period, placement.room_id)]

    def score_placement_conflicts(self, var: SessionVar, placement: Placement) -> int:
        """Calculates hard conflicts candidate placement would introduce with all OTHER variables.

        The session `var` is assumed already excluded from occupancy tables before calling this.
        Covers H1 (Teacher double-booking), H2 (Class double-booking), and H3 (Room double-booking)
        across all occupied periods of the candidate placement.

        Counting rule:
        For each occupied slot (day, period) in placement:
        - +1 per other placed session sharing var.teacher_id (H1).
        - +1 per other placed session sharing var.class_id (H2).
        - +1 per other placed session sharing placement.room_id (H3).

        Returns:
            Non-negative integer conflict count. 0 indicates perfect compatibility.
        """
        conflicts = 0
        for day, period in placement.get_occupied_slots(var.duration):
            conflicts += self.teacher_occupancy.get((day, period, var.teacher_id), 0)
            conflicts += self.class_occupancy.get((day, period, var.class_id), 0)
            conflicts += self.room_occupancy.get((day, period, placement.room_id), 0)
        return conflicts

    def get_variable_conflicts(self, var: SessionVar) -> int:
        """Returns the number of conflicting relationships var currently participates in.

        Evaluates var's current placement against all other assigned variables.
        """
        placement = self.assignments.get(var)
        if placement is None:
            return 0
        conflicts = 0
        for day, period in placement.get_occupied_slots(var.duration):
            # Subtract 1 because var itself is currently counted in the occupancy tables
            conflicts += (self.teacher_occupancy[(day, period, var.teacher_id)] - 1)
            conflicts += (self.class_occupancy[(day, period, var.class_id)] - 1)
            conflicts += (self.room_occupancy[(day, period, placement.room_id)] - 1)
        return conflicts

    def count_total_timetable_conflicts(self) -> int:
        """Calculates total atomic H1, H2, and H3 pairwise clashes in the current timetable.

        Uses the standard combination formula sum(k * (k - 1) / 2) over all occupancy counters.
        This corresponds exactly to the count of hard violations reported by check_h1, check_h2,
        and check_h3 in core.constraints.
        """
        total = 0
        for count in self.teacher_occupancy.values():
            if count > 1:
                total += (count * (count - 1)) // 2
        for count in self.class_occupancy.values():
            if count > 1:
                total += (count * (count - 1)) // 2
        for count in self.room_occupancy.values():
            if count > 1:
                total += (count * (count - 1)) // 2
        return total

    def _initialize_assignment(self) -> None:
        """Constructs the initial complete assignment according to initial_strategy."""
        self.assignments.clear()
        self.teacher_occupancy.clear()
        self.class_occupancy.clear()
        self.room_occupancy.clear()

        if self.initial_strategy == "greedy":
            # Greedy: assign each variable to the placement that minimizes conflicts with
            # already assigned variables
            for var in self.variables:
                domain = self.domains[var]
                candidate_scores = [(self.score_placement_conflicts(var, p), p) for p in domain]
                self.candidate_placements_evaluated += len(domain)
                min_c = min(s for s, p in candidate_scores)
                best_candidates = [p for s, p in candidate_scores if s == min_c]

                if self.deterministic or len(best_candidates) == 1:
                    chosen = best_candidates[0]
                else:
                    chosen = self.rng.choice(best_candidates)
                self._add_placement_to_tables(var, chosen)
        else:
            # Random: assign each variable to a uniformly chosen legal placement from its domain
            for var in self.variables:
                domain = self.domains[var]
                if self.deterministic:
                    chosen = domain[0]
                else:
                    chosen = self.rng.choice(domain)
                self._add_placement_to_tables(var, chosen)

    def _select_conflicted_variable(self, conflicted_vars: list[SessionVar]) -> SessionVar:
        """Selects a conflicted variable to repair on this iteration."""
        self.conflicted_variable_selections += 1
        if self.deterministic:
            # Deterministic: select the variable with the highest conflict count,
            # breaking ties by static variable index
            return max(
                conflicted_vars,
                key=lambda v: (self.get_variable_conflicts(v), -self.var_index[v]),
            )
        return self.rng.choice(conflicted_vars)

    def _choose_best_placement(
        self, var: SessionVar, current_placement: Placement, force_random_walk: bool = False
    ) -> Placement:
        """Evaluates all candidate placements for var and selects the move minimizing conflicts."""
        domain = self.domains[var]

        # Random walk for plateau escaping
        if force_random_walk and not self.deterministic and len(domain) > 1:
            alternatives = [p for p in domain if p != current_placement]
            return self.rng.choice(alternatives) if alternatives else domain[0]

        # Score every placement in the legal domain (var is currently removed from tables)
        scores: list[tuple[int, Placement]] = []
        for p in domain:
            c = self.score_placement_conflicts(var, p)
            scores.append((c, p))
            self.candidate_placements_evaluated += 1

        min_conflict = min(s for s, p in scores)
        best_placements = [p for s, p in scores if s == min_conflict]

        if len(best_placements) == 1:
            return best_placements[0]

        if self.deterministic:
            # Deterministic tie-breaker: first in domain order
            return best_placements[0]

        # If current_placement is among the best, prefer moving to an alternative best placement
        # with probability 0.8 to prevent stationary zero-move loops on flat plateaus
        alternatives = [p for p in best_placements if p != current_placement]
        if alternatives and self.rng.random() < 0.8:
            return self.rng.choice(alternatives)

        return self.rng.choice(best_placements)

    def _refine_soft_quality(
        self, initial_valid_tt: Timetable
    ) -> tuple[Timetable, CostBreakdown, int]:
        """Stage B: Local search refinement to optimize soft constraint quality.

        Starting from a conflict-free valid timetable, iteratively tests safe
        placement moves that preserve 0 hard violations while seeking lower soft cost.
        Never introduces hard conflicts. Preserves the best valid timetable found.
        """
        best_tt = initial_valid_tt
        best_cost_info = calculate_cost(best_tt)
        current_assignments = dict(initial_valid_tt.assignments)

        refinement_steps = 0
        vars_list = list(self.variables)

        for step in range(1, self.soft_refinement_iterations + 1):
            if self.time_limit_seconds is not None:
                if (time.perf_counter() - self.start_time) >= self.time_limit_seconds:
                    break
            refinement_steps += 1

            # Select a variable
            if self.deterministic:
                var = vars_list[step % len(vars_list)]
            else:
                var = self.rng.choice(vars_list)

            current_placement = current_assignments[var]
            domain = self.domains[var]
            if len(domain) <= 1:
                continue

            # Pick a candidate alternative placement
            if self.deterministic:
                candidate = domain[step % len(domain)]
            else:
                candidate = self.rng.choice(domain)

            if candidate == current_placement:
                continue

            # Test if candidate move creates any hard collision (H1, H2, H3) with other sessions
            has_collision = False
            for d, p in candidate.get_occupied_slots(var.duration):
                for other_v, other_pl in current_assignments.items():
                    if other_v == var or other_pl is None:
                        continue
                    if (d, p) in other_pl.get_occupied_slots(other_v.duration):
                        if (
                            other_v.teacher_id == var.teacher_id
                            or other_v.class_id == var.class_id
                            or other_pl.room_id == candidate.room_id
                        ):
                            has_collision = True
                            break
                if has_collision:
                    break

            if has_collision:
                continue

            # Feasible move! Check cost
            current_assignments[var] = candidate
            trial_tt = Timetable(dataset=self.dataset, assignments=dict(current_assignments))
            trial_cost_info = calculate_cost(trial_tt)

            if trial_cost_info.hard_violations == 0 and trial_cost_info.total_cost < best_cost_info.total_cost:
                best_tt = trial_tt
                best_cost_info = trial_cost_info
                if self.record_cost_history:
                    self.cost_history.append((self.iterations + step, trial_cost_info.total_cost))
            elif trial_cost_info.hard_violations == 0 and trial_cost_info.total_cost == best_cost_info.total_cost:
                if not self.deterministic and self.rng.random() < 0.20:
                    pass
                else:
                    current_assignments[var] = current_placement
            else:
                current_assignments[var] = current_placement

        return best_tt, best_cost_info, refinement_steps

    def solve(self) -> MinConflictsResult:
        """Executes the Min-Conflicts local search algorithm.

        Returns:
            MinConflictsResult with outcome status, solved timetable (if valid),
            and detailed search metrics.
        """
        self.start_time = time.perf_counter()
        self.iterations = 0
        self.conflicted_variable_selections = 0
        self.candidate_placements_evaluated = 0
        # Store cost per iteration for experimental analysis
        self.cost_history: list[tuple[int, int]] = []

        # 1. Check for empty domains upfront
        for var, domain in self.domains.items():
            if len(domain) == 0:
                elapsed = time.perf_counter() - self.start_time
                failure_msg = (
                    f"Session variable '{var.session_id}' has an empty domain (0 valid placements)."
                )
                partial_tt = Timetable(dataset=self.dataset, assignments={})
                cost_info = calculate_cost(partial_tt)
                result = MinConflictsResult(
                    success=False,
                    status="unsatisfiable",
                    timetable=None,
                    runtime_seconds=elapsed,
                    iterations=0,
                    conflicted_variable_selections=0,
                    candidate_placements_evaluated=0,
                    initial_hard_conflicts=0,
                    final_hard_violations=cost_info.hard_violations,
                    soft_penalties={
                        "s1_gaps": cost_info.s1_gaps,
                        "s2_repeats": cost_info.s2_repeats,
                        "s3_non_consecutive": cost_info.s3_non_consecutive,
                        "s4_teacher_preferences": cost_info.s4_teacher_preferences,
                        "s5_teacher_consecutive": cost_info.s5_teacher_consecutive,
                    },
                    total_cost=cost_info.total_cost,
                    random_seed=self.random_seed,
                    failure_reason=failure_msg,
                )
                if self.record_cost_history:
                    self.cost_history.append((0, cost_info.total_cost))
                    self.cost_history_detail.append((0, cost_info.total_cost, "current"))
                return result

        # 2. Construct initial assignment
        self._initialize_assignment()
        self.initial_hard_conflicts = self.count_total_timetable_conflicts()

        # Track best state found during search
        best_conflicts = self.initial_hard_conflicts
        best_assignments: dict[SessionVar, Placement] = dict(self.assignments)
        stagnation_counter = 0

        # Check if initial assignment is already conflict-free
        if self.initial_hard_conflicts == 0:
            tt = Timetable(dataset=self.dataset, assignments=dict(self.assignments))
            val = validate_timetable(tt)
            if val.is_valid:
                if self.refine_soft_costs:
                    tt, cost_info, extra_steps = self._refine_soft_quality(tt)
                    self.iterations += extra_steps
                else:
                    cost_info = calculate_cost(tt)
                elapsed = time.perf_counter() - self.start_time
                # Record initial cost (iteration 0)
                self.cost_history.append((self.iterations, cost_info.total_cost))
                return MinConflictsResult(
                    success=True,
                    status="solved",
                    timetable=tt,
                    runtime_seconds=elapsed,
                    iterations=self.iterations,
                    conflicted_variable_selections=0,
                    candidate_placements_evaluated=self.candidate_placements_evaluated,
                    initial_hard_conflicts=self.initial_hard_conflicts,
                    final_hard_violations=0,
                    soft_penalties={
                        "s1_gaps": cost_info.s1_gaps,
                        "s2_repeats": cost_info.s2_repeats,
                        "s3_non_consecutive": cost_info.s3_non_consecutive,
                        "s4_teacher_preferences": cost_info.s4_teacher_preferences,
                        "s5_teacher_consecutive": cost_info.s5_teacher_consecutive,
                    },
                    total_cost=cost_info.total_cost,
                    random_seed=self.random_seed,
                    cost_history=self.cost_history,
                )

        # 3. Min-Conflicts Repair Loop
        status = "limit_reached"
        failure_reason: str | None = None

        for iteration in range(1, self.max_iterations + 1):
            self.iterations = iteration

            # Check time budget
            if self.time_limit_seconds is not None:
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.time_limit_seconds:
                    failure_reason = f"Time limit exceeded ({self.time_limit_seconds:.2f}s)."
                    break

            # Find all conflicted variables
            conflicted_vars = [v for v in self.variables if self.get_variable_conflicts(v) > 0]

            if not conflicted_vars:
                # 0 conflicts! Verify with validate_timetable
                tt = Timetable(dataset=self.dataset, assignments=dict(self.assignments))
                val = validate_timetable(tt)
                if val.is_valid:
                    if self.refine_soft_costs:
                        tt, cost_info, extra_steps = self._refine_soft_quality(tt)
                        self.iterations += extra_steps
                    else:
                        cost_info = calculate_cost(tt)
                    elapsed = time.perf_counter() - self.start_time
                    result = MinConflictsResult(
                        success=True,
                        status="solved",
                        timetable=tt,
                        runtime_seconds=elapsed,
                        iterations=self.iterations,
                        conflicted_variable_selections=self.conflicted_variable_selections,
                        candidate_placements_evaluated=self.candidate_placements_evaluated,
                        initial_hard_conflicts=self.initial_hard_conflicts,
                        final_hard_violations=0,
                        soft_penalties={
                            "s1_gaps": cost_info.s1_gaps,
                            "s2_repeats": cost_info.s2_repeats,
                            "s3_non_consecutive": cost_info.s3_non_consecutive,
                            "s4_teacher_preferences": cost_info.s4_teacher_preferences,
                            "s5_teacher_consecutive": cost_info.s5_teacher_consecutive,
                        },
                        total_cost=cost_info.total_cost,
                        random_seed=self.random_seed,
                    )
                    if self.record_cost_history:
                        self.cost_history.append((self.iterations, cost_info.total_cost))
                        self.cost_history_detail.append((self.iterations, cost_info.total_cost, "current"))
                    return result

            # Select a conflicted variable
            var = self._select_conflicted_variable(conflicted_vars)
            current_placement = self.assignments[var]

            # Exclude var from tables before evaluating candidate moves
            self._remove_placement_from_tables(var, current_placement)

            # Determine if random walk is triggered (probabilistic or stagnation-based)
            force_random_walk = False
            if not self.deterministic:
                if stagnation_counter >= self.stagnation_limit:
                    force_random_walk = True
                    stagnation_counter = 0
                elif self.rng.random() < self.random_walk_probability:
                    force_random_walk = True

            # Choose placement minimizing conflicts
            new_placement = self._choose_best_placement(
                var, current_placement, force_random_walk=force_random_walk
            )

            # Apply new placement
            self._add_placement_to_tables(var, new_placement)

            # Update best state & stagnation counter
            current_conflicts = self.count_total_timetable_conflicts()
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_assignments = dict(self.assignments)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Record cost history for this iteration (if enabled)
            if self.record_cost_history:
                current_tt = Timetable(dataset=self.dataset, assignments=dict(self.assignments))
                current_cost_info = calculate_cost(current_tt)
                self.cost_history.append((self.iterations, current_cost_info.total_cost))
                self.cost_history_detail.append((self.iterations, current_cost_info.total_cost, "current"))

            if current_conflicts == 0:
                tt = Timetable(dataset=self.dataset, assignments=dict(self.assignments))
                val = validate_timetable(tt)
                if val.is_valid:
                    if self.refine_soft_costs:
                        tt, cost_info, extra_steps = self._refine_soft_quality(tt)
                        self.iterations += extra_steps
                    else:
                        cost_info = calculate_cost(tt)
                    elapsed = time.perf_counter() - self.start_time
                    return MinConflictsResult(
                        success=True,
                        status="solved",
                        timetable=tt,
                        runtime_seconds=elapsed,
                        iterations=self.iterations,
                        conflicted_variable_selections=self.conflicted_variable_selections,
                        candidate_placements_evaluated=self.candidate_placements_evaluated,
                        initial_hard_conflicts=self.initial_hard_conflicts,
                        final_hard_violations=0,
                        soft_penalties={
                            "s1_gaps": cost_info.s1_gaps,
                            "s2_repeats": cost_info.s2_repeats,
                            "s3_non_consecutive": cost_info.s3_non_consecutive,
                            "s4_teacher_preferences": cost_info.s4_teacher_preferences,
                            "s5_teacher_consecutive": cost_info.s5_teacher_consecutive,
                        },
                        total_cost=cost_info.total_cost,
                        random_seed=self.random_seed,
                        cost_history=self.cost_history if self.record_cost_history else None,
                    )

        # Search budget exhausted without reaching 0 hard conflicts
        elapsed = time.perf_counter() - self.start_time
        if failure_reason is None:
            failure_reason = (
                f"Maximum iteration limit ({self.max_iterations}) reached without finding a conflict-free schedule "
                f"(best conflicts achieved: {best_conflicts})."
            )

        best_tt = Timetable(dataset=self.dataset, assignments=dict(best_assignments))
        val_res = validate_timetable(best_tt)
        cost_info = calculate_cost(best_tt)

        return MinConflictsResult(
            success=False,
            status=status,
            timetable=best_tt,
            runtime_seconds=elapsed,
            iterations=self.iterations,
            conflicted_variable_selections=self.conflicted_variable_selections,
            candidate_placements_evaluated=self.candidate_placements_evaluated,
            initial_hard_conflicts=self.initial_hard_conflicts,
            final_hard_violations=val_res.hard_violation_count,
            soft_penalties={
                "s1_gaps": cost_info.s1_gaps,
                "s2_repeats": cost_info.s2_repeats,
                "s3_non_consecutive": cost_info.s3_non_consecutive,
                "s4_teacher_preferences": cost_info.s4_teacher_preferences,
                "s5_teacher_consecutive": cost_info.s5_teacher_consecutive,
            },
            total_cost=cost_info.total_cost,
            random_seed=self.random_seed,
            failure_reason=failure_reason,
            cost_history=self.cost_history,
        )


def solve_min_conflicts(
    dataset: TimetableDataset,
    max_iterations: int = 10_000,
    time_limit_seconds: float | None = 30.0,
    random_seed: int | None = 42,
    initial_strategy: str = "random",
    random_walk_probability: float = 0.05,
    stagnation_limit: int = 150,
    deterministic: bool = False,
    record_cost_history: bool = False,
    refine_soft_costs: bool = False,
    soft_refinement_iterations: int = 500,
) -> MinConflictsResult:
    """Convenience wrapper to solve a timetable dataset using Min-Conflicts CSP local search.

    Args:
        dataset: The target timetable dataset.
        max_iterations: Maximum allowed local repair steps (default: 10,000).
        time_limit_seconds: Maximum allowed wall-clock seconds (default: 30.0s).
        random_seed: Seed for random tie-breaking and random moves (default: 42).
        initial_strategy: Initial assignment mode: 'random' (default) or 'greedy'.
        random_walk_probability: Probability of random walk for plateau escaping (default: 0.05).
        stagnation_limit: Stagnation iterations before forced perturbation (default: 150).
        deterministic: If True, uses deterministic variable and placement selection.
        record_cost_history: If True, records (iteration, total_cost) tuples during search.

    Returns:
        MinConflictsResult object containing outcome, metrics, and timetable.
    """
    solver = MinConflictsSolver(
        dataset=dataset,
        max_iterations=max_iterations,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        initial_strategy=initial_strategy,
        random_walk_probability=random_walk_probability,
        stagnation_limit=stagnation_limit,
        deterministic=deterministic,
        record_cost_history=record_cost_history,
        refine_soft_costs=refine_soft_costs,
        soft_refinement_iterations=soft_refinement_iterations,
    )
    return solver.solve()
