"""Tests for Min-Conflicts Local Search CSP Solver (core/min_conflicts.py).

Verifies:
1. Solving small and large benchmark datasets.
2. Final schedules strictly pass validate_timetable (0 hard violations).
3. Initial assignments contain genuine conflicts under random strategy.
4. Isolated conflict scoring for H1 (teacher), H2 (class), H3 (room).
5. Multi-period lab overlap conflict scoring across all occupied periods.
6. Correct exclusion of the session being moved from its own conflict calculations.
7. Empty domain detection and unsatisfiable status.
8. Iteration limit enforcement (limit_reached).
9. Time limit enforcement (limit_reached).
10. Plateau escaping and cycle prevention.
11. Deterministic and randomized reproducibility with seeds.
12. Comprehensive result metric definitions.
"""

from __future__ import annotations

import pytest

from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.min_conflicts import (
    MinConflictsResult,
    MinConflictsSolver,
    solve_min_conflicts,
)
from core.models import (
    ClassGroup,
    ClassSubjectRequirement,
    Room,
    Subject,
    Teacher,
    TimeStructure,
    TimetableDataset,
)
from core.sessions import Placement, SessionVar, Timetable


# =============================================================================
# BENCHMARK DATASET TESTS
# =============================================================================


def test_solve_small_dataset_min_conflicts(small_dataset: TimetableDataset):
    """Verifies that Min-Conflicts solves small_dataset.json to 0 hard violations."""
    result = solve_min_conflicts(small_dataset, max_iterations=5000, random_seed=42)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "min_conflicts"
    assert result.timetable is not None
    assert result.runtime_seconds >= 0.0
    assert result.iterations > 0
    assert result.conflicted_variable_selections > 0
    assert result.candidate_placements_evaluated > 0
    assert result.initial_hard_conflicts > 0
    assert result.final_hard_violations == 0
    assert result.failure_reason is None

    # Validate with independent validator
    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0


def test_solve_large_dataset_min_conflicts(large_dataset: TimetableDataset):
    """Verifies that Min-Conflicts solves large_dataset.json to 0 hard violations."""
    result = solve_min_conflicts(
        large_dataset, max_iterations=10_000, time_limit_seconds=30.0, random_seed=42
    )

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "min_conflicts"
    assert result.timetable is not None
    assert result.final_hard_violations == 0

    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0


def test_min_conflicts_solution_passes_independent_validator(small_dataset: TimetableDataset):
    """Independently checks all pairwise placements for teacher, class, and room clashes."""
    result = solve_min_conflicts(small_dataset, random_seed=42)
    assert result.timetable is not None

    placed = result.timetable.get_placed_sessions()
    for i in range(len(placed)):
        var_a, place_a = placed[i]
        slots_a = set(place_a.get_occupied_slots(var_a.duration))

        for j in range(i + 1, len(placed)):
            var_b, place_b = placed[j]
            slots_b = set(place_b.get_occupied_slots(var_b.duration))

            overlap = slots_a.intersection(slots_b)
            if overlap:
                assert var_a.teacher_id != var_b.teacher_id, (
                    f"Teacher clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert var_a.class_id != var_b.class_id, (
                    f"Class clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert place_a.room_id != place_b.room_id, (
                    f"Room clash between {var_a.session_id} and {var_b.session_id} in {place_a.room_id} at {overlap}"
                )


def test_initial_assignment_contains_conflicts_under_random_strategy(small_dataset: TimetableDataset):
    """Verifies that random initial assignment generates an initial state with hard conflicts."""
    solver = MinConflictsSolver(small_dataset, initial_strategy="random", random_seed=42)
    solver._initialize_assignment()
    initial_conflicts = solver.count_total_timetable_conflicts()

    assert initial_conflicts > 0, "Random initial assignment must produce non-zero initial conflicts."


def test_greedy_initial_strategy_supported(small_dataset: TimetableDataset):
    """Verifies that greedy initial strategy is supported and solves cleanly."""
    result = solve_min_conflicts(small_dataset, initial_strategy="greedy", random_seed=42)
    assert result.success is True
    assert result.final_hard_violations == 0


# =============================================================================
# ISOLATED CONFLICT SCORING TESTS
# =============================================================================


def test_h1_teacher_conflict_scoring_isolated():
    """Verifies that conflict scoring accurately detects teacher clashes (H1)."""
    dataset = TimetableDataset(
        name="h1_conflict_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=3),
        subjects={
            "S1": Subject(id="S1", name="Sub 1", weekly_periods=1, type="theory"),
            "S2": Subject(id="S2", name="Sub 2", weekly_periods=1, type="theory"),
        },
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["S1", "S2"])},
        classes={
            "C1": ClassGroup(id="C1", name="C1", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S1", teacher_id="T1")]),
            "C2": ClassGroup(id="C2", name="C2", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S2", teacher_id="T1")]),
        },
        rooms={
            "R1": Room(id="R1", name="R1", capacity=30, type="classroom"),
            "R2": Room(id="R2", name="R2", capacity=30, type="classroom"),
        },
    )

    solver = MinConflictsSolver(dataset)
    v1, v2 = solver.variables

    # Place v1 at day 0, period 0, room R1
    p1 = Placement(day=0, start_period=0, room_id="R1")
    solver._add_placement_to_tables(v1, p1)

    # Candidate placements for v2 (sharing Teacher T1):
    # Clashing placement: day 0, period 0, room R2 (different room, but same teacher T1!)
    p_clash = Placement(day=0, start_period=0, room_id="R2")
    # Compatible placement: day 0, period 1, room R2
    p_ok = Placement(day=0, start_period=1, room_id="R2")

    assert solver.score_placement_conflicts(v2, p_clash) == 1
    assert solver.score_placement_conflicts(v2, p_ok) == 0


def test_h2_class_conflict_scoring_isolated():
    """Verifies that conflict scoring accurately detects class clashes (H2)."""
    dataset = TimetableDataset(
        name="h2_conflict_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=3),
        subjects={
            "S1": Subject(id="S1", name="Sub 1", weekly_periods=1, type="theory"),
            "S2": Subject(id="S2", name="Sub 2", weekly_periods=1, type="theory"),
        },
        teachers={
            "T1": Teacher(id="T1", name="Teacher 1", subject_ids=["S1"]),
            "T2": Teacher(id="T2", name="Teacher 2", subject_ids=["S2"]),
        },
        classes={
            "C1": ClassGroup(
                id="C1", name="C1", strength=20, lunch_slot=99,
                subjects=[
                    ClassSubjectRequirement(subject_id="S1", teacher_id="T1"),
                    ClassSubjectRequirement(subject_id="S2", teacher_id="T2"),
                ],
            ),
        },
        rooms={
            "R1": Room(id="R1", name="R1", capacity=30, type="classroom"),
            "R2": Room(id="R2", name="R2", capacity=30, type="classroom"),
        },
    )

    solver = MinConflictsSolver(dataset)
    v1, v2 = solver.variables

    # Place v1 at day 0, period 0, room R1
    p1 = Placement(day=0, start_period=0, room_id="R1")
    solver._add_placement_to_tables(v1, p1)

    # Candidate placements for v2 (sharing Class C1, different teacher, different room):
    p_clash = Placement(day=0, start_period=0, room_id="R2")
    p_ok = Placement(day=0, start_period=1, room_id="R2")

    assert solver.score_placement_conflicts(v2, p_clash) == 1
    assert solver.score_placement_conflicts(v2, p_ok) == 0


def test_h3_room_conflict_scoring_isolated():
    """Verifies that conflict scoring accurately detects room clashes (H3)."""
    dataset = TimetableDataset(
        name="h3_conflict_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=3),
        subjects={
            "S1": Subject(id="S1", name="Sub 1", weekly_periods=1, type="theory"),
            "S2": Subject(id="S2", name="Sub 2", weekly_periods=1, type="theory"),
        },
        teachers={
            "T1": Teacher(id="T1", name="Teacher 1", subject_ids=["S1"]),
            "T2": Teacher(id="T2", name="Teacher 2", subject_ids=["S2"]),
        },
        classes={
            "C1": ClassGroup(id="C1", name="C1", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S1", teacher_id="T1")]),
            "C2": ClassGroup(id="C2", name="C2", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S2", teacher_id="T2")]),
        },
        rooms={
            "R1": Room(id="R1", name="R1", capacity=30, type="classroom"),
            "R2": Room(id="R2", name="R2", capacity=30, type="classroom"),
        },
    )

    solver = MinConflictsSolver(dataset)
    v1, v2 = solver.variables

    # Place v1 at day 0, period 0, room R1
    p1 = Placement(day=0, start_period=0, room_id="R1")
    solver._add_placement_to_tables(v1, p1)

    # Candidate placements for v2 (different class C2, different teacher T2, same room R1):
    p_clash = Placement(day=0, start_period=0, room_id="R1")
    p_ok = Placement(day=0, start_period=0, room_id="R2")

    assert solver.score_placement_conflicts(v2, p_clash) == 1
    assert solver.score_placement_conflicts(v2, p_ok) == 0


def test_multi_period_lab_overlap_conflict_scoring():
    """Verifies that multi-period lab sessions account for all occupied periods in conflict scoring."""
    dataset = TimetableDataset(
        name="lab_conflict_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=4),
        subjects={
            "LAB1": Subject(id="LAB1", name="Lab 1", weekly_periods=2, type="lab", lab_duration=2),
            "THEO1": Subject(id="THEO1", name="Theo 1", weekly_periods=1, type="theory"),
        },
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["LAB1", "THEO1"])},
        classes={
            "C1": ClassGroup(
                id="C1", name="C1", strength=20, lunch_slot=99,
                subjects=[
                    ClassSubjectRequirement(subject_id="LAB1", teacher_id="T1"),
                    ClassSubjectRequirement(subject_id="THEO1", teacher_id="T1"),
                ],
            ),
        },
        rooms={
            "R_LAB": Room(id="R_LAB", name="Lab Room", capacity=30, type="lab"),
            "R_CLASS": Room(id="R_CLASS", name="Classroom", capacity=30, type="classroom"),
        },
    )

    solver = MinConflictsSolver(dataset)
    lab_var = [v for v in solver.variables if v.session_type == "lab"][0]
    theo_var = [v for v in solver.variables if v.session_type == "theory"][0]

    # Assign lab to period 1 (occupies period 1 AND period 2)
    lab_p = Placement(day=0, start_period=1, room_id="R_LAB")
    solver._add_placement_to_tables(lab_var, lab_p)

    # Test theo_var conflict score at:
    # Period 0: no overlap -> 0
    p0 = Placement(day=0, start_period=0, room_id="R_CLASS")
    assert solver.score_placement_conflicts(theo_var, p0) == 0

    # Period 1: overlaps lab's first period -> conflicts on Teacher and Class (each gives +1 -> total 2)
    p1 = Placement(day=0, start_period=1, room_id="R_CLASS")
    assert solver.score_placement_conflicts(theo_var, p1) == 2

    # Period 2: overlaps lab's second period -> conflicts on Teacher and Class (total 2)
    p2 = Placement(day=0, start_period=2, room_id="R_CLASS")
    assert solver.score_placement_conflicts(theo_var, p2) == 2

    # Period 3: no overlap -> 0
    p3 = Placement(day=0, start_period=3, room_id="R_CLASS")
    assert solver.score_placement_conflicts(theo_var, p3) == 0


def test_moving_session_excluded_from_its_own_conflict_score(small_dataset: TimetableDataset):
    """Verifies that a variable is not counted as conflicting with itself when evaluating moves."""
    solver = MinConflictsSolver(small_dataset, random_seed=42)
    var = solver.variables[0]
    p = solver.domains[var][0]

    # When no other session is in the tables, scoring p for var must return 0
    assert solver.score_placement_conflicts(var, p) == 0

    # If we add var at p, and then remove it before scoring, it still scores 0
    solver._add_placement_to_tables(var, p)
    assert solver.get_variable_conflicts(var) == 0  # isolated, no other session

    solver._remove_placement_from_tables(var, p)
    assert solver.score_placement_conflicts(var, p) == 0


# =============================================================================
# EDGE CASES, BUDGETS, AND METRICS
# =============================================================================


def test_empty_domain_returns_unsatisfiable():
    """Verifies that an instance with an empty domain returns unsatisfiable without searching."""
    dataset = TimetableDataset(
        name="empty_domain_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=1),
        subjects={"S1": Subject(id="S1", name="S1", weekly_periods=1, type="theory")},
        teachers={"T1": Teacher(id="T1", name="T1", subject_ids=["S1"])},
        classes={
            "C1": ClassGroup(
                id="C1", name="C1", strength=50, lunch_slot=99,
                subjects=[ClassSubjectRequirement(subject_id="S1", teacher_id="T1")],
            ),
        },
        # Room capacity 30 < class strength 50 -> 0 valid placements!
        rooms={"R1": Room(id="R1", name="R1", capacity=30, type="classroom")},
    )

    result = solve_min_conflicts(dataset)
    assert result.success is False
    assert result.status == "unsatisfiable"
    assert result.iterations == 0
    assert "empty domain" in result.failure_reason.lower()


def test_max_iterations_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exhausting max_iterations returns limit_reached."""
    result = solve_min_conflicts(
        small_dataset, max_iterations=1, initial_strategy="random", random_seed=42
    )

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.iterations == 1
    assert "iteration limit" in result.failure_reason.lower() and "reached" in result.failure_reason.lower()
    assert result.final_hard_violations > 0


def test_plateau_escaping_synthetic_trap():
    """Synthetic test demonstrating that plateau escaping (random walk and non-stationary tie-breaking)
    allows Min-Conflicts to escape a local plateau and find a conflict-free solution."""
    # Setup: 3 classes, 3 teachers, 2 classrooms, 2 periods on 1 day.
    # Total capacity = 4 slots. Total sessions = 3 sessions.
    # Pigeonhole-free (feasible!), but if sessions are grouped awkwardly on slot 0,
    # moving one session from (0,0) to (0,0,R2) or (0,1) might initially maintain equal conflict count.
    dataset = TimetableDataset(
        name="plateau_test",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=2),
        subjects={
            "S1": Subject(id="S1", name="S1", weekly_periods=1, type="theory"),
            "S2": Subject(id="S2", name="S2", weekly_periods=1, type="theory"),
            "S3": Subject(id="S3", name="S3", weekly_periods=1, type="theory"),
        },
        teachers={
            "T1": Teacher(id="T1", name="T1", subject_ids=["S1"]),
            "T2": Teacher(id="T2", name="T2", subject_ids=["S2"]),
            "T3": Teacher(id="T3", name="T3", subject_ids=["S3"]),
        },
        classes={
            "C1": ClassGroup(id="C1", name="C1", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S1", teacher_id="T1")]),
            "C2": ClassGroup(id="C2", name="C2", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S2", teacher_id="T2")]),
            "C3": ClassGroup(id="C3", name="C3", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S3", teacher_id="T3")]),
        },
        rooms={
            "R1": Room(id="R1", name="R1", capacity=30, type="classroom"),
            "R2": Room(id="R2", name="R2", capacity=30, type="classroom"),
        },
    )

    # With random walk probability and stagnation limit active, Min-Conflicts navigates the plateau
    result = solve_min_conflicts(
        dataset,
        max_iterations=500,
        random_seed=123,
        random_walk_probability=0.2,
        stagnation_limit=10,
    )

    assert result.success is True
    assert result.final_hard_violations == 0
    val = validate_timetable(result.timetable)
    assert val.is_valid is True


def test_time_limit_reached(small_dataset: TimetableDataset):
    """Verifies that time_limit_seconds=0.0 aborts and returns limit_reached."""
    result = solve_min_conflicts(small_dataset, time_limit_seconds=0.0, random_seed=42)

    assert result.success is False
    assert result.status == "limit_reached"
    assert "time limit exceeded" in result.failure_reason.lower()


def test_reproducibility_with_same_random_seed(small_dataset: TimetableDataset):
    """Verifies that two runs with the same random seed produce identical results."""
    res1 = solve_min_conflicts(small_dataset, random_seed=42)
    res2 = solve_min_conflicts(small_dataset, random_seed=42)

    assert res1.success is True and res2.success is True
    assert res1.iterations == res2.iterations
    assert res1.conflicted_variable_selections == res2.conflicted_variable_selections
    assert res1.candidate_placements_evaluated == res2.candidate_placements_evaluated
    assert res1.initial_hard_conflicts == res2.initial_hard_conflicts
    assert res1.total_cost == res2.total_cost
    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_different_random_seeds_produce_valid_solutions(small_dataset: TimetableDataset):
    """Verifies that Min-Conflicts reliably solves the dataset with different seeds."""
    for seed in [101, 202, 303]:
        res = solve_min_conflicts(small_dataset, max_iterations=5000, random_seed=seed)
        assert res.success is True
        assert res.final_hard_violations == 0
        assert res.timetable is not None
        val = validate_timetable(res.timetable)
        assert val.is_valid is True


def test_deterministic_mode_runs_cleanly(small_dataset: TimetableDataset):
    """Verifies that deterministic mode (deterministic=True) executes and succeeds."""
    res = solve_min_conflicts(small_dataset, deterministic=True, initial_strategy="greedy")
    assert res.success is True
    assert res.final_hard_violations == 0


def test_min_conflicts_result_to_dict(small_dataset: TimetableDataset):
    """Verifies serialization of MinConflictsResult to dictionary."""
    result = solve_min_conflicts(small_dataset, random_seed=42)
    d = result.to_dict()

    expected_keys = {
        "algorithm",
        "success",
        "status",
        "runtime_seconds",
        "iterations",
        "conflicted_variable_selections",
        "candidate_placements_evaluated",
        "initial_hard_conflicts",
        "final_hard_violations",
        "soft_penalties",
        "total_cost",
        "random_seed",
        "failure_reason",
    }
    assert expected_keys.issubset(d.keys())
    assert d["algorithm"] == "min_conflicts"
    assert d["success"] is True
    assert d["final_hard_violations"] == 0
