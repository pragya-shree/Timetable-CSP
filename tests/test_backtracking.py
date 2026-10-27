"""Unit tests for the plain Backtracking CSP baseline solver (core/backtracking.py)."""

from pathlib import Path
import pytest

from core.backtracking import (
    BacktrackingSolver,
    solve_backtracking,
    solve_backtracking_mrv,
    solve_backtracking_mrv_degree,
    solve_backtracking_mrv_degree_fc,
)
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.models import (
    ClassGroup,
    ClassSubjectRequirement,
    Room,
    Subject,
    Teacher,
    TimeStructure,
    TimetableDataset,
    load_dataset_from_file,
)
from core.sessions import generate_session_variables, get_valid_domain_for_session


@pytest.fixture
def small_dataset() -> TimetableDataset:
    return load_dataset_from_file(Path("datasets/small_dataset.json"))


@pytest.fixture
def large_dataset() -> TimetableDataset:
    return load_dataset_from_file(Path("datasets/large_dataset.json"))


def test_solve_small_dataset_success(small_dataset: TimetableDataset):
    """Verifies that plain backtracking successfully solves small_dataset.json."""
    result = solve_backtracking(small_dataset)

    assert result.success is True
    assert result.status == "solved"
    assert result.timetable is not None
    assert result.failure_reason is None
    assert result.runtime_seconds >= 0.0
    assert result.nodes_explored > 0
    assert result.assignments_tried > 0
    assert result.max_search_depth == len(generate_session_variables(small_dataset))


def test_solution_is_complete_and_valid(small_dataset: TimetableDataset):
    """Verifies that the returned timetable satisfies all hard constraints (0 violations)."""
    result = solve_backtracking(small_dataset)
    assert result.timetable is not None

    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0
    assert len(val_res.violations) == 0

    assert result.hard_violations == 0
    cost_info = calculate_cost(result.timetable)
    assert result.total_cost == cost_info.total_cost
    assert result.soft_penalties["s1_gaps"] == cost_info.s1_gaps
    assert result.soft_penalties["s2_repeats"] == cost_info.s2_repeats
    assert result.soft_penalties["s3_non_consecutive"] == cost_info.s3_non_consecutive


def test_solution_uses_legal_unary_domains(small_dataset: TimetableDataset):
    """Verifies that every session is assigned a placement strictly within its unary domain."""
    result = solve_backtracking(small_dataset)
    assert result.timetable is not None

    for var, placement in result.timetable.get_placed_sessions():
        valid_domain = get_valid_domain_for_session(var, small_dataset)
        assert placement in valid_domain, (
            f"Session {var.session_id} assigned illegal placement {placement}"
        )


def test_no_teacher_class_or_room_overlaps_including_labs(small_dataset: TimetableDataset):
    """Independently verifies that no 2 sessions overlap on occupied time-slots."""
    result = solve_backtracking(small_dataset)
    assert result.timetable is not None

    placed = result.timetable.get_placed_sessions()

    # Verify pairwise conflict freedom
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


def test_deterministic_behavior(small_dataset: TimetableDataset):
    """Ensures repeated runs with plain backtracking produce identical metrics and placements."""
    res1 = solve_backtracking(small_dataset)
    res2 = solve_backtracking(small_dataset)

    assert res1.success is True and res2.success is True
    assert res1.nodes_explored == res2.nodes_explored
    assert res1.backtracks == res2.backtracks
    assert res1.assignments_tried == res2.assignments_tried
    assert res1.total_cost == res2.total_cost

    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_empty_domain_failure():
    """Verifies that an impossible dataset with an empty domain terminates as unsatisfiable."""
    # Dataset where class strength exceeds all room capacities
    dataset = TimetableDataset(
        name="impossible_capacity",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=4),
        subjects={"SUB1": Subject(id="SUB1", name="Sub 1", weekly_periods=2, type="theory")},
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["SUB1"], max_weekly_load=10)},
        classes={
            "C1": ClassGroup(
                id="C1",
                name="Class 1",
                strength=100,  # strength 100
                lunch_slot=2,
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            )
        },
        rooms={"R1": Room(id="R1", name="Room 1", capacity=30, type="classroom")},  # capacity only 30
    )

    result = solve_backtracking(dataset)
    assert result.success is False
    assert result.status == "unsatisfiable"
    assert result.timetable is None
    assert result.nodes_explored == 0
    assert "empty domain" in result.failure_reason.lower()
    assert result.hard_violations > 0  # H7 deficit for missing periods


def test_impossible_schedule_search_exhaustion():
    """Verifies that an instance with valid unary domains but mathematically conflicting constraints
    exhausts the search tree and reports 'unsatisfiable' with backtracks > 0."""
    # 2 classes require 2 periods each of the same teacher, but the entire week only has 3 slots
    # Teacher needs 4 periods total in a 3-period week -> pigeonhole conflict!
    dataset = TimetableDataset(
        name="impossible_pigeonhole",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=3),
        subjects={"SUB1": Subject(id="SUB1", name="Sub 1", weekly_periods=2, type="theory")},
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["SUB1"], max_weekly_load=10)},
        classes={
            "C1": ClassGroup(
                id="C1",
                name="Class 1",
                strength=30,
                lunch_slot=0,  # lunch at 0 -> leaves periods 1, 2 (2 slots)
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
            "C2": ClassGroup(
                id="C2",
                name="Class 2",
                strength=30,
                lunch_slot=0,  # lunch at 0 -> leaves periods 1, 2 (2 slots)
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
        },
        rooms={
            "R1": Room(id="R1", name="Room 1", capacity=50, type="classroom"),
            "R2": Room(id="R2", name="Room 2", capacity=50, type="classroom"),
        },
    )

    result = solve_backtracking(dataset)
    assert result.success is False
    assert result.status == "unsatisfiable"
    assert result.timetable is None
    assert result.nodes_explored > 0
    assert result.backtracks > 0
    assert "exhausted" in result.failure_reason.lower()


def test_node_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding the configured node limit returns 'limit_reached'."""
    # Set a tiny node budget of 10 nodes (small dataset requires ~40 nodes)
    result = solve_backtracking(small_dataset, max_nodes=10)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert result.nodes_explored >= 10
    assert "node limit reached" in result.failure_reason.lower()


def test_time_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding the configured time limit returns 'limit_reached'."""
    # Set an infinitesimally small time limit (0.0 seconds)
    result = solve_backtracking(small_dataset, time_limit_seconds=0.0)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert "time limit exceeded" in result.failure_reason.lower()


def test_solve_large_dataset_success(large_dataset: TimetableDataset):
    """Verifies that plain backtracking also solves large_dataset.json cleanly."""
    result = solve_backtracking(large_dataset, max_nodes=500_000, time_limit_seconds=30.0)

    assert result.success is True
    assert result.status == "solved"
    assert result.timetable is not None
    assert result.hard_violations == 0

    val_res = validate_timetable(result.timetable)
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0


def test_backtracking_result_to_dict(small_dataset: TimetableDataset):
    """Verifies serialization of BacktrackingResult metrics."""
    result = solve_backtracking(small_dataset)
    data = result.to_dict()

    assert data["success"] is True
    assert data["status"] == "solved"
    assert data["algorithm"] == "plain_backtracking"
    assert "runtime_seconds" in data
    assert "nodes_explored" in data
    assert "backtracks" in data
    assert "assignments_tried" in data
    assert "max_search_depth" in data
    assert data["hard_violations"] == 0
    assert "s1_gaps" in data["soft_penalties"]


# =============================================================================
# MINIMUM REMAINING VALUES (MRV) HEURISTIC TESTS
# =============================================================================


def test_solve_small_dataset_mrv(small_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV successfully solves small_dataset.json."""
    result = solve_backtracking(small_dataset, use_mrv=True)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv"
    assert result.timetable is not None
    assert result.failure_reason is None
    assert result.runtime_seconds >= 0.0
    assert result.nodes_explored > 0
    assert result.assignments_tried > 0
    assert result.max_search_depth == len(generate_session_variables(small_dataset))


def test_mrv_solution_is_complete_and_valid(small_dataset: TimetableDataset):
    """Verifies that the timetable returned by MRV satisfies all hard constraints."""
    result = solve_backtracking(small_dataset, use_mrv=True)
    assert result.timetable is not None

    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0
    assert len(val_res.violations) == 0

    assert result.hard_violations == 0
    cost_info = calculate_cost(result.timetable)
    assert result.total_cost == cost_info.total_cost


def test_solve_large_dataset_mrv(large_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV successfully solves large_dataset.json."""
    result = solve_backtracking(large_dataset, max_nodes=500_000, time_limit_seconds=30.0, use_mrv=True)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv"
    assert result.timetable is not None
    assert result.hard_violations == 0

    val_res = validate_timetable(result.timetable)
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0


def test_mrv_no_teacher_class_or_room_overlaps_including_labs(small_dataset: TimetableDataset):
    """Independently verifies that MRV solution has zero pairwise collisions."""
    result = solve_backtracking(small_dataset, use_mrv=True)
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
                    f"MRV Teacher clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert var_a.class_id != var_b.class_id, (
                    f"MRV Class clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert place_a.room_id != place_b.room_id, (
                    f"MRV Room clash between {var_a.session_id} and {var_b.session_id} in {place_a.room_id} at {overlap}"
                )


def test_mrv_selects_minimum_legal_domain_size(small_dataset: TimetableDataset):
    """Verifies that select_next_variable chooses the variable with the minimum legal domain."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True)

    # Pick a session and place it to reduce other variables' domains
    var0 = solver.variables[0]
    placement0 = solver.domains[var0][0]
    solver._apply_placement(var0, placement0)

    unassigned = [v for v in solver.variables if solver.assignments[v] is None]
    counts = {v: solver.count_legal_placements(v) for v in unassigned}
    min_count = min(counts.values())

    selected = solver.select_next_variable()
    assert selected is not None
    assert counts[selected] == min_count, (
        f"Selected {selected.session_id} with count {counts[selected]}, but min is {min_count}"
    )


def test_mrv_equal_domain_sizes_use_deterministic_tie_breaker(small_dataset: TimetableDataset):
    """Verifies that equal remaining domain counts use the original variable order as tie-breaker."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True)
    # At initial state with no placements, evaluate unassigned variables
    unassigned = [v for v in solver.variables if solver.assignments[v] is None]
    counts = {v: solver.count_legal_placements(v) for v in unassigned}
    min_count = min(counts.values())

    # Filter all variables sharing the exact minimum count
    min_vars = [v for v in unassigned if counts[v] == min_count]
    assert len(min_vars) > 1, "Expected multiple variables to share the initial minimum domain size."

    expected_winner = min_vars[0]  # First in original generation order
    selected = solver.select_next_variable()
    assert selected == expected_winner
    assert solver.var_index[selected] == min(solver.var_index[v] for v in min_vars)


def test_mrv_deterministic_behavior(small_dataset: TimetableDataset):
    """Ensures repeated MRV runs produce identical placements and metrics."""
    res1 = solve_backtracking(small_dataset, use_mrv=True)
    res2 = solve_backtracking(small_dataset, use_mrv=True)

    assert res1.success is True and res2.success is True
    assert res1.nodes_explored == res2.nodes_explored
    assert res1.backtracks == res2.backtracks
    assert res1.assignments_tried == res2.assignments_tried
    assert res1.total_cost == res2.total_cost

    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_mrv_node_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding max_nodes in MRV mode returns limit_reached."""
    result = solve_backtracking(small_dataset, max_nodes=10, use_mrv=True)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert result.nodes_explored >= 10
    assert "node limit reached" in result.failure_reason.lower()


def test_mrv_time_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding time_limit in MRV mode returns limit_reached."""
    result = solve_backtracking(small_dataset, time_limit_seconds=0.0, use_mrv=True)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert "time limit exceeded" in result.failure_reason.lower()


def test_mrv_impossible_instance_exhausts_search():
    """Verifies that an unsatisfiable problem terminates cleanly with MRV."""
    dataset = TimetableDataset(
        name="impossible_pigeonhole",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=3),
        subjects={"SUB1": Subject(id="SUB1", name="Sub 1", weekly_periods=2, type="theory")},
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["SUB1"], max_weekly_load=10)},
        classes={
            "C1": ClassGroup(
                id="C1",
                name="Class 1",
                strength=30,
                lunch_slot=0,
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
            "C2": ClassGroup(
                id="C2",
                name="Class 2",
                strength=30,
                lunch_slot=0,
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
        },
        rooms={
            "R1": Room(id="R1", name="Room 1", capacity=50, type="classroom"),
            "R2": Room(id="R2", name="Room 2", capacity=50, type="classroom"),
        },
    )

    result = solve_backtracking(dataset, use_mrv=True)
    assert result.success is False
    assert result.status == "unsatisfiable"
    assert result.timetable is None
    assert result.nodes_explored > 0
    assert result.backtracks > 0


def test_plain_and_mrv_both_available_with_correct_algorithm_tags(small_dataset: TimetableDataset):
    """Verifies that both solver modes are independently available and identifiable."""
    res_plain = solve_backtracking(small_dataset, use_mrv=False)
    res_mrv = solve_backtracking(small_dataset, use_mrv=True)

    assert res_plain.algorithm == "plain_backtracking"
    assert res_mrv.algorithm == "backtracking_mrv"
    assert res_plain.success is True
    assert res_mrv.success is True
    assert res_plain.hard_violations == 0
    assert res_mrv.hard_violations == 0


# =============================================================================
# DEGREE HEURISTIC (MRV + DEGREE) TESTS
# =============================================================================


def test_solve_small_dataset_mrv_degree(small_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV + Degree solves small_dataset.json."""
    result = solve_backtracking_mrv_degree(small_dataset)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv_degree"
    assert result.timetable is not None
    assert result.failure_reason is None
    assert result.runtime_seconds >= 0.0
    assert result.nodes_explored > 0
    assert result.assignments_tried > 0
    assert result.max_search_depth == len(generate_session_variables(small_dataset))


def test_solve_large_dataset_mrv_degree(large_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV + Degree solves large_dataset.json."""
    result = solve_backtracking_mrv_degree(large_dataset, max_nodes=500_000, time_limit_seconds=30.0)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv_degree"
    assert result.timetable is not None
    assert result.hard_violations == 0


def test_mrv_degree_solution_is_complete_and_valid(small_dataset: TimetableDataset):
    """Verifies that the schedule produced by MRV + Degree has 0 hard violations."""
    result = solve_backtracking_mrv_degree(small_dataset)
    assert result.timetable is not None

    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0
    assert len(val_res.violations) == 0

    assert result.hard_violations == 0
    cost_info = calculate_cost(result.timetable)
    assert result.total_cost == cost_info.total_cost


def test_mrv_priority_over_degree_when_domain_sizes_differ(small_dataset: TimetableDataset):
    """Verifies that MRV strictly takes precedence over Degree when legal domain sizes differ."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)

    # Place an initial variable to constrain some variables more than others
    var0 = solver.variables[0]
    p0 = solver.domains[var0][0]
    solver._apply_placement(var0, p0)

    unassigned = [v for v in solver.variables if solver.assignments[v] is None]
    counts = {v: solver.count_legal_placements(v) for v in unassigned}
    min_count = min(counts.values())
    max_count = max(counts.values())
    assert min_count < max_count, "Setup requirement: domain sizes must differ."

    selected = solver.select_next_variable()
    assert selected is not None
    # Must choose a variable with strictly minimum remaining legal values
    assert counts[selected] == min_count, (
        f"MRV priority violated: selected {selected.session_id} with count {counts[selected]}, "
        f"expected min_count {min_count}"
    )


def test_degree_breaks_mrv_tie_correctly(small_dataset: TimetableDataset):
    """Verifies that when multiple variables have identical minimum domain counts,
    the Degree Heuristic strictly breaks the tie in favor of the variable with
    the highest degree among unassigned variables.
    
    Setup:
    Place variable 0 (CSE_A_CS101_s1) at (day=4, start_period=4, room='CR1').
    Under this partial state:
    - Variable CSE_A_CS101_LAB_s1 (index 12) has legal domain size 13 and unassigned degree 13.
    - Variable CSE_B_CS101_LAB_s1 (index 25) has legal domain size 13 and unassigned degree 14.
    - Both tie on minimum legal domain size (13).
    - MRV alone picks index 12 because 12 < 25 (original index tie-breaker).
    - MRV + Degree picks index 25 because degree 14 > 13.
    
    This test would FAIL if the Degree Heuristic were accidentally disabled.
    """
    from core.sessions import Placement

    solver = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)
    p0 = Placement(day=4, start_period=4, room_id="CR1")
    solver._apply_placement(solver.variables[0], p0)

    unassigned = [v for v in solver.variables if solver.assignments[v] is None]
    counts = {v: solver.count_legal_placements(v) for v in unassigned}
    min_count = min(counts.values())

    tied_vars = [v for v in unassigned if counts[v] == min_count]
    assert len(tied_vars) >= 2, "Setup requirement: multiple variables must tie on domain size."

    degrees = {v: solver.count_unassigned_degree(v, unassigned) for v in tied_vars}
    assert len(set(degrees.values())) > 1, f"Setup requirement: tied variables must have different degrees: {degrees}"

    highest_degree_var = max(tied_vars, key=lambda v: degrees[v])
    lowest_index_var = min(tied_vars, key=lambda v: solver.var_index[v])
    assert solver.var_index[highest_degree_var] > solver.var_index[lowest_index_var], (
        "Setup requirement: highest degree variable must have higher index to prove degree overrides index."
    )

    selected = solver.select_next_variable()
    assert selected is not None
    assert selected == highest_degree_var, (
        f"Degree tie-breaking violated: selected {selected.session_id} with degree {degrees[selected]}, "
        f"expected {highest_degree_var.session_id} with max degree {degrees[highest_degree_var]}."
    )

    # Prove that without Degree, MRV alone selects the lower-index variable
    solver_mrv_alone = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=False)
    solver_mrv_alone._apply_placement(solver_mrv_alone.variables[0], p0)
    selected_mrv = solver_mrv_alone.select_next_variable()
    assert selected_mrv == lowest_index_var, "MRV alone must select by lowest generation index."
    assert selected != selected_mrv, "MRV + Degree must demonstrably pick a different variable than MRV alone."


def test_degree_isolated_synthetic_demonstration():
    """Isolated unit test on a minimal synthetic dataset where:
    - Variable A (index 0) has legal domain size 2 and degree 1.
    - Variable B (index 1) has legal domain size 2 and degree 2.
    - Variable C (index 2) has legal domain size 2 and degree 1.
    
    Demonstrates in pure isolation that:
    - MRV alone picks Variable A (index 0).
    - MRV + Degree picks Variable B (index 1, due to degree 2 > 1).
    - If Degree were disabled, Variable A would be selected instead.
    """
    dataset = TimetableDataset(
        name="degree_synthetic_test",
        time_structure=TimeStructure(days=["Mon", "Tue"], periods_per_day=3),
        subjects={
            "SUB_A": Subject(id="SUB_A", name="A", weekly_periods=1, type="theory"),
            "SUB_B": Subject(id="SUB_B", name="B", weekly_periods=1, type="theory"),
            "SUB_C": Subject(id="SUB_C", name="C", weekly_periods=1, type="theory"),
        },
        teachers={
            "T_A": Teacher(id="T_A", name="T_A", subject_ids=["SUB_A"], unavailable_slots=[(1, 0), (1, 1)]),
            "T_B": Teacher(id="T_B", name="T_B", subject_ids=["SUB_B"], unavailable_slots=[(0, 1), (1, 1)]),
            "T_C": Teacher(id="T_C", name="T_C", subject_ids=["SUB_C"], unavailable_slots=[(0, 0), (0, 1)]),
        },
        classes={
            "C_A": ClassGroup(id="C_A", name="CA", strength=20, lunch_slot=2, subjects=[
                ClassSubjectRequirement(subject_id="SUB_A", teacher_id="T_A")
            ]),
            "C_B": ClassGroup(id="C_B", name="CB", strength=20, lunch_slot=2, subjects=[
                ClassSubjectRequirement(subject_id="SUB_B", teacher_id="T_B")
            ]),
            "C_C": ClassGroup(id="C_C", name="CC", strength=20, lunch_slot=2, subjects=[
                ClassSubjectRequirement(subject_id="SUB_C", teacher_id="T_C")
            ]),
        },
        rooms={
            "R1": Room(id="R1", name="R1", capacity=30, type="classroom"),
        },
    )

    solver_mrv = BacktrackingSolver(dataset, use_mrv=True, use_degree=False)
    solver_deg = BacktrackingSolver(dataset, use_mrv=True, use_degree=True)

    v_a, v_b, v_c = solver_deg.variables
    unassigned = solver_deg.variables

    # Verify identical domain sizes
    assert solver_deg.count_legal_placements(v_a) == 2
    assert solver_deg.count_legal_placements(v_b) == 2
    assert solver_deg.count_legal_placements(v_c) == 2

    # Verify differing degrees
    assert solver_deg.count_unassigned_degree(v_a, unassigned) == 1
    assert solver_deg.count_unassigned_degree(v_b, unassigned) == 2
    assert solver_deg.count_unassigned_degree(v_c, unassigned) == 1

    # MRV alone picks v_a (lowest index 0)
    assert solver_mrv.select_next_variable() == v_a

    # MRV + Degree picks v_b (highest degree 2)
    assert solver_deg.select_next_variable() == v_b


def test_configuration_independence_variable_ordering(small_dataset: TimetableDataset):
    """Verifies that on an identical partial state, all three configurations independently
    select according to their distinct mathematical specification:
    - Plain: next unassigned variable in static generation order (index 1).
    - MRV: minimum legal domain size, then lowest generation index (index 12).
    - MRV + Degree: minimum legal domain size, then highest unassigned degree (index 25).
    """
    from core.sessions import Placement

    p0 = Placement(day=4, start_period=4, room_id="CR1")

    # 1. Plain
    s_plain = BacktrackingSolver(small_dataset, use_mrv=False, use_degree=False)
    s_plain._apply_placement(s_plain.variables[0], p0)
    sel_plain = s_plain.select_next_variable()

    # 2. MRV
    s_mrv = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=False)
    s_mrv._apply_placement(s_mrv.variables[0], p0)
    sel_mrv = s_mrv.select_next_variable()

    # 3. MRV + Degree
    s_deg = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)
    s_deg._apply_placement(s_deg.variables[0], p0)
    sel_deg = s_deg.select_next_variable()

    assert sel_plain is not None and sel_mrv is not None and sel_deg is not None
    # All 3 choose distinct variables!
    assert sel_plain.session_id == "CSE_A_CS101_s2"
    assert s_plain.var_index[sel_plain] == 1

    assert sel_mrv.session_id == "CSE_A_CS101_LAB_s1"
    assert s_mrv.var_index[sel_mrv] == 12

    assert sel_deg.session_id == "CSE_B_CS101_LAB_s1"
    assert s_deg.var_index[sel_deg] == 25


def test_original_index_tie_breaking_when_both_mrv_and_degree_tie(small_dataset: TimetableDataset):
    """Verifies that when both MRV domain size and degree tie, the original variable index breaks the tie."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)

    unassigned = solver.variables
    counts = {v: solver.count_legal_placements(v) for v in unassigned}
    min_count = min(counts.values())

    tied_vars = [v for v in unassigned if counts[v] == min_count]
    degrees = {v: solver.count_unassigned_degree(v, unassigned) for v in tied_vars}
    max_deg = max(degrees.values())

    # Find all variables tying on both min domain count and max degree
    top_candidates = [v for v in tied_vars if degrees[v] == max_deg]
    assert len(top_candidates) > 1, "Setup requirement: multiple variables must tie on both count and degree."

    expected_winner = top_candidates[0]  # Lowest var_index in deterministic generation order
    selected = solver.select_next_variable()
    assert selected == expected_winner
    assert solver.var_index[selected] == min(solver.var_index[v] for v in top_candidates)


def test_degree_calculation_shared_teacher_class_room(small_dataset: TimetableDataset):
    """Verifies that constraint graph correctly identifies shared teachers, classes, and rooms."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)

    # 1. Shared teacher
    t1_vars = [v for v in solver.variables if v.teacher_id == "T1"]
    assert len(t1_vars) >= 2
    assert t1_vars[1] in solver.constraint_neighbors[t1_vars[0]], "Variables sharing teacher T1 must be neighbors."

    # 2. Shared class
    ca_vars = [v for v in solver.variables if v.class_id == "CSE_A"]
    assert len(ca_vars) >= 2
    assert ca_vars[1] in solver.constraint_neighbors[ca_vars[0]], "Variables sharing class CSE_A must be neighbors."

    # 3. Disjoint lab vs theory with different teacher and class
    lab_var = [v for v in solver.variables if v.session_type == "lab" and v.class_id == "CSE_A"][0]
    diff_theory = [
        v for v in solver.variables
        if v.session_type == "theory" and v.class_id == "CSE_B" and v.teacher_id != lab_var.teacher_id
    ][0]
    assert diff_theory not in solver.constraint_neighbors[lab_var], (
        "Variables with disjoint room types, different teachers, and different classes must NOT be neighbors."
    )


def test_mrv_degree_multi_period_lab_interactions(small_dataset: TimetableDataset):
    """Verifies that multi-period lab sessions account for all occupied periods in collision detection."""
    solver = BacktrackingSolver(small_dataset, use_mrv=True, use_degree=True)

    lab_vars = [v for v in solver.variables if v.session_type == "lab"]
    for lv in lab_vars:
        assert lv.duration == 2
        # Lab must have constraint edges with sessions of its own class
        same_class_theory = [v for v in solver.variables if v.class_id == lv.class_id and v != lv]
        for sct in same_class_theory:
            assert sct in solver.constraint_neighbors[lv], (
                f"Lab {lv.session_id} must have edge to same-class session {sct.session_id}."
            )


def test_mrv_degree_deterministic_behavior(small_dataset: TimetableDataset):
    """Ensures repeated MRV + Degree runs produce identical placements and metrics."""
    res1 = solve_backtracking_mrv_degree(small_dataset)
    res2 = solve_backtracking_mrv_degree(small_dataset)

    assert res1.success is True and res2.success is True
    assert res1.nodes_explored == res2.nodes_explored
    assert res1.backtracks == res2.backtracks
    assert res1.assignments_tried == res2.assignments_tried
    assert res1.total_cost == res2.total_cost

    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_mrv_degree_node_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding max_nodes in MRV + Degree mode returns limit_reached."""
    result = solve_backtracking_mrv_degree(small_dataset, max_nodes=10)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert result.nodes_explored >= 10
    assert "node limit reached" in result.failure_reason.lower()


def test_mrv_degree_time_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding time_limit in MRV + Degree mode returns limit_reached."""
    result = solve_backtracking_mrv_degree(small_dataset, time_limit_seconds=0.0)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert "time limit exceeded" in result.failure_reason.lower()


def test_all_three_configurations_available_independently(small_dataset: TimetableDataset):
    """Verifies that Plain, MRV, and MRV + Degree are all independently available."""
    res_plain = solve_backtracking(small_dataset, use_mrv=False, use_degree=False)
    res_mrv = solve_backtracking_mrv(small_dataset)
    res_degree = solve_backtracking_mrv_degree(small_dataset)

    assert res_plain.algorithm == "plain_backtracking"
    assert res_mrv.algorithm == "backtracking_mrv"
    assert res_degree.algorithm == "backtracking_mrv_degree"

    assert res_plain.success is True
    assert res_mrv.success is True
    assert res_degree.success is True
    assert res_plain.hard_violations == 0
    assert res_mrv.hard_violations == 0
    assert res_degree.hard_violations == 0


# =============================================================================
# FORWARD CHECKING (MRV + DEGREE + FC) TESTS
# =============================================================================


def test_solve_small_dataset_mrv_degree_fc(small_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV + Degree + FC solves small_dataset.json."""
    result = solve_backtracking_mrv_degree_fc(small_dataset)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv_degree_fc"
    assert result.timetable is not None
    assert result.failure_reason is None
    assert result.runtime_seconds >= 0.0
    assert result.nodes_explored > 0
    assert result.assignments_tried > 0
    assert result.backtracks == 0
    assert result.max_search_depth == len(generate_session_variables(small_dataset))


def test_solve_large_dataset_mrv_degree_fc(large_dataset: TimetableDataset):
    """Verifies that Backtracking + MRV + Degree + FC solves large_dataset.json."""
    result = solve_backtracking_mrv_degree_fc(large_dataset, max_nodes=500_000, time_limit_seconds=30.0)

    assert result.success is True
    assert result.status == "solved"
    assert result.algorithm == "backtracking_mrv_degree_fc"
    assert result.timetable is not None
    assert result.hard_violations == 0

    val_res = validate_timetable(result.timetable)
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0


def test_mrv_degree_fc_solution_is_complete_and_valid(small_dataset: TimetableDataset):
    """Verifies that the schedule produced by MRV + Degree + FC has 0 hard violations."""
    result = solve_backtracking_mrv_degree_fc(small_dataset)
    assert result.timetable is not None

    val_res = validate_timetable(result.timetable)
    assert val_res.is_complete is True
    assert val_res.is_valid is True
    assert val_res.hard_violation_count == 0
    assert len(val_res.violations) == 0

    assert result.hard_violations == 0
    cost_info = calculate_cost(result.timetable)
    assert result.total_cost == cost_info.total_cost


def test_mrv_degree_fc_no_teacher_class_or_room_overlaps(small_dataset: TimetableDataset):
    """Independently verifies that MRV + Degree + FC produces zero pairwise collisions."""
    result = solve_backtracking_mrv_degree_fc(small_dataset)
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
                    f"FC Teacher clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert var_a.class_id != var_b.class_id, (
                    f"FC Class clash between {var_a.session_id} and {var_b.session_id} at {overlap}"
                )
                assert place_a.room_id != place_b.room_id, (
                    f"FC Room clash between {var_a.session_id} and {var_b.session_id} in {place_a.room_id} at {overlap}"
                )


def test_fc_incompatible_placements_removed_after_assignment(small_dataset: TimetableDataset):
    """Verifies that assigning a variable removes all conflicting placements from unassigned neighbors."""
    solver = BacktrackingSolver(small_dataset, use_fc=True)
    var = solver.variables[0]  # CSE_A, T1, CR1/CR2/CR3
    placement = solver.current_domains[var][0]

    # Find unassigned constraint neighbors
    neighbors = [u for u in solver.variables if u in solver.constraint_neighbors[var] and u != var]
    assert len(neighbors) > 0

    # Record domains before FC
    pre_fc_counts = {u: len(solver.current_domains[u]) for u in neighbors}

    # Apply placement and forward check
    solver._apply_placement(var, placement)
    success, saved = solver.forward_check(var, placement)

    assert success is True
    # At least some neighbors must have had incompatible placements pruned
    pruned_neighbors = [u for u in neighbors if len(solver.current_domains[u]) < pre_fc_counts[u]]
    assert len(pruned_neighbors) > 0, "Forward checking must prune at least one conflicting placement."

    # Verify that every pruned placement actually conflicted under H1, H2, or H3
    for u in pruned_neighbors:
        pruned_placements = [p for p in saved[u] if p not in solver.current_domains[u]]
        for p in pruned_placements:
            assert solver._placements_conflict(var, placement, u, p) is True, (
                f"Pruned placement {p} for {u.session_id} must conflict with {placement}"
            )


def test_fc_multi_period_lab_conflicts_pruned_across_all_periods(small_dataset: TimetableDataset):
    """Verifies that multi-period lab assignments prune placements across all occupied periods."""
    solver = BacktrackingSolver(small_dataset, use_fc=True)

    # Locate a 2-period lab variable
    lab_vars = [v for v in solver.variables if v.session_type == "lab"]
    assert len(lab_vars) > 0
    lab_var = lab_vars[0]
    assert lab_var.duration == 2

    # Locate a theory variable sharing the same class group
    same_class_theory = [
        v for v in solver.variables
        if v.class_id == lab_var.class_id and v.session_type == "theory"
    ][0]

    # Assign lab to (day=1, start_period=1) -> occupies period 1 AND period 2
    lab_placement = [p for p in solver.current_domains[lab_var] if p.day == 1 and p.start_period == 1][0]

    solver._apply_placement(lab_var, lab_placement)
    success, saved = solver.forward_check(lab_var, lab_placement)

    assert success is True
    assert same_class_theory in saved

    theory_remaining = solver.current_domains[same_class_theory]
    # Verify that NO placement for same_class_theory exists on day 1 at period 1 OR period 2
    for p in theory_remaining:
        if p.day == 1:
            assert p.start_period not in (1, 2), (
                f"Theory placement at day 1 period {p.start_period} should have been pruned by 2-period lab at periods 1,2!"
            )


def test_fc_compatible_placements_not_removed(small_dataset: TimetableDataset):
    """Verifies that Forward Checking NEVER removes placements that are compatible with the assignment."""
    solver = BacktrackingSolver(small_dataset, use_fc=True)
    var = solver.variables[0]
    placement = solver.current_domains[var][0]

    solver._apply_placement(var, placement)
    success, saved = solver.forward_check(var, placement)
    assert success is True

    # For every unassigned variable, every remaining placement MUST be compatible
    for u in solver.variables:
        if solver.assignments[u] is None:
            for p in solver.current_domains[u]:
                assert not solver._placements_conflict(var, placement, u, p), (
                    f"Placement {p} for {u.session_id} remains in domain but conflicts with {placement}"
                )


def test_fc_domain_wipeout_triggers_immediate_backtrack():
    """Verifies that domain wipeout (an unassigned variable's domain becoming empty)
    is detected immediately by forward checking and triggers rejection/backtracking."""
    # Synthetic problem: 2 classes, 1 single slot available in the whole timetable, 1 room, 1 teacher
    # Variable 1 and Variable 2 both must be scheduled, but only 1 can fit.
    dataset = TimetableDataset(
        name="fc_wipeout_demo",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=1),
        subjects={"SUB1": Subject(id="SUB1", name="Sub 1", weekly_periods=1, type="theory")},
        teachers={"T1": Teacher(id="T1", name="Teacher 1", subject_ids=["SUB1"], max_weekly_load=5)},
        classes={
            "C1": ClassGroup(
                id="C1", name="C1", strength=20, lunch_slot=99,
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
            "C2": ClassGroup(
                id="C2", name="C2", strength=20, lunch_slot=99,
                subjects=[ClassSubjectRequirement(subject_id="SUB1", teacher_id="T1")],
            ),
        },
        rooms={"R1": Room(id="R1", name="R1", capacity=30, type="classroom")},
    )

    solver = BacktrackingSolver(dataset, use_fc=True)
    v1, v2 = solver.variables

    # Both variables start with domain of size 1 (day 0, period 0, room R1)
    assert len(solver.current_domains[v1]) == 1
    assert len(solver.current_domains[v2]) == 1

    p = solver.current_domains[v1][0]
    solver._apply_placement(v1, p)

    # Forward checking v1 = p should wipe out v2's domain because both share Teacher T1 and Room R1 at period 0
    success, saved = solver.forward_check(v1, p)

    assert success is False, "Forward checking must report False upon domain wipeout."
    assert len(saved) == 0, "On wipeout failure, forward_check must return empty saved map."
    # Crucially, v2's domain must have been restored immediately inside forward_check
    assert len(solver.current_domains[v2]) == 1


def test_fc_domains_restored_on_backtrack(small_dataset: TimetableDataset):
    """Verifies that domain changes made by forward checking are completely restored on backtrack."""
    solver = BacktrackingSolver(small_dataset, use_fc=True)
    var = solver.variables[0]
    placement = solver.current_domains[var][0]

    # Take snapshot of initial domains
    snapshot = {v: list(solver.current_domains[v]) for v in solver.variables}

    # Apply placement and forward check
    solver._apply_placement(var, placement)
    success, saved = solver.forward_check(var, placement)
    assert success is True
    assert len(saved) > 0, "Setup requirement: at least one variable domain must be modified."

    # Verify domains have changed
    assert any(solver.current_domains[v] != snapshot[v] for v in solver.variables)

    # Undo: restore domains and unapply placement
    solver.restore_domains(saved)
    solver._unapply_placement(var, placement)

    # Verify all domains are identical to the pre-assignment snapshot
    for v in solver.variables:
        assert solver.current_domains[v] == snapshot[v], (
            f"Domain for {v.session_id} was not restored cleanly after backtrack."
        )


def test_fc_synthetic_demonstrating_search_pruning():
    """Synthetic benchmark demonstrating that Forward Checking prunes search space
    and avoids exploring dead-end branches that Plain Backtracking explores.
    
    Setup:
    - 3 variables (V1, V2, V3) competing for 2 room slots.
    - V1 has 2 valid slots: (0, 0) and (0, 1).
    - V2 and V3 only have 1 valid slot: (0, 0) (e.g. blocked at slot 1).
    - If V1 is placed at (0, 0), V2 and V3 are immediately wiped out.
    - Plain Backtracking places V1 at (0, 0), then descends to V2 and explores before failing.
    - Forward Checking detects the wipeout on V2/V3 immediately at V1, rejecting (0, 0) in 1 step!
    """
    dataset = TimetableDataset(
        name="fc_pruning_synthetic",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=2),
        subjects={
            "S1": Subject(id="S1", name="S1", weekly_periods=1, type="theory"),
            "S2": Subject(id="S2", name="S2", weekly_periods=1, type="theory"),
            "S3": Subject(id="S3", name="S3", weekly_periods=1, type="theory"),
        },
        teachers={
            "T1": Teacher(id="T1", name="T1", subject_ids=["S1"]),
            # T2 and T3 unavailable at slot (0, 1) -> can only be at (0, 0)
            "T2": Teacher(id="T2", name="T2", subject_ids=["S2"], unavailable_slots=[(0, 1)]),
            "T3": Teacher(id="T3", name="T3", subject_ids=["S3"], unavailable_slots=[(0, 1)]),
        },
        classes={
            "C1": ClassGroup(id="C1", name="C1", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S1", teacher_id="T1")]),
            "C2": ClassGroup(id="C2", name="C2", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S2", teacher_id="T2")]),
            "C3": ClassGroup(id="C3", name="C3", strength=20, lunch_slot=99,
                             subjects=[ClassSubjectRequirement(subject_id="S3", teacher_id="T3")]),
        },
        rooms={"R1": Room(id="R1", name="R1", capacity=30, type="classroom")},
    )

    # 3 sessions need to be placed in 2 slots -> impossible pigeonhole
    res_plain = solve_backtracking(dataset, use_mrv=False, use_degree=False, use_fc=False)
    res_fc = solve_backtracking(dataset, use_mrv=True, use_degree=True, use_fc=True)

    assert res_plain.success is False
    assert res_fc.success is False

    # Forward Checking must explore fewer nodes than Plain Backtracking because
    # it detects domain wipeouts before descending into child subtrees!
    assert res_fc.nodes_explored < res_plain.nodes_explored, (
        f"FC nodes ({res_fc.nodes_explored}) must be strictly less than Plain ({res_plain.nodes_explored})"
    )


def test_mrv_degree_fc_deterministic_behavior(small_dataset: TimetableDataset):
    """Ensures repeated MRV + Degree + FC runs produce identical placements and metrics."""
    res1 = solve_backtracking_mrv_degree_fc(small_dataset)
    res2 = solve_backtracking_mrv_degree_fc(small_dataset)

    assert res1.success is True and res2.success is True
    assert res1.nodes_explored == res2.nodes_explored
    assert res1.backtracks == res2.backtracks
    assert res1.assignments_tried == res2.assignments_tried
    assert res1.total_cost == res2.total_cost

    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_mrv_degree_fc_node_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding max_nodes in MRV + Degree + FC mode returns limit_reached."""
    result = solve_backtracking_mrv_degree_fc(small_dataset, max_nodes=10)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert result.nodes_explored >= 10
    assert "node limit reached" in result.failure_reason.lower()


def test_mrv_degree_fc_time_limit_reached(small_dataset: TimetableDataset):
    """Verifies that exceeding time_limit in MRV + Degree + FC mode returns limit_reached."""
    result = solve_backtracking_mrv_degree_fc(small_dataset, time_limit_seconds=0.0)

    assert result.success is False
    assert result.status == "limit_reached"
    assert result.timetable is None
    assert "time limit exceeded" in result.failure_reason.lower()


def test_all_four_configurations_available_independently(small_dataset: TimetableDataset):
    """Verifies that Plain, MRV, MRV + Degree, and MRV + Degree + FC are all independently available."""
    res_plain = solve_backtracking(small_dataset, use_mrv=False, use_degree=False, use_fc=False)
    res_mrv = solve_backtracking_mrv(small_dataset)
    res_degree = solve_backtracking_mrv_degree(small_dataset)
    res_fc = solve_backtracking_mrv_degree_fc(small_dataset)

    assert res_plain.algorithm == "plain_backtracking"
    assert res_mrv.algorithm == "backtracking_mrv"
    assert res_degree.algorithm == "backtracking_mrv_degree"
    assert res_fc.algorithm == "backtracking_mrv_degree_fc"

    assert res_plain.success is True
    assert res_mrv.success is True
    assert res_degree.success is True
    assert res_fc.success is True

    assert res_plain.hard_violations == 0
    assert res_mrv.hard_violations == 0
    assert res_degree.hard_violations == 0
    assert res_fc.hard_violations == 0


def test_variable_ordering_alias_mrv_degree_fc(small_dataset: TimetableDataset):
    """Verifies that variable_ordering='mrv_degree_fc' correctly configures the solver."""
    result = solve_backtracking(small_dataset, variable_ordering="mrv_degree_fc")
    assert result.success is True
    assert result.algorithm == "backtracking_mrv_degree_fc"
    assert result.hard_violations == 0


def test_fc_multi_period_lab_wipeout_triggers_immediate_backtrack():
    """Synthetic test demonstrating that a 2-period lab occupying periods (0, 1) wipes out
    a variable whose only available slot is period 1, triggering immediate backtrack."""
    dataset = TimetableDataset(
        name="fc_lab_wipeout_demo",
        time_structure=TimeStructure(days=["Mon"], periods_per_day=2),
        subjects={
            "LAB1": Subject(id="LAB1", name="Lab 1", weekly_periods=2, type="lab", lab_duration=2),
            "THEO1": Subject(id="THEO1", name="Theo 1", weekly_periods=1, type="theory"),
        },
        teachers={
            # T1 teaches both LAB1 and THEO1
            # T1 is unavailable at slot (0, 0) for theory? No, let T1 be available at both slots,
            # but class C2 has blocked/unavailable slot (0, 0)
            "T1": Teacher(id="T1", name="T1", subject_ids=["LAB1", "THEO1"]),
        },
        classes={
            "C1": ClassGroup(
                id="C1", name="C1", strength=20, lunch_slot=99,
                subjects=[ClassSubjectRequirement(subject_id="LAB1", teacher_id="T1")],
            ),
            "C2": ClassGroup(
                id="C2", name="C2", strength=20, lunch_slot=0,  # lunch at period 0 -> only period 1 valid for C2
                subjects=[ClassSubjectRequirement(subject_id="THEO1", teacher_id="T1")],
            ),
        },
        rooms={
            "R_LAB": Room(id="R_LAB", name="Lab Room", capacity=30, type="lab"),
            "R_CLASS": Room(id="R_CLASS", name="Classroom", capacity=30, type="classroom"),
        },
    )

    solver = BacktrackingSolver(dataset, use_fc=True)
    lab_var = [v for v in solver.variables if v.session_type == "lab"][0]
    theo_var = [v for v in solver.variables if v.session_type == "theory"][0]

    # Lab var has 1 valid placement: (day 0, start_period 0, R_LAB)
    assert len(solver.current_domains[lab_var]) == 1
    # Theory var has 1 valid placement: (day 0, start_period 1, R_CLASS) because period 0 is lunch for C2
    assert len(solver.current_domains[theo_var]) == 1

    lab_placement = solver.current_domains[lab_var][0]
    solver._apply_placement(lab_var, lab_placement)

    # Forward checking lab_var: lab occupies period 0 AND period 1 for Teacher T1.
    # Therefore, theo_var at period 1 conflicts on Teacher T1 (H1)!
    success, saved = solver.forward_check(lab_var, lab_placement)

    assert success is False, "FC must report False because lab occupies period 1 and wipes out theo_var."
    assert len(saved) == 0
    assert len(solver.current_domains[theo_var]) == 1


def test_fc_preserves_soft_penalties_and_cost(small_dataset: TimetableDataset):
    """Verifies that MRV + Degree + FC correctly preserves and computes soft penalties and cost."""
    result = solve_backtracking_mrv_degree_fc(small_dataset)

    assert result.success is True
    assert result.hard_violations == 0
    assert "s1_gaps" in result.soft_penalties
    assert "s2_repeats" in result.soft_penalties
    assert "s3_non_consecutive" in result.soft_penalties
    assert "s4_teacher_preferences" in result.soft_penalties
    assert "s5_teacher_consecutive" in result.soft_penalties

    expected_cost = sum(result.soft_penalties.values())
    assert result.total_cost == expected_cost

    d = result.to_dict()
    assert d["algorithm"] == "backtracking_mrv_degree_fc"
    assert d["hard_violations"] == 0
    assert d["total_cost"] == expected_cost


def test_fc_solver_reentrancy(small_dataset: TimetableDataset):
    """Verifies that calling solve() multiple times on the same BacktrackingSolver instance works identically."""
    solver = BacktrackingSolver(small_dataset, use_fc=True)
    res1 = solver.solve()
    res2 = solver.solve()

    assert res1.success is True and res2.success is True
    assert res1.nodes_explored == res2.nodes_explored
    assert res1.assignments_tried == res2.assignments_tried
    assert res1.backtracks == res2.backtracks
    assert res1.total_cost == res2.total_cost
    assert res1.timetable is not None and res2.timetable is not None
    assert res1.timetable.assignments == res2.timetable.assignments


def test_backtracking_optimize_quality_flag(small_dataset: TimetableDataset):
    """Verifies that solve_backtracking with optimize_quality=True returns a valid timetable."""
    result = solve_backtracking(
        small_dataset,
        variable_ordering="mrv_degree_fc",
        optimize_quality=True,
    )
    assert result.success is True
    assert result.hard_violations == 0
    val = validate_timetable(result.timetable)
    assert val.is_valid is True
    assert val.hard_violation_count == 0

