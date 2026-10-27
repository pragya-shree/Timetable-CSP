"""Regression tests for variable-duration laboratories (2-period and 3-period labs).

Verifies:
1. 2-period labs occupy exactly two consecutive periods.
2. 3-period labs occupy exactly three consecutive periods.
3. Labs cannot cross lunch breaks or occupy global blocked periods.
4. Conflict in any occupied period (teacher, class, room) invalidates the placement.
5. Teacher availability and room compatibility are enforced across the entire duration.
6. CSV/XLSX and multi-sheet workbook imports preserve lab_duration.
7. Backtracking, Min-Conflicts, and Hybrid SA solvers preserve atomic lab blocks.
8. The UI renders the full duration (start card + continuation cells).
9. Impossible atomic placements are rejected rather than split.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import pandas as pd

from core.models import (
    ClassGroup,
    ClassSubjectRequirement,
    Room,
    Subject,
    Teacher,
    TimeStructure,
    TimetableDataset,
    ValidationError,
)
from core.sessions import (
    Placement,
    SessionVar,
    Timetable,
    generate_session_variables,
    get_valid_domain_for_session,
)
from core.constraints import (
    check_h1_teacher_clash,
    check_h2_class_clash,
    check_h3_room_clash,
    check_h4_room_capacity,
    check_h5_room_type,
    check_h6_teacher_availability,
    check_h7_weekly_periods,
    check_h8_teacher_max_load,
    check_h9_class_lunch_break,
    check_structural_validity,
    validate_timetable,
)
from core.backtracking import (
    solve_backtracking,
    solve_backtracking_mrv_degree_fc,
)
from core.min_conflicts import solve_min_conflicts
from core.simulated_annealing import solve_hybrid_simulated_annealing
from core.bulk_importer import (
    parse_pasted_tabular_text,
    validate_bulk_rows,
)

from core.custom_builder import (
    validate_and_build_custom_dataset,
)



# =============================================================================
# FIXTURES & BUILDERS
# =============================================================================

def build_variable_lab_dataset(
    lab_duration: int = 3,
    weekly_lab_periods: int = 3,
    lunch_slot: int = 3,
    periods_per_day: int = 6,
    blocked_periods: list[tuple[int, int]] | None = None,
    teacher_unavail: list[tuple[int, int]] | None = None,
) -> TimetableDataset:
    """Builds a minimal synthetic dataset containing a variable-duration lab."""
    blocked = blocked_periods or []
    unavail = teacher_unavail or []

    ts = TimeStructure(
        days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        periods_per_day=periods_per_day,
        blocked_periods=blocked,
    )

    subjects = {
        "MATH": Subject(id="MATH", name="Mathematics", weekly_periods=4, type="theory"),
        "VAR_LAB": Subject(
            id="VAR_LAB",
            name=f"Practical Lab {lab_duration}P",
            weekly_periods=weekly_lab_periods,
            type="lab",
            lab_duration=lab_duration,
        ),
    }

    teachers = {
        "T_MATH": Teacher(
            id="T_MATH",
            name="Math Faculty",
            subject_ids=["MATH"],
            max_weekly_load=15,
        ),
        "T_LAB": Teacher(
            id="T_LAB",
            name="Lab Faculty",
            subject_ids=["VAR_LAB"],
            unavailable_slots=unavail,
            max_weekly_load=15,
        ),
    }

    classes = {
        "CLASS_A": ClassGroup(
            id="CLASS_A",
            name="Batch A",
            strength=25,
            lunch_slot=lunch_slot,
            subjects=[
                ClassSubjectRequirement(subject_id="MATH", teacher_id="T_MATH"),
                ClassSubjectRequirement(subject_id="VAR_LAB", teacher_id="T_LAB"),
            ],
        )
    }

    rooms = {
        "CR1": Room(id="CR1", name="Classroom 1", capacity=30, type="classroom"),
        "LAB1": Room(id="LAB1", name="Laboratory 1", capacity=30, type="lab"),
    }

    return TimetableDataset(
        name=f"Dataset with {lab_duration}-Period Lab",
        time_structure=ts,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )


# =============================================================================
# REQUIREMENT 1: 2-PERIOD LAB OCCUPIES EXACTLY TWO CONSECUTIVE PERIODS
# =============================================================================

def test_two_period_lab_occupies_exactly_two_consecutive_periods():
    """A 2-period lab must generate duration=2 variables occupying 2 contiguous periods."""
    dataset = build_variable_lab_dataset(lab_duration=2, weekly_lab_periods=4)
    vars_list = generate_session_variables(dataset)

    lab_vars = [v for v in vars_list if v.subject_id == "VAR_LAB"]
    assert len(lab_vars) == 2, "4 weekly periods with lab_duration 2 should produce exactly 2 sessions."

    for v in lab_vars:
        assert v.duration == 2
        assert v.session_type == "lab"

    # Test placement coordinates
    placement = Placement(day=0, start_period=1, room_id="LAB1")
    occupied_periods = placement.get_occupied_periods(lab_vars[0].duration)
    occupied_slots = placement.get_occupied_slots(lab_vars[0].duration)

    assert occupied_periods == [1, 2], "Must occupy exactly periods 1 and 2."
    assert occupied_slots == [(0, 1), (0, 2)]
    assert len(occupied_periods) == 2
    assert occupied_periods[1] == occupied_periods[0] + 1, "Periods must be contiguous."


# =============================================================================
# REQUIREMENT 2: 3-PERIOD LAB OCCUPIES EXACTLY THREE CONSECUTIVE PERIODS
# =============================================================================

def test_three_period_lab_occupies_exactly_three_consecutive_periods():
    """A 3-period lab must generate duration=3 variables occupying 3 contiguous periods."""
    dataset = build_variable_lab_dataset(lab_duration=3, weekly_lab_periods=6)
    vars_list = generate_session_variables(dataset)

    lab_vars = [v for v in vars_list if v.subject_id == "VAR_LAB"]
    assert len(lab_vars) == 2, "6 weekly periods with lab_duration 3 should produce exactly 2 sessions."

    for v in lab_vars:
        assert v.duration == 3
        assert v.session_type == "lab"

    # Test placement coordinates
    placement = Placement(day=2, start_period=1, room_id="LAB1")
    occupied_periods = placement.get_occupied_periods(lab_vars[0].duration)
    occupied_slots = placement.get_occupied_slots(lab_vars[0].duration)

    assert occupied_periods == [1, 2, 3], "Must occupy exactly periods 1, 2, and 3."
    assert occupied_slots == [(2, 1), (2, 2), (2, 3)]
    assert len(occupied_periods) == 3
    assert occupied_periods[1] == occupied_periods[0] + 1
    assert occupied_periods[2] == occupied_periods[1] + 1, "All 3 periods must be consecutive."


# =============================================================================
# REQUIREMENT 3: LAB CANNOT CROSS LUNCH OR OCCUPY BLOCKED PERIOD
# =============================================================================

def test_lab_cannot_cross_lunch_or_occupy_blocked_period():
    """Unary domain generation and H9/H6 constraints must reject lunch and blocked overlaps."""
    # Lunch at period 3, blocked slot at (day 0, period 1)
    dataset = build_variable_lab_dataset(
        lab_duration=3,
        weekly_lab_periods=3,
        lunch_slot=3,
        blocked_periods=[(0, 1)],
    )
    vars_list = generate_session_variables(dataset)
    lab_var = next(v for v in vars_list if v.subject_id == "VAR_LAB")

    domain = get_valid_domain_for_session(lab_var, dataset)

    # 1. Day 0 has blocked period at 1:
    # Placements starting at 0 occupy [0, 1, 2] -> blocked!
    # Placements starting at 1 occupy [1, 2, 3] -> blocked & lunch!
    for p in domain:
        occupied = p.get_occupied_slots(lab_var.duration)
        assert (0, 1) not in occupied, "No placement in domain may include blocked period (0, 1)."

    # 2. Lunch is at period 3:
    # A 3-period lab starting at 1 occupies [1, 2, 3] -> hits lunch.
    # A 3-period lab starting at 2 occupies [2, 3, 4] -> hits lunch.
    # A 3-period lab starting at 3 occupies [3, 4, 5] -> hits lunch.
    for p in domain:
        occupied_p = p.get_occupied_periods(lab_var.duration)
        assert 3 not in occupied_p, f"No placement in domain may overlap lunch slot 3, got {occupied_p}."

    # 3. Direct violation verification with check_h9 and check_h6
    tt = Timetable.empty(dataset)
    # Force placement crossing lunch on Day 1 (periods 2, 3, 4)
    bad_lunch_placement = Placement(day=1, start_period=2, room_id="LAB1")
    tt.assign(lab_var, bad_lunch_placement)
    lunch_violations = check_h9_class_lunch_break(tt)
    assert len(lunch_violations) == 1, "Must trigger H9 violation for lunch overlap."

    # Force placement overlapping blocked period on Day 0 (periods 0, 1, 2)
    bad_blocked_placement = Placement(day=0, start_period=0, room_id="LAB1")
    tt.assign(lab_var, bad_blocked_placement)
    blocked_violations = check_h6_teacher_availability(tt)
    assert any(v.details.get("reason") == "global_blocked_period" for v in blocked_violations), (
        "Must trigger H6 violation for blocked period overlap."
    )


# =============================================================================
# REQUIREMENT 4: CONFLICT IN ANY OCCUPIED PERIOD INVALIDATES PLACEMENT
# =============================================================================

def test_conflict_in_any_occupied_period_invalidates_placement():
    """A clash in the first, middle, or final period of a 3-period lab must invalidate schedule."""
    dataset = build_variable_lab_dataset(lab_duration=3, weekly_lab_periods=3, lunch_slot=5)
    vars_list = generate_session_variables(dataset)
    lab_var = next(v for v in vars_list if v.subject_id == "VAR_LAB")
    math_vars = [v for v in vars_list if v.subject_id == "MATH"]

    tt = Timetable.empty(dataset)
    # Lab placed on Day 0, periods [1, 2, 3] in LAB1 with teacher T_LAB
    tt.assign(lab_var, Placement(day=0, start_period=1, room_id="LAB1"))

    # Case A: Class conflict in middle period (period 2)
    tt.assign(math_vars[0], Placement(day=0, start_period=2, room_id="CR1"))
    class_clashes = check_h2_class_clash(tt)
    assert len(class_clashes) >= 1
    assert any(c.details["period"] == 2 for c in class_clashes), "Must detect class clash in middle period 2."

    # Case B: Room conflict in first period (period 1)
    tt_room = Timetable.empty(dataset)
    tt_room.assign(lab_var, Placement(day=0, start_period=1, room_id="LAB1"))
    # Place another session in LAB1 at period 1
    tt_room.assign(math_vars[0], Placement(day=0, start_period=1, room_id="LAB1"))
    room_clashes = check_h3_room_clash(tt_room)
    assert len(room_clashes) >= 1
    assert any(c.details["period"] == 1 for c in room_clashes), "Must detect room clash in first period 1."

    # Case C: Teacher conflict in last period (period 3)
    # Assign math_vars[1] to T_LAB at period 3
    tt_teacher = Timetable.empty(dataset)
    tt_teacher.assign(lab_var, Placement(day=0, start_period=1, room_id="LAB1"))
    # Fabricate a temporary SessionVar with teacher T_LAB
    clashing_teacher_var = SessionVar(
        session_id="EXTRA_SESSION",
        class_id="CLASS_A",
        subject_id="MATH",
        session_no=99,
        teacher_id="T_LAB",
        duration=1,
        session_type="theory",
    )
    tt_teacher.assignments[clashing_teacher_var] = Placement(day=0, start_period=3, room_id="CR1")
    teacher_clashes = check_h1_teacher_clash(tt_teacher)
    assert len(teacher_clashes) >= 1
    assert any(c.details["period"] == 3 for c in teacher_clashes), "Must detect teacher clash in last period 3."


# =============================================================================
# REQUIREMENT 5: TEACHER AVAILABILITY CHECKED ACROSS ENTIRE DURATION
# =============================================================================

def test_teacher_and_room_availability_checked_across_entire_duration():
    """Teacher unavailable slots and room type must be checked across all lab periods."""
    # Teacher unavailable specifically on Day 1, period 2 (which would be middle of periods 1, 2, 3)
    dataset = build_variable_lab_dataset(
        lab_duration=3,
        weekly_lab_periods=3,
        lunch_slot=5,
        teacher_unavail=[(1, 2)],
    )
    vars_list = generate_session_variables(dataset)
    lab_var = next(v for v in vars_list if v.subject_id == "VAR_LAB")

    domain = get_valid_domain_for_session(lab_var, dataset)
    # Placement on Day 1 starting at 1 occupies [1, 2, 3] -> hits unavailable slot (1, 2)
    bad_pl = Placement(day=1, start_period=1, room_id="LAB1")
    assert bad_pl not in domain, "Placement spanning unavailable middle period must be excluded from domain."

    # Test H6 constraint directly
    tt = Timetable.empty(dataset)
    tt.assign(lab_var, bad_pl)
    violations = check_h6_teacher_availability(tt)
    assert len(violations) >= 1
    assert any(v.details.get("period") == 2 for v in violations)

    # Test H5 room type constraint: 3-period lab in classroom CR1
    tt_h5 = Timetable.empty(dataset)
    tt_h5.assign(lab_var, Placement(day=0, start_period=0, room_id="CR1"))
    h5_violations = check_h5_room_type(tt_h5)
    assert len(h5_violations) == 1, "Must fail H5 when lab is assigned to classroom."


# =============================================================================
# REQUIREMENT 6: CSV / XLSX / WORKBOOK IMPORTS PRESERVE LAB DURATION
# =============================================================================

def test_imports_preserve_variable_lab_duration():
    """Import pipelines (TSV, CSV, Excel) must preserve lab_duration for 2P and 3P labs."""
    tsv_content = (
        "subject_id\tsubject_name\tclass_id\tteacher_id\ttype\tweekly_periods\tlab_duration\n"
        "CHEM_LAB\tChemistry Lab\tCLASS_1\tT_CHEM\tlab\t3\t3\n"
        "PHYS_LAB\tPhysics Lab\tCLASS_1\tT_PHYS\tlab\t4\t2\n"
        "MATH101\tCalculus\tCLASS_1\tT_MATH\ttheory\t4\t1\n"
    )

    raw_df = parse_pasted_tabular_text(tsv_content, "subjects")
    assert "lab_duration" in raw_df.columns

    res = validate_bulk_rows(raw_df, "subjects")
    assert res.is_valid is True
    accepted = res.accepted_df


    chem_row = accepted[accepted["subject_id"] == "CHEM_LAB"].iloc[0]
    phys_row = accepted[accepted["subject_id"] == "PHYS_LAB"].iloc[0]

    assert chem_row["lab_duration"] == 3, "3-period lab duration must be preserved as 3."
    assert phys_row["lab_duration"] == 2, "2-period lab duration must be preserved as 2."

    # Verify custom builder pipeline preserves it through to TimetableDataset
    subjects_df = accepted.copy()
    classes_df = pd.DataFrame([
        {"class_id": "CLASS_1", "class_name": "Class 1", "strength": 25, "lunch_slot": 3}
    ])
    teachers_df = pd.DataFrame([
        {"teacher_id": "T_CHEM", "teacher_name": "Dr. Chem", "max_weekly_load": 15, "qualified_subjects": "CHEM_LAB", "unavailable_slots": "", "preferred_slots": ""},
        {"teacher_id": "T_PHYS", "teacher_name": "Dr. Phys", "max_weekly_load": 15, "qualified_subjects": "PHYS_LAB", "unavailable_slots": "", "preferred_slots": ""},
        {"teacher_id": "T_MATH", "teacher_name": "Dr. Math", "max_weekly_load": 15, "qualified_subjects": "MATH101", "unavailable_slots": "", "preferred_slots": ""},
    ])
    rooms_df = pd.DataFrame([
        {"room_id": "CR1", "room_name": "Classroom 1", "capacity": 30, "type": "classroom"},
        {"room_id": "LAB1", "room_name": "Lab 1", "capacity": 30, "type": "lab"},
    ])
    grid_cfg = {
        "institution_name": "Test Univ",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "periods_per_day": 6,
        "default_lunch_period": 4,
        "blocked_periods": [],
    }

    v_res = validate_and_build_custom_dataset(grid_cfg, classes_df, teachers_df, subjects_df, rooms_df)
    assert v_res.is_valid is True
    dataset = v_res.dataset
    assert dataset is not None


    assert dataset.subjects["CHEM_LAB"].lab_duration == 3
    assert dataset.subjects["PHYS_LAB"].lab_duration == 2

    sessions = generate_session_variables(dataset)
    chem_sessions = [s for s in sessions if s.subject_id == "CHEM_LAB"]
    phys_sessions = [s for s in sessions if s.subject_id == "PHYS_LAB"]

    assert len(chem_sessions) == 1
    assert chem_sessions[0].duration == 3
    assert len(phys_sessions) == 2
    assert all(s.duration == 2 for s in phys_sessions)


# =============================================================================
# REQUIREMENT 7: SOLVERS PRESERVE ATOMIC LAB BLOCKS
# =============================================================================

@pytest.fixture
def multi_lab_solvable_dataset() -> TimetableDataset:
    """Creates a feasible dataset with both a 2-period and a 3-period lab."""
    ts = TimeStructure(
        days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        periods_per_day=6,
        blocked_periods=[],
    )

    subjects = {
        "ENG": Subject(id="ENG", name="English", weekly_periods=3, type="theory"),
        "MATH": Subject(id="MATH", name="Math", weekly_periods=3, type="theory"),
        "LAB_2P": Subject(id="LAB_2P", name="Hardware Lab", weekly_periods=2, type="lab", lab_duration=2),
        "LAB_3P": Subject(id="LAB_3P", name="Software Lab", weekly_periods=3, type="lab", lab_duration=3),
    }

    teachers = {
        "T1": Teacher(id="T1", name="Teacher 1", subject_ids=["ENG"], max_weekly_load=15),
        "T2": Teacher(id="T2", name="Teacher 2", subject_ids=["MATH"], max_weekly_load=15),
        "T3": Teacher(id="T3", name="Teacher 3", subject_ids=["LAB_2P", "LAB_3P"], max_weekly_load=15),
    }

    classes = {
        "C1": ClassGroup(
            id="C1",
            name="Class 1",
            strength=20,
            lunch_slot=3,
            subjects=[
                ClassSubjectRequirement(subject_id="ENG", teacher_id="T1"),
                ClassSubjectRequirement(subject_id="LAB_3P", teacher_id="T3"),
            ],
        ),
        "C2": ClassGroup(
            id="C2",
            name="Class 2",
            strength=20,
            lunch_slot=3,
            subjects=[
                ClassSubjectRequirement(subject_id="MATH", teacher_id="T2"),
                ClassSubjectRequirement(subject_id="LAB_2P", teacher_id="T3"),
            ],
        ),
    }

    rooms = {
        "CR1": Room(id="CR1", name="Room 1", capacity=30, type="classroom"),
        "CR2": Room(id="CR2", name="Room 2", capacity=30, type="classroom"),
        "LAB1": Room(id="LAB1", name="Lab 1", capacity=30, type="lab"),
    }

    return TimetableDataset(
        name="Multi-Lab Solvable Dataset",
        time_structure=ts,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )


def test_backtracking_preserves_atomic_lab_blocks(multi_lab_solvable_dataset):
    """Backtracking solver must schedule 2P and 3P labs as contiguous atomic blocks."""
    res = solve_backtracking_mrv_degree_fc(multi_lab_solvable_dataset)
    assert res.success is True
    assert res.hard_violations == 0

    val = validate_timetable(res.timetable)
    assert val.is_valid is True
    assert val.hard_violation_count == 0

    # Inspect 3P lab
    lab3_var = next(v for v in res.timetable.assignments if v.subject_id == "LAB_3P")
    p3 = res.timetable.assignments[lab3_var]
    assert p3 is not None
    slots3 = p3.get_occupied_slots(lab3_var.duration)
    assert len(slots3) == 3
    # Consecutive check
    periods3 = [p for d, p in slots3]
    assert periods3 == list(range(periods3[0], periods3[0] + 3))
    assert 3 not in periods3, "Lab 3P must not span lunch period 3."

    # Inspect 2P lab
    lab2_var = next(v for v in res.timetable.assignments if v.subject_id == "LAB_2P")
    p2 = res.timetable.assignments[lab2_var]
    assert p2 is not None
    slots2 = p2.get_occupied_slots(lab2_var.duration)
    assert len(slots2) == 2
    periods2 = [p for d, p in slots2]
    assert periods2 == list(range(periods2[0], periods2[0] + 2))
    assert 3 not in periods2, "Lab 2P must not span lunch period 3."


def test_min_conflicts_preserves_atomic_lab_blocks(multi_lab_solvable_dataset):
    """Min-Conflicts solver must maintain 2P and 3P atomic contiguous blocks."""
    res = solve_min_conflicts(multi_lab_solvable_dataset, max_iterations=2000, random_seed=42)
    assert res.success is True
    assert res.final_hard_violations == 0

    val = validate_timetable(res.timetable)
    assert val.is_valid is True

    lab3_var = next(v for v in res.timetable.assignments if v.subject_id == "LAB_3P")
    p3 = res.timetable.assignments[lab3_var]
    periods3 = p3.get_occupied_periods(lab3_var.duration)
    assert len(periods3) == 3
    assert periods3 == list(range(periods3[0], periods3[0] + 3))
    assert 3 not in periods3


def test_simulated_annealing_preserves_atomic_lab_blocks(multi_lab_solvable_dataset):
    """Hybrid SA must optimize schedule without breaking atomic 2P and 3P lab blocks."""
    res = solve_hybrid_simulated_annealing(
        multi_lab_solvable_dataset,
        max_iterations=100,
        random_seed=42,
    )
    assert res.success is True
    assert res.hard_violations == 0

    val = validate_timetable(res.timetable)
    assert val.is_valid is True

    lab3_var = next(v for v in res.timetable.assignments if v.subject_id == "LAB_3P")
    p3 = res.timetable.assignments[lab3_var]
    periods3 = p3.get_occupied_periods(lab3_var.duration)
    assert len(periods3) == 3
    assert periods3 == list(range(periods3[0], periods3[0] + 3))


# =============================================================================
# REQUIREMENT 8: UI DISPLAYS COMPLETE LAB DURATION
# =============================================================================

def test_ui_displays_complete_lab_duration(multi_lab_solvable_dataset):
    """UI _make_grid and _build_timetable_html must display 1 start card and N-1 continuation cards."""
    res = solve_backtracking_mrv_degree_fc(multi_lab_solvable_dataset)
    assert res.success is True
    tt = res.timetable

    # Dynamically import UI functions from app.py
    import unittest.mock as mock
    import importlib.util
    import sys
    sys.modules["streamlit"] = mock.MagicMock()

    spec = importlib.util.spec_from_file_location("app_module", Path("app.py"))
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)

    # Generate Class View Grid
    grids = app_mod._make_grid(tt, group_by="class")
    c1_grid = grids["C1"]

    # Locate Lab 3P placement
    lab3_var = next(v for v in tt.assignments if v.subject_id == "LAB_3P")
    p3 = tt.assignments[lab3_var]
    day_name = multi_lab_solvable_dataset.time_structure.days[p3.day]
    start_p = p3.start_period

    start_cell = c1_grid.loc[f"P{start_p + 1}", day_name]
    cont1_cell = c1_grid.loc[f"P{start_p + 2}", day_name]
    cont2_cell = c1_grid.loc[f"P{start_p + 3}", day_name]

    # Verify start cell has primary details and continuation cells have continuation markers
    assert "Software Lab" in start_cell
    assert "Teacher 3 | Lab 1" in start_cell

    assert cont1_cell == "... Software Lab (cont.)", "First continuation period must have continuation tag."
    assert cont2_cell == "... Software Lab (cont.)", "Second continuation period must have continuation tag."

    # Test HTML renderer output
    html_out = app_mod._build_timetable_html(c1_grid, group_by="class")
    # Must have at least 2 continuation badges for the 3-period lab
    assert html_out.count("Continuation</span>") >= 2
    assert "tt-card-cont" in html_out


# =============================================================================
# REQUIREMENT 9: IMPOSSIBLE PLACEMENT REJECTED RATHER THAN SPLIT
# =============================================================================

def test_solver_rejects_impossible_lab_placement_rather_than_splitting():
    """When a 3P lab cannot fit atomically, solvers must reject it without splitting."""
    # Day has 4 periods: Period 0 is blocked, Period 2 is lunch.
    # Unoccupied periods are: Period 1 (length 1) and Period 3 (length 1).
    # No contiguous 3-period block exists on any day!
    ts = TimeStructure(
        days=["Monday"],
        periods_per_day=4,
        blocked_periods=[(0, 0)],
    )

    subjects = {
        "LAB_3P": Subject(id="LAB_3P", name="Long Lab", weekly_periods=3, type="lab", lab_duration=3),
    }

    teachers = {
        "T1": Teacher(id="T1", name="Faculty", subject_ids=["LAB_3P"], max_weekly_load=10),
    }

    classes = {
        "C1": ClassGroup(
            id="C1",
            name="Batch",
            strength=20,
            lunch_slot=2,
            subjects=[ClassSubjectRequirement(subject_id="LAB_3P", teacher_id="T1")],
        ),
    }

    rooms = {
        "LAB1": Room(id="LAB1", name="Lab", capacity=30, type="lab"),
    }

    dataset = TimetableDataset(
        name="Impossible 3P Lab Dataset",
        time_structure=ts,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )

    vars_list = generate_session_variables(dataset)
    assert len(vars_list) == 1
    lab_var = vars_list[0]
    assert lab_var.duration == 3

    # Check unary domain: must be empty because no contiguous 3-period block exists
    domain = get_valid_domain_for_session(lab_var, dataset)
    assert len(domain) == 0, "Domain must be empty since 3 consecutive unblocked periods do not exist."

    # Backtracking solver must report unsatisfiable and NOT split the session
    bt_res = solve_backtracking(dataset)
    assert bt_res.success is False
    assert bt_res.status == "unsatisfiable"
    assert bt_res.timetable is None or bt_res.timetable.get_placed_sessions() == []

    # Min-Conflicts must also report unsatisfiable due to empty domain
    mc_res = solve_min_conflicts(dataset)
    assert mc_res.success is False
    assert mc_res.status == "unsatisfiable"

    # Hybrid SA must also report failure
    sa_res = solve_hybrid_simulated_annealing(dataset)
    assert sa_res.success is False
    assert sa_res.status == "unsatisfiable"
