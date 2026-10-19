"""Unit tests for core/constraints.py (individual H1-H9 tests, STRUCT, and edge cases)."""

import pytest
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
from core.models import load_dataset_from_dict
from core.sessions import Placement, SessionVar, Timetable


def make_test_environment():
    """Builds a predictable dataset for isolated constraint tests."""
    data = {
        "name": "Constraint Test Rig",
        "time_structure": {
            "days": ["Monday", "Tuesday"],
            "periods_per_day": 5,
            "blocked_periods": [[0, 0]],  # Monday period 0 blocked
        },
        "subjects": [
            {"id": "MATH", "name": "Mathematics", "weekly_periods": 2, "type": "theory"},
            {"id": "PHYS", "name": "Physics", "weekly_periods": 2, "type": "theory"},
            {"id": "LAB", "name": "Computing Lab", "weekly_periods": 2, "type": "lab", "lab_duration": 2},
        ],
        "teachers": [
            {
                "id": "T1",
                "name": "Teacher One",
                "subject_ids": ["MATH", "PHYS"],
                "unavailable_slots": [[1, 4]],  # Tuesday period 4 unavailable
                "max_weekly_load": 3,           # Low max load to test H8
            },
            {
                "id": "T2",
                "name": "Teacher Two",
                "subject_ids": ["LAB"],
                "unavailable_slots": [],
                "max_weekly_load": 10,
            },
        ],
        "rooms": [
            {"id": "CR_BIG", "name": "Large Classroom", "capacity": 50, "type": "classroom"},
            {"id": "CR_SMALL", "name": "Small Classroom", "capacity": 20, "type": "classroom"},
            {"id": "LAB_ROOM", "name": "Lab Room", "capacity": 40, "type": "lab"},
        ],
        "classes": [
            {
                "id": "C1",
                "name": "Class 1",
                "strength": 30,  # larger than CR_SMALL
                "lunch_slot": 2,
                "subjects": [
                    {"subject_id": "MATH", "teacher_id": "T1", "weekly_periods": 2},
                    {"subject_id": "LAB", "teacher_id": "T2", "weekly_periods": 2},
                ],
            },
            {
                "id": "C2",
                "name": "Class 2",
                "strength": 20,
                "lunch_slot": 3,
                "subjects": [
                    {"subject_id": "PHYS", "teacher_id": "T1", "weekly_periods": 2},
                ],
            },
        ],
    }
    dataset = load_dataset_from_dict(data)
    return dataset


def test_clean_schedule_no_violations():
    """A perfectly placed schedule must have zero violations."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)

    # C1 MATH (2 sessions): Mon p1 in CR_BIG (T1), Tue p1 in CR_BIG (T1)
    # C1 LAB (1 session, dur 2): Mon p3-4 in LAB_ROOM (T2)
    # C2 PHYS (2 sessions): Mon p3 in CR_SMALL (T1), Tue p2 in CR_SMALL (T1)
    # Note: T1 teaches 4 periods total. But T1 max load is 3!
    # Let's adjust T1 max load in memory for clean test
    dataset.teachers["T1"] = dataset.teachers["T1"].__class__(
        id="T1", name="Teacher One", subject_ids=["MATH", "PHYS"],
        unavailable_slots=[(1, 4)], max_weekly_load=10
    )

    var_by_id = {v.session_id: v for v in tb.assignments}
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=1, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=3, room_id="LAB_ROOM"))
    tb.assign(var_by_id["C2_PHYS_s1"], Placement(day=0, start_period=4, room_id="CR_SMALL"))
    tb.assign(var_by_id["C2_PHYS_s2"], Placement(day=1, start_period=2, room_id="CR_SMALL"))

    res = validate_timetable(tb)
    assert res.is_complete is True
    assert res.is_valid is True
    assert res.hard_violation_count == 0
    assert len(res.violations) == 0


def test_h1_teacher_clash():
    """H1: Teacher scheduled in two different classes at the same slot."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Place C1_MATH_s1 and C2_PHYS_s1 at the exact same slot (Mon p1) with same teacher T1
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C2_PHYS_s1"], Placement(day=0, start_period=1, room_id="CR_SMALL"))

    violations = check_h1_teacher_clash(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H1"
    assert "T1" in violations[0].message
    assert set(violations[0].sessions_involved) == {"C1_MATH_s1", "C2_PHYS_s1"}


def test_h2_class_clash():
    """H2: Class scheduled in two different sessions at the same slot."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Double book class C1 at Mon p1
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=1, room_id="CR_SMALL"))

    violations = check_h2_class_clash(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H2"
    assert "C1" in violations[0].message
    assert set(violations[0].sessions_involved) == {"C1_MATH_s1", "C1_MATH_s2"}


def test_h3_room_clash():
    """H3: Two different sessions placed in the same room at the same time."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Put C1_MATH_s1 (T1) and C1_LAB_s1 (T2) in CR_BIG at Mon p1
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C2_PHYS_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))

    violations = check_h3_room_clash(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H3"
    assert "CR_BIG" in violations[0].message


def test_multi_period_lab_overlap_counting():
    """Multi-period lab overlap should count 1 violation per overlapping period."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # C1_LAB_s1 duration 2 placed at Mon p1-p2 in LAB_ROOM
    # Suppose another session overlaps on both p1 and p2 in same room:
    # We assign C1_LAB_s1 and C1_MATH_s1 at p1 and C1_MATH_s2 at p2 in LAB_ROOM
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=1, room_id="LAB_ROOM"))
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="LAB_ROOM"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=2, room_id="LAB_ROOM"))

    # In LAB_ROOM:
    # At p1: C1_LAB_s1 and C1_MATH_s1 clash (1 violation)
    # At p2: C1_LAB_s1 and C1_MATH_s2 clash (1 violation)
    violations = check_h3_room_clash(tb)
    assert len(violations) == 2
    assert all(v.constraint_id == "H3" for v in violations)


def test_h4_room_capacity():
    """H4: Session placed in room with capacity smaller than class strength."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Class C1 has strength 30. CR_SMALL capacity is 20.
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_SMALL"))

    violations = check_h4_room_capacity(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H4"
    assert "CR_SMALL" in violations[0].message
    assert "strength (30)" in violations[0].message


def test_h5_room_type():
    """H5: Lab session in classroom, theory session in lab."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Lab session in CR_BIG (classroom)
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    # Theory session in LAB_ROOM (lab)
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=3, room_id="LAB_ROOM"))

    violations = check_h5_room_type(tb)
    assert len(violations) == 2
    assert all(v.constraint_id == "H5" for v in violations)


def test_h6_teacher_availability_and_blocked():
    """H6: Placements on teacher unavailable slots or global blocked slots."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # (0, 0) is global blocked period (assembly)
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=0, room_id="CR_BIG"))
    # (1, 4) is teacher T1 unavailable slot
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=1, start_period=4, room_id="CR_BIG"))

    violations = check_h6_teacher_availability(tb)
    assert len(violations) == 2
    assert all(v.constraint_id == "H6" for v in violations)
    assert any("global blocked period" in v.message for v in violations)
    assert any("unavailable slot" in v.message for v in violations)


def test_h7_weekly_periods():
    """H7: Weekly periods met quota (|assigned - required|)."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # C1 needs: MATH: 2, LAB: 2.
    # C2 needs: PHYS: 2.
    # Total required = 6 periods.
    # Completely empty timetable has 6 missing periods!
    violations = check_h7_weekly_periods(tb)
    assert len(violations) == 6
    assert all(v.constraint_id == "H7" for v in violations)

    # Place only 1 session of C1 MATH (duration 1)
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    violations = check_h7_weekly_periods(tb)
    # C1 MATH now missing 1 period, others still missing 4 -> total 5 violations
    assert len(violations) == 5

    # Place C1_LAB_s1 (duration 2) -> should satisfy both required lab periods!
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=3, room_id="LAB_ROOM"))
    violations = check_h7_weekly_periods(tb)
    # C1 LAB is fully satisfied (discrepancy 0), so violations drops from 5 to 3!
    assert len(violations) == 3


def test_h8_teacher_max_load():
    """H8: Teacher exceeds max weekly load, accounting for multi-period lab duration."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Teacher T1 max load is 3 periods.
    # Place C1_MATH_s1 (1), C1_MATH_s2 (1), C2_PHYS_s1 (1), C2_PHYS_s2 (1) = 4 periods.
    # Excess = 4 - 3 = 1 period.
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=1, start_period=1, room_id="CR_BIG"))
    tb.assign(var_by_id["C2_PHYS_s1"], Placement(day=0, start_period=3, room_id="CR_SMALL"))
    tb.assign(var_by_id["C2_PHYS_s2"], Placement(day=1, start_period=2, room_id="CR_SMALL"))

    violations = check_h8_teacher_max_load(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H8"
    assert "T1" in violations[0].message
    assert "4 assigned periods" in violations[0].message

    # Test lab duration accounting on teacher load:
    # Set T2 max load to 1. T2 teaches C1_LAB_s1 which has duration 2.
    dataset.teachers["T2"] = dataset.teachers["T2"].__class__(
        id="T2", name="Teacher Two", subject_ids=["LAB"],
        unavailable_slots=[], max_weekly_load=1
    )
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=3, room_id="LAB_ROOM"))
    violations_t2 = [v for v in check_h8_teacher_max_load(tb) if "T2" in v.message]
    # T2 assigned 2 periods (from 1 lab session), exceeding max load 1 by 1 period
    assert len(violations_t2) == 1
    assert "T2" in violations_t2[0].message
    assert "2 assigned periods" in violations_t2[0].message


def test_h9_class_lunch_break():
    """H9: Session overlaps with class lunch break."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Class C1 has lunch at period 2
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=2, room_id="CR_BIG"))

    violations = check_h9_class_lunch_break(tb)
    assert len(violations) == 1
    assert violations[0].constraint_id == "H9"
    assert "C1" in violations[0].message
    assert "lunch slot" in violations[0].message


def test_structural_validity():
    """STRUCT: Catches out-of-bounds start periods, days, and labs extending past end of day."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Session placed on day 5 (only days 0 and 1 exist)
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=5, start_period=1, room_id="CR_BIG"))
    # Lab session (dur 2) placed at start_period 4 (periods_per_day is 5, so p4+2 = 6 > 5)
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=4, room_id="LAB_ROOM"))
    # Unknown room ID
    tb.assign(var_by_id["C2_PHYS_s1"], Placement(day=0, start_period=1, room_id="GHOST_ROOM"))

    violations = check_structural_validity(tb)
    assert len(violations) == 3
    assert all(v.constraint_id == "STRUCT" for v in violations)


def test_incomplete_timetable_is_never_valid():
    """An empty or partially assigned timetable is never marked is_valid."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)

    res = validate_timetable(tb)
    assert res.is_complete is False
    assert res.is_valid is False
    assert res.hard_violation_count > 0


def test_structural_validity_catches_corrupted_and_rogue_entities():
    """STRUCT must catch unknown class, unknown subject, wrong duration, and mismatched teacher."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)

    # 1. Unknown class ID
    rogue_var_class = SessionVar(
        session_id="GHOST_MATH_s1", class_id="UNKNOWN_CLASS", subject_id="MATH",
        session_no=1, teacher_id="T1", duration=1, session_type="theory"
    )
    tb.assignments[rogue_var_class] = Placement(day=0, start_period=1, room_id="CR_BIG")

    # 2. Unknown subject ID
    rogue_var_subj = SessionVar(
        session_id="C1_ASTRONOMY_s1", class_id="C1", subject_id="ASTRONOMY",
        session_no=1, teacher_id="T1", duration=1, session_type="theory"
    )
    tb.assignments[rogue_var_subj] = Placement(day=0, start_period=1, room_id="CR_BIG")

    # 3. Class C1 is not enrolled in PHYS (only MATH and LAB)
    rogue_var_enroll = SessionVar(
        session_id="C1_PHYS_s1", class_id="C1", subject_id="PHYS",
        session_no=1, teacher_id="T1", duration=1, session_type="theory"
    )
    tb.assignments[rogue_var_enroll] = Placement(day=0, start_period=1, room_id="CR_BIG")

    # 4. Wrong duration for lab (should be 2, but session has 1)
    # Wait, SessionVar.__post_init__ allows duration=1, but subject is lab with lab_duration=2
    rogue_var_dur = SessionVar(
        session_id="C1_LAB_s99", class_id="C1", subject_id="LAB",
        session_no=99, teacher_id="T2", duration=1, session_type="lab"
    )
    tb.assignments[rogue_var_dur] = Placement(day=0, start_period=1, room_id="LAB_ROOM")

    # 5. Teacher mismatched: C1 MATH requires T1, but session specifies T2
    # Note: T2 is not qualified for MATH, but even if T2 were qualified, class requirement designates T1
    rogue_var_teacher = SessionVar(
        session_id="C1_MATH_s99", class_id="C1", subject_id="MATH",
        session_no=99, teacher_id="T2", duration=1, session_type="theory"
    )
    tb.assignments[rogue_var_teacher] = Placement(day=0, start_period=1, room_id="CR_BIG")

    violations = check_structural_validity(tb)
    violation_msgs = [v.message for v in violations]

    assert any("references unknown class 'UNKNOWN_CLASS'" in m for m in violation_msgs)
    assert any("references unknown subject 'ASTRONOMY'" in m for m in violation_msgs)
    assert any("not enrolled in subject 'PHYS'" in m for m in violation_msgs)
    assert any("duration 1 does not match subject 'LAB' expected duration 2" in m for m in violation_msgs)
    assert any("class 'C1' requires teacher 'T1'" in m for m in violation_msgs)


def test_h7_detects_rogue_extra_subjects():
    """H7 must penalize periods of subjects assigned to a class that never required them."""
    dataset = make_test_environment()
    tb = Timetable.empty(dataset)

    # Class C1 only requires MATH (2) and LAB (2).
    # If a session for PHYS (dur 1) is placed for C1, H7 must record 1 violation for that extra period.
    rogue_var = SessionVar(
        session_id="C1_PHYS_s1", class_id="C1", subject_id="PHYS",
        session_no=1, teacher_id="T1", duration=1, session_type="theory"
    )
    tb.assignments[rogue_var] = Placement(day=0, start_period=1, room_id="CR_BIG")

    violations = check_h7_weekly_periods(tb)
    # Expected:
    # C1 MATH: 2 missing
    # C1 LAB: 2 missing
    # C2 PHYS: 2 missing
    # C1 PHYS: 1 excess!
    # Total = 7 violations
    assert len(violations) == 7
    excess_v = [v for v in violations if "excess" in v.message]
    assert len(excess_v) == 1
    assert "Class 'C1' subject 'PHYS' has 1 excess period(s)" in excess_v[0].message

