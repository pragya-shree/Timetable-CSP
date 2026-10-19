"""Unit tests for core/sessions.py (CSP variable generation and Placement representations)."""

import pytest
from core.models import load_dataset_from_dict
from core.sessions import (
    Placement,
    SessionVar,
    Timetable,
    generate_session_variables,
)


def sample_dataset():
    data = {
        "name": "Session Test",
        "time_structure": {"days": ["Mon", "Tue", "Wed"], "periods_per_day": 5},
        "subjects": [
            {"id": "MATH", "name": "Mathematics", "weekly_periods": 3, "type": "theory"},
            {"id": "PHY_LAB", "name": "Physics Lab", "weekly_periods": 4, "type": "lab", "lab_duration": 2},
        ],
        "teachers": [
            {"id": "T_M", "name": "Math Teacher", "subject_ids": ["MATH"], "max_weekly_load": 10},
            {"id": "T_P", "name": "Physics Teacher", "subject_ids": ["PHY_LAB"], "max_weekly_load": 10},
        ],
        "rooms": [
            {"id": "CR1", "name": "Classroom 1", "capacity": 30, "type": "classroom"},
            {"id": "LAB1", "name": "Lab 1", "capacity": 30, "type": "lab"},
        ],
        "classes": [
            {
                "id": "SEC_A",
                "name": "Section A",
                "strength": 20,
                "lunch_slot": 2,
                "subjects": [
                    {"subject_id": "MATH", "teacher_id": "T_M"},
                    {"subject_id": "PHY_LAB", "teacher_id": "T_P"},
                ],
            }
        ],
    }
    return load_dataset_from_dict(data)


def test_session_variable_generation():
    """Verify theory generates duration=1 sessions and lab generates duration=lab_duration sessions."""
    dataset = sample_dataset()
    vars = generate_session_variables(dataset)

    # MATH: 3 weekly periods -> 3 sessions of duration 1
    math_sessions = [v for v in vars if v.subject_id == "MATH"]
    assert len(math_sessions) == 3
    assert all(v.duration == 1 for v in math_sessions)
    assert all(v.session_type == "theory" for v in math_sessions)
    assert [v.session_no for v in math_sessions] == [1, 2, 3]

    # PHY_LAB: 4 weekly periods, lab_duration 2 -> 2 sessions of duration 2
    lab_sessions = [v for v in vars if v.subject_id == "PHY_LAB"]
    assert len(lab_sessions) == 2
    assert all(v.duration == 2 for v in lab_sessions)
    assert all(v.session_type == "lab" for v in lab_sessions)
    assert [v.session_no for v in lab_sessions] == [1, 2]

    # Total occupied periods = 3*1 + 2*2 = 7 periods
    total_periods = sum(v.duration for v in vars)
    assert total_periods == 7


def test_placement_occupied_slots():
    """Test Placement occupied slot computation."""
    p_theory = Placement(day=0, start_period=1, room_id="CR1")
    assert p_theory.get_occupied_periods(duration=1) == [1]
    assert p_theory.get_occupied_slots(duration=1) == [(0, 1)]

    p_lab = Placement(day=2, start_period=3, room_id="LAB1")
    assert p_lab.get_occupied_periods(duration=2) == [3, 4]
    assert p_lab.get_occupied_slots(duration=2) == [(2, 3), (2, 4)]


def test_timetable_lifecycle():
    """Test empty initialization, assignment, completion check, and serialization."""
    dataset = sample_dataset()
    tb = Timetable.empty(dataset)

    assert len(tb.assignments) == 5
    assert tb.is_complete() is False
    assert len(tb.get_placed_sessions()) == 0
    assert len(tb.get_unassigned_sessions()) == 5

    # Assign one variable
    var = list(tb.assignments.keys())[0]
    placement = Placement(day=0, start_period=0, room_id="CR1")
    tb.assign(var, placement)

    assert tb.is_complete() is False
    assert len(tb.get_placed_sessions()) == 1
    assert len(tb.get_unassigned_sessions()) == 4

    # Assign the rest
    for v in list(tb.assignments.keys())[1:]:
        tb.assign(v, Placement(day=1, start_period=0, room_id="CR1"))

    assert tb.is_complete() is True

    # Roundtrip serialization
    data = tb.to_dict()
    assert data["is_complete"] is True
    reconstructed = Timetable.from_dict(data, dataset)
    assert reconstructed.is_complete() is True
    assert len(reconstructed.get_placed_sessions()) == 5


def test_is_complete_with_partial_dictionary():
    """Timetable with only a subset of required variables must return is_complete == False."""
    dataset = sample_dataset()
    vars = generate_session_variables(dataset)
    # Only 1 variable out of 5 is in assignments dict
    partial_tb = Timetable(dataset=dataset, assignments={vars[0]: Placement(day=0, start_period=0, room_id="CR1")})
    assert partial_tb.is_complete() is False


def test_from_dict_rejects_duplicate_placements():
    """from_dict must raise ValidationError if the JSON schedule contains duplicate session placements."""
    dataset = sample_dataset()
    data = {
        "dataset_name": dataset.name,
        "is_complete": False,
        "placements": [
            {"session_id": "SEC_A_MATH_s1", "day": 0, "start_period": 0, "room_id": "CR1"},
            {"session_id": "SEC_A_MATH_s1", "day": 1, "start_period": 1, "room_id": "CR1"},
        ],
    }
    with pytest.raises(Exception, match="Duplicate placement for session 'SEC_A_MATH_s1'"):
        Timetable.from_dict(data, dataset)


def test_get_valid_domain_for_session():
    """get_valid_domain_for_session must filter out unary-incompatible placements."""
    from core.sessions import get_valid_domain_for_session
    dataset = sample_dataset()
    vars = generate_session_variables(dataset)
    math_var = [v for v in vars if v.subject_id == "MATH"][0]
    lab_var = [v for v in vars if v.subject_id == "PHY_LAB"][0]

    math_domain = get_valid_domain_for_session(math_var, dataset)
    lab_domain = get_valid_domain_for_session(lab_var, dataset)

    # 1. Math is theory -> all domain rooms must be classrooms
    assert all(dataset.rooms[p.room_id].type == "classroom" for p in math_domain)
    # 2. Lab is lab -> all domain rooms must be lab rooms
    assert all(dataset.rooms[p.room_id].type == "lab" for p in lab_domain)

    # 3. Class SEC_A lunch is period 2 -> no placement can occupy period 2
    for p in math_domain:
        assert 2 not in p.get_occupied_periods(math_var.duration)
    for p in lab_domain:
        assert 2 not in p.get_occupied_periods(lab_var.duration)

    # 4. Lab has duration 2 on a 5-period grid -> start_period can only be 0, 1, 2, 3 (not 4)
    # But period 2 is lunch, so:
    # start 0 -> [0, 1] (valid)
    # start 1 -> [1, 2] (hits lunch 2! invalid)
    # start 2 -> [2, 3] (hits lunch 2! invalid)
    # start 3 -> [3, 4] (valid)
    # So lab start_periods can only be 0 or 3!
    assert all(p.start_period in (0, 3) for p in lab_domain)
    # Lab duration 2 starting at 4 would overflow (4 + 2 = 6 > 5) -> must not exist
    assert not any(p.start_period == 4 for p in lab_domain)

