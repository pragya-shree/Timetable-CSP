"""Unit tests for core/cost.py (S1-S3 soft penalties and objective cost function)."""

import pytest
from core.cost import (
    CostBreakdown,
    SoftWeights,
    calculate_cost,
    calculate_s1_gaps,
    calculate_s2_repeats,
    calculate_s3_non_consecutive_labs,
)
from core.models import load_dataset_from_dict
from core.sessions import Placement, SessionVar, Timetable


def sample_cost_dataset():
    data = {
        "name": "Cost Test Dataset",
        "time_structure": {
            "days": ["Mon", "Tue"],
            "periods_per_day": 6,
            "blocked_periods": [[0, 2]],  # Mon period 2 blocked
        },
        "subjects": [
            {"id": "MATH", "name": "Math", "weekly_periods": 4, "type": "theory"},
            {"id": "LAB", "name": "Computing Lab", "weekly_periods": 4, "type": "lab", "lab_duration": 2},
        ],
        "teachers": [
            {"id": "T1", "name": "Teacher One", "subject_ids": ["MATH"], "max_weekly_load": 10},
            {"id": "T2", "name": "Teacher Two", "subject_ids": ["LAB"], "max_weekly_load": 10},
        ],
        "rooms": [
            {"id": "CR1", "name": "Classroom 1", "capacity": 30, "type": "classroom"},
            {"id": "LAB1", "name": "Lab 1", "capacity": 30, "type": "lab"},
        ],
        "classes": [
            {
                "id": "C1",
                "name": "Class 1",
                "strength": 20,
                "lunch_slot": 3,  # Lunch at period 3
                "subjects": [
                    {"subject_id": "MATH", "teacher_id": "T1", "weekly_periods": 4},
                    {"subject_id": "LAB", "teacher_id": "T2", "weekly_periods": 4},
                ],
            }
        ],
    }
    return load_dataset_from_dict(data)


def test_s1_gaps_calculation():
    """Hand-computed S1 gap calculation:
    - Occupied: Period 0 and Period 5.
    - Potential gap periods: 1, 2, 3, 4.
    - Exclusions: Period 2 is blocked period. Period 3 is class lunch.
    - True gaps: Period 1 and Period 4 -> exactly 2 gaps!
    """
    dataset = sample_cost_dataset()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Place MATH session 1 at period 0, MATH session 2 at period 5 on Day 0
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=0, room_id="CR1"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=5, room_id="CR1"))

    gaps = calculate_s1_gaps(tb)
    assert gaps == 2


def test_s2_repeats_calculation():
    """Hand-computed S2 calculation:
    - 3 theory MATH sessions scheduled on Monday -> 3 - 1 = 2 repeats.
    - 2 lab sessions scheduled on Tuesday -> exempt, 0 repeats!
    """
    dataset = sample_cost_dataset()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # 3 theory sessions on Day 0
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=0, room_id="CR1"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=1, room_id="CR1"))
    tb.assign(var_by_id["C1_MATH_s3"], Placement(day=0, start_period=4, room_id="CR1"))

    # 2 lab sessions on Day 1
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=1, start_period=0, room_id="LAB1"))
    tb.assign(var_by_id["C1_LAB_s2"], Placement(day=1, start_period=4, room_id="LAB1"))

    repeats = calculate_s2_repeats(tb)
    # Only the theory repeats count
    assert repeats == 2


def test_s3_non_consecutive_labs():
    """Hand-computed S3 calculation:
    - Two lab sessions of the same lab on Day 1:
      Session 1 at periods 0-1.
      Session 2 at periods 4-5.
      Gap between them = 4 - 2 = 2 periods gap -> 2 penalties!
    - Interrupted lab:
      A lab placed at start_period 2 on Day 0 (dur 2) covers periods 2 and 3.
      Period 2 is blocked period, period 3 is lunch slot! Both interrupt the lab -> 2 penalties!
    """
    dataset = sample_cost_dataset()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Interrupted lab: Day 0 start 2 (periods 2, 3)
    tb.assign(var_by_id["C1_LAB_s1"], Placement(day=0, start_period=2, room_id="LAB1"))
    penalties = calculate_s3_non_consecutive_labs(tb)
    assert penalties == 2  # 1 for blocked period 2, 1 for lunch period 3


def test_cost_calculation_formula():
    """Verify total_cost = 1000 * hard + (w1 * S1 + w2 * S2 + w3 * S3)."""
    dataset = sample_cost_dataset()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    # Place MATH s1 at Day 0 p0 and MATH s2 at Day 0 p4 (CR1)
    # S1 gaps: between 0 and 4: p1, p2 (blocked), p3 (lunch) -> 1 gap (p1)
    # S2 repeats: 2 sessions of MATH on Day 0 -> 1 repeat
    # S3 non-consecutive: 0
    # Soft penalties (default weights 1): 1 + 1 + 0 = 2
    # Incomplete timetable: missing 2 MATH periods, 4 LAB periods = 6 H7 violations.
    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=0, room_id="CR1"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=4, room_id="CR1"))

    cost_info = calculate_cost(tb)
    assert cost_info.is_complete is False
    assert cost_info.is_valid is False
    assert cost_info.hard_violations == 6
    assert cost_info.hard_cost == 6000
    assert cost_info.s1_gaps == 1
    assert cost_info.s1_cost == 1
    assert cost_info.s2_repeats == 1
    assert cost_info.s2_cost == 1
    assert cost_info.s3_non_consecutive == 0
    assert cost_info.total_soft_cost == 2
    assert cost_info.total_cost == 6002


def test_soft_weights_scaling():
    """Verify weights scale individual soft penalty contributions."""
    dataset = sample_cost_dataset()
    tb = Timetable.empty(dataset)
    var_by_id = {v.session_id: v for v in tb.assignments}

    tb.assign(var_by_id["C1_MATH_s1"], Placement(day=0, start_period=0, room_id="CR1"))
    tb.assign(var_by_id["C1_MATH_s2"], Placement(day=0, start_period=4, room_id="CR1"))

    custom_weights = SoftWeights(w_s1=10, w_s2=5, w_s3=20)
    cost_info = calculate_cost(tb, weights=custom_weights)

    # s1_gaps = 1 -> cost = 10
    # s2_repeats = 1 -> cost = 5
    assert cost_info.s1_cost == 10
    assert cost_info.s2_cost == 5
    assert cost_info.total_soft_cost == 15
    assert cost_info.total_cost == 6000 + 15


def test_empty_and_partial_timetables_cost_guarantee():
    """An empty or partial timetable must incur heavy hard cost proportional to missing periods."""
    from core.models import load_dataset_from_file
    from pathlib import Path
    small_dataset = load_dataset_from_file(Path("datasets/small_dataset.json"))
    tb_empty = Timetable.empty(small_dataset)

    # In small_dataset.json:
    # 3 classes, each requires:
    # CSE_A: CS101 (4), CS102 (4), MA101 (4), CS101_LAB (2) = 14 periods
    # CSE_B: CS101 (4), CS102 (4), MA101 (4), CS101_LAB (2) = 14 periods
    # ECE_A: EC101 (4), MA101 (4), CS101 (4), CS101_LAB (2) = 14 periods
    # Total required = 42 periods.
    # Empty timetable has 0 assigned periods -> exactly 42 missing period violations!
    cost_info = calculate_cost(tb_empty)
    assert cost_info.is_complete is False
    assert cost_info.is_valid is False
    assert cost_info.hard_violations == 42
    assert cost_info.hard_cost == 42000
    assert cost_info.total_cost >= 42000

