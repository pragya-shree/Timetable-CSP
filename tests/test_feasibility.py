"""Unit tests proving timetable solvability via hand-constructed feasible schedules.

No solver or search algorithm was used. Both schedules were constructed explicitly
by structured reasoning and mathematical patterns.
"""

import json
from pathlib import Path
import pytest
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.models import load_dataset_from_file
from core.sessions import Timetable


def test_small_dataset_feasible_schedule():
    """Validates that small_feasible_schedule.json achieves 0 violations."""
    dataset_path = Path("datasets/small_dataset.json")
    schedule_path = Path("datasets/small_feasible_schedule.json")

    assert dataset_path.exists(), "small_dataset.json must exist."
    assert schedule_path.exists(), "small_feasible_schedule.json must exist."

    dataset = load_dataset_from_file(dataset_path)
    with open(schedule_path, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)

    timetable = Timetable.from_dict(schedule_data, dataset)
    result = validate_timetable(timetable)

    assert result.is_complete is True, "Small schedule must be complete."
    assert result.is_valid is True, "Small schedule must be valid."
    assert result.hard_violation_count == 0, f"Expected 0 violations, got {result.violations}"
    assert len(result.violations) == 0

    cost_info = calculate_cost(timetable)
    assert cost_info.hard_violations == 0
    assert cost_info.hard_cost == 0
    assert cost_info.is_valid is True


def test_large_dataset_feasible_schedule():
    """Validates that large_feasible_schedule.json achieves 0 violations."""
    dataset_path = Path("datasets/large_dataset.json")
    schedule_path = Path("datasets/large_feasible_schedule.json")

    assert dataset_path.exists(), "large_dataset.json must exist."
    assert schedule_path.exists(), "large_feasible_schedule.json must exist."

    dataset = load_dataset_from_file(dataset_path)
    with open(schedule_path, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)

    timetable = Timetable.from_dict(schedule_data, dataset)
    result = validate_timetable(timetable)

    assert result.is_complete is True, "Large schedule must be complete."
    assert result.is_valid is True, "Large schedule must be valid."
    assert result.hard_violation_count == 0, f"Expected 0 violations, got {result.violations}"
    assert len(result.violations) == 0

    cost_info = calculate_cost(timetable)
    assert cost_info.hard_violations == 0
    assert cost_info.hard_cost == 0
    assert cost_info.is_valid is True


@pytest.mark.parametrize("dataset_file,schedule_file", [
    ("datasets/small_dataset.json", "datasets/small_feasible_schedule.json"),
    ("datasets/large_dataset.json", "datasets/large_feasible_schedule.json"),
])
def test_independent_audit_assertions_on_feasible_schedules(dataset_file, schedule_file):
    """Independently verifies every requirement of Audit 2 without relying solely on validators."""
    from collections import defaultdict
    from core.sessions import generate_session_variables

    dataset = load_dataset_from_file(dataset_file)
    with open(schedule_file, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)
    tb = Timetable.from_dict(schedule_data, dataset)

    # 1. All session variables accounted for exactly
    expected_vars = generate_session_variables(dataset)
    placed_sessions = tb.get_placed_sessions()
    assert len(placed_sessions) == len(expected_vars)
    assert tb.is_complete() is True

    # 2. Every class receives exactly its required weekly subject periods
    class_subject_periods = defaultdict(int)
    for var, placement in placed_sessions:
        class_subject_periods[(var.class_id, var.subject_id)] += var.duration

    for class_id, class_group in dataset.classes.items():
        for req in class_group.subjects:
            required = dataset.get_required_periods(class_id, req.subject_id)
            assigned = class_subject_periods[(class_id, req.subject_id)]
            assert assigned == required, (
                f"{dataset_file}: Class {class_id} subject {req.subject_id} has {assigned} periods, required {required}"
            )

    # 3. Every teacher is qualified to teach the assigned subject
    for var, placement in placed_sessions:
        teacher = dataset.teachers[var.teacher_id]
        assert var.subject_id in teacher.subject_ids, (
            f"Teacher {teacher.id} not qualified to teach {var.subject_id}"
        )

    # 4. No teacher exceeds their maximum load
    teacher_loads = defaultdict(int)
    for var, placement in placed_sessions:
        teacher_loads[var.teacher_id] += var.duration
    for teacher_id, teacher in dataset.teachers.items():
        assert teacher_loads[teacher_id] <= teacher.max_weekly_load, (
            f"Teacher {teacher_id} load {teacher_loads[teacher_id]} exceeds max {teacher.max_weekly_load}"
        )

    # 5. No teacher, class, or room clashes across all occupied periods
    teacher_occupancy = defaultdict(list)
    class_occupancy = defaultdict(list)
    room_occupancy = defaultdict(list)

    for var, placement in placed_sessions:
        for slot in placement.get_occupied_slots(var.duration):
            teacher_occupancy[(slot[0], slot[1], var.teacher_id)].append(var.session_id)
            class_occupancy[(slot[0], slot[1], var.class_id)].append(var.session_id)
            room_occupancy[(slot[0], slot[1], placement.room_id)].append(var.session_id)

    for key, sessions in teacher_occupancy.items():
        assert len(sessions) == 1, f"Teacher clash at {key}: {sessions}"
    for key, sessions in class_occupancy.items():
        assert len(sessions) == 1, f"Class clash at {key}: {sessions}"
    for key, sessions in room_occupancy.items():
        assert len(sessions) == 1, f"Room clash at {key}: {sessions}"

    # 6. Every room has adequate capacity and correct type
    for var, placement in placed_sessions:
        room = dataset.rooms[placement.room_id]
        class_group = dataset.classes[var.class_id]
        assert room.capacity >= class_group.strength, (
            f"Room {room.id} cap {room.capacity} < class {class_group.id} strength {class_group.strength}"
        )
        expected_type = "lab" if var.session_type == "lab" else "classroom"
        assert room.type == expected_type, (
            f"Room {room.id} type {room.type} != expected {expected_type}"
        )

    # 7. Teacher availability and global blocked periods respected
    ts = dataset.time_structure
    for var, placement in placed_sessions:
        teacher = dataset.teachers[var.teacher_id]
        for slot in placement.get_occupied_slots(var.duration):
            assert not ts.is_blocked(slot[0], slot[1]), (
                f"Session {var.session_id} on blocked slot {slot}"
            )
            assert slot not in teacher.unavailable_slots, (
                f"Teacher {teacher.id} on unavailable slot {slot}"
            )

    # 8. Lunch slots respected
    for var, placement in placed_sessions:
        class_group = dataset.classes[var.class_id]
        for slot in placement.get_occupied_slots(var.duration):
            assert slot[1] != class_group.lunch_slot, (
                f"Session {var.session_id} scheduled on class {class_group.id} lunch slot {slot}"
            )

    # 9. Labs occupy required duration consecutively
    for var, placement in placed_sessions:
        if var.session_type == "lab":
            subject = dataset.subjects[var.subject_id]
            assert var.duration == subject.lab_duration, (
                f"Lab {var.session_id} duration {var.duration} != expected {subject.lab_duration}"
            )
            periods = placement.get_occupied_periods(var.duration)
            assert periods == list(range(periods[0], periods[0] + var.duration)), (
                f"Lab {var.session_id} periods {periods} not consecutive"
            )

