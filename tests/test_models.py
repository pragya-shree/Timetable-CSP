"""Unit tests for core/models.py (schema loading, entity models, and validation)."""

import pytest
from core.models import (
    ClassGroup,
    ClassSubjectRequirement,
    Room,
    Subject,
    Teacher,
    TimeStructure,
    TimetableDataset,
    ValidationError,
    check_dataset_feasibility,
    load_dataset_from_dict,
    load_dataset_from_file,
    validate_dataset_schema,
)


def sample_valid_dict():
    """Minimal valid dataset dictionary for testing."""
    return {
        "name": "Test University",
        "time_structure": {
            "days": ["Monday", "Tuesday"],
            "periods_per_day": 4,
            "blocked_periods": [[0, 0]],
        },
        "subjects": [
            {"id": "SUB1", "name": "Subject 1", "weekly_periods": 2, "type": "theory"},
            {"id": "LAB1", "name": "Lab 1", "weekly_periods": 2, "type": "lab", "lab_duration": 2},
        ],
        "teachers": [
            {
                "id": "T1",
                "name": "Teacher One",
                "subject_ids": ["SUB1", "LAB1"],
                "unavailable_slots": [[1, 3]],
                "max_weekly_load": 10,
            }
        ],
        "rooms": [
            {"id": "R1", "name": "Room 1", "capacity": 30, "type": "classroom"},
            {"id": "LR1", "name": "Lab Room 1", "capacity": 30, "type": "lab"},
        ],
        "classes": [
            {
                "id": "C1",
                "name": "Class 1",
                "strength": 25,
                "lunch_slot": 2,
                "subjects": [
                    {"subject_id": "SUB1", "teacher_id": "T1", "weekly_periods": 2},
                    {"subject_id": "LAB1", "teacher_id": "T1", "weekly_periods": 2},
                ],
            }
        ],
    }


def test_valid_dataset_loading():
    """Test that a well-formed dataset dictionary loads and validates cleanly."""
    d = sample_valid_dict()
    dataset = load_dataset_from_dict(d)
    assert dataset.name == "Test University"
    assert len(dataset.subjects) == 2
    assert len(dataset.teachers) == 1
    assert len(dataset.classes) == 1
    assert len(dataset.rooms) == 2
    assert dataset.time_structure.periods_per_day == 4


def test_time_structure_checks():
    """Test TimeStructure helper methods."""
    ts = TimeStructure(days=["Mon", "Tue"], periods_per_day=4, blocked_periods=[(0, 0)])
    assert ts.total_slots == 8
    assert ts.is_valid_slot(0, 2) is True
    assert ts.is_valid_slot(1, 4) is False  # out of bounds
    assert ts.is_valid_slot(2, 0) is False  # day out of bounds
    assert ts.is_blocked(0, 0) is True
    assert ts.is_blocked(0, 1) is False


def test_subject_lab_duration_validation():
    """Test Subject validation rules for theory and lab."""
    # Lab without duration
    with pytest.raises(ValidationError, match="must specify lab_duration > 0"):
        Subject(id="L", name="Lab", weekly_periods=4, type="lab", lab_duration=None)

    # Lab with non-divisible duration
    with pytest.raises(ValidationError, match="must be divisible by lab_duration"):
        Subject(id="L", name="Lab", weekly_periods=5, type="lab", lab_duration=2)

    # Theory with lab_duration specified
    with pytest.raises(ValidationError, match="must not specify lab_duration"):
        Subject(id="T", name="Theory", weekly_periods=3, type="theory", lab_duration=2)

    # Invalid subject type
    with pytest.raises(ValidationError, match="type must be 'theory' or 'lab'"):
        Subject(id="X", name="Invalid", weekly_periods=2, type="seminar")


def test_duplicate_entity_ids():
    """Test that duplicate IDs across any entity type are caught."""
    # Duplicate subject ID
    d = sample_valid_dict()
    d["subjects"].append({"id": "SUB1", "name": "Duplicate Subject", "weekly_periods": 2, "type": "theory"})
    with pytest.raises(ValidationError, match="Duplicate subject ID 'SUB1'"):
        load_dataset_from_dict(d)

    # Duplicate teacher ID
    d = sample_valid_dict()
    d["teachers"].append({"id": "T1", "name": "Duplicate Teacher", "subject_ids": ["SUB1"], "max_weekly_load": 10})
    with pytest.raises(ValidationError, match="Duplicate teacher ID 'T1'"):
        load_dataset_from_dict(d)

    # Duplicate class ID
    d = sample_valid_dict()
    d["classes"].append({"id": "C1", "name": "Duplicate Class", "strength": 20, "lunch_slot": 2, "subjects": []})
    with pytest.raises(ValidationError, match="Duplicate class ID 'C1'"):
        load_dataset_from_dict(d)

    # Duplicate room ID
    d = sample_valid_dict()
    d["rooms"].append({"id": "R1", "name": "Duplicate Room", "capacity": 20, "type": "classroom"})
    with pytest.raises(ValidationError, match="Duplicate room ID 'R1'"):
        load_dataset_from_dict(d)


def test_ineligible_teacher_binding():
    """Test that assigning a teacher to a subject they cannot teach raises ValidationError."""
    d = sample_valid_dict()
    # T1 only knows SUB1, not LAB1
    d["teachers"][0]["subject_ids"] = ["SUB1"]
    with pytest.raises(ValidationError, match="not qualified to teach it"):
        load_dataset_from_dict(d)


def test_unknown_entity_references():
    """Test that dangling references to unknown subjects, teachers, or rooms fail."""
    # Teacher references unknown subject
    d = sample_valid_dict()
    d["teachers"][0]["subject_ids"].append("UNKNOWN_SUB")
    with pytest.raises(ValidationError, match="references unknown subject 'UNKNOWN_SUB'"):
        load_dataset_from_dict(d)

    # Class references unknown teacher
    d = sample_valid_dict()
    d["classes"][0]["subjects"][0]["teacher_id"] = "UNKNOWN_TEACHER"
    with pytest.raises(ValidationError, match="references unknown teacher 'UNKNOWN_TEACHER'"):
        load_dataset_from_dict(d)


def test_out_of_bounds_slots():
    """Test that out-of-bounds unavailable slots or blocked slots are rejected."""
    d = sample_valid_dict()
    d["time_structure"]["blocked_periods"] = [[5, 2]]  # Only 2 days (0, 1) exist
    with pytest.raises(ValidationError, match="Blocked period \\(5, 2\\) is out of bounds"):
        load_dataset_from_dict(d)

    d = sample_valid_dict()
    d["teachers"][0]["unavailable_slots"] = [[0, 10]]  # Only 4 periods (0 to 3)
    with pytest.raises(ValidationError, match="unavailable slot \\(0, 10\\) is out of bounds"):
        load_dataset_from_dict(d)


def test_class_lunch_slot_bounds():
    """Test lunch slot must fit inside periods_per_day."""
    d = sample_valid_dict()
    d["classes"][0]["lunch_slot"] = 4  # Valid range is 0 to 3
    with pytest.raises(ValidationError, match="lunch_slot .* must be between 0 and 3"):
        load_dataset_from_dict(d)


def test_check_dataset_feasibility_detects_overload():
    """Test check_dataset_feasibility detects when teacher load exceeds limit."""
    d = sample_valid_dict()
    # Class requires 12 periods, teacher max load is 10
    d["classes"][0]["subjects"][0]["weekly_periods"] = 12
    d["teachers"][0]["max_weekly_load"] = 8
    dataset = load_dataset_from_dict(d)
    issues = check_dataset_feasibility(dataset)
    assert any("exceeding max load" in s for s in issues)
