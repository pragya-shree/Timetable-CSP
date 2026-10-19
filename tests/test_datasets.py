"""Unit tests for verifying schema, size minimums, and mathematical feasibility of datasets."""

from pathlib import Path
import pytest
from core.models import check_dataset_feasibility, load_dataset_from_file


def test_small_dataset_structure_and_feasibility():
    """Verify small_dataset.json satisfies all spec requirements."""
    path = Path("datasets/small_dataset.json")
    assert path.exists(), "small_dataset.json must exist."

    dataset = load_dataset_from_file(path)

    # Minimum size assertions
    assert len(dataset.classes) >= 3, "Small dataset must have at least 3 classes."
    assert len(dataset.teachers) >= 6, "Small dataset must have at least 6 teachers."
    assert len(dataset.rooms) >= 4, "Small dataset must have at least 4 rooms."
    assert len(dataset.time_structure.days) == 5, "Must have 5 days."
    assert dataset.time_structure.periods_per_day == 6, "Must have 6 periods per day."

    # Must contain theory and lab, with at least one lab of duration 2
    has_lab_dur_2 = any(
        s.type == "lab" and s.lab_duration == 2 for s in dataset.subjects.values()
    )
    assert has_lab_dur_2, "Small dataset must contain at least one lab with lab_duration 2."

    # Programmatic feasibility checks
    issues = check_dataset_feasibility(dataset)
    assert len(issues) == 0, f"Small dataset has feasibility bottlenecks: {issues}"


def test_large_dataset_structure_and_feasibility():
    """Verify large_dataset.json satisfies all spec requirements."""
    path = Path("datasets/large_dataset.json")
    assert path.exists(), "large_dataset.json must exist."

    dataset = load_dataset_from_file(path)

    # Minimum size assertions
    assert len(dataset.classes) >= 8, "Large dataset must have at least 8 classes."
    assert len(dataset.teachers) >= 15, "Large dataset must have at least 15 teachers."
    assert len(dataset.rooms) >= 8, "Large dataset must have at least 8 rooms."
    assert len(dataset.time_structure.days) == 5, "Must have 5 days."
    assert dataset.time_structure.periods_per_day == 7, "Must have 7 periods per day."

    # Must contain lab of duration 2
    has_lab_dur_2 = any(
        s.type == "lab" and s.lab_duration == 2 for s in dataset.subjects.values()
    )
    assert has_lab_dur_2, "Large dataset must contain at least one lab with lab_duration 2."

    # Programmatic feasibility checks
    issues = check_dataset_feasibility(dataset)
    assert len(issues) == 0, f"Large dataset has feasibility bottlenecks: {issues}"
