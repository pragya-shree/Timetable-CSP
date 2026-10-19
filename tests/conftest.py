"""Shared pytest fixtures for Timetable Scheduler tests."""

from pathlib import Path
import pytest

from core.models import TimetableDataset, load_dataset_from_file


@pytest.fixture
def small_dataset() -> TimetableDataset:
    """Fixture providing loaded small_dataset.json."""
    return load_dataset_from_file(Path("datasets/small_dataset.json"))


@pytest.fixture
def large_dataset() -> TimetableDataset:
    """Fixture providing loaded large_dataset.json."""
    return load_dataset_from_file(Path("datasets/large_dataset.json"))
