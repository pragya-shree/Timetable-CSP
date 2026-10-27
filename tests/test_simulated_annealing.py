"""Unit and integration tests for Hybrid CSP + Simulated Annealing solver."""

from pathlib import Path
import pytest

from core.models import load_dataset_from_file, TimetableDataset
from core.simulated_annealing import (
    HybridSimulatedAnnealingSolver,
    SimulatedAnnealingResult,
    solve_hybrid_simulated_annealing,
)
from core.constraints import validate_timetable


@pytest.fixture
def small_dataset() -> TimetableDataset:
    return load_dataset_from_file(Path("datasets/small_dataset.json"))


def test_hybrid_sa_solves_small_dataset(small_dataset):
    """Hybrid SA should produce a 100% valid schedule with 0 hard violations."""
    res = solve_hybrid_simulated_annealing(
        small_dataset,
        initial_temperature=50.0,
        cooling_rate=0.95,
        max_iterations=100,
        random_seed=42,
    )
    assert res.success is True
    assert res.status == "solved"
    assert res.hard_violations == 0
    assert res.timetable is not None
    val = validate_timetable(res.timetable)
    assert val.is_valid is True
    assert val.hard_violation_count == 0


def test_hybrid_sa_metrics_and_penalties(small_dataset):
    """Verifies that all result metrics and soft penalties (S1-S5) are populated."""
    res = solve_hybrid_simulated_annealing(
        small_dataset,
        initial_temperature=20.0,
        cooling_rate=0.9,
        max_iterations=50,
        random_seed=123,
    )
    assert res.initial_cost >= res.final_cost
    assert res.total_cost == res.final_cost
    assert res.moves_accepted >= 0
    assert res.iterations > 0
    assert "s1_gaps" in res.soft_penalties
    assert "s2_repeats" in res.soft_penalties
    assert "s3_non_consecutive" in res.soft_penalties
    assert "s4_teacher_preferences" in res.soft_penalties
    assert "s5_teacher_consecutive" in res.soft_penalties

    d = res.to_dict()
    assert d["algorithm"] == "hybrid_simulated_annealing"
    assert d["success"] is True
    assert d["hard_violations"] == 0


def test_hybrid_sa_seed_reproducibility(small_dataset):
    """Fixed random seed should produce deterministic algorithmic outcome."""
    res1 = solve_hybrid_simulated_annealing(
        small_dataset,
        initial_temperature=30.0,
        cooling_rate=0.95,
        max_iterations=60,
        random_seed=999,
    )
    res2 = solve_hybrid_simulated_annealing(
        small_dataset,
        initial_temperature=30.0,
        cooling_rate=0.95,
        max_iterations=60,
        random_seed=999,
    )
    assert res1.final_cost == res2.final_cost
    assert res1.iterations == res2.iterations
    assert res1.moves_accepted == res2.moves_accepted


def test_hybrid_sa_solves_medium_tight_dataset():
    """Hybrid SA should solve medium_tight_dataset.json and maintain 0 hard violations."""
    dataset = load_dataset_from_file(Path("datasets/medium_tight_dataset.json"))
    res = solve_hybrid_simulated_annealing(
        dataset,
        initial_temperature=40.0,
        cooling_rate=0.95,
        max_iterations=100,
        random_seed=42,
    )
    assert res.success is True
    assert res.status == "solved"
    assert res.hard_violations == 0
    val = validate_timetable(res.timetable)
    assert val.is_valid is True
    assert val.hard_violation_count == 0


def test_hybrid_sa_zero_iterations(small_dataset):
    """When max_iterations=0, hybrid SA returns the initial solution with 0 hard violations."""
    res = solve_hybrid_simulated_annealing(
        small_dataset,
        max_iterations=0,
        random_seed=42,
    )
    assert res.success is True
    assert res.iterations == 0
    assert res.final_cost == res.initial_cost
    assert res.hard_violations == 0


def test_hybrid_sa_respects_time_budget(small_dataset):
    """Hybrid SA stops within specified time budget."""
    res = solve_hybrid_simulated_annealing(
        small_dataset,
        max_iterations=1000000,
        time_limit_seconds=0.15,
        random_seed=42,
    )
    assert res.success is True
    assert res.runtime_seconds >= 0.0
    assert res.iterations < 1000000
    assert res.hard_violations == 0
