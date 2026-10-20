"""Tests verifying complete removal of demo mode and ensuring custom timetable creation is primary.

Covers:
1. Complete removal of demo UI controls, banners, and default demo state.
2. Fresh application state has no active timetable and no fake statistics.
3. Custom timetable creation workflow (configure -> validate -> solve -> save).
4. Intentional empty states across all views (Dashboard, View & Edit, Export).
5. Benchmark dataset preservation and isolation from user timetable history.
6. User saved data protection and integrity in saved_timetables/.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from core.models import load_dataset_from_file, TimetableDataset
from core.sessions import Timetable
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.custom_builder import (
    create_default_custom_data,
    validate_and_build_custom_dataset,
)
from core.storage import TimetableRecord, TimetableStore
import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDemoModeRemovalVerification:
    def test_app_source_has_no_demo_selectors_or_buttons(self):
        """Source code of app.py must not contain demo selectors or demo loaders."""
        app_code = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        # Verify no demo source radio
        assert "📁 Predefined Benchmark Dataset" not in app_code
        assert "Clone Small Dataset" not in app_code
        assert "Quick Solve Small Dataset" not in app_code
        assert "Quick Solve Large Dataset" not in app_code
        assert '"dataset_source": "Demo"' not in app_code

    def test_default_session_state_is_clean(self):
        """_DEFAULTS must have no active timetable and dataset_source should not be Demo."""
        defaults = app._DEFAULTS
        assert defaults["active_timetable"] is None
        assert defaults["active_timetable_id"] is None
        assert defaults["active_generation_result"] is None
        assert defaults.get("dataset_source") != "Demo"

    def test_readme_has_no_demo_mode_claims(self):
        """README.md should reference Standardized Academic Benchmarks instead of Demo."""
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "Predefined Demo Benchmarks" not in readme
        assert "Standardized Academic Benchmarks" in readme


class TestCustomCreationPrimaryWorkflow:
    def test_custom_data_builds_and_generates_timetable(self):
        """The custom timetable builder constructs a valid dataset that solves cleanly."""
        data = create_default_custom_data()
        val = validate_and_build_custom_dataset(
            config=data["config"],
            classes_data=data["classes"],
            teachers_data=data["teachers"],
            subjects_data=data["subjects"],
            rooms_data=data["rooms"],
        )
        assert val.is_valid
        assert val.dataset is not None

        # Solve using Backtracking
        res_bt = solve_backtracking(val.dataset, use_mrv=True, use_fc=True)
        assert res_bt.success
        assert res_bt.timetable is not None
        check_bt = validate_timetable(res_bt.timetable)
        assert check_bt.is_valid

        # Solve using Min-Conflicts
        res_mc = solve_min_conflicts(val.dataset, max_iterations=2000, random_seed=42)
        assert res_mc.success
        assert res_mc.timetable is not None
        check_mc = validate_timetable(res_mc.timetable)
        assert check_mc.is_valid

    def test_saving_custom_timetable_to_store(self, tmp_path):
        """Custom generated timetable can be saved to persistent store with metrics."""
        data = create_default_custom_data()
        val = validate_and_build_custom_dataset(
            config=data["config"],
            classes_data=data["classes"],
            teachers_data=data["teachers"],
            subjects_data=data["subjects"],
            rooms_data=data["rooms"],
        )
        res = solve_backtracking(val.dataset, use_mrv=True, use_degree=True, use_fc=True)
        assert res.success

        store = TimetableStore(tmp_path)
        rec = TimetableRecord.create_from_timetable(
            timetable=res.timetable,
            name="My Engineering Department",
            algorithm="MRV+Degree+FC",
            runtime_seconds=res.runtime_seconds,
        )
        store.save(rec)

        saved = store.list_all()
        assert len(saved) == 1
        assert saved[0].name == "My Engineering Department"
        reloaded = store.get(rec.id).get_timetable()
        assert reloaded.is_complete()
        assert validate_timetable(reloaded).is_valid


class TestEmptyStateIntegrity:
    def test_grids_handle_none_timetable_cleanly(self):
        """When active timetable is None, make_grid returns empty dict without crashing."""
        grids = app._make_grid(None, "class")
        assert grids == {}

    def test_sessions_df_handles_none_timetable_cleanly(self):
        """When active timetable is None, sessions_df returns empty DataFrame."""
        df = app._sessions_df(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestBenchmarkDatasetsAndUserPreservation:
    def test_benchmark_datasets_remain_intact(self):
        """datasets/small_dataset.json and large_dataset.json must exist and be valid."""
        small_p = PROJECT_ROOT / "datasets" / "small_dataset.json"
        large_p = PROJECT_ROOT / "datasets" / "large_dataset.json"

        assert small_p.exists()
        assert large_p.exists()

        small_ds = load_dataset_from_file(small_p)
        large_ds = load_dataset_from_file(large_p)

        assert len(small_ds.classes) == 3
        assert len(large_ds.classes) == 8

    def test_benchmark_results_and_plots_preserved(self):
        """Experiment results directory and files must remain intact."""
        res_dir = PROJECT_ROOT / "experiments" / "results"
        assert (res_dir / "experiment_summary.csv").exists()
        assert (res_dir / "experiment_runs.csv").exists()
        assert (res_dir / "min_conflicts_cost_vs_iterations.png").exists()

    def test_user_saved_timetables_preserved(self):
        """All user timetables in saved_timetables/ must remain intact and parseable."""
        saved_dir = PROJECT_ROOT / "saved_timetables"
        store = TimetableStore(saved_dir)
        records = store.list_all()
        assert len(records) >= 2  # The 2 pre-existing user records

        for r in records:
            tt = r.get_timetable()
            assert isinstance(tt, Timetable)
            assert tt.is_complete()
            val = validate_timetable(tt)
            assert val.is_valid
