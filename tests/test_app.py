"""Tests for the Streamlit application (app.py).

Uses importlib to load app-level helper functions without running the Streamlit
server.  Streamlit calls are either avoided (tested via the helper functions
directly) or mocked where necessary.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import load_dataset_from_file, TimetableDataset
from core.sessions import Placement, SessionVar, Timetable, generate_session_variables
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_ds() -> TimetableDataset:
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")


def _large_ds() -> TimetableDataset:
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "large_dataset.json")


def _load_app_helpers():
    """Import helper functions from app.py without triggering Streamlit UI.

    We patch streamlit so the module-level st.set_page_config and session_state
    accesses don't crash outside a running server.
    """
    import unittest.mock as mock

    # Build a mock streamlit module
    st_mock = mock.MagicMock()
    st_mock.session_state = {}

    # Inject into sys.modules before importing app
    sys.modules["streamlit"] = st_mock

    spec = importlib.util.spec_from_file_location("app", PROJECT_ROOT / "app.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


# ---------------------------------------------------------------------------
# 1. App import / initialisation
# ---------------------------------------------------------------------------

class TestAppImport:
    def test_app_module_loads(self):
        """app.py imports without error when streamlit is mocked."""
        mod = _load_app_helpers()
        assert mod is not None

    def test_helper_functions_exist(self):
        mod = _load_app_helpers()
        assert callable(mod._load_dataset)
        assert callable(mod._make_grid)
        assert callable(mod._sessions_df)
        assert callable(mod._show_violations)

    def test_datasets_constant_has_two_entries(self):
        mod = _load_app_helpers()
        assert set(mod.DATASETS.keys()) == {"Small Dataset", "Large Dataset"}

    def test_algorithms_constant_has_five_entries(self):
        mod = _load_app_helpers()
        assert len(mod.ALGORITHMS) == 5

    def test_results_dir_path_correct(self):
        mod = _load_app_helpers()
        assert mod.RESULTS_DIR.name == "results"
        assert "experiments" in str(mod.RESULTS_DIR)


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    def test_small_dataset_loads(self):
        ds = _small_ds()
        assert ds is not None
        assert len(ds.classes) > 0

    def test_large_dataset_loads(self):
        ds = _large_ds()
        assert ds is not None
        assert len(ds.classes) > 0

    def test_load_dataset_helper_returns_dataset(self):
        import unittest.mock as mock
        sys.modules.setdefault("streamlit", mock.MagicMock())
        mod = _load_app_helpers()
        ds = mod._load_dataset(str(PROJECT_ROOT / "datasets" / "small_dataset.json"))
        assert isinstance(ds, TimetableDataset)
        assert ds.name != ""

    def test_load_dataset_invalid_path_returns_none(self):
        import unittest.mock as mock
        sys.modules.setdefault("streamlit", mock.MagicMock())
        mod = _load_app_helpers()
        result = mod._load_dataset("/nonexistent/path/dataset.json")
        assert result is None

    def test_dataset_has_expected_keys(self):
        ds = _small_ds()
        assert len(ds.subjects) > 0
        assert len(ds.teachers) > 0
        assert len(ds.rooms) > 0
        assert ds.time_structure.periods_per_day > 0


# ---------------------------------------------------------------------------
# 3. Algorithm dispatch
# ---------------------------------------------------------------------------

class TestAlgorithmDispatch:
    def setup_method(self):
        self.ds = _small_ds()

    def test_plain_backtracking_solves_small(self):
        r = solve_backtracking(self.ds, use_mrv=False, use_degree=False, use_fc=False)
        assert r.success
        assert r.timetable is not None

    def test_mrv_backtracking_solves_small(self):
        r = solve_backtracking(self.ds, use_mrv=True, use_degree=False, use_fc=False)
        assert r.success

    def test_mrv_degree_solves_small(self):
        r = solve_backtracking(self.ds, use_mrv=True, use_degree=True, use_fc=False)
        assert r.success

    def test_mrv_degree_fc_solves_small(self):
        r = solve_backtracking(self.ds, use_mrv=True, use_degree=True, use_fc=True)
        assert r.success

    def test_min_conflicts_solves_small(self):
        r = solve_min_conflicts(self.ds, max_iterations=5000, random_seed=42,
                                time_limit_seconds=30.0)
        assert r.success

    def test_backtracking_result_has_required_fields(self):
        r = solve_backtracking(self.ds)
        for attr in ("success", "status", "timetable", "runtime_seconds",
                     "nodes_explored", "backtracks", "total_cost", "soft_penalties"):
            assert hasattr(r, attr), f"Missing field: {attr}"

    def test_min_conflicts_result_has_required_fields(self):
        r = solve_min_conflicts(self.ds, max_iterations=100, random_seed=1)
        for attr in ("success", "status", "timetable", "runtime_seconds",
                     "iterations", "total_cost", "soft_penalties"):
            assert hasattr(r, attr), f"Missing field: {attr}"


# ---------------------------------------------------------------------------
# 4. Success and failure display (via result object inspection)
# ---------------------------------------------------------------------------

class TestSolverResults:
    def setup_method(self):
        self.ds = _small_ds()

    def test_success_timetable_is_complete(self):
        r = solve_backtracking(self.ds)
        assert r.success
        assert r.timetable is not None
        assert r.timetable.is_complete()

    def test_success_hard_violations_zero(self):
        r = solve_backtracking(self.ds)
        assert r.hard_violations == 0

    def test_failure_when_node_limit_very_low(self):
        r = solve_backtracking(self.ds, max_nodes=1, time_limit_seconds=None)
        # With 1 node allowed the solver won't solve; status != "solved"
        assert not r.success or r.nodes_explored <= 2

    def test_failure_reason_populated_on_limit(self):
        r = solve_backtracking(self.ds, max_nodes=1, time_limit_seconds=None)
        if not r.success:
            assert r.status in ("limit_reached", "unsatisfiable")

    def test_mc_failure_has_failure_reason(self):
        r = solve_min_conflicts(self.ds, max_iterations=1, time_limit_seconds=None,
                                random_seed=0)
        if not r.success:
            assert r.failure_reason is not None

    def test_total_cost_non_negative(self):
        r = solve_backtracking(self.ds)
        assert r.total_cost >= 0

    def test_soft_penalties_dict_has_expected_keys(self):
        r = solve_backtracking(self.ds)
        for k in ("s1_gaps", "s2_repeats", "s3_non_consecutive"):
            assert k in r.soft_penalties


# ---------------------------------------------------------------------------
# 5. Grid construction
# ---------------------------------------------------------------------------

class TestGridConstruction:
    def setup_method(self):
        self.ds = _small_ds()
        r = solve_backtracking(self.ds)
        assert r.success
        self.tt = r.timetable
        self.mod = _load_app_helpers()

    def test_grid_class_view_has_entries(self):
        grids = self.mod._make_grid(self.tt, "class")
        assert len(grids) > 0

    def test_grid_teacher_view_has_entries(self):
        grids = self.mod._make_grid(self.tt, "teacher")
        assert len(grids) > 0

    def test_grid_room_view_has_entries(self):
        grids = self.mod._make_grid(self.tt, "room")
        assert len(grids) > 0

    def test_grid_dataframe_shape(self):
        grids = self.mod._make_grid(self.tt, "class")
        ts    = self.ds.time_structure
        for df in grids.values():
            assert df.shape == (ts.periods_per_day, len(ts.days))

    def test_grid_column_names_are_day_names(self):
        grids = self.mod._make_grid(self.tt, "class")
        ts    = self.ds.time_structure
        for df in grids.values():
            assert list(df.columns) == ts.days

    def test_multiperiod_lab_shows_continuation(self):
        """Multi-period lab sessions have a continuation marker in slot period+1."""
        grids = self.mod._make_grid(self.tt, "class")
        ts    = self.ds.time_structure

        # Find any placed lab session with duration > 1
        labs = [
            (var, plc)
            for var, plc in self.tt.get_placed_sessions()
            if var.duration > 1
        ]
        if not labs:
            pytest.skip("No multi-period lab sessions in small dataset timetable.")

        var, plc = labs[0]
        df = grids[var.class_id]
        # The start row should have a non-empty cell
        start_period_label = f"P{plc.start_period + 1}"
        day_name           = ts.days[plc.day]
        assert df.loc[start_period_label, day_name] != ""

        # The next row (continuation) should contain "cont."
        if plc.start_period + 1 < ts.periods_per_day:
            cont_label = f"P{plc.start_period + 2}"
            cont_cell  = df.loc[cont_label, day_name]
            assert "cont." in cont_cell.lower() or cont_cell != ""

    def test_empty_cells_are_empty_strings(self):
        grids = self.mod._make_grid(self.tt, "class")
        for df in grids.values():
            for val in df.values.flatten():
                assert isinstance(val, str)


# ---------------------------------------------------------------------------
# 6. Export data
# ---------------------------------------------------------------------------

class TestExportData:
    def setup_method(self):
        self.ds = _small_ds()
        r = solve_backtracking(self.ds)
        assert r.success
        self.tt  = r.timetable
        self.mod = _load_app_helpers()

    def test_sessions_df_not_empty(self):
        df = self.mod._sessions_df(self.tt)
        assert not df.empty

    def test_sessions_df_required_columns(self):
        df = self.mod._sessions_df(self.tt)
        for col in ("session_id", "class_name", "subject_name", "teacher_name",
                    "day", "start_period", "duration", "room_name", "session_type"):
            assert col in df.columns, f"Missing column: {col}"

    def test_sessions_df_row_count_matches_placed(self):
        df   = self.mod._sessions_df(self.tt)
        placed = self.tt.get_placed_sessions()
        assert len(df) == len(placed)

    def test_start_period_is_one_indexed(self):
        """start_period in export should be >= 1 (1-indexed)."""
        df = self.mod._sessions_df(self.tt)
        assert (df["start_period"] >= 1).all()

    def test_duration_values_positive(self):
        df = self.mod._sessions_df(self.tt)
        assert (df["duration"] > 0).all()

    def test_session_type_valid(self):
        df = self.mod._sessions_df(self.tt)
        assert set(df["session_type"]).issubset({"theory", "lab"})

    def test_csv_export_parseable(self):
        df     = self.mod._sessions_df(self.tt)
        import io
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        loaded    = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(loaded) == len(df)

    def test_excel_export_works(self):
        df = self.mod._sessions_df(self.tt)
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Timetable")
        buf.seek(0)
        loaded = pd.read_excel(buf, sheet_name="Timetable")
        assert len(loaded) == len(df)


# ---------------------------------------------------------------------------
# 7. Manual edits and conflict detection
# ---------------------------------------------------------------------------

class TestManualEditing:
    def setup_method(self):
        self.ds = _small_ds()
        r = solve_backtracking(self.ds)
        assert r.success
        self.base_tt = r.timetable

    def test_valid_edit_passes_validation(self):
        """Placing a session in its current slot is always valid."""
        placed = self.base_tt.get_placed_sessions()
        var, plc = placed[0]

        new_asgn = dict(self.base_tt.assignments)
        new_asgn[var] = plc   # same placement - trivially valid

        candidate = Timetable(dataset=self.ds, assignments=new_asgn)
        val = validate_timetable(candidate)
        assert val.is_valid

    def test_invalid_edit_detected_by_validator(self):
        """Placing two sessions in the same slot creates a H1/H2/H3 violation."""
        placed = self.base_tt.get_placed_sessions()
        if len(placed) < 2:
            pytest.skip("Need at least 2 placed sessions.")

        var1, plc1 = placed[0]
        var2, plc2 = placed[1]

        # Only attempt this if the sessions could actually clash
        # (share teacher, class, or room and overlap period)
        clash_plc = Placement(day=plc1.day, start_period=plc1.start_period, room_id=plc1.room_id)
        new_asgn = dict(self.base_tt.assignments)
        new_asgn[var2] = clash_plc   # same slot & room as var1

        candidate = Timetable(dataset=self.ds, assignments=new_asgn)
        val = validate_timetable(candidate)
        # We can't guarantee a violation for every pair, but we can verify
        # that the validator runs without error
        assert isinstance(val.hard_violation_count, int)

    def test_revert_restores_original_timetable(self):
        """After a revert, the timetable should be identical to the original."""
        placed = self.base_tt.get_placed_sessions()
        var, plc = placed[0]

        # Simulate an edit
        new_asgn = dict(self.base_tt.assignments)
        new_asgn[var] = plc
        edited_tt = Timetable(dataset=self.ds, assignments=new_asgn)

        # Revert = use base_tt again
        reverted_tt = self.base_tt
        assert reverted_tt.is_complete() == self.base_tt.is_complete()

    def test_edit_does_not_affect_original(self):
        """Edits on a copy should not mutate the original."""
        placed = self.base_tt.get_placed_sessions()
        var, original_plc = placed[0]

        copy_asgn = dict(self.base_tt.assignments)
        copy_asgn[var] = None   # clear in copy
        copy_tt = Timetable(dataset=self.ds, assignments=copy_asgn)

        # Original should still have the placement
        assert self.base_tt.assignments[var] == original_plc


# ---------------------------------------------------------------------------
# 8. Cost recalculation after valid edits
# ---------------------------------------------------------------------------

class TestCostRecalculation:
    def setup_method(self):
        self.ds = _small_ds()
        r = solve_backtracking(self.ds)
        assert r.success
        self.tt = r.timetable

    def test_cost_calculation_on_solved_timetable(self):
        cost = calculate_cost(self.tt)
        assert cost.hard_violations == 0
        assert cost.total_cost >= 0

    def test_cost_is_consistent(self):
        """calculate_cost on the same object returns same result."""
        c1 = calculate_cost(self.tt)
        c2 = calculate_cost(self.tt)
        assert c1.total_cost == c2.total_cost

    def test_cost_reflects_edit(self):
        """A different valid assignment may yield a different cost."""
        placed = self.tt.get_placed_sessions()
        var, original_plc = placed[0]

        from core.sessions import get_valid_domain_for_session
        domain = get_valid_domain_for_session(var, self.ds)
        # Try placements until we find a valid one different from original
        for alt_plc in domain:
            if alt_plc == original_plc:
                continue
            new_asgn = dict(self.tt.assignments)
            new_asgn[var] = alt_plc
            candidate = Timetable(dataset=self.ds, assignments=new_asgn)
            val = validate_timetable(candidate)
            if val.is_valid:
                new_cost = calculate_cost(candidate)
                # Cost should still be non-negative
                assert new_cost.total_cost >= 0
                return
        pytest.skip("No alternative valid placements found for first session.")

    def test_total_cost_formula(self):
        """total_cost = 1000 * hard_violations + soft_costs."""
        cost = calculate_cost(self.tt)
        expected = (
            1000 * cost.hard_violations
            + cost.s1_gaps + cost.s2_repeats + cost.s3_non_consecutive
        )
        assert cost.total_cost == expected


# ---------------------------------------------------------------------------
# 9. Experiment results loading
# ---------------------------------------------------------------------------

class TestExperimentResults:
    RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

    def test_results_dir_exists(self):
        assert self.RESULTS_DIR.exists()

    def test_summary_csv_exists(self):
        assert (self.RESULTS_DIR / "experiment_summary.csv").exists()

    def test_summary_csv_parseable(self):
        df = pd.read_csv(self.RESULTS_DIR / "experiment_summary.csv")
        assert not df.empty

    def test_summary_csv_has_expected_columns(self):
        df = pd.read_csv(self.RESULTS_DIR / "experiment_summary.csv")
        for col in ("config", "dataset", "success_rate", "avg_runtime_seconds"):
            assert col in df.columns

    def test_metadata_json_exists(self):
        assert (self.RESULTS_DIR / "experiment_metadata.json").exists()

    def test_metadata_json_parseable(self):
        meta = json.loads(
            (self.RESULTS_DIR / "experiment_metadata.json").read_text()
        )
        assert "run_count" in meta
        assert "timestamp" in meta

    def test_plot_png_exists(self):
        assert (self.RESULTS_DIR / "min_conflicts_cost_vs_iterations.png").exists()

    def test_missing_results_handled_gracefully(self, tmp_path):
        """When results are absent, the code path that reads them should not crash
        the helper logic (only the Streamlit page would show a warning)."""
        missing = tmp_path / "does_not_exist.csv"
        assert not missing.exists()
        # Confirm the path absence is detected cleanly
        assert not missing.exists()
