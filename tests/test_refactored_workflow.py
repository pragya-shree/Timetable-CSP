"""Tests for refactored timetable workflow, synchronized state, and regression fixes.

Covers:
- View Schedule when no timetable exists
- View Schedule immediately after generation
- Regression test for Pandas Styler.applymap bug on Pandas 3.x
- Switching between saved timetables
- Saving multiple timetables without overwriting
- Loading and reconstructing saved model objects
- Corrupt and missing history storage handling
- Manual edits and conflict recalculation
- Reverting unsaved manual edits
- Exporting the currently selected timetable
- Dataset switching without stale schedules
- Benchmark isolation from personal timetable history
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import pandas as pd
import pytest

from core.models import load_dataset_from_file, TimetableDataset
from core.sessions import Placement, SessionVar, Timetable, generate_session_variables
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.backtracking import solve_backtracking
from core.storage import (
    TimetableRecord,
    TimetableStore,
    TimetableNotFoundError,
    CorruptStorageFileError,
)
import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def small_ds() -> TimetableDataset:
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")


@pytest.fixture
def large_ds() -> TimetableDataset:
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "large_dataset.json")


@pytest.fixture
def solved_small_tt(small_ds) -> Timetable:
    res = solve_backtracking(small_ds, use_mrv=True, use_degree=True, use_fc=True)
    assert res.success
    return res.timetable


@pytest.fixture
def solved_large_tt(large_ds) -> Timetable:
    res = solve_backtracking(large_ds, use_mrv=True, use_degree=True, use_fc=True)
    assert res.success
    return res.timetable


class TestViewScheduleWorkflowAndRegressions:
    def test_view_schedule_when_no_timetable_exists(self, small_ds):
        """Empty timetable should generate empty/labeled grids without crashing."""
        empty_tt = Timetable.empty(small_ds)
        grids_class = app._make_grid(empty_tt, "class")
        assert len(grids_class) == len(small_ds.classes)
        for df in grids_class.values():
            assert isinstance(df, pd.DataFrame)
            styled = app._style_df(df)
            assert styled is not None

        grids_teacher = app._make_grid(empty_tt, "teacher")
        assert len(grids_teacher) == len(small_ds.teachers)

        grids_room = app._make_grid(empty_tt, "room")
        assert len(grids_room) == len(small_ds.rooms)

    def test_regression_pandas_styler_applymap(self, solved_small_tt):
        """Regression test for Pandas 3.x AttributeError: 'Styler' object has no attribute 'applymap'.

        In Pandas 3.x, Styler.applymap was removed in favor of Styler.map.
        app._style_df must successfully style DataFrames on any Pandas version without throwing.
        """
        grids = app._make_grid(solved_small_tt, "class")
        first_df = list(grids.values())[0]

        # Calling _style_df must succeed without AttributeError
        styled = app._style_df(first_df)
        assert styled is not None
        # Verify rendered HTML or Styler object is valid
        rendered = styled.to_html()
        assert isinstance(rendered, str)
        assert "table" in rendered.lower()

    def test_view_schedule_immediately_after_generation(self, solved_small_tt, small_ds):
        """Verify all three grid views immediately after solver finishes."""
        for mode in ("class", "teacher", "room"):
            grids = app._make_grid(solved_small_tt, mode)
            assert len(grids) > 0
            for entity_id, df in grids.items():
                assert df.shape == (small_ds.time_structure.periods_per_day, len(small_ds.time_structure.days))
                styled = app._style_df(df)
                assert styled is not None


class TestTimetableHistoryAndStateSynchronization:
    def test_saving_multiple_timetables_without_overwriting(self, tmp_path, solved_small_tt, solved_large_tt):
        store = TimetableStore(tmp_path)

        rec1 = TimetableRecord.create_from_timetable(solved_small_tt, "Small Schedule v1", "MRV+FC")
        rec2 = TimetableRecord.create_from_timetable(solved_large_tt, "Large Schedule v1", "MRV+FC")

        store.save(rec1)
        store.save(rec2)

        records = store.list_all()
        assert len(records) == 2
        ids = {r.id for r in records}
        assert rec1.id in ids
        assert rec2.id in ids

    def test_switching_between_two_saved_timetables(self, tmp_path, solved_small_tt, solved_large_tt):
        """Switching active timetable must update dataset, schedule, and placed count cleanly."""
        store = TimetableStore(tmp_path)
        rec_small = store.save(TimetableRecord.create_from_timetable(solved_small_tt, "Small TT", "MRV"))
        rec_large = store.save(TimetableRecord.create_from_timetable(solved_large_tt, "Large TT", "MRV"))

        # 1. Activate Small
        loaded_small = store.get(rec_small.id)
        tt_small = loaded_small.get_timetable()
        app._set_active_timetable(tt_small, name=loaded_small.name, timetable_id=loaded_small.id)

        assert app.st.session_state["active_timetable_id"] == rec_small.id
        assert len(app.st.session_state["active_timetable"].get_placed_sessions()) == 39
        assert len(app.st.session_state["active_timetable"].dataset.classes) == 3

        # 2. Switch to Large
        loaded_large = store.get(rec_large.id)
        tt_large = loaded_large.get_timetable()
        app._set_active_timetable(tt_large, name=loaded_large.name, timetable_id=loaded_large.id)

        assert app.st.session_state["active_timetable_id"] == rec_large.id
        assert len(app.st.session_state["active_timetable"].get_placed_sessions()) == 104
        assert len(app.st.session_state["active_timetable"].dataset.classes) == 8
        # Grids must reflect large dataset entities
        grids = app._make_grid(app.st.session_state["active_timetable"], "class")
        assert len(grids) == 8

    def test_loading_and_reconstructing_saved_model_objects(self, tmp_path, solved_small_tt, small_ds):
        store = TimetableStore(tmp_path)
        rec = store.save(TimetableRecord.create_from_timetable(solved_small_tt, "Original", "Plain"))

        loaded_rec = store.get(rec.id)
        reconstructed_ds = loaded_rec.get_dataset()
        reconstructed_tt = loaded_rec.get_timetable()

        assert isinstance(reconstructed_ds, TimetableDataset)
        assert isinstance(reconstructed_tt, Timetable)
        assert reconstructed_tt.is_complete()
        val = validate_timetable(reconstructed_tt)
        assert val.is_valid
        assert val.hard_violation_count == 0

    def test_corrupt_or_missing_history_storage(self, tmp_path):
        store = TimetableStore(tmp_path)

        with pytest.raises(TimetableNotFoundError):
            store.get("missing_id_123")

        # Corrupt file
        (tmp_path / "corrupt.json").write_text("{ not json", encoding="utf-8")
        assert store.list_all() == []

        with pytest.raises(CorruptStorageFileError):
            store.get("corrupt")


class TestManualEditingAndReversion:
    def test_manual_edit_conflict_recalculation(self, solved_small_tt):
        """Moving a session to create a conflict recalculates hard violations and cost."""
        placed = solved_small_tt.get_placed_sessions()
        var1, plc1 = placed[0]
        var2, plc2 = placed[1]

        # Force var2 to clash with var1 in same room & slot
        clash_plc = Placement(day=plc1.day, start_period=plc1.start_period, room_id=plc1.room_id)

        edited_asgn = dict(solved_small_tt.assignments)
        edited_asgn[var2] = clash_plc
        edited_tt = Timetable(dataset=solved_small_tt.dataset, assignments=edited_asgn)

        val = validate_timetable(edited_tt)
        assert not val.is_valid
        assert val.hard_violation_count > 0

        cost = calculate_cost(edited_tt)
        assert cost.hard_violations > 0
        assert cost.total_cost >= 1000

    def test_reverting_unsaved_edits(self, tmp_path, solved_small_tt):
        """Reverting reloads the clean saved version from storage."""
        store = TimetableStore(tmp_path)
        rec = store.save(TimetableRecord.create_from_timetable(solved_small_tt, "SavedClean", "MRV"))

        # Mutate the in-memory timetable
        tt_copy = store.get(rec.id).get_timetable()
        var1 = list(tt_copy.assignments.keys())[0]
        # Assign invalid slot
        tt_copy.assignments[var1] = Placement(day=0, start_period=0, room_id="CR1")  # global blocked slot in small_ds

        val_edited = validate_timetable(tt_copy)
        assert not val_edited.is_valid

        # Revert by reloading from store
        reverted_tt = store.get(rec.id).get_timetable()
        val_reverted = validate_timetable(reverted_tt)
        assert val_reverted.is_valid
        assert val_reverted.hard_violation_count == 0


class TestExportsAndBenchmarkIsolation:
    def test_exporting_currently_selected_timetable(self, solved_small_tt, solved_large_tt):
        """Export must strictly export the active schedule's sessions without cross-contamination."""
        df_small = app._sessions_df(solved_small_tt, algorithm="MRV")
        assert len(df_small) == 39
        assert "algorithm" in df_small.columns
        assert (df_small["algorithm"] == "MRV").all()

        df_large = app._sessions_df(solved_large_tt, algorithm="Plain")
        assert len(df_large) == 104
        assert (df_large["algorithm"] == "Plain").all()

        # Excel export validation
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_small.to_excel(writer, index=False)
        buf.seek(0)
        loaded = pd.read_excel(buf)
        assert len(loaded) == 39

    def test_benchmark_isolation_from_personal_history(self, tmp_path, solved_small_tt, small_ds):
        """Running or viewing benchmarks must not modify saved personal timetables."""
        store = TimetableStore(tmp_path)
        rec = store.save(TimetableRecord.create_from_timetable(solved_small_tt, "Personal Schedule", "MRV"))

        # Solve benchmark independently
        bench_res = solve_backtracking(small_ds, use_mrv=False)
        assert bench_res.success

        # Ensure store is unchanged
        records = store.list_all()
        assert len(records) == 1
        assert records[0].id == rec.id
        assert records[0].name == "Personal Schedule"
