"""Tests for the timetable grid UI rendering improvements in app.py.

Covers:
- _make_grid for class, teacher, and room views (including blocked period labeling).
- _style_df: Pandas 3.x compatibility, high-contrast colors, 13px font.
- _build_timetable_html: HTML structure, all cell types, legend, scrollable container.
- Long subject/teacher/room names (no crash, XSS-safe HTML escaping).
- Multi-period lab continuation markers in HTML output.
- _render_timetable_grid: callable without crashing when st is mocked.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import load_dataset_from_file, TimetableDataset
from core.sessions import Timetable
from core.backtracking import solve_backtracking


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def _small_ds() -> TimetableDataset:
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")


def _load_app_helpers():
    """Import app.py helper functions without triggering Streamlit UI."""
    import unittest.mock as mock
    st_mock = mock.MagicMock()
    st_mock.session_state = {}
    sys.modules["streamlit"] = st_mock
    spec = importlib.util.spec_from_file_location("app_gridui", PROJECT_ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve_small() -> Timetable:
    ds = _small_ds()
    result = solve_backtracking(ds, time_limit_seconds=60.0)
    return result.timetable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mod():
    return _load_app_helpers()


@pytest.fixture(scope="module")
def solved_tt():
    return _solve_small()


@pytest.fixture(scope="module")
def small_ds():
    return _small_ds()


@pytest.fixture(scope="module")
def empty_tt(small_ds):
    return Timetable.empty(small_ds)


# ---------------------------------------------------------------------------
# 1. _style_df
# ---------------------------------------------------------------------------

class TestStyleDf:
    def test_returns_styler(self, mod, solved_tt):
        grids = mod._make_grid(solved_tt, "class")
        df = next(iter(grids.values()))
        assert mod._style_df(df) is not None

    def test_no_applymap_error(self, mod, solved_tt):
        """Pandas 3.x regression: must use map not applymap."""
        grids = mod._make_grid(solved_tt, "class")
        df = next(iter(grids.values()))
        rendered = mod._style_df(df).to_html()
        assert "table" in rendered.lower()

    def test_empty_cell_color(self, mod):
        df = pd.DataFrame({"Mon": [""]}, index=["P1"])
        rendered = mod._style_df(df).to_html()
        assert "f8fafc" in rendered

    def test_lunch_color(self, mod):
        df = pd.DataFrame({"Mon": ["[Lunch Break]"]}, index=["P1"])
        rendered = mod._style_df(df).to_html()
        assert "fef3c7" in rendered

    def test_blocked_color(self, mod):
        df = pd.DataFrame({"Mon": ["[Blocked Period]"]}, index=["P1"])
        rendered = mod._style_df(df).to_html()
        assert "f1f5f9" in rendered

    def test_session_color(self, mod):
        df = pd.DataFrame({"Mon": ["Maths\nDr. Jones | Room 101"]}, index=["P1"])
        rendered = mod._style_df(df).to_html()
        assert "f0f7ff" in rendered

    def test_continuation_color(self, mod):
        df = pd.DataFrame({"Mon": ["... Physics Lab (cont.)"]}, index=["P1"])
        rendered = mod._style_df(df).to_html()
        assert "f0fdf4" in rendered

    def test_font_size_13px(self, mod):
        df = pd.DataFrame({"Mon": ["[Lunch Break]", "Physics\nDr. X | R2"]}, index=["P1", "P2"])
        rendered = mod._style_df(df).to_html()
        assert "13px" in rendered

    def test_empty_df_does_not_crash(self, mod):
        df = pd.DataFrame({"Mon": ["", "", ""]}, index=["P1", "P2", "P3"])
        assert mod._style_df(df) is not None


# ---------------------------------------------------------------------------
# 2. _make_grid
# ---------------------------------------------------------------------------

class TestMakeGrid:
    def test_class_view_has_all_classes(self, mod, solved_tt, small_ds):
        grids = mod._make_grid(solved_tt, "class")
        for cid in small_ds.classes:
            assert cid in grids

    def test_teacher_view_has_all_teachers(self, mod, solved_tt, small_ds):
        grids = mod._make_grid(solved_tt, "teacher")
        for tid in small_ds.teachers:
            assert tid in grids

    def test_room_view_has_all_rooms(self, mod, solved_tt, small_ds):
        grids = mod._make_grid(solved_tt, "room")
        for rid in small_ds.rooms:
            assert rid in grids

    def test_grid_shape_matches_time_structure(self, mod, solved_tt, small_ds):
        ts = small_ds.time_structure
        for mode in ("class", "teacher", "room"):
            for df in mod._make_grid(solved_tt, mode).values():
                assert df.shape == (ts.periods_per_day, len(ts.days))

    def test_grid_index_is_period_labels(self, mod, solved_tt, small_ds):
        ts = small_ds.time_structure
        expected = [f"P{p + 1}" for p in range(ts.periods_per_day)]
        for df in mod._make_grid(solved_tt, "class").values():
            assert list(df.index) == expected

    def test_grid_columns_are_day_names(self, mod, solved_tt, small_ds):
        ts = small_ds.time_structure
        for df in mod._make_grid(solved_tt, "class").values():
            assert list(df.columns) == ts.days

    def test_class_grid_has_lunch_label(self, mod, empty_tt, small_ds):
        grids = mod._make_grid(empty_tt, "class")
        ts = small_ds.time_structure
        for cid, cg in small_ds.classes.items():
            if 0 <= cg.lunch_slot < ts.periods_per_day:
                df = grids[cid]
                period_label = f"P{cg.lunch_slot + 1}"
                for day in ts.days:
                    assert "[Lunch Break]" in df.loc[period_label, day]

    def test_class_grid_has_blocked_label(self, mod, empty_tt, small_ds):
        ts = small_ds.time_structure
        if not ts.blocked_periods:
            pytest.skip("No globally blocked periods in small dataset.")
        grids = mod._make_grid(empty_tt, "class")
        for cid in small_ds.classes:
            df = grids[cid]
            for d, p in ts.blocked_periods:
                if 0 <= d < len(ts.days) and 0 <= p < ts.periods_per_day:
                    assert "[Blocked Period]" in df.loc[f"P{p + 1}", ts.days[d]]

    def test_teacher_grid_has_blocked_label(self, mod, empty_tt, small_ds):
        ts = small_ds.time_structure
        if not ts.blocked_periods:
            pytest.skip("No globally blocked periods in small dataset.")
        grids = mod._make_grid(empty_tt, "teacher")
        for tid in small_ds.teachers:
            df = grids[tid]
            for d, p in ts.blocked_periods:
                if 0 <= d < len(ts.days) and 0 <= p < ts.periods_per_day:
                    assert "[Blocked Period]" in df.loc[f"P{p + 1}", ts.days[d]]

    def test_room_grid_has_blocked_label(self, mod, empty_tt, small_ds):
        ts = small_ds.time_structure
        if not ts.blocked_periods:
            pytest.skip("No globally blocked periods in small dataset.")
        grids = mod._make_grid(empty_tt, "room")
        for rid in small_ds.rooms:
            df = grids[rid]
            for d, p in ts.blocked_periods:
                if 0 <= d < len(ts.days) and 0 <= p < ts.periods_per_day:
                    assert "[Blocked Period]" in df.loc[f"P{p + 1}", ts.days[d]]

    def test_all_cells_are_strings(self, mod, solved_tt):
        for mode in ("class", "teacher", "room"):
            for df in mod._make_grid(solved_tt, mode).values():
                for val in df.values.flatten():
                    assert isinstance(val, str)

    def test_multiperiod_lab_continuation_marker(self, mod, solved_tt, small_ds):
        labs = [(var, plc) for var, plc in solved_tt.get_placed_sessions() if var.duration > 1]
        if not labs:
            pytest.skip("No multi-period lab sessions in solved small dataset.")
        grids = mod._make_grid(solved_tt, "class")
        ts = small_ds.time_structure
        for var, plc in labs:
            df = grids[var.class_id]
            start_label = f"P{plc.start_period + 1}"
            day_name = ts.days[plc.day]
            start_cell = df.loc[start_label, day_name]
            assert start_cell != ""
            assert "(cont.)" not in start_cell.lower()
            for offset in range(1, var.duration):
                cont_p = plc.start_period + offset
                if cont_p < ts.periods_per_day:
                    cont_cell = df.loc[f"P{cont_p + 1}", day_name]
                    assert "cont." in cont_cell.lower()

    def test_occupied_class_cells_contain_subject_name(self, mod, solved_tt, small_ds):
        grids = mod._make_grid(solved_tt, "class")
        ds = solved_tt.dataset
        ts = small_ds.time_structure
        for var, plc in solved_tt.get_placed_sessions():
            subj = ds.subjects.get(var.subject_id)
            subj_name = subj.name if subj else var.subject_id
            cell = grids[var.class_id].loc[f"P{plc.start_period + 1}", ts.days[plc.day]]
            assert subj_name in cell


# ---------------------------------------------------------------------------
# 3. _build_timetable_html
# ---------------------------------------------------------------------------

class TestBuildTimetableHtml:
    @staticmethod
    def _sample_df() -> pd.DataFrame:
        return pd.DataFrame({
            "Monday": [
                "Intro to CS\nDr. Turing | Lab 101",
                "... Intro to CS (cont.)",
                "[Lunch Break]",
                "",
            ],
            "Tuesday": [
                "[Blocked Period]",
                "Advanced Maths\nProf. Euler | Room 201",
                "",
                "Data Structures\nDr. Knuth | Lab",
            ],
        }, index=["P1", "P2", "P3", "P4"])

    def test_returns_string(self, mod):
        assert isinstance(mod._build_timetable_html(self._sample_df(), "class"), str)

    def test_contains_table(self, mod):
        result = mod._build_timetable_html(self._sample_df(), "class")
        assert "<table" in result.lower()

    def test_day_headers_present(self, mod):
        result = mod._build_timetable_html(self._sample_df(), "class")
        assert "Monday" in result and "Tuesday" in result

    def test_period_labels_present(self, mod):
        result = mod._build_timetable_html(self._sample_df(), "class")
        assert "P1" in result and "P4" in result

    def test_lunch_css_class(self, mod):
        assert "tt-card-lunch" in mod._build_timetable_html(self._sample_df(), "class")

    def test_blocked_css_class(self, mod):
        assert "tt-card-blocked" in mod._build_timetable_html(self._sample_df(), "class")

    def test_session_css_class(self, mod):
        assert "tt-card-session" in mod._build_timetable_html(self._sample_df(), "class")

    def test_continuation_css_class(self, mod):
        assert "tt-card-cont" in mod._build_timetable_html(self._sample_df(), "class")

    def test_empty_css_class(self, mod):
        assert "tt-card-empty" in mod._build_timetable_html(self._sample_df(), "class")

    def test_legend_bar(self, mod):
        assert "tt-legend-bar" in mod._build_timetable_html(self._sample_df(), "class")

    def test_scroll_container(self, mod):
        assert "tt-scroll-container" in mod._build_timetable_html(self._sample_df(), "class")

    def test_subject_names_present(self, mod):
        result = mod._build_timetable_html(self._sample_df(), "class")
        assert "Intro to CS" in result and "Advanced Maths" in result

    def test_teacher_and_room_names_present(self, mod):
        result = mod._build_timetable_html(self._sample_df(), "class")
        assert "Dr. Turing" in result and "Lab 101" in result

    def test_class_view_labels(self, mod):
        df = pd.DataFrame({"Mon": ["Calculus\nDr. Newton | Room A"]}, index=["P1"])
        result = mod._build_timetable_html(df, "class")
        assert "Teacher:" in result and "Room:" in result

    def test_teacher_view_labels(self, mod):
        df = pd.DataFrame({"Mon": ["Calculus\nClass A | Room B"]}, index=["P1"])
        result = mod._build_timetable_html(df, "teacher")
        assert "Class:" in result and "Room:" in result

    def test_room_view_labels(self, mod):
        df = pd.DataFrame({"Mon": ["Physics\nClass B | Prof. X"]}, index=["P1"])
        result = mod._build_timetable_html(df, "room")
        assert "Class:" in result and "Teacher:" in result

    def test_css_block_present(self, mod):
        assert "<style>" in mod._build_timetable_html(self._sample_df(), "class")

    def test_subj_title_class_present(self, mod):
        df = pd.DataFrame({"Mon": ["Physics\nDr. X | R1"]}, index=["P1"])
        assert "tt-subj-title" in mod._build_timetable_html(df, "class")

    def test_min_width_scales_with_days(self, mod):
        df_wide = pd.DataFrame({d: [""] for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]}, index=["P1"])
        df_narrow = pd.DataFrame({"Mon": [""]}, index=["P1"])

        def extract_table_min_width(h: str) -> int:
            # Look for the inline style on the <table> element: style="min-width: Npx;"
            m = re.search(r'<table[^>]*style="[^"]*min-width:\s*(\d+)px', h)
            return int(m.group(1)) if m else 0

        assert extract_table_min_width(mod._build_timetable_html(df_wide, "class")) > \
               extract_table_min_width(mod._build_timetable_html(df_narrow, "class"))


# ---------------------------------------------------------------------------
# 4. Long text and escaping
# ---------------------------------------------------------------------------

class TestLongTextHandling:
    @staticmethod
    def _long_df() -> pd.DataFrame:
        return pd.DataFrame({
            "Monday": [
                "Introduction to Distributed Systems and Cloud Computing in Modern Enterprise Environments"
                "\nProf. Dr. Evangelina Konstantinopoulos-Papadopoulos | Building 4 Wing C Room 24B"
            ]
        }, index=["P1"])

    def test_long_names_no_crash(self, mod):
        result = mod._build_timetable_html(self._long_df(), "class")
        assert isinstance(result, str) and len(result) > 100

    def test_xss_escaped(self, mod):
        df = pd.DataFrame({"Mon": ["<script>alert('xss')</script>\nT | R"]}, index=["P1"])
        result = mod._build_timetable_html(df, "class")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_ampersand_escaped(self, mod):
        df = pd.DataFrame({"Mon": ["Science & Technology\nDr. Smith | R1"]}, index=["P1"])
        assert "&amp;" in mod._build_timetable_html(df, "class")

    def test_long_names_no_crash_style_df(self, mod):
        rendered = mod._style_df(self._long_df()).to_html()
        assert "table" in rendered.lower()

    def test_long_names_preserved_in_output(self, mod):
        subj = "Advanced Quantum Mechanics and Relativistic Field Theory"
        teacher = "Dr. Bartholomew Feynman-Hawking"
        df = pd.DataFrame({"Mon": [f"{subj}\n{teacher} | Lab"]}, index=["P1"])
        result = mod._build_timetable_html(df, "class")
        assert subj in result
        assert teacher in result


# ---------------------------------------------------------------------------
# 5. Real timetable HTML generation
# ---------------------------------------------------------------------------

class TestBuildHtmlWithRealTimetable:
    def test_class_grids_produce_html_table(self, mod, solved_tt):
        for eid, df in mod._make_grid(solved_tt, "class").items():
            result = mod._build_timetable_html(df, "class")
            assert "<table" in result.lower(), f"Missing <table> for class {eid}"

    def test_teacher_grids_produce_html_table(self, mod, solved_tt):
        for eid, df in mod._make_grid(solved_tt, "teacher").items():
            result = mod._build_timetable_html(df, "teacher")
            assert "<table" in result.lower(), f"Missing <table> for teacher {eid}"

    def test_room_grids_produce_html_table(self, mod, solved_tt):
        for eid, df in mod._make_grid(solved_tt, "room").items():
            result = mod._build_timetable_html(df, "room")
            assert "<table" in result.lower(), f"Missing <table> for room {eid}"

    def test_lab_continuation_in_html(self, mod, solved_tt):
        labs = [(var, plc) for var, plc in solved_tt.get_placed_sessions() if var.duration > 1]
        if not labs:
            pytest.skip("No multi-period lab sessions.")
        grids = mod._make_grid(solved_tt, "class")
        for var, plc in labs:
            result = mod._build_timetable_html(grids[var.class_id], "class")
            assert "tt-card-cont" in result
            assert "Continuation" in result

    def test_lunch_label_in_class_html(self, mod, empty_tt):
        for df in mod._make_grid(empty_tt, "class").values():
            assert "Lunch Break" in mod._build_timetable_html(df, "class")

    def test_session_placements_not_mutated(self, mod, solved_tt):
        before = frozenset(
            (var.session_id, plc.day, plc.start_period, plc.room_id)
            for var, plc in solved_tt.get_placed_sessions()
        )
        for df in mod._make_grid(solved_tt, "class").values():
            mod._build_timetable_html(df, "class")
        after = frozenset(
            (var.session_id, plc.day, plc.start_period, plc.room_id)
            for var, plc in solved_tt.get_placed_sessions()
        )
        assert before == after


# ---------------------------------------------------------------------------
# 6. _render_timetable_grid
# ---------------------------------------------------------------------------

class TestRenderTimetableGrid:
    def test_callable(self, mod):
        assert callable(mod._render_timetable_grid)

    def test_does_not_crash_with_mocked_st(self, mod, solved_tt, small_ds):
        grids = mod._make_grid(solved_tt, "class")
        first_id = next(iter(grids.keys()))
        df = grids[first_id]
        cg = small_ds.classes.get(first_id)
        name = cg.name if cg else first_id
        try:
            mod._render_timetable_grid(solved_tt, "Class", first_id, name, df)
        except Exception as exc:
            if "MagicMock" not in str(type(exc)) and "MagicMock" not in str(exc):
                pytest.fail(f"Unexpected exception from _render_timetable_grid: {exc!r}")


# ---------------------------------------------------------------------------
# 7. Special characters, continuation cells, and conflict indicators
# ---------------------------------------------------------------------------

class TestSpecialCharactersAndContinuationCells:
    def test_subject_name_with_pipe_symbol(self, mod):
        """Subject name containing '|' must render properly without corrupting teacher/room metadata."""
        df = pd.DataFrame({
            "Mon": ["Theory | Computation\nDr. Turing | Room 101"]
        }, index=["P1"])
        html_out = mod._build_timetable_html(df, "class")
        assert "Theory | Computation" in html_out or "Theory | Computation" in html_out
        assert "Dr. Turing" in html_out
        assert "Room 101" in html_out
        assert "Teacher:" in html_out
        assert "Room:" in html_out

    def test_continuation_cell_exact_rendering(self, mod):
        """Continuation cell must show continuation badge and handle clean subject name."""
        df = pd.DataFrame({
            "Mon": ["... Data Structures Lab (cont.)"]
        }, index=["P2"])
        html_out = mod._build_timetable_html(df, "class")
        assert "tt-card-cont" in html_out
        assert "Continuation" in html_out
        assert "Data Structures Lab" in html_out

    def test_multiple_sessions_conflict_box(self, mod):
        """When multiple sessions overlap in a single slot, a conflict box must be rendered."""
        df = pd.DataFrame({
            "Mon": ["Physics\nDr. Smith | R1\nChemistry\nDr. Jones | R2"]
        }, index=["P1"])
        html_out = mod._build_timetable_html(df, "class")
        assert "tt-conflict-box" in html_out
        assert "Multiple Sessions / Conflict" in html_out

    def test_lunch_and_empty_distinguishable(self, mod):
        """Lunch break and empty periods must have distinct CSS classes."""
        df = pd.DataFrame({
            "Mon": ["[Lunch Break]", ""]
        }, index=["P3", "P4"])
        html_out = mod._build_timetable_html(df, "class")
        assert "tt-card-lunch" in html_out
        assert "tt-card-empty" in html_out
        assert "Lunch Break" in html_out
        assert "&mdash;" in html_out
