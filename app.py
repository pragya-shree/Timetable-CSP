"""AI Timetable Scheduler - Streamlit Application.

Unified Timetable Workflow:
- 📊 Dashboard: Central overview, active timetable status, and workflow guide.
- 🛠️ Create Timetable: Institutional configuration builder, real-time validation, and solver execution.
- 📁 My Timetables: Persistent local timetable history (save, open, rename, duplicate, delete).
- 👁️ View & Edit Timetable: Single source-of-truth schedule viewer and interactive manual editor.
- 🔬 Experiments & Benchmarks: Scientific benchmark suite (separate from personal timetables).
- 📥 Export: Export the active schedule to CSV, Excel, and JSON.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import html
import io
import json
import pathlib
import sys
import time
from typing import Any

import pandas as pd
import streamlit as st

# -- ensure project root is importable --
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.models import (
    TimetableDataset,
    load_dataset_from_file,
    check_dataset_feasibility,
)
from core.sessions import (
    Placement,
    SessionVar,
    Timetable,
    generate_session_variables,
    get_valid_domain_for_session,
)
from core.constraints import validate_timetable
from core.cost import calculate_cost
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.custom_builder import (
    create_default_custom_data,
    dataset_to_custom_tables,
    validate_and_build_custom_dataset,
    format_slots_to_string,
    CustomValidationResult,
)
from core.storage import (
    TimetableRecord,
    TimetableStore,
    TimetableNotFoundError,
    CorruptStorageFileError,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Timetable Scheduler",
    page_icon="\U0001f4c5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = {
    "Small Dataset": str(_ROOT / "datasets" / "small_dataset.json"),
    "Large Dataset": str(_ROOT / "datasets" / "large_dataset.json"),
}

ALGORITHMS = {
    "Plain Backtracking":              dict(use_mrv=False, use_degree=False, use_fc=False),
    "MRV Backtracking":                dict(use_mrv=True,  use_degree=False, use_fc=False),
    "MRV + Degree Heuristic":          dict(use_mrv=True,  use_degree=True,  use_fc=False),
    "MRV + Degree + Forward Checking": dict(use_mrv=True,  use_degree=True,  use_fc=True),
    "Min-Conflicts":                   None,   # handled separately
}

ALGO_DESCRIPTIONS = {
    "Plain Backtracking": (
        "Static variable ordering. Exhaustive depth-first search until a solution or budget limit."
    ),
    "MRV Backtracking": (
        "**Minimum Remaining Values** - selects the most constrained unassigned variable next."
    ),
    "MRV + Degree Heuristic": (
        "MRV with **Degree** tie-breaking: prefers variables involved in constraints with the most "
        "unassigned neighbours."
    ),
    "MRV + Degree + Forward Checking": (
        "MRV + Degree + **Forward Checking**: prunes neighbour domains after each placement; "
        "backtracks immediately on domain wipeout."
    ),
    "Min-Conflicts": (
        "Local search heuristic. Begins with a complete initial assignment and iteratively "
        "repairs the variable with the highest conflict count."
    ),
}

RESULTS_DIR = _ROOT / "experiments" / "results"
STORAGE_DIR = _ROOT / "saved_timetables"

# Initialize timetable store
_timetable_store = TimetableStore(STORAGE_DIR)

# ---------------------------------------------------------------------------
# Session state defaults (Safe dict and object access)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "active_timetable": None,
    "active_timetable_id": None,
    "active_timetable_name": None,
    "unsaved_edits": False,
    "active_generation_result": None,
    "custom_data": None,
    "custom_validation": None,
    "nav_selection": "Dashboard",
    # Backward-compatible keys for existing tests
    "timetable": None,
    "result": None,
    "dataset": None,
    "edited_tt": None,
    "dataset_source": "Custom",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.get("custom_data") is None:
    st.session_state["custom_data"] = create_default_custom_data()

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> TimetableDataset | None:
    """Loads a dataset from JSON file."""
    try:
        return load_dataset_from_file(pathlib.Path(path))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return None


def _set_active_timetable(timetable: Timetable | None, name: str | None = None, timetable_id: str | None = None, result: Any = None):
    """Authoritative helper to set the active timetable and synchronize legacy keys."""
    st.session_state["active_timetable"] = timetable
    st.session_state["active_timetable_id"] = timetable_id
    st.session_state["active_timetable_name"] = name or (timetable.dataset.name if timetable else "Untitled")
    st.session_state["unsaved_edits"] = False
    st.session_state["active_generation_result"] = result

    # Sync legacy keys
    st.session_state["timetable"] = timetable
    st.session_state["edited_tt"] = None
    st.session_state["result"] = result
    if timetable is not None:
        st.session_state["dataset"] = timetable.dataset
        st.session_state["dataset_source"] = name or timetable.dataset.name


def _style_df(df: pd.DataFrame) -> Any:
    """Applies cell styling compatible with both Pandas 2.x and Pandas 3.x.
    
    Uses high-contrast colors and readable typography for both light and dark themes.
    """
    def _colour(v: str) -> str:
        if not v:
            return "background-color: #f8fafc; color: #64748b; font-size: 13px;"
        if "[Lunch Break]" in v:
            return "background-color: #fef3c7; color: #92400e; font-weight: bold; font-size: 13px; border-left: 3px solid #d97706;"
        if "[Blocked Period]" in v:
            return "background-color: #f1f5f9; color: #475569; font-weight: bold; font-size: 13px; border-left: 3px solid #94a3b8;"
        if "cont." in v.lower():
            return "background-color: #f0fdf4; color: #166534; font-weight: 500; font-size: 13px; border-left: 3px solid #22c55e;"
        return "background-color: #f0f7ff; color: #1e293b; font-weight: 500; font-size: 13px; border-left: 3px solid #2563eb;"

    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(_colour)
    elif hasattr(styler, "applymap"):
        return styler.applymap(_colour)
    return styler


_TIMETABLE_CSS = """
<style>
.tt-wrapper {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0f172a;
    margin: 1rem 0;
    width: 100%;
}
.tt-scroll-container {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.tt-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
    background-color: #ffffff;
}
.tt-th {
    background-color: #0f172a;
    color: #ffffff;
    font-size: 13.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 11px 8px;
    text-align: center;
    border-right: 1px solid #334155;
    border-bottom: 2px solid #0f172a;
}
.tt-th-period {
    width: 75px;
    min-width: 75px;
    max-width: 80px;
    background-color: #020617;
    border-right: 2px solid #cbd5e1;
}
.tt-td-period {
    background-color: #f8fafc;
    color: #0f172a;
    font-weight: 700;
    font-size: 13.5px;
    text-align: center;
    vertical-align: middle;
    padding: 10px 4px;
    border-right: 2px solid #cbd5e1;
    border-bottom: 1px solid #e2e8f0;
}
.tt-td-cell {
    padding: 8px 6px;
    vertical-align: top;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    background-color: #ffffff;
    height: 100%;
}
.tt-card {
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 4px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: normal;
    white-space: normal;
    transition: background-color 0.15s ease;
}
.tt-card:last-child {
    margin-bottom: 0;
}
.tt-card-session {
    background-color: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
}
.tt-card-session:hover {
    background-color: #e0f0fe;
}
.tt-subj-title {
    font-size: 13.5px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.35;
    margin-bottom: 2px;
}
.tt-meta-line {
    font-size: 12px;
    color: #334155;
    line-height: 1.3;
}
.tt-room-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: #1e293b;
    background-color: #e2e8f0;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 1px 5px;
    margin-top: 2px;
    width: fit-content;
}
.tt-card-cont {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
}
.tt-card-cont:hover {
    background-color: #dcfce7;
}
.tt-subj-cont {
    font-size: 13px;
    font-weight: 600;
    color: #14532d;
    line-height: 1.35;
}
.tt-cont-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: #15803d;
    background-color: #dcfce7;
    border: 1px solid #86efac;
    border-radius: 4px;
    padding: 1px 5px;
    width: fit-content;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.tt-card-lunch {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #d97706;
    padding: 12px 8px;
    text-align: center;
    align-items: center;
    justify-content: center;
}
.tt-lunch-label {
    font-size: 12.5px;
    font-weight: 700;
    color: #92400e;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.tt-card-blocked {
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #64748b;
    padding: 12px 8px;
    text-align: center;
    align-items: center;
    justify-content: center;
}
.tt-blocked-label {
    font-size: 12.5px;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.04em;
}
.tt-card-empty {
    padding: 14px 8px;
    text-align: center;
    align-items: center;
    justify-content: center;
}
.tt-empty-dash {
    color: #cbd5e1;
    font-size: 15px;
}
.tt-conflict-box {
    border: 2px solid #ef4444;
    border-radius: 6px;
    padding: 4px;
    background-color: #fef2f2;
}
.tt-conflict-header {
    font-size: 11px;
    font-weight: 700;
    color: #b91c1c;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.tt-legend-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    padding: 10px 14px;
    background-color: #f8fafc;
    border-top: 1px solid #cbd5e1;
    font-size: 12px;
    color: #334155;
}
.tt-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.tt-swatch {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    vertical-align: middle;
}
.tt-swatch-session {
    background-color: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
}
.tt-swatch-cont {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a;
}
.tt-swatch-lunch {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #d97706;
}
.tt-swatch-blocked {
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #64748b;
}
.tt-swatch-empty {
    background-color: #ffffff;
    border: 1px dashed #cbd5e1;
}
</style>
"""


def _build_timetable_html(grid_df: pd.DataFrame, group_by: str = "class") -> str:
    """Builds a responsive, accessible HTML table for academic schedules."""
    days = list(grid_df.columns)
    periods = list(grid_df.index)
    min_width = max(780, len(days) * 140 + 80)

    def _render_cell_content(cell_text: str) -> str:
        if not cell_text or not cell_text.strip():
            return '<div class="tt-card tt-card-empty" title="Free Period"><span class="tt-empty-dash">&mdash;</span></div>'

        text = cell_text.strip()
        raw_lines = [l.strip() for l in text.split("\n") if l.strip()]

        cards = []
        i = 0
        session_count = 0

        while i < len(raw_lines):
            line = raw_lines[i]
            if line == "[Lunch Break]":
                cards.append(
                    '<div class="tt-card tt-card-lunch" title="Lunch Break">'
                    '<span class="tt-lunch-label">&#9749; Lunch Break</span>'
                    '</div>'
                )
                i += 1
                continue
            if line == "[Blocked Period]":
                cards.append(
                    '<div class="tt-card tt-card-blocked" title="Blocked Period (Global Break / Assembly)">'
                    '<span class="tt-blocked-label">&#9940; Blocked</span>'
                    '</div>'
                )
                i += 1
                continue
            if line.startswith("... ") or "(cont.)" in line.lower():
                clean_subj = line.replace("... ", "").replace("(cont.)", "").strip()
                cards.append(
                    f'<div class="tt-card tt-card-cont" title="{html.escape(line)}">'
                    f'<div class="tt-subj-cont">&rdsh; {html.escape(clean_subj)}</div>'
                    f'<span class="tt-cont-badge">Continuation</span>'
                    f'</div>'
                )
                i += 1
                continue

            subj_name = line
            meta_line = ""
            if i + 1 < len(raw_lines) and not raw_lines[i + 1].startswith("[") and not raw_lines[i + 1].startswith("..."):
                meta_line = raw_lines[i + 1]
                i += 2
            else:
                i += 1

            session_count += 1
            meta1, meta2 = "", ""
            if "|" in meta_line:
                parts = [p.strip() for p in meta_line.split("|", 1)]
                meta1, meta2 = parts[0], parts[1]
            elif meta_line:
                meta1 = meta_line

            if group_by == "class":
                label1, label2 = "Teacher: ", "Room: "
            elif group_by == "teacher":
                label1, label2 = "Class: ", "Room: "
            else:  # room
                label1, label2 = "Class: ", "Teacher: "

            meta_html = []
            if meta1:
                meta_html.append(
                    f'<div class="tt-meta-line"><span style="color:#64748b; font-weight:500;">{label1}</span>'
                    f'<span style="font-weight:600; color:#1e293b;">{html.escape(meta1)}</span></div>'
                )
            if meta2:
                meta_html.append(
                    f'<div class="tt-room-tag"><span style="color:#64748b; font-weight:500;">{label2}</span>'
                    f'{html.escape(meta2)}</div>'
                )

            tooltip = f"{subj_name}" + (f" | {meta_line}" if meta_line else "")
            cards.append(
                f'<div class="tt-card tt-card-session" title="{html.escape(tooltip)}">'
                f'<div class="tt-subj-title">{html.escape(subj_name)}</div>'
                f'{"".join(meta_html)}'
                f'</div>'
            )

        if session_count > 1 or (session_count >= 1 and any("Lunch" in c or "Blocked" in c for c in cards)):
            return f'<div class="tt-conflict-box"><div class="tt-conflict-header">&#9888; Multiple Sessions / Conflict</div>{"".join(cards)}</div>'

        return "".join(cards)

    rows_html = []
    for period in periods:
        cells_html = [f'<td class="tt-td-period">{html.escape(str(period))}</td>']
        for day in days:
            cell_val = str(grid_df.loc[period, day]) if day in grid_df.columns and period in grid_df.index else ""
            cells_html.append(f'<td class="tt-td-cell">{_render_cell_content(cell_val)}</td>')
        rows_html.append(f'<tr>{"".join(cells_html)}</tr>')

    day_headers = "".join([f'<th class="tt-th">{html.escape(str(d))}</th>' for d in days])

    return f"""{_TIMETABLE_CSS}
<div class="tt-wrapper">
  <div class="tt-scroll-container">
    <table class="tt-table" style="min-width: {min_width}px;">
      <thead>
        <tr>
          <th class="tt-th tt-th-period">Period</th>
          {day_headers}
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </div>
  <div class="tt-legend-bar">
    <span style="font-weight: 700; color: #0f172a;">Legend:</span>
    <span class="tt-legend-item"><span class="tt-swatch tt-swatch-session"></span> Lecture / Session</span>
    <span class="tt-legend-item"><span class="tt-swatch tt-swatch-cont"></span> Lab Continuation</span>
    <span class="tt-legend-item"><span class="tt-swatch tt-swatch-lunch"></span> Lunch Break</span>
    <span class="tt-legend-item"><span class="tt-swatch tt-swatch-blocked"></span> Blocked Slot</span>
    <span class="tt-legend-item"><span class="tt-swatch tt-swatch-empty"></span> Free Period</span>
  </div>
</div>
"""


def _make_grid(timetable: Timetable | None, group_by: str) -> dict[str, pd.DataFrame]:
    """Build {entity_id: DataFrame} grid. Rows=periods, Cols=days.
    
    Safe against missing entities and pre-populates all entities in dataset.
    """
    if timetable is None or getattr(timetable, "dataset", None) is None:
        return {}
    ds = timetable.dataset
    ts = ds.time_structure
    days = list(ts.days)
    periods = list(range(ts.periods_per_day))
    period_labels = [f"P{p + 1}" for p in periods]

    grids: dict[str, dict[tuple[int, int], str]] = {}

    def _add(entity: str, d: int, p: int, text: str) -> None:
        grids.setdefault(entity, {})
        prev = grids[entity].get((d, p), "")
        grids[entity][(d, p)] = (prev + "\n" + text).strip() if prev else text

    # Pre-populate all entities so even empty schedules can be viewed
    if group_by == "class":
        for cid, cg in ds.classes.items():
            grids.setdefault(cid, {})
            for d in range(len(days)):
                if 0 <= cg.lunch_slot < ts.periods_per_day:
                    _add(cid, d, cg.lunch_slot, "[Lunch Break]")
            for d, p in ts.blocked_periods:
                if 0 <= d < len(days) and 0 <= p < ts.periods_per_day:
                    _add(cid, d, p, "[Blocked Period]")
    elif group_by == "teacher":
        for tid in ds.teachers:
            grids.setdefault(tid, {})
            for d, p in ts.blocked_periods:
                if 0 <= d < len(days) and 0 <= p < ts.periods_per_day:
                    _add(tid, d, p, "[Blocked Period]")
    else:  # room
        for rid in ds.rooms:
            grids.setdefault(rid, {})
            for d, p in ts.blocked_periods:
                if 0 <= d < len(days) and 0 <= p < ts.periods_per_day:
                    _add(rid, d, p, "[Blocked Period]")

    for var, plc in timetable.get_placed_sessions():
        subj = ds.subjects.get(var.subject_id)
        subj_name = subj.name if subj else var.subject_id

        cg = ds.classes.get(var.class_id)
        class_name = cg.name if cg else var.class_id

        teacher = ds.teachers.get(var.teacher_id)
        teacher_name = teacher.name if teacher else var.teacher_id

        room = ds.rooms.get(plc.room_id)
        room_name = room.name if room else plc.room_id

        occupied = plc.get_occupied_slots(var.duration)
        first_p = min(p for _, p in occupied) if occupied else plc.start_period

        for d, p in occupied:
            if not (0 <= d < len(days) and 0 <= p < ts.periods_per_day):
                continue
            is_start = p == first_p
            if group_by == "class":
                cell = (
                    f"{subj_name}\n{teacher_name} | {room_name}"
                    if is_start else f"... {subj_name} (cont.)"
                )
                _add(var.class_id, d, p, cell)
            elif group_by == "teacher":
                cell = (
                    f"{subj_name}\n{class_name} | {room_name}"
                    if is_start else f"... {subj_name} (cont.)"
                )
                _add(var.teacher_id, d, p, cell)
            else:  # room
                cell = (
                    f"{subj_name}\n{class_name} | {teacher_name}"
                    if is_start else f"... {subj_name} (cont.)"
                )
                _add(plc.room_id, d, p, cell)

    dfs: dict[str, pd.DataFrame] = {}
    for entity in sorted(grids.keys()):
        slot_map = grids[entity]
        data = {
            day: [slot_map.get((d, p), "") for p in periods]
            for d, day in enumerate(days)
        }
        dfs[entity] = pd.DataFrame(data, index=period_labels)

    return dfs


def _render_timetable_grid(
    timetable: Timetable,
    group_by: str,
    selected_id: str,
    selected_name: str,
    grid_df: pd.DataFrame,
) -> None:
    """Renders a responsive, high-contrast timetable grid with full details.
    
    Reused across Class, Teacher, and Room timetable views.
    """
    ds = timetable.dataset
    ts = ds.time_structure

    # 1. Render primary responsive HTML timetable grid
    grid_html = _build_timetable_html(grid_df, group_by.lower())
    if hasattr(st, "html"):
        st.html(grid_html)
    else:
        st.markdown(grid_html, unsafe_allow_html=True)

    # 2. Detailed Placed Sessions List (handles long names, prevents any truncation)
    rel_sessions = []
    for var, plc in timetable.get_placed_sessions():
        if group_by == "Class" and var.class_id != selected_id:
            continue
        if group_by == "Teacher" and var.teacher_id != selected_id:
            continue
        if group_by == "Room" and plc.room_id != selected_id:
            continue

        subj = ds.subjects.get(var.subject_id)
        subj_name = subj.name if subj else var.subject_id
        cg = ds.classes.get(var.class_id)
        class_name = cg.name if cg else var.class_id
        teacher = ds.teachers.get(var.teacher_id)
        teacher_name = teacher.name if teacher else var.teacher_id
        room = ds.rooms.get(plc.room_id)
        room_name = room.name if room else plc.room_id

        day_name = ts.days[plc.day] if 0 <= plc.day < len(ts.days) else f"Day {plc.day}"
        end_p = plc.start_period + var.duration
        period_str = f"P{plc.start_period + 1}" if var.duration == 1 else f"P{plc.start_period + 1}–P{end_p}"

        rel_sessions.append({
            "Day": day_name,
            "Period": period_str,
            "DayIdx": plc.day,
            "PeriodIdx": plc.start_period,
            "Subject Name": subj_name,
            "Subject ID": var.subject_id,
            "Class Name": class_name,
            "Class ID": var.class_id,
            "Teacher Name": teacher_name,
            "Teacher ID": var.teacher_id,
            "Room Name": room_name,
            "Room ID": plc.room_id,
            "Duration": f"{var.duration} period{'s' if var.duration > 1 else ''}",
            "Type": "Lab" if var.duration > 1 else "Theory",
        })

    if rel_sessions:
        rel_sessions.sort(key=lambda x: (x["DayIdx"], x["PeriodIdx"]))
        detail_df = pd.DataFrame(rel_sessions).drop(columns=["DayIdx", "PeriodIdx"])
        with st.expander(f"📋 Detailed Session Breakdown for {selected_name} ({len(rel_sessions)} placed sessions)", expanded=False):
            st.dataframe(detail_df, use_container_width=True)

    # 3. Raw Data Table / Spreadsheet View Expander
    with st.expander("📊 View Raw Matrix / Spreadsheet View", expanded=False):
        st.dataframe(
            _style_df(grid_df),
            use_container_width=True,
            height=min(45 + 38 * len(grid_df), 500),
        )
        st.caption("Multi-period labs show '... (cont.)' in continuation rows. Blocked breaks and lunch slots are labeled.")


def _sessions_df(timetable: Timetable | None, algorithm: str = "") -> pd.DataFrame:
    """Flat DataFrame of all placed sessions for CSV/Excel export."""
    if timetable is None or getattr(timetable, "dataset", None) is None:
        return pd.DataFrame()
    ds = timetable.dataset
    rows = []
    for var, plc in timetable.get_placed_sessions():
        subj = ds.subjects.get(var.subject_id)
        subj_name = subj.name if subj else var.subject_id
        cg = ds.classes.get(var.class_id)
        class_name = cg.name if cg else var.class_id
        teacher = ds.teachers.get(var.teacher_id)
        teacher_name = teacher.name if teacher else var.teacher_id
        room = ds.rooms.get(plc.room_id)
        room_name = room.name if room else plc.room_id

        day_name = ds.time_structure.days[plc.day] if 0 <= plc.day < len(ds.time_structure.days) else f"Day {plc.day}"

        row_dict = {
            "session_id":   var.session_id,
            "class_id":     var.class_id,
            "class_name":   class_name,
            "subject_id":   var.subject_id,
            "subject_name": subj_name,
            "session_no":   var.session_no,
            "teacher_id":   var.teacher_id,
            "teacher_name": teacher_name,
            "day":          day_name,
            "start_period": plc.start_period + 1,
            "duration":     var.duration,
            "room_id":      plc.room_id,
            "room_name":    room_name,
            "session_type": var.session_type,
        }
        if algorithm:
            row_dict["algorithm"] = algorithm
        rows.append(row_dict)

    return pd.DataFrame(rows)


def _show_violations(violations: list[Any], max_show: int = 15) -> None:
    """Displays constraint violations in an actionable alert format."""
    for v in violations[:max_show]:
        st.warning(f"**{v.constraint_id}**: {v.message}")
    if len(violations) > max_show:
        st.caption(f"... and {len(violations) - max_show} more violations.")


# ---------------------------------------------------------------------------
# Sidebar Navigation (Reorganized as per Requirement 1)
# ---------------------------------------------------------------------------
NAV_PAGES = [
    "Dashboard",
    "Create Timetable",
    "My Timetables",
    "View & Edit Timetable",
    "Experiments & Benchmarks",
    "Export",
]

NAV_ICONS = {
    "Dashboard": "\U0001f4ca Dashboard",
    "Create Timetable": "\U0001f6e0\ufe0f Create Timetable",
    "My Timetables": "\U0001f4c1 My Timetables",
    "View & Edit Timetable": "\U0001f441\ufe0f View & Edit Timetable",
    "Experiments & Benchmarks": "\U0001f52c Experiments & Benchmarks",
    "Export": "\U0001f4e5 Export",
}

with st.sidebar:
    st.title("\U0001f4c5 Timetable Scheduler")
    st.caption("AI-Powered CSP & Local Search Engine")
    st.markdown("---")

    # Handle programmatic navigation
    cur_nav = st.session_state.get("nav_selection", "Dashboard")
    if cur_nav not in NAV_PAGES:
        cur_nav = "Dashboard"

    selected_nav = st.radio(
        "Navigation",
        NAV_PAGES,
        index=NAV_PAGES.index(cur_nav),
        format_func=lambda p: NAV_ICONS.get(p, p),
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )
    st.session_state["nav_selection"] = selected_nav

    st.markdown("---")
    st.markdown("### Active Timetable")
    active_tt = st.session_state.get("active_timetable")
    if active_tt is not None:
        tt_name = st.session_state.get("active_timetable_name", active_tt.dataset.name)
        val = validate_timetable(active_tt)
        status_badge = "\u2705 Valid" if val.is_valid else f"\u26a0 {val.hard_violation_count} Violations"
        st.success(f"**{tt_name}**\n\n{status_badge}")
        if st.session_state.get("unsaved_edits"):
            st.warning("\u26a0 Unsaved Manual Edits")
    else:
        st.info("No active timetable.\nGenerate one or open from My Timetables.")


# ===========================================================================
# 1. DASHBOARD
# ===========================================================================
if selected_nav == "Dashboard":
    st.title("\U0001f4ca Dashboard")
    st.markdown("Welcome to the **AI-Based College Timetable Scheduler**.")

    # Section A: Active Timetable Status
    st.header("Active Timetable Status")
    active_tt = st.session_state.get("active_timetable")
    if active_tt is not None:
        tt_name = st.session_state.get("active_timetable_name", active_tt.dataset.name)
        ds = active_tt.dataset
        val = validate_timetable(active_tt)
        cost = calculate_cost(active_tt)
        placed_cnt = len(active_tt.get_placed_sessions())
        total_sessions = len(generate_session_variables(ds))

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Schedule Name", tt_name)
        d_col2.metric("Validity", "Conflict-Free \u2705" if val.is_valid else f"{val.hard_violation_count} Conflicts \u274c")
        d_col3.metric("Total Cost", cost.total_cost)
        d_col4.metric("Placed Sessions", f"{placed_cnt}/{total_sessions}")

        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("\U0001f441\ufe0f Open in View & Edit Timetable", use_container_width=True):
                st.session_state["nav_selection"] = "View & Edit Timetable"
                st.rerun()
        with btn_c2:
            if st.button("\U0001f4e5 Jump to Export", use_container_width=True):
                st.session_state["nav_selection"] = "Export"
                st.rerun()
        with btn_c3:
            if st.session_state.get("unsaved_edits"):
                st.warning("You have unsaved edits on this timetable.")
    else:
        st.info("No timetable is currently active.")
        st.markdown(
            "Welcome to the **AI-Based College Timetable Scheduler**! "
            "Get started by configuring your institution and generating a timetable in **Create Timetable**, "
            "or open a previously saved timetable from **My Timetables**."
        )
        ec1, ec2, _ = st.columns([1, 1, 1])
        with ec1:
            if st.button("🛠️ Create Timetable", use_container_width=True, type="primary"):
                st.session_state["nav_selection"] = "Create Timetable"
                st.rerun()
        with ec2:
            if st.button("📁 Open Saved Timetable", use_container_width=True):
                st.session_state["nav_selection"] = "My Timetables"
                st.rerun()

    st.markdown("---")

    # Section B: System Workflow Guide
    st.header("System Workflow Guide")
    st.markdown("Follow these steps to configure, generate, and manage your schedules:")
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        st.markdown(
            "### 1. 🛠️ Configure Institution\n"
            "Define your working days, periods, classes, faculty, subjects, and rooms in **Create Timetable**. "
            "Real-time validation checks all H1–H9 hard constraint requirements and capacity limits."
        )
        st.markdown(
            "### 2. ⚡ Generate Timetable\n"
            "Select an algorithm (Backtracking with MRV, Degree, Forward Checking, or Min-Conflicts local search). "
            "Independent verification checks that every placement is completely conflict-free."
        )

    with w_col2:
        st.markdown(
            "### 3. 👁️ View & Manually Edit\n"
            "Inspect schedules in interactive Class, Teacher, and Room grids. "
            "Manually move sessions with instant feedback on constraint violations and cost recalculation."
        )
        st.markdown(
            "### 4. 📁 Save & Export\n"
            "Save generated and edited schedules to **My Timetables** with full version history. "
            "Export the active timetable to CSV, Excel (`.xlsx`), or JSON."
        )

    st.markdown("---")

    # Section C: Recent Saved Timetables
    st.header("Recently Saved Timetables")
    saved_records = _timetable_store.list_all()
    if saved_records:
        rec_data = []
        for r in saved_records[:5]:
            rec_data.append({
                "Name": r.name,
                "Algorithm": r.algorithm,
                "Last Modified": r.updated_at[:19].replace("T", " "),
                "Hard Violations": r.validation.get("hard_violation_count", 0),
                "Total Cost": r.cost.get("total_cost", 0),
                "Version": f"v{r.version}",
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True)
        if st.button("\U0001f4c1 Manage All Saved Timetables", use_container_width=True):
            st.session_state["nav_selection"] = "My Timetables"
            st.rerun()
    else:
        st.caption("No saved timetables yet in `saved_timetables/`. Once you generate and save a timetable, it will appear here.")


# ===========================================================================
# 2. CREATE TIMETABLE (Unified Workflow: Configure -> Validate -> Solve -> Act)
# ===========================================================================
elif selected_nav == "Create Timetable":
    st.title("🛠️ Create Timetable")
    st.markdown(
        "Configure institution requirements, validate the data schema, "
        "and generate conflict-free schedules using AI algorithms."
    )

    # Step 1: Institutional Configuration
    st.header("Step 1: Institutional Configuration")
    c_data = st.session_state["custom_data"]

    # Toolbar
    tb1, tb2 = st.columns(2)
    with tb1:
        if st.button("🔄 Reset Template", use_container_width=True):
            st.session_state["custom_data"] = create_default_custom_data()
            st.session_state["custom_validation"] = None
            st.rerun()
    with tb2:
        if st.button("🗑️ Clear All Tables", use_container_width=True):
            st.session_state["custom_data"] = {
                "config": {
                    "name": "My Custom Timetable",
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "periods_per_day": 6,
                    "default_lunch_slot": 4,
                    "blocked_periods": "",
                },
                "classes": pd.DataFrame(columns=["class_id", "class_name", "strength", "lunch_slot"]),
                "teachers": pd.DataFrame(columns=["teacher_id", "teacher_name", "max_weekly_load", "qualified_subjects", "unavailable_slots", "preferred_slots"]),
                "subjects": pd.DataFrame(columns=["subject_id", "subject_name", "class_id", "teacher_id", "type", "weekly_periods", "lab_duration"]),
                "rooms": pd.DataFrame(columns=["room_id", "room_name", "capacity", "type"]),
            }
            st.session_state["custom_validation"] = None
            st.rerun()

    # Custom Builder Tabs
    tabs = st.tabs([
        "📅 Time Grid & Breaks",
        "🏫 Classes",
        "👨‍🏫 Teachers",
        "📚 Subjects & Offerings",
        "🏛️ Rooms",
    ])

    # TAB 1: Time Grid
    with tabs[0]:
        cfg = c_data["config"]
        ds_name = st.text_input("Dataset Name", value=cfg.get("name", "Custom College Timetable"))
        tg1, tg2, tg3 = st.columns(3)
        with tg1:
            days_str = st.text_input(
                "Working Days (comma-separated)",
                value=", ".join(cfg.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])),
            )
        with tg2:
            periods_per_day = st.number_input(
                "Periods Per Day",
                min_value=2,
                max_value=12,
                value=int(cfg.get("periods_per_day", 6)),
            )
        with tg3:
            default_lunch = st.number_input(
                "Default Lunch Period (1-indexed)",
                min_value=1,
                max_value=int(periods_per_day),
                value=min(int(cfg.get("default_lunch_slot", 4)), int(periods_per_day)),
            )
        blocked_str = st.text_input(
            "Global Blocked Slots (e.g. Assembly)",
            value=str(cfg.get("blocked_periods", "")),
            help="Format: 'Monday P1, Wednesday P4' or '(0, 0)'",
        )
        cfg["name"] = ds_name
        cfg["days"] = [d.strip() for d in days_str.split(",") if d.strip()]
        cfg["periods_per_day"] = periods_per_day
        cfg["default_lunch_slot"] = default_lunch
        cfg["blocked_periods"] = blocked_str

    # TAB 2: Classes
    with tabs[1]:
        st.caption("Add student batches. Lunch slot is 1-indexed for display.")
        c_data["classes"] = st.data_editor(
            c_data["classes"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "class_id": st.column_config.TextColumn("Class ID", required=True),
                "class_name": st.column_config.TextColumn("Class Name", required=True),
                "strength": st.column_config.NumberColumn("Strength", min_value=1, required=True),
                "lunch_slot": st.column_config.NumberColumn("Lunch Period", min_value=1, max_value=int(periods_per_day), required=True),
            },
            key="classes_table_editor",
        )

    # TAB 3: Teachers
    with tabs[2]:
        st.caption("Manage faculty. Slots accept 'Monday P1' or '(0, 0)' notation.")
        c_data["teachers"] = st.data_editor(
            c_data["teachers"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "teacher_id": st.column_config.TextColumn("Teacher ID", required=True),
                "teacher_name": st.column_config.TextColumn("Teacher Name", required=True),
                "max_weekly_load": st.column_config.NumberColumn("Max Weekly Load", min_value=1, required=True),
                "qualified_subjects": st.column_config.TextColumn("Qualified Subjects"),
                "unavailable_slots": st.column_config.TextColumn("Unavailable Slots"),
                "preferred_slots": st.column_config.TextColumn("Preferred Slots"),
            },
            key="teachers_table_editor",
        )

    # TAB 4: Subjects
    with tabs[3]:
        st.caption("Bind subjects to classes and teachers. Lab duration applies to labs.")
        c_data["subjects"] = st.data_editor(
            c_data["subjects"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "subject_id": st.column_config.TextColumn("Subject ID", required=True),
                "subject_name": st.column_config.TextColumn("Subject Title", required=True),
                "class_id": st.column_config.TextColumn("Class ID", required=True),
                "teacher_id": st.column_config.TextColumn("Teacher ID", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["theory", "lab"], required=True),
                "weekly_periods": st.column_config.NumberColumn("Weekly Hours", min_value=1, required=True),
                "lab_duration": st.column_config.NumberColumn("Lab Duration", min_value=1, default=1),
            },
            key="subjects_table_editor",
        )

    # TAB 5: Rooms
    with tabs[4]:
        st.caption("Define classrooms and laboratories with seating capacity.")
        c_data["rooms"] = st.data_editor(
            c_data["rooms"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "room_id": st.column_config.TextColumn("Room ID", required=True),
                "room_name": st.column_config.TextColumn("Room Name", required=True),
                "capacity": st.column_config.NumberColumn("Capacity", min_value=1, required=True),
                "type": st.column_config.SelectboxColumn("Room Type", options=["classroom", "lab"], required=True),
            },
            key="rooms_table_editor",
        )

    # Validate Custom Input
    st.markdown("---")
    val_res = validate_and_build_custom_dataset(
        config=c_data["config"],
        classes_data=c_data["classes"],
        teachers_data=c_data["teachers"],
        subjects_data=c_data["subjects"],
        rooms_data=c_data["rooms"],
    )
    st.session_state["custom_validation"] = val_res

    validated_dataset: TimetableDataset | None = None
    if val_res.is_valid and val_res.dataset is not None:
        validated_dataset = val_res.dataset
        st.success(f"✅ Configuration for **'{val_res.dataset.name}'** is valid and ready for generation.")
        if val_res.warnings:
            with st.expander("⚠️ Feasibility Warnings", expanded=False):
                for w in val_res.warnings:
                    st.warning(w)
    else:
        st.error(f"❌ Configuration has {len(val_res.errors)} validation error(s):")
        for err in val_res.errors:
            st.error(f"• {err}")

    # Step 2: Solver Selection & Timetable Generation
    st.markdown("---")
    st.header("Step 2: Generate Timetable")

    if validated_dataset is None:
        st.info("Please configure your institution above and resolve any validation errors to enable timetable generation.")
    else:
        st.info(f"Target dataset: **{validated_dataset.name}** ({len(validated_dataset.classes)} classes, {len(validated_dataset.teachers)} teachers, {len(validated_dataset.rooms)} rooms).")

        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            algo_choice = st.selectbox("Select Algorithm", list(ALGORITHMS.keys()), key="create_algo_choice")
        with gen_col2:
            st.caption("Algorithm Overview:")
            st.markdown(ALGO_DESCRIPTIONS[algo_choice])

        with st.expander("Solver Parameters", expanded=False):
            if algo_choice == "Min-Conflicts":
                p1, p2, p3 = st.columns(3)
                mc_iter = p1.number_input("Max Iterations", 100, 100_000, 5000, 500)
                mc_time = p2.number_input("Time Limit (s)", 1, 300, 30)
                mc_seed = p3.number_input("Random Seed", 0, 999_999, 42)
                mc_strat = st.selectbox("Initial Strategy", ["random", "greedy"])
            else:
                p1, p2 = st.columns(2)
                bt_nodes = p1.number_input("Max Nodes", 1000, 10_000_000, 100_000, 10_000)
                bt_time = p2.number_input("Time Limit (s)", 1, 300, 30)

        # Run Solver Button
        if st.button("🚀 Generate Timetable", type="primary", use_container_width=True):
            with st.spinner(f"Running {algo_choice} on {validated_dataset.name}..."):
                if algo_choice == "Min-Conflicts":
                    gen_result = solve_min_conflicts(
                        dataset=validated_dataset,
                        max_iterations=int(mc_iter),
                        time_limit_seconds=float(mc_time),
                        random_seed=int(mc_seed),
                        initial_strategy=mc_strat,
                        record_cost_history=True,
                    )
                else:
                    gen_result = solve_backtracking(
                        dataset=validated_dataset,
                        max_nodes=int(bt_nodes),
                        time_limit_seconds=float(bt_time),
                        **(ALGORITHMS[algo_choice] or {}),
                    )

            # Independent Solution Verification
            if gen_result.success and gen_result.timetable is not None:
                indep_check = validate_timetable(gen_result.timetable)
                if not indep_check.is_valid:
                    gen_result.success = False
                    gen_result.status = "invalid_solution"
                    gen_result.failure_reason = f"Independent validation detected {indep_check.hard_violation_count} hard violations."

            st.session_state["active_generation_result"] = gen_result

            if gen_result.success and gen_result.timetable is not None:
                _set_active_timetable(
                    timetable=gen_result.timetable,
                    name=f"{validated_dataset.name} - {algo_choice}",
                    result=gen_result,
                )

        # Display Generation Result
        cur_res = st.session_state.get("active_generation_result")
        active_tt = st.session_state.get("active_timetable")

        if cur_res is not None:
            st.markdown("---")
            st.subheader("Generation Result")

            if cur_res.success:
                st.success(f"\u2705 Successfully generated conflict-free schedule using **{algo_choice}**!")
            else:
                st.error(f"\u274c Solver did not find a valid schedule ({cur_res.status}). Reason: {getattr(cur_res, 'failure_reason', 'Search budget limit reached')}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", cur_res.status)
            m2.metric("Runtime", f"{cur_res.runtime_seconds:.4f}s")
            hv = getattr(cur_res, "hard_violations", getattr(cur_res, "final_hard_violations", 0))
            m3.metric("Hard Violations", hv)
            m4.metric("Total Cost", cur_res.total_cost if cur_res.success else "N/A")

            # Post-Generation Actions
            if cur_res.success and active_tt is not None:
                st.markdown("### Actions")
                act_col1, act_col2, act_col3 = st.columns(3)

                with act_col1:
                    if st.button("👁️ View Timetable", use_container_width=True, type="primary"):
                        st.session_state["nav_selection"] = "View & Edit Timetable"
                        st.rerun()

                with act_col2:
                    save_name = st.text_input("Timetable Name", value=st.session_state.get("active_timetable_name", f"{validated_dataset.name} - {algo_choice}"))
                    if st.button("💾 Save Timetable to History", use_container_width=True):
                        search_stats = {
                            "nodes_explored": getattr(cur_res, "nodes_explored", None),
                            "backtracks": getattr(cur_res, "backtracks", None),
                            "iterations": getattr(cur_res, "iterations", None),
                        }
                        rec = TimetableRecord.create_from_timetable(
                            timetable=active_tt,
                            name=save_name,
                            algorithm=algo_choice,
                            runtime_seconds=cur_res.runtime_seconds,
                            search_metrics=search_stats,
                        )
                        _timetable_store.save(rec)
                        st.session_state["active_timetable_id"] = rec.id
                        st.session_state["active_timetable_name"] = rec.name
                        st.success(f"Saved to My Timetables as '{rec.name}'!")

                with act_col3:
                    st.caption("Want to try another solver?")
                    st.write("Select a different algorithm above and click Generate again.")


# ===========================================================================
# 3. MY TIMETABLES (Persistent Storage & History)
# ===========================================================================
elif selected_nav == "My Timetables":
    st.title("\U0001f4c1 My Timetables")
    st.markdown("View, open, rename, duplicate, or delete saved timetables.")
    st.caption(f"Storage location: `{_timetable_store.storage_dir}` (stored locally as JSON files).")

    records = _timetable_store.list_all()

    if not records:
        st.info("No saved timetables found. Generate a timetable in **Create Timetable** and click Save!")
    else:
        st.subheader(f"Saved Timetable Records ({len(records)})")

        active_id = st.session_state.get("active_timetable_id")

        for idx, rec in enumerate(records):
            is_active = (rec.id == active_id)
            box_type = st.success if is_active else st.container

            with st.expander(f"{'⭐ [ACTIVE] ' if is_active else ''}{rec.name} — ({rec.algorithm}, Total Cost: {rec.cost.get('total_cost', '?')})", expanded=is_active):
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.write(f"**ID:** `{rec.id}`")
                rc2.write(f"**Algorithm:** {rec.algorithm}")
                rc3.write(f"**Version:** v{rec.version}")
                rc4.write(f"**Modified:** {rec.updated_at[:19].replace('T', ' ')}")

                val_cnt = rec.validation.get("hard_violation_count", 0)
                if val_cnt == 0 and rec.validation.get("is_valid", False):
                    st.caption("\u2705 Status: Valid Conflict-Free Timetable")
                else:
                    st.caption(f"\u26a0 Status: {val_cnt} Hard Violations (Conflicts Detected)")

                # Actions row
                a_col1, a_col2, a_col3, a_col4, a_col5 = st.columns(5)

                with a_col1:
                    if st.button("\U0001f442 Set Active", key=f"set_active_{rec.id}", disabled=is_active, use_container_width=True):
                        try:
                            tt = rec.get_timetable()
                            _set_active_timetable(tt, name=rec.name, timetable_id=rec.id)
                            st.success(f"Activated '{rec.name}'!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error loading timetable: {exc}")

                with a_col2:
                    new_name = st.text_input("New Name", value=rec.name, key=f"rename_input_{rec.id}")
                    if st.button("✏️ Rename", key=f"rename_btn_{rec.id}", use_container_width=True):
                        if new_name.strip() and new_name.strip() != rec.name:
                            _timetable_store.rename(rec.id, new_name.strip())
                            if is_active:
                                st.session_state["active_timetable_name"] = new_name.strip()
                            st.success("Renamed!")
                            st.rerun()

                with a_col3:
                    if st.button("📋 Duplicate", key=f"dup_btn_{rec.id}", use_container_width=True):
                        dup = _timetable_store.duplicate(rec.id)
                        st.success(f"Duplicated as '{dup.name}'!")
                        st.rerun()

                with a_col4:
                    confirm_del = st.checkbox("Confirm", key=f"confirm_del_{rec.id}")
                    if st.button("🗑️ Delete", key=f"del_btn_{rec.id}", disabled=not confirm_del, use_container_width=True):
                        _timetable_store.delete(rec.id)
                        if is_active:
                            _set_active_timetable(None)
                        st.success(f"Deleted '{rec.name}'.")
                        st.rerun()

                with a_col5:
                    rec_json = json.dumps(rec.to_dict(), indent=2).encode("utf-8")
                    st.download_button(
                        "📥 JSON Backup",
                        rec_json,
                        file_name=f"{rec.id}.json",
                        mime="application/json",
                        key=f"dl_json_{rec.id}",
                        use_container_width=True,
                    )


# ===========================================================================
# 4. VIEW & EDIT TIMETABLE (Single Source-of-Truth Viewer & Manual Editor)
# ===========================================================================
elif selected_nav == "View & Edit Timetable":
    st.title("\U0001f441\ufe0f View & Edit Timetable")

    active_tt = st.session_state.get("active_timetable")

    if active_tt is None:
        st.warning("No active timetable is currently loaded.")
        st.markdown("Please generate a timetable in **Create Timetable** or open an existing schedule in **My Timetables**.")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("➕ Go to Create Timetable", use_container_width=True, type="primary"):
                st.session_state["nav_selection"] = "Create Timetable"
                st.rerun()
        with btn_c2:
            if st.button("📁 Go to My Timetables", use_container_width=True):
                st.session_state["nav_selection"] = "My Timetables"
                st.rerun()
        st.stop()

    ds = active_tt.dataset
    tt_name = st.session_state.get("active_timetable_name", ds.name)
    val = validate_timetable(active_tt)
    cost = calculate_cost(active_tt)

    # Status Banner
    st.markdown(f"### Schedule: **{tt_name}**")
    sb1, sb2, sb3, sb4 = st.columns(4)
    sb1.metric("Status", "Conflict-Free \u2705" if val.is_valid else f"Conflicts: {val.hard_violation_count} \u26a0")
    sb2.metric("Total Cost", cost.total_cost)
    sb3.metric("Soft Penalties (S1/S2/S3)", f"{cost.s1_gaps} / {cost.s2_repeats} / {cost.s3_non_consecutive}")
    sb4.metric("Placed Sessions", len(active_tt.get_placed_sessions()))

    if st.session_state.get("unsaved_edits"):
        st.warning("\u26a0 You have unsaved manual edits on this schedule. Click **Save Changes** below to persist them.")

    # Section 1: Timetable Grids (Class, Teacher, Room)
    st.markdown("---")
    st.header("Schedule Grids")

    v_choice = st.selectbox("View Timetable By", ["Class", "Teacher", "Room"], key="grid_view_mode")
    grids = _make_grid(active_tt, v_choice.lower())

    if not grids:
        st.info("Timetable is empty — no sessions to display.")
    else:
        eid_list = sorted(grids.keys())
        if v_choice == "Class":
            labels = {e: ds.classes[e].name for e in eid_list if e in ds.classes}
        elif v_choice == "Teacher":
            labels = {e: ds.teachers[e].name for e in eid_list if e in ds.teachers}
        else:
            labels = {e: ds.rooms[e].name for e in eid_list if e in ds.rooms}

        sel_eid = st.selectbox(
            f"Select {v_choice}",
            eid_list,
            format_func=lambda x: f"{labels.get(x, x)} ({x})",
            key="grid_entity_select",
        )

        grid_df = grids[sel_eid]
        sel_name = labels.get(sel_eid, sel_eid)
        _render_timetable_grid(active_tt, v_choice, sel_eid, sel_name, grid_df)

    # Section 2: Manual Schedule Editor
    st.markdown("---")
    st.header("Manual Schedule Editor")
    st.markdown("Move any session to another day, period, or room. Violations and cost are immediately recomputed.")

    placed_sessions = active_tt.get_placed_sessions()
    if not placed_sessions:
        st.warning("No placed sessions available to edit.")
    else:
        sess_lookup = {var.session_id: (var, plc) for var, plc in placed_sessions}
        sess_labels = {
            sid: (
                f"{ds.classes.get(v.class_id, v).name if hasattr(ds.classes.get(v.class_id, None), 'name') else v.class_id} — "
                f"{ds.subjects.get(v.subject_id, v).name if hasattr(ds.subjects.get(v.subject_id, None), 'name') else v.subject_id} "
                f"(s{v.session_no}, {v.duration}p) [{ds.teachers.get(v.teacher_id, v).name if hasattr(ds.teachers.get(v.teacher_id, None), 'name') else v.teacher_id}]"
            )
            for sid, (v, _) in sess_lookup.items()
        }

        sel_sid = st.selectbox(
            "Select Session to Move",
            sorted(sess_lookup.keys()),
            format_func=lambda s: sess_labels.get(s, s),
            key="edit_session_select",
        )
        sel_var, sel_plc = sess_lookup[sel_sid]

        cur_day_str = ds.time_structure.days[sel_plc.day] if 0 <= sel_plc.day < len(ds.time_structure.days) else f"Day {sel_plc.day}"
        cur_room_str = ds.rooms[sel_plc.room_id].name if sel_plc.room_id in ds.rooms else sel_plc.room_id
        st.markdown(f"**Current Placement:** `{cur_day_str}` | `P{sel_plc.start_period + 1}` | `{cur_room_str}`")

        # Placement Selection
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            new_day = st.selectbox("New Day", range(len(ds.time_structure.days)), format_func=lambda d: ds.time_structure.days[d], index=sel_plc.day, key="edit_new_day")
        with e_col2:
            max_p = ds.time_structure.periods_per_day - sel_var.duration
            new_period = st.selectbox("New Start Period", range(max_p + 1), format_func=lambda p: f"P{p + 1}", index=min(sel_plc.start_period, max_p), key="edit_new_period")
        with e_col3:
            room_keys = sorted(ds.rooms.keys())
            def_idx = room_keys.index(sel_plc.room_id) if sel_plc.room_id in room_keys else 0
            new_room = st.selectbox("New Room", room_keys, format_func=lambda r: f"{ds.rooms[r].name} ({ds.rooms[r].type}, cap={ds.rooms[r].capacity})", index=def_idx, key="edit_new_room")

        chosen_plc = Placement(day=new_day, start_period=new_period, room_id=new_room)

        act1, act2, act3 = st.columns(3)
        with act1:
            if st.button("⚡ Apply Move", type="primary", use_container_width=True):
                # Update assignments in place on active timetable
                new_asgn = dict(active_tt.assignments)
                new_asgn[sel_var] = chosen_plc
                active_tt.assignments = new_asgn
                st.session_state["unsaved_edits"] = True

                # Immediate validation & feedback
                new_val = validate_timetable(active_tt)
                new_cost = calculate_cost(active_tt)

                if new_val.is_valid:
                    st.success(f"Move applied cleanly! Hard violations: 0, Total cost: {new_cost.total_cost}")
                else:
                    st.error(f"Move introduces {new_val.hard_violation_count} hard violations (Conflicts Detected).")
                    _show_violations(new_val.violations)
                st.rerun()

        with act2:
            if st.button("💾 Save Changes to History", use_container_width=True):
                active_id = st.session_state.get("active_timetable_id")
                if active_id:
                    rec = _timetable_store.get(active_id)
                    rec = TimetableRecord.create_from_timetable(
                        timetable=active_tt,
                        name=rec.name,
                        algorithm=rec.algorithm,
                        timetable_id=rec.id,
                        version=rec.version + 1,
                    )
                    _timetable_store.save(rec)
                    st.session_state["unsaved_edits"] = False
                    st.success(f"Saved changes to '{rec.name}' (v{rec.version})!")
                    st.rerun()
                else:
                    # New save
                    rec = TimetableRecord.create_from_timetable(active_tt, name=tt_name)
                    _timetable_store.save(rec)
                    st.session_state["active_timetable_id"] = rec.id
                    st.session_state["unsaved_edits"] = False
                    st.success(f"Saved as new record '{rec.name}' in My Timetables!")
                    st.rerun()

        with act3:
            if st.button("↩️ Revert Changes", use_container_width=True):
                active_id = st.session_state.get("active_timetable_id")
                if active_id:
                    try:
                        rec = _timetable_store.get(active_id)
                        _set_active_timetable(rec.get_timetable(), name=rec.name, timetable_id=rec.id)
                        st.success("Reverted to last saved version.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error reverting: {exc}")
                else:
                    st.warning("No saved version to revert to.")


# ===========================================================================
# 5. EXPERIMENTS & BENCHMARKS
# ===========================================================================
elif selected_nav == "Experiments & Benchmarks":
    st.title("\U0001f52c Experiments & Benchmarks")
    st.markdown(
        "Empirical evaluation of all 5 CSP algorithms across standardized benchmarks. "
        "Benchmark runs are isolated from your personal timetable history."
    )

    summary_csv = RESULTS_DIR / "experiment_summary.csv"
    runs_csv    = RESULTS_DIR / "experiment_runs.csv"
    plot_png    = RESULTS_DIR / "min_conflicts_cost_vs_iterations.png"
    meta_json   = RESULTS_DIR / "experiment_metadata.json"

    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            st.caption(
                f"Generated: {meta.get('timestamp', '??')} | "
                f"Total Runs: {meta.get('run_count', '??')} | "
                f"Pandas {meta.get('tool_versions', {}).get('pandas', '?')}"
            )
        except Exception:
            pass

    if summary_csv.exists():
        st.subheader("Benchmark Summary Statistics")
        summary_df = pd.read_csv(summary_csv)
        summary_df["success_rate"] = (summary_df["success_rate"] * 100).round(1).astype(str) + "%"
        summary_df["avg_runtime_seconds"] = summary_df["avg_runtime_seconds"].round(4)
        summary_df["avg_total_cost"] = summary_df["avg_total_cost"].round(1)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("No summary results found. Run `python experiments/run_experiments.py`.")

    if plot_png.exists():
        st.subheader("Min-Conflicts: Cost vs. Iterations")
        st.image(str(plot_png), use_container_width=True)

    if runs_csv.exists():
        with st.expander("All Individual Benchmark Runs (28 Trials)"):
            st.dataframe(pd.read_csv(runs_csv), use_container_width=True)


# ===========================================================================
# 6. EXPORT
# ===========================================================================
elif selected_nav == "Export":
    st.title("\U0001f4e5 Export Timetable")

    active_tt = st.session_state.get("active_timetable")

    if active_tt is None:
        st.warning("No active timetable available to export. Please generate or open one first.")
        st.stop()

    ds = active_tt.dataset
    tt_name = st.session_state.get("active_timetable_name", ds.name)
    val = validate_timetable(active_tt)

    st.markdown(f"Exporting active schedule: **{tt_name}** ({len(active_tt.get_placed_sessions())} scheduled sessions).")
    if not val.is_valid:
        st.warning(f"\u26a0 Note: This schedule currently has {val.hard_violation_count} hard violations (manual edits with conflicts).")

    raw_df = _sessions_df(active_tt, algorithm=st.session_state.get("active_generation_result", {}).get("algorithm", "") if isinstance(st.session_state.get("active_generation_result"), dict) else getattr(st.session_state.get("active_generation_result"), "algorithm", ""))

    export_cols = [
        "class_name", "subject_name", "session_id", "teacher_name",
        "day", "start_period", "duration", "room_name", "session_type",
    ]
    display_df = (
        raw_df[export_cols]
        .rename(columns={
            "class_name":   "Class",
            "subject_name": "Subject",
            "session_id":   "Session ID",
            "teacher_name": "Teacher",
            "day":          "Day",
            "start_period": "Start Period",
            "duration":     "Duration",
            "room_name":    "Room",
            "session_type": "Type",
        })
        .sort_values(["Class", "Day", "Start Period"])
        .reset_index(drop=True)
    )

    st.dataframe(display_df, use_container_width=True)

    # Export Buttons
    e1, e2, e3 = st.columns(3)

    with e1:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "\u2b07\ufe0f Download CSV",
            csv_bytes,
            "timetable.csv",
            "text/csv",
            use_container_width=True,
        )

    with e2:
        try:
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
                display_df.to_excel(writer, index=False, sheet_name="Timetable")
            xls_buf.seek(0)
            st.download_button(
                "\u2b07\ufe0f Download Excel",
                xls_buf.read(),
                "timetable.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel export unavailable: {exc}")

    with e3:
        ds_json = json.dumps(ds.to_dict(), indent=2).encode("utf-8")
        st.download_button(
            "\u2b07\ufe0f Download Dataset JSON",
            ds_json,
            "dataset.json",
            "application/json",
            use_container_width=True,
        )
