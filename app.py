"""AI Timetable Scheduler - Streamlit Application.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import time

import pandas as pd
import streamlit as st

# -- ensure project root is importable --
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.models import TimetableDataset, load_dataset_from_file, check_dataset_feasibility
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
        "Static variable ordering. Exhaustive search until a solution or budget limit."
    ),
    "MRV Backtracking": (
        "**Minimum Remaining Values** - picks the most constrained unassigned variable."
    ),
    "MRV + Degree Heuristic": (
        "MRV with **Degree** tie-breaking: prefers variables constrained by the most "
        "unassigned neighbours."
    ),
    "MRV + Degree + Forward Checking": (
        "MRV + Degree + **Forward Checking**: prunes domains after each assignment; "
        "backtracks on wipeout."
    ),
    "Min-Conflicts": (
        "Local search. Starts with a random complete assignment; iteratively repairs "
        "the most conflicted variable."
    ),
}

RESULTS_DIR = _ROOT / "experiments" / "results"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for _k in ("timetable", "result", "dataset", "edited_tt"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> TimetableDataset | None:
    try:
        return load_dataset_from_file(pathlib.Path(path))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return None


def _make_grid(timetable: Timetable, group_by: str) -> dict[str, pd.DataFrame]:
    """Build {entity_id: DataFrame} grid.  Rows=periods, Cols=days."""
    ds  = timetable.dataset
    ts  = ds.time_structure
    days = ts.days
    periods = list(range(ts.periods_per_day))
    period_labels = [f"P{p + 1}" for p in periods]

    grids: dict[str, dict[tuple[int, int], str]] = {}

    def _add(entity: str, d: int, p: int, text: str) -> None:
        grids.setdefault(entity, {})
        prev = grids[entity].get((d, p), "")
        grids[entity][(d, p)] = (prev + "\n" + text).strip() if prev else text

    for var, plc in timetable.get_placed_sessions():
        subj_name    = ds.subjects[var.subject_id].name
        class_name   = ds.classes[var.class_id].name
        teacher_name = ds.teachers[var.teacher_id].name
        room_name    = ds.rooms[plc.room_id].name

        occupied = plc.get_occupied_slots(var.duration)
        first_p  = min(p for _, p in occupied)

        for d, p in occupied:
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
            else:   # room
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


def _sessions_df(timetable: Timetable) -> pd.DataFrame:
    """Flat DataFrame of all placed sessions for CSV/Excel export."""
    ds = timetable.dataset
    rows = []
    for var, plc in timetable.get_placed_sessions():
        subj = ds.subjects[var.subject_id]
        rows.append({
            "session_id":   var.session_id,
            "class_id":     var.class_id,
            "class_name":   ds.classes[var.class_id].name,
            "subject_id":   var.subject_id,
            "subject_name": subj.name,
            "session_no":   var.session_no,
            "teacher_id":   var.teacher_id,
            "teacher_name": ds.teachers[var.teacher_id].name,
            "day":          ds.time_structure.days[plc.day],
            "start_period": plc.start_period + 1,
            "duration":     var.duration,
            "room_id":      plc.room_id,
            "room_name":    ds.rooms[plc.room_id].name,
            "session_type": var.session_type,
        })
    return pd.DataFrame(rows)


def _show_violations(violations, max_show: int = 10) -> None:
    for v in violations[:max_show]:
        st.warning(f"**{v.constraint_id}**: {v.message}")
    if len(violations) > max_show:
        st.caption(f"... and {len(violations) - max_show} more.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001f4c5 Timetable Scheduler")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "\U0001f3e0 Overview & Dataset",
            "\u2699\ufe0f Generate Timetable",
            "\U0001f4ca View Schedule",
            "\u270f\ufe0f Edit Schedule",
            "\U0001f4e5 Export",
            "\U0001f52c Experiment Results",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("AI Timetable Scheduler")
    st.caption("CSP + Min-Conflicts")


# ===========================================================================
# PAGE 1 - Overview & Dataset
# ===========================================================================
if page == "\U0001f3e0 Overview & Dataset":
    st.title("\U0001f3e0 AI-Based Timetable Scheduler")
    st.markdown(
        "Generate conflict-free timetables using **Constraint Satisfaction Problem (CSP)** "
        "techniques. Select a dataset below, inspect its structure, then move to "
        "**Generate Timetable**."
    )

    st.header("Dataset Selection")
    ds_choice = st.selectbox("Choose a dataset", list(DATASETS.keys()))
    ds_path   = DATASETS[ds_choice]

    ds = _load_dataset(ds_path)
    if ds is None:
        st.stop()
    st.session_state.dataset = ds

    st.success(f"Loaded **{ds.name}** from `{pathlib.Path(ds_path).name}`")

    # Metrics
    st.header("Dataset Summary")
    ts = ds.time_structure
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Classes",          len(ds.classes))
    c2.metric("Teachers",         len(ds.teachers))
    c3.metric("Rooms",            len(ds.rooms))
    c4.metric("Subjects",         len(ds.subjects))
    c5.metric("Days x Periods",   f"{len(ts.days)} x {ts.periods_per_day}")

    with st.expander("Time Structure"):
        st.markdown(f"**Days:** {', '.join(ts.days)}")
        st.markdown(f"**Periods per day:** {ts.periods_per_day}")
        if ts.blocked_periods:
            bp_txt = ", ".join(f"{ts.days[d]} P{p+1}" for d, p in ts.blocked_periods)
            st.markdown(f"**Blocked periods:** {bp_txt}")
        else:
            st.markdown("**Blocked periods:** None")

    with st.expander("Classes"):
        for cid, cg in sorted(ds.classes.items()):
            subj_txt = ", ".join(
                f"{ds.subjects[r.subject_id].name}"
                f"({r.weekly_periods or ds.subjects[r.subject_id].weekly_periods}p)"
                for r in cg.subjects if r.subject_id in ds.subjects
            )
            st.markdown(
                f"**{cg.name}** ({cid}) — Strength: {cg.strength}, "
                f"Lunch: P{cg.lunch_slot + 1} — Subjects: {subj_txt}"
            )

    with st.expander("Teachers"):
        for tid, t in sorted(ds.teachers.items()):
            snames = [ds.subjects[s].name for s in t.subject_ids if s in ds.subjects]
            st.markdown(
                f"**{t.name}** ({tid}) — Teaches: {', '.join(snames)} "
                f"— Max load: {t.max_weekly_load}p/wk "
                f"— Unavailable: {len(t.unavailable_slots)} slots"
            )

    with st.expander("Rooms"):
        for rid, r in sorted(ds.rooms.items()):
            st.markdown(
                f"**{r.name}** ({rid}) — {r.type.title()} — Capacity: {r.capacity}"
            )

    # Feasibility check
    st.header("Feasibility Check")
    try:
        issues = check_dataset_feasibility(ds)
        if issues:
            st.warning(f"{len(issues)} issue(s) detected:")
            for iss in issues:
                st.warning(f"\u26a0 {iss}")
        else:
            st.success("\u2705 Dataset passes all structural and feasibility checks.")
    except Exception as exc:
        st.error(f"Validation error: {exc}")


# ===========================================================================
# PAGE 2 - Generate Timetable
# ===========================================================================
elif page == "\u2699\ufe0f Generate Timetable":
    st.title("\u2699\ufe0f Generate Timetable")

    ds = st.session_state.dataset
    if ds is None:
        st.warning("Please select a dataset on the **Overview & Dataset** page first.")
        st.stop()

    st.info(f"Active dataset: **{ds.name}**")

    algo_name = st.selectbox("Algorithm", list(ALGORITHMS.keys()))

    with st.expander("Algorithm description"):
        st.info(ALGO_DESCRIPTIONS[algo_name])

    with st.expander("Solver configuration", expanded=True):
        if algo_name == "Min-Conflicts":
            cc1, cc2, cc3 = st.columns(3)
            max_iter    = cc1.number_input("Max iterations",  100, 100_000, 5000, 500)
            time_limit  = cc2.number_input("Time limit (s)",    1,    300,    30)
            random_seed = cc3.number_input("Random seed",       0, 999_999,   42)
            mc_strategy = st.selectbox("Initial strategy", ["random", "greedy"])
        else:
            bc1, bc2 = st.columns(2)
            bt_nodes      = bc1.number_input("Max nodes",       1_000, 10_000_000, 100_000, 10_000)
            bt_time_limit = bc2.number_input("Time limit (s)",      1,        300,      30)

    if st.button("\U0001f680 Generate Timetable", type="primary", use_container_width=True):
        st.session_state.timetable = None
        st.session_state.result    = None
        st.session_state.edited_tt = None

        with st.spinner(f"Running {algo_name} ..."):
            try:
                if algo_name == "Min-Conflicts":
                    result = solve_min_conflicts(
                        dataset=ds,
                        max_iterations=int(max_iter),
                        time_limit_seconds=float(time_limit),
                        random_seed=int(random_seed),
                        initial_strategy=mc_strategy,
                        record_cost_history=True,
                    )
                else:
                    result = solve_backtracking(
                        dataset=ds,
                        max_nodes=int(bt_nodes),
                        time_limit_seconds=float(bt_time_limit),
                        **ALGORITHMS[algo_name],
                    )
            except Exception as exc:
                st.error(f"Solver error: {exc}")
                st.stop()

        st.session_state.result = result
        if result.success and result.timetable is not None:
            st.session_state.timetable = result.timetable

    # Display result
    result = st.session_state.result
    if result is not None:
        st.markdown("---")
        st.subheader("Result")

        if result.success:
            st.success("\u2705 Solved - timetable generated successfully.")
        else:
            status_txt = result.status.replace("_", " ").title()
            st.error(f"\u274c {status_txt} - no valid timetable found.")
            if getattr(result, "failure_reason", None):
                st.warning(f"Reason: {result.failure_reason}")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Algorithm",       algo_name)
        rc2.metric("Status",          result.status)
        rc3.metric("Runtime",         f"{result.runtime_seconds:.3f}s")
        hv = getattr(result, "hard_violations",
                     getattr(result, "final_hard_violations", 0))
        rc4.metric("Hard violations", hv)

        if result.success:
            # Search metrics (show only applicable ones)
            search_data = {
                "Nodes explored":        getattr(result, "nodes_explored",                None),
                "Backtracks":            getattr(result, "backtracks",                    None),
                "Assignments tried":     getattr(result, "assignments_tried",             None),
                "Max depth":             getattr(result, "max_search_depth",              None),
                "Iterations":            getattr(result, "iterations",                    None),
                "Conflicted selections": getattr(result, "conflicted_variable_selections", None),
                "Candidates evaluated":  getattr(result, "candidate_placements_evaluated", None),
                "Initial conflicts":     getattr(result, "initial_hard_conflicts",         None),
            }
            valid_m = [(k, v) for k, v in search_data.items() if v is not None]
            if valid_m:
                st.subheader("Search Metrics")
                mcols = st.columns(min(4, len(valid_m)))
                for i, (k, v) in enumerate(valid_m):
                    mcols[i % 4].metric(k, v)

            # Cost breakdown
            st.subheader("Cost Breakdown")
            soft = getattr(result, "soft_penalties", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("S1 - Gaps",             soft.get("s1_gaps", 0))
            sc2.metric("S2 - Same-day repeats", soft.get("s2_repeats", 0))
            sc3.metric("S3 - Non-consec labs",  soft.get("s3_non_consecutive", 0))
            sc4.metric("Total cost",             result.total_cost)

            # Cost history plot (Min-Conflicts only)
            ch = getattr(result, "cost_history", None)
            if ch and len(ch) > 1:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 3))
                iters, costs = zip(*ch)
                ax.plot(iters, costs, linewidth=1.5, color="#4C72B0")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Total Cost")
                ax.set_title("Min-Conflicts: Cost vs. Iterations")
                ax.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig)
                plt.close(fig)


# ===========================================================================
# PAGE 3 - View Schedule
# ===========================================================================
elif page == "\U0001f4ca View Schedule":
    st.title("\U0001f4ca View Schedule")

    tt = st.session_state.edited_tt or st.session_state.timetable
    if tt is None:
        st.info("No timetable available. Please generate one first.")
        st.stop()

    ds          = tt.dataset
    total_vars  = len(generate_session_variables(ds))
    placed_cnt  = len(tt.get_placed_sessions())
    st.info(f"{placed_cnt}/{total_vars} sessions placed.")

    view_choice = st.selectbox("View by", ["Class", "Teacher", "Room"])
    grids       = _make_grid(tt, view_choice.lower())

    if not grids:
        st.warning("Timetable is empty - no sessions placed.")
        st.stop()

    eid_list = sorted(grids.keys())
    if view_choice == "Class":
        labels = {e: ds.classes[e].name   for e in eid_list}
    elif view_choice == "Teacher":
        labels = {e: ds.teachers[e].name  for e in eid_list}
    else:
        labels = {e: ds.rooms[e].name     for e in eid_list}

    sel_eid = st.selectbox(
        f"Select {view_choice}",
        eid_list,
        format_func=lambda x: f"{labels[x]} ({x})",
    )

    df = grids[sel_eid]

    def _colour(v):
        return (
            "background-color:#d4edda; font-size:0.78rem;"
            if v else
            "background-color:#f8f9fa; color:#bbb;"
        )

    st.dataframe(
        df.style.applymap(_colour),
        use_container_width=True,
        height=min(45 + 38 * len(df), 520),
    )
    st.caption("Multi-period labs show '... (cont.)' in continuation rows.")


# ===========================================================================
# PAGE 4 - Edit Schedule
# ===========================================================================
elif page == "\u270f\ufe0f Edit Schedule":
    st.title("\u270f\ufe0f Manual Schedule Editor")

    base_tt = st.session_state.timetable
    if base_tt is None:
        st.info("No timetable available. Please generate one first.")
        st.stop()

    ds = base_tt.dataset
    ts = ds.time_structure

    # Initialise editable copy
    if st.session_state.edited_tt is None:
        st.session_state.edited_tt = Timetable(
            dataset=ds,
            assignments=dict(base_tt.assignments),
        )

    current_tt = st.session_state.edited_tt

    # Current cost banner
    cost_now = calculate_cost(current_tt)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Hard violations", cost_now.hard_violations)
    b2.metric("S1 gaps",         cost_now.s1_gaps)
    b3.metric("S2 repeats",      cost_now.s2_repeats)
    b4.metric("Total cost",      cost_now.total_cost)

    st.markdown("---")
    st.markdown("Select a session, choose a new placement, then apply.")

    placed = current_tt.get_placed_sessions()
    if not placed:
        st.warning("No placed sessions to edit.")
        st.stop()

    sess_map = {var.session_id: (var, plc) for var, plc in placed}
    sess_labels = {
        sid: (
            f"{ds.classes[v.class_id].name} - "
            f"{ds.subjects[v.subject_id].name} "
            f"(s{v.session_no})"
        )
        for sid, (v, _) in sess_map.items()
    }

    sel_sid      = st.selectbox(
        "Session to move",
        sorted(sess_map.keys()),
        format_func=lambda x: sess_labels[x],
    )
    sel_var, sel_plc = sess_map[sel_sid]

    st.markdown(
        f"**Current:** {ts.days[sel_plc.day]} | "
        f"P{sel_plc.start_period + 1} | "
        f"{ds.rooms[sel_plc.room_id].name}"
    )

    valid_domain = get_valid_domain_for_session(sel_var, ds)
    if not valid_domain:
        st.error("No valid placements exist for this session.")
        st.stop()

    domain_labels = [
        f"{ts.days[p.day]}  |  P{p.start_period + 1}  |  {ds.rooms[p.room_id].name}"
        for p in valid_domain
    ]
    chosen_idx = st.selectbox(
        "New placement",
        range(len(valid_domain)),
        format_func=lambda i: domain_labels[i],
    )
    chosen_plc = valid_domain[chosen_idx]

    apply_col, revert_col = st.columns(2)

    with apply_col:
        if st.button("\u2705 Apply Edit", type="primary"):
            new_asgn = dict(current_tt.assignments)
            new_asgn[sel_var] = chosen_plc
            candidate = Timetable(dataset=ds, assignments=new_asgn)
            val = validate_timetable(candidate)

            if val.is_valid:
                new_cost = calculate_cost(candidate)
                st.session_state.edited_tt = candidate
                st.success(
                    f"Edit applied. Hard violations: 0 | "
                    f"Total cost: {new_cost.total_cost}"
                )
                st.rerun()
            else:
                st.error(
                    f"Edit rejected - {val.hard_violation_count} hard violation(s). "
                    "Choose a different placement or revert."
                )
                _show_violations(val.violations)

    with revert_col:
        if st.button("\u21a9\ufe0f Revert All Edits"):
            st.session_state.edited_tt = None
            st.success("Reverted to the original generated timetable.")
            st.rerun()


# ===========================================================================
# PAGE 5 - Export
# ===========================================================================
elif page == "\U0001f4e5 Export":
    st.title("\U0001f4e5 Export Schedule")

    tt = st.session_state.edited_tt or st.session_state.timetable
    if tt is None:
        st.info("No timetable available. Please generate one first.")
        st.stop()

    raw_df = _sessions_df(tt)
    if raw_df.empty:
        st.warning("No sessions to export.")
        st.stop()

    export_cols = [
        "class_name", "subject_name", "session_id", "teacher_name",
        "day", "start_period", "duration", "room_name", "session_type",
    ]
    export_df = (
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

    st.dataframe(export_df, use_container_width=True)
    st.caption(f"Total sessions: {len(export_df)}")

    dl1, dl2 = st.columns(2)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    dl1.download_button(
        "\u2b07\ufe0f Download CSV",
        csv_bytes,
        "timetable.csv",
        "text/csv",
        use_container_width=True,
    )

    try:
        xls_buf = io.BytesIO()
        with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Timetable")
        xls_buf.seek(0)
        dl2.download_button(
            "\u2b07\ufe0f Download Excel",
            xls_buf.read(),
            "timetable.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as exc:
        dl2.error(f"Excel export unavailable: {exc}")


# ===========================================================================
# PAGE 6 - Experiment Results
# ===========================================================================
elif page == "\U0001f52c Experiment Results":
    st.title("\U0001f52c Experiment Results")

    summary_csv = RESULTS_DIR / "experiment_summary.csv"
    runs_csv    = RESULTS_DIR / "experiment_runs.csv"
    plot_png    = RESULTS_DIR / "min_conflicts_cost_vs_iterations.png"
    meta_json   = RESULTS_DIR / "experiment_metadata.json"

    if not summary_csv.exists():
        st.warning(
            "No experiment results found.\n\n"
            "Generate them by running:\n"
            "```\npython experiments/run_experiments.py\n```"
        )
        st.stop()

    if meta_json.exists():
        meta = json.loads(meta_json.read_text())
        st.caption(
            f"Generated: {meta.get('timestamp', '??')}  |  "
            f"Runs: {meta.get('run_count', '??')}  |  "
            f"Pandas {meta.get('tool_versions', {}).get('pandas', '?')}"
        )

    st.subheader("Summary Statistics")
    summary_df = pd.read_csv(summary_csv)
    summary_df["success_rate"] = (
        summary_df["success_rate"] * 100
    ).round(1).astype(str) + "%"
    summary_df["avg_runtime_seconds"] = summary_df["avg_runtime_seconds"].round(4)
    summary_df["avg_total_cost"]      = summary_df["avg_total_cost"].round(1)
    st.dataframe(summary_df, use_container_width=True)

    if runs_csv.exists():
        with st.expander("All individual runs"):
            runs_df = pd.read_csv(runs_csv)
            st.dataframe(runs_df, use_container_width=True)

    if plot_png.exists():
        st.subheader("Min-Conflicts: Cost vs. Iterations")
        st.image(str(plot_png), use_container_width=True)
    else:
        st.info(
            "Cost plot not yet generated. Run the experiment script to produce it."
        )
