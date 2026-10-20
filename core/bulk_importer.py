"""Bulk Timetable Setup: Importer, Parsers, Templates, and Availability Matrix.

This module provides high-speed bulk input capabilities for the AI Timetable Scheduler:
1. Multi-row tabular ingestion supporting TSV (copied from Excel/Google Sheets), CSV, and XLSX.
2. Robust header normalization (maps aliases like 'code', 'hours', 'cap' to canonical fields).
3. Row-level validation reporting exact errors per row while preserving valid entries.
4. Downloadable starter templates (individual CSV/XLSX and complete multi-sheet institution workbooks).
5. Two-way availability matrix conversion (Grid DataFrame <-> slot coordinate strings).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.custom_builder import format_slots_to_string, parse_slot_string


# ---------------------------------------------------------------------------
# Canonical schema definitions and header aliases
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, dict[str, list[str]]] = {
    "classes": {
        "class_id":   ["class_id", "id", "class_code", "batch_id", "code", "classid"],
        "class_name": ["class_name", "name", "batch", "class", "section", "classname"],
        "strength":   ["strength", "students", "student_count", "size", "capacity", "count"],
        "lunch_slot": ["lunch_slot", "lunch_period", "lunch", "lunchslot", "break_period"],
    },
    "teachers": {
        "teacher_id":         ["teacher_id", "id", "teacher_code", "faculty_id", "code", "teacherid"],
        "teacher_name":       ["teacher_name", "name", "faculty_name", "faculty", "instructor", "teachername"],
        "max_weekly_load":    ["max_weekly_load", "max_load", "load", "weekly_load", "maxload", "max_hours"],
        "qualified_subjects": ["qualified_subjects", "qualified", "subjects", "courses", "skills"],
        "unavailable_slots":  ["unavailable_slots", "unavailable", "blocked", "busy_slots", "unavail"],
        "preferred_slots":    ["preferred_slots", "preferred", "preferences", "pref_slots", "prefs"],
    },
    "subjects": {
        "subject_id":     ["subject_id", "id", "subject_code", "code", "course_code", "subjectid"],
        "subject_name":   ["subject_name", "name", "title", "subject_title", "course_name", "subjectname"],
        "class_id":       ["class_id", "class", "batch", "target_class", "section", "classid"],
        "teacher_id":     ["teacher_id", "teacher", "faculty", "instructor", "assigned_teacher", "teacherid"],
        "type":           ["type", "subject_type", "theory_lab", "session_type", "mode"],
        "weekly_periods": ["weekly_periods", "weekly_hours", "hours", "periods", "sessions", "weeklyhours"],
        "lab_duration":   ["lab_duration", "duration", "lab_length", "span", "labduration"],
    },
    "rooms": {
        "room_id":   ["room_id", "id", "room_code", "code", "number", "roomid"],
        "room_name": ["room_name", "name", "room_title", "venue", "roomname"],
        "capacity":  ["capacity", "cap", "seats", "size", "seating"],
        "type":      ["type", "room_type", "category", "venue_type", "roomtype"],
    },
}

CANONICAL_COLUMNS: dict[str, list[str]] = {
    "classes":  ["class_id", "class_name", "strength", "lunch_slot"],
    "teachers": ["teacher_id", "teacher_name", "max_weekly_load", "qualified_subjects", "unavailable_slots", "preferred_slots"],
    "subjects": ["subject_id", "subject_name", "class_id", "teacher_id", "type", "weekly_periods", "lab_duration"],
    "rooms":    ["room_id", "room_name", "capacity", "type"],
}

DEFAULT_FALLBACKS: dict[str, dict[str, Any]] = {
    "classes":  {"strength": 30, "lunch_slot": 4},
    "teachers": {"max_weekly_load": 14, "qualified_subjects": "", "unavailable_slots": "", "preferred_slots": ""},
    "subjects": {"type": "theory", "weekly_periods": 4, "lab_duration": 1},
    "rooms":    {"capacity": 40, "type": "classroom"},
}


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class RowError:
    row_number: int
    raw_data: dict[str, Any]
    errors: list[str]


@dataclass
class BulkValidationResult:
    is_valid: bool
    accepted_df: pd.DataFrame
    rejected_rows: list[RowError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary_message: str = ""


# ---------------------------------------------------------------------------
# Header Normalization
# ---------------------------------------------------------------------------

def normalize_column_name(col_name: str, entity_type: str) -> str:
    """Normalizes an arbitrary user-supplied header into a canonical schema column name."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "", str(col_name).strip().lower().replace(" ", "_"))
    aliases_map = COLUMN_ALIASES.get(entity_type, {})
    for canonical, aliases in aliases_map.items():
        if clean == canonical or clean in aliases:
            return canonical
    return clean


def normalize_dataframe_headers(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """Renames all columns in df to their canonical names if recognized."""
    new_cols = {col: normalize_column_name(col, entity_type) for col in df.columns}
    renamed = df.rename(columns=new_cols)
    return renamed


# ---------------------------------------------------------------------------
# Tabular Data Ingestion (TSV Paste, CSV, Excel)
# ---------------------------------------------------------------------------

def parse_pasted_tabular_text(text: str, entity_type: str) -> pd.DataFrame:
    """Parses raw text pasted from clipboard (TSV from Excel/Sheets or CSV/semicolon).
    
    Handles:
    - Tab-delimited rows with or without header.
    - Comma or semicolon-delimited rows.
    - Strips blank lines and whitespace.
    """
    if not text or not text.strip():
        return pd.DataFrame(columns=CANONICAL_COLUMNS.get(entity_type, []))

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame(columns=CANONICAL_COLUMNS.get(entity_type, []))

    first_line = lines[0]
    # Detect delimiter
    if "\t" in first_line:
        delimiter = "\t"
    elif ";" in first_line:
        delimiter = ";"
    else:
        delimiter = ","

    stream = io.StringIO("\n".join(lines))
    try:
        raw_df = pd.read_csv(stream, sep=delimiter, dtype=str, keep_default_na=False)
    except Exception:
        # Fallback to python csv reader if pandas parser encounters jagged lines
        stream = io.StringIO("\n".join(lines))
        reader = csv.reader(stream, delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return pd.DataFrame(columns=CANONICAL_COLUMNS.get(entity_type, []))
        raw_df = pd.DataFrame(rows[1:], columns=rows[0])

    # Check if first line was actually data rather than header
    norm_first = [normalize_column_name(c, entity_type) for c in raw_df.columns]
    canonical_set = set(CANONICAL_COLUMNS.get(entity_type, []))
    has_header = any(col in canonical_set for col in norm_first)

    if not has_header:
        # Re-read without header
        stream = io.StringIO("\n".join(lines))
        canonical_cols = CANONICAL_COLUMNS.get(entity_type, [])
        col_names = canonical_cols[:len(first_line.split(delimiter))]
        raw_df = pd.read_csv(stream, sep=delimiter, names=col_names, dtype=str, keep_default_na=False)

    df_clean = normalize_dataframe_headers(raw_df, entity_type)
    return df_clean


def parse_uploaded_file(file_bytes: bytes, filename: str, entity_type: str) -> pd.DataFrame:
    """Parses an uploaded CSV or XLSX file into a DataFrame with normalized headers."""
    fn = filename.lower()
    if fn.endswith(".xlsx") or fn.endswith(".xls"):
        # If excel, check sheet names matching entity_type or use first sheet
        wb = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, dtype=str)
        sheet_match = None
        for sname in wb.keys():
            if entity_type.lower() in sname.lower():
                sheet_match = sname
                break
        df = wb[sheet_match] if sheet_match else list(wb.values())[0]
    else:
        # Treat as CSV/TSV
        raw_str = file_bytes.decode("utf-8", errors="replace")
        return parse_pasted_tabular_text(raw_str, entity_type)

    df = df.fillna("")
    return normalize_dataframe_headers(df, entity_type)


def parse_complete_workbook(file_bytes: bytes) -> tuple[dict[str, Any], list[str]]:
    """Imports a multi-sheet institution workbook (.xlsx) containing all tables.
    
    Looks for sheets matching: 'Config'/'TimeGrid', 'Classes', 'Teachers', 'Subjects', 'Rooms'.
    Returns (custom_data_dict, list_of_import_messages).
    """
    wb = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, dtype=str)
    messages: list[str] = []

    sheet_keys = {k.strip().lower(): k for k in wb.keys()}

    # 1. Config
    config_dict: dict[str, Any] = {
        "name": "Imported College Timetable",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "periods_per_day": 6,
        "default_lunch_slot": 4,
        "blocked_periods": "",
    }
    for alias in ["config", "timegrid", "time_structure", "settings"]:
        if alias in sheet_keys:
            c_sheet = wb[sheet_keys[alias]].fillna("")
            if not c_sheet.empty:
                for _, row in c_sheet.iterrows():
                    k = str(row.iloc[0]).strip().lower() if len(row) > 0 else ""
                    v = str(row.iloc[1]).strip() if len(row) > 1 else ""
                    if "institution" in k or "name" in k:
                        config_dict["name"] = v or config_dict["name"]
                    elif "working" in k or "days" in k:
                        if v:
                            config_dict["days"] = [d.strip() for d in v.split(",") if d.strip()]
                    elif "periods_per_day" in k or "periods per day" in k or k == "periods":
                        try:
                            config_dict["periods_per_day"] = int(v)
                        except ValueError:
                            pass
                    elif "lunch" in k:
                        try:
                            config_dict["default_lunch_slot"] = int(v)
                        except ValueError:
                            pass
                    elif "blocked" in k:
                        config_dict["blocked_periods"] = v
            messages.append(f"Loaded institution time structure from sheet '{sheet_keys[alias]}'.")
            break

    # Helper to extract and normalize sheet
    def _extract_sheet(entity: str) -> pd.DataFrame:
        for s_alias in [entity, entity[:-1]]:
            if s_alias in sheet_keys:
                df = wb[sheet_keys[s_alias]].fillna("")
                messages.append(f"Loaded {len(df)} rows from sheet '{sheet_keys[s_alias]}'.")
                return normalize_dataframe_headers(df, entity)
        return pd.DataFrame(columns=CANONICAL_COLUMNS[entity])

    custom_data: dict[str, Any] = {
        "config": config_dict,
        "classes": _extract_sheet("classes"),
        "teachers": _extract_sheet("teachers"),
        "subjects": _extract_sheet("subjects"),
        "rooms": _extract_sheet("rooms"),
    }
    return custom_data, messages


# ---------------------------------------------------------------------------
# Row-Level Validation & Sanitation
# ---------------------------------------------------------------------------

def validate_bulk_rows(
    df: pd.DataFrame,
    entity_type: str,
    known_classes: set[str] | None = None,
    known_teachers: set[str] | None = None,
) -> BulkValidationResult:
    """Performs rigorous row-level validation and schema sanitation.
    
    Returns:
    - accepted_df: DataFrame of clean, parsed rows with correct types.
    - rejected_rows: list of RowError containing 1-indexed row number, raw data, and errors.
    - warnings: non-fatal notices.
    """
    canon_cols = CANONICAL_COLUMNS.get(entity_type, [])
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[RowError] = []
    warnings: list[str] = []

    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_num = int(idx) + 1  # 1-indexed for human display
        row_dict = {col: str(row.get(col, "")).strip() for col in canon_cols}
        errors: list[str] = []

        if entity_type == "classes":
            cid = row_dict.get("class_id", "")
            cname = row_dict.get("class_name", "")
            if not cid:
                errors.append("Missing required 'class_id'")
            elif cid in seen_ids:
                errors.append(f"Duplicate class_id '{cid}'")
            else:
                seen_ids.add(cid)

            if not cname:
                row_dict["class_name"] = cid

            # Strength
            try:
                strength = int(row_dict.get("strength", 30))
                if strength <= 0:
                    errors.append("Student strength must be a positive integer (> 0)")
                row_dict["strength"] = max(1, strength)
            except ValueError:
                errors.append(f"Invalid numeric strength: '{row_dict.get('strength')}'")

            # Lunch slot
            try:
                lunch = int(row_dict.get("lunch_slot", 4))
                if lunch <= 0:
                    errors.append("Lunch period must be a positive period number (>= 1)")
                row_dict["lunch_slot"] = max(1, lunch)
            except ValueError:
                errors.append(f"Invalid numeric lunch slot: '{row_dict.get('lunch_slot')}'")

        elif entity_type == "teachers":
            tid = row_dict.get("teacher_id", "")
            tname = row_dict.get("teacher_name", "")
            if not tid:
                errors.append("Missing required 'teacher_id'")
            elif tid in seen_ids:
                errors.append(f"Duplicate teacher_id '{tid}'")
            else:
                seen_ids.add(tid)

            if not tname:
                row_dict["teacher_name"] = tid

            # Max load
            try:
                load = int(row_dict.get("max_weekly_load", 14))
                if load <= 0:
                    errors.append("Max weekly load must be > 0")
                row_dict["max_weekly_load"] = max(1, load)
            except ValueError:
                errors.append(f"Invalid numeric max load: '{row_dict.get('max_weekly_load')}'")

        elif entity_type == "subjects":
            sid = row_dict.get("subject_id", "")
            sname = row_dict.get("subject_name", "")
            cid = row_dict.get("class_id", "")
            tid = row_dict.get("teacher_id", "")
            stype = row_dict.get("type", "theory").lower()

            if not sid:
                errors.append("Missing required 'subject_id'")
            elif sid in seen_ids:
                errors.append(f"Duplicate subject_id '{sid}' in subjects list")
            else:
                seen_ids.add(sid)

            if not sname:
                row_dict["subject_name"] = sid

            if not cid:
                errors.append("Missing required 'class_id'")
            elif known_classes and cid not in known_classes:
                errors.append(f"Unknown class '{cid}'. Must match a defined Class ID")

            if not tid:
                errors.append("Missing required 'teacher_id'")
            elif known_teachers and tid not in known_teachers:
                errors.append(f"Unknown teacher '{tid}'. Must match a defined Teacher ID")

            if stype not in ("theory", "lab"):
                errors.append(f"Invalid subject type '{stype}'. Must be 'theory' or 'lab'")
            row_dict["type"] = stype if stype in ("theory", "lab") else "theory"

            # Weekly periods
            try:
                hrs = int(row_dict.get("weekly_periods", 4))
                if hrs <= 0:
                    errors.append("Weekly periods must be > 0")
                row_dict["weekly_periods"] = max(1, hrs)
            except ValueError:
                errors.append(f"Invalid numeric weekly periods: '{row_dict.get('weekly_periods')}'")

            # Lab duration
            try:
                dur = int(row_dict.get("lab_duration", 1 if stype == "theory" else 2))
                if dur <= 0:
                    errors.append("Lab duration must be > 0")
                row_dict["lab_duration"] = max(1, dur)
            except ValueError:
                errors.append(f"Invalid lab duration: '{row_dict.get('lab_duration')}'")

        elif entity_type == "rooms":
            rid = row_dict.get("room_id", "")
            rname = row_dict.get("room_name", "")
            rtype = row_dict.get("type", "classroom").lower()

            if not rid:
                errors.append("Missing required 'room_id'")
            elif rid in seen_ids:
                errors.append(f"Duplicate room_id '{rid}'")
            else:
                seen_ids.add(rid)

            if not rname:
                row_dict["room_name"] = rid

            try:
                cap = int(row_dict.get("capacity", 40))
                if cap <= 0:
                    errors.append("Room capacity must be a positive integer (> 0)")
                row_dict["capacity"] = max(1, cap)
            except ValueError:
                errors.append(f"Invalid numeric capacity: '{row_dict.get('capacity')}'")

            if rtype not in ("classroom", "lab"):
                errors.append(f"Invalid room type '{rtype}'. Must be 'classroom' or 'lab'")
            row_dict["type"] = rtype if rtype in ("classroom", "lab") else "classroom"

        if errors:
            rejected_rows.append(RowError(row_number=row_num, raw_data=dict(row), errors=errors))
        else:
            accepted_rows.append(row_dict)

    accepted_df = pd.DataFrame(accepted_rows, columns=canon_cols) if accepted_rows else pd.DataFrame(columns=canon_cols)
    is_valid = len(rejected_rows) == 0

    summary = (
        f"Processed {len(df)} row(s): {len(accepted_rows)} accepted cleanly, "
        f"{len(rejected_rows)} rejected with errors."
    )
    return BulkValidationResult(
        is_valid=is_valid,
        accepted_df=accepted_df,
        rejected_rows=rejected_rows,
        warnings=warnings,
        summary_message=summary,
    )


# ---------------------------------------------------------------------------
# Template Generators
# ---------------------------------------------------------------------------

SAMPLE_ROWS: dict[str, list[dict[str, Any]]] = {
    "classes": [
        {"class_id": "CSE_A", "class_name": "CS Year 3 Sec A", "strength": 40, "lunch_slot": 4},
        {"class_id": "CSE_B", "class_name": "CS Year 3 Sec B", "strength": 35, "lunch_slot": 4},
        {"class_id": "ECE_A", "class_name": "EC Year 2", "strength": 30, "lunch_slot": 4},
    ],
    "teachers": [
        {"teacher_id": "T_TURING", "teacher_name": "Dr. Alan Turing", "max_weekly_load": 14, "qualified_subjects": "CS301, CS302", "unavailable_slots": "Friday P6", "preferred_slots": "Monday P1, Tuesday P2"},
        {"teacher_id": "T_ADA", "teacher_name": "Dr. Ada Lovelace", "max_weekly_load": 14, "qualified_subjects": "CS301, CS301_LAB", "unavailable_slots": "Monday P1", "preferred_slots": "Wednesday P3"},
        {"teacher_id": "T_SHANNON", "teacher_name": "Dr. Claude Shannon", "max_weekly_load": 12, "qualified_subjects": "EC201", "unavailable_slots": "", "preferred_slots": ""},
    ],
    "subjects": [
        {"subject_id": "CS301", "subject_name": "Operating Systems", "class_id": "CSE_A", "teacher_id": "T_TURING", "type": "theory", "weekly_periods": 4, "lab_duration": 1},
        {"subject_id": "CS301_LAB", "subject_name": "OS Laboratory", "class_id": "CSE_A", "teacher_id": "T_ADA", "type": "lab", "weekly_periods": 2, "lab_duration": 2},
        {"subject_id": "CS302", "subject_name": "Database Systems", "class_id": "CSE_B", "teacher_id": "T_TURING", "type": "theory", "weekly_periods": 4, "lab_duration": 1},
        {"subject_id": "EC201", "subject_name": "Signals & Systems", "class_id": "ECE_A", "teacher_id": "T_SHANNON", "type": "theory", "weekly_periods": 4, "lab_duration": 1},
    ],
    "rooms": [
        {"room_id": "LH_101", "room_name": "Lecture Hall 101", "capacity": 60, "type": "classroom"},
        {"room_id": "LH_102", "room_name": "Lecture Hall 102", "capacity": 50, "type": "classroom"},
        {"room_id": "LAB_CS", "room_name": "Computing Systems Lab", "capacity": 45, "type": "lab"},
    ],
}


def generate_csv_template(entity_type: str) -> str:
    """Generates a downloadable CSV template with canonical headers and sample rows."""
    samples = SAMPLE_ROWS.get(entity_type, [])
    df = pd.DataFrame(samples, columns=CANONICAL_COLUMNS.get(entity_type, []))
    return df.to_csv(index=False)


def generate_xlsx_template(entity_type: str) -> bytes:
    """Generates an Excel (.xlsx) single-entity template."""
    samples = SAMPLE_ROWS.get(entity_type, [])
    df = pd.DataFrame(samples, columns=CANONICAL_COLUMNS.get(entity_type, []))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=entity_type.capitalize())
    return buf.getvalue()


def generate_combined_workbook_template() -> bytes:
    """Generates a complete multi-sheet Excel (.xlsx) workbook for the entire institution."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: TimeGrid Config
        config_df = pd.DataFrame([
            {"Parameter": "Institution Name", "Value": "University Institute of Technology"},
            {"Parameter": "Working Days (comma-separated)", "Value": "Monday, Tuesday, Wednesday, Thursday, Friday"},
            {"Parameter": "Periods Per Day", "Value": "6"},
            {"Parameter": "Default Lunch Period (1-indexed)", "Value": "4"},
            {"Parameter": "Global Blocked Slots", "Value": "Monday P1"},
        ])
        config_df.to_excel(writer, index=False, sheet_name="Config")

        # Sheets 2-5: Classes, Teachers, Subjects, Rooms
        for entity in ["classes", "teachers", "subjects", "rooms"]:
            samples = SAMPLE_ROWS.get(entity, [])
            df = pd.DataFrame(samples, columns=CANONICAL_COLUMNS.get(entity, []))
            df.to_excel(writer, index=False, sheet_name=entity.capitalize())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Teacher Availability Matrix Helpers
# ---------------------------------------------------------------------------

def teacher_slots_to_matrix(
    unavailable_slots_str: str,
    preferred_slots_str: str,
    days: list[str],
    periods_per_day: int,
) -> pd.DataFrame:
    """Transforms a teacher's slot specification strings into an interactive matrix DataFrame.
    
    Rows: Period labels ('P1', 'P2', ...)
    Columns: Day names ('Monday', 'Tuesday', ...)
    Values: 'Available' (default), 'Unavailable', or 'Preferred'
    """
    unavail_slots = set(parse_slot_string(unavailable_slots_str, days, periods_per_day))
    pref_slots = set(parse_slot_string(preferred_slots_str, days, periods_per_day))

    data: dict[str, list[str]] = {}
    period_labels = [f"P{p + 1}" for p in range(periods_per_day)]

    for d_idx, day_name in enumerate(days):
        col_vals: list[str] = []
        for p_idx in range(periods_per_day):
            slot = (d_idx, p_idx)
            if slot in unavail_slots:
                col_vals.append("🔴 Unavailable")
            elif slot in pref_slots:
                col_vals.append("⭐ Preferred")
            else:
                col_vals.append("🟢 Available")
        data[day_name] = col_vals

    matrix_df = pd.DataFrame(data, index=period_labels)
    return matrix_df


def matrix_to_teacher_slots(
    matrix_df: pd.DataFrame,
    days: list[str],
) -> tuple[str, str]:
    """Converts an interactive matrix DataFrame back into unavailable and preferred slot strings."""
    unavail_slots: list[tuple[int, int]] = []
    pref_slots: list[tuple[int, int]] = []

    for d_idx, day_name in enumerate(days):
        if day_name not in matrix_df.columns:
            continue
        col_series = matrix_df[day_name]
        for p_idx, val in enumerate(col_series):
            s_val = str(val).strip().lower()
            if "unavail" in s_val or "🔴" in s_val:
                unavail_slots.append((d_idx, p_idx))
            elif "prefer" in s_val or "⭐" in s_val:
                pref_slots.append((d_idx, p_idx))

    unavail_str = format_slots_to_string(unavail_slots, days)
    pref_str = format_slots_to_string(pref_slots, days)
    return unavail_str, pref_str


def append_blank_rows(df: pd.DataFrame, entity_type: str, count: int = 5) -> pd.DataFrame:
    """Appends N blank / default template rows to an existing entity table."""
    canon_cols = CANONICAL_COLUMNS.get(entity_type, [])
    fallbacks = DEFAULT_FALLBACKS.get(entity_type, {})
    new_rows = []
    start_num = len(df) + 1
    for i in range(count):
        row_dict = {col: "" for col in canon_cols}
        row_dict.update(fallbacks)
        if entity_type == "classes":
            row_dict["class_id"] = f"CLASS_{start_num + i}"
            row_dict["class_name"] = f"Class {start_num + i}"
        elif entity_type == "teachers":
            row_dict["teacher_id"] = f"T_{start_num + i}"
            row_dict["teacher_name"] = f"Faculty {start_num + i}"
        elif entity_type == "subjects":
            row_dict["subject_id"] = f"SUB_{start_num + i}"
            row_dict["subject_name"] = f"Subject {start_num + i}"
        elif entity_type == "rooms":
            row_dict["room_id"] = f"R_{start_num + i}"
            row_dict["room_name"] = f"Room {start_num + i}"
        new_rows.append(row_dict)

    add_df = pd.DataFrame(new_rows, columns=canon_cols)
    if df.empty:
        return add_df
    return pd.concat([df, add_df], ignore_index=True)


def merge_imported_rows(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    entity_type: str,
    mode: str = "append",
) -> pd.DataFrame:
    """Merges new rows into existing table with 'append' or 'replace' mode.
    
    In 'append' mode, updates existing rows with matching ID or appends new rows.
    """
    id_col = CANONICAL_COLUMNS[entity_type][0]
    if mode == "replace" or existing_df.empty:
        return new_df.copy()

    ex_dict = {str(row[id_col]).strip(): row for _, row in existing_df.iterrows() if str(row[id_col]).strip()}
    for _, row in new_df.iterrows():
        nid = str(row[id_col]).strip()
        if nid:
            ex_dict[nid] = row

    canon_cols = CANONICAL_COLUMNS[entity_type]
    merged = pd.DataFrame(list(ex_dict.values()), columns=canon_cols)
    return merged

