"""Custom Timetable Builder, Parser, and Validation Module.

This module provides data structures and conversion logic for the "Create My Timetable"
custom dataset mode. It allows users to build timetables interactively, converts user
inputs into standard TimetableDataset objects, and performs multi-tier validation:
1. Field and syntax validation (required fields, positive numbers, valid formats).
2. Duplicate identifier checks (classes, teachers, subjects, rooms, days).
3. Relational integrity checks (references between classes, teachers, subjects, rooms).
4. Physical and workload constraints (room capacities, teacher max loads, available slots).
5. Dataset schema compliance (validate_dataset_schema) and mathematical feasibility (check_dataset_feasibility).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.models import (
    ClassGroup,
    ClassSubjectRequirement,
    Room,
    Subject,
    Teacher,
    TimeStructure,
    TimetableDataset,
    ValidationError,
    check_dataset_feasibility,
    validate_dataset_schema,
)


@dataclass
class CustomValidationResult:
    """Outcome of custom timetable validation.
    
    Attributes:
        is_valid: True iff all syntactic, relational, and schema checks pass.
        errors: Actionable validation error messages explaining invalid input.
        is_feasible: True iff the dataset has no mathematical capacity bottlenecks.
        warnings: Warnings indicating potential feasibility or capacity issues.
        dataset: The constructed TimetableDataset if is_valid is True, else None.
    """
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    is_feasible: bool = True
    warnings: list[str] = field(default_factory=list)
    dataset: TimetableDataset | None = None


def parse_slot_string(
    text: Any, days: list[str], periods_per_day: int
) -> list[tuple[int, int]]:
    """Parses slot specifications from user input into a list of (day, period) tuples.
    
    Supported formats (comma/semicolon/newline-delimited):
    - "Monday P1, Tuesday P3" or "Mon P1, Tue P3"
    - "Monday:1, Tuesday:3"
    - "(0, 0), (1, 2)" or "[0, 0], [1, 2]" or "0:0, 1:2"
    - "Monday 1, Friday 5"
    
    Note: Period numbers like "P1" or "1" (after a day name) are interpreted as 1-indexed
    period numbers (P1 -> 0, P2 -> 1). Direct numeric coordinates (d, p) without a day
    name are interpreted as 0-indexed grid coordinates.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        # Already parsed or list of slots
        result: list[tuple[int, int]] = []
        for item in text:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                d, p = int(item[0]), int(item[1])
                if 0 <= d < len(days) and 0 <= p < periods_per_day:
                    result.append((d, p))
        return sorted(set(result))

    s = str(text).strip()
    if not s or s.lower() in ("none", "nil", "[]", "()", "-"):
        return []

    # Map day names (full and 3-letter prefix) to day index
    day_map: dict[str, int] = {}
    for idx, d_name in enumerate(days):
        day_map[d_name.strip().lower()] = idx
        if len(d_name.strip()) >= 3:
            day_map[d_name.strip()[:3].lower()] = idx

    # Regex pattern to match:
    # 1. Bracketed/parenthesized coordinates like (0, 1), [0, 1], (0:1)
    # 2. Named days like "Monday P1", "Mon:1", "Monday 2"
    # 3. Colon/space separated coordinates like "0:1" or "0 1"
    token_pattern = re.compile(
        r"[\(\[]\s*\d+\s*[,:]\s*\d+\s*[\)\]]|"
        r"[A-Za-z]+\s*[:\-]?\s*[Pp]?\s*\d+|"
        r"\b\d+\s*[:\s]\s*\d+\b"
    )

    matches = token_pattern.findall(s)
    # Check if there are unrecognized characters beyond delimiters
    delims_removed = token_pattern.sub("", s)
    if re.search(r"[A-Za-z0-9]", delims_removed):
        # Found extra alphanumeric text that didn't match any valid slot format
        bad_token = delims_removed.strip(" ,;\n\t")
        raise ValidationError(
            f"Cannot parse slot '{bad_token}'. Expected format like 'Monday P1' or '(0, 0)'."
        )

    slots: set[tuple[int, int]] = set()

    for token in matches:
        token = token.strip()
        # Check coordinate notation like "(0, 1)" or "[0, 1]" or "0:1"
        coord_match = re.match(r"^[\(\[]?\s*(\d+)\s*[:,\s]\s*(\d+)\s*[\)\]]?$", token)
        if coord_match:
            d_idx = int(coord_match.group(1))
            p_idx = int(coord_match.group(2))
            if not (0 <= d_idx < len(days)):
                raise ValidationError(
                    f"Day index {d_idx} in slot '{token}' is out of bounds (0-{len(days) - 1})."
                )
            if not (0 <= p_idx < periods_per_day):
                raise ValidationError(
                    f"Period index {p_idx} in slot '{token}' is out of bounds (0-{periods_per_day - 1})."
                )
            slots.add((d_idx, p_idx))
            continue

        # Check named day notation like "Monday P1" or "Mon:1" or "Monday 2"
        named_match = re.match(
            r"^([A-Za-z]+)\s*[:\-]?\s*[Pp]?\s*(\d+)$", token
        )
        if named_match:
            d_str = named_match.group(1).lower()
            p_num = int(named_match.group(2))
            if d_str not in day_map:
                raise ValidationError(
                    f"Unknown day '{named_match.group(1)}' in slot '{token}'. Valid days: {', '.join(days)}."
                )
            d_idx = day_map[d_str]
            # p_num is 1-indexed (e.g. P1 = 0)
            if p_num < 1 or p_num > periods_per_day:
                raise ValidationError(
                    f"Period P{p_num} in slot '{token}' is out of bounds (valid: P1 to P{periods_per_day})."
                )
            slots.add((d_idx, p_num - 1))
            continue

        raise ValidationError(
            f"Cannot parse slot '{token}'. Expected format like 'Monday P1' or '(0, 0)'."
        )

    return sorted(slots)


def format_slots_to_string(slots: list[tuple[int, int]], days: list[str]) -> str:
    """Formats a list of (day, period) coordinate tuples into a readable string."""
    if not slots:
        return ""
    formatted = []
    for d, p in sorted(slots):
        day_name = days[d] if 0 <= d < len(days) else f"Day {d}"
        formatted.append(f"{day_name} P{p + 1}")
    return ", ".join(formatted)


def create_default_custom_data() -> dict[str, Any]:
    """Returns default initial datasets and tables for a new custom timetable."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    periods_per_day = 6

    classes_df = pd.DataFrame([
        {"class_id": "CSE_1", "class_name": "CS Year 1", "strength": 30, "lunch_slot": 4},
        {"class_id": "CSE_2", "class_name": "CS Year 2", "strength": 25, "lunch_slot": 4},
    ])

    teachers_df = pd.DataFrame([
        {
            "teacher_id": "T1",
            "teacher_name": "Dr. Alan Turing",
            "max_weekly_load": 14,
            "qualified_subjects": "CS101, CS201",
            "unavailable_slots": "Friday P6",
            "preferred_slots": "Monday P1, Tuesday P2",
        },
        {
            "teacher_id": "T2",
            "teacher_name": "Dr. Ada Lovelace",
            "max_weekly_load": 14,
            "qualified_subjects": "CS102, CS202",
            "unavailable_slots": "Monday P1",
            "preferred_slots": "Wednesday P2, Thursday P3",
        },
        {
            "teacher_id": "T3",
            "teacher_name": "Dr. Claude Shannon",
            "max_weekly_load": 12,
            "qualified_subjects": "MA101, CS101_LAB",
            "unavailable_slots": "",
            "preferred_slots": "",
        },
    ])

    subjects_df = pd.DataFrame([
        {
            "subject_id": "CS101",
            "subject_name": "Programming Fundamentals",
            "class_id": "CSE_1",
            "teacher_id": "T1",
            "type": "theory",
            "weekly_periods": 4,
            "lab_duration": 1,
        },
        {
            "subject_id": "CS102",
            "subject_name": "Digital Logic",
            "class_id": "CSE_1",
            "teacher_id": "T2",
            "type": "theory",
            "weekly_periods": 4,
            "lab_duration": 1,
        },
        {
            "subject_id": "CS101_LAB",
            "subject_name": "Programming Lab",
            "class_id": "CSE_1",
            "teacher_id": "T3",
            "type": "lab",
            "weekly_periods": 2,
            "lab_duration": 2,
        },
        {
            "subject_id": "CS201",
            "subject_name": "Data Structures",
            "class_id": "CSE_2",
            "teacher_id": "T1",
            "type": "theory",
            "weekly_periods": 4,
            "lab_duration": 1,
        },
        {
            "subject_id": "CS202",
            "subject_name": "Computer Architecture",
            "class_id": "CSE_2",
            "teacher_id": "T2",
            "type": "theory",
            "weekly_periods": 4,
            "lab_duration": 1,
        },
        {
            "subject_id": "MA101",
            "subject_name": "Applied Mathematics",
            "class_id": "CSE_2",
            "teacher_id": "T3",
            "type": "theory",
            "weekly_periods": 4,
            "lab_duration": 1,
        },
    ])

    rooms_df = pd.DataFrame([
        {"room_id": "CR101", "room_name": "Lecture Hall 101", "capacity": 35, "type": "classroom"},
        {"room_id": "CR102", "room_name": "Lecture Hall 102", "capacity": 35, "type": "classroom"},
        {"room_id": "LAB01", "room_name": "Computing Lab 01", "capacity": 30, "type": "lab"},
    ])

    config = {
        "name": "Custom College Timetable",
        "days": days,
        "periods_per_day": periods_per_day,
        "default_lunch_slot": 4,  # Period 4 (1-indexed) -> 3 in 0-indexed
        "blocked_periods": "Monday P1",  # Weekly assembly
    }

    return {
        "config": config,
        "classes": classes_df,
        "teachers": teachers_df,
        "subjects": subjects_df,
        "rooms": rooms_df,
    }


def dataset_to_custom_tables(dataset: TimetableDataset) -> dict[str, Any]:
    """Converts an existing TimetableDataset into the editable custom tables format."""
    ts = dataset.time_structure
    days = list(ts.days)
    periods_per_day = ts.periods_per_day

    # Classes DataFrame
    class_rows = []
    for c in dataset.classes.values():
        class_rows.append({
            "class_id": c.id,
            "class_name": c.name,
            "strength": int(c.strength),
            "lunch_slot": int(c.lunch_slot) + 1,  # Store 1-indexed for UI display
        })
    classes_df = pd.DataFrame(class_rows)

    # Teachers DataFrame
    teacher_rows = []
    for t in dataset.teachers.values():
        teacher_rows.append({
            "teacher_id": t.id,
            "teacher_name": t.name,
            "max_weekly_load": int(t.max_weekly_load),
            "qualified_subjects": ", ".join(t.subject_ids),
            "unavailable_slots": format_slots_to_string(t.unavailable_slots, days),
            "preferred_slots": format_slots_to_string(t.preferred_slots, days),
        })
    teachers_df = pd.DataFrame(teacher_rows)

    # Subjects & Offerings DataFrame
    subject_rows = []
    for cid, c in dataset.classes.items():
        for req in c.subjects:
            subj = dataset.subjects[req.subject_id]
            wp = req.weekly_periods if req.weekly_periods is not None else subj.weekly_periods
            subject_rows.append({
                "subject_id": subj.id,
                "subject_name": subj.name,
                "class_id": cid,
                "teacher_id": req.teacher_id,
                "type": subj.type,
                "weekly_periods": int(wp),
                "lab_duration": int(subj.lab_duration) if subj.lab_duration else 1,
            })
    subjects_df = pd.DataFrame(subject_rows)

    # Rooms DataFrame
    room_rows = []
    for r in dataset.rooms.values():
        room_rows.append({
            "room_id": r.id,
            "room_name": r.name,
            "capacity": int(r.capacity),
            "type": r.type,
        })
    rooms_df = pd.DataFrame(room_rows)

    config = {
        "name": dataset.name,
        "days": days,
        "periods_per_day": periods_per_day,
        "default_lunch_slot": class_rows[0]["lunch_slot"] if class_rows else 4,
        "blocked_periods": format_slots_to_string(ts.blocked_periods, days),
    }

    return {
        "config": config,
        "classes": classes_df,
        "teachers": teachers_df,
        "subjects": subjects_df,
        "rooms": rooms_df,
    }


def validate_and_build_custom_dataset(
    config: dict[str, Any],
    classes_data: Any,
    teachers_data: Any,
    subjects_data: Any,
    rooms_data: Any,
) -> CustomValidationResult:
    """Validates user input and constructs a TimetableDataset.
    
    Accepts DataFrames or lists of dicts for each entity type.
    Returns a CustomValidationResult with distinct errors, warnings, and the dataset.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Helper to convert DataFrame or list of dicts to clean list of dicts
    def _to_records(data: Any) -> list[dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, pd.DataFrame):
            # Drop empty rows
            df = data.dropna(how="all")
            return df.to_dict(orient="records")
        if isinstance(data, list):
            return [dict(r) for r in data if isinstance(r, dict)]
        return []

    classes_records = _to_records(classes_data)
    teachers_records = _to_records(teachers_data)
    subjects_records = _to_records(subjects_data)
    rooms_records = _to_records(rooms_data)

    # ---------------------------------------------------------
    # 1. Validate Time Structure
    # ---------------------------------------------------------
    raw_days = config.get("days", [])
    if isinstance(raw_days, str):
        days = [d.strip() for d in raw_days.split(",") if d.strip()]
    else:
        days = [str(d).strip() for d in raw_days if str(d).strip()]

    if not days:
        errors.append("Time Structure: At least one working day must be specified.")
    elif len(days) != len(set(days)):
        errors.append("Time Structure: Working day names must be unique.")

    try:
        periods_per_day = int(config.get("periods_per_day", 0))
        if periods_per_day <= 0:
            errors.append("Time Structure: Periods per day must be a positive integer (> 0).")
    except (ValueError, TypeError):
        errors.append("Time Structure: Periods per day must be a valid integer.")
        periods_per_day = 0

    blocked_periods: list[tuple[int, int]] = []
    if days and periods_per_day > 0:
        raw_blocked = config.get("blocked_periods", "")
        try:
            blocked_periods = parse_slot_string(raw_blocked, days, periods_per_day)
        except ValidationError as ve:
            errors.append(f"Blocked Periods: {ve}")

    # ---------------------------------------------------------
    # 2. Validate Classes
    # ---------------------------------------------------------
    if not classes_records:
        errors.append("Classes: At least one class/batch must be added.")

    seen_class_ids: set[str] = set()
    cleaned_classes: dict[str, dict[str, Any]] = {}

    for idx, r in enumerate(classes_records, start=1):
        cid = str(r.get("class_id", "")).strip()
        cname = str(r.get("class_name", cid)).strip()
        if not cid:
            errors.append(f"Class row #{idx}: Class ID is required.")
            continue
        if cid in seen_class_ids:
            errors.append(f"Classes: Duplicate Class ID '{cid}'. Class IDs must be unique.")
            continue
        seen_class_ids.add(cid)

        # Strength validation
        try:
            strength = int(r.get("strength", 0))
            if strength <= 0:
                errors.append(f"Class '{cid}': Student strength must be positive (> 0).")
        except (ValueError, TypeError):
            errors.append(f"Class '{cid}': Strength must be a valid integer.")
            strength = 0

        # Lunch slot validation (UI is 1-indexed, internally 0-indexed)
        try:
            raw_lunch = r.get("lunch_slot", config.get("default_lunch_slot", 4))
            lunch_int = int(raw_lunch)
            # If user entered 1-indexed (e.g. 4 for 4th period), map to 3
            if 1 <= lunch_int <= periods_per_day:
                lunch_slot = lunch_int - 1
            elif 0 <= lunch_int < periods_per_day:
                lunch_slot = lunch_int
            else:
                errors.append(
                    f"Class '{cid}': Lunch slot {lunch_int} is out of bounds (1-{periods_per_day})."
                )
                lunch_slot = 0
        except (ValueError, TypeError):
            errors.append(f"Class '{cid}': Lunch slot must be an integer.")
            lunch_slot = 0

        cleaned_classes[cid] = {
            "id": cid,
            "name": cname or cid,
            "strength": strength,
            "lunch_slot": lunch_slot,
        }

    # ---------------------------------------------------------
    # 3. Validate Teachers
    # ---------------------------------------------------------
    if not teachers_records:
        errors.append("Teachers: At least one teacher must be added.")

    seen_teacher_ids: set[str] = set()
    cleaned_teachers: dict[str, dict[str, Any]] = {}

    for idx, r in enumerate(teachers_records, start=1):
        tid = str(r.get("teacher_id", "")).strip()
        tname = str(r.get("teacher_name", tid)).strip()
        if not tid:
            errors.append(f"Teacher row #{idx}: Teacher ID is required.")
            continue
        if tid in seen_teacher_ids:
            errors.append(f"Teachers: Duplicate Teacher ID '{tid}'. Teacher IDs must be unique.")
            continue
        seen_teacher_ids.add(tid)

        try:
            max_load = int(r.get("max_weekly_load", 20))
            if max_load <= 0:
                errors.append(f"Teacher '{tid}': Max weekly load must be positive (> 0).")
        except (ValueError, TypeError):
            errors.append(f"Teacher '{tid}': Max weekly load must be an integer.")
            max_load = 20

        # Parse qualified subjects
        raw_quals = r.get("qualified_subjects", "")
        if isinstance(raw_quals, str):
            qual_list = [s.strip() for s in re.split(r"[,;\s]+", raw_quals) if s.strip()]
        elif isinstance(raw_quals, (list, tuple)):
            qual_list = [str(s).strip() for s in raw_quals if str(s).strip()]
        else:
            qual_list = []

        # Parse unavailable and preferred slots
        unavail_slots: list[tuple[int, int]] = []
        preferred_slots: list[tuple[int, int]] = []
        if days and periods_per_day > 0:
            try:
                unavail_slots = parse_slot_string(
                    r.get("unavailable_slots", ""), days, periods_per_day
                )
            except ValidationError as ve:
                errors.append(f"Teacher '{tid}' unavailable slots: {ve}")

            try:
                preferred_slots = parse_slot_string(
                    r.get("preferred_slots", ""), days, periods_per_day
                )
            except ValidationError as ve:
                errors.append(f"Teacher '{tid}' preferred slots: {ve}")

        cleaned_teachers[tid] = {
            "id": tid,
            "name": tname or tid,
            "max_weekly_load": max_load,
            "qualified_subjects": set(qual_list),
            "unavailable_slots": unavail_slots,
            "preferred_slots": preferred_slots,
        }

    # ---------------------------------------------------------
    # 4. Validate Rooms
    # ---------------------------------------------------------
    if not rooms_records:
        errors.append("Rooms: At least one room must be added.")

    seen_room_ids: set[str] = set()
    cleaned_rooms: dict[str, dict[str, Any]] = {}

    for idx, r in enumerate(rooms_records, start=1):
        rid = str(r.get("room_id", "")).strip()
        rname = str(r.get("room_name", rid)).strip()
        if not rid:
            errors.append(f"Room row #{idx}: Room ID is required.")
            continue
        if rid in seen_room_ids:
            errors.append(f"Rooms: Duplicate Room ID '{rid}'. Room IDs must be unique.")
            continue
        seen_room_ids.add(rid)

        try:
            capacity = int(r.get("capacity", 0))
            if capacity <= 0:
                errors.append(f"Room '{rid}': Capacity must be positive (> 0).")
        except (ValueError, TypeError):
            errors.append(f"Room '{rid}': Capacity must be an integer.")
            capacity = 0

        rtype = str(r.get("type", "classroom")).strip().lower()
        if rtype not in ("classroom", "lab"):
            errors.append(f"Room '{rid}': Type must be 'classroom' or 'lab', got '{rtype}'.")
            rtype = "classroom"

        cleaned_rooms[rid] = {
            "id": rid,
            "name": rname or rid,
            "capacity": capacity,
            "type": rtype,
        }

    # ---------------------------------------------------------
    # 5. Validate Subjects & Course Offerings
    # ---------------------------------------------------------
    if not subjects_records:
        errors.append("Subjects: At least one subject must be added.")

    cleaned_subjects: dict[str, dict[str, Any]] = {}
    class_requirements: dict[str, list[dict[str, Any]]] = {
        cid: [] for cid in cleaned_classes
    }
    class_seen_subjects: dict[str, set[str]] = {
        cid: set() for cid in cleaned_classes
    }

    for idx, r in enumerate(subjects_records, start=1):
        sid = str(r.get("subject_id", "")).strip()
        sname = str(r.get("subject_name", sid)).strip()
        cid = str(r.get("class_id", "")).strip()
        tid = str(r.get("teacher_id", "")).strip()

        if not sid:
            errors.append(f"Subject row #{idx}: Subject ID is required.")
            continue

        # Validate Class Reference
        if not cid:
            errors.append(f"Subject '{sid}' (row #{idx}): Associated Class ID is required.")
            continue
        if cid not in cleaned_classes:
            errors.append(
                f"Subject '{sid}': Class '{cid}' does not exist in defined Classes. "
                f"Valid classes: {', '.join(cleaned_classes.keys())}."
            )
            continue

        # Prevent duplicate subject assignment to the same class
        if sid in class_seen_subjects[cid]:
            errors.append(
                f"Class '{cid}': Duplicate subject requirement for '{sid}'. "
                "Each subject can only be assigned once per class."
            )
            continue
        class_seen_subjects[cid].add(sid)

        # Validate Teacher Reference
        if not tid:
            errors.append(f"Subject '{sid}' for Class '{cid}': Assigned Teacher ID is required.")
            continue
        if tid not in cleaned_teachers:
            errors.append(
                f"Subject '{sid}': Teacher '{tid}' does not exist in defined Teachers. "
                f"Valid teachers: {', '.join(cleaned_teachers.keys())}."
            )
            continue

        stype = str(r.get("type", "theory")).strip().lower()
        if stype not in ("theory", "lab"):
            errors.append(
                f"Subject '{sid}': Type must be 'theory' or 'lab', got '{stype}'."
            )
            stype = "theory"

        # Weekly periods
        try:
            wp = int(r.get("weekly_periods", 0))
            if wp <= 0:
                errors.append(f"Subject '{sid}': Weekly periods must be positive (> 0).")
        except (ValueError, TypeError):
            errors.append(f"Subject '{sid}': Weekly periods must be an integer.")
            wp = 0

        # Lab duration
        lab_dur: int | None = None
        if stype == "lab":
            try:
                raw_dur = r.get("lab_duration", 2)
                lab_dur = int(raw_dur) if raw_dur is not None else 2
                if lab_dur <= 0:
                    errors.append(f"Lab subject '{sid}': Lab duration must be positive (> 0).")
                elif periods_per_day > 0 and lab_dur > periods_per_day:
                    errors.append(
                        f"Lab subject '{sid}': Lab duration ({lab_dur}) cannot exceed periods per day ({periods_per_day})."
                    )
                elif wp > 0 and wp % lab_dur != 0:
                    errors.append(
                        f"Lab subject '{sid}': Weekly periods ({wp}) must be divisible by lab duration ({lab_dur})."
                    )
            except (ValueError, TypeError):
                errors.append(f"Lab subject '{sid}': Lab duration must be an integer.")
        else:
            lab_dur = None

        # Record subject definition
        if sid in cleaned_subjects:
            # Check consistency if defined multiple times
            prev = cleaned_subjects[sid]
            if prev["type"] != stype or prev["weekly_periods"] != wp or prev["lab_duration"] != lab_dur:
                errors.append(
                    f"Subject '{sid}' has conflicting definitions across classes. "
                    "Use distinct Subject IDs (e.g. CS101_A, CS101_B) if requirements differ."
                )
        else:
            cleaned_subjects[sid] = {
                "id": sid,
                "name": sname or sid,
                "type": stype,
                "weekly_periods": wp,
                "lab_duration": lab_dur,
            }

        # Auto-qualify teacher for assigned subject
        cleaned_teachers[tid]["qualified_subjects"].add(sid)

        # Add requirement to class
        class_requirements[cid].append({
            "subject_id": sid,
            "teacher_id": tid,
            "weekly_periods": wp,
        })

    # ---------------------------------------------------------
    # 6. Cross-Entity Physical & Relational Validation
    # ---------------------------------------------------------
    # Check that rooms exist for required types and capacities
    classrooms = [r for r in cleaned_rooms.values() if r["type"] == "classroom"]
    labs = [r for r in cleaned_rooms.values() if r["type"] == "lab"]

    for cid, cinfo in cleaned_classes.items():
        reqs = class_requirements.get(cid, [])
        has_theory = any(
            cleaned_subjects[r["subject_id"]]["type"] == "theory"
            for r in reqs if r["subject_id"] in cleaned_subjects
        )
        has_lab = any(
            cleaned_subjects[r["subject_id"]]["type"] == "lab"
            for r in reqs if r["subject_id"] in cleaned_subjects
        )

        c_str = cinfo["strength"]
        if has_theory:
            if not classrooms:
                errors.append(
                    f"Class '{cid}' requires theory lectures, but no classrooms have been added."
                )
            elif not any(r["capacity"] >= c_str for r in classrooms):
                max_cr = max(r["capacity"] for r in classrooms)
                errors.append(
                    f"Class '{cid}' (strength {c_str}) exceeds capacity of all classrooms (max: {max_cr}). "
                    "Increase classroom capacity or reduce class strength."
                )

        if has_lab:
            if not labs:
                errors.append(
                    f"Class '{cid}' requires lab sessions, but no laboratory rooms have been added."
                )
            elif not any(r["capacity"] >= c_str for r in labs):
                max_lab = max(r["capacity"] for r in labs)
                errors.append(
                    f"Class '{cid}' (strength {c_str}) exceeds capacity of all labs (max: {max_lab}). "
                    "Increase lab capacity or reduce class strength."
                )

    # Check Teacher Workload Limits
    teacher_loads: dict[str, int] = {tid: 0 for tid in cleaned_teachers}
    for cid, reqs in class_requirements.items():
        for req in reqs:
            teacher_loads[req["teacher_id"]] += req["weekly_periods"]

    total_grid_slots = len(days) * periods_per_day
    blocked_set = set(blocked_periods)

    for tid, tinfo in cleaned_teachers.items():
        assigned = teacher_loads[tid]
        if assigned > tinfo["max_weekly_load"]:
            errors.append(
                f"Teacher '{tid}' ({tinfo['name']}) is assigned {assigned} periods, "
                f"which exceeds maximum weekly load of {tinfo['max_weekly_load']}."
            )
        unavail_set = set(tinfo["unavailable_slots"])
        valid_teacher_slots = total_grid_slots - len(blocked_set | unavail_set)
        if assigned > valid_teacher_slots:
            errors.append(
                f"Teacher '{tid}' ({tinfo['name']}) is assigned {assigned} periods, "
                f"which exceeds their {valid_teacher_slots} available slots."
            )

    # Check Class Usable Slots against Required Periods
    for cid, cinfo in cleaned_classes.items():
        reqs = class_requirements.get(cid, [])
        total_req = sum(r["weekly_periods"] for r in reqs)
        # Class cannot schedule during lunch slot on any day
        lunch_slots = {(d, cinfo["lunch_slot"]) for d in range(len(days))}
        usable_class_slots = total_grid_slots - len(lunch_slots | blocked_set)
        if total_req > usable_class_slots:
            errors.append(
                f"Class '{cid}' ({cinfo['name']}) requires {total_req} periods, "
                f"which exceeds its {usable_class_slots} usable timetable slots."
            )

    # If any hard syntax/relational errors occurred, stop here
    if errors:
        return CustomValidationResult(
            is_valid=False,
            errors=errors,
            is_feasible=False,
            warnings=warnings,
            dataset=None,
        )

    # ---------------------------------------------------------
    # 7. Construct TimetableDataset Object
    # ---------------------------------------------------------
    try:
        ts = TimeStructure(
            days=days,
            periods_per_day=periods_per_day,
            blocked_periods=blocked_periods,
        )

        subjects_dict = {
            sid: Subject(
                id=s["id"],
                name=s["name"],
                weekly_periods=s["weekly_periods"],
                type=s["type"],
                lab_duration=s["lab_duration"],
            )
            for sid, s in cleaned_subjects.items()
        }

        teachers_dict = {
            tid: Teacher(
                id=t["id"],
                name=t["name"],
                subject_ids=sorted(t["qualified_subjects"]),
                unavailable_slots=t["unavailable_slots"],
                max_weekly_load=t["max_weekly_load"],
                preferred_slots=t["preferred_slots"],
            )
            for tid, t in cleaned_teachers.items()
        }

        rooms_dict = {
            rid: Room(
                id=r["id"],
                name=r["name"],
                capacity=r["capacity"],
                type=r["type"],
            )
            for rid, r in cleaned_rooms.items()
        }

        classes_dict = {}
        for cid, c in cleaned_classes.items():
            reqs = [
                ClassSubjectRequirement(
                    subject_id=r["subject_id"],
                    teacher_id=r["teacher_id"],
                    weekly_periods=r["weekly_periods"],
                )
                for r in class_requirements.get(cid, [])
            ]
            classes_dict[cid] = ClassGroup(
                id=c["id"],
                name=c["name"],
                strength=c["strength"],
                lunch_slot=c["lunch_slot"],
                subjects=reqs,
            )

        dataset = TimetableDataset(
            name=str(config.get("name", "Custom Timetable")),
            time_structure=ts,
            subjects=subjects_dict,
            teachers=teachers_dict,
            classes=classes_dict,
            rooms=rooms_dict,
        )

        # Deep dataset schema check
        validate_dataset_schema(dataset)

        # Feasibility check for mathematical bottlenecks
        feasibility_issues = check_dataset_feasibility(dataset)
        if feasibility_issues:
            warnings.extend(feasibility_issues)

        return CustomValidationResult(
            is_valid=True,
            errors=[],
            is_feasible=len(feasibility_issues) == 0,
            warnings=warnings,
            dataset=dataset,
        )

    except ValidationError as ve:
        return CustomValidationResult(
            is_valid=False,
            errors=[str(ve)],
            is_feasible=False,
            warnings=warnings,
            dataset=None,
        )
    except Exception as exc:
        return CustomValidationResult(
            is_valid=False,
            errors=[f"Unexpected error constructing dataset: {exc}"],
            is_feasible=False,
            warnings=warnings,
            dataset=None,
        )
