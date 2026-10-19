"""Data models, schema validation, and dataset loading for Timetable CSP.

This module defines the foundational entities of the timetable scheduling problem:
- TimeStructure: Days, periods per day, and global blocked slots.
- Subject: Theory or Lab subject details (with lab duration divisibility check).
- Teacher: Qualified subjects, unavailable slots, and maximum weekly load.
- ClassGroup: Student strength, lunch break slot, and required subjects with assigned teachers.
- Room: Classroom or Lab room with seating capacity.
- TimetableDataset: Complete container for all problem inputs.

All models use Python dataclasses with full type annotations. Clear validation
functions ensure that datasets loaded from JSON meet all structural and logical
requirements before scheduling or validation begins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Raised when dataset schema, references, or logical constraints are invalid."""
    pass


@dataclass(frozen=True)
class TimeStructure:
    """Defines the weekly time grid and global break periods.
    
    Attributes:
        days: List of day names, e.g. ["Monday", "Tuesday", ...].
        periods_per_day: Number of teaching/break periods in each day (0-indexed).
        blocked_periods: List of (day_index, period_index) tuples that are globally
                         unavailable for all classes and teachers (e.g. assembly).
    """
    days: list[str]
    periods_per_day: int
    blocked_periods: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_slots(self) -> int:
        """Total time slots available in the weekly grid."""
        return len(self.days) * self.periods_per_day

    def is_valid_slot(self, day: int, period: int) -> bool:
        """Checks if a (day, period) coordinate exists within the time grid."""
        return 0 <= day < len(self.days) and 0 <= period < self.periods_per_day

    def is_blocked(self, day: int, period: int) -> bool:
        """Checks if a given slot is marked as a global blocked period."""
        return (day, period) in self.blocked_periods


@dataclass(frozen=True)
class Subject:
    """Academic subject or lab course.
    
    Attributes:
        id: Unique subject identifier (e.g. "CS101", "PHY_LAB").
        name: Human-readable subject title.
        weekly_periods: Total periods required per week.
        type: "theory" or "lab".
        lab_duration: Duration (in consecutive periods) per lab session.
                      Must be specified for labs (e.g. 2); must be None for theory.
    """
    id: str
    name: str
    weekly_periods: int
    type: str  # "theory" | "lab"
    lab_duration: int | None = None

    def __post_init__(self) -> None:
        if self.type not in ("theory", "lab"):
            raise ValidationError(
                f"Subject '{self.id}' type must be 'theory' or 'lab', got '{self.type}'."
            )
        if self.weekly_periods <= 0:
            raise ValidationError(
                f"Subject '{self.id}' weekly_periods must be > 0, got {self.weekly_periods}."
            )
        if self.type == "lab":
            if self.lab_duration is None or self.lab_duration <= 0:
                raise ValidationError(
                    f"Lab subject '{self.id}' must specify lab_duration > 0, got {self.lab_duration}."
                )
            if self.weekly_periods % self.lab_duration != 0:
                raise ValidationError(
                    f"Lab subject '{self.id}' weekly_periods ({self.weekly_periods}) "
                    f"must be divisible by lab_duration ({self.lab_duration})."
                )
        elif self.type == "theory":
            if self.lab_duration is not None:
                raise ValidationError(
                    f"Theory subject '{self.id}' must not specify lab_duration."
                )


@dataclass(frozen=True)
class Teacher:
    """Faculty member teaching one or more subjects.
    
    Attributes:
        id: Unique teacher identifier (e.g. "T01").
        name: Teacher's full name.
        subject_ids: List of subject IDs the teacher is certified to teach.
        unavailable_slots: Slots (day_index, period_index) when teacher cannot teach.
        max_weekly_load: Maximum periods the teacher can teach per week.
                         (A 2-period lab counts as 2 periods towards load).
        preferred_slots: Optional preference slots (stored for future solver use).
    """
    id: str
    name: str
    subject_ids: list[str]
    unavailable_slots: list[tuple[int, int]] = field(default_factory=list)
    max_weekly_load: int = 20
    preferred_slots: list[tuple[int, int]] = field(default_factory=list)

    def is_available(self, day: int, period: int) -> bool:
        """Checks if teacher is available at the specified slot."""
        return (day, period) not in self.unavailable_slots


@dataclass(frozen=True)
class ClassSubjectRequirement:
    """Subject requirement assigned to a class batch, binding a specific teacher.
    
    Attributes:
        subject_id: ID of the subject to be taught.
        teacher_id: Designated teacher ID for this class-subject.
        weekly_periods: Periods required per week. If None, inherits from Subject.
    """
    subject_id: str
    teacher_id: str
    weekly_periods: int | None = None


@dataclass(frozen=True)
class ClassGroup:
    """Class or student batch that attends sessions together.
    
    Attributes:
        id: Unique class identifier (e.g. "CSE_2A").
        name: Class display name.
        strength: Number of enrolled students.
        lunch_slot: Period index reserved for daily lunch break (0-indexed).
        subjects: List of subject requirements with designated teachers.
    """
    id: str
    name: str
    strength: int
    lunch_slot: int
    subjects: list[ClassSubjectRequirement] = field(default_factory=list)

    def get_subject_requirement(self, subject_id: str) -> ClassSubjectRequirement | None:
        """Finds the requirement for a given subject_id, or None if not taken."""
        for req in self.subjects:
            if req.subject_id == subject_id:
                return req
        return None


@dataclass(frozen=True)
class Room:
    """Physical venue where sessions take place.
    
    Attributes:
        id: Unique room identifier (e.g. "CR101", "LAB_01").
        name: Room display name.
        capacity: Maximum number of students that can be seated.
        type: "classroom" (for theory) or "lab" (for lab sessions).
    """
    id: str
    name: str
    capacity: int
    type: str  # "classroom" | "lab"

    def __post_init__(self) -> None:
        if self.type not in ("classroom", "lab"):
            raise ValidationError(
                f"Room '{self.id}' type must be 'classroom' or 'lab', got '{self.type}'."
            )
        if self.capacity <= 0:
            raise ValidationError(
                f"Room '{self.id}' capacity must be > 0, got {self.capacity}."
            )


@dataclass
class TimetableDataset:
    """Complete problem input container holding all entities and grid structure."""
    name: str
    time_structure: TimeStructure
    subjects: dict[str, Subject]
    teachers: dict[str, Teacher]
    classes: dict[str, ClassGroup]
    rooms: dict[str, Room]

    def get_required_periods(self, class_id: str, subject_id: str) -> int:
        """Returns required weekly periods for a specific (class, subject)."""
        cls = self.classes.get(class_id)
        if not cls:
            raise ValidationError(f"Unknown class '{class_id}'.")
        req = cls.get_subject_requirement(subject_id)
        if not req:
            raise ValidationError(f"Class '{class_id}' does not take subject '{subject_id}'.")
        if req.weekly_periods is not None:
            return req.weekly_periods
        subj = self.subjects.get(subject_id)
        if not subj:
            raise ValidationError(f"Unknown subject '{subject_id}'.")
        return subj.weekly_periods

    def to_dict(self) -> dict[str, Any]:
        """Serializes dataset to a standard JSON-compatible dictionary."""
        return {
            "name": self.name,
            "time_structure": {
                "days": self.time_structure.days,
                "periods_per_day": self.time_structure.periods_per_day,
                "blocked_periods": [list(slot) for slot in self.time_structure.blocked_periods],
            },
            "subjects": [
                {
                    "id": s.id,
                    "name": s.name,
                    "weekly_periods": s.weekly_periods,
                    "type": s.type,
                    "lab_duration": s.lab_duration,
                }
                for s in self.subjects.values()
            ],
            "teachers": [
                {
                    "id": t.id,
                    "name": t.name,
                    "subject_ids": t.subject_ids,
                    "unavailable_slots": [list(slot) for slot in t.unavailable_slots],
                    "max_weekly_load": t.max_weekly_load,
                    "preferred_slots": [list(slot) for slot in t.preferred_slots],
                }
                for t in self.teachers.values()
            ],
            "classes": [
                {
                    "id": c.id,
                    "name": c.name,
                    "strength": c.strength,
                    "lunch_slot": c.lunch_slot,
                    "subjects": [
                        {
                            "subject_id": r.subject_id,
                            "teacher_id": r.teacher_id,
                            "weekly_periods": r.weekly_periods,
                        }
                        for r in c.subjects
                    ],
                }
                for c in self.classes.values()
            ],
            "rooms": [
                {
                    "id": r.id,
                    "name": r.name,
                    "capacity": r.capacity,
                    "type": r.type,
                }
                for r in self.rooms.values()
            ],
        }


def _parse_slot_list(raw_slots: Any, field_name: str) -> list[tuple[int, int]]:
    """Helper to parse slots from [[day, period], ...] or [{'day': d, 'period': p}, ...]."""
    if not raw_slots:
        return []
    slots: list[tuple[int, int]] = []
    for item in raw_slots:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            slots.append((int(item[0]), int(item[1])))
        elif isinstance(item, dict) and "day" in item and "period" in item:
            slots.append((int(item["day"]), int(item["period"])))
        else:
            raise ValidationError(
                f"Invalid slot format in {field_name}: expected [day, period] or "
                f"{{'day': d, 'period': p}}, got {item}."
            )
    return slots


def validate_dataset_schema(dataset: TimetableDataset) -> None:
    """Performs deep cross-entity integrity checks on the dataset.
    
    Raises:
        ValidationError: If any logical inconsistency or dangling reference is found.
    """
    ts = dataset.time_structure

    # 1. Time structure validation
    if not ts.days or len(ts.days) != len(set(ts.days)):
        raise ValidationError("Dataset must define non-empty, unique day names.")
    if ts.periods_per_day <= 0:
        raise ValidationError("periods_per_day must be positive.")
    for d, p in ts.blocked_periods:
        if not ts.is_valid_slot(d, p):
            raise ValidationError(
                f"Blocked period ({d}, {p}) is out of bounds for "
                f"{len(ts.days)} days x {ts.periods_per_day} periods."
            )

    # 2. Subjects validation
    if not dataset.subjects:
        raise ValidationError("Dataset must contain at least one subject.")

    # 3. Teachers validation
    if not dataset.teachers:
        raise ValidationError("Dataset must contain at least one teacher.")
    for t in dataset.teachers.values():
        if t.max_weekly_load <= 0:
            raise ValidationError(f"Teacher '{t.id}' max_weekly_load must be > 0.")
        for d, p in t.unavailable_slots:
            if not ts.is_valid_slot(d, p):
                raise ValidationError(
                    f"Teacher '{t.id}' unavailable slot ({d}, {p}) is out of bounds."
                )
        for sid in t.subject_ids:
            if sid not in dataset.subjects:
                raise ValidationError(
                    f"Teacher '{t.id}' references unknown subject '{sid}'."
                )

    # 4. Rooms validation
    if not dataset.rooms:
        raise ValidationError("Dataset must contain at least one room.")

    # 5. Classes validation
    if not dataset.classes:
        raise ValidationError("Dataset must contain at least one class.")
    for c in dataset.classes.values():
        if c.strength <= 0:
            raise ValidationError(f"Class '{c.id}' strength must be > 0.")
        if not (0 <= c.lunch_slot < ts.periods_per_day):
            raise ValidationError(
                f"Class '{c.id}' lunch_slot ({c.lunch_slot}) must be between 0 "
                f"and {ts.periods_per_day - 1}."
            )
        seen_subjects: set[str] = set()
        for req in c.subjects:
            if req.subject_id in seen_subjects:
                raise ValidationError(
                    f"Class '{c.id}' has duplicate requirement for subject '{req.subject_id}'."
                )
            seen_subjects.add(req.subject_id)

            if req.subject_id not in dataset.subjects:
                raise ValidationError(
                    f"Class '{c.id}' references unknown subject '{req.subject_id}'."
                )
            subj = dataset.subjects[req.subject_id]

            if req.teacher_id not in dataset.teachers:
                raise ValidationError(
                    f"Class '{c.id}' references unknown teacher '{req.teacher_id}'."
                )
            teacher = dataset.teachers[req.teacher_id]

            if req.subject_id not in teacher.subject_ids:
                raise ValidationError(
                    f"Teacher '{teacher.id}' assigned to class '{c.id}' for subject "
                    f"'{subj.id}' is not qualified to teach it."
                )

            wp = req.weekly_periods if req.weekly_periods is not None else subj.weekly_periods
            if wp <= 0:
                raise ValidationError(
                    f"Class '{c.id}' subject '{subj.id}' required periods must be > 0."
                )
            if subj.type == "lab" and subj.lab_duration:
                if wp % subj.lab_duration != 0:
                    raise ValidationError(
                        f"Class '{c.id}' subject '{subj.id}' periods ({wp}) must be "
                        f"divisible by lab_duration ({subj.lab_duration})."
                    )


def check_dataset_feasibility(dataset: TimetableDataset) -> list[str]:
    """Runs programmatic capacity and mathematical solvability checks.
    
    Returns a list of warning or error strings if any mathematical bottlenecks
    make scheduling impossible. An empty list indicates all preliminary sanity
    checks passed.
    """
    issues: list[str] = []
    ts = dataset.time_structure
    total_slots = ts.total_slots

    # Check 1: Every subject has at least 1 qualified teacher
    for s_id, subj in dataset.subjects.items():
        qualified = [t.id for t in dataset.teachers.values() if s_id in t.subject_ids]
        if not qualified:
            issues.append(f"Subject '{s_id}' has no qualified teachers.")

    # Check 2: Per-teacher workload and availability
    teacher_assigned_periods: dict[str, int] = {t_id: 0 for t_id in dataset.teachers}
    for c in dataset.classes.values():
        for req in c.subjects:
            wp = req.weekly_periods if req.weekly_periods is not None else dataset.subjects[req.subject_id].weekly_periods
            teacher_assigned_periods[req.teacher_id] += wp

    for t_id, teacher in dataset.teachers.items():
        assigned = teacher_assigned_periods[t_id]
        if assigned > teacher.max_weekly_load:
            issues.append(
                f"Teacher '{t_id}' assigned {assigned} periods, exceeding max load {teacher.max_weekly_load}."
            )
        # Compute teacher available slots (excluding unavailable slots and global blocked slots)
        blocked_set = set(ts.blocked_periods)
        unavail_set = set(teacher.unavailable_slots)
        valid_slots = total_slots - len(blocked_set | unavail_set)
        if assigned > valid_slots:
            issues.append(
                f"Teacher '{t_id}' assigned {assigned} periods, exceeding their {valid_slots} available slots."
            )

    # Check 3: Per-class capacity against available teaching periods
    for c in dataset.classes.values():
        class_periods = sum(
            req.weekly_periods if req.weekly_periods is not None else dataset.subjects[req.subject_id].weekly_periods
            for req in c.subjects
        )
        # Class cannot use its lunch slot on any day (len(days) lunch slots) + blocked slots
        lunch_slots = {(d, c.lunch_slot) for d in range(len(ts.days))}
        blocked_set = set(ts.blocked_periods)
        usable_class_slots = total_slots - len(lunch_slots | blocked_set)
        if class_periods > usable_class_slots:
            issues.append(
                f"Class '{c.id}' requires {class_periods} periods, exceeding its {usable_class_slots} usable slots."
            )

    # Check 4: Room capacity per class
    classrooms = [r for r in dataset.rooms.values() if r.type == "classroom"]
    labs = [r for r in dataset.rooms.values() if r.type == "lab"]

    for c in dataset.classes.values():
        has_theory = any(dataset.subjects[req.subject_id].type == "theory" for req in c.subjects)
        has_lab = any(dataset.subjects[req.subject_id].type == "lab" for req in c.subjects)

        if has_theory:
            if not any(r.capacity >= c.strength for r in classrooms):
                issues.append(
                    f"Class '{c.id}' (strength {c.strength}) exceeds capacity of all classrooms."
                )
        if has_lab:
            if not any(r.capacity >= c.strength for r in labs):
                issues.append(
                    f"Class '{c.id}' (strength {c.strength}) exceeds capacity of all labs."
                )

    # Check 5: Aggregate room capacity
    total_theory_periods = sum(
        (req.weekly_periods if req.weekly_periods is not None else dataset.subjects[req.subject_id].weekly_periods)
        for c in dataset.classes.values()
        for req in c.subjects
        if dataset.subjects[req.subject_id].type == "theory"
    )
    total_lab_periods = sum(
        (req.weekly_periods if req.weekly_periods is not None else dataset.subjects[req.subject_id].weekly_periods)
        for c in dataset.classes.values()
        for req in c.subjects
        if dataset.subjects[req.subject_id].type == "lab"
    )

    room_avail_slots_per_room = total_slots - len(ts.blocked_periods)
    total_classroom_slot_cap = len(classrooms) * room_avail_slots_per_room
    total_lab_slot_cap = len(labs) * room_avail_slots_per_room

    if total_theory_periods > total_classroom_slot_cap:
        issues.append(
            f"Total theory periods ({total_theory_periods}) exceeds total classroom slot capacity ({total_classroom_slot_cap})."
        )
    if total_lab_periods > total_lab_slot_cap:
        issues.append(
            f"Total lab periods ({total_lab_periods}) exceeds total lab slot capacity ({total_lab_slot_cap})."
        )

    return issues


def load_dataset_from_dict(raw: dict[str, Any]) -> TimetableDataset:
    """Parses and validates a raw dictionary into a TimetableDataset."""
    name = raw.get("name", "Timetable Dataset")

    raw_ts = raw.get("time_structure", {})
    ts = TimeStructure(
        days=list(raw_ts.get("days", [])),
        periods_per_day=int(raw_ts.get("periods_per_day", 0)),
        blocked_periods=_parse_slot_list(raw_ts.get("blocked_periods", []), "time_structure.blocked_periods"),
    )

    # Parse subjects
    subjects: dict[str, Subject] = {}
    for item in raw.get("subjects", []):
        sid = item.get("id") or item.get("code")
        if not sid:
            raise ValidationError("Every subject must have an 'id' or 'code'.")
        if sid in subjects:
            raise ValidationError(f"Duplicate subject ID '{sid}'.")
        # Handle 'weekly_hours' alias from spec
        wp = item.get("weekly_periods") if "weekly_periods" in item else item.get("weekly_hours")
        if wp is None:
            raise ValidationError(f"Subject '{sid}' is missing 'weekly_periods'.")
        subjects[sid] = Subject(
            id=sid,
            name=str(item.get("name", sid)),
            weekly_periods=int(wp),
            type=str(item.get("type", "theory")).lower(),
            lab_duration=int(item["lab_duration"]) if item.get("lab_duration") is not None else None,
        )

    # Parse teachers
    teachers: dict[str, Teacher] = {}
    for item in raw.get("teachers", []):
        tid = item.get("id") or item.get("name")
        if not tid:
            raise ValidationError("Every teacher must have an 'id' or 'name'.")
        if tid in teachers:
            raise ValidationError(f"Duplicate teacher ID '{tid}'.")
        teachers[tid] = Teacher(
            id=tid,
            name=str(item.get("name", tid)),
            subject_ids=list(item.get("subject_ids", item.get("subjects_can_teach", []))),
            unavailable_slots=_parse_slot_list(
                item.get("unavailable_slots", []), f"teacher '{tid}'.unavailable_slots"
            ),
            max_weekly_load=int(item.get("max_weekly_load", 20)),
            preferred_slots=_parse_slot_list(
                item.get("preferred_slots", []), f"teacher '{tid}'.preferred_slots"
            ),
        )

    # Parse rooms
    rooms: dict[str, Room] = {}
    for item in raw.get("rooms", []):
        rid = item.get("id") or item.get("name")
        if not rid:
            raise ValidationError("Every room must have an 'id' or 'name'.")
        if rid in rooms:
            raise ValidationError(f"Duplicate room ID '{rid}'.")
        rooms[rid] = Room(
            id=rid,
            name=str(item.get("name", rid)),
            capacity=int(item.get("capacity", 0)),
            type=str(item.get("type", "classroom")).lower(),
        )

    # Parse classes
    classes: dict[str, ClassGroup] = {}
    for item in raw.get("classes", raw.get("batches", [])):
        cid = item.get("id") or item.get("name")
        if not cid:
            raise ValidationError("Every class must have an 'id' or 'name'.")
        if cid in classes:
            raise ValidationError(f"Duplicate class ID '{cid}'.")

        reqs: list[ClassSubjectRequirement] = []
        for req_item in item.get("subjects", []):
            if isinstance(req_item, str):
                raise ValidationError(
                    f"Class '{cid}' subject must be an object with subject_id and teacher_id."
                )
            sid = req_item.get("subject_id") or req_item.get("code")
            tid = req_item.get("teacher_id")
            wp = req_item.get("weekly_periods") if "weekly_periods" in req_item else req_item.get("weekly_hours")
            reqs.append(
                ClassSubjectRequirement(
                    subject_id=str(sid),
                    teacher_id=str(tid),
                    weekly_periods=int(wp) if wp is not None else None,
                )
            )

        classes[cid] = ClassGroup(
            id=cid,
            name=str(item.get("name", cid)),
            strength=int(item.get("strength", 0)),
            lunch_slot=int(item.get("lunch_slot", item.get("lunch_break_slot", 3))),
            subjects=reqs,
        )

    dataset = TimetableDataset(
        name=name,
        time_structure=ts,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
    )

    validate_dataset_schema(dataset)
    return dataset


def load_dataset_from_file(file_path: str | Path) -> TimetableDataset:
    """Reads a JSON file from disk and parses it into a TimetableDataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_dataset_from_dict(data)
