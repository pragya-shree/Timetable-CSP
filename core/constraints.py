"""Hard constraint validation module (H1 - H9 + STRUCT).

Every college timetable must strictly satisfy all hard constraints.
This module implements individual validators for each constraint:
- H1: Teacher conflict (cannot teach two sessions at the same time).
- H2: Class conflict (cannot attend two sessions at the same time).
- H3: Room conflict (cannot host two sessions at the same time).
- H4: Room capacity (room capacity must be at least class strength).
- H5: Room type (theory in classroom, lab in lab).
- H6: Teacher availability & blocked periods (respect teacher unavailable slots and breaks).
- H7: Subject weekly hours (syllabus quota: |assigned - required| periods).
- H8: Teacher maximum weekly load (assigned periods <= max_weekly_load).
- H9: Class lunch break (no session scheduled during class lunch slot).
- STRUCT: Structural integrity (valid IDs, within grid bounds, teacher qualification).

Result is aggregated into a ValidationResult.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from core.sessions import Placement, SessionVar, Timetable


@dataclass(frozen=True)
class Violation:
    """Represents a single constraint violation in a timetable.
    
    Attributes:
        constraint_id: Short constraint code (e.g. 'H1', 'H4', 'STRUCT').
        message: Descriptive explanation of the clash.
        sessions_involved: List of session IDs associated with the violation.
        details: Additional diagnostic metadata (e.g. slot, room, teacher).
    """
    constraint_id: str
    message: str
    sessions_involved: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Aggregated timetable validation outcome.
    
    Attributes:
        violations: Full list of all detected Violation objects.
        hard_violation_count: Total number of hard violations.
        is_complete: True iff all session variables have been assigned.
        is_valid: True ONLY if is_complete is True AND hard_violation_count == 0.
    """
    violations: list[Violation]
    hard_violation_count: int
    is_complete: bool
    is_valid: bool


def check_h1_teacher_clash(timetable: Timetable) -> list[Violation]:
    """H1: A teacher cannot teach two sessions at the same time.
    
    Counting rule: 1 violation per conflicting pair of sessions per overlapping period.
    """
    violations: list[Violation] = []
    # Map (day, period, teacher_id) -> list of SessionVars
    slot_map: dict[tuple[int, int, str], list[SessionVar]] = defaultdict(list)

    for var, placement in timetable.get_placed_sessions():
        for slot in placement.get_occupied_slots(var.duration):
            slot_map[(slot[0], slot[1], var.teacher_id)].append(var)

    for (day, period, teacher_id), sessions in sorted(slot_map.items()):
        if len(sessions) > 1:
            for s1, s2 in combinations(sorted(sessions, key=lambda s: s.session_id), 2):
                violations.append(
                    Violation(
                        constraint_id="H1",
                        message=(
                            f"Teacher '{teacher_id}' is double-booked on day {day}, "
                            f"period {period} between '{s1.session_id}' and '{s2.session_id}'."
                        ),
                        sessions_involved=[s1.session_id, s2.session_id],
                        details={"day": day, "period": period, "teacher_id": teacher_id},
                    )
                )

    return violations


def check_h2_class_clash(timetable: Timetable) -> list[Violation]:
    """H2: A class cannot have two sessions at the same time.
    
    Counting rule: 1 violation per conflicting pair of sessions per overlapping period.
    """
    violations: list[Violation] = []
    slot_map: dict[tuple[int, int, str], list[SessionVar]] = defaultdict(list)

    for var, placement in timetable.get_placed_sessions():
        for slot in placement.get_occupied_slots(var.duration):
            slot_map[(slot[0], slot[1], var.class_id)].append(var)

    for (day, period, class_id), sessions in sorted(slot_map.items()):
        if len(sessions) > 1:
            for s1, s2 in combinations(sorted(sessions, key=lambda s: s.session_id), 2):
                violations.append(
                    Violation(
                        constraint_id="H2",
                        message=(
                            f"Class '{class_id}' has multiple sessions on day {day}, "
                            f"period {period} ('{s1.session_id}' and '{s2.session_id}')."
                        ),
                        sessions_involved=[s1.session_id, s2.session_id],
                        details={"day": day, "period": period, "class_id": class_id},
                    )
                )

    return violations


def check_h3_room_clash(timetable: Timetable) -> list[Violation]:
    """H3: A room cannot be used for two sessions at the same time.
    
    Counting rule: 1 violation per conflicting pair of sessions per overlapping period.
    """
    violations: list[Violation] = []
    slot_map: dict[tuple[int, int, str], list[SessionVar]] = defaultdict(list)

    for var, placement in timetable.get_placed_sessions():
        for slot in placement.get_occupied_slots(var.duration):
            slot_map[(slot[0], slot[1], placement.room_id)].append(var)

    for (day, period, room_id), sessions in sorted(slot_map.items()):
        if len(sessions) > 1:
            for s1, s2 in combinations(sorted(sessions, key=lambda s: s.session_id), 2):
                violations.append(
                    Violation(
                        constraint_id="H3",
                        message=(
                            f"Room '{room_id}' is double-booked on day {day}, "
                            f"period {period} between '{s1.session_id}' and '{s2.session_id}'."
                        ),
                        sessions_involved=[s1.session_id, s2.session_id],
                        details={"day": day, "period": period, "room_id": room_id},
                    )
                )

    return violations


def check_h4_room_capacity(timetable: Timetable) -> list[Violation]:
    """H4: Room capacity must be at least the class strength.
    
    Counting rule: 1 violation per session placed in an undersized room.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset

    for var, placement in timetable.get_placed_sessions():
        room = dataset.rooms.get(placement.room_id)
        class_group = dataset.classes.get(var.class_id)
        if room and class_group:
            if room.capacity < class_group.strength:
                violations.append(
                    Violation(
                        constraint_id="H4",
                        message=(
                            f"Room '{room.id}' capacity ({room.capacity}) is smaller than "
                            f"class '{class_group.id}' strength ({class_group.strength}) "
                            f"for session '{var.session_id}'."
                        ),
                        sessions_involved=[var.session_id],
                        details={
                            "room_id": room.id,
                            "room_capacity": room.capacity,
                            "class_strength": class_group.strength,
                        },
                    )
                )

    return violations


def check_h5_room_type(timetable: Timetable) -> list[Violation]:
    """H5: Lab sessions must be in lab rooms; theory sessions in classrooms.
    
    Counting rule: 1 violation per session placed in an incompatible room type.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset

    for var, placement in timetable.get_placed_sessions():
        room = dataset.rooms.get(placement.room_id)
        if room:
            if var.session_type == "lab" and room.type != "lab":
                violations.append(
                    Violation(
                        constraint_id="H5",
                        message=(
                            f"Lab session '{var.session_id}' placed in non-lab room "
                            f"'{room.id}' (type '{room.type}')."
                        ),
                        sessions_involved=[var.session_id],
                        details={"room_id": room.id, "expected_type": "lab", "actual_type": room.type},
                    )
                )
            elif var.session_type == "theory" and room.type != "classroom":
                violations.append(
                    Violation(
                        constraint_id="H5",
                        message=(
                            f"Theory session '{var.session_id}' placed in non-classroom room "
                            f"'{room.id}' (type '{room.type}')."
                        ),
                        sessions_involved=[var.session_id],
                        details={"room_id": room.id, "expected_type": "classroom", "actual_type": room.type},
                    )
                )

    return violations


def check_h6_teacher_availability(timetable: Timetable) -> list[Violation]:
    """H6: A teacher can only be scheduled in their available slots (and not during blocked periods).
    
    Counting rule: 1 violation per occupied slot falling on an unavailable or blocked slot.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset
    ts = dataset.time_structure

    for var, placement in timetable.get_placed_sessions():
        teacher = dataset.teachers.get(var.teacher_id)

        for day, period in placement.get_occupied_slots(var.duration):
            # Check global blocked periods (applies to all sessions)
            if ts.is_blocked(day, period):
                violations.append(
                    Violation(
                        constraint_id="H6",
                        message=(
                            f"Session '{var.session_id}' occupies global blocked period "
                            f"on day {day}, period {period}."
                        ),
                        sessions_involved=[var.session_id],
                        details={"day": day, "period": period, "reason": "global_blocked_period"},
                    )
                )
            # Check teacher personal unavailability
            if teacher and (day, period) in teacher.unavailable_slots:
                violations.append(
                    Violation(
                        constraint_id="H6",
                        message=(
                            f"Teacher '{teacher.id}' scheduled during unavailable slot "
                            f"(day {day}, period {period}) for session '{var.session_id}'."
                        ),
                        sessions_involved=[var.session_id],
                        details={"day": day, "period": period, "teacher_id": teacher.id},
                    )
                )

    return violations


def check_h7_weekly_periods(timetable: Timetable) -> list[Violation]:
    """H7: Every subject gets exactly its required weekly hours.
    
    Counting rule: |assigned_periods - required_periods| violations per (class, subject).
    Unassigned sessions contribute here as missing periods!
    """
    violations: list[Violation] = []
    dataset = timetable.dataset

    # Count assigned periods per (class_id, subject_id)
    assigned_periods: dict[tuple[str, str], int] = defaultdict(int)
    for var, placement in timetable.get_placed_sessions():
        assigned_periods[(var.class_id, var.subject_id)] += var.duration

    # Collect all (class_id, subject_id) pairs from both requirements and actual assignments
    all_pairs: set[tuple[str, str]] = set()
    for class_id, class_group in dataset.classes.items():
        for req in class_group.subjects:
            all_pairs.add((class_id, req.subject_id))
    for pair in assigned_periods.keys():
        all_pairs.add(pair)

    for class_id, subject_id in sorted(all_pairs):
        class_group = dataset.classes.get(class_id)
        req = class_group.get_subject_requirement(subject_id) if class_group else None
        if req is not None:
            required = dataset.get_required_periods(class_id, subject_id)
        else:
            required = 0

        assigned = assigned_periods[(class_id, subject_id)]
        discrepancy = abs(assigned - required)

        if discrepancy > 0:
            deficit_or_excess = "missing" if assigned < required else "excess"
            for step in range(1, discrepancy + 1):
                violations.append(
                    Violation(
                        constraint_id="H7",
                        message=(
                            f"Class '{class_id}' subject '{subject_id}' has {discrepancy} "
                            f"{deficit_or_excess} period(s) (assigned {assigned}, required {required}). "
                            f"[Unit {step}/{discrepancy}]"
                        ),
                        sessions_involved=[],
                        details={
                            "class_id": class_id,
                            "subject_id": subject_id,
                            "assigned": assigned,
                            "required": required,
                            "discrepancy": discrepancy,
                        },
                    )
                )

    return violations


def check_h8_teacher_max_load(timetable: Timetable) -> list[Violation]:
    """H8: A teacher's load should not exceed their maximum weekly load.
    
    Counting rule: max(0, assigned_periods - max_load) violations per teacher.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset

    # Count total assigned periods per teacher
    teacher_loads: dict[str, int] = defaultdict(int)
    for var, placement in timetable.get_placed_sessions():
        teacher_loads[var.teacher_id] += var.duration

    for teacher_id, teacher in sorted(dataset.teachers.items()):
        assigned = teacher_loads[teacher_id]
        excess = max(0, assigned - teacher.max_weekly_load)
        if excess > 0:
            for step in range(1, excess + 1):
                violations.append(
                    Violation(
                        constraint_id="H8",
                        message=(
                            f"Teacher '{teacher_id}' exceeds max weekly load ({teacher.max_weekly_load}) "
                            f"with {assigned} assigned periods. [Excess period {step}/{excess}]"
                        ),
                        sessions_involved=[],
                        details={
                            "teacher_id": teacher_id,
                            "assigned_load": assigned,
                            "max_load": teacher.max_weekly_load,
                            "excess": excess,
                        },
                    )
                )

    return violations


def check_h9_class_lunch_break(timetable: Timetable) -> list[Violation]:
    """H9: Each class must have its lunch break in its assigned time slot.
    
    Counting rule: 1 violation per occupied slot falling on the class's lunch slot.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset

    for var, placement in timetable.get_placed_sessions():
        class_group = dataset.classes.get(var.class_id)
        if not class_group:
            continue

        for day, period in placement.get_occupied_slots(var.duration):
            if period == class_group.lunch_slot:
                violations.append(
                    Violation(
                        constraint_id="H9",
                        message=(
                            f"Session '{var.session_id}' overlaps with class '{class_group.id}' "
                            f"lunch slot on day {day}, period {period}."
                        ),
                        sessions_involved=[var.session_id],
                        details={"class_id": class_group.id, "day": day, "lunch_period": period},
                    )
                )

    return violations


def check_structural_validity(timetable: Timetable) -> list[Violation]:
    """STRUCT: Checks that placements conform to grid bounds and entity references.
    
    Checks:
    - Day within [0, len(days) - 1].
    - Start period >= 0.
    - Session does not exceed periods_per_day (start_period + duration <= periods_per_day).
    - Room exists in dataset.
    - Teacher exists and is qualified for the subject.
    """
    violations: list[Violation] = []
    dataset = timetable.dataset
    ts = dataset.time_structure

    for var, placement in timetable.get_placed_sessions():
        # Day bounds
        if not (0 <= placement.day < len(ts.days)):
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=(
                        f"Session '{var.session_id}' placed on invalid day {placement.day} "
                        f"(valid: 0 to {len(ts.days) - 1})."
                    ),
                    sessions_involved=[var.session_id],
                    details={"field": "day", "value": placement.day},
                )
            )

        # Period bounds
        if placement.start_period < 0:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=(
                        f"Session '{var.session_id}' placed at negative start_period "
                        f"{placement.start_period}."
                    ),
                    sessions_involved=[var.session_id],
                    details={"field": "start_period", "value": placement.start_period},
                )
            )
        elif placement.start_period + var.duration > ts.periods_per_day:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=(
                        f"Session '{var.session_id}' with duration {var.duration} extends past end "
                        f"of day: start {placement.start_period} + duration {var.duration} > {ts.periods_per_day}."
                    ),
                    sessions_involved=[var.session_id],
                    details={
                        "start_period": placement.start_period,
                        "duration": var.duration,
                        "periods_per_day": ts.periods_per_day,
                    },
                )
            )

        # Class existence check
        class_group = dataset.classes.get(var.class_id)
        if not class_group:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=f"Session '{var.session_id}' references unknown class '{var.class_id}'.",
                    sessions_involved=[var.session_id],
                    details={"class_id": var.class_id},
                )
            )

        # Subject existence check
        subject = dataset.subjects.get(var.subject_id)
        if not subject:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=f"Session '{var.session_id}' references unknown subject '{var.subject_id}'.",
                    sessions_involved=[var.session_id],
                    details={"subject_id": var.subject_id},
                )
            )
        else:
            # Verify session_type and duration match subject definition
            expected_duration = subject.lab_duration if subject.type == "lab" else 1
            if var.duration != expected_duration:
                violations.append(
                    Violation(
                        constraint_id="STRUCT",
                        message=(
                            f"Session '{var.session_id}' duration {var.duration} does not match "
                            f"subject '{subject.id}' expected duration {expected_duration}."
                        ),
                        sessions_involved=[var.session_id],
                        details={"duration": var.duration, "expected_duration": expected_duration},
                    )
                )
            if var.session_type != subject.type:
                violations.append(
                    Violation(
                        constraint_id="STRUCT",
                        message=(
                            f"Session '{var.session_id}' session_type '{var.session_type}' does not "
                            f"match subject '{subject.id}' type '{subject.type}'."
                        ),
                        sessions_involved=[var.session_id],
                        details={"session_type": var.session_type, "expected_type": subject.type},
                    )
                )

        # Class enrollment and designated teacher binding check
        if class_group and subject:
            req = class_group.get_subject_requirement(var.subject_id)
            if not req:
                violations.append(
                    Violation(
                        constraint_id="STRUCT",
                        message=f"Class '{class_group.id}' is not enrolled in subject '{subject.id}'.",
                        sessions_involved=[var.session_id],
                        details={"class_id": class_group.id, "subject_id": subject.id},
                    )
                )
            elif req.teacher_id != var.teacher_id:
                violations.append(
                    Violation(
                        constraint_id="STRUCT",
                        message=(
                            f"Session '{var.session_id}' has teacher '{var.teacher_id}', but "
                            f"class '{class_group.id}' requires teacher '{req.teacher_id}' for subject '{subject.id}'."
                        ),
                        sessions_involved=[var.session_id],
                        details={"teacher_id": var.teacher_id, "expected_teacher_id": req.teacher_id},
                    )
                )

        # Room existence
        if placement.room_id not in dataset.rooms:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=f"Session '{var.session_id}' placed in unknown room '{placement.room_id}'.",
                    sessions_involved=[var.session_id],
                    details={"room_id": placement.room_id},
                )
            )

        # Teacher qualification check
        teacher = dataset.teachers.get(var.teacher_id)
        if not teacher:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=f"Session '{var.session_id}' has unknown teacher '{var.teacher_id}'.",
                    sessions_involved=[var.session_id],
                    details={"teacher_id": var.teacher_id},
                )
            )
        elif var.subject_id not in teacher.subject_ids:
            violations.append(
                Violation(
                    constraint_id="STRUCT",
                    message=(
                        f"Teacher '{teacher.id}' for session '{var.session_id}' is not "
                        f"qualified to teach subject '{var.subject_id}'."
                    ),
                    sessions_involved=[var.session_id],
                    details={"teacher_id": teacher.id, "subject_id": var.subject_id},
                )
            )

    return violations


def validate_timetable(timetable: Timetable) -> ValidationResult:
    """Runs all hard constraint checks and returns the comprehensive ValidationResult.
    
    A timetable is marked is_valid == True ONLY if:
    1. Every session variable has a placement (is_complete == True).
    2. Zero hard violations exist across H1-H9 and STRUCT.
    """
    violations: list[Violation] = []

    # Run structural checks first
    violations.extend(check_structural_validity(timetable))

    # Run H1 - H9 checks
    violations.extend(check_h1_teacher_clash(timetable))
    violations.extend(check_h2_class_clash(timetable))
    violations.extend(check_h3_room_clash(timetable))
    violations.extend(check_h4_room_capacity(timetable))
    violations.extend(check_h5_room_type(timetable))
    violations.extend(check_h6_teacher_availability(timetable))
    violations.extend(check_h7_weekly_periods(timetable))
    violations.extend(check_h8_teacher_max_load(timetable))
    violations.extend(check_h9_class_lunch_break(timetable))

    is_comp = timetable.is_complete()
    hard_count = len(violations)
    is_val = is_comp and (hard_count == 0)

    return ValidationResult(
        violations=violations,
        hard_violation_count=hard_count,
        is_complete=is_comp,
        is_valid=is_val,
    )
