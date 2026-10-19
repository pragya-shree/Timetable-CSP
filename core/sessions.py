"""CSP Variable and Domain definitions for Timetable Scheduling.

In this CSP formulation:
- Variable: A teaching session (SessionVar). For example, Class A's 1st Math session.
  A 2-period lab is represented as a single SessionVar with duration=2, occupying
  2 consecutive time periods.
- Domain Value: A placement (Placement) defined by (day, start_period, room_id).
- Assignment / Timetable: Mapping from SessionVar to Placement (or None if unassigned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.models import TimetableDataset, ValidationError


@dataclass(frozen=True)
class SessionVar:
    """CSP variable representing an atomic scheduled unit.
    
    Attributes:
        session_id: Unique string key, e.g. "CSE_2A_MATH_s1".
        class_id: Class/batch attending the session.
        subject_id: Subject being taught.
        session_no: 1-indexed counter for this subject and class.
        teacher_id: Pre-assigned teacher for this session.
        duration: Number of consecutive periods occupied (1 for theory, e.g. 2 for labs).
        session_type: "theory" or "lab".
    """
    session_id: str
    class_id: str
    subject_id: str
    session_no: int
    teacher_id: str
    duration: int
    session_type: str  # "theory" | "lab"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValidationError(
                f"SessionVar '{self.session_id}' duration must be > 0, got {self.duration}."
            )
        if self.session_type not in ("theory", "lab"):
            raise ValidationError(
                f"SessionVar '{self.session_id}' session_type must be 'theory' or 'lab'."
            )


@dataclass(frozen=True)
class Placement:
    """CSP domain value representing a time and room allocation.
    
    Attributes:
        day: 0-indexed day (e.g. 0 for Monday).
        start_period: 0-indexed starting period of the session.
        room_id: Identifier of the allocated room.
    """
    day: int
    start_period: int
    room_id: str

    def get_occupied_periods(self, duration: int) -> list[int]:
        """Returns the list of period indices occupied given the session duration."""
        return list(range(self.start_period, self.start_period + duration))

    def get_occupied_slots(self, duration: int) -> list[tuple[int, int]]:
        """Returns the list of (day, period) coordinate tuples occupied."""
        return [(self.day, p) for p in range(self.start_period, self.start_period + duration)]


def generate_session_variables(dataset: TimetableDataset) -> list[SessionVar]:
    """Generates the complete set of CSP variables from class subject requirements.
    
    Rules:
    - Theory subjects: Each session has duration 1. Total sessions = weekly_periods.
    - Lab subjects: Each session has duration = lab_duration (e.g. 2).
      Total sessions = weekly_periods / lab_duration.
      A 2-period lab is ONE session, not two separate sessions.
    
    Returns:
        List of SessionVar objects deterministically sorted.
    """
    variables: list[SessionVar] = []

    for class_id in sorted(dataset.classes.keys()):
        class_group = dataset.classes[class_id]
        for req in class_group.subjects:
            subject = dataset.subjects[req.subject_id]
            weekly_periods = (
                req.weekly_periods if req.weekly_periods is not None else subject.weekly_periods
            )

            if subject.type == "theory":
                duration = 1
                session_count = weekly_periods
            else:  # lab
                duration = subject.lab_duration or 2
                if weekly_periods % duration != 0:
                    raise ValidationError(
                        f"Class '{class_id}' subject '{subject.id}' weekly periods ({weekly_periods}) "
                        f"not divisible by lab duration ({duration})."
                    )
                session_count = weekly_periods // duration

            for s_no in range(1, session_count + 1):
                session_id = f"{class_id}_{subject.id}_s{s_no}"
                var = SessionVar(
                    session_id=session_id,
                    class_id=class_id,
                    subject_id=subject.id,
                    session_no=s_no,
                    teacher_id=req.teacher_id,
                    duration=duration,
                    session_type=subject.type,
                )
                variables.append(var)

    return variables


@dataclass
class Timetable:
    """Represents a complete or partial timetable assignment.
    
    Maps each SessionVar to a Placement (or None if unassigned).
    """
    dataset: TimetableDataset
    assignments: dict[SessionVar, Placement | None] = field(default_factory=dict)

    @classmethod
    def empty(cls, dataset: TimetableDataset) -> Timetable:
        """Creates an empty timetable with all session variables initialized to None."""
        variables = generate_session_variables(dataset)
        return cls(dataset=dataset, assignments={v: None for v in variables})

    def assign(self, session: SessionVar, placement: Placement | None) -> None:
        """Assigns or clears placement for a given session variable."""
        if session not in self.assignments:
            raise KeyError(f"SessionVar '{session.session_id}' not found in timetable.")
        self.assignments[session] = placement

    def is_complete(self) -> bool:
        """Returns True if every required session variable has an assigned Placement."""
        required_vars = generate_session_variables(self.dataset)
        if len(self.assignments) < len(required_vars):
            return False
        for v in required_vars:
            if self.assignments.get(v) is None:
                return False
        return True

    def get_placed_sessions(self) -> list[tuple[SessionVar, Placement]]:
        """Returns all (session, placement) pairs where placement is not None."""
        return [(v, p) for v, p in self.assignments.items() if p is not None]

    def get_unassigned_sessions(self) -> list[SessionVar]:
        """Returns all session variables that have not yet been assigned a placement."""
        return [v for v, p in self.assignments.items() if p is None]

    def to_dict(self) -> dict[str, Any]:
        """Serializes placements to a dictionary format."""
        placements_data = []
        for var, placement in self.assignments.items():
            if placement is not None:
                placements_data.append({
                    "session_id": var.session_id,
                    "class_id": var.class_id,
                    "subject_id": var.subject_id,
                    "session_no": var.session_no,
                    "teacher_id": var.teacher_id,
                    "day": placement.day,
                    "start_period": placement.start_period,
                    "room_id": placement.room_id,
                })
        return {
            "dataset_name": self.dataset.name,
            "is_complete": self.is_complete(),
            "placements": placements_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], dataset: TimetableDataset) -> Timetable:
        """Reconstructs a Timetable from a dictionary of placements."""
        tb = cls.empty(dataset)
        # Create a lookup map for session variables by session_id
        var_by_id = {v.session_id: v for v in tb.assignments.keys()}
        seen_sessions: set[str] = set()

        for item in data.get("placements", []):
            sid = item["session_id"]
            if sid in seen_sessions:
                raise ValidationError(f"Duplicate placement for session '{sid}' in timetable data.")
            seen_sessions.add(sid)

            if sid not in var_by_id:
                raise ValidationError(f"Placement references unknown session_id '{sid}'.")
            var = var_by_id[sid]
            placement = Placement(
                day=int(item["day"]),
                start_period=int(item["start_period"]),
                room_id=str(item["room_id"]),
            )
            tb.assign(var, placement)

        return tb


def get_valid_domain_for_session(
    session: SessionVar, dataset: TimetableDataset
) -> list[Placement]:
    """Enumerates all unary-compatible placements for a given session variable.
    
    A placement (day, start_period, room_id) is unary-compatible if:
    1. Fits completely within the daily periods: start_period + duration <= periods_per_day.
    2. Room type matches session type: 'lab' -> lab rooms, 'theory' -> classrooms.
    3. Room capacity >= class strength.
    4. None of the occupied slots fall on a global blocked period.
    5. None of the occupied slots fall on the teacher's unavailable slots.
    6. None of the occupied slots fall on the class's daily lunch slot.
    
    Note: Dynamic binary conflicts (H1 teacher clash, H2 class clash, H3 room clash)
    depend on other session assignments and are checked during search/validation.
    """
    ts = dataset.time_structure
    class_group = dataset.classes[session.class_id]
    teacher = dataset.teachers[session.teacher_id]

    expected_room_type = "lab" if session.session_type == "lab" else "classroom"
    valid_rooms = [
        r for r in dataset.rooms.values()
        if r.type == expected_room_type and r.capacity >= class_group.strength
    ]

    valid_placements: list[Placement] = []

    for day in range(len(ts.days)):
        for start_p in range(ts.periods_per_day - session.duration + 1):
            occupied = [(day, p) for p in range(start_p, start_p + session.duration)]

            # Check global blocked periods
            if any(ts.is_blocked(d, p) for d, p in occupied):
                continue

            # Check teacher unavailable slots
            if any((d, p) in teacher.unavailable_slots for d, p in occupied):
                continue

            # Check class lunch slot
            if any(p == class_group.lunch_slot for d, p in occupied):
                continue

            # Compatible for all valid rooms
            for room in valid_rooms:
                valid_placements.append(
                    Placement(day=day, start_period=start_p, room_id=room.id)
                )

    return valid_placements
