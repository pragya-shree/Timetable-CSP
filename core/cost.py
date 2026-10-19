"""Soft constraints evaluation and objective cost function calculation.

Soft constraints represent educational quality preferences:
- S1: Minimize free periods (gaps) in a class's daily schedule.
- S2: Spread subject lectures across different days (avoid duplicate theory lectures on the same day).
- S3: Ensure lab sessions occupy consecutive periods (penalize interrupted or split lab sessions).

Cost Function:
    total_cost = 1000 * hard_violation_count + sum(weighted soft penalties)

Exact Definition of hard_violations:
    'hard_violations' is the total count of atomic violation records across H1-H9 + STRUCT:
    - H1, H2, H3: 1 violation record per conflicting pair of sessions per overlapping occupied period.
    - H4: 1 violation record per session placed in an undersized room.
    - H5: 1 violation record per session placed in an incompatible room type.
    - H6: 1 violation record per occupied slot falling on an unavailable or global blocked slot.
    - H7: 1 violation record per missing or excess weekly period per (class, subject).
    - H8: 1 violation record per excess assigned period over a teacher's max weekly load.
    - H9: 1 violation record per occupied slot falling on the class's daily lunch break.
    - STRUCT: 1 violation record per structural integrity defect.

Incomplete Timetable Cost Guarantee:
    Unassigned sessions are never ignored. Every unassigned period is counted as an H7
    quota deficit (1 violation record per missing period). A timetable with M missing
    periods automatically incurs at least 1000 * M hard cost, and flags is_complete=False
    and is_valid=False. An incomplete timetable can never receive a deceptively low cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.constraints import validate_timetable
from core.sessions import Timetable


@dataclass(frozen=True)
class SoftWeights:
    """Configurable penalty multipliers for soft constraint evaluation."""
    w_s1: int = 1  # Weight for class timetable gaps
    w_s2: int = 1  # Weight for same-subject duplicate lectures in a day
    w_s3: int = 1  # Weight for non-consecutive or interrupted lab periods


@dataclass
class CostBreakdown:
    """Detailed decomposition of hard and soft penalty contributions."""
    hard_violations: int
    hard_cost: int
    s1_gaps: int
    s1_cost: int
    s2_repeats: int
    s2_cost: int
    s3_non_consecutive: int
    s3_cost: int
    total_soft_cost: int
    total_cost: int
    is_complete: bool
    is_valid: bool


def calculate_s1_gaps(timetable: Timetable) -> int:
    """S1: Count free periods (idle gaps) in each class's daily schedule.
    
    A gap is an empty period strictly between the first and last occupied period
    of a class on a given day.
    Exclusions:
    - The class's official lunch break is NOT a gap.
    - Global blocked periods are NOT gaps.
    """
    total_gaps = 0
    dataset = timetable.dataset
    ts = dataset.time_structure

    # Map (class_id, day) -> set of occupied period indices
    class_day_periods: dict[tuple[str, int], set[int]] = defaultdict(set)
    for var, placement in timetable.get_placed_sessions():
        for slot in placement.get_occupied_slots(var.duration):
            class_day_periods[(var.class_id, slot[0])].add(slot[1])

    for (class_id, day), periods in class_day_periods.items():
        if len(periods) <= 1:
            continue
        p_min = min(periods)
        p_max = max(periods)
        class_group = dataset.classes[class_id]

        for p in range(p_min + 1, p_max):
            if p not in periods:
                # Ignore class lunch break
                if p == class_group.lunch_slot:
                    continue
                # Ignore global blocked periods
                if ts.is_blocked(day, p):
                    continue
                total_gaps += 1

    return total_gaps


def calculate_s2_repeats(timetable: Timetable) -> int:
    """S2: Penalize multiple theory lectures of the same subject on the same day.
    
    For each (class, subject, day), counts sessions beyond the first.
    Lab sessions are strictly exempt: a 2-period lab is 1 session and is not a repeat.
    """
    total_repeats = 0
    dataset = timetable.dataset

    # Map (class_id, subject_id, day) -> count of placed sessions
    subject_day_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for var, placement in timetable.get_placed_sessions():
        subject = dataset.subjects.get(var.subject_id)
        if subject and subject.type == "theory":
            subject_day_counts[(var.class_id, var.subject_id, placement.day)] += 1

    for count in subject_day_counts.values():
        if count > 1:
            total_repeats += (count - 1)

    return total_repeats


def calculate_s3_non_consecutive_labs(timetable: Timetable) -> int:
    """S3: Ensure lab sessions occupy consecutive periods without interruption.
    
    Penalizes:
    1. Interrupted lab placement: A multi-period lab whose occupied slots span
       over a class lunch slot or a global blocked period.
    2. Multiple lab sessions of the same lab subject scheduled on the same day
       for the same class that are placed with a gap between them.
    """
    penalties = 0
    dataset = timetable.dataset
    ts = dataset.time_structure

    # 1. Check for single lab placements interrupted by lunch or blocked periods
    for var, placement in timetable.get_placed_sessions():
        if var.session_type == "lab" and var.duration > 1:
            class_group = dataset.classes.get(var.class_id)
            occupied = placement.get_occupied_slots(var.duration)
            for day, p in occupied:
                # If a lab slot falls on lunch or blocked slot, the lab is interrupted
                if class_group and p == class_group.lunch_slot:
                    penalties += 1
                if ts.is_blocked(day, p):
                    penalties += 1

    # 2. Check for multiple lab sessions of same subject on same day separated by a gap
    lab_sessions_by_day: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)
    for var, placement in timetable.get_placed_sessions():
        if var.session_type == "lab":
            lab_sessions_by_day[(var.class_id, var.subject_id, placement.day)].append(
                (placement.start_period, placement.start_period + var.duration)
            )

    for sessions in lab_sessions_by_day.values():
        if len(sessions) > 1:
            sorted_sessions = sorted(sessions, key=lambda s: s[0])
            for i in range(len(sorted_sessions) - 1):
                end_current = sorted_sessions[i][1]
                start_next = sorted_sessions[i + 1][0]
                if start_next > end_current:
                    # Gap between lab sessions on same day
                    penalties += (start_next - end_current)

    return penalties


def calculate_cost(
    timetable: Timetable,
    weights: SoftWeights | None = None,
) -> CostBreakdown:
    """Computes total cost and detailed breakdown of hard and soft constraint violations.
    
    Cost Formula:
        total_cost = 1000 * hard_violations + (w1 * S1 + w2 * S2 + w3 * S3)
    """
    if weights is None:
        weights = SoftWeights()

    # Validate hard constraints
    val_result = validate_timetable(timetable)
    hard_violations = val_result.hard_violation_count
    hard_cost = 1000 * hard_violations

    # Evaluate soft constraints on placed sessions
    s1_gaps = calculate_s1_gaps(timetable)
    s1_cost = s1_gaps * weights.w_s1

    s2_repeats = calculate_s2_repeats(timetable)
    s2_cost = s2_repeats * weights.w_s2

    s3_non_consec = calculate_s3_non_consecutive_labs(timetable)
    s3_cost = s3_non_consec * weights.w_s3

    total_soft_cost = s1_cost + s2_cost + s3_cost
    total_cost = hard_cost + total_soft_cost

    return CostBreakdown(
        hard_violations=hard_violations,
        hard_cost=hard_cost,
        s1_gaps=s1_gaps,
        s1_cost=s1_cost,
        s2_repeats=s2_repeats,
        s2_cost=s2_cost,
        s3_non_consecutive=s3_non_consec,
        s3_cost=s3_cost,
        total_soft_cost=total_soft_cost,
        total_cost=total_cost,
        is_complete=val_result.is_complete,
        is_valid=val_result.is_valid,
    )
