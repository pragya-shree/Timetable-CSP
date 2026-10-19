"""Core package for AI-Based Timetable Scheduler (CSP formulation & Solvers).

Modules provided:
- Data models and dataset loading/validation (models.py)
- CSP variable generation and Placement representations (sessions.py)
- Hard constraint validation H1-H9 + STRUCT (constraints.py)
- Soft constraint penalties S1-S3 and cost calculation (cost.py)
- Backtracking CSP solvers: Plain, MRV, MRV + Degree, and MRV + Degree + FC (backtracking.py)
- Min-Conflicts Local Search CSP solver (min_conflicts.py)
"""

from core.min_conflicts import (
    MinConflictsResult,
    MinConflictsSolver,
    solve_min_conflicts,
)

__all__ = [
    "MinConflictsResult",
    "MinConflictsSolver",
    "solve_min_conflicts",
]
