"""Core package for AI-Based Timetable Scheduler (CSP formulation & Solvers).

Modules provided:
- Data models and dataset loading/validation (models.py)
- CSP variable generation and Placement representations (sessions.py)
- Hard constraint validation H1-H9 + STRUCT (constraints.py)
- Soft constraint penalties S1-S5 and cost calculation (cost.py)
- Backtracking CSP solvers: Plain, MRV, MRV + Degree, and MRV + Degree + FC (backtracking.py)
- Min-Conflicts Local Search CSP solver (min_conflicts.py)
- Hybrid CSP + Simulated Annealing solver (simulated_annealing.py)
"""

from core.backtracking import (
    BacktrackingResult,
    BacktrackingSolver,
    solve_backtracking,
    solve_backtracking_mrv,
    solve_backtracking_mrv_degree,
    solve_backtracking_mrv_degree_fc,
)
from core.min_conflicts import (
    MinConflictsResult,
    MinConflictsSolver,
    solve_min_conflicts,
)
from core.simulated_annealing import (
    HybridSimulatedAnnealingSolver,
    SimulatedAnnealingResult,
    solve_hybrid_simulated_annealing,
)

__all__ = [
    "BacktrackingResult",
    "BacktrackingSolver",
    "solve_backtracking",
    "solve_backtracking_mrv",
    "solve_backtracking_mrv_degree",
    "solve_backtracking_mrv_degree_fc",
    "MinConflictsResult",
    "MinConflictsSolver",
    "solve_min_conflicts",
    "HybridSimulatedAnnealingSolver",
    "SimulatedAnnealingResult",
    "solve_hybrid_simulated_annealing",
]
