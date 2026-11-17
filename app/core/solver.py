"""Re-export the Sudoku solver.

The implementation lives in :mod:`app.core.sudoku_solver`. This module keeps
the earlier import path working.
"""

from app.core.sudoku_solver import (
    SolveResult,
    SolveStatistics,
    SolveStatus,
    SolveStep,
    StepKind,
    solve,
)

__all__ = [
    "SolveResult",
    "SolveStatistics",
    "SolveStatus",
    "SolveStep",
    "StepKind",
    "solve",
]
