"""Puzzle geometry, validation, candidate sets, and solving."""

from app.core.candidate_generator import (
    candidate_grid,
    cell_candidates,
    peer_coordinates,
)
from app.core.grid import Cell, GridSpec
from app.core.puzzle import Puzzle
from app.core.sudoku_solver import (
    SolveResult,
    SolveStatistics,
    SolveStatus,
    SolveStep,
    StepKind,
    solve,
)
from app.core.sudoku_validator import (
    UnitKind,
    ValidationIssue,
    ValidationResult,
    validate_board,
    validate_puzzle,
)

__all__ = [
    "Cell",
    "GridSpec",
    "Puzzle",
    "SolveResult",
    "SolveStatistics",
    "SolveStatus",
    "SolveStep",
    "StepKind",
    "UnitKind",
    "ValidationIssue",
    "ValidationResult",
    "candidate_grid",
    "cell_candidates",
    "peer_coordinates",
    "solve",
    "validate_board",
    "validate_puzzle",
]
