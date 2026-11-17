"""Puzzle geometry, validation, candidate sets, and solving."""

from app.core.candidates import candidate_grid, cell_candidates, peer_coordinates
from app.core.grid import Cell, GridSpec
from app.core.puzzle import Puzzle
from app.core.solver import (
    SolveResult,
    SolveStatistics,
    SolveStatus,
    SolveStep,
    StepKind,
    solve,
)
from app.core.validation import (
    UnitKind,
    ValidationIssue,
    ValidationResult,
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
    "validate_puzzle",
]
