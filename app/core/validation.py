"""Re-export the Sudoku validator.

The implementation lives in :mod:`app.core.sudoku_validator`. This module
keeps the earlier import path working.
"""

from app.core.sudoku_validator import (
    UnitKind,
    ValidationIssue,
    ValidationResult,
    validate_board,
    validate_puzzle,
)

__all__ = [
    "UnitKind",
    "ValidationIssue",
    "ValidationResult",
    "validate_board",
    "validate_puzzle",
]
