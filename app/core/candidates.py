"""Re-export candidate generation.

The implementation lives in :mod:`app.core.candidate_generator`. This module
keeps the earlier import path working.
"""

from app.core.candidate_generator import (
    candidate_grid,
    cell_candidates,
    peer_coordinates,
)

__all__ = ["candidate_grid", "cell_candidates", "peer_coordinates"]
