"""Backtracking Sudoku search with MRV and forward checking.

The search works on a copy of the clues. A solved grid is returned only when
the search proves that exactly one completion exists. Finding several
completions does not pick one of them.
"""

from dataclasses import dataclass
from enum import Enum

from app.core.candidates import candidate_grid, peer_coordinates
from app.core.grid import GridSpec
from app.core.puzzle import Puzzle
from app.core.validation import ValidationIssue, validate_puzzle

Grid = tuple[tuple[int, ...], ...]
Domains = list[list[set[int]]]


class SolveStatus(Enum):
    """Outcome of validation plus a bounded solution search."""

    INVALID = "invalid"
    UNSOLVABLE = "unsolvable"
    UNIQUE = "unique"
    MULTIPLE = "multiple"
    LIMIT_REACHED = "limit_reached"


class StepKind(Enum):
    """One recorded action in the search."""

    ASSIGN = "assign"
    BACKTRACK = "backtrack"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class SolveStep:
    """A placement, an undone placement, or a value rejected by forward checking."""

    kind: StepKind
    row: int
    col: int
    value: int


@dataclass(frozen=True, slots=True)
class SolveStatistics:
    """Counts from the search that produced a result.

    ``solution_count`` on the result is capped by the caller's limit, so these
    figures describe the work that was actually done.
    """

    assignments: int = 0
    backtracks: int = 0
    forward_check_rejections: int = 0
    candidate_eliminations: int = 0
    mrv_selections: int = 0
    empty_domains: int = 0


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Structured outcome of :func:`solve`.

    ``solution`` is present only for :attr:`SolveStatus.UNIQUE`. A puzzle with
    several completions yields :attr:`SolveStatus.MULTIPLE` and no grid, even
    when the search stopped at ``solution_limit``. :attr:`SolveStatus.LIMIT_REACHED`
    means exactly one completion was found and the limit stopped the search
    before another completion could be ruled out.
    """

    status: SolveStatus
    solution: Grid | None
    solution_count: int
    statistics: SolveStatistics
    steps: tuple[SolveStep, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(slots=True)
class _Placement:
    row: int
    col: int
    value: int
    domain: set[int]
    removed: list[tuple[int, int]]


@dataclass(slots=True)
class _Counts:
    assignments: int = 0
    backtracks: int = 0
    forward_check_rejections: int = 0
    candidate_eliminations: int = 0
    mrv_selections: int = 0


def solve(puzzle: Puzzle, *, solution_limit: int = 2) -> SolveResult:
    """Validate ``puzzle`` and search for completions on a copy of its clues.

    ``solution_limit`` is the maximum number of completions to collect. It
    must be at least 1. The default of 2 is enough to tell a unique puzzle
    from one with several solutions.
    """
    _require_limit(solution_limit)
    validation = validate_puzzle(puzzle)
    if not validation.valid:
        return SolveResult(
            status=SolveStatus.INVALID,
            solution=None,
            solution_count=0,
            statistics=SolveStatistics(),
            issues=validation.issues,
        )

    spec = puzzle.spec
    board = puzzle.working_board()
    domains: Domains = [
        [set(cell) for cell in row] for row in candidate_grid(spec, board)
    ]
    empty_domains = sum(
        1
        for row in range(spec.size)
        for col in range(spec.size)
        if board[row][col] == 0 and not domains[row][col]
    )
    if empty_domains:
        return SolveResult(
            status=SolveStatus.UNSOLVABLE,
            solution=None,
            solution_count=0,
            statistics=SolveStatistics(empty_domains=empty_domains),
        )

    solutions: list[Grid] = []
    steps: list[SolveStep] = []
    trail: list[_Placement] = []
    counts = _Counts()
    truncated = False

    def search() -> None:
        nonlocal truncated
        if len(solutions) >= solution_limit:
            truncated = True
            return
        selected = _select_mrv(spec, board, domains)
        if selected is None:
            solutions.append(tuple(tuple(row) for row in board))
            return
        counts.mrv_selections += 1
        row, col = selected
        for value in sorted(domains[row][col]):
            if len(solutions) >= solution_limit:
                truncated = True
                return
            eliminations = _place(spec, board, domains, trail, row, col, value)
            if eliminations is None:
                counts.forward_check_rejections += 1
                steps.append(SolveStep(StepKind.REJECT, row, col, value))
                continue
            counts.assignments += 1
            counts.candidate_eliminations += eliminations
            steps.append(SolveStep(StepKind.ASSIGN, row, col, value))
            search()
            _undo(board, domains, trail)
            counts.backtracks += 1
            steps.append(SolveStep(StepKind.BACKTRACK, row, col, value))
            if truncated:
                return

    search()
    _ensure_restored(puzzle, board)
    if len(set(solutions)) != len(solutions):
        raise RuntimeError("search recorded the same completion more than once")

    status, solution = _status_and_grid(solutions, truncated)
    return SolveResult(
        status=status,
        solution=solution,
        solution_count=len(solutions),
        statistics=SolveStatistics(
            assignments=counts.assignments,
            backtracks=counts.backtracks,
            forward_check_rejections=counts.forward_check_rejections,
            candidate_eliminations=counts.candidate_eliminations,
            mrv_selections=counts.mrv_selections,
        ),
        steps=tuple(steps),
    )


def _require_limit(solution_limit: int) -> None:
    if isinstance(solution_limit, bool) or not isinstance(solution_limit, int):
        raise TypeError("solution_limit must be an int of at least 1")
    if solution_limit < 1:
        raise ValueError("solution_limit must be at least 1")


def _select_mrv(
    spec: GridSpec, board: list[list[int]], domains: Domains
) -> tuple[int, int] | None:
    """Choose the empty cell with the fewest remaining candidates.

    Ties break toward the smallest row, then the smallest column, so the
    search order does not depend on set iteration.
    """
    selected: tuple[int, int] | None = None
    smallest = spec.max_digit + 1
    for row in range(spec.size):
        for col in range(spec.size):
            if board[row][col] != 0:
                continue
            remaining = len(domains[row][col])
            if remaining < smallest:
                smallest = remaining
                selected = (row, col)
    return selected


def _place(
    spec: GridSpec,
    board: list[list[int]],
    domains: Domains,
    trail: list[_Placement],
    row: int,
    col: int,
    value: int,
) -> int | None:
    """Assign ``value`` and remove it from peer domains.

    Returns how many peer candidates were removed, or ``None`` when a peer
    would be left with no candidates. A failed attempt restores those peers
    and does not change the board.
    """
    removed: list[tuple[int, int]] = []
    for other_row, other_col in peer_coordinates(spec, row, col):
        if board[other_row][other_col] != 0:
            continue
        peer_domain = domains[other_row][other_col]
        if value not in peer_domain:
            continue
        peer_domain.remove(value)
        removed.append((other_row, other_col))
        if not peer_domain:
            for undo_row, undo_col in removed:
                domains[undo_row][undo_col].add(value)
            return None
    trail.append(_Placement(row, col, value, domains[row][col], removed))
    board[row][col] = value
    domains[row][col] = set()
    return len(removed)


def _undo(board: list[list[int]], domains: Domains, trail: list[_Placement]) -> None:
    placement = trail.pop()
    board[placement.row][placement.col] = 0
    domains[placement.row][placement.col] = placement.domain
    for other_row, other_col in placement.removed:
        domains[other_row][other_col].add(placement.value)


def _ensure_restored(puzzle: Puzzle, board: list[list[int]]) -> None:
    for row in range(puzzle.spec.size):
        for col in range(puzzle.spec.size):
            if board[row][col] != puzzle.clue(row, col):
                raise RuntimeError("solver did not restore the working board")


def _status_and_grid(
    solutions: list[Grid], truncated: bool
) -> tuple[SolveStatus, Grid | None]:
    found = len(solutions)
    if found == 0:
        return SolveStatus.UNSOLVABLE, None
    if found == 1 and not truncated:
        return SolveStatus.UNIQUE, solutions[0]
    if found == 1:
        return SolveStatus.LIMIT_REACHED, None
    return SolveStatus.MULTIPLE, None
