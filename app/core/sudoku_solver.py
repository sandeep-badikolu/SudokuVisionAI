"""Backtracking Sudoku search with MRV and forward checking.

Each recursive call receives its own board and candidate grid. A solved grid
is returned only when the search proves that exactly one completion exists.
"""

from dataclasses import dataclass, fields, replace
from enum import Enum

from app.core.candidate_generator import CandidateGrid, candidate_grid, peer_coordinates
from app.core.grid import GridSpec
from app.core.puzzle import Puzzle
from app.core.sudoku_validator import ValidationIssue, validate_puzzle

Grid = tuple[tuple[int, ...], ...]


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


@dataclass(frozen=True, slots=True)
class _Outcome:
    solutions: tuple[Grid, ...]
    statistics: SolveStatistics
    steps: tuple[SolveStep, ...]
    truncated: bool


def solve(puzzle: Puzzle, *, solution_limit: int = 2) -> SolveResult:
    """Validate ``puzzle`` and search for completions without changing it.

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
    board = puzzle.clues
    domains = candidate_grid(spec, board)
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

    outcome = _search(spec, board, domains, solution_limit)
    if len(set(outcome.solutions)) != len(outcome.solutions):
        raise RuntimeError("search recorded the same completion more than once")
    status, solution = _status_and_grid(outcome.solutions, outcome.truncated)
    return SolveResult(
        status=status,
        solution=solution,
        solution_count=len(outcome.solutions),
        statistics=outcome.statistics,
        steps=outcome.steps,
    )


def _require_limit(solution_limit: int) -> None:
    if isinstance(solution_limit, bool) or not isinstance(solution_limit, int):
        raise TypeError("solution_limit must be an int of at least 1")
    if solution_limit < 1:
        raise ValueError("solution_limit must be at least 1")


def _search(
    spec: GridSpec,
    board: Grid,
    domains: CandidateGrid,
    limit: int,
) -> _Outcome:
    """Search a private copy of the board and candidate sets."""
    if limit < 1:
        return _Outcome((), SolveStatistics(), (), True)
    selected = _select_mrv(spec, board, domains)
    if selected is None:
        return _Outcome((board,), SolveStatistics(), (), False)

    row, col = selected
    found: list[Grid] = []
    steps: list[SolveStep] = []
    stats = SolveStatistics(mrv_selections=1)
    truncated = False
    values = tuple(sorted(domains[row][col]))
    for index, value in enumerate(values):
        assigned = _assign(spec, board, domains, row, col, value)
        if assigned is None:
            stats = _combine(stats, forward_check_rejections=1)
            steps.append(SolveStep(StepKind.REJECT, row, col, value))
            continue
        child_board, child_domains, eliminated = assigned
        stats = _combine(stats, assignments=1, candidate_eliminations=eliminated)
        steps.append(SolveStep(StepKind.ASSIGN, row, col, value))
        child = _search(spec, child_board, child_domains, limit - len(found))
        found.extend(child.solutions)
        stats = _merge(stats, child.statistics)
        steps.extend(child.steps)
        stats = _combine(stats, backtracks=1)
        steps.append(SolveStep(StepKind.BACKTRACK, row, col, value))
        skipped_values = index < len(values) - 1 and len(found) >= limit
        if child.truncated or skipped_values:
            truncated = True
            break
    return _Outcome(tuple(found), stats, tuple(steps), truncated)


def _select_mrv(
    spec: GridSpec, board: Grid, domains: CandidateGrid
) -> tuple[int, int] | None:
    """Choose the empty cell with the fewest remaining candidates.

    Ties break toward the smallest row, then the smallest column.
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


def _assign(
    spec: GridSpec,
    board: Grid,
    domains: CandidateGrid,
    row: int,
    col: int,
    value: int,
) -> tuple[Grid, CandidateGrid, int] | None:
    """Return a new board and candidate grid with ``value`` placed.

    ``None`` means a peer would be left with no candidates. The inputs are
    not modified.
    """
    updated = [list(domain_row) for domain_row in domains]
    removed = 0
    for other_row, other_col in peer_coordinates(spec, row, col):
        if board[other_row][other_col] != 0:
            continue
        domain = updated[other_row][other_col]
        if value not in domain:
            continue
        reduced = domain - {value}
        if not reduced:
            return None
        updated[other_row][other_col] = reduced
        removed += 1
    updated[row][col] = frozenset()
    new_row = tuple(
        value if column == col else board[row][column] for column in range(spec.size)
    )
    new_board = tuple(
        new_row if current == row else board[current] for current in range(spec.size)
    )
    new_domains = tuple(tuple(domain_row) for domain_row in updated)
    return new_board, new_domains, removed


def _combine(stats: SolveStatistics, **increments: int) -> SolveStatistics:
    return replace(
        stats,
        **{key: getattr(stats, key) + amount for key, amount in increments.items()},
    )


def _merge(left: SolveStatistics, right: SolveStatistics) -> SolveStatistics:
    return SolveStatistics(
        **{
            field.name: getattr(left, field.name) + getattr(right, field.name)
            for field in fields(SolveStatistics)
        }
    )


def _status_and_grid(
    solutions: tuple[Grid, ...], truncated: bool
) -> tuple[SolveStatus, Grid | None]:
    found = len(solutions)
    if found == 0:
        return SolveStatus.UNSOLVABLE, None
    if found == 1 and not truncated:
        return SolveStatus.UNIQUE, solutions[0]
    if found == 1:
        return SolveStatus.LIMIT_REACHED, None
    return SolveStatus.MULTIPLE, None
