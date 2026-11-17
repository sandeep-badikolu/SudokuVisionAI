
# SudokuVision AI - Project Instructions

## Project Objective
Build a Python-based application that accepts Sudoku
photographs and screenshots, recognizes the puzzle,
solves it, and displays the solution.

## Role
Act as a Senior Python Engineer, Computer Vision
Architect and AI/ML Engineer.

## Technology Stack
- Python 3.12+
- OpenCV
- NumPy
- Pillow
- PyTorch
- Streamlit
- Pytest
- Ruff

## Supported Sudoku Types
- 4x4 grid with 2x2 boxes
- 6x6 grid with 2x3 boxes
- 9x9 grid with 3x3 boxes

## Architecture Principles
- Separate computer vision from puzzle solving.
- Separate digit recognition from the solver.
- Use modular, maintainable Python code.
- Apply SOLID principles where appropriate.
- Use type hints and docstrings.
- Avoid unnecessary dependencies.
- Never hardcode puzzle solutions.
- Preserve the original puzzle.
- Never silently accept invalid puzzles.

## Sudoku Solver Requirements
- Validate rows, columns and boxes.
- Support backtracking.
- Implement Minimum Remaining Values (MRV).
- Implement forward checking.
- Detect zero, one or multiple solutions.
- Record actual solving steps.
- Distinguish invalid puzzles from unsolvable puzzles.

## Computer Vision Requirements
- Detect the Sudoku boundary.
- Correct image perspective.
- Extract individual cells.
- Remove grid lines without destroying digits.
- Handle image rotation and varying lighting.
- Produce diagnostic images.

## Machine Learning Requirements
- Start with a baseline digit recognizer.
- Later introduce a PyTorch CNN.
- Return confidence scores.
- Request manual correction for uncertain digits.
- Evaluate using appropriate test datasets.
- Never fabricate accuracy metrics.

## Testing
- Use pytest.
- Test valid, invalid and unsolvable puzzles.
- Test multiple solutions.
- Test all supported grid sizes.
- Keep tests independent of the UI.

## Development Workflow
- Implement one milestone at a time.
- Do not generate the entire application at once.
- Explain the files being created.
- Run tests after every major implementation.
- Fix failures before proceeding.
- Ask before introducing major dependencies.
- Never overwrite working code without explaining why.
