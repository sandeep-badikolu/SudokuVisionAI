"""Streamlit review screen for a photographed Sudoku.

Processing, recognition, validation, and solving stay in the existing modules.
This file only displays their results and records manual corrections.
"""

from __future__ import annotations

import streamlit as st

from app.ui.review import (
    MAX_UPLOAD_BYTES,
    DisplayCell,
    ProcessingError,
    PuzzleReview,
    SolutionCell,
    UploadError,
    process_upload,
)

_BOXES = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


def main() -> None:
    """Render the upload, review, and solution sections."""
    st.set_page_config(
        page_title="SudokuVision AI",
        page_icon=":material/grid_on:",
        layout="wide",
    )
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.title("SudokuVision AI")
    st.write(
        "Upload a Sudoku photograph, review the recognized digits, "
        "correct anything uncertain, then validate and solve."
    )
    st.caption(
        "Template matching can miss a digit. Uncertain cells stay unresolved "
        "until you edit them. This screen does not report a measured "
        "photograph accuracy."
    )
    uploaded = st.file_uploader(
        "Photograph",
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, or JPEG, up to 10 MB.",
    )
    order_choice = st.selectbox(
        "Grid size",
        ("Detect automatically", "4×4", "6×6", "9×9"),
    )
    if st.button("Process image", type="primary", disabled=uploaded is None):
        _process(uploaded, order_choice)
    review = st.session_state.get("review")
    if not isinstance(review, PuzzleReview):
        st.info("Upload a PNG or JPEG, then choose Process image.")
        return
    if uploaded is not None and uploaded.name != review.filename:
        st.warning("This file has not been processed yet. Choose Process image.")
    _images(review)
    _grid_section(review)
    _editor(review)
    _actions(review)


def _process(uploaded: object, order_choice: str) -> None:
    order = None if order_choice == "Detect automatically" else int(order_choice[0])
    payload = uploaded.getvalue()  # type: ignore[attr-defined]
    name = str(uploaded.name)  # type: ignore[attr-defined]
    if len(payload) > MAX_UPLOAD_BYTES:
        st.error("The uploaded file is larger than 10 MB.")
        return
    with st.spinner("Detecting the grid and reading the cells."):
        try:
            review = process_upload(name, payload, order=order)
        except (UploadError, ProcessingError) as exc:
            st.error(str(exc))
            return
    st.session_state.review = review


def _images(review: PuzzleReview) -> None:
    st.subheader("Processing")
    uncertain = sum(cell.uncertain for cell in review.display_cells())
    st.write(
        f"Detected a {review.order}×{review.order} grid. "
        f"{uncertain} cells are uncertain."
    )
    labels = (
        ("original", "Original"),
        ("detected_grid", "Detected grid"),
        ("perspective_grid", "Perspective-corrected"),
        ("contact_sheet", "Cell contact sheet"),
    )
    columns = st.columns(2)
    for index, (key, label) in enumerate(labels):
        image = review.images.get(key)
        with columns[index % 2]:
            if image is None:
                st.caption(f"{label} is unavailable.")
            else:
                st.image(image, caption=label, use_container_width=True)


def _grid_section(review: PuzzleReview) -> None:
    st.subheader("Recognized puzzle")
    grid = _grid_html(review.display_cells(), review.order)
    st.markdown(grid, unsafe_allow_html=True)
    st.caption(
        "Amber cells are uncertain. Indigo cells were edited. A dash is a blank."
    )
    uncertain = [cell for cell in review.display_cells() if cell.uncertain]
    if uncertain:
        with st.expander("Uncertain cells", expanded=True):
            for cell in uncertain:
                st.write(
                    f"Row {cell.row + 1}, column {cell.col + 1}. "
                    f"{_confidence_text(cell)}"
                )


def _editor(review: PuzzleReview) -> None:
    st.subheader("Correct a cell")
    prefix = f"{review.filename}-{review.order}"
    row_col, col_col, digit_col = st.columns(3)
    row = row_col.number_input(
        "Row",
        min_value=1,
        max_value=review.order,
        key=f"{prefix}-row",
    )
    column = col_col.number_input(
        "Column", min_value=1, max_value=review.order, key=f"{prefix}-col"
    )
    choices = ["Blank"] + [str(digit) for digit in range(1, review.order + 1)]
    digit_label = digit_col.selectbox("Value", choices, key=f"{prefix}-digit")
    selected = review.display_cells()[(int(row) - 1) * review.order + (int(column) - 1)]
    st.caption(
        f"OCR value: {_value_text(selected.ocr_value)}. "
        f"Current value: {_value_text(selected.value)}. {_confidence_text(selected)}"
    )
    apply, restore = st.columns(2)
    if apply.button("Apply correction", use_container_width=True):
        digit = 0 if digit_label == "Blank" else int(str(digit_label))
        review.set_cell(int(row) - 1, int(column) - 1, digit)
        st.rerun()
    if restore.button("Restore OCR value", use_container_width=True):
        review.clear_cell(int(row) - 1, int(column) - 1)
        st.rerun()


def _actions(review: PuzzleReview) -> None:
    st.subheader("Validate and solve")
    validate, solve, reset = st.columns(3)
    if validate.button("Validate puzzle", use_container_width=True):
        review.validate()
        st.rerun()
    if solve.button("Solve puzzle", use_container_width=True):
        review.solve_puzzle()
        st.rerun()
    if reset.button("Reset corrections", use_container_width=True):
        review.reset()
        st.rerun()
    if review.check is not None:
        if review.check.status == "valid":
            st.success("The corrected puzzle is valid.")
        elif review.check.errors:
            for error in review.check.errors:
                st.error(error)
    if review.solve_result is not None:
        st.write(review.solve_result.message)
        overlay = review.solution_cells()
        if overlay is not None:
            st.markdown(_solution_html(overlay, review.order), unsafe_allow_html=True)
            st.caption("Bold cells are clues. Blue cells were filled by the solver.")


def _grid_html(cells: tuple[DisplayCell, ...], order: int) -> str:
    box_rows, box_cols = _BOXES[order]
    by_position = {(cell.row, cell.col): cell for cell in cells}
    rows: list[str] = []
    for row in range(order):
        parts = []
        for col in range(order):
            cell = by_position[(row, col)]
            classes = ["sv-cell"]
            if cell.uncertain:
                classes.append("sv-uncertain")
            elif cell.value == 0:
                classes.append("sv-blank")
            if cell.source == "correction":
                classes.append("sv-edited")
            if row % box_rows == 0:
                classes.append("sv-box-top")
            if col % box_cols == 0:
                classes.append("sv-box-left")
            if cell.uncertain:
                label = "?"
            elif cell.value == 0:
                label = "·"
            else:
                label = str(cell.value)
            parts.append(f'<td class="{" ".join(classes)}">{label}</td>')
        rows.append("<tr>" + "".join(parts) + "</tr>")
    return '<table class="sv-grid">' + "".join(rows) + "</table>"


def _solution_html(cells: tuple[SolutionCell, ...], order: int) -> str:
    box_rows, box_cols = _BOXES[order]
    by_position = {(cell.row, cell.col): cell for cell in cells}
    rows: list[str] = []
    for row in range(order):
        parts = []
        for col in range(order):
            cell = by_position[(row, col)]
            classes = ["sv-cell", "sv-clue" if cell.role == "clue" else "sv-generated"]
            if row % box_rows == 0:
                classes.append("sv-box-top")
            if col % box_cols == 0:
                classes.append("sv-box-left")
            parts.append(f'<td class="{" ".join(classes)}">{cell.value}</td>')
        rows.append("<tr>" + "".join(parts) + "</tr>")
    return '<table class="sv-grid">' + "".join(rows) + "</table>"


def _value_text(value: int | None) -> str:
    if value is None:
        return "uncertain"
    if value == 0:
        return "blank"
    return str(value)


def _confidence_text(cell: DisplayCell) -> str:
    if cell.value == 0 and cell.blank_confidence is not None:
        return f"Blank confidence {cell.blank_confidence:.2f}."
    if cell.confidence is not None:
        return f"Digit confidence {cell.confidence:.2f}."
    return "No digit confidence is available."


_STYLE = """
<style>
.sv-grid { border-collapse: collapse; margin: 0.25rem 0 0.75rem; }
.sv-cell {
  width: 2.25rem; height: 2.25rem; text-align: center;
  border: 1px solid #d0d5dd; font-size: 1rem; color: #101828;
}
.sv-box-top { border-top: 2px solid #344054; }
.sv-box-left { border-left: 2px solid #344054; }
.sv-uncertain { background: #fef3c7; color: #92400e; font-weight: 600; }
.sv-blank { color: #98a2b3; }
.sv-edited { background: #eef2ff; }
.sv-clue { font-weight: 700; }
.sv-generated { color: #175cd3; background: #eff8ff; }
</style>
"""


main()
