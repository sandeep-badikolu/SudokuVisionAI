# 4×4 photographed import diagnosis

The eight JPEGs in `incoming_photographed/4x4/` were not imported. The dataset, the 20 evaluation photographs, the CNN checkpoint, and the production line threshold were not changed.

## Where it fails

The failure is the grid-line check inside `analyze_grid_lines`, after the boundary and the warp have already succeeded.

For every file, `detect_grid` finds a quadrilateral with confidence 0.933 to 0.992 and warps it to 360×360. The message `warped region does not contain a regular 4, 6, or 9 grid` is raised only when that warped square does not contain a regular 4, 6, or 9 line grid. It is not the “no quadrilateral” rejection and it is not a perspective-transform failure.

On seven photographs both axes keep exactly three peaks: the two outer borders and the heavy centre line, near 0, 180, and 359, with a gap ratio of about 1.00. A 4×4 grid needs five lines. The two lighter lines, at one quarter and three quarters, stay below the production peak cutoff of 0.45 times the strongest line. Their measured strength is about 0.03 to 0.40 of the bold box lines. Digit ink in the same projection is often as strong as those light lines, so the cutoff is doing what it was set to do.

`4*4 8.jpeg` is the same stage and a weaker case. The horizontal projection keeps one peak, the centre line. The vertical projection keeps two. The outer border does not clear 0.45 of that centre line. The warp still exists. The page is softer than the other seven.

## What it is not

The outer boundary is found. Perspective correction produces a square whose bold lines sit on a regular half-grid. The order is not misclassified as 6 or 9. The page around the puzzle is outside the warp. The comparison photograph `046d0ec1-360e-4df3-b668-961a70b2c88d.jpg`, already in the dataset, has all five 4×4 lines at 0.99 to 1.00 of the strongest line, so the same check accepts it.

## Diagnostics

Each rejected file has a folder under `evaluation_output/grid_diagnostics/4x4_incoming/`:

| File | What it is |
| --- | --- |
| `01_original.jpg` | The camera image |
| `02_gray.png` | Grayscale |
| `03_threshold.png` | Adaptive threshold used to find the boundary |
| `04_overlay.jpg` | Detected quadrilateral on the original |
| `05_warped.jpg` | Perspective-corrected square |
| `06_horizontal_profile.png` | Horizontal projection and the peaks that passed |
| `07_vertical_profile.png` | Vertical projection and the peaks that passed |
| `08_lines_on_warp.jpg` | Red and blue for accepted peaks, green for the missing quarter positions |

The green lines on the sharper photographs fall on the printed light lines. That shows the warp is aligned. It does not put those light lines over the 0.45 cutoff.

## Recommended next action

Do not lower the 0.45 line cutoff and do not add an equal-slice bypass. On several of these pages the light line is no stronger in the projection than the digits beside it. A global change would accept strokes that are not grid lines. An equal slice from only the three bold lines would also accept a rectangle with one centre cross and no 4×4 lines at all.

Recapture the puzzles so the grid fills more of the frame and the light lines are as dark as the box lines, closer to the 4×4 photographs that already import. Leave `4*4 8.jpeg` out until a sharper frame shows five lines on both axes. The importer should keep refusing this batch.

## Tests

`tests/unit/test_photographed_grid_rejection.py` locks the 0.45 cutoff and the rejection of `4*4 1.jpeg`, `4*4 7.jpeg`, and `4*4 8.jpeg` after a successful warp. Ruff passed. Pytest passed: 322 tests.
