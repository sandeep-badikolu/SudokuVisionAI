# Milestone 3.27 — Genuine camera dataset collection

The import workflow is ready. No camera photographs were added in this milestone, and none are in the dataset. The saved CNN checkpoint was not loaded or changed. Template OCR was not changed. The 20 evaluation photographs are still the frozen evaluation split.

**GO** for continuing collection and for reserving the new photographed holdout before any future training.

This is not a generalisation result. That evaluation has not been run.

## Import

Put the camera files somewhere outside the dataset, then import them explicitly:

```bash
python -m scripts.import_photographed \
  --dataset evaluation_output/dataset \
  --provenance photographed \
  --pool holdout \
  --confirm \
  path/to/puzzle-1.jpg path/to/puzzle-2.jpg
```

`--pool train` uses the same checks for photographs that are meant for a later training pool. Reserve the holdout photographs first, across 4×4, 6×6, and 9×9, and only then import a separate batch with `--pool train`.

Each accepted file is copied into the dataset, its cells are extracted, and every digit is left empty. A 4×4 grid produces 16 cells, a 6×6 produces 36, and a 9×9 produces 81. If any file in the batch is rejected, the batch is removed and the manifest is restored.

The importer refuses the batch when:

- `--confirm` is missing, or `--provenance` is not the word `photographed`.
- The file is an exact duplicate of a photograph already stored, including one of the frozen evaluation photographs.
- The average hash is within distance 4 of a frozen evaluation photograph.
- A photograph with the same filename is already present.
- Grid detection does not find a 4, 6, or 9 grid.

A photograph that is only similar to an existing template photograph is imported and reported as a warning. It is not dropped into the evaluation split.

## Provenance

`source_type=photographed` is written only when the command confirms that word. The folder the file came from is not read as provenance. A file is not marked synthetic or unknown by this command. Existing rows keep the source they already have.

## Annotation

Labels stay manual. Open the Ground truth page and choose the new photograph. The cells start empty. Do not copy a recognizer prediction into the digit column.

The “Next unlabelled photograph” control on the template view walks training photographs only. Holdout photographs stay out of that pass. Select the holdout photograph from the list to label it. Those labels are ground truth for the future holdout. They are not training labels.

## Isolation

| Pool | Manifest split | Development role | Used for |
| --- | --- | --- | --- |
| Frozen evaluation | `evaluation` | empty | The existing 20 photographs only |
| Photographed holdout | `template` | `holdout` | A future photographed evaluation |
| Future training photographs | `template` | `train` | A later training run, after the holdout exists |

Every cell of one photograph gets the same split, the same role, and the same source. New photographs are not given `split=evaluation`. The CNN export and the training partition skip `holdout` cells, so the reserved pool is not training data, validation data, or a threshold set. The checkpoint file is not an input to the importer.

## Coverage now

The collection targets are 8 photographs of 4×4, 8 of 6×6, and 12 of 9×9. They are collection goals, not a model gate.

| Source | Photographs |
| --- | ---: |
| Synthetic | 25 |
| Unknown | 60 |
| Photographed | 0 |
| Unset | 0 |

Photographed photographs by grid size: 4×4 is 0, 6×6 is 0, and 9×9 is 0. Each size is short of its target, and none has a holdout photograph. Labelled photographed cells for digits 0 through 9 are all 0. No paper, print, lighting, or perspective notes are recorded, because there are no photographed photographs and no condition file.

Digits 7, 8, and 9 cannot come from a 4×4 or 6×6 puzzle. Those classes stay missing until photographed 9×9 puzzles are labelled. The report does not fill that gap from the unknown or synthetic photographs.

Optional notes live in a CSV with the columns `image,paper,print_style,lighting,perspective`. Empty cells stay “not recorded”. The coverage command is:

```bash
python -m scripts.report_photographed_coverage \
  --manifest evaluation_output/dataset/manifest.csv \
  --conditions evaluation_output/dataset/photograph_conditions.csv
```

## Readiness for a future generalisation evaluation

The holdout does not yet cover 4×4, 6×6, and 9×9, and it has no labels. `generalisation_evaluated` is false. Score a frozen model on that holdout only after the reserved photographs exist and have been labelled by hand. Do not choose the holdout from the current 20, and do not train on it.

## Tests

The new tests check that a folder named `photographed` is not provenance, that a confirmed holdout import keeps existing labels and stays out of training, that a failed batch and an evaluation duplicate roll back, and that coverage names missing digits instead of inventing paper or light. Ruff passed. Pytest passed: 318 tests.

The checkpoint hash is unchanged: `2cd75902d6d05b34a9ee8f4b1ed085430312041d785c6fbfda3323719fce2eb6`.
