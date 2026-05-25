# Project Inventory

This inventory records the initial cleanup performed before the first GitHub push. The goal was quick triage, not a perfect refactor.

## Likely Main / Current Version

Kept as active source:

- `src/cv_pipeline/new_cv.py`
- `src/cv_pipeline/mobilenet_robust.py`
- `src/cv_pipeline/MN_rob_2.py`
- `src/cv_pipeline/MN_plot.py`
- `src/cv_pipeline/traditional_cv.py`
- `src/cv_pipeline/make_figures.py`
- `src/cv_pipeline/makefile`
- `src/cv_pipeline/CV_3/traditional_cv.py`
- `src/cv_pipeline/CV_3/trad_cv.ipynb`

Reasoning: these files came from `COMPVISION/.venv/CV_pipeline`, which contained the newest experiment scripts and appeared to match the final generated model/result artifacts in the local workspace.

## Kept

- `README.md`: setup and run notes.
- `.gitignore`: excludes local environments, datasets, model checkpoints, generated outputs, and archived material.
- `docs/PROJECT_INVENTORY.md`: this inventory.
- `src/cv_pipeline/`: current runnable/project code.

## Archived

Moved to `archive/legacy_pipeline_export/`:

- `CV_pipeline-20250501T105732Z-001/`

This appears to be an older duplicate/exported pipeline with traditional CV scripts, cached descriptors, model artifacts, results, and figures.

Moved to `archive/local_environment_and_artifacts/`:

- `COMPVISION/`

This contains the original local workspace leftovers after extracting the current pipeline source. Notable contents include:

- `.venv/` local Python environment.
- `datatset/iCubWorld/` image dataset.
- `A_CV/` older/uncertain code.
- `models/` trained checkpoints and serialized models.
- `results/`, `debug_outputs/`, `occlusion_debug/`, and top-level generated figures.
- `studies/` exploratory scripts and pickled study files.

Moved to `archive/local_environment_and_artifacts/current_pipeline_outputs/`:

- Generated outputs that were next to the extracted current pipeline, including caches, models, results, and figures.

Moved to `archive/reports/`:

- `Comp_vision_3.pdf`

## Uncertain

- `COMPVISION/A_CV/local_features.py` looked like a more developed standalone baseline, but it imports helpers that were not present in the active tree (`icub_data`). It was archived rather than treated as current.
- `CV_3/trad_cv.ipynb` may be exploratory, but it was kept beside `CV_3/traditional_cv.py` because notebooks often explain experiment context.
- Several scripts still contain old absolute paths. The README recommends passing dataset/output paths explicitly rather than editing core logic during this triage.

## GitHub Push Notes

The `archive/` directory is intentionally ignored by Git. It remains available locally, but the initial GitHub push should contain the cleaned source, README, gitignore, and inventory without large local data, virtual environment files, checkpoints, or generated outputs.
