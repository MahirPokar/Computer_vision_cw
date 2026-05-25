# Computer Vision Project

This repository contains a cleaned first-push version of a computer vision coursework/project workspace. The main code kept for active use is in `src/cv_pipeline`.

## What This Project Does

The project contains two main experiment tracks:

- Traditional computer vision: SIFT/ORB descriptors, Bag of Visual Words, KMeans, and SVM classifiers.
- CNN robustness experiments: MobileNetV2/ResNet-style image classifiers tested under distortions such as occlusion, noise, and blur.

The original local workspace also contained datasets, virtual environments, model checkpoints, cached descriptors, generated figures, and older duplicate exports. Those files were moved into `archive/` and are intentionally excluded from Git by `.gitignore`.

## Repository Layout

```text
src/cv_pipeline/
  new_cv.py              # BoVW SIFT/ORB training and robustness evaluation
  traditional_cv.py      # Older BoVW grid experiment
  mobilenet_robust.py    # CNN robustness experiments
  MN_rob_2.py            # CNN robustness variant
  MN_plot.py             # CNN training/evaluation with plot outputs
  make_figures.py        # Plot generation from result summaries
  makefile               # Original make targets
  CV_3/
    traditional_cv.py    # Later traditional CV experiment script
    trad_cv.ipynb        # Notebook version / exploratory work

docs/
  PROJECT_INVENTORY.md   # Cleanup inventory and uncertainty notes

archive/
  ...                    # Local-only archived artifacts, ignored by Git
```

## Setup

Create and activate a Python environment, then install the likely dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python opencv-contrib-python scikit-learn joblib tqdm matplotlib torch torchvision pillow
```

`opencv-contrib-python` is needed for SIFT support.

## Dataset

The scripts expect an image dataset arranged as class folders:

```text
dataset_root/
  train/
    class_a/
      image001.ppm
      ...
    class_b/
      ...
  test/
    class_a/
      ...
```

Some scripts still have old absolute Windows paths as defaults. Prefer passing paths explicitly.

## Run Examples

From the repository root:

```powershell
python src\cv_pipeline\new_cv.py --train_dir path\to\train --test_dir path\to\test --output_dir outputs\models
```

```powershell
python src\cv_pipeline\mobilenet_robust.py --train_dir path\to\train --test_dir path\to\test --epochs 1
```

```powershell
python src\cv_pipeline\MN_plot.py --train_dir path\to\train --test_dir path\to\test --epochs 1 --out_dir outputs\results
```

The full experiments can take a long time and may require a GPU, depending on the dataset size.

## Notes

- `archive/` is kept locally for recovery/reference but is ignored for GitHub.
- Generated outputs should go into `outputs/`, `results/`, or `models/`, which are ignored by Git.
- The cleanup was intentionally conservative: files were moved, not deleted, and core logic was not rewritten.
