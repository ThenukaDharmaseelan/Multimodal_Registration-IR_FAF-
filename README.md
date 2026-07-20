# Multimodal_Registration-(IR_FAF)

**Robust FAF ↔ IR retinal image registration.** A modular pipeline that aligns
wide-field fundus autofluorescence (FAF, 55°) images onto infrared reflectance
(IR, 30°) images using a validated **FOV-seed → LoFTR → optical-flow cascade**,
with anatomical-score candidate selection and built-in quality flagging.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Input data & CSV format](#input-data--csv-format)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [Method overview](#method-overview)
- [Metrics](#metrics)
- [Quality flagging](#quality-flagging)
- [Ablation study](#ablation-study)
- [Reproducibility](#reproducibility)
- [Project structure](#project-structure)


---

## Overview

Multimodal retinal registration is hard: FAF and IR do not share intensity
polarity, have different fields of view (55° vs 30°), and are captured at
different resolutions. This package addresses that by registering on
**vessel structure** rather than raw intensity, seeding the transform from the
known FOV ratio and refining it with feature matching and dense optical flow.

The public contract is a single function:

```python
from retina_multimodal_reg import register

result = register(
    ir_image="ir.png",           # IR (30°)  — the FIXED frame
    faf_image="faf.png",         # FAF (55°) — the MOVING image
    ir_vessel="ir_vessel.png",   # optional vessel masks
    faf_vessel="faf_vessel.png",
)

result.transform          # 2x3 affine mapping FAF (moving) -> IR (fixed)
result.registered_image   # FAF warped into the IR frame (RGB, uint8)
result.registered_vessel  # FAF vessel mask warped into the IR frame
result.metrics            # before/after quality metrics (dict)
```

Vessel masks are **optional** — when they are absent the pipeline derives a
provisional mask and additionally falls back to cross-modal image-intensity
LoFTR, so registration is still attempted.

---

## Key features

- **FOV-aware seeding** — the transform is seeded from the 55/30 ≈ 1.833 field
  ratio, with an optional narrow scale search around the seed.
- **LoFTR feature matching** on vessel distance transforms, with an
  image-intensity fallback for missing or weak masks.
- **Optical-flow cascade** — multi-resolution TV-L1 flow plus an iterative
  compositional refiner, each step gated so refinement can never lower the
  anatomical score.
- **Anatomical candidate selection** — a directed-ASD selector (falling back to
  a composite anatomical score, or cross-modal NMI when masks are empty).
- **Post-registration quality flagging** — reject noisy images, sparse /
  unsegmentable vessel masks, or low-quality registrations without ever
  altering the matching itself.
- **Rich metrics** — Dice, NCC, HD95, ASD, Wasserstein, centerline recall,
  connectivity, and image-intensity metrics (MI/NMI/SSIM/PSNR).
- **Diagnostic visualizations** — per-pair panel grids, overlays,
  checkerboards, and vessel-overlap views.
- **Leave-one-out ablation mode** that emits a paper-ready ✓/✗ table.

---

## Installation

Requires **Python 3.9+**.

### Option A — Conda (recommended)

```bash
git clone https://github.com/ThenukaDharmaseelan/Multimodal_Registration-IR_FAF-.git
cd retina_multimodal_reg

# create and activate the environment
conda create -n irfaf_env python=3.10 -y
conda activate irfaf_env

# install the package (pulls the dependencies below)
pip install - requirements.txt
```

Or create it in one step from an `environment.yml`:

```yaml
# environment.yml
name: irfaf_env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - -e .
```

```bash
conda env create -f environment.yml
conda activate irfaf_env
```

> Dependencies are installed via pip (not conda-forge) because
> `opencv-contrib-python` and `kornia` are most reliably obtained from PyPI.
> Avoid installing the conda `opencv` package alongside them — mixing the two
> OpenCV builds causes conflicts.

### Option B — venv

```bash
git clone https://github.com/ThenukaDharmaseelan/Multimodal_Registration-IR_FAF-.git
cd retina_multimodal_reg
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Dependencies

```text
numpy>=1.24
scipy>=1.10
opencv-contrib-python>=4.8    # NOT opencv-python — see note below
scikit-image>=0.21
torch>=2.0
kornia>=0.7                   # provides LoFTR
pandas>=2.0
POT>=0.9                      # optional: exact Wasserstein (sliced fallback otherwise)
```

> **Important — use `opencv-contrib-python`, not `opencv-python`.**
> The pipeline uses `cv2.ximgproc.thinning` (vessel skeletonization) and
> `cv2.optflow.DualTVL1OpticalFlow_create` (optical flow). These live in the
> **contrib** build. With plain `opencv-python`, skeletonization silently
> returns the un-thinned mask (degrading HD95/ASD/Wasserstein/centerline
> metrics) and flow falls back to Farneback. Do not install both OpenCV
> packages in the same environment — they conflict.

**GPU (optional but recommended):** LoFTR runs on CUDA automatically when a GPU
is available (`torch.cuda.is_available()`), otherwise on CPU. Install the CUDA
build of PyTorch from <https://pytorch.org> if you have a GPU.

---

## Quickstart

### Python API

```python
from retina_multimodal_reg import register

result = register(
    ir_image="data/ir.png",
    faf_image="data/faf.png",
    ir_vessel="data/ir_vessel.png",   # omit for image-based registration
    faf_vessel="data/faf_vessel.png",
)

print(result.label)                    # which internal candidate won
print(result.metrics["dice_after"])    # vessel Dice after registration

# Save the full diagnostic image set
result.save("out/", name="pair_001")
```

### Batch CLI

```bash
python -m retina_multimodal_reg.cli sample.csv out/ --flag-sparse --flag-min-anat 0.52
```

This registers every pair in `sample.csv`, writes diagnostic images and
`out/results.csv`, prints mean ± std summaries, and sets aside sparse-mask or
low-quality pairs under `out/flagged/`.

> If you installed the console script (via `pip install -e .`), the same run is:
> ```bash
> fafir-register sample.csv out/ --flag-sparse --flag-min-anat 0.52
> ```

---

## Input data & CSV format

The batch CLI reads a CSV with **one image pair per row**. By default it expects
these columns:

| Column               | Contents                    | Maps to (in registration) |
| -------------------- | --------------------------- | ------------------------- |
| `moving`             | IR (30°) image path         | fixed frame               |
| `fixed`              | FAF (55°) image path        | moving image              |
| `moving_vessel_mask` | IR vessel-mask path (opt.)  | fixed vessel              |
| `fixed_vessel_mask`  | FAF vessel-mask path (opt.) | moving vessel             |

> **Note on the column names.** The default column named `moving` holds the
> **IR** image and `fixed` holds the **FAF** image — the opposite of the
> registration roles (IR is the fixed frame, FAF is moving). This naming is a
> historical quirk of the CLI defaults. If you'd rather name your columns
> intuitively, point the CLI at them explicitly:
> ```bash
> python -m retina_multimodal_reg.cli pairs.csv out/ \
>     --ir-col ir_path --faf-col faf_path \
>     --ir-vessel-col ir_mask --faf-vessel-col faf_mask
> ```

**Vessel-mask columns are optional.** Blank/missing cells are passed as `None`,
and the pipeline automatically switches that pair to image-based matching.

Example `sample.csv`:

```csv
moving,fixed,moving_vessel_mask,fixed_vessel_mask
patientA/ir.png,patientA/faf.png,patientA/ir_vessel.png,patientA/faf_vessel.png
patientB/ir.png,patientB/faf.png,,
```

(Row 2 has no vessel masks → image-based registration.)

---

## Command-line reference

```
python -m retina_multimodal_reg.cli CSV OUTPUT [options]
```

### Core

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `CSV` | — | CSV listing image pairs (one per row). |
| `OUTPUT` | — | Output directory for images + `results.csv`. |
| `--size N` | `224` | Working resolution; **must be a multiple of 14**. |
| `--fov-scale F` | `1.833` | FOV scale seed (55/30). |
| `--no-images` | off | Only write `results.csv`; skip diagnostic images. |

### Pipeline toggles (also used for ablation)

| Flag | Effect |
| ---- | ------ |
| `--no-preprocess` | Skip CLAHE/blur preprocessing. |
| `--no-loftr` | Disable LoFTR feature matching (flow only). |
| `--scale-search [HALFWIDTH]` | Enable the narrow scale search around the seed. |

### Low-resolution retry (fallback for weak cases)

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--lowres-retry-size N` | `336` | Retry size when a 224px run looks weak. |
| `--lowres-retry-dice D` | `0.25` | Trigger retry when `dice_after` is below this. |
| `--lowres-retry-highres-size N` | `1022` | Second-stage retry size for stubborn cases. |
| `--no-lowres-retry` | off | Disable the retry fallback entirely. |

### Quality flagging (all opt-in)

| Flag | Description |
| ---- | ----------- |
| `--flag-min-anat V` | Flag pairs whose final anatomical score is below `V` (e.g. `0.52`). |
| `--flag-sparse` | Flag pairs whose vessel mask is thin/sparse **or** dots/speckle. |
| `--flag-unsegmentable` | Flag **only** dots/speckle masks; keep decent thin ones. |
| `--flag-noise-over V` | Flag noisy images whose noise estimate exceeds `V` (e.g. `16`). |
| `--good-images-only` | Skip a pair unless both masks are good, both images readable. |

### Ablation

| Flag | Description |
| ---- | ----------- |
| `--ablation` | Run the leave-one-out ablation study instead of a normal batch. |
| `--configs NAME [NAME ...]` | Subset of ablation configurations to run. |

Run `python -m retina_multimodal_reg.cli --help` for the complete list.

---

## Outputs

### Normal batch

```
out/
├── results.csv               # one row per pair, all before/after metrics
├── summary.csv               # mean ± std per metric (tidy)
├── registered/               # FAF warped into the IR frame
├── overlap_image/            # intensity overlays
├── overlap_checker/          # checkerboard views
├── overlap_vessels/          # vessel-overlap views
├── grid/                     # annotated per-pair panel grid
└── flagged/                  # pairs set aside by quality flags
```

The console also prints mean ± std of Dice / HD95 / NCC / Wasserstein / ASD
(before / after / delta).

### Ablation mode

```
out/
├── ablation_results.csv      # one row per (pair × config)
├── ablation_summary.csv      # mean ± std per (config × metric)
└── ablation_table.md         # paper-ready ✓/✗ markdown table
```

---

## Method overview

The pipeline runs entirely below the single `register()` call:

1. **Load & preprocess** — images resized to the working resolution (multiple
   of 14); optional CLAHE + mild blur. Missing vessel masks are estimated from
   the image (black-hat vesselness).
2. **FOV-scale seed** — seed the affine from the 55/30 field ratio, with an
   optional narrow scale search that only moves off the seed on a real gain.
3. **LoFTR matching** — dense correspondences on vessel *distance transforms*
   → partial affine (RANSAC). Falls back to modality-invariant image-intensity
   LoFTR when masks are missing or vessel candidates are weak.
4. **Optical-flow refinement** — multi-resolution TV-L1 flow fit to a local
   affine, then an iterative compositional refiner. Every step is gated by the
   anatomical score, so refinement can only improve (or hold) quality.
5. **Candidate selection** — all candidates are scored and the winner is chosen
   by a directed-ASD selector (falling back to the composite anatomical score,
   or cross-modal NMI when vessel masks are empty).
6. **Cascade polish** — the winner is handed to the optical-flow cascade for a
   final refinement pass.

All stages can be switched off individually (`use_loftr`, `multires_flow`,
`comp_flow`, `cascade`, `asd_selection`, `scale_search`, `preprocess`) for the
ablation study.

---

## Metrics

For every pair the pipeline reports **before** and **after** registration:

**Vessel-mask (structure):**
- **Dice** — vessel overlap (higher is better)
- **NCC** — normalized cross-correlation
- **HD95** — 95th-percentile Hausdorff surface distance (lower is better)
- **ASD** — average symmetric surface distance (lower is better)
- **Wasserstein** — optimal-transport distance (exact via POT, sliced fallback)
- **Centerline recall** — skeleton coverage at multiple tolerances
- **Connectivity** — connected-component and Euler-number agreement

**Image intensity (inside the common FOV):**
- **MI / NMI** — mutual information (robust to cross-modal intensity)
- **SSIM**, **PSNR**, **NCC**, **SSD**

---

## Quality flagging

Flagging is **post-registration only**: every pair is registered fully, then
optionally set aside. Matching is never altered by a flag — it only decides
whether a result is kept or moved to `flagged/`. Three independent gates:

- **Noisy image** (`--flag-noise-over`) — high Immerkær noise estimate. A noisy
  image whose masks are still well-segmented is kept (registration is
  mask-driven).
- **Sparse / unsegmentable mask** (`--flag-sparse`, `--flag-unsegmentable`) —
  distinguishes a genuinely thin-but-real vessel tree ("sparse") from
  dots/speckle with no coherent tree ("unsegmentable").
- **Low registration quality** (`--flag-min-anat`) — final anatomical score
  below a threshold.

Flagged pairs are recorded with `status=flagged` in `results.csv` and excluded
from the summary statistics.

---

## Ablation study

Run a leave-one-out ablation over the pipeline components:

```bash
# full study over every configuration
python -m retina_multimodal_reg.cli pairs.csv out/ --ablation

# limit to specific configurations
python -m retina_multimodal_reg.cli pairs.csv out/ --ablation --configs full no_loftr
```

Configurations: `full`, `no_loftr`, `no_optical_flow`, `no_cascade`,
`no_asd_selection`, `no_scale_search`, `no_preprocess`, `seed_only`.

Produces `ablation_table.md` — a ready-to-paste ✓/✗ table with mean ± std per
metric and the best configuration bolded per column.

---

## Reproducibility

Every run calls `_seed_everything(0)`, which seeds NumPy, Python `random`,
OpenCV RANSAC, and Torch/cuDNN (deterministic convolutions) so the winning
transform is stable run-to-run. The sliced-Wasserstein fallback also uses its
own seeded RNG.

For fully deterministic results, keep `--size` fixed (the low-res retry ladder
changes the working resolution, which can change resolution-dependent scores).

---

## Project structure

```
retina_multimodal_reg/
├── __init__.py         # public API: register, RegistrationResult
├── cli.py              # batch + ablation command-line entry point
├── pipeline.py         # the single register() entry point
├── registration.py     # candidate generation + selection
├── matching.py         # LoFTR -> affine (vessel DT and image intensity)
├── flow.py             # optical-flow refinement + cascade
├── metrics.py          # Dice/NCC/HD95/ASD/Wasserstein + scoring
├── preprocessing.py    # CLAHE, vessel closing, mask quality heuristics
├── models/loftr.py     # LoFTR matcher state + loader
├── io.py               # image / vessel-mask loading
├── utils.py            # configuration, geometry, affine sanity
└── visualization.py    # diagnostic images + grids
```



