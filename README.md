# Reproducing UniDepth for Mobile Robot Perception

**A Robustness and Open-Vocabulary Object-Level Fusion Study**

Muhammad Mudassir Shakeel — Department of Mechatronics Engineering, University of Engineering and Technology, Lahore
Course: Mobile Robotics Systems — Group B, Perception & Semantic Mapping — Instructor: Dr. Maria Akram

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository reproduces [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) — a universal
monocular *metric* depth model that predicts a dense depth map **and** its own camera intrinsics
from a single uncalibrated RGB image — and contributes three original extensions aimed at mobile
robot deployment:

| Extension | What it does |
|---|---|
| **I — Robustness sweep** | Degrades a test photo under low light, Gaussian noise, and motion blur (9 conditions), and measures how far each prediction drifts from the clean-image baseline. |
| **II — Object-level depth fusion** | Pairs an open-vocabulary detector (Grounding DINO + BLIP captioning) with UniDepth's dense output to produce a `(object label, distance)` list per scene — the representation a planner like [VLFM](https://arxiv.org/abs/2312.03176) actually consumes. |
| **III — Cross-backbone consistency** | Runs all three official UniDepthV2 checkpoints (ViT-S/B/L) on the same image and quantifies how much their absolute depth scale disagrees. |

Full methodology, results, and discussion are in the paper: [`docs/UniDepth_Report_Final.pdf`](docs/UniDepth_Report_Final.pdf).

> **Note on ROS2 / robot platform.** This project is a **perception-only** reproduction, evaluated
> purely as a camera-in-the-loop module (see the report's Limitations section). No physical or
> simulated robot was used this semester, so there is no ROS2 workspace in this repository. The
> object-distance list produced by Extension II is formatted so it could feed a frontier- or
> obstacle-avoidance planner in future work — see "Future Work" in the report.

---

## Repository Structure

```
.
├── README.md                          # this file
├── LICENSE
├── requirements.txt                   # Python dependencies
├── docs/
│   └── UniDepth_Report_Final.pdf      # full IEEE-style report (methodology, results, discussion)
├── notebooks/
│   └── UniDepth_Extensions_Colab.py   # the exact, runnable Colab script (Jupytext "percent"-less
│                                       # cell script) used to produce every result in the report —
│                                       # run top-to-bottom in Colab: File > Upload notebook, or
│                                       # `jupytext --to notebook` locally if you prefer a .ipynb
├── src/
│   ├── setup.py
│   └── unidepth_ext/                  # reusable library code factored out of the notebook
│       ├── model.py                   # load_unidepth / run_inference / free_model
│       ├── degradations.py            # Extension I degradation functions
│       ├── metrics.py                 # self-referenced depth-map consistency metrics
│       ├── detection.py               # Extension II: Grounding DINO + BLIP pipeline
│       ├── visualization.py           # shared "Depth (m)" colorbar plotting helper
│       └── extensions/                # CLI entry points, runnable outside Colab
│           ├── ext1_robustness.py
│           ├── ext2_object_fusion.py
│           └── ext3_backbone_consistency.py
├── sample_images/                     # put your own test photos here (not included — see below)
└── results/
    ├── figures/                       # sample output figures, extracted from the report
    │   ├── fig1a_classroom_rgb_depth.png
    │   ├── fig1b_livingroom_rgb_depth.png
    │   ├── fig2_extension1_robustness_grid.png
    │   └── fig3_extension2_object_depth_detections.png
    └── tables/                        # sample output tables, matching the report's Tables I-IV
        ├── table1_reproduction_summary.csv
        ├── table2_benchmark_validation.csv
        ├── table3_ext1_robustness_results.csv
        ├── table4_ext2_object_depths.csv
        └── table_ext2_spatial_consistency.csv
```

---

## Installation

Tested on Google Colab (T4 GPU) and locally on Python 3.9+ with a CUDA GPU (CPU also works, just
slower — the notebook defaults to the small `vits14` backbone on CPU runtimes).

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/unidepth-mobile-robot-perception.git
cd unidepth-mobile-robot-perception

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install this project's dependencies
pip install -r requirements.txt

# 4. Install the official UniDepth repo (required — not on PyPI)
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth && pip install -e . && cd ..

# 5. Install this repo's own library code in editable mode
pip install -e src/
```

### Dependencies

| Package | Used for |
|---|---|
| `torch`, `torchvision` | Model inference, NMS |
| `unidepth` (official repo, installed above) | UniDepthV2 model + Hugging Face checkpoint loading |
| `transformers` | Grounding DINO + BLIP (Extension II) |
| `opencv-python-headless` | Image degradations (Extension I) |
| `scikit-image` | SSIM metric |
| `scipy` | Spearman rank correlation (Extension II) |
| `numpy`, `pandas` | Array ops, results tables |
| `matplotlib`, `Pillow` | Visualization, image I/O |

GPU is strongly recommended for Extension II (Grounding DINO + BLIP) and for running the ViT-L
backbone; everything else is usable on CPU.

---

## Usage

### Option A — Colab (recommended, matches the report exactly)

1. Open Google Colab, set **Runtime > Change runtime type > T4 GPU**.
2. Upload and run [`notebooks/UniDepth_Extensions_Colab.py`](notebooks/UniDepth_Extensions_Colab.py)
   top to bottom (each `# STEP` banner is one cell). Steps 1–4 are common setup; Extensions I, II,
   and III are independent after that — run whichever you need.
3. When prompted, upload your own test photo(s) (e.g. a classroom or living-room photo).
4. Outputs (CSVs + figures) are saved locally in the Colab runtime and optionally copied to
   Google Drive in the final cell.

### Option B — Local CLI (library code, one extension at a time)

```bash
# Extension I — robustness sweep
python -m unidepth_ext.extensions.ext1_robustness \
    --image sample_images/classroom.jpg --backbone vits14 --out results/local_run/

# Extension II — open-vocabulary object-depth fusion
python -m unidepth_ext.extensions.ext2_object_fusion \
    --image sample_images/classroom.jpg --backbone vitl14 --out results/local_run/

# Extension III — cross-backbone consistency
python -m unidepth_ext.extensions.ext3_backbone_consistency \
    --image sample_images/classroom.jpg --out results/local_run/
```

Each command prints its results table to stdout and writes the corresponding CSV to `--out`.

> `sample_images/` is intentionally empty in this repo — add your own RGB photos there (any
> resolution; no camera calibration needed, that's the whole point of UniDepth).

---

## Sample Results

Full tables and discussion are in the report; a few headline numbers:

- **Benchmark validation** (published zero-shot numbers, exact checkpoint used here):
  UniDepth-ViT-L reaches δ₁ = 98.4%, RMSE = 0.201 m on NYU-Depth-v2 (−40.4% RMSE vs. Metric3D) and
  δ₁ = 98.6%, RMSE = 1.75 m on KITTI (−22.6% vs. Metric3D). See `results/tables/table2_benchmark_validation.csv`.
- **Extension I:** graceful degradation to γ = 0.7, then a sharp scale collapse at γ = 0.5
  (Agree@5% drops from 99.6% to 0.02%) — see `results/tables/table3_ext1_robustness_results.csv`
  and `results/figures/fig2_extension1_robustness_grid.png`.
- **Extension II:** 16/17 (94%) classroom objects confidently named and fused with metric distance;
  Spearman ρ = −0.76 (p = 0.0006) between vertical frame position and depth confirms the fused
  distances are geometrically sensible — see `results/tables/table4_ext2_object_depths.csv` and
  `results/figures/fig3_extension2_object_depth_detections.png`.

| ![Classroom RGB + depth](results/figures/fig1a_classroom_rgb_depth.png) |
|:---:|
| *Fig. 1 — RGB input and predicted metric depth for an unseen classroom photograph.* |

---

## Reference

If you build on this work, please cite the original UniDepth paper:

```bibtex
@inproceedings{piccinelli2024unidepth,
  title     = {UniDepth: Universal Monocular Metric Depth Estimation},
  author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia
               and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## License

Code in this repository is released under the [MIT License](LICENSE). See `LICENSE` for the note
on the upstream UniDepth dependency's own license terms.
