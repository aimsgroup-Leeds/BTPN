<div align="center">

# Bayesian Temporal Pose Networks

### Uncertainty-Calibrated Laparoscopic Tool Pose Tracking

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue)](#citation)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

<!-- **[Paper](link) | [Poster](link) | [Video](link)** -->

*A probabilistic framework for 7-DoF vision-only laparoscopic tool pose tracking with calibrated Bayesian uncertainty, validated on 114 peg transfer trials across three electromagnetic tracking datasets.*

</div>

---

<p align="center">
  <img src="figures/architecture.png" alt="BTPN Architecture" width="95%"/>
</p>

**Architecture overview.** **(a)** A two-stage CNN detection pipeline: YOLOv26m-seg localises and segments both tools, then ROI crops are fed to YOLOv26m-pose for per-tool 8-keypoint detection (4 shaft, joint, tip, 2 jaw). **(b)** The Kinematic Foundation Model uses a Hierarchical Temporal Transformer (HTT) with cross-scale attention at clinically motivated resolutions — local for micro-gestures (grasp onset, release timing), medium for purposeful actions (reach, transfer, place) and global for full-sequence context. A Memory Enhanced Encoder (MEE) splits per-tool projections with bimanual cross-attention gates for inter-tool coordination. **(c)** The full BTPN fuses segmentation, depth (DepthAnything V2) and keypoint features with the frozen kinematic prior via multi-head cross-attention. Per-channel confidence gates (position &le; 0.5, rotation &le; 0.3, jaw &le; 0.5) prevent noisy visual estimates from corrupting modalities where kinematics are already accurate. Probabilistic pose heads predict residual corrections with calibrated Bayesian uncertainty.

## Abstract

Accurate pose tracking of laparoscopic instruments from monocular endoscopic video in surgical training tasks is essential for computer-assisted surgery and objective skill assessment. However, current methods require geometric priors unavailable in non-robotic settings and lack temporal reasoning across multimodal cues and uncertainty quantification. We introduce **Bayesian Temporal Pose Network (BTPN)**, a framework that fuses visual and kinematic features through hierarchical multi-scale temporal attention with calibrated Bayesian uncertainty. A fine-tuned segmentation backbone achieves **91.1% mAP<sub>50-95</sub>** and keypoint detection reaches **94.6% mAP<sub>50-95</sub>**. End-to-end visual pose tracking attains **6.8 mm** position and **12.0°** rotation RMSE with **0.016** uncertainty error. The framework is validated on three electromagnetic tracking datasets with 114 peg transfer trials, demonstrating that uncertainty-aware, vision-only tracking can support interpretable surgical skill assessment.

## Highlights

| | Metric | Value |
|---|---|---|
| :dart: | **Position RMSE** | 6.8 mm on held-out surgical trials |
| :triangular_ruler: | **Rotation RMSE** | 12.0° with calibrated uncertainty |
| :microscope: | **Segmentation** | 91.1% mAP<sub>50-95</sub> |
| :straight_ruler: | **Keypoints** | 94.6% mAP<sub>50-95</sub> |
| :bar_chart: | **Calibration (ECE)** | 0.013 — well-calibrated uncertainty |
| :hospital: | **Validation** | 114 peg transfer trials across 3 datasets |

---

## Key Results

### (a) Visual Pipeline Components

| Component | Precision | Recall | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:----------|:---------:|:------:|:-----------------:|:--------------------:|
| Detection | 98.1 | 98.3 | 99.1 | 97.5 |
| Segmentation | 98.1 | 98.3 | 99.1 | 91.1 |
| Keypoints | 97.3 | 97.3 | 98.3 | 94.6 |

### (b) Pose Prediction — Dataset A Held-Out Trials

| Method | Pos *x* | Pos *y* | Pos *z* | Pos ‖v‖ | Roll | Pitch | Yaw | Geo | Jaw (°) | ECE |
|:-------|:-------:|:-------:|:-------:|:--------:|:----:|:-----:|:---:|:---:|:-------:|:---:|
| ART-Net | 19.7 | 20.7 | 14.9 | 32.2 | 65.6 | 30.8 | 67.5 | 55.3 | 5.73 | 0.154 |
| Visual regr. | 18.3 | 17.0 | 13.1 | 28.2 | 53.0 | 27.2 | 54.8 | 44.1 | 6.30 | 0.137 |
| Visual + LSTM | 17.2 | 13.3 | 12.4 | 25.0 | 76.9 | 36.8 | 68.1 | 67.9 | 5.73 | 0.099 |
| Visual + TCN | 15.1 | 11.9 | 11.0 | 22.2 | 72.1 | 37.3 | 68.7 | 66.6 | 5.73 | 0.240 |
| Visual + VTT | 16.4 | 13.7 | 12.5 | 24.8 | 74.0 | 45.7 | 79.8 | 69.1 | 5.73 | 0.115 |
| Kinematic regr. | 5.2 | 5.8 | 3.9 | 8.7 | 32.8 | 17.3 | 30.9 | 27.7 | 1.72 | 0.092 |
| **Full BTPN** | **4.1** | **4.4** | **3.5** | **6.8** | **14.7** | **7.3** | **15.8** | **11.9** | **1.72** | **0.013** |

> Position errors in mm, rotation errors in degrees. Geo = geodesic distance on SO(3).

### (c) Cross-Dataset Generalisation

| Dataset | Role | Pos *x* | Pos *y* | Pos *z* | Pos ‖v‖ | &Delta;Rot (°/step) | Jaw (°) |
|:--------|:----:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:-------:|
| A (21 trials) | Held-out | 4.1 | 4.4 | 3.5 | 6.8 | 14.0 | 1.7 |
| B (30 trials) | In-dist. | 6.2 | 5.5 | 4.9 | 9.6 | 20.4 | 1.7 |
| C (24 trials) | OOD | 5.4 | 7.3 | 7.2 | 11.6 | 19.6 | N/A |

### Qualitative Results

<p align="center">
  <img src="figures/output_comparison.png" alt="7-DoF Trajectory Predictions" width="90%"/>
</p>

**7-DoF predictions for a held-out trial sequence.** Ground truth (black solid) vs BTPN predictions (blue dashed) with &plusmn;2&sigma; uncertainty bands (shaded). Left column shows position (X, Y, Z in mm); right column shows Euler rotation components and jaw angle in degrees. Per-axis RMSE annotations range from 1.9–2.5 mm for position and 4.8–17.2° for rotation. The model accurately tracks rapid tool motions with well-calibrated uncertainty that widens during challenging periods.

<p align="center">
  <img src="figures/uncertainty_quality.png" alt="Uncertainty Quality Assessment" width="90%"/>
</p>

**Uncertainty quality assessment.** **(a)** Reliability diagram: position (ECE = 0.017) and jaw angle (ECE = 0.061) are well-calibrated, while rotation is over-conservative (ECE = 0.179), reflecting the inherent difficulty of recovering orientation from monocular images. **(b)** Mean position error binned by predicted &sigma; (*r* = 0.62): a clear monotonic trend confirms higher predicted uncertainty corresponds to higher actual error. **(c)** Sparsification curve: discarding the most uncertain 50% of predictions reduces mean error from 5.9 mm to ~3.5 mm (AUSE = 1.08 mm), close to the oracle ordering by true error. **(d)** Position error stratified by detection confidence — error is 5.9 mm at high confidence (*n* = 8,120) and degrades to 23.4 mm only for rare low-confidence frames (*n* = 48).

### Datasets

<table>
<tr>
<td align="center" width="33%">
<img src="figures/dataset_a.png" alt="Dataset A" width="100%"/><br/>
<b>Dataset A</b><br/>60 trials &middot; 7-DoF &middot; 13 fps
</td>
<td align="center" width="33%">
<img src="figures/dataset_b.png" alt="Dataset B" width="100%"/><br/>
<b>Dataset B</b><br/>30 trials &middot; 7-DoF &middot; 13 fps
</td>
<td align="center" width="33%">
<img src="figures/dataset_c.png" alt="Dataset C" width="100%"/><br/>
<b>Dataset C</b><br/>24 trials &middot; 6-DoF &middot; 26 fps
</td>
</tr>
</table>

---

## Installation

```bash
git clone https://github.com/omariosc/BTPN.git
cd BTPN
pip install -e .
```

> **Requirements:** Python 3.10+, PyTorch 2.0+, CUDA-capable GPU recommended.

## Quick Start

### Inference with Pre-trained Model

```python
from btpn import BTPN, BTPNConfig

config = BTPNConfig.from_yaml("configs/btpn.yaml")
model = BTPN.load_pretrained("checkpoints/btpn_supervised.pt", config)
model.eval()

# Run inference on a trial
predictions = model.predict(trial_data)
print(f"Position: {predictions['position'].shape}")      # (T, 2, 3)
print(f"Quaternion: {predictions['quaternion'].shape}")    # (T, 2, 4)
print(f"Sigma (pos): {predictions['sigma_pos'].shape}")   # (T, 2, 3)
```

### Training from Scratch

```bash
# Stage 1: Train Kinematic Foundation Model
python scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml

# Stage 2: Visual SSL Pre-training
python scripts/train.py --stage ssl --config configs/btpn.yaml

# Stage 3: Supervised Fine-tuning
python scripts/train.py --stage supervised --config configs/btpn.yaml

# (Optional) Train detection models
python scripts/train.py --stage detection --task segmentation --config configs/detection.yaml
python scripts/train.py --stage detection --task keypoints --config configs/detection.yaml
```

### Evaluation

```bash
# Evaluate on Dataset A (held-out trials)
python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset A

# Evaluate on all datasets and generate LaTeX tables
python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset all --output-tables
```

---

## Pre-trained Checkpoints

| Model | File | Size | Description |
|:------|:-----|:----:|:------------|
| Kinematic Foundation | `kinematic_foundation.pt` | 143 MB | Multi-scale temporal pose predictor (frozen prior) |
| Kinematic Norm Stats | `kinematic_foundation_norm.npz` | <1 KB | Z-score normalization statistics |
| BTPN SSL | `btpn_ssl.pt` | 126 MB | Stage 1: Visual SSL pre-trained encoder |
| **BTPN Supervised** | **`btpn_supervised.pt`** | **131 MB** | **Stage 2: Final model (paper results)** |
| BTPN Norm Stats | `btpn_norm.npz` | <1 KB | Stage 2 normalization statistics |
| YOLO Segmentation | `yolo_segmentation.pt` | 52 MB | YOLOv26m-seg (91.1% mAP<sub>50-95</sub>) |
| YOLO Keypoints | `yolo_keypoints.pt` | 136 MB | YOLOv26m-pose (94.6% mAP<sub>50-95</sub>) |

> Checkpoints are tracked with [Git LFS](https://git-lfs.github.com/). Run `git lfs pull` after cloning.

---

## Project Structure

```
BTPN/
├── btpn/                          # Python package
│   ├── config.py                  # BTPNConfig dataclass (YAML-loaded)
│   ├── model.py                   # KinematicFoundationModel + BTPN
│   ├── components.py              # HTT, MEE, attention, embeddings
│   ├── visual.py                  # Visual projections, fusion, gates
│   ├── losses.py                  # All loss functions
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── detection.py               # YOLO training and feature extraction
│   ├── metrics.py                 # Evaluation metrics (RMSE, ECE, AUSE)
│   └── utils.py                   # Schedulers, checkpointing, plotting
│
├── scripts/
│   ├── train.py                   # Unified training: --stage {foundation,ssl,supervised,detection}
│   ├── evaluate.py                # Evaluation: --dataset {A,B,C,all}
│   ├── inference.py               # Single-trial inference demo
│   ├── generate_figures.py        # Reproduce paper figures
│   └── annotate.py                # Create COCO annotations from video
│
├── configs/                       # YAML configuration files
│   ├── paths.yaml                 # Data paths (edit for your setup)
│   ├── kinematic_foundation.yaml  # Foundation model hyperparameters
│   ├── btpn.yaml                  # Full BTPN hyperparameters
│   └── detection.yaml             # YOLO training configs
│
├── checkpoints/                   # Pre-trained weights (Git LFS)
├── figures/                       # Paper figures
├── results/                       # LaTeX tables and evaluation data
├── data/                          # Sample data for testing
│   ├── sample_a/                  # 1 trial from Dataset A (7-DoF)
│   └── sample_c/                  # 1 trial from Dataset C (6-DoF)
└── docs/                          # Extended documentation
    ├── TRAINING.md                # Step-by-step training guide
    ├── DATA_FORMAT.md             # Dataset format specification
    └── ARCHITECTURE.md            # Architecture details
```

## Data Format

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for full specification. Brief overview:

**7-DoF (Datasets A, B):** Each trial is a directory with `label.json` containing Tool1, Tool2, Camera, and World kinematic streams. Each stream has position (3D, mm), quaternion (wxyz), and jaw angle (voltage).

**6-DoF (Dataset C):** Each trial is a directory with numbered `.txt` files (one per frame) containing Reference, Fenestrated, Curved, and Camera sensor readings.

---

## Training Pipeline

The full training pipeline takes approximately **52 hours** on a single NVIDIA RTX 3060:

| Stage | Epochs (total / best) | Wall Time | Patience |
|:------|:---------------------:|:---------:|:--------:|
| YOLO segmentation fine-tune | 124 / 104 | 4.0 h | 20 |
| YOLO keypoint fine-tune | 169 / 144 | 3.6 h | 25 |
| Kinematic Foundation Model | 80 / 50 | 23.6 h | 30 |
| Visual SSL pre-training | 39 / 14 | 9.1 h | 25 |
| Visual supervised fine-tuning | 50 / 10 | 11.9 h | 20 |

See [`docs/TRAINING.md`](docs/TRAINING.md) for detailed instructions.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{choudhry2026btpn,
  title     = {Bayesian Temporal Pose Networks for Uncertainty-Calibrated
               Laparoscopic Tool Pose Tracking},
  author    = {Choudhry, Omar and Ali, Sharib and Biyani, Chandra Shekhar
               and Jones, Dominic},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               (MICCAI)},
  year      = {2026}
}
```

## License

This project is licensed under [CC BY-NC 4.0](LICENSE). You may use and adapt this work for non-commercial purposes with attribution.
