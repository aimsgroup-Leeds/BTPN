# Training Guide

Step-by-step instructions for training the full BTPN pipeline from scratch.

## Prerequisites

1. **Hardware:** NVIDIA GPU with 8+ GB VRAM (tested on RTX 3060 12 GB)
2. **Data:** At least Dataset A (7DOF2024) with video frames and kinematic data
3. **Install:** `pip install -e .` from the repo root

## Pipeline Overview

```
Detection Models ──> Feature Precomputation ──> Kinematic Foundation ──> Visual SSL ──> Supervised
     (4h each)            (1h/dataset)              (24h)                (9h)           (12h)
```

Total: ~52 hours on RTX 3060. Stages are independent except for dependencies shown.

---

## Stage 0: Detection Models (Optional)

Skip this if using the provided `checkpoints/yolo_segmentation.pt` and `checkpoints/yolo_keypoints.pt`.

### 0a. Create Annotations

```bash
python scripts/annotate.py --data-root /path/to/7DOF2024 --output annotations/
```

This generates COCO-format annotations with instance segmentation masks and 8-keypoint skeletons.

### 0b. Train Segmentation

```bash
python scripts/train.py --stage detection --task segmentation --config configs/detection.yaml
```

### 0c. Train Keypoints

```bash
python scripts/train.py --stage detection --task keypoints --config configs/detection.yaml
```

### 0d. Precompute Visual Features

After training detection models, precompute features for all trials:

```bash
python scripts/train.py --stage detection --task precompute --data-root /path/to/7DOF2024
```

This generates `YOLO_FEATURES/`, `POSE_FEATURES/`, `DEPTH/embeddings/`, and `_visual_cache.npz` per trial.

---

## Stage 1: Kinematic Foundation Model

Trains a multi-scale temporal pose predictor on kinematic data only. This becomes the frozen prior for the full BTPN.

```bash
python scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml
```

**Key settings:**
- 3 window scales: 10, 50, 100 frames
- Beta-NLL loss with diagonal covariance
- Cosine annealing with warmup
- Early stopping (patience=30)

**Expected output:**
- `outputs/foundation/checkpoints/best_model.pt`
- `outputs/foundation/checkpoints/norm_stats.npz`
- Position MAE ~5.4 mm, Rotation ~87° geodesic (kinematic-only baseline)

**Typical convergence:** ~80 epochs (best at ~50), ~24h on RTX 3060.

---

## Stage 2: Visual SSL Pre-training

Pre-trains the visual encoder with four self-supervised tasks:

| Task | Weight | Description |
|:-----|:------:|:------------|
| MVR (Masked Visual Reconstruction) | 1.0 | Reconstruct masked visual features |
| VCL (Visual Contrastive Learning) | 0.5 | Temporal contrastive learning on visual sequences |
| VTOP (Visual Temporal Order Prediction) | 0.2 | Predict temporal ordering of visual frames |
| VKA (Visual-Kinematic Alignment) | 0.3 | Align visual and kinematic representations |

```bash
python scripts/train.py --stage ssl --config configs/btpn.yaml
```

**Prerequisites:**
- Kinematic Foundation checkpoint at path specified in `configs/paths.yaml`
- Precomputed visual features (`_visual_cache.npz`) for all training trials

**Expected output:**
- `outputs/ssl/checkpoints/best_model.pt`

**Typical convergence:** ~39 epochs (best at ~14), ~9h on RTX 3060.

---

## Stage 3: Supervised Fine-tuning

Fine-tunes the full BTPN with three phases:

| Phase | Epochs | What trains | LR |
|:------|:------:|:------------|:---|
| Warmup | 0–20 | Fusion + gates only | Base |
| Full | 20–150 | All visual parameters | Base + cosine decay |
| Fine-tune | 150+ | All parameters | 0.1x base |

```bash
python scripts/train.py --stage supervised --config configs/btpn.yaml
```

**Prerequisites:**
- Visual SSL checkpoint from Stage 2
- Kinematic Foundation checkpoint

**Expected output:**
- `outputs/supervised/checkpoints/best_model.pt`
- `outputs/supervised/checkpoints/norm_stats.npz`

**Typical convergence:** ~50 epochs (best at ~10), ~12h on RTX 3060.

---

## Evaluation

```bash
# Single dataset
python scripts/evaluate.py --checkpoint outputs/supervised/checkpoints/best_model.pt --dataset A

# All datasets with LaTeX tables
python scripts/evaluate.py --checkpoint outputs/supervised/checkpoints/best_model.pt --dataset all --output-tables

# Using provided checkpoints
python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset all
```

---

## Configuration

Edit `configs/paths.yaml` to point to your data:

```yaml
data_root: "/path/to/your/data"
output_root: "outputs"
dataset_a: "7DOF2024"
dataset_b: "BAPES2024"
dataset_c: "6DOF2023"
```

All hyperparameters are in `configs/btpn.yaml` and `configs/kinematic_foundation.yaml`. Key parameters to tune:

| Parameter | Default | Notes |
|:----------|:-------:|:------|
| `batch_size` | 16/32 | Reduce if GPU memory limited |
| `stage2_lr` | 3e-4 | Main learning rate for supervised |
| `max_gate_rotation` | 0.1 | Higher = more visual influence on rotation |
| `pivot_warmup_epochs` | 60 | When trocar constraint activates |
| `early_stopping_patience` | 30 | Increase for longer training |

---

## Troubleshooting

**CUDA out of memory:** Reduce `batch_size` in the config YAML.

**Rotation error ~90°:** Ensure `set_quat_norm_stats()` is called on the loss function before evaluation. Without it, quaternion targets remain z-scored.

**Visual features not found:** Run the precompute step (Stage 0d) to generate `_visual_cache.npz` for each trial.

**Windows multiprocessing crash:** Set `num_workers: 0` in config (default).
