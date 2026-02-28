# Data Format Specification

This document describes the data formats for Datasets A, B, and C.

## Overview

| Property | Dataset A | Dataset B | Dataset C |
|:---------|:---------:|:---------:|:---------:|
| Trials | 60 | 30 | 24 |
| Degrees of Freedom | 7 | 7 | 6 |
| Frame Rate | ~13 fps | ~13 fps | ~26 fps |
| Instruments | 2 straight graspers | 2 straight graspers | Fenestrated + Curved |
| Jaw Angle | Yes (voltage) | Yes (voltage) | No |
| Format | `label.json` | `label.json` | Numbered `.txt` files |

---

## Dataset A & B — 7-DoF Format

Each trial is a directory containing:

```
Trial1/
├── label.json           # Kinematic data (all frames)
├── Frames/              # Video frames (PNG)
├── YOLO_FEATURES/       # Precomputed segmentation features
├── DEPTH/embeddings/    # Precomputed depth embeddings
├── POSE_FEATURES/       # Precomputed pose/keypoint features
└── _visual_cache.npz    # Consolidated visual features
```

### label.json Structure

```json
{
  "Tool1": {
    "position": [[x, y, z], ...],        // mm, Aurora frame
    "quaternion": [[w, x, y, z], ...],    // unit quaternions
    "angle": [v1, v2, ...],              // raw voltage (uncalibrated)
    "timestamps": [t1, t2, ...]          // ms
  },
  "Tool2": { ... },
  "Camera": {
    "position": [[x, y, z], ...],
    "quaternion": [[w, x, y, z], ...]
  },
  "World": {
    "position": [[x, y, z], ...],
    "quaternion": [[w, x, y, z], ...]
  }
}
```

### Feature Vector (30D)

```
Index 0-2:   Tool1 position (x, y, z) in mm
Index 3-6:   Tool1 quaternion (w, x, y, z)
Index 7:     Tool1 jaw angle (calibrated radians)
Index 8-10:  Tool2 position (x, y, z) in mm
Index 11-14: Tool2 quaternion (w, x, y, z)
Index 15:    Tool2 jaw angle (calibrated radians)
Index 16-18: Camera position (x, y, z) in mm
Index 19-22: Camera quaternion (w, x, y, z)
Index 23-25: World position (x, y, z) in mm
Index 26-29: World quaternion (w, x, y, z)
```

### Jaw Angle Calibration

Raw jaw angle values are uncalibrated voltages. Calibration uses the 10-90 percentile method:
1. Compute 10th and 90th percentile of raw values per trial
2. Map [P10, P90] to [0, pi/2] (closed to fully open)
3. Values outside the range are clipped

---

## Dataset C — 6-DoF Format

Each trial is a directory with numbered files:

```
Test 1/
├── 0.txt    # Frame 0 kinematic data
├── 1.txt    # Frame 1
├── ...
└── 1793.txt # Last frame
```

### .txt File Format

Each file contains one line per sensor:

```
Time (ms)\t<timestamp>
Reference\t<x>\t<y>\t<z>\t<qw>\t<qx>\t<qy>\t<qz>
Fenestrated\t<x>\t<y>\t<z>\t<qw>\t<qx>\t<qy>\t<qz>
Curved\t<x>\t<y>\t<z>\t<qw>\t<qx>\t<qy>\t<qz>
Camera\t<x>\t<y>\t<z>\t<qw>\t<qx>\t<qy>\t<qz>
```

### Feature Vector (28D)

```
Index 0-2:   Reference position (x, y, z) in mm
Index 3-6:   Reference quaternion (w, x, y, z)
Index 7-9:   Fenestrated position (x, y, z) in mm
Index 10-13: Fenestrated quaternion (w, x, y, z)
Index 14-16: Curved position (x, y, z) in mm
Index 17-20: Curved quaternion (w, x, y, z)
Index 21-23: Camera position (x, y, z) in mm
Index 24-27: Camera quaternion (w, x, y, z)
```

> **Note:** Dataset C has no jaw angle (6-DoF only). During training, jaw channels are zero-padded and masked.

---

## Trial Discovery

### Dataset A (7DOF2024)

Trials 1-32 are nested under attempt directories:
```
7DOF2024/Attempt 1 - Day 1 with Latency/Trial1/
7DOF2024/Attempt 1 - Day 1 with Latency/Trial2/
...
7DOF2024/Attempt 4 - Day 2 Bimanual/Trial32/
```

Trials 33-60 are at the top level:
```
7DOF2024/Trial33 - camera cut out once/
7DOF2024/Trial34/
...
7DOF2024/Trial60/
```

### Dataset B (BAPES2024)

All trials at the top level:
```
BAPES2024/Trial1/
BAPES2024/Trial2/
...
BAPES2024/Trial30/
```

### Dataset C (6DOF2023)

```
6DOF2023/Test 1/
6DOF2023/Test 2/
...
6DOF2023/Test 24/
```

---

## Precomputed Visual Features

Visual features are precomputed once and cached for efficient training.

### Segmentation Features (`YOLO_FEATURES/`)

Per-frame `.npz` files with:
- `seg_neck_t1`: (256,) FPN neck features for Tool 1
- `seg_neck_t2`: (256,) FPN neck features for Tool 2
- `seg_backbone`: (512,) Global backbone features

### Depth Features (`DEPTH/embeddings/`)

Per-frame `.npz` files with:
- `embedding`: (384,) DINOv2-S CLS token from Depth Anything V2

### Pose Features (`POSE_FEATURES/`)

Per-frame `.npz` files with:
- `kp_t1`: (24,) Keypoint coordinates for Tool 1 (8 keypoints x 3)
- `kp_t2`: (24,) Keypoint coordinates for Tool 2
- `enc_t1`: (512,) Pose encoder backbone features for Tool 1
- `enc_t2`: (512,) Pose encoder backbone features for Tool 2
- `geometric`: (6,) Geometric features (shaft width, midline angle, jaw opening)

### Consolidated Cache (`_visual_cache.npz`)

All per-frame features consolidated into a single file per trial for fast loading:
- `seg_neck_t1`: (N, 256) all frames
- `seg_neck_t2`: (N, 256)
- `seg_backbone`: (N, 512)
- `depth_embedding`: (N, 384)
- `pose_kp_t1`: (N, 24)
- `pose_kp_t2`: (N, 24)
- `pose_enc_t1`: (N, 512)
- `pose_enc_t2`: (N, 512)
- `pose_geometric`: (N, 6)
- `frame_indices`: (N,) mapping to original frame numbers

---

## Normalization

All kinematic features are z-score normalized during training:
- **Position:** Per-channel mean/std computed across training set
- **Quaternion:** Per-channel mean/std (note: quaternions are renormalized to unit length after denormalization for evaluation)
- **Jaw angle:** Per-channel mean/std after calibration

Normalization statistics are saved in `norm_stats.npz` alongside checkpoints.
