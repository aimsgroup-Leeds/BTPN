# Pre-trained Checkpoints

## Model Files

| File | Size | Description | Expected Metrics |
|:-----|:----:|:------------|:-----------------|
| `kinematic_foundation.pt` | 143 MB | Multi-scale temporal pose predictor | Pos MAE 5.4 mm |
| `kinematic_foundation_norm.npz` | <1 KB | Z-score normalization statistics | — |
| `btpn_ssl.pt` | 126 MB | Visual SSL pre-trained encoder | — |
| `btpn_supervised.pt` | 131 MB | Final model (paper results) | Pos RMSE 6.8 mm, Rot 12.0° |
| `btpn_norm.npz` | <1 KB | Stage 2 normalization statistics | — |
| `yolo_segmentation.pt` | 52 MB | YOLOv26m-seg instance segmentation | 91.1% mAP50-95 |
| `yolo_keypoints.pt` | 136 MB | YOLOv26m-pose keypoint detection | 94.6% mAP50-95 |

## Training Details

- **Hardware:** NVIDIA RTX 3060 12 GB
- **Framework:** PyTorch 2.0+, Ultralytics (YOLO)
- **Total training time:** ~52 hours

## Git LFS

These files are tracked with [Git LFS](https://git-lfs.github.com/).

After cloning, run:
```bash
git lfs pull
```

If you see small pointer files instead of model weights, Git LFS is not installed or not pulling.

## Loading Checkpoints

```python
from btpn import BTPN, BTPNConfig

config = BTPNConfig.from_yaml("configs/btpn.yaml")
model = BTPN.load_pretrained("checkpoints/btpn_supervised.pt", config)
```

## State Dict Keys

Checkpoint state dicts use internal attribute names for backward compatibility:
- `btpn_v5.*` — Kinematic Foundation Model parameters (within BTPN)
- `clinical_attention.*` — Clinical Attention Encoder
- `multi_gate.*` — Multi-Channel Confidence Gate
- `residual_head.*` — Residual Pose Head

These names are preserved from development and do not affect functionality.
