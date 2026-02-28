# Architecture Details

Detailed description of the BTPN architecture with paper cross-references.

## Overview

The BTPN consists of two main components:

1. **Kinematic Foundation Model** — A multi-scale temporal pose predictor trained on kinematic data only. Serves as a frozen prior providing baseline predictions with uncertainty.

2. **Visual-Temporal BTPN** — Fuses visual features (segmentation, depth, keypoints) with kinematic predictions through clinical attention and confidence-gated corrections.

---

## Kinematic Foundation Model

*Paper Section 3.1 — Temporal Kinematic Foundation*

### Input Processing

- **Multi-scale windowing:** Input sequences at 3 scales (10, 50, 100 frames at 13 fps)
  - 10 frames (~0.77s): micro-gesture level
  - 50 frames (~3.85s): action phase level
  - 100 frames (~7.69s): full peg transfer cycle
- **Pose Input Embedding:** Linear projection from 30D kinematic features to 256D
- **Learnable Temporal Encoding:** Learned positional embeddings added to each timestep

### Hierarchical Temporal Transformer (HTT)

*Paper Section 3.2*

Three-level hierarchical attention operating at different temporal resolutions:

| Level | Window | Layers | Purpose |
|:------|:------:|:------:|:--------|
| Local | 5 frames | 2 | Fine motion details (~0.4s) |
| Medium | 20 frames | 2 | Action-level patterns (~1.5s) |
| Global | Full | 2 | Long-range dependencies |

Each level uses standard multi-head self-attention (8 heads, 256D, GELU activation).

### Memory-Enhanced Encoder (MEE)

*Paper Section 3.2*

Learnable memory bank (64 slots x 256D) that stores prototypical motion patterns. Cross-attention between encoded features and memory enables retrieval of relevant prior knowledge.

### Bimanual Cross-Attention

*Paper Section 3.2*

Cross-attention between Tool 1 and Tool 2 representations captures coordinated bimanual motion patterns (e.g., peg handoffs). Uses 4 attention heads.

### Scale Fusion

Cross-attention fuses representations from the 3 temporal scales into a unified encoding.

### Probabilistic Output Heads

Per-tool prediction heads:
- **Position head:** 3D position + 3D diagonal sigma (Gaussian)
- **Rotation head:** 4D quaternion + scalar kappa (von Mises-Fisher on S3)
- **Jaw angle head:** 1D angle + 1D sigma (Gaussian)
- **Jaw state head:** Binary open/closed classification

Total parameters: ~1.2M

---

## Visual Pipeline

*Paper Section 3.3*

### Feature Extraction (Precomputed)

| Modality | Model | Output | Description |
|:---------|:------|:------:|:------------|
| Segmentation | YOLOv26m-seg | 256D neck + 512D backbone | Instance masks + features |
| Depth | Depth Anything V2 (DINOv2-S) | 384D CLS token | Monocular depth embeddings |
| Keypoints | YOLOv26m-pose | 8 x 3D per tool | Anatomical keypoint locations |

### Visual Projections

Each modality is projected to a common 256D space:

- **SegmentationProjection:** Learnable fusion of neck (per-tool) and backbone (global) features
- **DepthProjection:** MLP projection of 384D depth embeddings
- **KeypointProjection:** Keypoint coordinate encoding + geometric features (shaft width, midline angle, jaw opening)
- **PoseProjection:** Pose encoder backbone features

### Scene Fusion

Concatenation + MLP fusion of all projected visual modalities into a unified visual representation.

---

## Visual-Temporal BTPN

*Paper Section 3.4*

### Clinical Attention Encoder

*Paper Section 3.4.1*

6-layer hierarchical windowed transformer operating at clinically motivated timescales:

| Layers | Window | Timescale | Clinical Relevance |
|:------:|:------:|:---------:|:-------------------|
| 1-2 | 8 frames | 0.6s | Micro-gesture (grasp, release) |
| 3-4 | 20 frames | 1.5s | Transfer action |
| 5-6 | Full | 7.7s+ | Task phase / global context |

### Kinematic-Visual Fusion

Cross-attention fusing kinematic representations (from frozen foundation model) with visual representations (from clinical attention encoder). 4 attention heads, 256D.

### Multi-Channel Confidence Gate

*Paper Section 3.4.2*

Separate gates for position, rotation, and jaw angle corrections with learned ceilings:

| Channel | Ceiling | Rationale |
|:--------|:-------:|:----------|
| Position | 0.5 | Visual provides absolute position reference |
| Rotation | 0.1 | Conservative — visual rotation estimates are noisy |
| Jaw angle | 0.5 | Visual provides jaw state information |

Gates are initialized with high temperature (5.0) for conservative early behavior. Anti-saturation regularization prevents gate collapse.

### Residual Pose Head

Predicts corrections (deltas) to the kinematic prior, scaled by the confidence gates:

```
final_prediction = kinematic_prior + gate * visual_correction
```

Position/quaternion/angle corrections are clamped to prevent catastrophic deviations.

### Relative Displacement Tracking

Auxiliary head predicting inter-frame displacements for temporal consistency.

### Pivot Point Estimation

EMA-based estimation of the trocar (entry point) location. Used for:
1. **Training loss:** Penalizes predictions inconsistent with trocar constraint
2. **Uncertainty inflation:** Increases sigma for predictions far from estimated trocar axis

---

## Loss Functions

### Kinematic Foundation Loss

| Component | Weight | Description |
|:----------|:------:|:------------|
| Position Beta-NLL | 1.0 | Gaussian NLL with beta weighting (beta=0.5) |
| Rotation Beta-vMF | 3.0 | von Mises-Fisher NLL on S3 with beta weighting |
| Jaw angle Beta-NLL | 1.0 | Gaussian NLL for jaw angle |
| Jaw state BCE | 0.5 | Binary cross-entropy for open/closed |
| Differentiable ECE | 0.1 | Calibration regularization (starts epoch 10) |

### Full BTPN Loss (adds to above)

| Component | Weight | Description |
|:----------|:------:|:------------|
| Gate entropy | 0.01 | Prevents gate saturation |
| Displacement | 0.5 | Inter-frame consistency |
| Pivot residual | 0.1 | Trocar constraint (starts epoch 60) |
| Smoothness | 0.01 | Temporal smoothness regularization |
| Residual reg. | 0.005 | Penalizes large corrections |

---

## Uncertainty Quantification

### Aleatoric Uncertainty
- **Position:** Diagonal Gaussian with learned sigma per axis
- **Rotation:** von Mises-Fisher on S3 with learned concentration kappa
- Calibrated via differentiable ECE penalty during training

### Epistemic Uncertainty
- MC Dropout (rate=0.1) with 30 forward passes at inference
- Provides predictive variance decomposition: aleatoric vs epistemic

### Beta-NLL Loss

Standard NLL overweights high-uncertainty predictions. Beta-NLL (Seitzer et al., 2022) reweights:

```
L_beta = sigma^(2*beta) * NLL(x, mu, sigma)
```

With beta=0.5, this balances learning accuracy and calibration.
