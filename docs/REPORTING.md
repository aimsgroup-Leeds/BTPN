# Reporting of Experimental Results

This document addresses the [MICCAI 2026 experimental-reporting guidelines](https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html) for *Bayesian Temporal Pose Networks for Uncertainty-Calibrated Laparoscopic Tool Pose Tracking* (BTPN). All headline numbers are the honest values reproduced from the released checkpoint and are recomputable on CPU via `python scripts/evaluate.py --from-npz results/evaluation_data.npz`. Where a guideline item is only partially met (e.g. multi-seed error bars), this is stated explicitly rather than glossed over.

## 1. Hyperparameters — range, selection, and values
**Selection.** The best checkpoint per configuration is chosen by lowest validation loss on the Dataset-A held-out split (early stopping, patience 15–30 epochs). High-level architectural choices (kinematic prior, bimanual cross-attention, multi-scale prior, calibration loss) were validated by the component ablations in `results/table2b.tex` rather than a continuous grid search.

**Values used.**
- Optimiser: AdamW (β₁=0.9, β₂=0.999, weight decay 1e-5); gradient clipping 1.0.
- Schedule: cosine LR 3e-4 → 1e-7; batch size 16; ≈47K training windows (dense sliding window, stride 2).
- Kinematic foundation model: multi-scale temporal windows **[10, 50, 100]** frames; single-phase supervised next-frame regression.
- Visual encoder: two stages — stage 1 self-supervised (masked **visual** reconstruction, 30% of frames, + visual–kinematic alignment); stage 2 supervised fine-tuning (visual projections first, then all parameters).
- Loss: β-NLL (β=0.5) for position/jaw, β-vMF for orientation, differentiable ECE calibration; auxiliary weights λ_reconstruction=0.1, λ_pivot=0.1.
- Pose-head confidence-gate ceilings: position ≤ 0.5, jaw ≤ 0.5, rotation ≤ 0.1.
- Detection/keypoints: YOLOv26m-seg / YOLOv26m-pose. Depth: DepthAnything V2 (Small), frozen.

A continuous hyperparameter *sweep* (e.g. over β, gate ceilings, or window scales) was not performed; this is noted as a limitation.

## 2. Sensitivity to parameter changes
Component sensitivity is quantified by the four ablations in `results/table2b.tex`:
- Removing the **kinematic prior** is by far the most damaging (‖v‖ 7.0 → 25.2 mm).
- Removing the **calibration** loss collapses calibration (ECE 0.028 → 0.259).
- Removing **bimanual** cross-attention costs ≈2.1 mm (‖v‖ 7.0 → 9.1).
- Removing the **multi-scale** prior (single-scale [10]) gives a small position change (7.0 → 7.3 mm) with rotation/jaw essentially unchanged.

A finer continuous sensitivity analysis was not performed.

## 3. Number of training and evaluation runs
- **Training:** one run per configuration (single seed). Multi-seed training is **not** performed — a stated limitation (see §7–§8).
- **Evaluation:** the released model and each ablation are evaluated on the Dataset-A held-out split at dense sampling interval 2 (**n = 21,048 frames / 42,096 tool-instances**), and across Datasets B and C. Run-to-run variation from cuDNN non-determinism was measured at ≈ **1.5%** on the position metric (e.g. 6.91 vs 7.02 mm across passes).

## 4. Baselines — implementation and tuning
No published full-7-DoF, non-robotic, monocular pose comparator exists for this setting. We therefore (a) re-implement **ART-Net** (Hasan et al.) as the closest geometric-prior method, and (b) implement four in-house temporal baselines — **Visual regression, Visual+LSTM, Visual+TCN, Visual+VTT** — and a **Kinematic regression** baseline. All baselines share BTPN's identical data splits, optimiser/schedule/early-stopping recipe, and evaluation protocol, and are tuned on the same Dataset-A validation split.

## 5. Train / validation / test splits
- **Dataset A** (60 trials): **39 training / 21 held-out**. All models — including the kinematic foundation model — are trained **only** on Dataset A.
- **Dataset B** (30 trials): in-distribution external validation; never used in training.
- **Dataset C** (24 trials): out-of-distribution test (different box trainer, camera, tools, insertion angles; 6-DoF, no jaw sensor); never used in training.

Datasets B and C use per-trial normalisation to absorb inter-trial electromagnetic-reference offsets.

## 6. Evaluation metrics (definitions)
- **Position ‖v‖ (mm):** Euclidean (L2) RMSE over (x, y, z) tool-tip position in the camera frame.
- **Rotation, Geo (°):** geodesic distance on SO(3) between predicted and ground-truth orientation.
- **Jaw (% opening):** RMSE of jaw aperture as a percentage of the per-trial 10th/90th-percentile opening range (the jaw signal is a raw sensor voltage with no voltage-to-angle calibration; only the relative improvement is unit-invariant).
- **ECE:** Expected Calibration Error — position in L2-norm Euclidean space; rotation in physical (Fisher) space.
- **ΔRot (°/step):** RMSE of the frame-invariant per-step angular velocity, comparable across datasets with different EM coordinate frames.
- **AR@5 / AR@10:** autoregressive rollout position error at 5 / 10 future frames.

## 7. Central tendency and variation
Errors are reported as the mean (root-mean-square) over all evaluation frames. **Variation across training seeds (error bars) is not reported** — a single training run per configuration was used. We do report the measured cuDNN run-to-run band (≈ 1.5% on position). Multi-seed error bars are a clear limitation and recommended future work.

## 8. Statistical significance
No formal seed-level significance test was performed; differences are reported descriptively. Where a small effect is interpreted (the multi-scale prior's ≈0.3 mm position gain), it was checked against a same-model GPU-noise placebo and found to exceed the noise floor (≈ 2.2×) across 18/20 held-out trials — but this is **not** a seed-level significance test. (An earlier internal "p < 0.001" claim was found unsubstantiated and removed.)

## 9. Runtime / energy
**Measured** per-frame inference latency on an **NVIDIA RTX 3060** (batch 1; 15 warm-up + 60 timed iterations; `torch.cuda.synchronize()` around each stage; PyTorch 2.7.1 + CUDA 11.8):

| Stage | Latency |
|---|---|
| YOLO segmentation | ≈ 20 ms |
| YOLO keypoints / pose | ≈ 26 ms |
| DepthAnything V2 (Small) | ≈ 107 ms (37 ms CPU preprocessing + 70 ms GPU forward) |
| BTPN temporal model (full `VisualTemporalBTPNv3` forward) | ≈ 117 ms |
| **End-to-end total** | **≈ 270 ms / frame (≈ 3.7 FPS)** |

The BTPN and depth stages are co-dominant (≈ 43% / ≈ 40%); the two YOLO stages together are ≈ 17%. Inference is a **single deterministic forward pass** (no test-time MC sampling). At batch 1 the BTPN forward is host-dispatch / kernel-launch bound (≈ 16 ms of actual GPU compute spread over ~4,000 small kernels), so this latency is substantially reducible via CUDA graphs / TorchScript / batching. Energy cost was not measured.

## 10. Memory footprint
≈ **0.1 GB** GPU memory at inference; ≈ **19.8 M** parameters (the majority frozen: DepthAnything V2 encoder + the frozen kinematic prior). *(Reported values; confirm on your hardware.)*

## 11. Failure-mode analysis
- **Rotation** is the hardest channel and its uncertainty is **over-conservative / under-confident**: rotation ECE ≈ 0.30 vs position 0.028 / jaw 0.079.
- **Low detection-confidence frames** degrade sharply: position ≈ 5.3 mm at high detection confidence (n = 20,258) vs ≈ 22.2 mm at low confidence (n = 121).
- **Out-of-distribution (Dataset C):** position ‖v‖ rises to 11.0 mm (vs 7.0 mm in-distribution).
- **Jaw** has very low signal variance (near "predict-the-mean"); only the relative 3.7× improvement over baselines is meaningful.
- **MC-Dropout** epistemic uncertainty does **not** improve calibration (it worsens position/jaw ECE), so single-pass deterministic inference is used.

## 12. Computing infrastructure
NVIDIA RTX 3060 (12 GB) GPU; PyTorch 2.7, CUDA 11.8, Python 3.13.

## 13. Clinical significance
BTPN performs vision-only 7-DoF tool tracking on non-robotic laparoscopic peg-transfer (a Fundamentals-of-Laparoscopic-Surgery training task) **without** CAD/geometric priors or electromagnetic tracking at inference. Calibrated, vision-only pose enables objective, interpretable surgical-skill assessment (motion economy, bimanual coordination) in settings without instrumented tools — relevant to low-resource and surgical-training environments. Sub-centimetre position tracking is sufficient to capture peg-transfer tool trajectories for downstream skill analysis.

---

## Exploratory: inference-efficiency / modality tradeoff (repo-only, not in the paper)

Two lighter visual variants were trained (single seed, reusing the frozen kinematic prior + the same ablation recipe) to probe the accuracy ↔ latency tradeoff on an RTX 3060:

| Variant | Pos ‖v‖ (mm) | Geo (°) | Jaw (%) | ECE | End-to-end | FPS |
|---|---|---|---|---|---|---|
| Full (seg + keypoints + depth) | 7.0 | 11.7 | 13.6 | 0.027 | ~270 ms | 3.7 |
| w/o depth (seg + keypoints) | 7.1 | 11.8 | 13.4 | 0.016 | ~163 ms | 6.1 |
| seg-only (no depth, no keypoints) | 7.0 | 11.9 | 13.4 | 0.013 | ~137 ms | 7.3 |

**Read:** accuracy is preserved within single-seed noise while end-to-end latency drops ~40% (no-depth) to ~49% (seg-only) — dropping DepthAnything (the ~107 ms dominant stage) and keypoints (~26 ms) costs essentially no accuracy here. The variants are genuine (depth/pose parameters absent, not zeroed).

**Caveats (do not over-read):**
- **Single seed** — accuracy differences (pos 7.0/7.1, geo 11.7→11.9) are within noise; the honest takeaway is **parity**, not that fewer modalities improve accuracy.
- **Latency is by construction** — perception stages are measured and the BTPN forward is held at its ~117 ms reference (self-timed forwards confirm it is modality-independent); the savings (depth ≈107 ms, keypoints ≈26 ms) are real, but the end-to-end totals are estimates, not a fresh wall-clock end-to-end.
- **ECE** appears lower without depth, but this is single-seed and the Full baseline came from a different run context — **not** a claim that dropping modalities improves calibration.

## Ethics Approval Statement
Approval for the dataset collection was granted by the University of Leeds Faculty of Engineering and Physical Sciences Research Ethics Committee (ref: MEEC 22-023). It contains no personally identifiable information nor human body parts. All other datasets used in this study are entirely non-identifiable, open-source, or simulated data and meet the requirements set by the standards at the University of Leeds, so there are no ethical concerns.
