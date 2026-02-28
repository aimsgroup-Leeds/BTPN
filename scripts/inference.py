#!/usr/bin/env python3
"""Single-trial inference demo for BTPN.

Loads a pre-trained BTPN model, runs inference on a single trial,
and outputs predictions with uncertainty estimates.

Usage:
    python scripts/inference.py --checkpoint checkpoints/btpn_supervised.pt --trial data/sample_a
    python scripts/inference.py --checkpoint checkpoints/btpn_supervised.pt --trial /path/to/trial --visualize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure btpn package is importable when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from btpn import BTPNConfig, KinematicFoundationModel, enable_mc_dropout
from btpn.dataset import NormalizationStats, load_trial, DEFAULT_SCALES
from btpn.config import BTPNFeatureConfig

FEATURES = BTPNFeatureConfig()


# =============================================================================
# Multi-Scale Window Construction
# =============================================================================


def build_multiscale_windows(
    normalized_data: np.ndarray,
    frame_idx: int,
    scales: list[int],
) -> list[torch.Tensor]:
    """Build causal multi-scale windows ending at a given frame.

    Each window contains the most recent ``scale`` frames up to and
    including ``frame_idx``.  If the window extends before the trial
    start, it is zero-padded on the left.

    Args:
        normalized_data: Normalized trial data (N, 30).
        frame_idx: Target frame index (window ends here, inclusive).
        scales: List of window sizes.

    Returns:
        List of tensors, one per scale, each shaped (1, window_size, 30).
    """
    windows: list[torch.Tensor] = []
    for window_size in scales:
        start = frame_idx - window_size + 1
        end = frame_idx + 1

        if start >= 0:
            window = normalized_data[start:end].copy()
        else:
            available = normalized_data[:end].copy()
            pad_size = window_size - available.shape[0]
            padding = np.zeros(
                (pad_size, normalized_data.shape[1]), dtype=np.float32
            )
            window = np.concatenate([padding, available], axis=0)

        windows.append(torch.from_numpy(window).unsqueeze(0))
    return windows


# =============================================================================
# Inference with MC Dropout
# =============================================================================


@torch.no_grad()
def run_inference(
    model: KinematicFoundationModel,
    normalized_data: np.ndarray,
    scales: list[int],
    device: torch.device,
    mc_samples: int = 20,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Run frame-by-frame inference with MC Dropout uncertainty.

    For each frame t in [0, N-2], constructs multi-scale causal windows
    ending at t and predicts frame t+1.  When ``mc_samples > 1``, runs
    multiple stochastic forward passes with MC Dropout enabled and
    computes empirical mean and standard deviation.

    Args:
        model: Trained model in eval mode.
        normalized_data: Z-score normalized trial data (N, 30).
        scales: Multi-scale window sizes.
        device: Torch device.
        mc_samples: Number of MC Dropout forward passes.
        batch_size: Frames per batch.

    Returns:
        Dictionary with stacked arrays (all in normalized space):
            mu_position (M, 2, 3), sigma_position (M, 2, 3),
            mu_quaternion (M, 2, 4), kappa_quaternion (M, 2, 1),
            mu_angle (M, 2, 1), sigma_angle (M, 2, 1),
            mc_position_std (M, 2, 3) -- epistemic uncertainty,
            frame_indices (M,)
        where M = N - 1.
    """
    n_frames = normalized_data.shape[0]
    n_predictions = n_frames - 1

    # Accumulators for MC samples
    mc_pos_samples: list[np.ndarray] = []
    mc_quat_samples: list[np.ndarray] = []

    # Aleatoric uncertainty (from single pass -- averaged if MC > 1)
    all_sigma_pos = np.zeros((n_predictions, 2, 3), dtype=np.float32)
    all_kappa = np.zeros((n_predictions, 2, 1), dtype=np.float32)
    all_mu_angle = np.zeros((n_predictions, 2, 1), dtype=np.float32)
    all_sigma_angle = np.zeros((n_predictions, 2, 1), dtype=np.float32)

    use_mc = mc_samples > 1

    for mc_idx in range(mc_samples):
        pos_this = np.zeros((n_predictions, 2, 3), dtype=np.float32)
        quat_this = np.zeros((n_predictions, 2, 4), dtype=np.float32)
        sigma_pos_this = np.zeros((n_predictions, 2, 3), dtype=np.float32)
        kappa_this = np.zeros((n_predictions, 2, 1), dtype=np.float32)
        angle_this = np.zeros((n_predictions, 2, 1), dtype=np.float32)
        sigma_angle_this = np.zeros((n_predictions, 2, 1), dtype=np.float32)

        for batch_start in range(0, n_predictions, batch_size):
            batch_end = min(batch_start + batch_size, n_predictions)

            batch_scales: list[list[torch.Tensor]] = [[] for _ in scales]
            for i in range(batch_start, batch_end):
                windows = build_multiscale_windows(normalized_data, i, scales)
                for s_idx, w in enumerate(windows):
                    batch_scales[s_idx].append(w)

            multi_scale_inputs = [
                torch.cat(batch_scales[s_idx], dim=0).to(device)
                for s_idx in range(len(scales))
            ]

            if use_mc:
                with enable_mc_dropout(model):
                    outputs = model(
                        multi_scale_inputs=multi_scale_inputs,
                        force_diagonal=True,
                    )
            else:
                outputs = model(
                    multi_scale_inputs=multi_scale_inputs,
                    force_diagonal=True,
                )

            sl = slice(batch_start, batch_end)
            pos_this[sl] = outputs["mu_position"].cpu().numpy()
            quat_this[sl] = outputs["mu_quaternion"].cpu().numpy()
            sigma_pos_this[sl] = outputs["sigma_position"].cpu().numpy()
            kappa_this[sl] = outputs["kappa_quaternion"].cpu().numpy()
            angle_this[sl] = outputs["mu_angle"].cpu().numpy()
            sigma_angle_this[sl] = outputs["sigma_angle"].cpu().numpy()

        mc_pos_samples.append(pos_this)
        mc_quat_samples.append(quat_this)

        # Accumulate aleatoric estimates (average across MC runs)
        all_sigma_pos += sigma_pos_this / mc_samples
        all_kappa += kappa_this / mc_samples
        all_mu_angle += angle_this / mc_samples
        all_sigma_angle += sigma_angle_this / mc_samples

    # Stack MC samples: (S, M, 2, 3) and (S, M, 2, 4)
    pos_stack = np.stack(mc_pos_samples, axis=0)
    quat_stack = np.stack(mc_quat_samples, axis=0)

    # Posterior mean position and quaternion
    mu_pos = pos_stack.mean(axis=0)
    mu_quat = quat_stack.mean(axis=0)
    # Renormalize mean quaternion to unit sphere
    quat_norms = np.linalg.norm(mu_quat, axis=-1, keepdims=True)
    mu_quat = mu_quat / np.clip(quat_norms, 1e-8, None)

    # Epistemic uncertainty: std across MC samples
    mc_pos_std = pos_stack.std(axis=0) if use_mc else np.zeros_like(mu_pos)

    return {
        "mu_position": mu_pos,
        "sigma_position": all_sigma_pos,
        "mc_position_std": mc_pos_std,
        "mu_quaternion": mu_quat,
        "kappa_quaternion": all_kappa,
        "mu_angle": all_mu_angle,
        "sigma_angle": all_sigma_angle,
        "frame_indices": np.arange(1, n_frames, dtype=np.int64),
    }


# =============================================================================
# Denormalization
# =============================================================================


def denormalize_predictions(
    predictions: dict[str, np.ndarray],
    raw_data: np.ndarray,
    norm_stats: NormalizationStats,
) -> dict[str, np.ndarray]:
    """Convert normalized predictions back to physical units.

    Position and angle predictions are denormalized using the stored
    z-score statistics. Quaternions are already unit-normalized and
    sigma values are scaled by the corresponding feature std.

    Args:
        predictions: Model outputs in normalized space.
        raw_data: Original un-normalized trial data (N, 30).
        norm_stats: Normalization statistics used during training.

    Returns:
        Dictionary with arrays in physical units (mm, unit quaternions).
    """
    mean, std = norm_stats.mean, norm_stats.std
    frame_indices = predictions["frame_indices"]
    t1p = slice(*FEATURES.tool1_pos)
    t2p = slice(*FEATURES.tool2_pos)

    # Position (mm)
    mu_pos = predictions["mu_position"]
    pred_pos = np.zeros_like(mu_pos)
    pred_pos[:, 0] = mu_pos[:, 0] * std[t1p] + mean[t1p]
    pred_pos[:, 1] = mu_pos[:, 1] * std[t2p] + mean[t2p]

    # Position aleatoric uncertainty (mm)
    sigma_pos = predictions["sigma_position"]
    sigma_pos_mm = np.zeros_like(sigma_pos)
    sigma_pos_mm[:, 0] = sigma_pos[:, 0] * std[t1p]
    sigma_pos_mm[:, 1] = sigma_pos[:, 1] * std[t2p]

    # Epistemic uncertainty (mm)
    mc_std = predictions["mc_position_std"]
    mc_std_mm = np.zeros_like(mc_std)
    mc_std_mm[:, 0] = mc_std[:, 0] * std[t1p]
    mc_std_mm[:, 1] = mc_std[:, 1] * std[t2p]

    # Quaternion (already unit-normalized)
    pred_quat = predictions["mu_quaternion"]

    # Kappa (VMF concentration -- no denormalization)
    kappa = predictions["kappa_quaternion"]

    # Jaw angle (physical units)
    mu_angle = predictions["mu_angle"]
    t1a, t2a = FEATURES.tool1_angle, FEATURES.tool2_angle
    pred_angle = np.zeros_like(mu_angle)
    pred_angle[:, 0] = mu_angle[:, 0] * std[t1a] + mean[t1a]
    pred_angle[:, 1] = mu_angle[:, 1] * std[t2a] + mean[t2a]

    sigma_angle = predictions["sigma_angle"]
    sigma_angle_phys = np.zeros_like(sigma_angle)
    sigma_angle_phys[:, 0] = sigma_angle[:, 0] * std[t1a]
    sigma_angle_phys[:, 1] = sigma_angle[:, 1] * std[t2a]

    # Ground-truth targets
    targets = raw_data[frame_indices]
    tgt_pos = np.stack(
        [targets[:, t1p], targets[:, t2p]], axis=1
    )
    t1q = slice(*FEATURES.tool1_quat)
    t2q = slice(*FEATURES.tool2_quat)
    tgt_quat = np.stack(
        [targets[:, t1q], targets[:, t2q]], axis=1
    )
    tgt_angle = np.stack(
        [targets[:, t1a:t1a + 1], targets[:, t2a:t2a + 1]], axis=1
    )

    return {
        "pred_position": pred_pos,
        "pred_quaternion": pred_quat,
        "pred_angle": pred_angle,
        "sigma_position": sigma_pos_mm,
        "mc_position_std": mc_std_mm,
        "kappa_quaternion": kappa,
        "sigma_angle": sigma_angle_phys,
        "target_position": tgt_pos,
        "target_quaternion": tgt_quat,
        "target_angle": tgt_angle,
        "frame_indices": frame_indices,
    }


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: dict[str, np.ndarray], trial_name: str) -> None:
    """Print a human-readable summary of prediction accuracy.

    Reports position MAE, geodesic rotation error, jaw angle MAE,
    and uncertainty statistics for each tool.

    Args:
        results: Denormalized predictions dictionary.
        trial_name: Trial name for display.
    """
    pred_pos = results["pred_position"]
    tgt_pos = results["target_position"]
    sigma_pos = results["sigma_position"]
    mc_std = results["mc_position_std"]
    pred_quat = results["pred_quaternion"]
    tgt_quat = results["target_quaternion"]
    kappa = results["kappa_quaternion"]

    n_frames = pred_pos.shape[0]
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  BTPN Inference Summary: {trial_name}")
    print(f"  Frames predicted: {n_frames}")
    print(sep)

    # -- Position --
    pos_error = np.linalg.norm(pred_pos - tgt_pos, axis=-1)  # (M, 2)
    for t, name in enumerate(["Tool 1", "Tool 2"]):
        mae = pos_error[:, t].mean()
        p95 = np.percentile(pos_error[:, t], 95)
        sig = sigma_pos[:, t].mean()
        mc = mc_std[:, t].mean()
        print(f"\n  {name} Position:")
        print(f"    MAE  = {mae:.2f} mm   (P95 = {p95:.2f} mm)")
        print(f"    Aleatoric sigma = {sig:.2f} mm")
        if mc > 0:
            print(f"    Epistemic std   = {mc:.2f} mm")

    combined = pos_error.mean()
    print(f"\n  Combined Position MAE: {combined:.2f} mm")

    # -- Rotation --
    for t, name in enumerate(["Tool 1", "Tool 2"]):
        pq = pred_quat[:, t]
        tq = tgt_quat[:, t]
        dot = np.abs(np.sum(pq * tq, axis=-1))
        dot = np.clip(dot, 0.0, 1.0)
        geo_deg = np.degrees(2.0 * np.arccos(dot))
        k = kappa[:, t, 0].mean()
        print(f"\n  {name} Rotation:")
        print(f"    Geodesic = {geo_deg.mean():.2f} deg  (median {np.median(geo_deg):.2f} deg)")
        print(f"    Mean kappa = {k:.1f}")

    # -- Jaw angle --
    pred_angle = results["pred_angle"]
    tgt_angle = results["target_angle"]
    angle_err = np.abs(pred_angle - tgt_angle)
    for t, name in enumerate(["Tool 1", "Tool 2"]):
        print(f"  {name} Jaw Angle MAE: {angle_err[:, t, 0].mean():.4f}")

    print(f"\n{sep}\n")


# =============================================================================
# Visualization
# =============================================================================


def visualize_trajectories(
    results: dict[str, np.ndarray],
    trial_name: str,
    output_path: Path,
) -> None:
    """Plot 3D predicted vs ground-truth trajectories with uncertainty.

    Creates a two-panel figure (one per tool) showing predicted and
    ground-truth 3D positions, with translucent uncertainty tubes around
    the predicted trajectory sized by the aleatoric position sigma.

    Args:
        results: Denormalized predictions dictionary.
        trial_name: Trial name for the figure title.
        output_path: Path to save the figure (PNG).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed -- skipping visualization.")
        return

    pred_pos = results["pred_position"]
    tgt_pos = results["target_position"]
    sigma_pos = results["sigma_position"]

    fig = plt.figure(figsize=(16, 7))
    tool_colors = [("#2196F3", "#90CAF9"), ("#F44336", "#EF9A9A")]
    tool_names = ["Tool 1", "Tool 2"]

    for t in range(2):
        ax = fig.add_subplot(1, 2, t + 1, projection="3d")
        gt = tgt_pos[:, t]
        pr = pred_pos[:, t]

        ax.plot(
            gt[:, 0], gt[:, 1], gt[:, 2],
            color=tool_colors[t][1], linewidth=1.0, alpha=0.7,
            label="Ground truth",
        )
        ax.plot(
            pr[:, 0], pr[:, 1], pr[:, 2],
            color=tool_colors[t][0], linewidth=1.2,
            label="Predicted",
        )

        # Uncertainty band: sample every 20th frame for clarity
        sig = sigma_pos[:, t]
        total_sig = np.linalg.norm(sig, axis=-1)
        step = max(1, len(pr) // 50)
        for i in range(0, len(pr), step):
            r = total_sig[i] * 0.5
            ax.scatter(
                pr[i, 0], pr[i, 1], pr[i, 2],
                s=max(5, r * 10), color=tool_colors[t][0],
                alpha=0.15, edgecolors="none",
            )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{tool_names[t]} -- {trial_name}")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"BTPN Pose Prediction: {trial_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BTPN single-trial inference with uncertainty estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to pre-trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--trial", type=Path, required=True,
        help="Path to trial directory (must contain label.json or .txt frames).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML config override file.",
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=None,
        help="Path to normalization stats (.npz). "
             "Defaults to norm_stats.npz next to the checkpoint.",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=20,
        help="Number of MC Dropout forward passes for epistemic uncertainty.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate 3D trajectory plot and save as PNG.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.npz"),
        help="Output file path for predictions (.npz or .csv).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device string (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    """Run single-trial BTPN inference pipeline."""
    args = parse_args()

    # --- Device ---
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # --- Build config ---
    if args.config is not None:
        config = BTPNConfig.from_yaml(args.config)
    elif "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        field_names = {f.name for f in BTPNConfig.__dataclass_fields__.values()}
        config = BTPNConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
    else:
        config = BTPNConfig()
    print(f"Config: d_model={config.d_model}, scales={config.window_scales}")

    # --- Build model and load weights ---
    model = KinematicFoundationModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # --- Load normalization stats ---
    norm_stats_path = args.norm_stats or ckpt_path.parent / "norm_stats.npz"
    if norm_stats_path.exists():
        norm_stats = NormalizationStats.load(norm_stats_path)
        print(f"Normalization stats loaded: {norm_stats_path}")
    elif "norm_mean" in checkpoint and "norm_std" in checkpoint:
        norm_stats = NormalizationStats(
            mean=checkpoint["norm_mean"],
            std=checkpoint["norm_std"],
        )
        print("Normalization stats loaded from checkpoint.")
    else:
        print(f"ERROR: Normalization stats not found at {norm_stats_path} "
              f"and not embedded in checkpoint.")
        sys.exit(1)

    # --- Load trial ---
    trial_path: Path = args.trial
    if not trial_path.exists():
        print(f"ERROR: Trial directory not found: {trial_path}")
        sys.exit(1)

    result = load_trial(trial_path)
    if result is None:
        print(f"ERROR: Could not load trial data from {trial_path}")
        sys.exit(1)

    raw_data: np.ndarray = result["data"]
    metadata: dict = result["metadata"]
    trial_name = metadata.get("trial_name", trial_path.name)
    print(f"Trial: {trial_name}  ({raw_data.shape[0]} frames, {raw_data.shape[1]}D)")

    # Normalize
    normalized_data = norm_stats.normalize(raw_data)

    # --- Run inference ---
    scales = config.window_scales
    print(f"Running inference: scales={scales}, mc_samples={args.mc_samples} ...")
    t0 = time.perf_counter()

    predictions = run_inference(
        model=model,
        normalized_data=normalized_data,
        scales=scales,
        device=device,
        mc_samples=args.mc_samples,
    )

    elapsed = time.perf_counter() - t0
    n_pred = predictions["frame_indices"].shape[0]
    fps = n_pred / elapsed
    print(f"Inference complete: {n_pred} frames in {elapsed:.2f}s ({fps:.0f} fps)")

    # --- Denormalize ---
    results = denormalize_predictions(predictions, raw_data, norm_stats)

    # --- Save ---
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        _save_csv(results, output_path)
    else:
        np.savez(
            output_path,
            **{k: v for k, v in results.items()},
        )
    print(f"Predictions saved: {output_path}")

    # --- Summary ---
    print_summary(results, trial_name)

    # --- Visualization ---
    if args.visualize:
        fig_path = output_path.with_suffix(".png")
        visualize_trajectories(results, trial_name, fig_path)


def _save_csv(results: dict[str, np.ndarray], path: Path) -> None:
    """Save predictions as a flat CSV file.

    Columns: frame, tool, pred_x, pred_y, pred_z, sigma_x, sigma_y,
    sigma_z, pred_qw, pred_qx, pred_qy, pred_qz, kappa, pred_angle,
    sigma_angle, gt_x, gt_y, gt_z, gt_qw, gt_qx, gt_qy, gt_qz, gt_angle.

    Args:
        results: Denormalized predictions dictionary.
        path: Output CSV path.
    """
    rows: list[str] = []
    header = (
        "frame,tool,"
        "pred_x,pred_y,pred_z,sigma_x,sigma_y,sigma_z,"
        "pred_qw,pred_qx,pred_qy,pred_qz,kappa,"
        "pred_angle,sigma_angle,"
        "gt_x,gt_y,gt_z,gt_qw,gt_qx,gt_qy,gt_qz,gt_angle"
    )
    rows.append(header)

    frames = results["frame_indices"]
    for i, f in enumerate(frames):
        for t in range(2):
            pp = results["pred_position"][i, t]
            sp = results["sigma_position"][i, t]
            pq = results["pred_quaternion"][i, t]
            k = results["kappa_quaternion"][i, t, 0]
            pa = results["pred_angle"][i, t, 0]
            sa = results["sigma_angle"][i, t, 0]
            gp = results["target_position"][i, t]
            gq = results["target_quaternion"][i, t]
            ga = results["target_angle"][i, t, 0]

            vals = [
                str(f), str(t + 1),
                f"{pp[0]:.4f}", f"{pp[1]:.4f}", f"{pp[2]:.4f}",
                f"{sp[0]:.4f}", f"{sp[1]:.4f}", f"{sp[2]:.4f}",
                f"{pq[0]:.6f}", f"{pq[1]:.6f}", f"{pq[2]:.6f}", f"{pq[3]:.6f}",
                f"{k:.2f}",
                f"{pa:.6f}", f"{sa:.6f}",
                f"{gp[0]:.4f}", f"{gp[1]:.4f}", f"{gp[2]:.4f}",
                f"{gq[0]:.6f}", f"{gq[1]:.6f}", f"{gq[2]:.6f}", f"{gq[3]:.6f}",
                f"{ga:.6f}",
            ]
            rows.append(",".join(vals))

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
