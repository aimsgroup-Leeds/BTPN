#!/usr/bin/env python3
"""Reproduce paper figures from evaluation data.

Generates publication-quality figures for the BTPN MICCAI 2026 paper from the
released evaluation predictions. Reads results/evaluation_data.npz (Full-BTPN
outputs under bare keys + embedded mean/std) and converts to physical units
internally. Supports individual figure generation or batch mode.

The paper's four-panel calibration figure (figures/uncertainty_quality.{png,pdf},
with the physical-space Fisher rotation ECE and the per-trial jaw channel) is
produced by scripts/make_uncertainty_figure.py; figure 4 here is a lighter
position-only uncertainty summary (uncertainty.png).

Usage:
    python scripts/generate_figures.py --data results/evaluation_data.npz
    python scripts/generate_figures.py --data results/evaluation_data.npz --figure 3
    python scripts/generate_figures.py --data results/evaluation_data.npz --all

Figures:
    3  -- Trajectory predictions with uncertainty bands (trajectories.png)
    4  -- Uncertainty quality: sparsification, calibration, correlation
           (uncertainty.png)
    S1 -- Per-trial error breakdown box plots (supp_per_trial.png)
    S2 -- Training curves from JSON logs (supp_training_curves.png)

Author: BTPN Publication Repository
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import norm as normal_dist, spearmanr

# ---------------------------------------------------------------------------
# Add parent to path so ``btpn`` package is importable when running as script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btpn.metrics import (
    compute_ause,
    compute_coverage,
    compute_ece,
    compute_geodesic_error,
    compute_position_metrics,
)

# =============================================================================
# Publication style
# =============================================================================

# MICCAI single-column width ~12 cm; double-column ~17.5 cm
SINGLE_COL_CM = 12.0
DOUBLE_COL_CM = 17.5
CM_PER_INCH = 2.54

TOOL1_COLOR = "#2196F3"  # Blue
TOOL2_COLOR = "#F44336"  # Red
GT_COLOR = "#333333"  # Near-black for ground truth
KIN_COLOR = "#9E9E9E"  # Gray for kinematic baseline
ORACLE_COLOR = "#4CAF50"  # Green for oracle curve
RANDOM_COLOR = "#FF9800"  # Orange for random baseline

CONFIDENCE_BAND_ALPHA = 0.20


def setup_mpl_style() -> None:
    """Configure matplotlib for MICCAI publication figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def cm_to_inch(cm: float) -> float:
    """Convert centimetres to inches."""
    return cm / CM_PER_INCH


# =============================================================================
# Data loading
# =============================================================================


def _denormalize_eval_data(
    data: dict[str, np.ndarray],
    norm_path: str | Path | None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Denormalize position/quaternion/sigma arrays in-place to physical units.

    The saved predictions .npz stores ``target_*`` and ``v3_mu_position`` in
    z-score *normalized* space (target quaternion norms are not 1). Figures
    need millimetres and unit quaternions, so this multiplies by the stored
    std (and adds the mean for means; sigma is scaled by std only). Prediction
    quaternions (``*_mu_quaternion``) are already unit-norm and left untouched.

    Feature layout (30-D): T1 pos 0:3, T1 quat 3:7, T1 jaw 7,
    T2 pos 8:11, T2 quat 11:15, T2 jaw 15.

    Args:
        data: Loaded evaluation arrays (modified copy returned).
        norm_path: Path to normalization stats .npz (mean/std, 30-D). Ignored
            when ``mean``/``std`` are passed directly (e.g. embedded in the npz).
        mean, std: 30-D normalization stats to use instead of reading norm_path.

    Returns:
        The denormalized dictionary.
    """
    if mean is None or std is None:
        norm_path = Path(norm_path)
        if not norm_path.exists():
            print(f"  WARNING: norm stats not found at {norm_path}; "
                  f"figures will be in normalized units.")
            return data
        # norm stats are plain numeric arrays -> allow_pickle=False.
        ns = np.load(norm_path, allow_pickle=False)
        mean = ns["mean"]; std = ns["std"]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    pos = (slice(0, 3), slice(8, 11))
    quat = (slice(3, 7), slice(11, 15))
    jaw = (7, 15)

    for t in range(2):
        # Position means: x*std + mean
        for key in ("target_position",):
            if key in data:
                data[key][:, t] = data[key][:, t] * std[pos[t]] + mean[pos[t]]
        for key in ("v3_mu_position", "v2_mu_position", "kin_mu_position"):
            if key in data:
                data[key][:, t] = data[key][:, t] * std[pos[t]] + mean[pos[t]]
        # Position sigmas: scale by std only
        for key in ("v3_sigma_position", "v2_sigma_position", "kin_sigma_position"):
            if key in data:
                data[key][:, t] = data[key][:, t] * std[pos[t]]
        # Target quaternion: x*std + mean (predictions already unit-norm)
        if "target_quaternion" in data:
            data["target_quaternion"][:, t] = (
                data["target_quaternion"][:, t] * std[quat[t]] + mean[quat[t]]
            )
        # Jaw angle means
        for key in ("target_angle", "v3_mu_angle", "v2_mu_angle", "kin_mu_angle"):
            if key in data:
                data[key][:, t] = data[key][:, t] * std[jaw[t]] + mean[jaw[t]]
    return data


def load_eval_data(
    path: str | Path,
    norm_stats: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load evaluation .npz and return contents as a flat dictionary.

    Args:
        path: Path to evaluation_data.npz.
        norm_stats: Optional path to normalization stats (.npz). If given,
            position/quaternion/sigma arrays are denormalized to physical
            units (mm, unit quaternions) so figures show real-world scales.

    Returns:
        Dictionary mapping array names to NumPy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {path}")
    # npz holds only plain numeric arrays -> allow_pickle=False (no code-exec risk).
    data = dict(np.load(path, allow_pickle=False))

    # The released results/evaluation_data.npz stores Full-BTPN outputs under
    # bare keys (mu_position, ...) and embeds the mean/std stats. Alias them to
    # the v3_* keys the figure functions expect, and denormalize using the
    # embedded stats (no external norm-stats file needed).
    if "v3_mu_position" not in data and "mu_position" in data:
        for src, dst in (("mu_position", "v3_mu_position"),
                         ("mu_quaternion", "v3_mu_quaternion"),
                         ("sigma_position", "v3_sigma_position"),
                         ("mu_angle", "v3_mu_angle")):
            if src in data:
                data[dst] = data[src]
        if "mean" in data and "std" in data:
            data = _denormalize_eval_data(data, None,
                                          mean=data["mean"], std=data["std"])
            return data

    if norm_stats is not None:
        data = _denormalize_eval_data(data, norm_stats)
    return data


def infer_trial_boundaries(
    data: dict[str, np.ndarray],
    n_trials: int = 21,
) -> list[tuple[int, int]]:
    """Estimate per-trial (start, end) frame ranges.

    If the evaluation data does not store explicit trial boundaries we
    split the frames into *n_trials* equal-length segments.  This is a
    reasonable approximation when trials have similar durations after
    window-based evaluation.

    Args:
        data: Evaluation arrays (needs ``target_position``).
        n_trials: Expected number of held-out trials.

    Returns:
        List of (start, end) index tuples, one per trial.
    """
    n_frames = data["target_position"].shape[0]
    trial_len = n_frames // n_trials
    boundaries: list[tuple[int, int]] = []
    for i in range(n_trials):
        start = i * trial_len
        end = start + trial_len if i < n_trials - 1 else n_frames
        boundaries.append((start, end))
    return boundaries


# =============================================================================
# Figure 3: Trajectory Predictions
# =============================================================================


def figure_trajectory(
    data: dict[str, np.ndarray],
    output_dir: Path,
    fmt: str = "both",
) -> None:
    """Generate Figure 3 -- trajectory predictions with uncertainty.

    Shows a representative segment of predicted vs ground-truth 3D tool
    positions for both tools, overlaid with +/-2 sigma confidence bands.
    Three rows (X, Y, Z axes) x two columns (Tool 1, Tool 2).

    Args:
        data: Loaded evaluation arrays.
        output_dir: Directory to save figures.
        fmt: Output format -- "png", "pdf", or "both".
    """
    print("Generating Figure 3: Trajectory Predictions ...")

    # Select a representative 400-frame segment (~30 s at 13 fps)
    n_frames = data["target_position"].shape[0]
    seg_len = 400
    # Pick segment from middle of data for diverse motion
    start = n_frames // 3
    end = start + seg_len
    t = np.arange(seg_len) / 13.0  # seconds

    pred_pos = data["v3_mu_position"][start:end]  # (T, 2, 3)
    gt_pos = data["target_position"][start:end]
    sigma_pos = data["v3_sigma_position"][start:end]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(cm_to_inch(DOUBLE_COL_CM), cm_to_inch(10.0)),
        sharex=True,
    )

    axis_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    tool_labels = ["Tool 1", "Tool 2"]
    tool_colors = [TOOL1_COLOR, TOOL2_COLOR]

    for col, (tool_label, color) in enumerate(zip(tool_labels, tool_colors)):
        for row, ax_label in enumerate(axis_labels):
            ax = axes[row, col]
            gt = gt_pos[:, col, row]
            pred = pred_pos[:, col, row]
            sigma = sigma_pos[:, col, row]

            # Ground truth
            ax.plot(t, gt, color=GT_COLOR, linewidth=0.8, label="Ground truth")
            # Prediction
            ax.plot(t, pred, color=color, linewidth=0.8, label="BTPN")
            # +/- 2 sigma band
            ax.fill_between(
                t,
                pred - 2 * sigma,
                pred + 2 * sigma,
                color=color,
                alpha=CONFIDENCE_BAND_ALPHA,
                label=r"$\pm 2\sigma$",
            )

            ax.set_ylabel(ax_label)
            if row == 0:
                ax.set_title(tool_label, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Time (s)")

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, output_dir / "trajectories", fmt)
    plt.close(fig)
    print("  Saved trajectories figure.")


# =============================================================================
# Figure 4: Uncertainty Quality
# =============================================================================


def figure_uncertainty(
    data: dict[str, np.ndarray],
    output_dir: Path,
    fmt: str = "both",
) -> None:
    """Generate Figure 4 -- uncertainty calibration and quality.

    Three panels:
        (a) Reliability diagram (observed vs expected coverage)
        (b) Sparsification error curve (model vs oracle vs random)
        (c) Predicted sigma vs actual error scatter/hexbin

    Args:
        data: Loaded evaluation arrays.
        output_dir: Directory to save figures.
        fmt: Output format -- "png", "pdf", or "both".
    """
    print("Generating Figure 4: Uncertainty Quality ...")

    # Combine both tools for aggregate statistics
    pred_pos = data["v3_mu_position"]  # (N, 2, 3)
    gt_pos = data["target_position"]
    sigma_pos = data["v3_sigma_position"]

    # Per-sample Euclidean position error (both tools)
    err_t1 = np.linalg.norm(pred_pos[:, 0] - gt_pos[:, 0], axis=-1)
    err_t2 = np.linalg.norm(pred_pos[:, 1] - gt_pos[:, 1], axis=-1)
    errors = np.concatenate([err_t1, err_t2])

    # Mean sigma per sample (average across 3 axes)
    sig_t1 = np.linalg.norm(sigma_pos[:, 0], axis=-1)
    sig_t2 = np.linalg.norm(sigma_pos[:, 1], axis=-1)
    sigmas = np.concatenate([sig_t1, sig_t2])

    fig, axes = plt.subplots(
        1, 3,
        figsize=(cm_to_inch(DOUBLE_COL_CM), cm_to_inch(5.5)),
    )

    # --- (a) Reliability diagram ---
    ax = axes[0]
    coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.683, 0.7, 0.8, 0.9, 0.95, 0.99]
    expected = []
    observed = []
    for level in coverage_levels:
        z = normal_dist.ppf(0.5 + level / 2.0)
        within = float((errors <= z * sigmas).mean())
        expected.append(level)
        observed.append(within)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, label="Ideal")
    ax.plot(expected, observed, "o-", color=TOOL1_COLOR, markersize=3, linewidth=1.0, label="BTPN")
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("(a) Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")

    # Compute and annotate ECE
    ece_result = compute_ece(errors, sigmas)
    ece_val = ece_result["ece"]
    ax.text(
        0.05, 0.90,
        f"ECE = {ece_val:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
    )

    # --- (b) Sparsification curve ---
    ax = axes[1]
    ause_result = compute_ause(errors, sigmas, n_steps=50)
    fracs = np.array(ause_result["fractions"])
    sigma_curve = np.array(ause_result["sigma_curve"])
    oracle_curve = np.array(ause_result["oracle_curve"])
    random_curve = np.array(ause_result["random_curve"])

    ax.plot(fracs, sigma_curve, color=TOOL1_COLOR, linewidth=1.0, label="BTPN")
    ax.plot(fracs, oracle_curve, color=ORACLE_COLOR, linewidth=1.0, linestyle="--", label="Oracle")
    ax.plot(fracs, random_curve, color=RANDOM_COLOR, linewidth=1.0, linestyle=":", label="Random")

    # Shade area between model and oracle
    ax.fill_between(
        fracs, oracle_curve, sigma_curve,
        color=TOOL1_COLOR, alpha=0.12,
    )

    ax.set_xlabel("Fraction removed")
    ax.set_ylabel("Mean error (mm)")
    ax.set_title("(b) Sparsification curve")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # Annotate AUSE
    ause_norm = ause_result["ause_normalized"]
    ax.text(
        0.95, 0.90,
        f"AUSE = {ause_norm:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        verticalalignment="top",
    )

    # --- (c) Sigma vs error scatter ---
    ax = axes[2]

    # Subsample for readability
    rng = np.random.default_rng(42)
    n_show = min(5000, len(errors))
    idx = rng.choice(len(errors), size=n_show, replace=False)

    ax.hexbin(
        sigmas[idx], errors[idx],
        gridsize=40,
        cmap="Blues",
        mincnt=1,
        linewidths=0.2,
    )

    # Linear fit line
    slope, intercept = np.polyfit(sigmas[idx], errors[idx], 1)
    x_fit = np.linspace(sigmas[idx].min(), sigmas[idx].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color=TOOL2_COLOR, linewidth=1.0, linestyle="--")

    # Spearman correlation
    rho, p_val = spearmanr(sigmas[idx], errors[idx])
    ax.text(
        0.05, 0.90,
        rf"$\rho_s$ = {rho:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
    )

    ax.set_xlabel(r"Predicted $\|\sigma\|$ (mm)")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("(c) Error vs uncertainty")

    plt.tight_layout()
    _save_figure(fig, output_dir / "uncertainty", fmt)
    plt.close(fig)
    print("  Saved uncertainty figure.")


# =============================================================================
# Supplementary S1: Per-trial Breakdown
# =============================================================================


def figure_per_trial(
    data: dict[str, np.ndarray],
    output_dir: Path,
    fmt: str = "both",
) -> None:
    """Generate Supplementary Figure S1 -- per-trial error box plots.

    Two panels: (a) position RMSE per trial, (b) rotation RMSE per trial,
    each showing Tool 1 and Tool 2 side by side.

    Args:
        data: Loaded evaluation arrays.
        output_dir: Directory to save figures.
        fmt: Output format -- "png", "pdf", or "both".
    """
    print("Generating Supplementary Figure S1: Per-trial Breakdown ...")

    boundaries = infer_trial_boundaries(data)
    n_trials = len(boundaries)

    pred_pos = data["v3_mu_position"]  # (N, 2, 3)
    gt_pos = data["target_position"]
    pred_quat = data["v3_mu_quaternion"]  # (N, 2, 4)
    gt_quat = data["target_quaternion"]

    pos_errors_t1: list[np.ndarray] = []
    pos_errors_t2: list[np.ndarray] = []
    rot_errors_t1: list[np.ndarray] = []
    rot_errors_t2: list[np.ndarray] = []

    for s, e in boundaries:
        # Position: per-frame Euclidean distance
        pe1 = np.linalg.norm(pred_pos[s:e, 0] - gt_pos[s:e, 0], axis=-1)
        pe2 = np.linalg.norm(pred_pos[s:e, 1] - gt_pos[s:e, 1], axis=-1)
        pos_errors_t1.append(pe1)
        pos_errors_t2.append(pe2)

        # Rotation: geodesic error
        geo1 = compute_geodesic_error(pred_quat[s:e, 0], gt_quat[s:e, 0])
        geo2 = compute_geodesic_error(pred_quat[s:e, 1], gt_quat[s:e, 1])
        rot_errors_t1.append(np.full(1, geo1["rmse_deg"]))
        rot_errors_t2.append(np.full(1, geo2["rmse_deg"]))

    # Per-trial RMSE scalars for box plots
    pos_rmse_t1 = [float(np.sqrt(np.mean(e ** 2))) for e in pos_errors_t1]
    pos_rmse_t2 = [float(np.sqrt(np.mean(e ** 2))) for e in pos_errors_t2]
    rot_rmse_t1 = [float(e[0]) for e in rot_errors_t1]
    rot_rmse_t2 = [float(e[0]) for e in rot_errors_t2]

    trial_ids = list(range(1, n_trials + 1))
    x = np.arange(n_trials)
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(cm_to_inch(DOUBLE_COL_CM), cm_to_inch(10.0)),
        sharex=True,
    )

    # --- (a) Position RMSE ---
    ax1.bar(
        x - bar_width / 2, pos_rmse_t1, bar_width,
        color=TOOL1_COLOR, alpha=0.85, label="Tool 1",
    )
    ax1.bar(
        x + bar_width / 2, pos_rmse_t2, bar_width,
        color=TOOL2_COLOR, alpha=0.85, label="Tool 2",
    )
    # Overall mean line
    overall_pos = np.mean(pos_rmse_t1 + pos_rmse_t2)
    ax1.axhline(overall_pos, color=GT_COLOR, linestyle="--", linewidth=0.8, label=f"Mean: {overall_pos:.1f} mm")
    ax1.set_ylabel("Position RMSE (mm)")
    ax1.set_title("(a) Per-trial position error")
    ax1.legend(frameon=False, fontsize=7, ncol=3)

    # --- (b) Rotation RMSE ---
    ax2.bar(
        x - bar_width / 2, rot_rmse_t1, bar_width,
        color=TOOL1_COLOR, alpha=0.85, label="Tool 1",
    )
    ax2.bar(
        x + bar_width / 2, rot_rmse_t2, bar_width,
        color=TOOL2_COLOR, alpha=0.85, label="Tool 2",
    )
    overall_rot = np.mean(rot_rmse_t1 + rot_rmse_t2)
    ax2.axhline(overall_rot, color=GT_COLOR, linestyle="--", linewidth=0.8, label=f"Mean: {overall_rot:.1f}\u00b0")
    ax2.set_ylabel("Rotation RMSE (\u00b0)")
    ax2.set_xlabel("Held-out trial index")
    ax2.set_title("(b) Per-trial rotation error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(trial_ids, fontsize=7)
    ax2.legend(frameon=False, fontsize=7, ncol=3)

    plt.tight_layout()
    _save_figure(fig, output_dir / "supp_per_trial", fmt)
    plt.close(fig)
    print("  Saved per-trial breakdown figure.")


# =============================================================================
# Supplementary S2: Training Curves
# =============================================================================


def figure_training_curves(
    log_path: Path | None,
    output_dir: Path,
    fmt: str = "both",
) -> None:
    """Generate Supplementary Figure S2 -- training loss curves.

    Expects a JSON-lines log file where each line is a dict with at
    minimum ``epoch``, ``train_loss``, and optionally ``val_loss``.
    Produces a two-panel figure: (a) total loss and (b) component losses
    if available.

    Args:
        log_path: Path to training log JSON. Skipped if *None* or missing.
        output_dir: Directory to save figures.
        fmt: Output format -- "png", "pdf", or "both".
    """
    if log_path is None or not log_path.exists():
        print("  Skipping Figure S2: no training log found.")
        return

    print("Generating Supplementary Figure S2: Training Curves ...")

    records: list[dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("  Skipping Figure S2: log file is empty or unparseable.")
        return

    epochs = [r["epoch"] for r in records if "epoch" in r]
    train_loss = [r.get("train_loss") for r in records if "epoch" in r]
    val_loss = [r.get("val_loss") for r in records if "epoch" in r]

    # Filter None values
    has_val = any(v is not None for v in val_loss)

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(SINGLE_COL_CM), cm_to_inch(6.0)),
    )

    valid_train = [(e, l) for e, l in zip(epochs, train_loss) if l is not None]
    if valid_train:
        ep, tl = zip(*valid_train)
        ax.plot(ep, tl, color=TOOL1_COLOR, linewidth=1.0, label="Train loss")

    if has_val:
        valid_val = [(e, l) for e, l in zip(epochs, val_loss) if l is not None]
        if valid_val:
            ep, vl = zip(*valid_val)
            ax.plot(ep, vl, color=TOOL2_COLOR, linewidth=1.0, label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False)

    # Log scale if range spans >2 orders of magnitude
    if valid_train:
        loss_vals = [v for _, v in valid_train]
        if max(loss_vals) / max(min(loss_vals), 1e-12) > 100:
            ax.set_yscale("log")

    plt.tight_layout()
    _save_figure(fig, output_dir / "supp_training_curves", fmt)
    plt.close(fig)
    print("  Saved training curves figure.")


# =============================================================================
# Save helpers
# =============================================================================


def _save_figure(fig: plt.Figure, stem: Path, fmt: str) -> None:
    """Save figure in requested format(s).

    Args:
        fig: Matplotlib figure to save.
        stem: Output path without extension (e.g. ``figures/trajectories``).
        fmt: ``"png"``, ``"pdf"``, or ``"both"``.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        fig.savefig(str(stem) + ".png", dpi=300)
    if fmt in ("pdf", "both"):
        fig.savefig(str(stem) + ".pdf")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate BTPN paper figures from evaluation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to evaluation_data.npz (required).",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help=(
            "Generate a specific figure. Options: 3, 4, S1, S2. "
            "Omit to generate only main figures (3, 4)."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all figures (main + supplementary).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory (default: figures/).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "both"],
        default="both",
        help="Output format (default: both).",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Path to JSON-lines training log for Figure S2.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path("checkpoints/btpn_norm.npz"),
        help=(
            "Normalization stats (.npz) used to denormalize the saved arrays "
            "to physical units (mm, unit quaternions). "
            "Default: checkpoints/btpn_norm.npz. Pass an empty string to skip."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse arguments and dispatch figure generation."""
    args = parse_args()
    setup_mpl_style()

    # Denormalize to physical units when a norm-stats file is given. The
    # committed evaluation_data.npz stores targets in z-score space, so the
    # default points at checkpoints/btpn_norm.npz; pass --norm-stats "" to skip.
    norm_stats = args.norm_stats if str(args.norm_stats) != "" else None
    if norm_stats is not None and not Path(norm_stats).is_absolute():
        norm_stats = Path(__file__).resolve().parent.parent / norm_stats
    data = load_eval_data(args.data, norm_stats=norm_stats)
    output_dir = args.output_dir
    fmt = args.format

    # Resolve which figures to generate
    if args.all:
        targets = {"3", "4", "S1", "S2"}
    elif args.figure is not None:
        targets = {args.figure.upper()}
    else:
        # Default: main figures only
        targets = {"3", "4"}

    print(f"Output directory: {output_dir}")
    print(f"Format: {fmt}")
    print(f"Figures: {', '.join(sorted(targets))}")
    print()

    if "3" in targets:
        figure_trajectory(data, output_dir, fmt)

    if "4" in targets:
        figure_uncertainty(data, output_dir, fmt)

    if "S1" in targets:
        figure_per_trial(data, output_dir, fmt)

    if "S2" in targets:
        log_path = args.training_log
        # Auto-detect common log locations
        if log_path is None:
            candidates = [
                Path("results/training_log.json"),
                Path("results/train_log.jsonl"),
                Path("logs/training.json"),
            ]
            for c in candidates:
                if c.exists():
                    log_path = c
                    break
        figure_training_curves(log_path, output_dir, fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
