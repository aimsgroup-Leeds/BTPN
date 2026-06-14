#!/usr/bin/env python3
"""Reproduce Figure 4 (uncertainty quality) from the released evaluation data.

Self-contained, CPU-only. Reads ``results/evaluation_data.npz`` -- which carries
the Full-BTPN per-sample predictions, their uncertainties, the held-out targets,
the z-score normalisation statistics, a per-sample ``trial_ids`` index and the
detector confidence ``det_conf`` -- and renders the four-panel uncertainty /
calibration figure used in the paper:

  (a) Reliability diagram: position / rotation / jaw observed-vs-expected
      coverage, annotated with each channel's ECE.
  (b) Mean position error binned by predicted sigma (Pearson r).
  (c) Sparsification curve (model-by-sigma vs oracle-by-error), AUSE.
  (d) Position error stratified by detection confidence.

All ECE / coverage maths (physical-space quaternion denormalisation, Fisher
rotation sigma from kappa, quantile-binned ECE) match the canonical pipeline, so
panel (a)'s ECEs equal the committed calibration set. A hard gate asserts the
position / rotation / jaw ECEs match the honest targets before the figure is
written.

Usage:
    python scripts/make_uncertainty_figure.py
    python scripts/make_uncertainty_figure.py --data results/evaluation_data.npz \
        --out figures/uncertainty_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = REPO_ROOT / "results" / "evaluation_data.npz"
DEFAULT_OUT = REPO_ROOT / "figures" / "uncertainty_quality"

# Honest ECE targets (interval-2, physical-space rotation) for the hard gate.
ECE_TARGETS = {"position": 0.028, "rotation": 0.301, "jaw": 0.079}
GATE_TOL = 0.005

COLORS = {
    "gt": "#212121", "tool1": "#2196F3", "tool2": "#F44336",
    "aleatoric": "#FF9800", "ideal": "#9E9E9E",
    "gap_good": "#4CAF50", "gap_bad": "#FF5722",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


# --------------------------------------------------------------------------
# Metric maths (match the canonical evaluation pipeline)
# --------------------------------------------------------------------------

def compute_ece(errors, sigmas, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sigmas, quantiles)
    levels = [0.5, 0.683, 0.9, 0.95, 0.99]
    z_scores = [0.6745, 1.0, 1.6449, 1.96, 2.576]
    total_ece, total_count = 0.0, 0
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i < n_bins - 1 else np.inf
        mask = (sigmas >= lo) & (sigmas <= hi) if i == 0 else (sigmas > lo) & (sigmas <= hi)
        n = mask.sum()
        if n == 0:
            continue
        observed_68 = (errors[mask] <= sigmas[mask]).mean()
        total_ece += abs(observed_68 - 0.683) * n
        total_count += n
    ece = total_ece / max(total_count, 1)
    gc = {f"{lv:.3f}": float((errors <= z * sigmas).mean()) for lv, z in zip(levels, z_scores)}
    return {"ece": float(ece), "n_total": int(total_count), "global_coverages": gc}


def denormalize_quaternions(norm_quats, mean, std):
    result = np.zeros_like(norm_quats)
    result[:, 0] = norm_quats[:, 0] * std[3:7] + mean[3:7]
    result[:, 1] = norm_quats[:, 1] * std[11:15] + mean[11:15]
    for tool in range(2):
        norms = np.linalg.norm(result[:, tool], axis=-1, keepdims=True)
        result[:, tool] /= np.clip(norms, 1e-8, None)
    return result


def geodesic_deg(pred_q, tgt_q):
    pred_q = pred_q / np.linalg.norm(pred_q, axis=-1, keepdims=True)
    tgt_q = tgt_q / np.linalg.norm(tgt_q, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(pred_q * tgt_q, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def _plot_reliability(ax, ece_data) -> None:
    ax.plot([0.35, 1.05], [0.35, 1.05], "--", color=COLORS["ideal"],
            linewidth=0.8, label="Ideal", zorder=1)
    levels = ["0.500", "0.683", "0.900", "0.950", "0.990"]
    expected = [float(l) for l in levels]
    for key, color, marker in [("position", COLORS["tool1"], "o"),
                               ("rotation", COLORS["tool2"], "s"),
                               ("jaw", COLORS["aleatoric"], "^")]:
        cov = ece_data[key]["global_coverages"]
        ece = ece_data[key]["ece"]
        obs = [cov[l] for l in levels]
        label = {"position": "Position", "rotation": "Rotation", "jaw": "Jaw"}[key]
        ax.plot(expected, obs, marker + "-", color=color, markersize=5,
                linewidth=1.2, zorder=3, label=f"{label} ({ece:.3f})")
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85,
              bbox_to_anchor=(1.12, 0.02), borderaxespad=0)
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0.35, 1.05)
    ax.set_aspect("equal")
    ax.text(-0.02, 1.05, "(a) Reliability diagram", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom")


def _plot_uncertainty_error_binned(ax, sigma_mm, error_mm, n_bins=10) -> float:
    clip = np.percentile(sigma_mm, 99)
    sigma_clip = sigma_mm[sigma_mm <= clip]
    error_clip = error_mm[sigma_mm <= clip]
    bin_edges = np.linspace(sigma_clip.min(), sigma_clip.max(), n_bins + 1)
    bin_indices = np.clip(np.digitize(sigma_clip, bin_edges, right=True), 1, n_bins)
    means, stds, centers, widths = [], [], [], []
    for b in range(1, n_bins + 1):
        bmask = bin_indices == b
        if bmask.sum() < 5:
            continue
        means.append(error_clip[bmask].mean())
        stds.append(error_clip[bmask].std() / np.sqrt(bmask.sum()))
        lo, hi = bin_edges[b - 1], bin_edges[b]
        centers.append((lo + hi) / 2)
        widths.append(hi - lo)
    means = np.array(means); stds = np.array(stds)
    centers = np.array(centers); widths = np.array(widths)
    ax.bar(centers, means, width=widths * 0.85, color=COLORS["tool1"], alpha=0.55,
           edgecolor=COLORS["tool1"], linewidth=0.6, zorder=2)
    ax.errorbar(centers, means, yerr=stds, fmt="none", ecolor=COLORS["gt"],
                elinewidth=0.8, capsize=2, capthick=0.6, zorder=3)
    r_value = stats.pearsonr(sigma_mm, error_mm)[0]
    ax.text(0.97, 0.92, f"$r$ = {r_value:.2f}", transform=ax.transAxes, fontsize=7,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#BDBDBD", linewidth=0.5, alpha=0.85))
    ax.set_xlabel(r"Predicted $\sigma$ bin (mm)")
    ax.set_ylabel("Mean position error (mm)")
    ax.text(-0.02, 1.05, r"(b) Error by predicted uncertainty", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom")
    return float(r_value)


def _plot_sparsification(ax, sigma_mm, error_mm) -> float:
    n = len(error_mm)
    fractions = np.linspace(0, 0.95, 50)
    oracle = []
    for frac in fractions:
        n_remove = int(n * frac)
        remaining = np.sort(error_mm)[:n - n_remove] if n_remove < n else np.array([0.0])
        oracle.append(remaining.mean())
    model_order = np.argsort(sigma_mm)[::-1]
    model = []
    for frac in fractions:
        n_remove = int(n * frac)
        idx = model_order[n_remove:]
        model.append(error_mm[idx].mean() if len(idx) else 0.0)
    random = [error_mm.mean()] * len(fractions)
    oracle = np.array(oracle); model = np.array(model)
    ax.plot(fractions * 100, oracle, "--", color=COLORS["ideal"], linewidth=1.0,
            label="Oracle (by error)")
    ax.plot(fractions * 100, model, "-", color=COLORS["tool1"], linewidth=1.2,
            label=r"Model (by $\sigma$)")
    ax.plot(fractions * 100, random, ":", color=COLORS["gt"], linewidth=0.8, alpha=0.5,
            label="Random")
    ause = np.trapezoid(model - oracle, fractions)
    ax.fill_between(fractions * 100, oracle, model, alpha=0.12, color=COLORS["tool1"],
                    label=f"AUSE = {ause:.2f} mm")
    ax.set_xlabel("Fraction removed (%)")
    ax.set_ylabel("Mean position error (mm)")
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
    ax.text(-0.02, 1.05, "(c) Sparsification curve", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom")
    return float(ause)


def _plot_confidence_boxplots(ax, det_conf, error_mm) -> dict:
    mean_conf = det_conf.mean(axis=1)
    bins = {
        "High\n(>0.7)": error_mm[mean_conf > 0.7],
        "Medium\n(0.3-0.7)": error_mm[(mean_conf >= 0.3) & (mean_conf <= 0.7)],
        "Low\n(<0.3)": error_mm[mean_conf < 0.3],
    }
    box_data = [bins[k] for k in bins]
    labels = list(bins.keys())
    counts = [len(d) for d in box_data]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showfliers=False,
                    widths=0.5, medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
    for patch, color in zip(bp["boxes"], [COLORS["gap_good"], "#FFC107", COLORS["gap_bad"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for i, (count, data) in enumerate(zip(counts, box_data)):
        if len(data) == 0:
            continue
        if i < 2:
            q3 = np.percentile(data, 75)
            iqr = q3 - np.percentile(data, 25)
            y = min(data[data <= q3 + 1.5 * iqr].max(), data.max()) + 1.0
            va = "bottom"
        else:
            q1 = np.percentile(data, 25)
            iqr = np.percentile(data, 75) - q1
            y = max(data[data >= q1 - 1.5 * iqr].min(), data.min()) - 1.0
            va = "top"
        ax.text(i + 1, y, f"n={count}", ha="center", va=va, fontsize=7, color="#616161")
    ax.set_ylabel("Position error (mm)")
    ax.text(-0.02, 1.05, "(d) Error by detection confidence", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom")
    return {
        "high": (float(box_data[0].mean()), counts[0]),
        "medium": (float(box_data[1].mean()) if counts[1] else float("nan"), counts[1]),
        "low": (float(box_data[2].mean()) if counts[2] else float("nan"), counts[2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output stem (writes <stem>.png and <stem>.pdf).")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    setup_style()
    # npz holds only plain numeric arrays -> allow_pickle=False (no code-exec risk).
    d = np.load(args.data, allow_pickle=False)
    mean, std = d["mean"], d["std"]
    mu_pos, sig_pos = d["mu_position"], d["sigma_position"]
    mu_q, kappa = d["mu_quaternion"], d["kappa_quaternion"]
    mu_a, sig_a = d["mu_angle"], d["sigma_angle"]
    t_pos, t_q, t_a = d["target_position"], d["target_quaternion"], d["target_angle"]
    det_conf = d["det_conf"]
    N = mu_pos.shape[0]

    t1s, t1m = std[0:3], mean[0:3]
    t2s, t2m = std[8:11], mean[8:11]
    p1 = mu_pos[:, 0] * t1s + t1m; p2 = mu_pos[:, 1] * t2s + t2m
    g1 = t_pos[:, 0] * t1s + t1m; g2 = t_pos[:, 1] * t2s + t2m
    err_l2_t1 = np.linalg.norm(p1 - g1, axis=1)
    err_l2_t2 = np.linalg.norm(p2 - g2, axis=1)
    s1 = sig_pos[:, 0] * t1s; s2 = sig_pos[:, 1] * t2s

    # panel (a) ECEs
    pos_ece = compute_ece(np.concatenate([err_l2_t1, err_l2_t2]),
                          np.concatenate([np.linalg.norm(s1, axis=1),
                                          np.linalg.norm(s2, axis=1)]))
    tgt_phys = denormalize_quaternions(t_q, mean, std)
    geo_all_rad = np.radians(np.concatenate([geodesic_deg(mu_q[:, 0], tgt_phys[:, 0]),
                                             geodesic_deg(mu_q[:, 1], tgt_phys[:, 1])]))
    kappas = kappa.reshape(-1)
    rot_ece = compute_ece(geo_all_rad, 1.0 / np.sqrt(np.maximum(kappas, 1.0)))
    jaw_ece = compute_ece(np.abs(mu_a - t_a).reshape(-1), sig_a.reshape(-1))
    ece_data = {"position": pos_ece, "rotation": rot_ece, "jaw": jaw_ece}

    # hard gate
    measured = {"position": pos_ece["ece"], "rotation": rot_ece["ece"], "jaw": jaw_ece["ece"]}
    print("=== ECE GATE (target / measured / |diff|) ===")
    gate_ok = True
    for k in ("position", "rotation", "jaw"):
        diff = abs(measured[k] - ECE_TARGETS[k])
        ok = diff <= GATE_TOL
        gate_ok &= ok
        print(f"  {k:9s} target={ECE_TARGETS[k]:.3f}  measured={measured[k]:.4f}  "
              f"|diff|={diff:.4f}  {'OK' if ok else 'FAIL'}")
    rot_cov50 = rot_ece["global_coverages"]["0.500"]
    print(f"  rotation over-coverage @ expected 0.50 = {rot_cov50:.3f} "
          f"(must be >>0.50; honest under-confidence)")
    if rot_cov50 <= 0.50:
        gate_ok = False
    if not gate_ok and not args.no_gate:
        print("\nGATE FAILED -- not writing figure.")
        return 1
    print("GATE PASSED.\n" if gate_ok else "GATE FAILED (continuing: --no-gate).\n")

    # panels (b)(c)(d): per-frame sigma (mm) & error (mm)
    sigma_mm = np.stack([s1, s2], axis=1).mean(axis=(1, 2))
    error_mm = (err_l2_t1 + err_l2_t2) / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))
    _plot_reliability(axes[0, 0], ece_data)
    r_value = _plot_uncertainty_error_binned(axes[0, 1], sigma_mm, error_mm)
    ause = _plot_sparsification(axes[1, 0], sigma_mm, error_mm)
    det_stats = _plot_confidence_boxplots(axes[1, 1], det_conf, error_mm)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.out.with_suffix(".png")
    pdf_path = args.out.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    n = len(error_mm)
    keep = np.argsort(sigma_mm)[: n - n // 2]
    print("=== CAPTION NUMBERS ===")
    print(f"N frames = {N}; tool-points = {2 * N}")
    print(f"Panel (a) ECE  position={pos_ece['ece']:.3f}  rotation={rot_ece['ece']:.3f}  "
          f"jaw={jaw_ece['ece']:.3f}")
    print(f"Panel (b) Pearson r = {r_value:.3f}")
    print(f"Panel (c) AUSE = {ause:.3f} mm;  discard 50%: "
          f"{error_mm.mean():.3f} -> {error_mm[keep].mean():.3f} mm")
    print(f"Panel (d) MAE  high={det_stats['high'][0]:.2f}mm(n={det_stats['high'][1]})  "
          f"medium={det_stats['medium'][0]:.2f}mm(n={det_stats['medium'][1]})  "
          f"low={det_stats['low'][0]:.2f}mm(n={det_stats['low'][1]})")
    print(f"\nWrote {png_path}\nWrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
