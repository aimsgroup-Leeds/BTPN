#!/usr/bin/env python3
"""Evaluate BTPN model and reproduce paper tables.

Runs inference with MC Dropout for uncertainty estimation, computes all
paper metrics (position RMSE, rotation geodesic RMSE, jaw angle MAE, ECE,
AUSE), and optionally generates LaTeX tables matching the paper format.

Supports evaluation on individual datasets (A = 7DOF2024, B = BAPES2024,
C = 6DOF2023) or all three for cross-dataset analysis.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset A
    python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset all --output-tables
    python scripts/evaluate.py --checkpoint checkpoints/kinematic_foundation.pt --stage foundation --dataset A

Author: BTPN Publication Repository
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from btpn.config import BTPNConfig, BTPNFeatureConfig
from btpn.model import KinematicFoundationModel, BTPN
from btpn.dataset import (
    NormalizationStats,
    discover_trials,
    load_trial,
    split_7dof_fixed,
    KinematicDataset,
    VisualTemporalDataset,
    collate_multiscale,
    collate_visual_temporal,
    DEFAULT_SCALES,
)
from btpn.metrics import (
    compute_position_metrics,
    compute_geodesic_error,
    compute_euler_errors,
    compute_jaw_metrics,
    compute_ece,
    compute_ause,
    compute_coverage,
    format_results_table,
    format_latex_table,
    format_latex_row,
    _normalize_quaternions,
)
from btpn.utils import (
    enable_mc_dropout,
    set_seed,
    count_parameters,
)


# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_MAP: dict[str, str] = {
    "A": "7DOF2024",
    "B": "BAPES2024",
    "C": "6DOF2023",
}


# ============================================================================
# Data Loading
# ============================================================================


def _load_paths_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load paths.yaml configuration.

    Args:
        config_path: Path to paths.yaml. Defaults to configs/paths.yaml.

    Returns:
        Parsed YAML dictionary.
    """
    if config_path is None:
        config_path = REPO_ROOT / "configs" / "paths.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_dataset(
    dataset_key: str,
    paths_config: dict[str, Any],
    config: BTPNConfig,
    norm_stats: NormalizationStats | None = None,
    is_visual: bool = False,
) -> tuple[torch.utils.data.DataLoader, NormalizationStats, list[dict[str, Any]]]:
    """Load a dataset and create a DataLoader for evaluation.

    For Dataset A (7DOF2024), uses the fixed validation split.
    For Datasets B (BAPES2024) and C (6DOF2023), evaluates on all trials.

    Args:
        dataset_key: One of "A", "B", "C".
        paths_config: Paths configuration dict.
        config: BTPN configuration.
        norm_stats: Pre-computed normalization stats. If None, computed from
            Dataset A training split.
        is_visual: Whether to create VisualTemporalDataset (True) or
            KinematicDataset (False).

    Returns:
        Tuple of (data_loader, norm_stats, trial_metadata_list).
    """
    data_root = Path(paths_config["data_root"])
    dataset_name = DATASET_MAP[dataset_key]

    print(f"Loading Dataset {dataset_key} ({dataset_name})...")
    trial_dirs = discover_trials(data_root, dataset_name)
    print(f"  Found {len(trial_dirs)} trials")

    # For Dataset A, use fixed val split. For B and C, use all trials.
    if dataset_key == "A":
        _, val_dirs = split_7dof_fixed(trial_dirs, seed=config.seed)
        eval_dirs = val_dirs
    else:
        eval_dirs = trial_dirs

    # Load trial data
    trial_data: list[np.ndarray] = []
    trial_meta: list[dict[str, Any]] = []
    for td in eval_dirs:
        result = load_trial(td)
        if result is not None:
            trial_data.append(result["data"])
            trial_meta.append(result["metadata"])

    print(f"  Loaded {len(trial_data)} trials for evaluation")

    # Compute or reuse norm stats from Dataset A training split
    if norm_stats is None:
        print("  Computing normalisation stats from Dataset A training split...")
        a_dirs = discover_trials(data_root, "7DOF2024")
        train_dirs, _ = split_7dof_fixed(a_dirs, seed=config.seed)
        train_data: list[np.ndarray] = []
        for td in train_dirs:
            r = load_trial(td)
            if r is not None:
                train_data.append(r["data"])
        norm_stats = NormalizationStats.from_trials(train_data)

    window_scales = config.window_scales
    batch_size = config.batch_size

    if is_visual:
        # Build trial info for VisualTemporalDataset
        from btpn.dataset import _build_trial_info

        trials_info = _build_trial_info(eval_dirs, trial_data, trial_meta)
        visual_scales = config.visual_window_scales

        dataset = VisualTemporalDataset(
            trials=trials_info,
            scales=window_scales,
            visual_scales=visual_scales,
            norm_stats=norm_stats,
            sample_interval=config.sample_interval,
            seg_features_dir=paths_config.get("seg_features_dir", "YOLO_FEATURES"),
            depth_embeddings_dir=paths_config.get("depth_embeddings_dir", "DEPTH/embeddings"),
            pose_features_dir=paths_config.get("pose_features_dir", "POSE_FEATURES"),
            use_pose_features=config.use_pose_features,
            augment=False,
        )
        collate_fn = collate_visual_temporal
    else:
        dataset = KinematicDataset(
            trial_data_list=trial_data,
            trial_metadata_list=trial_meta,
            window_scales=window_scales,
            normalize=True,
            norm_stats=norm_stats,
            augment=False,
        )
        collate_fn = collate_multiscale

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    print(f"  Dataset samples: {len(dataset)}, batches: {len(loader)}")

    return loader, norm_stats, trial_meta


# ============================================================================
# Model Loading
# ============================================================================


def _detect_stage(checkpoint_path: Path) -> str:
    """Auto-detect model stage from checkpoint contents.

    Foundation checkpoints contain a 'config' dict without visual keys.
    BTPN checkpoints contain visual encoder weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        "foundation" or "btpn".
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", {})

    # If any key starts with visual-specific prefixes, it is a full BTPN
    visual_prefixes = ("seg_neck_proj.", "clinical_encoder.", "confidence_gate.")
    for key in state_dict:
        if any(key.startswith(p) for p in visual_prefixes):
            return "btpn"
    return "foundation"


def _load_model(
    checkpoint_path: Path,
    config: BTPNConfig,
    stage: str,
    device: torch.device,
) -> tuple[KinematicFoundationModel | BTPN, NormalizationStats | None]:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: BTPN configuration.
        stage: "foundation" or "btpn".
        device: Torch device.

    Returns:
        Tuple of (model, norm_stats_from_checkpoint_or_None).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if stage == "foundation":
        model = KinematicFoundationModel(config).to(device)
    else:
        # Build full BTPN with kinematic prior
        kin_ckpt_path = Path(config.kinematic_checkpoint)
        if not kin_ckpt_path.is_absolute():
            kin_ckpt_path = REPO_ROOT / kin_ckpt_path

        kinematic_model = KinematicFoundationModel(config).to(device)
        if kin_ckpt_path.exists():
            kin_ckpt = torch.load(kin_ckpt_path, map_location=device, weights_only=False)
            kinematic_model.load_state_dict(kin_ckpt["model_state_dict"])
            print(f"  Kinematic prior loaded (epoch {kin_ckpt.get('epoch', '?')})")
        else:
            print(f"  WARNING: Kinematic prior not found at {kin_ckpt_path}")
        kinematic_model.eval()

        model = BTPN(config, kinematic_model=kinematic_model).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    best_loss = ckpt.get("best_val_loss", "?")
    print(f"  Model loaded from epoch {epoch} (best loss: {best_loss})")

    # Extract norm stats from checkpoint if available
    norm_stats = None
    if "norm_mean" in ckpt and "norm_std" in ckpt:
        norm_stats = NormalizationStats(
            mean=np.array(ckpt["norm_mean"], dtype=np.float32),
            std=np.array(ckpt["norm_std"], dtype=np.float32),
        )
    elif "norm_stats" in ckpt:
        ns = ckpt["norm_stats"]
        if isinstance(ns, NormalizationStats):
            norm_stats = ns
        elif isinstance(ns, dict):
            norm_stats = NormalizationStats(
                mean=np.array(ns["mean"], dtype=np.float32),
                std=np.array(ns["std"], dtype=np.float32),
            )

    model.eval()
    param_info = count_parameters(model)
    print(f"  Parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")

    return model, norm_stats


# ============================================================================
# Inference with MC Dropout
# ============================================================================


@torch.no_grad()
def _collect_predictions_foundation(
    model: KinematicFoundationModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    mc_samples: int = 20,
) -> dict[str, np.ndarray]:
    """Run MC Dropout inference for the kinematic foundation model.

    Args:
        model: Foundation model in eval mode.
        data_loader: Evaluation data loader.
        device: Torch device.
        mc_samples: Number of MC forward passes.

    Returns:
        Dictionary of aggregated predictions as numpy arrays.
    """
    collectors: dict[str, list[np.ndarray]] = {
        "mu_position": [],
        "sigma_position": [],
        "mu_quaternion": [],
        "kappa_quaternion": [],
        "mu_angle": [],
        "sigma_angle": [],
        "target_position": [],
        "target_quaternion": [],
        "target_angle": [],
        "trial_idx": [],
    }

    model.eval()

    for batch in tqdm(data_loader, desc="Inference", leave=False):
        kin_windows = [w.to(device) for w in batch["scales"]]
        target = batch["target"].cpu().numpy()

        # MC Dropout: T forward passes
        mc_outputs: list[dict[str, torch.Tensor]] = []
        with enable_mc_dropout(model):
            for _ in range(mc_samples):
                out = model(multi_scale_inputs=kin_windows, force_diagonal=True)
                mc_outputs.append(out)

        # Average predictions across MC samples
        mu_pos = torch.stack([o["mu_position"] for o in mc_outputs]).mean(0)
        sigma_pos_aleatoric = torch.stack([o["sigma_position"] for o in mc_outputs]).mean(0)
        mu_pos_epistemic = torch.stack([o["mu_position"] for o in mc_outputs]).std(0)
        # Total sigma: sqrt(aleatoric^2 + epistemic^2)
        sigma_pos = torch.sqrt(sigma_pos_aleatoric ** 2 + mu_pos_epistemic ** 2)

        mu_quat = torch.stack([o["mu_quaternion"] for o in mc_outputs]).mean(0)
        kappa_quat = torch.stack([o["kappa_quaternion"] for o in mc_outputs]).mean(0)
        mu_angle = torch.stack([o["mu_angle"] for o in mc_outputs]).mean(0)
        sigma_angle = torch.stack([o["sigma_angle"] for o in mc_outputs]).mean(0)

        collectors["mu_position"].append(mu_pos.float().cpu().numpy())
        collectors["sigma_position"].append(sigma_pos.float().cpu().numpy())
        collectors["mu_quaternion"].append(mu_quat.float().cpu().numpy())
        collectors["kappa_quaternion"].append(kappa_quat.float().cpu().numpy())
        collectors["mu_angle"].append(mu_angle.float().cpu().numpy())
        collectors["sigma_angle"].append(sigma_angle.float().cpu().numpy())

        # Targets: Tool 1 [0:8], Tool 2 [8:16]
        collectors["target_position"].append(
            np.stack([target[:, 0:3], target[:, 8:11]], axis=1)
        )
        collectors["target_quaternion"].append(
            np.stack([target[:, 3:7], target[:, 11:15]], axis=1)
        )
        collectors["target_angle"].append(
            np.stack([target[:, 7:8], target[:, 15:16]], axis=1)
        )
        if "trial_idx" in batch:
            collectors["trial_idx"].append(batch["trial_idx"].numpy())

    return {k: np.concatenate(v, axis=0) for k, v in collectors.items() if v}


@torch.no_grad()
def _collect_predictions_btpn(
    model: BTPN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    mc_samples: int = 20,
) -> dict[str, np.ndarray]:
    """Run MC Dropout inference for the full BTPN model.

    Args:
        model: Full BTPN model in eval mode.
        data_loader: Evaluation data loader.
        device: Torch device.
        mc_samples: Number of MC forward passes.

    Returns:
        Dictionary of aggregated predictions as numpy arrays.
    """
    collectors: dict[str, list[np.ndarray]] = {
        "mu_position": [],
        "sigma_position": [],
        "mu_quaternion": [],
        "kappa_quaternion": [],
        "mu_angle": [],
        "sigma_angle": [],
        "kin_mu_position": [],
        "kin_mu_quaternion": [],
        "kin_mu_angle": [],
        "target_position": [],
        "target_quaternion": [],
        "target_angle": [],
        "trial_idx": [],
    }

    model.eval()

    for batch in tqdm(data_loader, desc="Inference", leave=False):
        kin_windows = [w.to(device) for w in batch["kinematic_windows"]]
        seg_neck = [w.to(device) for w in batch["seg_neck_windows"]]
        seg_backbone = [w.to(device) for w in batch["seg_backbone_windows"]]
        depth = [w.to(device) for w in batch["depth_windows"]]
        det_conf = [w.to(device) for w in batch["detection_conf"]]
        target = batch["target"].cpu().numpy()

        pose_kp = pose_bb = pose_geo = pose_conf = None
        if "pose_kp_windows" in batch:
            pose_kp = [w.to(device) for w in batch["pose_kp_windows"]]
            pose_bb = [w.to(device) for w in batch["pose_backbone_windows"]]
            pose_geo = [w.to(device) for w in batch["pose_geometric_windows"]]
            pose_conf = [w.to(device) for w in batch["pose_conf"]]

        current_pos = batch.get("current_position")
        if current_pos is not None:
            current_pos = current_pos.to(device)

        # MC Dropout: T forward passes
        mc_outputs: list[dict[str, torch.Tensor]] = []
        with enable_mc_dropout(model):
            for _ in range(mc_samples):
                out = model(
                    kinematic_windows=kin_windows,
                    seg_neck_windows=seg_neck,
                    seg_backbone_windows=seg_backbone,
                    depth_windows=depth,
                    detection_conf=det_conf,
                    pose_kp_windows=pose_kp,
                    pose_backbone_windows=pose_bb,
                    pose_geometric_windows=pose_geo,
                    pose_conf=pose_conf,
                    current_position=current_pos,
                )
                mc_outputs.append(out)

        # Average predictions across MC samples
        mu_pos = torch.stack([o["mu_position"] for o in mc_outputs]).mean(0)
        sigma_pos_aleatoric = torch.stack([o["sigma_position"] for o in mc_outputs]).mean(0)
        mu_pos_epistemic = torch.stack([o["mu_position"] for o in mc_outputs]).std(0)
        sigma_pos = torch.sqrt(sigma_pos_aleatoric ** 2 + mu_pos_epistemic ** 2)

        mu_quat = torch.stack([o["mu_quaternion"] for o in mc_outputs]).mean(0)
        kappa_quat = torch.stack([o["kappa_quaternion"] for o in mc_outputs]).mean(0)
        mu_angle = torch.stack([o["mu_angle"] for o in mc_outputs]).mean(0)
        sigma_angle = torch.stack([o["sigma_angle"] for o in mc_outputs]).mean(0)

        collectors["mu_position"].append(mu_pos.float().cpu().numpy())
        collectors["sigma_position"].append(sigma_pos.float().cpu().numpy())
        collectors["mu_quaternion"].append(mu_quat.float().cpu().numpy())
        collectors["kappa_quaternion"].append(kappa_quat.float().cpu().numpy())
        collectors["mu_angle"].append(mu_angle.float().cpu().numpy())
        collectors["sigma_angle"].append(sigma_angle.float().cpu().numpy())

        # Kinematic prior predictions (from first MC sample, deterministic)
        collectors["kin_mu_position"].append(
            mc_outputs[0]["kin_mu_position"].float().cpu().numpy()
        )
        collectors["kin_mu_quaternion"].append(
            mc_outputs[0]["kin_mu_quaternion"].float().cpu().numpy()
        )
        collectors["kin_mu_angle"].append(
            mc_outputs[0]["kin_mu_angle"].float().cpu().numpy()
        )

        # Targets
        collectors["target_position"].append(
            np.stack([target[:, 0:3], target[:, 8:11]], axis=1)
        )
        collectors["target_quaternion"].append(
            np.stack([target[:, 3:7], target[:, 11:15]], axis=1)
        )
        collectors["target_angle"].append(
            np.stack([target[:, 7:8], target[:, 15:16]], axis=1)
        )
        if "trial_idx" in batch:
            collectors["trial_idx"].append(batch["trial_idx"].numpy())

    return {k: np.concatenate(v, axis=0) for k, v in collectors.items() if v}


# ============================================================================
# Metrics Computation
# ============================================================================


def _denormalize_positions(
    pred: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    norm_stats: NormalizationStats,
    tool_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denormalize predicted and target positions back to millimetres.

    Args:
        pred: (N, 3) predicted positions in normalized space.
        target: (N, 3) target positions in normalized space.
        sigma: (N, 3) predicted sigma in normalized space.
        norm_stats: Normalization statistics.
        tool_idx: 0 for Tool 1, 1 for Tool 2.

    Returns:
        Tuple of (pred_mm, target_mm, sigma_mm).
    """
    # Tool 1: indices 0:3, Tool 2: indices 8:11
    offset = tool_idx * 8
    mean = norm_stats.mean[offset:offset + 3]
    std = norm_stats.std[offset:offset + 3]
    return pred * std + mean, target * std + mean, sigma * std


def compute_per_tool_metrics(
    pred_data: dict[str, np.ndarray],
    norm_stats: NormalizationStats,
    dataset_key: str = "A",
) -> dict[str, Any]:
    """Compute all paper metrics from collected predictions.

    Produces per-tool and averaged metrics for position, rotation, jaw
    angle, and uncertainty calibration.

    Args:
        pred_data: Dictionary of numpy arrays from collection functions.
        norm_stats: Normalization statistics for denormalization.
        dataset_key: "A", "B", or "C" (affects jaw metric reporting).

    Returns:
        Nested dictionary of all metrics.
    """
    metrics: dict[str, Any] = {}
    is_6dof = dataset_key == "C"

    # ---- Position metrics per tool ----
    tool_pos_errors: list[np.ndarray] = []
    tool_pos_metrics: list[dict[str, Any]] = []
    tool_sigma_mm: list[np.ndarray] = []

    for t in range(2):
        pred_mm, tgt_mm, sigma_mm = _denormalize_positions(
            pred_data["mu_position"][:, t, :],
            pred_data["target_position"][:, t, :],
            pred_data["sigma_position"][:, t, :],
            norm_stats,
            tool_idx=t,
        )
        pos_m = compute_position_metrics(pred_mm, tgt_mm)
        tool_pos_metrics.append(pos_m)

        err = np.linalg.norm(pred_mm - tgt_mm, axis=-1)
        tool_pos_errors.append(err)
        tool_sigma_mm.append(np.linalg.norm(sigma_mm, axis=-1))

    # Averaged position metrics
    all_err = np.concatenate(tool_pos_errors)
    metrics["position"] = {
        "mae_mm": float(np.mean([m["mae_mm"] for m in tool_pos_metrics])),
        "rmse_mm": float(np.sqrt(np.mean(all_err ** 2))),
        "std_mm": float(all_err.std()),
        "p50_mm": float(np.percentile(all_err, 50)),
        "p75_mm": float(np.percentile(all_err, 75)),
        "p90_mm": float(np.percentile(all_err, 90)),
        "p95_mm": float(np.percentile(all_err, 95)),
        "p99_mm": float(np.percentile(all_err, 99)),
        "per_tool": {
            "tool1": tool_pos_metrics[0],
            "tool2": tool_pos_metrics[1],
        },
    }

    # Averaged per-axis
    per_axis: dict[str, dict[str, float]] = {}
    for ax_name in ["X", "Y", "Z"]:
        ax_vals = []
        for m in tool_pos_metrics:
            ax_vals.append(m["per_axis"][ax_name]["rmse_mm"])
        per_axis[ax_name] = {
            "rmse_mm": float(np.mean(ax_vals)),
        }
        # Add individual tool per-axis
        per_axis[ax_name]["tool1_rmse_mm"] = tool_pos_metrics[0]["per_axis"][ax_name]["rmse_mm"]
        per_axis[ax_name]["tool2_rmse_mm"] = tool_pos_metrics[1]["per_axis"][ax_name]["rmse_mm"]
    metrics["position"]["per_axis"] = per_axis

    # ---- Rotation metrics per tool ----
    tool_rot_metrics: list[dict[str, float]] = []
    tool_rot_errors: list[np.ndarray] = []

    for t in range(2):
        rot_m = compute_geodesic_error(
            pred_data["mu_quaternion"][:, t, :],
            pred_data["target_quaternion"][:, t, :],
        )
        tool_rot_metrics.append(rot_m)

        # Raw errors for AUSE
        from btpn.metrics import _normalize_quaternions
        q1 = _normalize_quaternions(pred_data["mu_quaternion"][:, t, :])
        q2 = _normalize_quaternions(pred_data["target_quaternion"][:, t, :])
        dot = np.abs(np.sum(q1 * q2, axis=-1))
        dot = np.clip(dot, 0.0, 1.0)
        geo_err = 2.0 * np.arccos(dot) * 180.0 / np.pi
        tool_rot_errors.append(geo_err)

    all_rot = np.concatenate(tool_rot_errors)
    metrics["rotation"] = {
        "mean_deg": float(np.mean([m["mean_deg"] for m in tool_rot_metrics])),
        "rmse_deg": float(np.sqrt(np.mean(all_rot ** 2))),
        "std_deg": float(all_rot.std()),
        "median_deg": float(np.median(all_rot)),
        "p50_deg": float(np.percentile(all_rot, 50)),
        "p75_deg": float(np.percentile(all_rot, 75)),
        "p90_deg": float(np.percentile(all_rot, 90)),
        "p95_deg": float(np.percentile(all_rot, 95)),
        "p99_deg": float(np.percentile(all_rot, 99)),
        "per_tool": {
            "tool1": tool_rot_metrics[0],
            "tool2": tool_rot_metrics[1],
        },
    }

    # Euler angle decomposition
    euler_metrics: dict[str, dict[str, float]] = {}
    for t in range(2):
        euler_m = compute_euler_errors(
            pred_data["mu_quaternion"][:, t, :],
            pred_data["target_quaternion"][:, t, :],
        )
        for axis_name, axis_m in euler_m.items():
            if axis_name not in euler_metrics:
                euler_metrics[axis_name] = {}
            for k, v in axis_m.items():
                key = f"tool{t + 1}_{k}"
                euler_metrics[axis_name][key] = v
    # Average across tools
    for axis_name in ["roll", "pitch", "yaw"]:
        for metric_key in ["mae_deg", "rmse_deg"]:
            t1 = euler_metrics[axis_name][f"tool1_{metric_key}"]
            t2 = euler_metrics[axis_name][f"tool2_{metric_key}"]
            euler_metrics[axis_name][metric_key] = float(np.mean([t1, t2]))
    metrics["euler"] = euler_metrics

    # ---- Jaw angle metrics ----
    if not is_6dof and "mu_angle" in pred_data and "target_angle" in pred_data:
        jaw_all_pred = pred_data["mu_angle"].ravel()
        jaw_all_tgt = pred_data["target_angle"].ravel()
        jaw_m = compute_jaw_metrics(jaw_all_pred, jaw_all_tgt)
        metrics["jaw_angle"] = jaw_m

        # Per-tool
        for t in range(2):
            j_m = compute_jaw_metrics(
                pred_data["mu_angle"][:, t, :],
                pred_data["target_angle"][:, t, :],
            )
            metrics["jaw_angle"][f"tool{t + 1}_mae"] = j_m["mae"]
            metrics["jaw_angle"][f"tool{t + 1}_rmse"] = j_m["rmse"]

    # ---- Uncertainty calibration ----
    sigma_pos_all = np.concatenate(tool_sigma_mm)
    pos_errors_all = np.concatenate(tool_pos_errors)

    ece_result = compute_ece(pos_errors_all, sigma_pos_all, n_bins=10)
    ause_result = compute_ause(pos_errors_all, sigma_pos_all, n_steps=20)
    coverage_result = compute_coverage(pos_errors_all, sigma_pos_all)

    # Rotation uncertainty via kappa inverse
    kappa_all = np.concatenate([
        pred_data["kappa_quaternion"][:, 0, 0],
        pred_data["kappa_quaternion"][:, 1, 0],
    ])
    rot_sigma_proxy = 1.0 / (kappa_all + 1e-8)
    rot_ause = compute_ause(all_rot, rot_sigma_proxy, n_steps=20)

    metrics["uncertainty"] = {
        "ece": ece_result["ece"],
        "ause": ause_result["ause"],
        "ause_normalized": ause_result["ause_normalized"],
        "ause_rot_normalized": rot_ause["ause_normalized"],
        "mean_sigma_pos_mm": float(sigma_pos_all.mean()),
        "mean_kappa": float(kappa_all.mean()),
        "global_coverages": ece_result["global_coverages"],
        "coverage": coverage_result,
    }

    # ---- Store raw error arrays (not serialized to JSON) ----
    metrics["_raw"] = {
        "pos_err_t1": tool_pos_errors[0],
        "pos_err_t2": tool_pos_errors[1],
        "rot_err_t1": tool_rot_errors[0],
        "rot_err_t2": tool_rot_errors[1],
        "sigma_pos_all": sigma_pos_all,
        "pos_errors_all": pos_errors_all,
        "rot_errors_all": all_rot,
        "ause_curves": ause_result,
    }

    return metrics


# ============================================================================
# Per-Trial Breakdown
# ============================================================================


def compute_per_trial_breakdown(
    pred_data: dict[str, np.ndarray],
    norm_stats: NormalizationStats,
    trial_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute metrics broken down by individual trial.

    Args:
        pred_data: Dictionary of numpy arrays from collection.
        norm_stats: Normalization statistics.
        trial_metadata: List of trial metadata dicts.

    Returns:
        List of per-trial metric dictionaries.
    """
    if "trial_idx" not in pred_data:
        return []

    trial_indices = pred_data["trial_idx"]
    unique_trials = np.unique(trial_indices)
    per_trial: list[dict[str, Any]] = []

    for trial_idx in unique_trials:
        mask = trial_indices == trial_idx
        n_samples = int(mask.sum())
        if n_samples < 10:
            continue

        trial_pred = {k: v[mask] for k, v in pred_data.items() if k != "trial_idx"}
        trial_metrics = compute_per_tool_metrics(trial_pred, norm_stats)

        # Strip raw arrays
        trial_metrics.pop("_raw", None)

        meta = trial_metadata[int(trial_idx)] if int(trial_idx) < len(trial_metadata) else {}
        per_trial.append({
            "trial_idx": int(trial_idx),
            "trial_name": meta.get("trial_name", f"Trial_{trial_idx}"),
            "n_samples": n_samples,
            "skill_category": meta.get("skill_category", "unknown"),
            "metrics": trial_metrics,
        })

    return per_trial


# ============================================================================
# LaTeX Table Generation
# ============================================================================


def _build_flat_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten nested metrics dict into the format expected by format_latex_row.

    Args:
        metrics: Nested metrics from compute_per_tool_metrics.

    Returns:
        Flat dictionary with keys like pos_rmse_x_mm, rot_geodesic_deg, etc.
    """
    flat: dict[str, float] = {}

    # Per-axis position RMSE
    for axis in ["X", "Y", "Z"]:
        key = f"pos_rmse_{axis.lower()}_mm"
        flat[key] = metrics["position"]["per_axis"][axis]["rmse_mm"]

    flat["pos_rmse_all_mm"] = metrics["position"]["rmse_mm"]

    # Per-axis rotation RMSE
    for axis in ["roll", "pitch", "yaw"]:
        key = f"rot_rmse_{axis}_deg"
        flat[key] = metrics.get("euler", {}).get(axis, {}).get("rmse_deg", 0.0)

    flat["rot_geodesic_deg"] = metrics["rotation"]["rmse_deg"]

    # Jaw angle
    flat["jaw_rmse_rad"] = metrics.get("jaw_angle", {}).get("rmse", 0.0)

    # ECE
    flat["ece"] = metrics.get("uncertainty", {}).get("ece", 0.0)

    return flat


def generate_latex_table_2b(
    results_by_dataset: dict[str, dict[str, Any]],
    model_label: str = "BTPN",
) -> str:
    """Generate LaTeX Table 2(b): per-axis RMSE across datasets.

    Matches the paper Table 2(b) format with per-axis position RMSE,
    per-axis rotation RMSE, geodesic RMSE, jaw RMSE, and ECE.

    Args:
        results_by_dataset: Mapping from dataset key to metrics dict.
        model_label: Display name for the model row.

    Returns:
        LaTeX table string.
    """
    lines: list[str] = [
        "% Table 2(b): Cross-dataset evaluation",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-dataset pose prediction evaluation.}",
        "\\label{tab:cross-dataset}",
        "\\small",
        "\\begin{tabular}{ll rrr r rrr r r r}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{Position RMSE (mm)} & \\multicolumn{4}{c}{Rotation RMSE ($^\\circ$)} & Jaw & ECE \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        "Dataset & Method & $x$ & $y$ & $z$ & All & Roll & Pitch & Yaw & Geo. & RMSE & \\\\",
        "\\midrule",
    ]

    dataset_labels = {"A": "A (7DOF)", "B": "B (BAPES)", "C": "C (6DOF)"}

    for ds_key in ["A", "B", "C"]:
        if ds_key not in results_by_dataset:
            continue

        m = results_by_dataset[ds_key]
        flat = _build_flat_metrics(m)
        jaw_na = ds_key == "C"  # 6DOF has no jaw angle

        label = f"{dataset_labels.get(ds_key, ds_key)} & {model_label}"
        row = format_latex_row(label, flat, bold=False, jaw_na=jaw_na, include_ece=True)
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_table_2c(
    results_by_dataset: dict[str, dict[str, Any]],
    kin_results: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Generate LaTeX Table 2(c): ablation / comparison table.

    Compares kinematic-only baseline vs full BTPN across datasets.

    Args:
        results_by_dataset: Full BTPN metrics per dataset.
        kin_results: Kinematic-only metrics per dataset (if available).

    Returns:
        LaTeX table string.
    """
    lines: list[str] = [
        "% Table 2(c): Model comparison",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Kinematic-only vs.\\ full BTPN.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\begin{tabular}{ll rrr r rrr r r r}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{Position RMSE (mm)} & \\multicolumn{4}{c}{Rotation RMSE ($^\\circ$)} & Jaw & ECE \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        "Dataset & Method & $x$ & $y$ & $z$ & All & Roll & Pitch & Yaw & Geo. & RMSE & \\\\",
        "\\midrule",
    ]

    dataset_labels = {"A": "A (7DOF)", "B": "B (BAPES)", "C": "C (6DOF)"}

    for ds_key in ["A", "B", "C"]:
        ds_label = dataset_labels.get(ds_key, ds_key)
        jaw_na = ds_key == "C"

        # Kinematic-only row
        if kin_results and ds_key in kin_results:
            flat_kin = _build_flat_metrics(kin_results[ds_key])
            row = format_latex_row(
                f"{ds_label} & Kinematic only", flat_kin, jaw_na=jaw_na,
            )
            lines.append(row)

        # Full BTPN row
        if ds_key in results_by_dataset:
            flat_btpn = _build_flat_metrics(results_by_dataset[ds_key])
            row = format_latex_row(
                f"{ds_label} & \\textbf{{Full BTPN}}",
                flat_btpn,
                bold=True,
                jaw_na=jaw_na,
            )
            lines.append(row)

        # Add midrule between datasets
        if ds_key != "C":
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ============================================================================
# Results Summary Printing
# ============================================================================


def print_results_summary(
    metrics: dict[str, Any],
    dataset_key: str,
    stage: str,
) -> None:
    """Print a formatted summary of evaluation results.

    Args:
        metrics: Metrics dict from compute_per_tool_metrics.
        dataset_key: "A", "B", or "C".
        stage: "foundation" or "btpn".
    """
    ds_name = DATASET_MAP[dataset_key]
    print()
    print("=" * 72)
    print(f"  Dataset {dataset_key} ({ds_name}) -- {stage.upper()}")
    print("=" * 72)

    pos = metrics["position"]
    print(f"\n  Position (mm):")
    print(f"    MAE:     {pos['mae_mm']:.3f}")
    print(f"    RMSE:    {pos['rmse_mm']:.3f}")
    print(f"    P50:     {pos['p50_mm']:.3f}")
    print(f"    P90:     {pos['p90_mm']:.3f}")
    print(f"    P99:     {pos['p99_mm']:.3f}")

    print(f"\n  Per-tool Position MAE (mm):")
    print(f"    Tool 1:  {pos['per_tool']['tool1']['mae_mm']:.3f}")
    print(f"    Tool 2:  {pos['per_tool']['tool2']['mae_mm']:.3f}")

    rot = metrics["rotation"]
    print(f"\n  Rotation (deg):")
    print(f"    Mean:    {rot['mean_deg']:.2f}")
    print(f"    RMSE:    {rot['rmse_deg']:.2f}")
    print(f"    Median:  {rot['median_deg']:.2f}")
    print(f"    P90:     {rot['p90_deg']:.2f}")

    print(f"\n  Per-tool Rotation Mean (deg):")
    print(f"    Tool 1:  {rot['per_tool']['tool1']['mean_deg']:.2f}")
    print(f"    Tool 2:  {rot['per_tool']['tool2']['mean_deg']:.2f}")

    if "euler" in metrics:
        print(f"\n  Euler Decomposition RMSE (deg):")
        for axis in ["roll", "pitch", "yaw"]:
            rmse = metrics["euler"].get(axis, {}).get("rmse_deg", 0.0)
            print(f"    {axis.capitalize():<8} {rmse:.2f}")

    if "jaw_angle" in metrics:
        jaw = metrics["jaw_angle"]
        print(f"\n  Jaw Angle:")
        print(f"    MAE:     {jaw['mae']:.4f}")
        print(f"    RMSE:    {jaw['rmse']:.4f}")

    unc = metrics.get("uncertainty", {})
    if unc:
        print(f"\n  Uncertainty Calibration:")
        print(f"    ECE:           {unc.get('ece', 0):.4f}")
        print(f"    AUSE (norm):   {unc.get('ause_normalized', 0):.4f}")
        print(f"    Mean sigma:    {unc.get('mean_sigma_pos_mm', 0):.4f} mm")
        print(f"    Mean kappa:    {unc.get('mean_kappa', 0):.2f}")
        cov = unc.get("global_coverages", {})
        if cov:
            print(f"    Coverages:")
            for level, obs in cov.items():
                print(f"      {float(level)*100:.1f}% expected -> {obs*100:.1f}% observed")

    print()


# ============================================================================
# Main Evaluation
# ============================================================================


def evaluate(
    checkpoint_path: Path,
    config: BTPNConfig,
    stage: str,
    dataset_keys: list[str],
    paths_config: dict[str, Any],
    output_dir: Path,
    mc_samples: int = 20,
    output_tables: bool = False,
    batch_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: BTPN configuration.
        stage: "foundation" or "btpn".
        dataset_keys: List of dataset keys to evaluate ("A", "B", "C").
        paths_config: Paths configuration dict.
        output_dir: Directory for saving results.
        mc_samples: Number of MC Dropout samples.
        output_tables: Whether to generate LaTeX tables.
        batch_size: Override batch size.

    Returns:
        Dictionary mapping dataset key to metrics dict.
    """
    t_start = time.time()

    if batch_size is not None:
        config.batch_size = batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_visual = stage == "btpn"

    print("=" * 72)
    print("  BTPN Evaluation")
    print("=" * 72)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Stage:       {stage}")
    print(f"  Datasets:    {', '.join(dataset_keys)}")
    print(f"  MC samples:  {mc_samples}")
    print(f"  Device:      {device}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load model
    print("Loading model...")
    model, ckpt_norm_stats = _load_model(checkpoint_path, config, stage, device)
    print()

    # Try to load norm stats from checkpoint path companion file
    norm_stats = ckpt_norm_stats
    if norm_stats is None:
        norm_stats_path = Path(config.kinematic_norm_stats)
        if not norm_stats_path.is_absolute():
            norm_stats_path = REPO_ROOT / norm_stats_path
        if norm_stats_path.exists():
            norm_stats = NormalizationStats.load(norm_stats_path)
            print(f"  Loaded norm stats from {norm_stats_path}")

    all_results: dict[str, dict[str, Any]] = {}
    all_pred_data: dict[str, dict[str, np.ndarray]] = {}

    for ds_key in dataset_keys:
        print("-" * 72)
        print(f"Evaluating Dataset {ds_key} ({DATASET_MAP[ds_key]})...")
        print("-" * 72)

        loader, norm_stats, trial_meta = _load_dataset(
            ds_key, paths_config, config,
            norm_stats=norm_stats,
            is_visual=is_visual,
        )
        print()

        # Collect predictions with MC Dropout
        print(f"Running inference (T={mc_samples} MC samples)...")
        t_infer = time.time()
        if stage == "foundation":
            pred_data = _collect_predictions_foundation(
                model, loader, device, mc_samples=mc_samples,
            )
        else:
            pred_data = _collect_predictions_btpn(
                model, loader, device, mc_samples=mc_samples,
            )
        n_samples = pred_data["mu_position"].shape[0]
        print(f"  {n_samples:,} samples in {time.time() - t_infer:.1f}s")
        print()

        # Compute metrics
        print("Computing metrics...")
        metrics = compute_per_tool_metrics(pred_data, norm_stats, dataset_key=ds_key)

        # Per-trial breakdown
        per_trial = compute_per_trial_breakdown(pred_data, norm_stats, trial_meta)
        metrics["per_trial"] = per_trial

        # Print summary
        print_results_summary(metrics, ds_key, stage)

        all_results[ds_key] = metrics
        all_pred_data[ds_key] = pred_data

    # ================================================================
    # Save Results
    # ================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    def _strip_raw(d: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable _raw arrays for JSON export."""
        result = {}
        for k, v in d.items():
            if k == "_raw":
                continue
            if isinstance(v, dict):
                result[k] = _strip_raw(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                result[k] = float(v)
            else:
                result[k] = v
        return result

    # JSON results
    json_results: dict[str, Any] = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "stage": stage,
            "mc_samples": mc_samples,
            "device": str(device),
            "evaluation_time_s": time.time() - t_start,
        },
    }
    for ds_key, m in all_results.items():
        json_results[f"dataset_{ds_key}"] = _strip_raw(m)

    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {json_path}")

    # NPZ evaluation data (for figure generation)
    for ds_key, pred_data in all_pred_data.items():
        npz_path = output_dir / f"evaluation_data_{ds_key}.npz"
        save_arrays: dict[str, np.ndarray] = {}
        for k, v in pred_data.items():
            if isinstance(v, np.ndarray):
                save_arrays[k] = v
        # Add denormalized errors
        raw = all_results[ds_key].get("_raw", {})
        for k, v in raw.items():
            if isinstance(v, np.ndarray):
                save_arrays[f"error_{k}"] = v
        np.savez(npz_path, **save_arrays)
        print(f"Evaluation data saved to {npz_path}")

    # Plain text table
    if len(all_results) > 0:
        # Build results dict for format_results_table
        table_results: dict[str, dict[str, Any]] = {}
        for ds_key, m in all_results.items():
            table_results[f"Dataset {ds_key}"] = m
        table_str = format_results_table(table_results, title="BTPN Evaluation")
        print()
        print(table_str)

        table_path = output_dir / "results_table.txt"
        with open(table_path, "w") as f:
            f.write(table_str)
        print(f"\nTable saved to {table_path}")

    # LaTeX tables
    if output_tables:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)

        # Table 2(b)
        table_2b = generate_latex_table_2b(all_results, model_label="BTPN")
        table_2b_path = results_dir / "table_2b.tex"
        with open(table_2b_path, "w") as f:
            f.write(table_2b)
        print(f"Table 2(b) saved to {table_2b_path}")

        # Table 2(c) -- requires kinematic baseline
        # If we have a BTPN model with kinematic prior, extract its metrics too
        kin_results: dict[str, dict[str, Any]] | None = None
        if stage == "btpn" and "kin_mu_position" in all_pred_data.get("A", {}):
            kin_results = {}
            for ds_key in dataset_keys:
                pred = all_pred_data.get(ds_key, {})
                if "kin_mu_position" not in pred:
                    continue
                # Build kinematic-only prediction dict
                kin_pred: dict[str, np.ndarray] = {
                    "mu_position": pred["kin_mu_position"],
                    "sigma_position": pred["sigma_position"],  # reuse BTPN sigma as proxy
                    "mu_quaternion": pred["kin_mu_quaternion"],
                    "kappa_quaternion": pred["kappa_quaternion"],
                    "mu_angle": pred["kin_mu_angle"],
                    "sigma_angle": pred["sigma_angle"],
                    "target_position": pred["target_position"],
                    "target_quaternion": pred["target_quaternion"],
                    "target_angle": pred["target_angle"],
                }
                kin_m = compute_per_tool_metrics(kin_pred, norm_stats, dataset_key=ds_key)
                kin_results[ds_key] = kin_m

        table_2c = generate_latex_table_2c(all_results, kin_results=kin_results)
        table_2c_path = results_dir / "table_2c.tex"
        with open(table_2c_path, "w") as f:
            f.write(table_2c)
        print(f"Table 2(c) saved to {table_2c_path}")

    # Summary
    total_time = time.time() - t_start
    print()
    print("=" * 72)
    print(f"  Evaluation complete in {total_time:.1f}s")
    print("=" * 72)

    return all_results


# ============================================================================
# Offline Reproduction from Saved Predictions (.npz)
# ============================================================================


# Reference numbers from the accepted MICCAI 2026 paper, Table 2(b),
# "Full BTPN" row (Dataset A held-out). Used only for side-by-side display.
_PAPER_FULL_BTPN: dict[str, float] = {
    "pos_x": 4.1, "pos_y": 4.4, "pos_z": 3.5, "pos_v": 6.8,
    "roll": 14.7, "pitch": 7.3, "yaw": 15.8, "geo": 11.9,
    "jaw_deg": 1.72, "ece": 0.013,
}


def evaluate_from_npz(
    npz_path: Path,
    norm_path: Path,
    output_dir: Path,
) -> dict[str, float]:
    """Recompute Dataset-A Full-BTPN metrics from a saved predictions .npz.

    This is the **offline, CPU-only** reproduction path. It needs neither a
    model checkpoint nor the full dataset: it reads stored per-frame
    predictions (``v3_*`` = Full BTPN, ``kin_*`` = in-model kinematic prior
    branch) and ground-truth ``target_*`` arrays, denormalizes them with the
    z-score stats, and evaluates the same leaf metric functions used by the
    online path.

    IMPORTANT: the ``target_*`` arrays in the .npz are stored in *normalized*
    space (quaternion norms are not 1). They are denormalized here, up front,
    via ``norm_path`` (mean/std). ``v3_mu_position`` is also denormalized to
    millimetres; ``v3_mu_quaternion`` is already unit-norm. The leaf functions
    (:func:`compute_geodesic_error`, :func:`compute_euler_errors`,
    :func:`compute_ece`, ...) are called on these denormalized arrays directly
    — the .npz is *not* routed through :func:`compute_per_tool_metrics`, which
    only denormalizes positions.

    Args:
        npz_path: Path to predictions .npz (e.g. results/evaluation_data.npz).
        norm_path: Path to normalization stats .npz (mean/std, 30-D).
        output_dir: Directory to write the reproduced table + JSON.

    Returns:
        Flat dict of the reproduced Full-BTPN metrics.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Predictions npz not found: {npz_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

    print("=" * 72)
    print("  BTPN Offline Reproduction (from saved predictions .npz)")
    print("=" * 72)
    print(f"  Predictions:  {npz_path}")
    print(f"  Norm stats:   {norm_path}")
    print(f"  Device:       cpu (no model, no full dataset required)")
    print()

    data = dict(np.load(npz_path, allow_pickle=True))
    ns = np.load(norm_path, allow_pickle=True)
    mean = np.asarray(ns["mean"], dtype=np.float64)
    std = np.asarray(ns["std"], dtype=np.float64)

    # Feature layout (30-D): T1 pos 0:3, T1 quat 3:7, T1 jaw 7,
    #                        T2 pos 8:11, T2 quat 11:15, T2 jaw 15.
    sl = {
        "pos": (slice(0, 3), slice(8, 11)),
        "quat": (slice(3, 7), slice(11, 15)),
        "jaw": (7, 15),
    }
    n = data["v3_mu_position"].shape[0]
    print(f"  Samples: {n:,} (Dataset A held-out, 2 tools)")
    print()

    def _denorm(arr: np.ndarray, idx: slice | int) -> np.ndarray:
        return arr * std[idx] + mean[idx]

    # --- Denormalize targets (stored normalized) ---
    tgt_pos = [_denorm(data["target_position"][:, t], sl["pos"][t]) for t in range(2)]
    tgt_quat = [_denorm(data["target_quaternion"][:, t], sl["quat"][t]) for t in range(2)]
    tgt_jaw = [_denorm(data["target_angle"][:, t, 0], sl["jaw"][t]) for t in range(2)]

    results: dict[str, float] = {}

    def _eval_model(prefix: str) -> dict[str, float]:
        """Compute metrics for predictions stored under *prefix* (v3 / kin)."""
        # Position: denormalize to mm; quaternion already unit-norm.
        pred_pos = [_denorm(data[f"{prefix}_mu_position"][:, t], sl["pos"][t]) for t in range(2)]
        pred_quat = [data[f"{prefix}_mu_quaternion"][:, t] for t in range(2)]

        # Per-axis position RMSE, averaged over the two tools (paper convention).
        out: dict[str, float] = {}
        for ax_i, ax_name in enumerate(["pos_x", "pos_y", "pos_z"]):
            rmses = [
                float(np.sqrt(np.mean((pred_pos[t][:, ax_i] - tgt_pos[t][:, ax_i]) ** 2)))
                for t in range(2)
            ]
            out[ax_name] = float(np.mean(rmses))
        # Overall ||v|| RMSE over both tools.
        pe = np.concatenate([
            np.linalg.norm(pred_pos[t] - tgt_pos[t], axis=-1) for t in range(2)
        ])
        out["pos_v"] = float(np.sqrt(np.mean(pe ** 2)))
        out["pos_v_mean"] = float(pe.mean())

        # Rotation: geodesic RMSE (both tools) + Euler RMSE per axis (avg tools).
        # Concatenate raw per-frame geodesic errors across tools so the overall
        # RMSE is over all samples (not an RMSE-of-RMSEs).
        rot_err = []
        for t in range(2):
            q1 = _normalize_quaternions(pred_quat[t])
            q2 = _normalize_quaternions(tgt_quat[t])
            dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
            rot_err.append(2.0 * np.arccos(dot) * 180.0 / np.pi)
        all_rot = np.concatenate(rot_err)
        out["geo"] = float(np.sqrt(np.mean(all_rot ** 2)))
        for ax_name in ["roll", "pitch", "yaw"]:
            vals = [compute_euler_errors(pred_quat[t], tgt_quat[t])[ax_name]["rmse_deg"]
                    for t in range(2)]
            out[ax_name] = float(np.mean(vals))

        # Jaw angle RMSE (paper reports degrees; stored values are calibrated rad).
        pred_jaw = [_denorm(data[f"{prefix}_mu_angle"][:, t, 0], sl["jaw"][t])
                    for t in range(2)] if f"{prefix}_mu_angle" in data else None
        if pred_jaw is not None:
            jerr = np.concatenate([np.abs(pred_jaw[t] - tgt_jaw[t]) for t in range(2)])
            out["jaw_rmse_rad"] = float(np.sqrt(np.mean(jerr ** 2)))
            out["jaw_deg"] = float(np.degrees(out["jaw_rmse_rad"]))

        # Position calibration (ECE / AUSE): denormalize sigma to mm.
        if f"{prefix}_sigma_position" in data:
            sig = np.concatenate([
                np.linalg.norm(data[f"{prefix}_sigma_position"][:, t] * std[sl["pos"][t]], axis=-1)
                for t in range(2)
            ])
            out["ece"] = float(compute_ece(pe, sig, n_bins=10)["ece"])
            out["ause_norm"] = float(compute_ause(pe, sig, n_steps=20)["ause_normalized"])
        return out

    full = _eval_model("v3")
    results = full

    # --- Print side-by-side comparison against the locked paper ---
    print("  Full BTPN -- Dataset A held-out (reproduced from npz vs paper Table 2)")
    print("  " + "-" * 64)
    print(f"  {'Metric':<18}{'Reproduced':>14}{'Paper':>12}{'Delta':>12}")
    print("  " + "-" * 64)
    rows = [
        ("Pos x (mm)", "pos_x"), ("Pos y (mm)", "pos_y"), ("Pos z (mm)", "pos_z"),
        ("Pos |v| (mm)", "pos_v"), ("Roll (deg)", "roll"), ("Pitch (deg)", "pitch"),
        ("Yaw (deg)", "yaw"), ("Geo (deg)", "geo"), ("Jaw (deg)", "jaw_deg"),
        ("ECE", "ece"),
    ]
    for label, key in rows:
        rep = full.get(key)
        pap = _PAPER_FULL_BTPN.get(key)
        if rep is None or pap is None:
            continue
        prec = 3 if key == "ece" else (2 if key == "jaw_deg" else 1)
        print(f"  {label:<18}{rep:>14.{prec}f}{pap:>12.{prec}f}{rep - pap:>+12.{prec}f}")
    print("  " + "-" * 64)
    print(f"  (Pos |v| mean Euclidean = {full['pos_v_mean']:.2f} mm; the table 'All' "
          f"column is RMSE.)")
    print()

    # --- Write reproduced LaTeX row + JSON (NOT overwriting committed tables) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "table2b_reproduced.tex"
    row = (
        f"\\textbf{{Full BTPN}} & "
        f"{full['pos_x']:.1f} & {full['pos_y']:.1f} & {full['pos_z']:.1f} & {full['pos_v']:.1f} & "
        f"{full['roll']:.1f} & {full['pitch']:.1f} & {full['yaw']:.1f} & {full['geo']:.1f} & "
        f"{full.get('jaw_deg', float('nan')):.2f} & {full['ece']:.3f} \\\\"
    )
    tex_path.write_text(
        "% Reproduced on CPU from results/evaluation_data.npz via\n"
        "%   python scripts/evaluate.py --from-npz results/evaluation_data.npz\n"
        "% Columns: x y z |v| (mm); roll pitch yaw geo (deg); jaw (deg); ECE.\n"
        "% NOTE: not auto-substituted into the locked paper table; see README.\n"
        + row + "\n",
        encoding="utf-8",
    )
    json_path = output_dir / "evaluation_reproduced.json"
    with open(json_path, "w") as f:
        json.dump(
            {"full_btpn_dataset_a": full, "paper_reference": _PAPER_FULL_BTPN},
            f, indent=2,
        )
    print(f"  Reproduced LaTeX row -> {tex_path}")
    print(f"  Reproduced metrics   -> {json_path}")
    print()
    print("=" * 72)
    print("  Offline reproduction complete.")
    print("=" * 72)
    return results


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """Parse command-line arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate BTPN model and reproduce paper tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset A\n"
            "  python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset all --output-tables\n"
            "  python scripts/evaluate.py --checkpoint checkpoints/kinematic_foundation.pt --stage foundation\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file). Required unless --from-npz.",
    )
    parser.add_argument(
        "--from-npz",
        type=str,
        default=None,
        metavar="NPZ",
        help=(
            "Offline reproduction: recompute Dataset A (Full BTPN) pose + "
            "calibration metrics directly from a saved predictions .npz "
            "(e.g. results/evaluation_data.npz). Runs on CPU with NO model "
            "checkpoint and NO full dataset. See --norm-stats."
        ),
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="checkpoints/btpn_norm.npz",
        help=(
            "Normalization stats (.npz with mean/std) used to denormalize "
            "the targets stored in --from-npz. Default: checkpoints/btpn_norm.npz."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file. Defaults to configs/btpn.yaml for "
            "stage=btpn and configs/kinematic_foundation.yaml for stage=foundation."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="A",
        choices=["A", "B", "C", "all"],
        help="Dataset to evaluate: A (7DOF2024), B (BAPES2024), C (6DOF2023), or all.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["foundation", "btpn"],
        help="Model stage. Auto-detected from checkpoint if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation outputs. Default: results/evaluation/.",
    )
    parser.add_argument(
        "--output-tables",
        action="store_true",
        help="Generate LaTeX tables (Table 2b, 2c) in results/ directory.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of MC Dropout forward passes (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Offline reproduction path: recompute metrics from a saved .npz with
    # no checkpoint and no full dataset (CPU-only). This regenerates the
    # Full-BTPN / Dataset-A numbers of the paper from the committed
    # results/evaluation_data.npz. See evaluate_from_npz().
    # ------------------------------------------------------------------
    if args.from_npz is not None:
        npz_path = Path(args.from_npz)
        if not npz_path.is_absolute():
            npz_path = REPO_ROOT / npz_path
        norm_path = Path(args.norm_stats)
        if not norm_path.is_absolute():
            norm_path = REPO_ROOT / norm_path
        out_dir = (
            Path(args.output_dir) if args.output_dir is not None
            else REPO_ROOT / "results"
        )
        evaluate_from_npz(npz_path, norm_path, out_dir)
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required (or use --from-npz for offline reproduction).")

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    # Auto-detect stage
    stage = args.stage
    if stage is None:
        stage = _detect_stage(checkpoint_path)
        print(f"Auto-detected stage: {stage}")

    # Load config
    if args.config is not None:
        config_path = Path(args.config)
    elif stage == "foundation":
        config_path = REPO_ROOT / "configs" / "kinematic_foundation.yaml"
    else:
        config_path = REPO_ROOT / "configs" / "btpn.yaml"

    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    config = BTPNConfig.from_yaml(config_path)
    config.seed = args.seed

    # Load paths config
    paths_config = _load_paths_config()

    # Resolve dataset keys
    if args.dataset == "all":
        dataset_keys = ["A", "B", "C"]
    else:
        dataset_keys = [args.dataset]

    # Output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "results" / "evaluation"

    evaluate(
        checkpoint_path=checkpoint_path,
        config=config,
        stage=stage,
        dataset_keys=dataset_keys,
        paths_config=paths_config,
        output_dir=output_dir,
        mc_samples=args.mc_samples,
        output_tables=args.output_tables,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
