"""BTPN — Bayesian Temporal Pose Networks.

Uncertainty-calibrated laparoscopic tool pose tracking using hierarchical
multi-scale temporal attention with visual-kinematic fusion.

Reference:
    Choudhry et al., "Bayesian Temporal Pose Networks for Uncertainty-Calibrated
    Laparoscopic Tool Pose Tracking", MICCAI 2026.
"""

__version__ = "1.0.0"

from .config import BTPNConfig, BTPNFeatureConfig
from .model import BTPN, KinematicFoundationModel, PivotPointEstimator
from .losses import KinematicLoss, BTPNLoss, geodesic_distance, beta_nll_loss
from .metrics import (
    compute_position_metrics,
    compute_geodesic_error,
    compute_ece,
    compute_ause,
)
from .utils import (
    enable_mc_dropout,
    save_checkpoint,
    load_checkpoint,
    CosineWarmupScheduler,
    EarlyStopping,
    set_seed,
)

__all__ = [
    # Config
    "BTPNConfig",
    "BTPNFeatureConfig",
    # Models
    "BTPN",
    "KinematicFoundationModel",
    "PivotPointEstimator",
    # Losses
    "KinematicLoss",
    "BTPNLoss",
    "geodesic_distance",
    "beta_nll_loss",
    # Metrics
    "compute_position_metrics",
    "compute_geodesic_error",
    "compute_ece",
    "compute_ause",
    # Utils
    "enable_mc_dropout",
    "save_checkpoint",
    "load_checkpoint",
    "CosineWarmupScheduler",
    "EarlyStopping",
    "set_seed",
]
