"""Training utilities for the Bayesian Temporal Pose Network (BTPN).

Provides common training infrastructure: learning rate scheduling,
checkpointing, Monte Carlo dropout, reproducibility, parameter counting,
early stopping, and visualization helpers.

Paper reference: Section 4 (Training Procedure).
"""

from __future__ import annotations

import contextlib
import logging
import random
from collections.abc import Generator
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup.

    Linearly ramps the learning rate from 0 to the base LR over
    ``warmup_epochs``, then decays following a cosine half-period
    to ``min_lr`` by the end of training.

    Supports multiple parameter groups with different base learning
    rates -- each group's LR is scaled by the same warmup/decay factor.

    Args:
        optimizer: PyTorch optimizer with at least one parameter group.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs (warmup + cosine).
        min_lr: Minimum learning rate floor.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        >>> scheduler = CosineWarmupScheduler(optimizer, 15, 200, min_lr=1e-7)
        >>> for epoch in range(200):
        ...     lr = scheduler.step(epoch)
        ...     train_one_epoch(...)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        """Update learning rates for the given epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Learning rate of the first parameter group after update.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to base_lr
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing from base_lr to min_lr
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)

        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler | Any | None,
    epoch: int,
    path: str | Path,
    *,
    best_loss: float = float("inf"),
    scaler: Any | None = None,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    **extra: Any,
) -> None:
    """Save a training checkpoint to disk.

    Stores model weights, optimizer state, scheduler metadata, epoch,
    best loss, optional AMP scaler state, and optional normalization
    statistics. Any additional keyword arguments are also stored.

    Args:
        model: PyTorch model.
        optimizer: Optimizer whose state should be saved.
        scheduler: Learning rate scheduler (saved as dict if it has state_dict).
        epoch: Current epoch number.
        path: File path for the checkpoint.
        best_loss: Best validation loss so far.
        scaler: Optional AMP GradScaler.
        norm_mean: Optional normalization mean vector.
        norm_std: Optional normalization std vector.
        **extra: Additional key-value pairs to store (e.g. config, phase).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "best_val_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        else:
            # Store scheduler attributes for custom schedulers
            checkpoint["scheduler_info"] = {
                "warmup_epochs": getattr(scheduler, "warmup_epochs", None),
                "total_epochs": getattr(scheduler, "total_epochs", None),
                "min_lr": getattr(scheduler, "min_lr", None),
                "base_lrs": getattr(scheduler, "base_lrs", None),
            }

    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if norm_mean is not None:
        checkpoint["norm_mean"] = norm_mean
    if norm_std is not None:
        checkpoint["norm_std"] = norm_std

    checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: CosineWarmupScheduler | Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load a training checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state into. If None, skip.
        scheduler: Scheduler to restore state into. If None, skip.
        scaler: AMP GradScaler to restore. If None, skip.
        map_location: Device mapping for torch.load.
        strict: If True, require exact state dict match. Default False
            to handle auxiliary modules or config differences gracefully.

    Returns:
        Dictionary with metadata from the checkpoint:
            epoch (int), best_loss (float), and any extra keys.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict,
    )
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.info("Unexpected keys in checkpoint (ignored): %d keys", len(unexpected))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        if hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    meta: dict[str, Any] = {
        "epoch": checkpoint.get("epoch", 0),
        "best_loss": checkpoint.get("best_val_loss", float("inf")),
    }

    # Pass through any extra keys
    skip_keys = {
        "model_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "scaler_state_dict",
        "epoch", "best_val_loss",
    }
    for key, value in checkpoint.items():
        if key not in skip_keys:
            meta[key] = value

    return meta


# =============================================================================
# MC Dropout
# =============================================================================


class MCDropout(nn.Dropout):
    """Dropout layer that remains active during inference for MC sampling.

    Identical to ``nn.Dropout`` except that ``forward()`` always applies
    dropout regardless of the module's training mode. Used in conjunction
    with :func:`enable_mc_dropout` to selectively enable dropout at
    inference time without affecting batch normalization or other layers.

    Args:
        p: Dropout probability.
        inplace: If True, operate in-place.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout unconditionally (ignores self.training)."""
        return nn.functional.dropout(x, self.p, training=True, inplace=self.inplace)


@contextlib.contextmanager
def enable_mc_dropout(model: nn.Module) -> Generator[None, None, None]:
    """Enable only MCDropout layers while keeping the model in eval mode.

    Standard ``model.train()`` enables dropout but also changes batch
    normalization behavior, corrupting uncertainty estimates. This context
    manager solves the problem by keeping the model in eval mode and only
    setting MCDropout submodules to training mode.

    Args:
        model: PyTorch model containing MCDropout layers.

    Yields:
        None. Model state is temporarily modified.

    Example:
        >>> model.eval()
        >>> with enable_mc_dropout(model):
        ...     samples = [model(x) for _ in range(30)]
        >>> # Model is restored to its original state.
    """
    was_training = model.training
    model.eval()

    # Find and enable MCDropout layers
    mc_states: list[tuple[MCDropout, bool]] = []
    for module in model.modules():
        if isinstance(module, MCDropout):
            mc_states.append((module, module.training))
            module.train()

    try:
        yield
    finally:
        for module, was_train in mc_states:
            module.train(was_train)
        model.train(was_training)


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training.

    Seeds Python's built-in ``random``, NumPy, and PyTorch (CPU and
    CUDA). Also sets ``torch.backends.cudnn.deterministic`` for full
    reproducibility (at the cost of some performance).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Parameter Counting
# =============================================================================


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with keys "total" and "trainable", each an int.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """Early stopping tracker that monitors a validation metric.

    Stops training when the monitored metric has not improved for
    ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to qualify as an improvement.

    Example:
        >>> stopper = EarlyStopping(patience=20, min_delta=1e-4)
        >>> for epoch in range(max_epochs):
        ...     val_loss = validate(...)
        ...     if stopper.step(val_loss):
        ...         print(f"Early stop at epoch {epoch}")
        ...         break
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float = float("inf")
        self.counter: int = 0
        self.best_epoch: int = 0

    def step(self, value: float, epoch: int = 0) -> bool:
        """Record a new metric value and check for stopping.

        Args:
            value: Current validation metric (lower is better).
            epoch: Current epoch number (for tracking best epoch).

        Returns:
            True if training should stop (patience exhausted).
        """
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False

        self.counter += 1
        return self.counter >= self.patience

    @property
    def should_stop(self) -> bool:
        """Whether the patience budget has been exhausted."""
        return self.counter >= self.patience


# =============================================================================
# Visualization Helpers
# =============================================================================


def plot_training_curves(
    history: list[dict[str, Any]],
    save_path: str | Path,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Plot training and validation loss/metric curves.

    Creates a 2x2 grid of subplots:
        (0,0) Training and validation loss
        (0,1) Position MAE (mm)
        (1,0) Rotation error (degrees)
        (1,1) Learning rate schedule

    Args:
        history: List of per-epoch dictionaries. Each should contain
            "epoch", "train" (sub-dict with "loss"), "val" (sub-dict
            with "loss", "position_mae_mm", "rotation_error_deg"),
            and "lr".
        save_path: Path to save the figure (PNG or PDF).
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt

    epochs = [r.get("epoch", i) for i, r in enumerate(history)]
    train_loss = [r.get("train", {}).get("loss", float("nan")) for r in history]
    val_loss = [r.get("val", {}).get("loss", float("nan")) for r in history]
    pos_mae = [r.get("val", {}).get("position_mae_mm", float("nan")) for r in history]
    rot_err = [r.get("val", {}).get("rotation_error_deg", float("nan")) for r in history]
    lrs = [r.get("lr", float("nan")) for r in history]

    # Also support flat history format (e.g., from Stage 2)
    if all(np.isnan(tl) for tl in train_loss):
        train_loss = [r.get("train_loss", float("nan")) for r in history]
        val_loss = [r.get("val_loss", float("nan")) for r in history]
        pos_mae = [r.get("val_pos_mae_mm", float("nan")) for r in history]
        rot_err = [r.get("val_rot_err_deg", float("nan")) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train", alpha=0.8)
    ax.plot(epochs, val_loss, label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position MAE
    ax = axes[0, 1]
    ax.plot(epochs, pos_mae, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mm)")
    ax.set_title("Position MAE")
    ax.grid(True, alpha=0.3)

    # Rotation error
    ax = axes[1, 0]
    ax.plot(epochs, rot_err, color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (degrees)")
    ax.set_title("Rotation Error")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, lrs, color="#4CAF50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path,
    tool_labels: tuple[str, str] = ("Tool 1", "Tool 2"),
    figsize: tuple[float, float] = (16, 6),
) -> None:
    """Plot 3D predicted vs ground-truth tool trajectories.

    Creates two side-by-side 3D scatter plots (one per tool) showing
    predicted positions overlaid on ground-truth positions.

    Args:
        predictions: (N, 2, 3) predicted positions in mm.
        targets: (N, 2, 3) ground-truth positions in mm.
        save_path: Path to save the figure.
        tool_labels: Display labels for Tool 1 and Tool 2.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt

    if predictions.ndim != 3 or predictions.shape[1:] != (2, 3):
        raise ValueError(
            f"Expected predictions shape (N, 2, 3), got {predictions.shape}"
        )

    fig = plt.figure(figsize=figsize)

    colors = {"pred": "#2196F3", "target": "#9E9E9E"}

    for tool_idx in range(2):
        ax = fig.add_subplot(1, 2, tool_idx + 1, projection="3d")

        pred_t = predictions[:, tool_idx]
        tgt_t = targets[:, tool_idx]

        ax.scatter(
            tgt_t[:, 0], tgt_t[:, 1], tgt_t[:, 2],
            c=colors["target"], s=1, alpha=0.3, label="Ground Truth",
        )
        ax.scatter(
            pred_t[:, 0], pred_t[:, 1], pred_t[:, 2],
            c=colors["pred"], s=1, alpha=0.3, label="Predicted",
        )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(tool_labels[tool_idx])
        ax.legend(markerscale=5)

    fig.suptitle("3D Tool Trajectories: Predicted vs Ground Truth")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
