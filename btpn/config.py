"""Unified configuration for the Bayesian Temporal Pose Network (BTPN).

Flattens the five-level configuration hierarchy from the research codebase
(BTPNConfig -> BTPNConfigV3 -> BTPNConfigV5 -> VisualTemporalConfig ->
VisualTemporalConfigV2 -> VisualTemporalConfigV3) into a single flat
dataclass with YAML override support.

All field names are preserved exactly for checkpoint compatibility.

Paper reference: Sections 3-5 (architecture, training, experiments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Feature Index Configuration
# =============================================================================


@dataclass
class BTPNFeatureConfig:
    """Feature index mapping for 7DOF kinematic data (30D).

    The 30-dimensional feature vector is organized as:
        Tool1: Position(3) + Quaternion(4) + Angle(1) = indices 0-7
        Tool2: Position(3) + Quaternion(4) + Angle(1) = indices 8-15
        Camera: Position(3) + Quaternion(4) = indices 16-22
        World: Position(3) + Quaternion(4) = indices 23-29

    All index tuples are (start, stop) for Python slicing.
    """

    # Tool 1 indices
    tool1_pos: tuple[int, int] = (0, 3)
    tool1_quat: tuple[int, int] = (3, 7)
    tool1_angle: int = 7

    # Tool 2 indices
    tool2_pos: tuple[int, int] = (8, 11)
    tool2_quat: tuple[int, int] = (11, 15)
    tool2_angle: int = 15

    # Camera indices
    camera_pos: tuple[int, int] = (16, 19)
    camera_quat: tuple[int, int] = (19, 23)

    # World indices
    world_pos: tuple[int, int] = (23, 26)
    world_quat: tuple[int, int] = (26, 30)

    # Output (prediction) indices -- tools only
    output_tool1: tuple[int, int] = (0, 8)
    output_tool2: tuple[int, int] = (8, 16)

    # Feature names for logging
    feature_names: list[str] = field(default_factory=lambda: [
        "Tool1_X", "Tool1_Y", "Tool1_Z",
        "Tool1_qw", "Tool1_qx", "Tool1_qy", "Tool1_qz",
        "Tool1_Angle",
        "Tool2_X", "Tool2_Y", "Tool2_Z",
        "Tool2_qw", "Tool2_qx", "Tool2_qy", "Tool2_qz",
        "Tool2_Angle",
        "Camera_X", "Camera_Y", "Camera_Z",
        "Camera_qw", "Camera_qx", "Camera_qy", "Camera_qz",
        "World_X", "World_Y", "World_Z",
        "World_qw", "World_qx", "World_qy", "World_qz",
    ])

    output_names: list[str] = field(default_factory=lambda: [
        "Tool1_X", "Tool1_Y", "Tool1_Z",
        "Tool1_qw", "Tool1_qx", "Tool1_qy", "Tool1_qz",
        "Tool1_Angle",
        "Tool2_X", "Tool2_Y", "Tool2_Z",
        "Tool2_qw", "Tool2_qx", "Tool2_qy", "Tool2_qz",
        "Tool2_Angle",
    ])


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class BTPNConfig:
    """Unified configuration for the BTPN architecture and training.

    Consolidates all parameters from the research codebase's five-level
    config hierarchy into a single flat dataclass. Fields are organized
    by section with comments indicating their origin and purpose.

    Use ``BTPNConfig.from_yaml(path)`` to load a YAML file and override
    any subset of defaults.

    Attributes are grouped into:
        - Input/output dimensions
        - Transformer architecture
        - Hierarchical temporal attention (Paper Section 3.2)
        - Bimanual cross-attention (Paper Section 3.3)
        - Memory-enhanced encoder (Paper Section 3.2)
        - Multi-scale configuration (Paper Section 3.4)
        - Probabilistic output heads (Paper Section 3.5)
        - Loss weights (Paper Section 4.1)
        - Training schedule
        - Visual modality (Paper Section 3.6)
        - Multi-channel gate and residual correction (Paper Section 3.7)
        - Pivot estimation (Paper Section 3.8)
        - Data and augmentation
    """

    # ========================= Input / Output ================================
    input_dim: int = 30           # Tool1(8) + Tool2(8) + Camera(7) + World(7)
    output_dim: int = 16          # Tool1(8) + Tool2(8) poses only
    tool_dim: int = 8             # Position(3) + Quaternion(4) + Angle(1)
    position_dim: int = 3
    quaternion_dim: int = 4
    angle_dim: int = 1

    # ========================= Transformer Architecture ======================
    d_model: int = 256            # Hidden dimension
    n_heads: int = 8              # Attention heads
    n_encoder_layers: int = 6     # Total encoder depth (2 local + 2 med + 2 glob)
    ff_dim: int = 1024            # Feed-forward dimension (4x d_model)
    dropout: float = 0.1          # Training dropout
    activation: str = "gelu"      # Activation function

    # ========================= Hierarchical Temporal Attention ================
    # Paper Section 3.2: Multi-scale temporal attention hierarchy
    use_hierarchical: bool = True
    local_window: int = 5         # ~0.4s at 13 fps -- fine motion, tremor
    medium_window: int = 20       # ~1.5s -- motion phrases, gestures
    global_window: int = 64       # Full context -- surgical subtasks
    local_layers: int = 2
    medium_layers: int = 2
    global_layers: int = 2

    # Cross-scale attention between local/medium/global
    use_cross_scale_attention: bool = True

    # ========================= Bimanual Cross-Attention ======================
    # Paper Section 3.3: Tool1 <-> Tool2 coordination
    use_cross_attention: bool = True
    cross_attention_heads: int = 4

    # ========================= Memory-Enhanced Encoder =======================
    # Paper Section 3.2: Learnable memory slots + bidirectional context
    use_memory: bool = True
    use_bidirectional: bool = True
    use_gated_fusion: bool = True
    memory_size: int = 64         # Number of learnable memory slots
    memory_heads: int = 4
    context_before: int = 5       # Frames before target in window
    context_after: int = 4        # Frames after target in window

    # ========================= Multi-Scale Configuration =====================
    # Paper Section 3.4: Causal multi-scale windowing
    window_scales: list[int] = field(default_factory=lambda: [10, 50, 100])
    # 10 frames (~0.77s): micro-gesture level
    # 50 frames (~3.85s): action phase level
    # 100 frames (~7.69s): full peg transfer cycle

    # Cross-scale fusion
    n_scale_fusion_heads: int = 4
    scale_token_dim: int = 256    # Same as d_model
    use_confidence_weighted_fusion: bool = True

    # ========================= Sequence Configuration ========================
    sequence_length: int = 64     # Input context (~5s)
    prediction_horizon: int = 1   # Steps ahead to predict
    max_sequence_length: int = 256

    # ========================= Probabilistic Output ==========================
    # Paper Section 3.5: Position(Gaussian), Rotation(VMF), Angle(Gaussian)
    covariance_type: str = "diagonal"  # "diagonal" or "cholesky"
    use_joint_covariance: bool = False  # Joint 6x6 bimanual Cholesky covariance
    cholesky_warmup_epochs: int = 20   # Diagonal-only for first N epochs
    beta_nll: float = 0.5         # Beta parameter for Beta-NLL loss
    min_sigma: float = 1e-3       # Minimum predicted sigma
    max_sigma: float = 10.0       # Maximum predicted sigma
    min_kappa: float = 1.0        # Minimum VMF concentration

    # MC Dropout
    mc_dropout_rate: float = 0.1  # MC Dropout at inference
    mc_samples: int = 30          # Number of MC forward passes

    # MDN configuration
    use_mdn: bool = True
    num_mdn_components: int = 5
    mdn_hidden_dim: int = 128

    # ========================= Loss Weights ==================================
    # Paper Section 4.1: Multi-task loss formulation
    lambda_position: float = 1.0       # Position Beta-NLL
    lambda_quaternion: float = 3.0     # Rotation Beta-VMF
    lambda_angle: float = 1.0          # Jaw angle Beta-NLL
    lambda_jaw_state: float = 0.5      # Jaw state BCE
    lambda_calibration: float = 0.1    # Differentiable ECE penalty
    lambda_reconstruction: float = 0.1 # Auxiliary SSL loss
    lambda_geodesic: float = 0.0       # Geodesic rotation loss

    # Smoothness regularization
    lambda_velocity: float = 0.3
    lambda_acceleration: float = 0.1
    lambda_jerk: float = 0.05

    # Auxiliary decoders
    use_auxiliary_reconstruction: bool = True
    use_auxiliary_decoders: bool = True
    lambda_auxiliary: float = 0.2

    # Calibration settings
    n_calibration_bins: int = 10
    calibration_start_epoch: int = 10

    # ========================= Training ======================================
    epochs: int = 10000           # High ceiling; patience stops early
    batch_size: int = 32
    lr: float = 3e-4
    warmup_epochs: int = 15
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 30

    # Learning rate scheduling
    use_cosine_annealing: bool = True
    cosine_min_lr: float = 1e-7

    # Mixed precision
    use_amp: bool = False         # AMP float16 can overflow in ECE loss

    # Checkpointing
    checkpoint_every: int = 10
    visualize_every: int = 20
    seed: int = 42
    num_workers: int = 0          # 0 for Windows multiprocessing compatibility
    pin_memory: bool = True
    experiment_name: str = "btpn"

    # ========================= Jaw Angle =====================================
    calibrate_jaw_angles: bool = True
    jaw_lower_percentile: float = 10.0
    jaw_upper_percentile: float = 90.0
    jaw_state_threshold: float = 50.0
    predict_jaw_state: bool = True
    num_jaw_classes: int = 1  # Binary (open/closed), 1 logit per tool

    # ========================= Data ==========================================
    train_dataset: str = "7DOF2024"
    val_dataset: str = "BAPES2024"
    test_dataset: str = "6DOF2023"
    train_val_split: float = 0.85

    # Augmentation
    augment_train: bool = True
    noise_std: float = 0.01
    speed_range: tuple[float, float] = (0.8, 1.2)
    tool_swap_prob: float = 0.5
    temporal_frame_rate: float = 13.0  # fps for temporal derivatives

    # ========================= Visual Modality ===============================
    # Paper Section 3.6: Segmentation + depth + pose keypoint branches

    # YOLO segmentation features
    seg_neck_dim: int = 256
    seg_backbone_dim: int = 512
    seg_proj_dim: int = 256

    # Depth features (DINOv2-S)
    depth_feature_dim: int = 384
    depth_proj_dim: int = 128
    use_depth_features: bool = True

    # Pose keypoint features
    pose_keypoint_dim: int = 48       # 2 tools x 8 kp x 3
    pose_backbone_dim: int = 512
    pose_geometric_dim: int = 6       # shaft_width + midline_angle + jaw_opening
    pose_kp_proj_dim: int = 64
    pose_enc_proj_dim: int = 64
    pose_proj_dim: int = 128
    use_pose_features: bool = True

    # Cross-modal fusion
    cross_modal_heads: int = 4
    cross_modal_dropout: float = 0.1
    use_gated_cross_modal: bool = True

    # Vision-guided uncertainty
    use_vision_uncertainty: bool = True
    uncertainty_alpha_init: float = 2.0
    lambda_vision_uncertainty: float = 0.05

    # Visual learning rates
    visual_lr_scale: float = 0.5
    fusion_lr_scale: float = 1.0

    # Visual data
    positive_only: bool = True
    sample_interval: int = 2

    # Kinematic prior (frozen)
    kinematic_checkpoint: str = ""
    kinematic_norm_stats: str = ""
    freeze_kinematic: bool = True

    # Visual temporal encoder
    use_bidirectional_visual: bool = True
    visual_n_layers: int = 6
    visual_n_heads: int = 4
    visual_d_model: int = 256
    visual_dropout: float = 0.1

    # ========================= Multi-Channel Gate ============================
    # Paper Section 3.7: Per-channel gate ceilings
    max_gate_position: float = 0.5
    max_gate_rotation: float = 0.1     # Conservative to prevent rotation corruption
    max_gate_angle: float = 0.5
    gate_init_temperature: float = 5.0 # High temp -> conservative initial gates
    gate_entropy_weight: float = 0.01  # Anti-saturation regularizer

    # Residual correction
    max_position_delta: float = 5.0
    max_quaternion_delta: float = 0.15
    max_angle_delta: float = 1.0

    # Confidence gate
    positive_only_loss: bool = True
    min_detection_conf: float = 0.3
    gate_hidden_dim: int = 64

    # Residual regularization
    lambda_residual_reg: float = 0.005
    lambda_residual_reg_finetune: float = 0.002

    # ========================= Relative Tracking =============================
    use_relative_tracking: bool = True
    lambda_displacement: float = 0.5
    max_displacement_delta: float = 5.0
    lambda_smoothness: float = 0.01

    # ========================= Pivot Estimation ==============================
    # Paper Section 3.8: Trocar-aware uncertainty inflation
    use_pivot_estimation: bool = True
    pivot_warmup_epochs: int = 60
    pivot_decay: float = 0.99
    pivot_inflation_threshold: float = 5.0   # mm
    pivot_max_inflation: float = 3.0
    lambda_pivot: float = 0.1

    # ========================= Clinical Attention Spans ======================
    visual_window_scales: list[int] = field(
        default_factory=lambda: [8, 40, 100, 200]
    )
    visual_local_window: int = 8
    visual_medium_window: int = 20

    # ========================= 2-Stage Training ==============================
    # Stage 1: Visual SSL pre-training
    stage1_max_epochs: int = 500
    stage1_patience: int = 25
    stage1_lr: float = 5e-4
    lambda_mvr: float = 1.0       # Masked Visual Reconstruction
    lambda_vcl: float = 0.5       # Visual Contrastive Learning
    lambda_vtop: float = 0.2      # Visual Temporal Order Prediction
    lambda_vka: float = 0.3       # Visual-Kinematic Alignment
    mvr_mask_ratio: float = 0.3

    # Stage 2: Supervised fine-tuning
    stage2_max_epochs: int = 1000
    stage2_patience: int = 30
    stage2_lr: float = 3e-4
    stage2_warmup_epochs: int = 20
    stage2_full_end_epoch: int = 150
    stage2_finetune_lr_scale: float = 0.1
    stage2_finetune_conf_threshold: float = 0.5

    # ========================= SSL Pre-training (kinematic) ==================
    mask_ratio_frames: float = 0.15
    mask_ratio_features: float = 0.20
    mask_block_size: int = 4
    contrastive_temperature: float = 0.07
    num_temporal_segments: int = 4
    lambda_mta: float = 1.0
    lambda_cml: float = 0.5
    lambda_top: float = 0.3

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> BTPNConfig:
        """Load configuration from a YAML file, overriding defaults.

        Any key present in the YAML that matches a field name will
        override the default value. Unknown keys are silently ignored.

        Args:
            path: Path to a YAML configuration file.

        Returns:
            BTPNConfig instance with overridden values.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        path = Path(path)
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        # Collect only keys that match dataclass fields
        field_names = {fld.name for fld in cls.__dataclass_fields__.values()}
        overrides = {k: v for k, v in raw.items() if k in field_names}

        return cls(**overrides)

    # ------------------------------------------------------------------
    # Training helper methods
    # ------------------------------------------------------------------

    def get_stage2_phase(self, epoch: int) -> str:
        """Get the current training phase within stage 2.

        Stage 2 has three phases:
            warmup (0 to stage2_warmup_epochs):
                Train fusion + gate only; visual encoder frozen.
            full (stage2_warmup_epochs to stage2_full_end_epoch):
                All parameters unfrozen with cosine LR.
            finetune (after stage2_full_end_epoch):
                Reduced LR, stricter confidence, pivot active.

        Args:
            epoch: Current epoch within stage 2 (0-indexed).

        Returns:
            Phase name: "warmup", "full", or "finetune".
        """
        if epoch < self.stage2_warmup_epochs:
            return "warmup"
        elif epoch < self.stage2_full_end_epoch:
            return "full"
        return "finetune"

    def get_stage2_lr_scale(self, epoch: int) -> float:
        """Get learning rate scale factor for the current stage 2 phase.

        Args:
            epoch: Current epoch within stage 2 (0-indexed).

        Returns:
            LR scale factor in (0, 1].
        """
        phase = self.get_stage2_phase(epoch)
        if phase == "finetune":
            return self.stage2_finetune_lr_scale
        return 1.0

    def get_pivot_warmup_weight(self, epoch: int) -> float:
        """Get pivot consistency loss weight for a given epoch.

        Linearly ramps from 0 to ``lambda_pivot`` over 20 epochs
        starting at ``pivot_warmup_epochs``. Returns 0 if pivot
        estimation is disabled.

        Args:
            epoch: Current training epoch.

        Returns:
            Pivot loss weight in [0, lambda_pivot].
        """
        if not self.use_pivot_estimation or epoch < self.pivot_warmup_epochs:
            return 0.0
        ramp_epochs = 20
        progress = min(1.0, (epoch - self.pivot_warmup_epochs) / ramp_epochs)
        return self.lambda_pivot * progress

    def get_calibration_weight(self, epoch: int) -> float:
        """Get calibration loss weight for a given epoch.

        Linearly ramps from 0 to ``lambda_calibration`` over 20 epochs
        starting at ``calibration_start_epoch``.

        Args:
            epoch: Current training epoch.

        Returns:
            Calibration loss weight in [0, lambda_calibration].
        """
        if epoch < self.calibration_start_epoch:
            return 0.0
        ramp_epochs = 20
        progress = min(1.0, (epoch - self.calibration_start_epoch) / ramp_epochs)
        return self.lambda_calibration * progress

    def get_cholesky_enabled(self, epoch: int) -> bool:
        """Check whether full Cholesky covariance is active at a given epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            True if Cholesky covariance should be used.
        """
        return (
            self.covariance_type == "cholesky"
            and epoch >= self.cholesky_warmup_epochs
        )

    def get_visual_warmup_weight(
        self,
        epoch: int,
        warmup_epochs: int = 10,
    ) -> float:
        """Get visual modality warmup weight for a given epoch.

        Linearly ramps cross-modal fusion from 0 to 1 over
        ``warmup_epochs``.

        Args:
            epoch: Current training epoch.
            warmup_epochs: Number of warmup epochs for visual fusion.

        Returns:
            Visual warmup weight in [0, 1].
        """
        if epoch >= warmup_epochs:
            return 1.0
        return epoch / warmup_epochs

    def get_residual_reg_weight(self, epoch: int) -> float:
        """Get residual regularization weight for the current epoch.

        Switches to the reduced fine-tuning weight after
        ``stage2_full_end_epoch``.

        Args:
            epoch: Current training epoch.

        Returns:
            Regularization weight.
        """
        if epoch >= self.stage2_full_end_epoch:
            return self.lambda_residual_reg_finetune
        return self.lambda_residual_reg

    def get_conf_threshold(self, epoch: int) -> float:
        """Get detection confidence threshold for the current epoch.

        Uses the stricter fine-tuning threshold after
        ``stage2_full_end_epoch``.

        Args:
            epoch: Current training epoch.

        Returns:
            Confidence threshold.
        """
        if epoch >= self.stage2_full_end_epoch:
            return self.stage2_finetune_conf_threshold
        return self.min_detection_conf
