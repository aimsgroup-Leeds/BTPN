"""Core BTPN model architectures for surgical tool pose prediction.

This module contains the two main model classes:

1. ``KinematicFoundationModel`` -- A multi-scale transformer that predicts
   next-frame tool poses from kinematic (electromagnetic tracker) data alone,
   with full probabilistic uncertainty quantification (Gaussian, VMF, Beta-NLL).

2. ``BTPN`` -- The full Bayesian Temporal Pose Network that extends the
   kinematic foundation model with visual features (segmentation, depth,
   pose keypoints) through a residual correction architecture with
   multi-channel confidence gating.

Also includes:

3. ``PivotPointEstimator`` -- EMA-based trocar (entry point) estimator that
   exploits the physical constraint that all tool shafts pass through the
   abdominal wall entry point.

Architecture overview::

    KinematicFoundationModel (frozen prior):
        Multi-scale kinematic windows [10, 50, 100 frames]
            --> Shared PoseInputEmbedding
            --> Shared HierarchicalTemporalTransformer
            --> CrossScaleFusion
            --> ProbabilisticPoseHead
            --> mu_pos, sigma_pos, mu_quat, kappa, mu_angle, sigma_angle

    BTPN (full model):
        Frozen KinematicFoundationModel (Stage 0)
            |
        Visual Feature Encoding (Stage 1)
            --> SegmentationProjection + DepthProjection + KeypointProjection
            --> SceneFusion + PoseProjection
            --> visual_proj
            |
        ClinicalAttentionEncoder (Stage 2)
            --> visual_repr (B, 256) from CLS token
            |
        KinematicVisualFusion (Stage 3)
            --> gated cross-attention fusion
            |
        ConfidenceGate + ResidualPoseHead (Stage 4)
            --> per-channel gated corrections to kinematic prior

Paper cross-references:
    - Section 3.1: Kinematic Foundation Model (KinematicFoundationModel)
    - Section 3.2: Visual Feature Extraction (visual.py)
    - Section 3.3: Clinical Attention Encoder (visual.py)
    - Section 3.4: Kinematic-Visual Fusion (visual.py)
    - Section 3.5: Multi-Channel Gating and Residual Correction
    - Section 3.6: Pivot Point Estimation
    - Figure 2: Full BTPN architecture overview
    - Table 1: Model parameter counts

Note on internal attribute naming:
    All ``self.xxx`` attribute names are preserved exactly from the research
    codebase for checkpoint compatibility:
    - KinematicFoundationModel uses BTPNv5 attribute names
      (self.encoder -> self.input_embed, self.prob_head, etc.)
    - BTPN uses VisualTemporalBTPNv3 attribute names
      (self.btpn_v5 -> holds KinematicFoundationModel instance,
       self.clinical_encoder, self.confidence_gate, self.residual_head, etc.)
    The public class names use clean paper terminology, but the internal
    structure must not be changed to preserve checkpoint loading.

Author: BTPN Publication Repository
"""

from __future__ import annotations

import contextlib
import math
import warnings
from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BTPNConfig
from .components import (
    PoseInputEmbedding,
    LearnableTemporalEncoding,
    HierarchicalTemporalTransformer,
    BimanualCrossAttention,
    MCDropout,
    MemoryEnhancedEncoder,
    GatedToolFusion,
)
from .visual import (
    SegmentationProjection,
    DepthProjection,
    KeypointProjection,
    PoseProjection,
    SceneFusion,
    KinematicVisualFusion,
    ClinicalAttentionEncoder,
    ConfidenceGate,
    ResidualPoseHead,
    gate_entropy_loss,
    # V3-compatible sub-components for checkpoint loading in BTPN
    _SegNeckProjection,
    _SegBackboneProjection,
    _LearnableSegFusion,
    _PoseKeypointProjection,
    _PoseGeometricProjection,
    _LearnablePoseFusion,
)


# =============================================================================
# MC Dropout Context Manager
# =============================================================================


@contextlib.contextmanager
def enable_mc_dropout(model: nn.Module) -> Generator[None, None, None]:
    """Enable only MCDropout layers while keeping the model in eval mode.

    This context manager solves the issue where calling ``model.train()`` to
    enable dropout for MC inference also enables batch norm training behavior
    and other regularization, corrupting uncertainty estimates.

    Only ``MCDropout`` layers (which always apply dropout, even in eval mode)
    are switched to training mode. All other layers (LayerNorm, etc.) remain
    in eval mode.

    Paper: Section 3.1, "Epistemic Uncertainty via MC Dropout"

    Args:
        model: PyTorch model containing MCDropout layers.

    Yields:
        None. Model state is temporarily modified.

    Example::

        model.eval()
        with enable_mc_dropout(model):
            samples = [model(x) for _ in range(30)]
        # Model is back to original state
    """
    was_training = model.training
    model.eval()

    mc_dropout_states: list[tuple[MCDropout, bool]] = []
    for module in model.modules():
        if isinstance(module, MCDropout):
            mc_dropout_states.append((module, module.training))
            module.train()

    try:
        yield
    finally:
        for module, was_train in mc_dropout_states:
            module.train(was_train)
        model.train(was_training)


# =============================================================================
# Probabilistic Output Head
# =============================================================================


class ProbabilisticPoseHead(nn.Module):
    """Probabilistic output head with full Cholesky covariance support.

    Produces per-tool (x2 tools) probabilistic predictions:

    - **Position**: mu(3) + sigma(3) or Cholesky L(6) for full 3x3
      covariance. Loss: Beta-NLL (Seitzer et al., 2022).
    - **Rotation**: mu_quat(4) + kappa(1) for von Mises-Fisher on S^3.
      Loss: Beta-VMF negative log-likelihood.
    - **Jaw angle**: mu(1) + sigma(1) for Gaussian. Loss: Beta-NLL.
    - **Jaw state**: logits(1) for Bernoulli. Loss: BCE.

    Paper: Section 3.1, "Probabilistic Output Head"
    Equation 3: Beta-NLL loss formulation
    Equation 5: VMF rotation loss

    Checkpoint compatibility:
        Renamed from ProbabilisticPoseHeadV5 (BTPNv5). Uses attribute
        names: self.hidden, self.pos_mu, self.pos_sigma, self.pos_cholesky,
        self.quat_mu, self.quat_kappa, self.angle_mu, self.angle_sigma,
        self.jaw_state_head.

    Args:
        d_model: Input model dimension.
        use_cholesky: Output full Cholesky covariance parameters.
        use_joint_covariance: Output joint 6x6 bimanual Cholesky.
        min_sigma: Minimum predicted sigma.
        max_sigma: Maximum predicted sigma.
        min_kappa: Minimum VMF concentration.
        mc_dropout_rate: MC Dropout rate for epistemic uncertainty.
        num_jaw_classes: Number of jaw state classes (2=binary).
    """

    def __init__(
        self,
        d_model: int = 256,
        use_cholesky: bool = False,
        use_joint_covariance: bool = False,
        min_sigma: float = 1e-4,
        max_sigma: float = 10.0,
        min_kappa: float = 1.0,
        mc_dropout_rate: float = 0.1,
        num_jaw_classes: int = 2,
    ):
        super().__init__()
        self.use_cholesky = use_cholesky
        self.use_joint_covariance = use_joint_covariance
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kappa = min_kappa
        self.num_jaw_classes = num_jaw_classes

        hidden_dim = d_model // 2

        # Shared hidden layer with MC Dropout for epistemic uncertainty
        self.hidden = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            MCDropout(mc_dropout_rate),
        )

        # Position outputs (2 tools x 3D)
        self.pos_mu = nn.Linear(hidden_dim, 6)
        self.pos_sigma = nn.Linear(hidden_dim, 6)  # diagonal sigma
        if use_cholesky:
            self.pos_cholesky = nn.Linear(hidden_dim, 12)  # 2 x 6 Cholesky
        if use_joint_covariance:
            self.pos_joint_cholesky = nn.Linear(hidden_dim, 21)  # 6x6 joint

        # Quaternion outputs (2 tools x 4D mu + 1D kappa)
        self.quat_mu = nn.Linear(hidden_dim, 8)
        self.quat_kappa = nn.Linear(hidden_dim, 2)

        # Jaw angle outputs (2 tools x 1D each)
        self.angle_mu = nn.Linear(hidden_dim, 2)
        self.angle_sigma = nn.Linear(hidden_dim, 2)

        # Jaw state classification
        jaw_out_dim = 2 * num_jaw_classes
        self.jaw_state_head = nn.Linear(hidden_dim, jaw_out_dim)

    def forward(
        self,
        x: torch.Tensor,
        force_diagonal: bool = False,
        use_joint_covariance: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute probabilistic pose outputs.

        Args:
            x: (B, D) fused multi-scale representation.
            force_diagonal: Force diagonal covariance even if Cholesky
                is enabled (used during Cholesky warmup).
            use_joint_covariance: Enable joint 6x6 bimanual Cholesky.

        Returns:
            Dict with all predictions and uncertainty parameters:
                mu_position (B,2,3), sigma_position (B,2,3),
                mu_quaternion (B,2,4), kappa_quaternion (B,2,1),
                mu_angle (B,2,1), sigma_angle (B,2,1),
                jaw_state_logits (B,2) or (B,2,C),
                and optionally cholesky_position (B,2,6).
        """
        hidden = self.hidden(x)
        outputs: dict[str, torch.Tensor] = {}

        # ---- Position ----
        pos_mu = self.pos_mu(hidden).view(-1, 2, 3)
        outputs["mu_position"] = pos_mu

        if (
            use_joint_covariance
            and self.use_joint_covariance
            and not force_diagonal
        ):
            # Joint 6x6 Cholesky for bimanual covariance
            joint_L_params = self.pos_joint_cholesky(hidden)  # (B, 21)
            outputs["cholesky_joint_position"] = joint_L_params
            # Extract per-tool diagonal sigma for compatibility
            B = hidden.shape[0]
            L = torch.zeros(
                B, 6, 6, device=hidden.device, dtype=hidden.dtype
            )
            idx = 0
            for i in range(6):
                for j in range(i + 1):
                    if i == j:
                        L[:, i, j] = (
                            F.softplus(joint_L_params[:, idx]) + 1e-4
                        )
                    else:
                        L[:, i, j] = joint_L_params[:, idx]
                    idx += 1
            Sigma_diag = (L ** 2).sum(dim=-1)  # (B, 6)
            sigma_per_dim = torch.sqrt(Sigma_diag)
            outputs["sigma_position"] = sigma_per_dim.view(-1, 2, 3)
        elif self.use_cholesky and not force_diagonal:
            cholesky = self.pos_cholesky(hidden).view(-1, 2, 6)
            outputs["cholesky_position"] = cholesky
            # Extract diagonal from Cholesky for metrics
            L0 = F.softplus(cholesky[..., 0]) + 1e-4
            L2 = F.softplus(cholesky[..., 2]) + 1e-4
            L5 = F.softplus(cholesky[..., 5]) + 1e-4
            outputs["sigma_position"] = torch.stack([L0, L2, L5], dim=-1)
        else:
            pos_sigma = (
                F.softplus(self.pos_sigma(hidden)).view(-1, 2, 3)
                + self.min_sigma
            )
            pos_sigma = torch.clamp(pos_sigma, max=self.max_sigma)
            outputs["sigma_position"] = pos_sigma

        # ---- Quaternion (VMF) ----
        quat_mu = self.quat_mu(hidden).view(-1, 2, 4)
        quat_mu = F.normalize(quat_mu, dim=-1)
        kappa = (
            F.softplus(self.quat_kappa(hidden)).view(-1, 2, 1)
            + self.min_kappa
        )
        outputs["mu_quaternion"] = quat_mu
        outputs["kappa_quaternion"] = kappa

        # ---- Jaw Angle (Gaussian) ----
        angle_mu = self.angle_mu(hidden).view(-1, 2, 1)
        angle_sigma = (
            F.softplus(self.angle_sigma(hidden)).view(-1, 2, 1)
            + self.min_sigma
        )
        angle_sigma = torch.clamp(angle_sigma, max=self.max_sigma)
        outputs["mu_angle"] = angle_mu
        outputs["sigma_angle"] = angle_sigma

        # ---- Jaw State ----
        jaw_raw = self.jaw_state_head(hidden)
        if self.num_jaw_classes > 2:
            outputs["jaw_state_logits"] = jaw_raw.view(
                -1, 2, self.num_jaw_classes
            )
        else:
            outputs["jaw_state_logits"] = jaw_raw

        return outputs


# =============================================================================
# Cross-Scale Fusion
# =============================================================================


class CrossScaleFusion(nn.Module):
    """Fuse scale tokens from multiple temporal scales via cross-attention.

    Takes scale tokens extracted from each temporal scale's encoded
    representation and combines them using multi-head self-attention,
    producing a unified multi-scale representation.

    When confidence-weighted fusion is enabled, each scale predicts a
    confidence score. Scales with higher predicted confidence contribute
    more to the fused representation, creating an uncertainty-attention
    interaction.

    Paper: Section 3.1, "Cross-Scale Temporal Fusion"
    Equation 8: Confidence-weighted scale aggregation

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        n_scales: Number of temporal scales.
        use_confidence_weighted: Enable confidence-weighted pooling.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_scales: int = 3,
        use_confidence_weighted: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.use_confidence_weighted = use_confidence_weighted

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_confidence_weighted:
            self.confidence_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, 1),
                )
                for _ in range(n_scales)
            ])

    def forward(
        self,
        scale_tokens: list[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse scale tokens via cross-attention.

        Args:
            scale_tokens: List of scale token tensors, each (B, d_model).

        Returns:
            If confidence-weighted: (fused (B, d_model), weights (B, n_scales)).
            Otherwise: fused (B, d_model).
        """
        tokens = torch.stack(scale_tokens, dim=1)  # (B, n_scales, d_model)

        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        if self.use_confidence_weighted:
            confidences = []
            for i in range(self.n_scales):
                c_i = (
                    F.softplus(self.confidence_heads[i](tokens[:, i, :]))
                    + 1e-6
                )
                confidences.append(c_i.squeeze(-1))
            conf = torch.stack(confidences, dim=1)  # (B, n_scales)
            weights = conf / conf.sum(dim=1, keepdim=True)

            fused = (weights.unsqueeze(-1) * tokens).sum(dim=1)
            return fused, weights
        else:
            return tokens.mean(dim=1)


# =============================================================================
# Kinematic Foundation Model
# =============================================================================


class KinematicFoundationModel(nn.Module):
    """Multi-scale probabilistic kinematic pose prediction model.

    Processes three temporal scales (default 10, 50, 100 frames) through
    a shared encoder pipeline, fuses scale-specific summaries via
    cross-attention, and outputs calibrated probabilistic predictions.

    All windows are causal: they end at the target prediction frame t.

    Architecture:
        1. PoseInputEmbedding (shared) -- specialized per component type
        2. LearnableTemporalEncoding (shared) -- position + sinusoidal
        3. Scale token prepended to each scale's sequence
        4. HierarchicalTemporalTransformer (shared) -- local/medium/global
        5. MemoryEnhancedEncoder (shared) -- memory slots + bidirectional
        6. GatedToolFusion (shared) -- tool reliability weighting
        7. BimanualCrossAttention (shared) -- Tool1 <-> Tool2
        8. Scale token extraction --> CrossScaleFusion
        9. ProbabilisticPoseHead -- calibrated probabilistic outputs

    Paper: Section 3.1, "Kinematic Foundation Model"
    Figure 2a: Kinematic encoder architecture

    Checkpoint compatibility:
        Renamed from BTPNv5. All internal attribute names are preserved
        exactly: self.input_embed (was self.input_embed), self.encoder
        (mapped to temporal_transformer), self.pos_head / self.rot_head /
        self.angle_head (mapped through prob_head), etc. The state_dict
        keys match the original BTPNv5 checkpoint format.

    Args:
        config: Model configuration. Uses BTPNConfig defaults if None.
    """

    def __init__(self, config: BTPNConfig | None = None):
        super().__init__()
        self.config = config or BTPNConfig()
        cfg = self.config

        # ---- Shared Input Pipeline ----
        # NOTE: Attribute name 'input_embed' matches BTPNv5 for checkpoint compat
        self.input_embed = PoseInputEmbedding(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )

        # NOTE: Attribute name 'temporal_encoding' matches BTPNv5
        self.temporal_encoding = LearnableTemporalEncoding(
            d_model=cfg.d_model,
            max_len=max(cfg.window_scales) + 1,
            dropout=cfg.dropout,
        )

        # ---- Scale Tokens ----
        n_scales = len(cfg.window_scales)
        # NOTE: Attribute name 'scale_tokens' matches BTPNv5
        self.scale_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            for _ in range(n_scales)
        ])

        # ---- Shared Hierarchical Transformer ----
        # NOTE: Attribute name 'temporal_transformer' matches BTPNv5
        self.temporal_transformer = HierarchicalTemporalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            local_window=cfg.local_window,
            medium_window=cfg.medium_window,
            local_layers=cfg.local_layers,
            medium_layers=cfg.medium_layers,
            global_layers=cfg.global_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_cross_scale_attention=True,
        )

        # ---- Memory-Enhanced Encoder ----
        # NOTE: Attribute name 'memory_encoder' matches BTPNv5
        self.memory_encoder = MemoryEnhancedEncoder(
            d_model=cfg.d_model,
            memory_size=cfg.memory_size,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            context_before=cfg.context_before,
            context_after=cfg.context_after,
            use_memory=cfg.use_memory,
            use_bidirectional=cfg.use_bidirectional,
            use_gated_fusion=cfg.use_gated_fusion,
        )

        # ---- Tool Projections & Cross-Attention ----
        # NOTE: Attribute names 'tool1_proj', 'tool2_proj', 'tool_fusion',
        #       'cross_attention', 'cross_fusion' all match BTPNv5
        self.tool1_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.tool2_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.tool_fusion = GatedToolFusion(
            d_model=cfg.d_model,
            hidden_dim=cfg.d_model // 2,
            dropout=cfg.dropout,
        )

        if cfg.use_cross_attention:
            self.cross_attention = BimanualCrossAttention(
                d_model=cfg.d_model,
                n_heads=cfg.cross_attention_heads,
                dropout=cfg.dropout,
            )
            self.cross_fusion = nn.Linear(2 * cfg.d_model, cfg.d_model)

        # ---- Cross-Scale Fusion ----
        # NOTE: Attribute name 'scale_fusion' matches BTPNv5
        self.scale_fusion = CrossScaleFusion(
            d_model=cfg.d_model,
            n_heads=cfg.n_scale_fusion_heads,
            dropout=cfg.dropout,
            n_scales=n_scales,
            use_confidence_weighted=cfg.use_confidence_weighted_fusion,
        )

        # ---- Final Projection ----
        # NOTE: Attribute name 'final_proj' matches BTPNv5
        self.final_proj = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            MCDropout(cfg.mc_dropout_rate),
        )

        # ---- Probabilistic Output Head ----
        # NOTE: Attribute name 'prob_head' matches BTPNv5
        self.prob_head = ProbabilisticPoseHead(
            d_model=cfg.d_model,
            use_cholesky=(cfg.covariance_type == "cholesky"),
            use_joint_covariance=cfg.use_joint_covariance,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            min_kappa=cfg.min_kappa,
            mc_dropout_rate=cfg.mc_dropout_rate,
            num_jaw_classes=cfg.num_jaw_classes,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_scale(
        self,
        x: torch.Tensor,
        scale_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single temporal scale through the shared pipeline.

        Args:
            x: (B, T_scale, input_dim) kinematic input for one scale.
            scale_idx: Index into scale_tokens.

        Returns:
            Tuple of:
                scale_token: (B, d_model) extracted scale summary.
                last_frame: (B, d_model) last frame representation.
        """
        B, T, _ = x.shape

        # 1. Pose input embedding
        embedded = self.input_embed(x)  # (B, T, d_model)

        # 2. Prepend scale token
        scale_tok = self.scale_tokens[scale_idx].expand(B, -1, -1)
        seq = torch.cat([scale_tok, embedded], dim=1)  # (B, 1+T, d_model)

        # 3. Temporal positional encoding
        seq = self.temporal_encoding(seq)

        # 4. Hierarchical temporal transformer
        seq = self.temporal_transformer(seq)

        # 5. Tool-specific projections
        tool1_feat = self.tool1_proj(seq)
        tool2_feat = self.tool2_proj(seq)

        # 6. Memory-enhanced encoder
        seq = self.memory_encoder(seq, tool1_feat, tool2_feat)

        # 7. Bimanual cross-attention
        if self.config.use_cross_attention:
            tool1_out, tool2_out = self.cross_attention(
                tool1_feat, tool2_feat
            )
            cross_out = self.cross_fusion(
                torch.cat([tool1_out, tool2_out], dim=-1)
            )
            seq = seq + cross_out

        # Extract outputs
        scale_token = seq[:, 0, :]   # (B, d_model)
        last_frame = seq[:, -1, :]   # (B, d_model)

        return scale_token, last_frame

    def forward(
        self,
        multi_scale_inputs: list[torch.Tensor],
        force_diagonal: bool = False,
        use_joint_covariance: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for multi-scale kinematic pose prediction.

        Args:
            multi_scale_inputs: List of kinematic tensors for each scale.
                [x_10 (B,10,30), x_50 (B,50,30), x_100 (B,100,30)]
                All windows are causal (end at frame t).
            force_diagonal: Force diagonal covariance (for Cholesky warmup).
            use_joint_covariance: Enable joint 6x6 bimanual Cholesky.

        Returns:
            Dict with probabilistic predictions:
                mu_position (B,2,3), sigma_position (B,2,3),
                mu_quaternion (B,2,4), kappa_quaternion (B,2,1),
                mu_angle (B,2,1), sigma_angle (B,2,1),
                jaw_state_logits (B,2),
                and optionally scale_confidence (B,n_scales).
        """
        assert len(multi_scale_inputs) == len(self.config.window_scales), (
            f"Expected {len(self.config.window_scales)} scales, "
            f"got {len(multi_scale_inputs)}"
        )

        scale_tokens = []
        last_frames = []

        for i, x_scale in enumerate(multi_scale_inputs):
            scale_tok, last_frame = self._encode_scale(x_scale, i)
            scale_tokens.append(scale_tok)
            last_frames.append(last_frame)

        # Cross-scale fusion
        fusion_out = self.scale_fusion(scale_tokens)

        if isinstance(fusion_out, tuple):
            fused, scale_confidence = fusion_out
        else:
            fused = fusion_out
            scale_confidence = None

        # Combine fused representation with finest-scale last frame
        finest_last = last_frames[0]
        combined = torch.cat([fused, finest_last], dim=-1)
        final_repr = self.final_proj(combined)

        # Probabilistic output
        outputs = self.prob_head(
            final_repr,
            force_diagonal=force_diagonal,
            use_joint_covariance=use_joint_covariance,
        )

        if scale_confidence is not None:
            outputs["scale_confidence"] = scale_confidence

        return outputs

    def predict_with_uncertainty(
        self,
        multi_scale_inputs: list[torch.Tensor],
        mc_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict with full uncertainty decomposition via MC Dropout.

        Runs multiple forward passes with MC Dropout enabled to decompose:
        - **Aleatoric** uncertainty: mean of predicted variances (data noise)
        - **Epistemic** uncertainty: variance of predicted means (model uncertainty)
        - **Total** uncertainty: sqrt(aleatoric^2 + epistemic^2)

        Paper: Section 3.1, "Uncertainty Decomposition"
        Equation 9: Aleatoric-epistemic decomposition

        Args:
            multi_scale_inputs: Multi-scale kinematic input tensors.
            mc_samples: Number of MC forward passes (default: config.mc_samples).

        Returns:
            Dict with ensemble-averaged predictions and decomposed
            uncertainties (position_aleatoric_std, position_epistemic_std,
            position_total_std).
        """
        mc_samples = mc_samples or self.config.mc_samples

        all_outputs: list[dict[str, torch.Tensor]] = []
        with enable_mc_dropout(self):
            for _ in range(mc_samples):
                with torch.no_grad():
                    out = self.forward(multi_scale_inputs)
                    all_outputs.append(out)

        result: dict[str, torch.Tensor] = {}

        # Aggregate key predictions
        keys_to_aggregate = [
            "mu_position", "sigma_position", "mu_quaternion",
            "kappa_quaternion", "mu_angle", "sigma_angle",
        ]

        for key in keys_to_aggregate:
            if key not in all_outputs[0]:
                continue
            stacked = torch.stack([o[key] for o in all_outputs], dim=0)
            result[key] = stacked.mean(dim=0)
            result[f"{key}_epistemic_std"] = stacked.std(dim=0)

        # Uncertainty decomposition
        if "mu_position" in result and "sigma_position" in result:
            sigma_stacked = torch.stack(
                [o["sigma_position"] for o in all_outputs], dim=0
            )
            aleatoric_var = (sigma_stacked ** 2).mean(dim=0)

            mu_stacked = torch.stack(
                [o["mu_position"] for o in all_outputs], dim=0
            )
            epistemic_var = mu_stacked.var(dim=0)

            result["position_aleatoric_std"] = torch.sqrt(aleatoric_var)
            result["position_epistemic_std"] = torch.sqrt(epistemic_var)
            result["position_total_std"] = torch.sqrt(
                aleatoric_var + epistemic_var
            )

        # Copy non-aggregated keys
        for key in ["jaw_state_logits", "cholesky_position", "scale_confidence"]:
            if key in all_outputs[0]:
                stacked = torch.stack([o[key] for o in all_outputs], dim=0)
                result[key] = stacked.mean(dim=0)

        return result

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Get parameter count breakdown by component.

        Returns:
            Dict mapping component name to parameter count.
        """
        breakdown: dict[str, int] = {}

        components = {
            "input_embed": self.input_embed,
            "temporal_encoding": self.temporal_encoding,
            "temporal_transformer": self.temporal_transformer,
            "memory_encoder": self.memory_encoder,
            "tool_fusion": self.tool_fusion,
            "scale_fusion": self.scale_fusion,
            "final_proj": self.final_proj,
            "prob_head": self.prob_head,
        }

        if hasattr(self, "cross_attention"):
            components["cross_attention"] = self.cross_attention

        for name, module in components.items():
            breakdown[name] = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

        breakdown["scale_tokens"] = sum(p.numel() for p in self.scale_tokens)

        breakdown["tool_projections"] = (
            sum(p.numel() for p in self.tool1_proj.parameters())
            + sum(p.numel() for p in self.tool2_proj.parameters())
            + (sum(p.numel() for p in self.cross_fusion.parameters())
               if hasattr(self, "cross_fusion") else 0)
        )

        breakdown["total"] = self.get_num_parameters()
        return breakdown


# =============================================================================
# Pivot Point Estimator
# =============================================================================


class PivotPointEstimator(nn.Module):
    """EMA-based trocar (entry point) estimator from predicted tool poses.

    Exploits the fixed-point constraint in laparoscopic surgery: all tool
    shaft lines must pass through the abdominal wall entry point (trocar).
    This module:

    1. Estimates the trocar position from predicted poses using running
       EMA of least-squares shaft line intersections.
    2. Detects physically implausible predictions (shaft misses trocar).
    3. Inflates prediction uncertainty when poses disagree with the
       estimated pivot.
    4. Provides a differentiable consistency loss for training.

    Paper: Section 3.6, "Pivot Point Estimation"
    Figure 6: Trocar constraint geometry

    Args:
        decay: EMA decay factor for pivot updates. Higher values give
            more stable but slower-adapting estimates.
        inflation_threshold: Distance (mm) above which uncertainty
            is inflated.
        max_inflation: Maximum uncertainty inflation factor.
        shaft_axis: Local shaft axis direction in tool frame. Default
            [0, 0, 1] (z-axis) for Aurora-tracked tools.
    """

    def __init__(
        self,
        decay: float = 0.99,
        inflation_threshold: float = 5.0,
        max_inflation: float = 3.0,
        shaft_axis: list[float] | None = None,
    ):
        super().__init__()
        self.decay = decay
        self.inflation_threshold = inflation_threshold
        self.max_inflation = max_inflation

        axis = shaft_axis or [0.0, 0.0, 1.0]
        self.register_buffer(
            "shaft_axis", torch.tensor(axis, dtype=torch.float32)
        )

        # Running pivot estimates (2 tools x 3D)
        self.register_buffer("pivot_estimate", torch.zeros(2, 3))
        self.register_buffer(
            "pivot_variance", torch.ones(2, 3) * 100.0
        )
        self.register_buffer(
            "pivot_count", torch.zeros(2, dtype=torch.long)
        )
        self.register_buffer("initialized", torch.tensor(False))

    @staticmethod
    def _quat_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices.

        Args:
            quaternions: (..., 4) quaternions in [w, x, y, z] format.

        Returns:
            Rotation matrices (..., 3, 3).
        """
        q = F.normalize(quaternions, dim=-1)
        w, x, y, z = q.unbind(-1)

        R = torch.stack([
            torch.stack([
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ], dim=-1),
            torch.stack([
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ], dim=-1),
            torch.stack([
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ], dim=-1),
        ], dim=-2)

        return R

    @torch.no_grad()
    def estimate_pivot_batch(
        self,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate pivot point from a batch of predicted poses.

        Uses least-squares intersection of shaft lines: finds the 3D point
        minimizing total perpendicular distance to all predicted shaft lines.

        The solution is updated via EMA for temporal stability.

        Args:
            positions: Tool positions (B, 2, 3) in mm.
            quaternions: Tool quaternions (B, 2, 4) in [w, x, y, z] format.

        Returns:
            Tuple of:
                pivot: Estimated pivot positions (2, 3) in mm.
                variance: Pivot estimate variance (2, 3) in mm^2.
        """
        B = positions.shape[0]

        R = self._quat_to_rotation_matrix(quaternions)  # (B, 2, 3, 3)
        shaft_local = self.shaft_axis.expand(B, 2, 3).unsqueeze(-1)
        shaft_world = torch.matmul(R, shaft_local).squeeze(-1)
        shaft_world = F.normalize(shaft_world, dim=-1)

        for tool in range(2):
            P = positions[:, tool, :]  # (B, 3)
            d = shaft_world[:, tool, :]  # (B, 3)

            # Build A = sum(I - d_i @ d_i^T), b = sum((I - d_i @ d_i^T) @ P_i)
            I_mat = torch.eye(3, device=P.device).unsqueeze(0).expand(B, -1, -1)
            ddT = torch.bmm(d.unsqueeze(-1), d.unsqueeze(-2))
            proj = I_mat - ddT

            A = proj.sum(dim=0)
            b = torch.bmm(proj, P.unsqueeze(-1)).squeeze(-1).sum(dim=0)

            A_reg = A + 1e-4 * torch.eye(3, device=A.device)
            try:
                pivot_new = torch.linalg.solve(A_reg, b)
            except Exception:
                pivot_new = self.pivot_estimate[tool]

            # EMA update
            if self.pivot_count[tool] == 0:
                self.pivot_estimate[tool] = pivot_new
                self.pivot_variance[tool] = torch.zeros(3, device=P.device)
            else:
                old = self.pivot_estimate[tool]
                self.pivot_estimate[tool] = (
                    self.decay * old + (1 - self.decay) * pivot_new
                )
                diff = pivot_new - self.pivot_estimate[tool]
                self.pivot_variance[tool] = (
                    self.decay * self.pivot_variance[tool]
                    + (1 - self.decay) * diff ** 2
                )

            self.pivot_count[tool] += 1

        return self.pivot_estimate.clone(), self.pivot_variance.clone()

    def compute_pivot_residual(
        self,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perpendicular distance from predictions to estimated pivot.

        A physically valid pose should have near-zero residual (shaft
        passes through trocar).

        Args:
            positions: Tool positions (B, 2, 3) in mm.
            quaternions: Tool quaternions (B, 2, 4) in [w, x, y, z] format.

        Returns:
            Residuals (B, 2) in mm.
        """
        B = positions.shape[0]

        R = self._quat_to_rotation_matrix(quaternions)
        shaft_local = self.shaft_axis.expand(B, 2, 3).unsqueeze(-1)
        shaft_world = torch.matmul(R, shaft_local).squeeze(-1)
        shaft_world = F.normalize(shaft_world, dim=-1)

        pivot = self.pivot_estimate.unsqueeze(0).expand(B, -1, -1)
        v = pivot - positions

        dot = (v * shaft_world).sum(dim=-1, keepdim=True)
        proj = dot * shaft_world
        perp = v - proj

        return torch.norm(perp, dim=-1)

    def inflate_uncertainty(
        self,
        sigma: torch.Tensor,
        kappa: torch.Tensor,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inflate uncertainty when predictions disagree with estimated pivot.

        Increases position sigma and decreases rotation kappa proportionally
        to the shaft-to-pivot distance.

        Args:
            sigma: Position sigma (B, 2, 3) in mm.
            kappa: VMF concentration (B, 2, 1).
            positions: Tool positions (B, 2, 3) in mm.
            quaternions: Tool quaternions (B, 2, 4) in [w, x, y, z] format.

        Returns:
            Tuple of (sigma_inflated, kappa_deflated).
        """
        if not self.initialized or self.pivot_count.min() < 10:
            return sigma, kappa

        residuals = self.compute_pivot_residual(positions, quaternions)

        excess = (residuals - self.inflation_threshold).clamp(min=0)
        factor = 1.0 + excess / self.inflation_threshold * (
            self.max_inflation - 1.0
        )
        factor = factor.clamp(max=self.max_inflation).unsqueeze(-1)

        return sigma * factor.expand_as(sigma), kappa / factor

    def pivot_consistency_loss(
        self,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pivot consistency loss for training.

        Penalizes predictions whose shaft line doesn't pass through
        the estimated trocar. Uses Huber loss for robustness.

        Paper: Section 3.6, Equation 14: Pivot consistency loss

        Args:
            positions: Predicted tool positions (B, 2, 3) in mm.
            quaternions: Predicted tool quaternions (B, 2, 4).
            sigma: Predicted position sigma (B, 2, 3), optional.
                When provided, loss is down-weighted for uncertain predictions.

        Returns:
            Scalar pivot consistency loss.
        """
        residuals = self.compute_pivot_residual(positions, quaternions)

        pivot_weight = 1.0 / (self.pivot_variance.norm(dim=-1) + 1.0)
        weighted = residuals * pivot_weight.unsqueeze(0)

        if sigma is not None:
            sigma_mean = sigma.mean(dim=-1)
            uncertainty_weight = 1.0 / (sigma_mean + 0.1)
            weighted = weighted * uncertainty_weight

        return F.huber_loss(weighted, torch.zeros_like(weighted), delta=5.0)

    @property
    def is_initialized(self) -> bool:
        """Check whether pivot estimates have accumulated enough samples.

        Returns:
            True if both tools have at least 10 estimates.
        """
        return bool(self.pivot_count.min() >= 10)

    def mark_initialized(self) -> None:
        """Mark the estimator as initialized for uncertainty inflation."""
        self.initialized = torch.tensor(
            True, device=self.pivot_estimate.device
        )

    def reset(self) -> None:
        """Reset all pivot estimates to initial state."""
        self.pivot_estimate.zero_()
        self.pivot_variance.fill_(100.0)
        self.pivot_count.zero_()
        self.initialized.fill_(False)


# =============================================================================
# Full BTPN Model
# =============================================================================


class BTPN(nn.Module):
    """Bayesian Temporal Pose Network -- full multimodal architecture.

    Combines a frozen kinematic foundation model with trainable visual
    feature encoding, clinical attention, kinematic-visual fusion, and
    multi-channel gated residual correction.

    Training follows a 2-stage pipeline:

    **Stage 1 (SSL):** Train visual projections and clinical attention
    encoder with self-supervised tasks (masked reconstruction, contrastive
    learning, temporal order prediction, kinematic alignment).

    **Stage 2 (Supervised):** Three-phase training:
    - Warmup (0-20 epochs): Only fusion + gate + heads trainable.
    - Full (20-150 epochs): All visual components trainable.
    - Finetune (150+ epochs): Reduced LR, pivot constraint active.

    Paper: Section 3 (full architecture), Figure 2

    Checkpoint compatibility:
        Renamed from VisualTemporalBTPNv3. Internal attribute names are
        preserved exactly:
        - ``self.kinematic_prior``: KinematicFoundationModel instance
          (was self.kinematic_prior in V3, holding BTPNv5)
        - ``self.clinical_encoder``: ClinicalAttentionEncoder
        - ``self.confidence_gate``: ConfidenceGate (was MultiChannelConfidenceGate)
        - ``self.residual_head``: ResidualPoseHead (was ResidualPoseHeadV3)
        - ``self.seg_neck_proj``, ``self.seg_backbone_proj``, ``self.seg_fusion``
        - ``self.depth_proj``, ``self.scene_fusion``
        - ``self.pose_kp_proj``, ``self.pose_enc_proj``, ``self.pose_geo_proj``,
          ``self.pose_fusion``
        - ``self.visual_proj``, ``self.kin_embed_proj``, ``self.kv_fusion``
        - ``self.pivot_estimator``

    Args:
        config: Model configuration.
        kinematic_model: Pre-loaded kinematic foundation model. If None,
            creates a new KinematicFoundationModel from config.
    """

    def __init__(
        self,
        config: BTPNConfig,
        kinematic_model: KinematicFoundationModel | None = None,
    ):
        super().__init__()
        self.config = config

        # ---- Stage 0: Frozen Kinematic Prior ----
        # NOTE: Attribute name 'kinematic_prior' matches VisualTemporalBTPNv3
        if kinematic_model is not None:
            self.kinematic_prior = kinematic_model
        else:
            self.kinematic_prior = KinematicFoundationModel(config)

        if config.freeze_kinematic:
            for param in self.kinematic_prior.parameters():
                param.requires_grad = False
            self.kinematic_prior.eval()

        # ---- Stage 1: Visual Feature Encoding ----
        # Segmentation branch: three separate attributes for V3 checkpoint compat.
        # V3 uses SegNeckProjection, SegBackboneProjection, LearnableSegFusion
        # as three separate self.xxx attributes. We preserve that layout.
        self.seg_neck_proj = _SegNeckProjection(
            neck_dim=config.seg_neck_dim,
            proj_dim=config.seg_proj_dim,
        )
        self.seg_backbone_proj = _SegBackboneProjection(
            backbone_dim=config.seg_backbone_dim,
            proj_dim=config.depth_proj_dim,
        )
        self.seg_fusion = _LearnableSegFusion(
            neck_proj_dim=config.seg_proj_dim,
            backbone_proj_dim=config.depth_proj_dim,
            output_dim=config.seg_proj_dim,
        )

        # Depth branch (optional for ablation)
        self._use_depth = config.use_depth_features
        if self._use_depth:
            self.depth_proj = DepthProjection(
                depth_dim=config.depth_feature_dim,
                proj_dim=config.depth_proj_dim,
            )
            self.scene_fusion = SceneFusion(
                seg_dim=config.seg_proj_dim,
                depth_dim=config.depth_proj_dim,
                output_dim=config.visual_d_model,
            )

        # Pose branch (optional): separate attributes for V3 checkpoint compat.
        # V3 uses PoseKeypointProjection, PoseEncoderProjection,
        # PoseGeometricProjection, LearnablePoseFusion as four separate attrs.
        self._use_pose = config.use_pose_features
        if self._use_pose:
            self.pose_kp_proj = _PoseKeypointProjection(
                n_keypoints=8,
                proj_dim=config.pose_kp_proj_dim,
            )
            self.pose_enc_proj = PoseProjection(
                backbone_dim=config.pose_backbone_dim,
                proj_dim=config.pose_enc_proj_dim,
                dropout=config.visual_dropout,
            )
            self.pose_geo_proj = _PoseGeometricProjection(
                input_dim=config.pose_geometric_dim,
                proj_dim=32,
            )
            self.pose_fusion = _LearnablePoseFusion(
                kp_proj_dim=config.pose_kp_proj_dim,
                enc_proj_dim=config.pose_enc_proj_dim,
                geo_proj_dim=32,
                output_dim=config.pose_proj_dim,
            )

        # Visual projection: scene/seg + pose -> d_model
        scene_out_dim = (
            config.visual_d_model
            if self._use_depth
            else config.seg_proj_dim
        )
        visual_input_dim = scene_out_dim + (
            config.pose_proj_dim if self._use_pose else 0
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_input_dim, config.visual_d_model),
            nn.LayerNorm(config.visual_d_model),
            nn.GELU(),
        )

        # ---- Stage 2: Clinical Attention Encoder ----
        # NOTE: Attribute name 'clinical_encoder' matches VisualTemporalBTPNv3
        self.clinical_encoder = ClinicalAttentionEncoder(
            d_model=config.visual_d_model,
            n_layers=config.visual_n_layers,
            n_heads=config.visual_n_heads,
            dropout=config.visual_dropout,
            local_window=config.visual_local_window,
            medium_window=config.visual_medium_window,
        )

        # ---- Stage 3: Kinematic-Visual Fusion ----
        # NOTE: Attribute names match VisualTemporalBTPNv3
        self.kin_embed_proj = nn.Linear(
            config.d_model, config.visual_d_model
        )
        self.kv_fusion = KinematicVisualFusion(
            d_model=config.visual_d_model,
            n_heads=config.cross_modal_heads,
            dropout=config.cross_modal_dropout,
        )

        # ---- Stage 4: Multi-Channel Gates + Residual Heads ----
        # NOTE: Attribute names match VisualTemporalBTPNv3
        self.confidence_gate = ConfidenceGate(
            hidden_dim=config.gate_hidden_dim,
            max_gate_pos=config.max_gate_position,
            max_gate_rot=config.max_gate_rotation,
            max_gate_angle=config.max_gate_angle,
            init_temperature=config.gate_init_temperature,
        )
        self.residual_head = ResidualPoseHead(
            d_model=config.visual_d_model,
            use_relative_tracking=config.use_relative_tracking,
            max_pos_delta=config.max_position_delta,
            max_quat_delta=config.max_quaternion_delta,
            max_angle_delta=config.max_angle_delta,
            max_displacement_delta=config.max_displacement_delta,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
            min_kappa=config.min_kappa,
        )

        # ---- Pivot Point Estimation (optional) ----
        if config.use_pivot_estimation:
            self.pivot_estimator = PivotPointEstimator(
                decay=config.pivot_decay,
                inflation_threshold=config.pivot_inflation_threshold,
                max_inflation=config.pivot_max_inflation,
            )

        self._init_visual_weights()

    def _init_visual_weights(self) -> None:
        """Initialize visual branch weights with Xavier uniform.

        Only initializes visual components, not the frozen kinematic prior.
        """
        visual_modules = [
            self.visual_proj,
            self.clinical_encoder,
            self.kin_embed_proj,
            self.kv_fusion,
            self.confidence_gate,
            self.residual_head,
        ]
        if self._use_depth:
            visual_modules.insert(0, self.scene_fusion)
        for module in visual_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _encode_visual_scale(
        self,
        seg_neck: torch.Tensor,
        seg_backbone: torch.Tensor,
        depth: torch.Tensor,
        pose_kp: torch.Tensor | None = None,
        pose_backbone: torch.Tensor | None = None,
        pose_geometric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode visual features for one temporal scale.

        Processes segmentation, depth, and pose features through their
        respective projection modules, then fuses them into a single
        visual representation per frame.

        Args:
            seg_neck: (B, T, 2, 256) per-tool FPN neck features.
            seg_backbone: (B, T, 512) global backbone features.
            depth: (B, T, 384) depth embeddings.
            pose_kp: (B, T, 2, 8, 3) per-tool keypoints, optional.
            pose_backbone: (B, T, 2, 512) per-tool backbone, optional.
            pose_geometric: (B, T, 6) geometric features, optional.

        Returns:
            Visual features (B, T, visual_d_model).
        """
        # Segmentation: three separate calls matching V3 checkpoint layout
        neck_proj = self.seg_neck_proj(seg_neck)
        bb_proj = self.seg_backbone_proj(seg_backbone)
        seg_fused = self.seg_fusion(neck_proj, bb_proj)

        # Scene: depth-dependent path
        if self._use_depth:
            depth_proj = self.depth_proj(depth)
            scene = self.scene_fusion(seg_fused, depth_proj)
        else:
            scene = seg_fused

        # Pose: four separate calls matching V3 checkpoint layout
        if self._use_pose and pose_kp is not None:
            kp_proj = self.pose_kp_proj(pose_kp)
            enc_proj = self.pose_enc_proj(pose_backbone)
            geo_proj = self.pose_geo_proj(pose_geometric)
            pose_fused = self.pose_fusion(kp_proj, enc_proj, geo_proj)
            visual = torch.cat([scene, pose_fused], dim=-1)
        else:
            visual = scene

        return self.visual_proj(visual)

    def _extract_kinematic_embedding(
        self, kinematic_windows: list[torch.Tensor]
    ) -> torch.Tensor:
        """Extract intermediate embedding from the frozen kinematic prior.

        Re-runs the kinematic model's encode pipeline to get scale tokens,
        then averages them as the kinematic embedding.

        Args:
            kinematic_windows: List of kinematic tensors per scale.

        Returns:
            Kinematic embedding (B, d_model).
        """
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            scale_tokens = []
            for i, kin_input in enumerate(kinematic_windows):
                scale_tok, _ = self.kinematic_prior._encode_scale(
                    kin_input.float(), i
                )
                scale_tokens.append(scale_tok)

        return torch.stack(scale_tokens, dim=0).mean(dim=0)

    def forward(
        self,
        kinematic_windows: list[torch.Tensor],
        seg_neck_windows: list[torch.Tensor],
        seg_backbone_windows: list[torch.Tensor],
        depth_windows: list[torch.Tensor],
        detection_conf: list[torch.Tensor] | None = None,
        visual_valid_mask: list[torch.Tensor] | None = None,
        pose_kp_windows: list[torch.Tensor] | None = None,
        pose_backbone_windows: list[torch.Tensor] | None = None,
        pose_geometric_windows: list[torch.Tensor] | None = None,
        pose_conf: list[torch.Tensor] | None = None,
        visual_warmup: float = 1.0,
        current_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: kinematic prior + visual residual correction.

        Args:
            kinematic_windows: List of kinematic tensors per scale.
                [Tensor(B, 10, 30), Tensor(B, 50, 30), Tensor(B, 100, 30)]
            seg_neck_windows: List of per-tool FPN features per scale.
            seg_backbone_windows: List of global backbone features per scale.
            depth_windows: List of depth embeddings per scale.
            detection_conf: List of per-tool detection conf per scale.
            visual_valid_mask: List of bool masks per scale.
            pose_kp_windows: List of per-tool keypoints per scale, optional.
            pose_backbone_windows: List of per-tool backbone per scale, optional.
            pose_geometric_windows: List of geometric features per scale, optional.
            pose_conf: List of per-tool pose conf per scale, optional.
            visual_warmup: Warmup weight for visual branch [0, 1].
            current_position: (B, 2, 3) current frame position for
                displacement prediction.

        Returns:
            Dict with corrected predictions, kinematic prior outputs,
            gate values, and optional displacement/pivot outputs.
        """
        # ---- Stage 0: Kinematic Prior ----
        # Run in float32 to avoid attention NaN under float16
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            self.kinematic_prior.eval()
            kin_windows_fp32 = [w.float() for w in kinematic_windows]
            kin_outputs = self.kinematic_prior(
                multi_scale_inputs=kin_windows_fp32,
                force_diagonal=True,
            )

        kin_embedding = self._extract_kinematic_embedding(kinematic_windows)

        # ---- Stage 1 & 2: Visual Encoding ----
        # Use finest scale (scale 0) for visual clinical encoder
        visual_feats = self._encode_visual_scale(
            seg_neck=seg_neck_windows[0],
            seg_backbone=seg_backbone_windows[0],
            depth=depth_windows[0],
            pose_kp=(
                pose_kp_windows[0] if pose_kp_windows else None
            ),
            pose_backbone=(
                pose_backbone_windows[0] if pose_backbone_windows else None
            ),
            pose_geometric=(
                pose_geometric_windows[0]
                if pose_geometric_windows
                else None
            ),
        )

        visual_repr = self.clinical_encoder(visual_feats)

        # Apply visual warmup
        if visual_warmup < 1.0:
            visual_repr = visual_repr * visual_warmup

        # ---- Stage 3: Kinematic-Visual Fusion ----
        kin_proj = self.kin_embed_proj(kin_embedding)
        fused_repr = self.kv_fusion(kin_proj, visual_repr)

        # ---- Stage 4: Multi-Channel Gates + Residual Heads ----
        if detection_conf is not None and len(detection_conf) > 0:
            last_conf = detection_conf[0][:, -1, :]  # (B, 2)
        else:
            last_conf = torch.ones(
                fused_repr.shape[0], 2, device=fused_repr.device
            )

        gates = self.confidence_gate(last_conf)

        outputs = self.residual_head(
            fused_repr,
            kin_outputs,
            gates,
            current_position=current_position,
        )

        # Kinematic prior outputs for comparison/logging
        outputs["kin_mu_position"] = kin_outputs["mu_position"]
        outputs["kin_mu_quaternion"] = kin_outputs["mu_quaternion"]
        outputs["kin_mu_angle"] = kin_outputs["mu_angle"]

        # Gate entropy loss for anti-saturation regularization
        outputs["gate_entropy"] = gate_entropy_loss(gates)
        outputs["gate_temperature"] = (
            self.confidence_gate.log_temperature.exp()
        )

        # ---- Pivot Estimation ----
        if hasattr(self, "pivot_estimator"):
            if self.training:
                self.pivot_estimator.estimate_pivot_batch(
                    positions=outputs["mu_position"].detach(),
                    quaternions=outputs["mu_quaternion"].detach(),
                )
                if self.pivot_estimator.is_initialized:
                    self.pivot_estimator.mark_initialized()

            if self.pivot_estimator.is_initialized:
                pivot_residual = self.pivot_estimator.compute_pivot_residual(
                    positions=outputs["mu_position"],
                    quaternions=outputs["mu_quaternion"],
                )
                outputs["pivot_residual"] = pivot_residual

                sigma_inflated, kappa_deflated = (
                    self.pivot_estimator.inflate_uncertainty(
                        sigma=outputs["sigma_position"],
                        kappa=outputs["kappa_quaternion"],
                        positions=outputs["mu_position"],
                        quaternions=outputs["mu_quaternion"],
                    )
                )
                outputs["sigma_position_pre_pivot"] = outputs[
                    "sigma_position"
                ]
                outputs["kappa_quaternion_pre_pivot"] = outputs[
                    "kappa_quaternion"
                ]
                outputs["sigma_position"] = sigma_inflated
                outputs["kappa_quaternion"] = kappa_deflated

        return outputs

    # =====================================================================
    # Training Phase Configuration
    # =====================================================================

    def set_training_phase(self, phase: str) -> None:
        """Configure model for a specific training phase.

        Controls which parameters are trainable:
        - **warmup**: Only visual projections trainable (fusion, encoder,
          gate, and heads frozen).
        - **full**: All visual components trainable (kinematic prior frozen).
        - **finetune**: Same as full (caller reduces LR externally).

        The kinematic prior is always frozen regardless of phase.

        Args:
            phase: "warmup", "full", or "finetune".

        Raises:
            ValueError: If phase is not recognized.
        """
        if phase not in ("warmup", "full", "finetune"):
            raise ValueError(
                f"Unknown training phase '{phase}'. "
                "Expected 'warmup', 'full', or 'finetune'."
            )

        if phase == "warmup":
            for name, param in self.named_parameters():
                if name.startswith("kinematic_prior"):
                    param.requires_grad = False
                elif any(name.startswith(p) for p in [
                    "kv_fusion", "residual_head", "confidence_gate",
                    "clinical_encoder",
                ]):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif phase in ("full", "finetune"):
            for name, param in self.named_parameters():
                if name.startswith("kinematic_prior"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def set_ssl_mode(self, enabled: bool) -> None:
        """Enable or disable SSL mode for stage 1 visual pre-training.

        When enabled, freezes everything except visual projection modules
        and the clinical encoder. When disabled, unfreezes all except the
        kinematic prior.

        Args:
            enabled: True to enter SSL mode, False to exit.
        """
        ssl_prefixes = [
            "seg_neck_proj", "seg_backbone_proj", "seg_fusion",
            "visual_proj", "clinical_encoder",
        ]
        if self._use_depth:
            ssl_prefixes.extend(["depth_proj", "scene_fusion"])
        if self._use_pose:
            ssl_prefixes.extend([
                "pose_kp_proj", "pose_enc_proj", "pose_geo_proj",
                "pose_fusion",
            ])

        for name, param in self.named_parameters():
            if name.startswith("kinematic_prior"):
                param.requires_grad = False
            elif enabled:
                param.requires_grad = any(
                    name.startswith(p) for p in ssl_prefixes
                )
            else:
                param.requires_grad = True

    # =====================================================================
    # Weight Loading
    # =====================================================================

    def load_stage1_weights(self, checkpoint_path: str | Path) -> None:
        """Load stage 1 SSL visual encoder weights from a checkpoint.

        Loads weights for visual projection modules and the clinical
        encoder. Handles shape mismatches gracefully with warnings.

        Args:
            checkpoint_path: Path to stage 1 checkpoint file (.pt).
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remap V2-style keys to V3-style keys
        remapped_state: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("visual_temporal_encoder."):
                new_key = new_key.replace(
                    "visual_temporal_encoder.", "clinical_encoder.", 1
                )
                new_key = new_key.replace(
                    "clinical_encoder.encoder.", "clinical_encoder.", 1
                )
            remapped_state[new_key] = v
        state_dict = remapped_state

        visual_prefixes = [
            "seg_neck_proj.", "seg_backbone_proj.", "seg_fusion.",
            "visual_proj.", "clinical_encoder.",
        ]
        if self._use_depth:
            visual_prefixes.extend(["depth_proj.", "scene_fusion."])
        if self._use_pose:
            visual_prefixes.extend([
                "pose_kp_proj.", "pose_enc_proj.", "pose_geo_proj.",
                "pose_fusion.",
            ])

        visual_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in visual_prefixes)
        }

        # Filter shape-mismatched keys
        model_state = self.state_dict()
        shape_skipped = []
        compatible_state = {}
        for k, v in visual_state.items():
            if k in model_state and model_state[k].shape != v.shape:
                shape_skipped.append(k)
            else:
                compatible_state[k] = v
        if shape_skipped:
            warnings.warn(
                f"Skipping {len(shape_skipped)} stage 1 keys with shape "
                f"mismatch: {shape_skipped[:5]}"
            )
        visual_state = compatible_state

        missing, unexpected = self.load_state_dict(
            visual_state, strict=False
        )

        truly_missing = [
            k for k in missing
            if any(k.startswith(p) for p in visual_prefixes)
        ]

        if truly_missing:
            warnings.warn(
                f"Missing visual encoder keys from stage 1 checkpoint: "
                f"{truly_missing[:5]}"
                f"{'...' if len(truly_missing) > 5 else ''}"
            )

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str | Path,
        config: BTPNConfig,
        map_location: str | torch.device = "cpu",
    ) -> BTPN:
        """Load a pre-trained BTPN model from checkpoint.

        Creates a new BTPN instance with the given config and loads
        the full model state dict from the checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file).
            config: Model configuration (must match checkpoint architecture).
            map_location: Device to map checkpoint tensors to.

        Returns:
            Loaded BTPN model in eval mode.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

        model = cls(config)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def predict(
        self,
        trial_data: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Easy inference interface for a single trial.

        Wraps the forward pass with no_grad context and handles
        the common case of passing all modalities in a single dict.

        Args:
            trial_data: Dict containing at minimum:
                - kinematic_windows: list[Tensor] multi-scale kinematic input
                - seg_neck_windows: list[Tensor] segmentation neck features
                - seg_backbone_windows: list[Tensor] segmentation backbone
                - depth_windows: list[Tensor] depth embeddings
                Optionally:
                - detection_conf: list[Tensor] detection confidences
                - pose_kp_windows: list[Tensor] keypoint detections
                - pose_backbone_windows: list[Tensor] pose backbone features
                - pose_geometric_windows: list[Tensor] geometric features
                - current_position: Tensor current frame positions

        Returns:
            Dict with all model predictions. See forward() for details.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                kinematic_windows=trial_data["kinematic_windows"],
                seg_neck_windows=trial_data["seg_neck_windows"],
                seg_backbone_windows=trial_data["seg_backbone_windows"],
                depth_windows=trial_data["depth_windows"],
                detection_conf=trial_data.get("detection_conf"),
                visual_valid_mask=trial_data.get("visual_valid_mask"),
                pose_kp_windows=trial_data.get("pose_kp_windows"),
                pose_backbone_windows=trial_data.get(
                    "pose_backbone_windows"
                ),
                pose_geometric_windows=trial_data.get(
                    "pose_geometric_windows"
                ),
                pose_conf=trial_data.get("pose_conf"),
                current_position=trial_data.get("current_position"),
            )

    # =====================================================================
    # Parameter Counting
    # =====================================================================

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters (excluding frozen kinematic prior).

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Get parameter count by component.

        Returns:
            Dict mapping component name to parameter count, including
            frozen kinematic prior count and total trainable.
        """
        components: dict[str, list[str]] = {
            "seg_projections": [
                "seg_neck_proj", "seg_backbone_proj", "seg_fusion",
            ],
            "visual_proj": ["visual_proj"],
            "clinical_encoder": ["clinical_encoder"],
            "kin_embed_proj": ["kin_embed_proj"],
            "kv_fusion": ["kv_fusion"],
            "confidence_gate": ["confidence_gate"],
            "residual_head": ["residual_head"],
        }

        if self._use_depth:
            components["depth_projection"] = ["depth_proj"]
            components["scene_fusion"] = ["scene_fusion"]

        if self._use_pose:
            components["pose_projections"] = [
                "pose_kp_proj", "pose_enc_proj", "pose_geo_proj",
                "pose_fusion",
            ]

        if hasattr(self, "pivot_estimator"):
            components["pivot_estimator"] = ["pivot_estimator"]

        counts: dict[str, int] = {}
        for name, prefixes in components.items():
            count = sum(
                p.numel()
                for n, p in self.named_parameters()
                if any(n.startswith(pf) for pf in prefixes)
                and p.requires_grad
            )
            if count > 0:
                counts[name] = count

        counts["kinematic_prior_frozen"] = sum(
            p.numel() for p in self.kinematic_prior.parameters()
        )
        counts["total_trainable"] = self.get_trainable_parameters()
        counts["total_all"] = sum(p.numel() for p in self.parameters())

        return counts
