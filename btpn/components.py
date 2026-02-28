"""Reusable neural network components for the BTPN architecture.

This module contains all building-block modules used by the main BTPN
model. Every class preserves its original attribute names for checkpoint
compatibility with the research codebase.

Components (in dependency order):

    Utility:
        MCDropout             -- Monte Carlo Dropout (always active)
        get_activation        -- Activation factory

    Positional Encoding:
        LearnableTemporalEncoding  -- Sinusoidal + learnable + time-delta

    Input Embedding:
        QuaternionEmbedding   -- Tangent-space-aware quaternion embedding
        PoseInputEmbedding    -- Specialized per-component pose embedding

    Temporal Processing:
        HierarchicalTemporalTransformer  -- Multi-scale attention (Section 3.2)
        CrossScaleAttention              -- Bidirectional inter-scale attention

    Bimanual Coordination:
        BimanualCrossAttention  -- Gated Tool1 <-> Tool2 attention (Section 3.3)

    Memory:
        MemoryBank                   -- Learnable motion prototypes
        MemoryAttentiveFusion        -- Gated memory read
        BidirectionalTemporalContext -- Past / future context splitting
        GatedToolFusion              -- Dynamic tool reliability weighting
        MemoryEnhancedEncoder        -- Combined memory encoder (Section 3.2)

    Output Heads:
        MixtureDensityHead     -- MDN with K Gaussian components
        ProbabilisticPoseHead  -- Position/Rotation/Angle/JawState (Section 3.5)

    Cross-Scale:
        CrossScaleFusion       -- Multi-scale token fusion (Section 3.4)

    Masking:
        create_masking         -- Frame / feature masking for SSL

Paper: "Bayesian Temporal Pose Network for Surgical Tool Tracking"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BTPNFeatureConfig


# =============================================================================
# Utility Modules
# =============================================================================


class MCDropout(nn.Dropout):
    """Monte Carlo Dropout -- remains active during inference.

    Standard dropout is disabled at eval time, but MC Dropout stays
    on so that multiple stochastic forward passes can be used to
    estimate epistemic (model) uncertainty.

    Paper Section 3.5: Epistemic uncertainty via MC Dropout.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout unconditionally (even in eval mode).

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with elements randomly zeroed with probability ``p``.
        """
        return F.dropout(x, self.p, training=True, inplace=self.inplace)


def get_activation(name: str) -> nn.Module:
    """Get an activation module by name.

    Args:
        name: One of ``"gelu"``, ``"relu"``, ``"silu"``.

    Returns:
        Corresponding ``nn.Module``. Defaults to GELU.
    """
    activations: dict[str, nn.Module] = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    return activations.get(name.lower(), nn.GELU())


# =============================================================================
# Positional Encoding
# =============================================================================


class LearnableTemporalEncoding(nn.Module):
    """Learnable temporal positional encoding with time-delta support.

    Combines three sources of positional information:
        1. Fixed sinusoidal encoding (absolute position awareness).
        2. Learnable position embeddings (data-adaptive offsets).
        3. Optional time-delta MLP (handles irregular ~13 fps sampling).

    Paper Section 3.2: Temporal positional encoding.

    Attributes:
        d_model: Model hidden dimension.
        max_len: Maximum supported sequence length.
        use_time_delta: Whether the time-delta MLP is active.
        pe: Fixed sinusoidal encoding buffer ``(1, max_len, d_model)``.
        pos_embed: Learnable position parameter ``(1, max_len, d_model)``.
        time_mlp: Optional MLP mapping ``(B, T, 1)`` to ``(B, T, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
        dropout: float = 0.1,
        use_time_delta: bool = True,
    ) -> None:
        """Initialize temporal encoding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
            use_time_delta: Whether to encode inter-frame time differences.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_time_delta = use_time_delta
        self.dropout = nn.Dropout(dropout)

        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        # Time-delta encoding for irregular sampling
        if use_time_delta:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add positional encoding to the input.

        Args:
            x: Input tensor ``(B, T, D)``.
            time_deltas: Optional time differences ``(B, T)``.

        Returns:
            Positionally encoded tensor ``(B, T, D)``.
        """
        seq_len = x.size(1)

        # Add fixed + learnable position encoding
        x = x + self.pe[:, :seq_len, :] + self.pos_embed[:, :seq_len, :]

        # Add time-delta encoding if provided
        if self.use_time_delta and time_deltas is not None:
            time_embed = self.time_mlp(time_deltas.unsqueeze(-1))
            x = x + time_embed

        return self.dropout(x)


# =============================================================================
# Quaternion Embedding
# =============================================================================


class QuaternionEmbedding(nn.Module):
    """Quaternion-aware embedding that respects the S^3 manifold.

    Maps quaternions to a latent space using two complementary paths:
        1. Direct 4D -> embed_dim linear path.
        2. Tangent-space projection (log map at identity) for gradient
           stability, producing embed_dim // 2 features.

    The final output concatenates the tangent half with the second half
    of the direct embedding.

    Paper Section 3.2: Quaternion embedding with tangent-space projection.

    Attributes:
        embed_dim: Output embedding dimension.
        quat_embed: Direct quaternion MLP.
        tangent_embed: Tangent-space MLP.
    """

    def __init__(self, embed_dim: int = 64, dropout: float = 0.1) -> None:
        """Initialize quaternion embedding.

        Args:
            embed_dim: Output embedding dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Direct embedding path
        self.quat_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Tangent-space projection path
        self.tangent_embed = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )

    @staticmethod
    def _quat_to_tangent(q: torch.Tensor) -> torch.Tensor:
        """Project quaternion to tangent space at identity via log map.

        Args:
            q: Quaternion ``(..., 4)`` in ``[w, x, y, z]`` order.

        Returns:
            Tangent vector ``(..., 3)``.
        """
        q = F.normalize(q, dim=-1)

        w = q[..., 0:1]
        xyz = q[..., 1:4]

        # Resolve sign ambiguity (q and -q represent the same rotation)
        sign = torch.sign(w + 1e-8)
        w = w * sign
        xyz = xyz * sign

        # Log map: rotation vector = axis * angle
        angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
        axis_norm = torch.norm(xyz, dim=-1, keepdim=True) + 1e-8
        axis = xyz / axis_norm

        return axis * angle

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Embed quaternion to latent space.

        Args:
            q: Quaternion tensor ``(..., 4)``.

        Returns:
            Embedding ``(..., embed_dim)``.
        """
        q = F.normalize(q, dim=-1)

        direct = self.quat_embed(q)

        tangent = self._quat_to_tangent(q)
        tangent_emb = self.tangent_embed(tangent)

        # First half from tangent, second half from direct
        combined = torch.cat(
            [tangent_emb, direct[..., self.embed_dim // 2 :]], dim=-1
        )
        return combined


# =============================================================================
# Pose Input Embedding
# =============================================================================


class PoseInputEmbedding(nn.Module):
    """Embed raw 30D kinematic features into a rich representation space.

    Uses specialized sub-embeddings for each component type:
        - Positions (3D)  -> 64D via MLP
        - Quaternions (4D) -> 64D via QuaternionEmbedding
        - Jaw angles (1D) -> 32D via MLP

    The concatenated embeddings (576D total) are projected to ``d_model``.

    Paper Section 3.2: Pose input embedding.

    Attributes:
        pos_embed: Position MLP (3 -> 64).
        quat_embed: QuaternionEmbedding (4 -> 64).
        angle_embed: Angle MLP (1 -> 32).
        feature_config: Index mapping for 30D features.
        projection: Final linear projection to d_model.
    """

    def __init__(
        self,
        input_dim: int = 30,
        d_model: int = 256,
        dropout: float = 0.1,
        derived_feature_dim: int = 0,
    ) -> None:
        """Initialize pose embedding.

        Args:
            input_dim: Raw input dimension (30 for 7DOF).
            d_model: Output model dimension.
            dropout: Dropout probability.
            derived_feature_dim: Extra feature dimension appended after
                the core 30D. When > 0, these are projected separately
                and summed into the output.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.derived_feature_dim = derived_feature_dim

        # Position embedding (3D -> 64D)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Quaternion embedding (4D -> 64D)
        self.quat_embed = QuaternionEmbedding(embed_dim=64, dropout=dropout)

        # Angle embedding (1D -> 32D)
        self.angle_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Feature configuration for 7DOF
        self.feature_config = BTPNFeatureConfig()

        # Embedded dim:
        # Tool1: pos(64) + quat(64) + angle(32) = 160
        # Tool2: pos(64) + quat(64) + angle(32) = 160
        # Camera: pos(64) + quat(64) = 128
        # World:  pos(64) + quat(64) = 128
        # Total: 576
        embed_dim = 2 * (64 + 64 + 32) + 2 * (64 + 64)  # = 576

        # Projection to d_model
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optional derived-feature projection
        if derived_feature_dim > 0:
            self.derived_proj = nn.Sequential(
                nn.Linear(derived_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed pose features.

        Args:
            x: Raw pose tensor ``(B, T, input_dim)`` or ``(B, input_dim)``.
                First 30 dims are specialized pose features; remaining
                dims (if any) are derived features.

        Returns:
            Embedded tensor ``(B, T, d_model)`` or ``(B, d_model)``.
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Split raw pose (30D) from optional derived features
        x_pose = x[..., :30]
        x_derived = x[..., 30:] if self.derived_feature_dim > 0 else None

        cfg = self.feature_config
        embeddings: list[torch.Tensor] = []

        # Tool 1: Position + Quaternion + Angle
        t1_pos = self.pos_embed(x_pose[..., cfg.tool1_pos[0]:cfg.tool1_pos[1]])
        t1_quat = self.quat_embed(x_pose[..., cfg.tool1_quat[0]:cfg.tool1_quat[1]])
        t1_angle = self.angle_embed(x_pose[..., cfg.tool1_angle:cfg.tool1_angle + 1])
        embeddings.extend([t1_pos, t1_quat, t1_angle])

        # Tool 2: Position + Quaternion + Angle
        t2_pos = self.pos_embed(x_pose[..., cfg.tool2_pos[0]:cfg.tool2_pos[1]])
        t2_quat = self.quat_embed(x_pose[..., cfg.tool2_quat[0]:cfg.tool2_quat[1]])
        t2_angle = self.angle_embed(x_pose[..., cfg.tool2_angle:cfg.tool2_angle + 1])
        embeddings.extend([t2_pos, t2_quat, t2_angle])

        # Camera: Position + Quaternion
        cam_pos = self.pos_embed(x_pose[..., cfg.camera_pos[0]:cfg.camera_pos[1]])
        cam_quat = self.quat_embed(x_pose[..., cfg.camera_quat[0]:cfg.camera_quat[1]])
        embeddings.extend([cam_pos, cam_quat])

        # World: Position + Quaternion
        world_pos = self.pos_embed(x_pose[..., cfg.world_pos[0]:cfg.world_pos[1]])
        world_quat = self.quat_embed(x_pose[..., cfg.world_quat[0]:cfg.world_quat[1]])
        embeddings.extend([world_pos, world_quat])

        # Concatenate all sub-embeddings and project
        embedded = torch.cat(embeddings, dim=-1)
        output = self.projection(embedded)

        # Add derived feature embedding if present
        if x_derived is not None and hasattr(self, "derived_proj"):
            output = output + self.derived_proj(x_derived)

        if squeeze:
            output = output.squeeze(1)

        return output


# =============================================================================
# Cross-Scale Attention
# =============================================================================


class CrossScaleAttention(nn.Module):
    """Bidirectional cross-attention between temporal scales.

    Enables information flow between local, medium, and global
    representations so that each scale can attend to the others.
    Gated residual connections control how much cross-scale information
    is incorporated.

    Paper Section 3.2: Cross-scale attention within the HTT.

    Attributes:
        local_to_medium, local_to_global: Local queries other scales.
        medium_to_local, medium_to_global: Medium queries other scales.
        global_to_local, global_to_medium: Global queries other scales.
        gate_local, gate_medium, gate_global: Gating networks.
        norm_local, norm_medium, norm_global: Layer normalizations.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize cross-scale attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        # Local cross-attentions
        self.local_to_medium = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.local_to_global = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Medium cross-attentions
        self.medium_to_local = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.medium_to_global = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Global cross-attentions
        self.global_to_local = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.global_to_medium = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Gating mechanisms
        self.gate_local = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.Sigmoid(),
        )
        self.gate_medium = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.Sigmoid(),
        )
        self.gate_global = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.Sigmoid(),
        )

        # Layer norms
        self.norm_local = nn.LayerNorm(d_model)
        self.norm_medium = nn.LayerNorm(d_model)
        self.norm_global = nn.LayerNorm(d_model)

    def forward(
        self,
        x_local: torch.Tensor,
        x_medium: torch.Tensor,
        x_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-scale attention.

        Args:
            x_local: Local features ``(B, T, D)``.
            x_medium: Medium features ``(B, T, D)``.
            x_global: Global features ``(B, T, D)``.

        Returns:
            Updated (local, medium, global) feature tensors.
        """
        # Local cross-attention
        local_from_med, _ = self.local_to_medium(x_local, x_medium, x_medium)
        local_from_glob, _ = self.local_to_global(x_local, x_global, x_global)
        gate_l = self.gate_local(
            torch.cat([x_local, local_from_med, local_from_glob], dim=-1)
        )
        x_local_out = self.norm_local(
            x_local + gate_l * (local_from_med + local_from_glob)
        )

        # Medium cross-attention
        med_from_local, _ = self.medium_to_local(x_medium, x_local, x_local)
        med_from_glob, _ = self.medium_to_global(x_medium, x_global, x_global)
        gate_m = self.gate_medium(
            torch.cat([x_medium, med_from_local, med_from_glob], dim=-1)
        )
        x_medium_out = self.norm_medium(
            x_medium + gate_m * (med_from_local + med_from_glob)
        )

        # Global cross-attention
        glob_from_local, _ = self.global_to_local(x_global, x_local, x_local)
        glob_from_med, _ = self.global_to_medium(x_global, x_medium, x_medium)
        gate_g = self.gate_global(
            torch.cat([x_global, glob_from_local, glob_from_med], dim=-1)
        )
        x_global_out = self.norm_global(
            x_global + gate_g * (glob_from_local + glob_from_med)
        )

        return x_local_out, x_medium_out, x_global_out


# =============================================================================
# Hierarchical Temporal Transformer
# =============================================================================


class HierarchicalTemporalTransformer(nn.Module):
    """Multi-scale temporal transformer with local, medium, and global attention.

    The hierarchy captures motion at different time scales:
        Local (5 frames / ~0.4s):  Fine motion details, tremor.
        Medium (20 frames / ~1.5s): Motion phrases, gestures.
        Global (full sequence):     Long-term context, surgical subtasks.

    Each scale uses windowed attention (local/medium) or full attention
    (global). An optional ``CrossScaleAttention`` module enables
    bidirectional information flow between the three scales before
    fusion.

    Paper Section 3.2: Hierarchical Temporal Transformer (HTT).

    Attributes:
        d_model: Hidden dimension.
        local_window: Window size for local attention.
        medium_window: Window size for medium attention.
        use_cross_scale_attention: Whether cross-scale attention is active.
        local_layers: Local windowed transformer layers.
        medium_layers: Medium windowed transformer layers.
        global_layers: Global full-attention transformer layers.
        cross_scale_attn: Optional CrossScaleAttention module.
        scale_fusion: Linear + norm + activation fusion of 3 scales.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 1024,
        local_window: int = 5,
        medium_window: int = 20,
        local_layers: int = 2,
        medium_layers: int = 2,
        global_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_cross_scale_attention: bool = False,
    ) -> None:
        """Initialize hierarchical transformer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            local_window: Local attention window size.
            medium_window: Medium attention window size.
            local_layers: Number of local attention layers.
            medium_layers: Number of medium attention layers.
            global_layers: Number of global attention layers.
            dropout: Dropout probability.
            activation: Activation function name.
            use_cross_scale_attention: Enable cross-scale attention.
        """
        super().__init__()
        self.d_model = d_model
        self.local_window = local_window
        self.medium_window = medium_window
        self.use_cross_scale_attention = use_cross_scale_attention

        # Local attention layers (windowed, fewer heads)
        self.local_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads // 2,
                dim_feedforward=ff_dim // 2,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(local_layers)
        ])

        # Medium attention layers
        self.medium_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(medium_layers)
        ])

        # Global attention layers (full sequence)
        self.global_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(global_layers)
        ])

        # Cross-scale attention (optional)
        if use_cross_scale_attention:
            self.cross_scale_attn = CrossScaleAttention(
                d_model=d_model,
                n_heads=n_heads // 2,
                dropout=dropout,
            )

        # Scale fusion: concat 3 scales -> d_model
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _create_window_mask(
        seq_len: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a boolean attention mask for windowed attention.

        Args:
            seq_len: Sequence length.
            window_size: Attention window size.
            device: Target device.

        Returns:
            Boolean mask ``(seq_len, seq_len)`` where ``True`` = masked out.
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = False
        return mask

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply hierarchical temporal attention.

        Args:
            x: Input tensor ``(B, T, D)``.
            src_key_padding_mask: Optional padding mask ``(B, T)``.

        Returns:
            Output tensor ``(B, T, D)``.
        """
        _B, T, _D = x.shape
        device = x.device

        # Local attention (windowed)
        local_mask = self._create_window_mask(T, self.local_window, device)
        x_local = x
        for layer in self.local_layers:
            x_local = layer(x_local, src_mask=local_mask)

        # Medium attention (larger window)
        medium_mask = self._create_window_mask(T, self.medium_window, device)
        x_medium = x
        for layer in self.medium_layers:
            x_medium = layer(x_medium, src_mask=medium_mask)

        # Global attention (full sequence)
        x_global = x
        for layer in self.global_layers:
            x_global = layer(
                x_global, src_key_padding_mask=src_key_padding_mask,
            )

        # Optional cross-scale attention
        if self.use_cross_scale_attention:
            x_local, x_medium, x_global = self.cross_scale_attn(
                x_local, x_medium, x_global,
            )

        # Fuse multi-scale representations
        x_multi = torch.cat([x_local, x_medium, x_global], dim=-1)
        return self.scale_fusion(x_multi)


# =============================================================================
# Bimanual Cross-Attention
# =============================================================================


class BimanualCrossAttention(nn.Module):
    """Gated cross-attention between Tool 1 and Tool 2 representations.

    Models bimanual coordination during surgical tasks. Expert surgeons
    exhibit higher tool coordination, so this module captures how each
    tool's motion is influenced by the other.

    Bidirectional: Tool 1 attends to Tool 2 and vice versa, with
    learnable gating to control the mixing ratio.

    Paper Section 3.3: Bimanual Cross-Attention.

    Attributes:
        cross_attn_1to2: MHA where Tool 1 queries Tool 2.
        cross_attn_2to1: MHA where Tool 2 queries Tool 1.
        gate_1, gate_2: Sigmoid gating networks.
        norm1, norm2: Layer normalizations.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize bimanual cross-attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Tool 1 attends to Tool 2
        self.cross_attn_1to2 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Tool 2 attends to Tool 1
        self.cross_attn_2to1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Gating mechanism
        self.gate_1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.Sigmoid(),
        )
        self.gate_2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.Sigmoid(),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        tool1_features: torch.Tensor,
        tool2_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-attention between tools.

        Args:
            tool1_features: Tool 1 features ``(B, T, D)``.
            tool2_features: Tool 2 features ``(B, T, D)``.

        Returns:
            Updated features for (tool1, tool2).
        """
        # Tool 1 attends to Tool 2
        t1_cross, _ = self.cross_attn_1to2(
            query=tool1_features,
            key=tool2_features,
            value=tool2_features,
        )

        # Tool 2 attends to Tool 1
        t2_cross, _ = self.cross_attn_2to1(
            query=tool2_features,
            key=tool1_features,
            value=tool1_features,
        )

        # Gated fusion with residual
        gate_1 = self.gate_1(torch.cat([tool1_features, t1_cross], dim=-1))
        gate_2 = self.gate_2(torch.cat([tool2_features, t2_cross], dim=-1))

        tool1_out = self.norm1(tool1_features + gate_1 * t1_cross)
        tool2_out = self.norm2(tool2_features + gate_2 * t2_cross)

        return tool1_out, tool2_out


# =============================================================================
# Memory Components
# =============================================================================


class MemoryBank(nn.Module):
    """Learnable memory bank storing surgical motion pattern prototypes.

    Maintains a set of learnable embedding vectors that capture common
    motion patterns across training data. Input features attend to these
    memory slots to retrieve relevant long-term context.

    Paper Section 3.2: Memory-Enhanced Encoder.

    Attributes:
        memory_size: Number of memory slots.
        d_model: Dimension of each slot.
        memory: Learnable parameter ``(memory_size, d_model)``.
    """

    def __init__(
        self,
        memory_size: int = 64,
        d_model: int = 256,
        init_std: float = 0.02,
    ) -> None:
        """Initialize memory bank.

        Args:
            memory_size: Number of memory slots (prototypes).
            d_model: Dimension of each memory slot.
            init_std: Standard deviation for initialization.
        """
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model

        self.memory = nn.Parameter(
            torch.randn(memory_size, d_model) * init_std
        )

    def forward(self) -> torch.Tensor:
        """Return the memory bank.

        Returns:
            Memory tensor ``(memory_size, d_model)``.
        """
        return self.memory

    def read(
        self,
        query: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using scaled dot-product attention.

        Args:
            query: Query tensor ``(B, T, D)`` or ``(B, D)``.
            return_weights: Whether to return attention weights.

        Returns:
            Retrieved memory ``(B, T, D)`` or ``(B, D)``.
            Optionally: attention weights ``(B, T, M)`` or ``(B, M)``.
        """
        squeezed = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeezed = True

        B, _T, _D = query.shape

        memory = self.memory.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        scores = torch.bmm(query, memory.transpose(1, 2))    # (B, T, M)
        scores = scores / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)

        retrieved = torch.bmm(weights, memory)  # (B, T, D)

        if squeezed:
            retrieved = retrieved.squeeze(1)
            weights = weights.squeeze(1)

        if return_weights:
            return retrieved, weights
        return retrieved


class MemoryAttentiveFusion(nn.Module):
    """Memory-based attention fusion for long-term pattern capture.

    Combines:
        1. Learnable memory bank with motion prototypes.
        2. Multi-head attention to query the memory.
        3. Gated fusion of current features with retrieved memory.

    The memory captures long-term dependencies that purely local
    attention mechanisms may miss.

    Paper Section 3.2: Memory-Enhanced Encoder.

    Attributes:
        memory_bank: Learnable memory slots.
        memory_attention: Multi-head attention for memory querying.
        gate: Gating network for fusion control.
        norm: Layer normalization.
        output_proj: Residual output MLP.
    """

    def __init__(
        self,
        d_model: int = 256,
        memory_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize memory-based attentive fusion.

        Args:
            d_model: Model dimension.
            memory_size: Number of memory slots.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.n_heads = n_heads

        self.memory_bank = MemoryBank(memory_size, d_model)

        self.memory_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply memory-based attentive fusion.

        Args:
            x: Input tensor ``(B, T, D)``.
            return_attention: Whether to return attention weights.

        Returns:
            Fused tensor ``(B, T, D)``.
            Optionally: attention weights ``(B, T, M)``.
        """
        B, _T, _D = x.shape

        memory = self.memory_bank().unsqueeze(0).expand(B, -1, -1)

        mem_out, attn_weights = self.memory_attention(
            query=x, key=memory, value=memory, need_weights=True,
        )

        gate = self.gate(torch.cat([x, mem_out], dim=-1))
        fused = x + gate * mem_out

        fused = self.norm(fused)
        output = self.output_proj(fused) + fused  # Residual

        if return_attention:
            return output, attn_weights
        return output


class BidirectionalTemporalContext(nn.Module):
    """Process past and future context separately for bidirectional understanding.

    Splits the input window into past / current / future segments,
    processes each direction with separate attention, then fuses them
    with learnable combination weights.

    Paper Section 3.2: Bidirectional context in the MEE.

    Attributes:
        context_before: Number of past frames.
        context_after: Number of future frames.
        past_attention: Transformer layer for past context.
        future_attention: Transformer layer for future context.
        fusion: MLP fusing past + current + future.
        alpha: Learnable combination weights.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_before: int = 5,
        context_after: int = 4,
    ) -> None:
        """Initialize bidirectional temporal context.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            dropout: Dropout probability.
            context_before: Frames before the target.
            context_after: Frames after the target.
        """
        super().__init__()
        self.d_model = d_model
        self.context_before = context_before
        self.context_after = context_after

        self.past_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )

        self.future_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.alpha = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional temporal context processing.

        Assumes input is organized as ``[past_frames..., current, future_frames...]``.

        Args:
            x: Input tensor ``(B, T, D)`` where
                ``T = context_before + 1 + context_after``.

        Returns:
            Fused tensor ``(B, T, D)`` with enhanced representations.
        """
        B, T, D = x.shape

        past = x[:, :self.context_before, :]
        future = x[:, self.context_before + 1:, :]

        # Process past
        if past.shape[1] > 0:
            past_repr = self.past_attention(past)
            past_pooled = past_repr.mean(dim=1, keepdim=True)
        else:
            past_pooled = torch.zeros(B, 1, D, device=x.device)

        # Process future
        if future.shape[1] > 0:
            future_repr = self.future_attention(future)
            future_pooled = future_repr.mean(dim=1, keepdim=True)
        else:
            future_pooled = torch.zeros(B, 1, D, device=x.device)

        # Fuse past, current, future
        past_expanded = past_pooled.expand(-1, T, -1)
        future_expanded = future_pooled.expand(-1, T, -1)

        combined = torch.cat([past_expanded, x, future_expanded], dim=-1)
        fused = self.fusion(combined)

        # Weighted combination with original
        weights = F.softmax(self.alpha, dim=0)
        output = weights[0] * fused + weights[1] * x + weights[2] * (fused + x) / 2

        return output


class GatedToolFusion(nn.Module):
    """Dynamic fusion of Tool 1 and Tool 2 representations.

    Learns per-timestep reliability weights for each tool and fuses
    them accordingly. Useful when one tool may be occluded or have
    noisy measurements.

    Paper Section 3.2: Gated tool fusion in the MEE.

    Attributes:
        tool1_proj, tool2_proj: Per-tool linear projections.
        gate_network: MLP predicting ``(B, T, 2)`` softmax weights.
        output_proj: Output projection with layer norm.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize gated tool fusion.

        Args:
            d_model: Model dimension.
            hidden_dim: Hidden dimension for gate network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model

        self.tool1_proj = nn.Linear(d_model, d_model)
        self.tool2_proj = nn.Linear(d_model, d_model)

        self.gate_network = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        tool1_features: torch.Tensor,
        tool2_features: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply gated fusion of tool features.

        Args:
            tool1_features: Tool 1 features ``(B, T, D)``.
            tool2_features: Tool 2 features ``(B, T, D)``.
            return_weights: Whether to return fusion weights.

        Returns:
            Fused features ``(B, T, D)``.
            Optionally: fusion weights ``(B, T, 2)``.
        """
        t1 = self.tool1_proj(tool1_features)
        t2 = self.tool2_proj(tool2_features)

        combined = torch.cat([t1, t2], dim=-1)
        weights = self.gate_network(combined)  # (B, T, 2)

        fused = weights[..., 0:1] * t1 + weights[..., 1:2] * t2
        output = self.output_proj(fused)

        if return_weights:
            return output, weights
        return output


class MemoryEnhancedEncoder(nn.Module):
    """Combined encoder with memory attention and bidirectional context.

    Integrates all MEE components into a single encoder:
        1. Bidirectional temporal context (past/future splitting).
        2. Memory-based attentive fusion (learnable prototypes).
        3. Gated tool fusion (tool reliability weighting).

    Used as the foundation model encoder that can be frozen during
    fine-tuning.

    Paper Section 3.2: Memory-Enhanced Encoder (MEE).

    Attributes:
        use_memory: Whether memory attention is active.
        use_bidirectional: Whether bidirectional context is active.
        use_gated_fusion: Whether gated tool fusion is active.
        bidirectional: BidirectionalTemporalContext (conditional).
        memory_fusion: MemoryAttentiveFusion (conditional).
        tool_fusion: GatedToolFusion (conditional).
        norm: Final layer normalization.
    """

    def __init__(
        self,
        d_model: int = 256,
        memory_size: int = 64,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_before: int = 5,
        context_after: int = 4,
        use_memory: bool = True,
        use_bidirectional: bool = True,
        use_gated_fusion: bool = True,
    ) -> None:
        """Initialize memory-enhanced encoder.

        Args:
            d_model: Model dimension.
            memory_size: Number of memory slots.
            n_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            dropout: Dropout probability.
            context_before: Frames before target.
            context_after: Frames after target.
            use_memory: Whether to use memory attention.
            use_bidirectional: Whether to use bidirectional context.
            use_gated_fusion: Whether to use gated tool fusion.
        """
        super().__init__()
        self.d_model = d_model
        self.use_memory = use_memory
        self.use_bidirectional = use_bidirectional
        self.use_gated_fusion = use_gated_fusion

        if use_bidirectional:
            self.bidirectional = BidirectionalTemporalContext(
                d_model=d_model,
                n_heads=n_heads // 2,
                ff_dim=ff_dim,
                dropout=dropout,
                context_before=context_before,
                context_after=context_after,
            )

        if use_memory:
            self.memory_fusion = MemoryAttentiveFusion(
                d_model=d_model,
                memory_size=memory_size,
                n_heads=n_heads // 2,
                dropout=dropout,
            )

        if use_gated_fusion:
            self.tool_fusion = GatedToolFusion(
                d_model=d_model,
                hidden_dim=d_model // 2,
                dropout=dropout,
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        tool1_features: Optional[torch.Tensor] = None,
        tool2_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply memory-enhanced encoding.

        Args:
            x: Input tensor ``(B, T, D)``.
            tool1_features: Optional separate Tool 1 features ``(B, T, D)``.
            tool2_features: Optional separate Tool 2 features ``(B, T, D)``.

        Returns:
            Enhanced tensor ``(B, T, D)``.
        """
        if self.use_bidirectional:
            x = self.bidirectional(x)

        if self.use_memory:
            x = self.memory_fusion(x)

        if (
            self.use_gated_fusion
            and tool1_features is not None
            and tool2_features is not None
        ):
            tool_fused = self.tool_fusion(tool1_features, tool2_features)
            x = x + tool_fused  # Residual

        return self.norm(x)


# =============================================================================
# Mixture Density Network Output Head
# =============================================================================


class MixtureDensityHead(nn.Module):
    """Mixture Density Network head for multi-modal pose predictions.

    Outputs a mixture of K Gaussians for each pose dimension, enabling
    the model to represent multi-modal predictive distributions (e.g.,
    when the tool could plausibly move in multiple directions).

    Paper Section 3.5: MDN output head.

    Attributes:
        output_dim: Pose output dimension.
        num_components: Number of mixture components K.
        hidden: Shared hidden MLP.
        pi_head: Mixture weight logits ``(B, K)``.
        mu_head: Mixture means ``(B, K, output_dim)``.
        sigma_head: Mixture stds ``(B, K, output_dim)``.
    """

    def __init__(
        self,
        d_model: int = 256,
        output_dim: int = 16,
        num_components: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize MDN head.

        Args:
            d_model: Input model dimension.
            output_dim: Output pose dimension.
            num_components: Number of mixture components.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_components = num_components

        self.hidden = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.pi_head = nn.Linear(hidden_dim, num_components)
        self.mu_head = nn.Linear(hidden_dim, num_components * output_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_components * output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute MDN outputs.

        Args:
            x: Input tensor ``(B, D)``.

        Returns:
            Dictionary with keys ``mdn_pi``, ``mdn_mu``, ``mdn_sigma``.
        """
        B = x.shape[0]
        hidden = self.hidden(x)

        pi = self.pi_head(hidden)  # (B, K)

        mu = self.mu_head(hidden).view(B, self.num_components, self.output_dim)

        sigma = F.softplus(self.sigma_head(hidden)) + 1e-4
        sigma = sigma.view(B, self.num_components, self.output_dim)

        return {"mdn_pi": pi, "mdn_mu": mu, "mdn_sigma": sigma}


# =============================================================================
# Probabilistic Pose Head (v5 -- with Cholesky support)
# =============================================================================


class ProbabilisticPoseHead(nn.Module):
    """Probabilistic output head with Beta-NLL loss and Cholesky support.

    Outputs per tool (x2 tools):
        Position:  mu(3) + sigma(3) or L_cholesky(6) -> Gaussian.
        Rotation:  mu_quat(4) + kappa(1) -> von Mises-Fisher on S^3.
        Jaw angle: mu(1) + sigma(1) -> Gaussian.
        Jaw state: logits -> Bernoulli (binary open/closed).

    During the Cholesky warmup period, the head outputs diagonal sigma
    even when Cholesky is configured. After warmup, it outputs Cholesky
    parameters alongside diagonal sigma for metrics.

    Paper Section 3.5: Probabilistic output head with Beta-NLL.

    Attributes:
        use_cholesky: Whether Cholesky covariance parameters are available.
        min_sigma: Minimum predicted sigma.
        max_sigma: Maximum predicted sigma.
        min_kappa: Minimum VMF concentration.
        num_jaw_classes: Number of jaw state classes (2 = binary).
        hidden: Shared hidden layer with MC Dropout.
        pos_mu: Position mean head.
        pos_sigma: Position diagonal sigma head.
        pos_cholesky: Position Cholesky head (conditional).
        quat_mu: Quaternion mean head.
        quat_kappa: VMF concentration head.
        angle_mu: Jaw angle mean head.
        angle_sigma: Jaw angle sigma head.
        jaw_state_head: Jaw state classification head.
    """

    def __init__(
        self,
        d_model: int = 256,
        use_cholesky: bool = False,
        min_sigma: float = 1e-3,
        max_sigma: float = 10.0,
        min_kappa: float = 1.0,
        mc_dropout_rate: float = 0.1,
        num_jaw_classes: int = 2,
    ) -> None:
        """Initialize probabilistic pose head.

        Args:
            d_model: Input model dimension.
            use_cholesky: Whether to output Cholesky covariance parameters.
            min_sigma: Minimum predicted sigma.
            max_sigma: Maximum predicted sigma.
            min_kappa: Minimum VMF concentration.
            mc_dropout_rate: MC Dropout rate for epistemic uncertainty.
            num_jaw_classes: Number of jaw state classes.
        """
        super().__init__()
        self.use_cholesky = use_cholesky
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kappa = min_kappa
        self.num_jaw_classes = num_jaw_classes

        hidden_dim = d_model // 2

        # Shared hidden layer with MC Dropout
        self.hidden = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            MCDropout(mc_dropout_rate),
        )

        # Position outputs (2 tools x 3D)
        self.pos_mu = nn.Linear(hidden_dim, 6)
        self.pos_sigma = nn.Linear(hidden_dim, 6)
        if use_cholesky:
            # Cholesky L: 6 lower-triangular params per tool
            self.pos_cholesky = nn.Linear(hidden_dim, 12)

        # Quaternion outputs (2 tools x 4D mu + 1D kappa)
        self.quat_mu = nn.Linear(hidden_dim, 8)
        self.quat_kappa = nn.Linear(hidden_dim, 2)

        # Jaw angle outputs (2 tools)
        self.angle_mu = nn.Linear(hidden_dim, 2)
        self.angle_sigma = nn.Linear(hidden_dim, 2)

        # Jaw state classification
        jaw_out_dim = 2 * num_jaw_classes
        self.jaw_state_head = nn.Linear(hidden_dim, jaw_out_dim)

    def forward(
        self,
        x: torch.Tensor,
        force_diagonal: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute probabilistic pose outputs.

        Args:
            x: Input tensor ``(B, D)`` from fused multi-scale representation.
            force_diagonal: Force diagonal covariance even if Cholesky is
                configured (used during warmup).

        Returns:
            Dictionary with all predictions and uncertainty parameters:
                mu_position: ``(B, 2, 3)``
                sigma_position: ``(B, 2, 3)``
                cholesky_position: ``(B, 2, 6)`` (when Cholesky active)
                mu_quaternion: ``(B, 2, 4)`` (unit normalized)
                kappa_quaternion: ``(B, 2, 1)``
                mu_angle: ``(B, 2, 1)``
                sigma_angle: ``(B, 2, 1)``
                jaw_state_logits: ``(B, 2)`` or ``(B, 2, C)``
        """
        hidden = self.hidden(x)
        outputs: dict[str, torch.Tensor] = {}

        # ---- Position ----
        pos_mu = self.pos_mu(hidden).view(-1, 2, 3)
        outputs["mu_position"] = pos_mu

        if self.use_cholesky and not force_diagonal:
            cholesky = self.pos_cholesky(hidden).view(-1, 2, 6)
            outputs["cholesky_position"] = cholesky
            # Extract diagonal sigma from Cholesky for metrics
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
        quat_mu = F.normalize(quat_mu, dim=-1)  # Unit quaternion
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
                -1, 2, self.num_jaw_classes,
            )
        else:
            outputs["jaw_state_logits"] = jaw_raw  # (B, 2)

        return outputs


# =============================================================================
# Cross-Scale Fusion
# =============================================================================


class CrossScaleFusion(nn.Module):
    """Fuse scale tokens from multiple temporal scales via cross-attention.

    Takes scale tokens extracted from each scale's encoder output and
    combines them using multi-head self-attention, producing a unified
    multi-scale representation.

    Paper Section 3.4: Cross-scale fusion.

    Attributes:
        d_model: Model dimension.
        n_scales: Number of temporal scales.
        cross_attn: Multi-head self-attention for scale tokens.
        ffn: Feed-forward network after attention.
        norm1, norm2: Layer normalizations.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_scales: int = 3,
    ) -> None:
        """Initialize cross-scale fusion.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            n_scales: Number of temporal scales.
        """
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales

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

    def forward(
        self,
        scale_tokens: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse scale tokens via self-attention and mean pooling.

        Args:
            scale_tokens: List of scale token tensors, each ``(B, d_model)``.

        Returns:
            Fused representation ``(B, d_model)``.
        """
        tokens = torch.stack(scale_tokens, dim=1)  # (B, n_scales, d_model)

        # Self-attention across scales
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Feed-forward
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # Pool across scales
        fused = tokens.mean(dim=1)  # (B, d_model)
        return fused


# =============================================================================
# Masking Utility
# =============================================================================


def create_masking(
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    mask_ratio_frames: float = 0.15,
    mask_ratio_features: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a boolean masking tensor for SSL pre-training.

    Supports two modes:
        Per-frame masking: Randomly mask entire frames.
        Per-feature masking: Randomly mask individual features per frame.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        feature_dim: Feature dimension.
        mask_ratio_frames: Fraction of frames to mask (per-frame mode).
        mask_ratio_features: Fraction of features to mask per frame.
            When > 0, uses per-feature mode instead of per-frame.
        device: Target device.

    Returns:
        Boolean mask where ``True`` = masked:
            ``(B, T, D)`` if per-feature masking,
            ``(B, T)`` if per-frame masking.
    """
    if mask_ratio_features > 0:
        mask = torch.rand(
            batch_size, seq_len, feature_dim, device=device,
        )
        return mask < mask_ratio_features
    else:
        mask = torch.rand(batch_size, seq_len, device=device)
        return mask < mask_ratio_frames
