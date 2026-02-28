"""Visual feature projections and fusion modules for multimodal BTPN.

This module implements all visual processing components for the Bayesian
Temporal Pose Network (BTPN). These modules project pre-extracted visual
features (segmentation, depth, pose keypoints) into a shared embedding
space and fuse them with kinematic representations.

Architecture (per temporal window of T frames):

    Segmentation pathway:
        Neck features (B,T,2,256)   --> SegmentationProjection --> (B,T,256)
        Backbone features (B,T,512) --|

    Depth pathway:
        DA2 embeddings (B,T,384) --> DepthProjection --> (B,T,128)

    Pose keypoint pathway:
        Keypoints (B,T,2,8,3) --> KeypointProjection --> (B,T,64)
        Geometric (B,T,6)     --|

    Pose backbone pathway:
        Backbone (B,T,2,512) --> PoseProjection --> (B,T,64)

    Scene-level fusion:
        Seg(256) + Depth(128) --> SceneFusion --> (B,T,256)
        Kp(64) + PoseBackbone(64) + Geo(32) --> internal fusion
        Scene(256) + Pose(128) --> visual_proj --> (B,T,256)

    Temporal encoding:
        Visual sequence --> ClinicalAttentionEncoder --> (B,256) CLS token

    Kinematic-visual fusion:
        KinematicVisualFusion(kin_repr, visual_repr) --> (B,256)

    Multi-channel gating:
        ConfidenceGate(detection_conf) --> pos/rot/angle gates

    Residual prediction:
        ResidualPoseHead(fused, kin_outputs, gates) --> corrected pose

Paper cross-references:
    - Section 3.2: Visual Feature Extraction and Projection
    - Section 3.3: Clinical Attention Encoder
    - Section 3.4: Kinematic-Visual Fusion
    - Section 3.5: Multi-Channel Confidence Gating
    - Figure 3: Visual processing pipeline
    - Table 2: Modality ablation study

Note on attribute naming:
    Internal attribute names (self.xxx) are preserved exactly from the V3
    research codebase for checkpoint compatibility. Public class names use
    clean paper terminology. See each class docstring for the mapping.

Author: BTPN Publication Repository
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Segmentation Feature Projection
# =============================================================================


class SegmentationProjection(nn.Module):
    """Project and fuse YOLO segmentation features from neck and backbone.

    Merges two segmentation feature streams with learnable gating:

    1. Per-tool FPN neck features (B, T, 2, 256) from ROI-pooled P3/P4/P5
       layers. These carry fine-grained, tool-localized information.
    2. Global backbone features (B, T, 512) from the YOLO backbone.
       These carry scene-level context (layout, other objects).

    Each stream is projected independently, then fused via a learned gate
    that decides per-frame whether tool-specific or global features are
    more informative.

    Paper: Section 3.2, "Segmentation Feature Projection"

    Checkpoint compatibility:
        Internally uses self.per_tool_proj, self.tool_embedding, self.pool
        (from SegNeckProjection), self.proj (from SegBackboneProjection),
        self.backbone_up, self.gate, self.fuse, self.norm (from
        LearnableSegFusion).

    Args:
        neck_dim: Per-tool FPN neck feature dimension.
        backbone_dim: Global backbone feature dimension.
        proj_dim: Output projection dimension for neck path.
        backbone_proj_dim: Intermediate projection dimension for backbone path.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        neck_dim: int = 256,
        backbone_dim: int = 512,
        proj_dim: int = 256,
        backbone_proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.backbone_proj_dim = backbone_proj_dim

        # --- Neck projection (per-tool FPN features) ---
        # Kept as separate attributes for checkpoint compatibility with
        # SegNeckProjection from V1.
        self.per_tool_proj = nn.Sequential(
            nn.LayerNorm(neck_dim),
            nn.Linear(neck_dim, proj_dim),
            nn.GELU(),
        )
        self.tool_embedding = nn.Embedding(2, proj_dim)
        self.pool = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        # --- Backbone projection (global scene features) ---
        # Kept as separate attribute for checkpoint compatibility with
        # SegBackboneProjection from V1.
        self.backbone_proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.LayerNorm(backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim // 2, backbone_proj_dim),
        )

        # --- Learnable fusion gate ---
        # Kept as separate attributes for checkpoint compatibility with
        # LearnableSegFusion from V1.
        concat_dim = proj_dim + backbone_proj_dim
        self.backbone_up = nn.Linear(backbone_proj_dim, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, proj_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Linear(concat_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(
        self,
        neck_features: torch.Tensor,
        backbone_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project and fuse segmentation features.

        Args:
            neck_features: (B, T, 2, neck_dim) per-tool FPN features.
            backbone_features: (B, T, backbone_dim) global backbone features.

        Returns:
            Fused segmentation features (B, T, proj_dim).
        """
        B, T, n_tools, D = neck_features.shape

        # --- Neck pathway ---
        proj = self.per_tool_proj(neck_features)  # (B, T, 2, proj_dim)

        tool_ids = torch.arange(n_tools, device=neck_features.device)
        tool_emb = self.tool_embedding(tool_ids)  # (2, proj_dim)
        proj = proj + tool_emb.unsqueeze(0).unsqueeze(0)

        tool1 = proj[:, :, 0, :]  # (B, T, proj_dim)
        tool2 = proj[:, :, 1, :]  # (B, T, proj_dim)
        neck_proj = self.pool(
            torch.cat([tool1, tool2], dim=-1)
        )  # (B, T, proj_dim)

        # --- Backbone pathway ---
        bb_proj = self.backbone_proj(backbone_features)  # (B, T, bb_proj_dim)

        # --- Gated fusion ---
        concat = torch.cat([neck_proj, bb_proj], dim=-1)
        gate_val = self.gate(concat)  # (B, T, proj_dim)
        backbone_up = self.backbone_up(bb_proj)  # (B, T, proj_dim)
        fused = gate_val * neck_proj + (1 - gate_val) * backbone_up

        return self.norm(fused)  # (B, T, proj_dim)


# =============================================================================
# V3-Compatible Segmentation Sub-Components
# =============================================================================
# The following three classes provide checkpoint-compatible sub-modules
# matching the V3 research codebase attribute names (seg_neck_proj,
# seg_backbone_proj, seg_fusion). They are used by BTPN internally
# to preserve checkpoint loading. For new code, prefer the merged
# SegmentationProjection class above.


class _SegNeckProjection(nn.Module):
    """Project per-tool FPN neck features (V3-compatible).

    Input:  (B, T, 2, neck_dim) per-tool FPN features.
    Output: (B, T, proj_dim).
    """

    def __init__(
        self,
        neck_dim: int = 256,
        proj_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.per_tool_proj = nn.Sequential(
            nn.LayerNorm(neck_dim),
            nn.Linear(neck_dim, proj_dim),
            nn.GELU(),
        )
        self.tool_embedding = nn.Embedding(2, proj_dim)
        self.pool = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

    def forward(self, neck_features: torch.Tensor) -> torch.Tensor:
        """Project neck features.

        Args:
            neck_features: (B, T, 2, neck_dim).

        Returns:
            Projected features (B, T, proj_dim).
        """
        B, T, n_tools, D = neck_features.shape
        proj = self.per_tool_proj(neck_features)
        tool_ids = torch.arange(n_tools, device=neck_features.device)
        tool_emb = self.tool_embedding(tool_ids)
        proj = proj + tool_emb.unsqueeze(0).unsqueeze(0)
        tool1 = proj[:, :, 0, :]
        tool2 = proj[:, :, 1, :]
        return self.pool(torch.cat([tool1, tool2], dim=-1))


class _SegBackboneProjection(nn.Module):
    """Project global backbone features (V3-compatible).

    Input:  (B, T, backbone_dim).
    Output: (B, T, proj_dim).
    """

    def __init__(
        self,
        backbone_dim: int = 512,
        proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.LayerNorm(backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim // 2, proj_dim),
        )

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """Project backbone features.

        Args:
            backbone_features: (B, T, backbone_dim).

        Returns:
            Projected features (B, T, proj_dim).
        """
        return self.proj(backbone_features)


class _LearnableSegFusion(nn.Module):
    """Fuse neck + backbone with learned gating (V3-compatible).

    Input:  neck_proj (B, T, neck_dim) + backbone_proj (B, T, bb_dim).
    Output: (B, T, output_dim).
    """

    def __init__(
        self,
        neck_proj_dim: int = 256,
        backbone_proj_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        concat_dim = neck_proj_dim + backbone_proj_dim
        self.backbone_up = nn.Linear(backbone_proj_dim, neck_proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Linear(concat_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        neck_proj: torch.Tensor,
        backbone_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse neck and backbone features.

        Args:
            neck_proj: (B, T, neck_proj_dim).
            backbone_proj: (B, T, backbone_proj_dim).

        Returns:
            Fused features (B, T, output_dim).
        """
        concat = torch.cat([neck_proj, backbone_proj], dim=-1)
        gate_val = self.gate(concat)
        backbone_up = self.backbone_up(backbone_proj)
        fused = gate_val * neck_proj + (1 - gate_val) * backbone_up
        return self.norm(fused)


class _LearnablePoseFusion(nn.Module):
    """Fuse keypoint + encoder + geometric pose features (V3-compatible).

    Input:  kp_proj(kp_dim) + enc_proj(enc_dim) + geo_proj(geo_dim).
    Output: (B, T, output_dim).
    """

    def __init__(
        self,
        kp_proj_dim: int = 64,
        enc_proj_dim: int = 64,
        geo_proj_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()
        concat_dim = kp_proj_dim + enc_proj_dim + geo_proj_dim
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Linear(concat_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        kp_proj: torch.Tensor,
        enc_proj: torch.Tensor,
        geo_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse pose sub-features.

        Args:
            kp_proj: (B, T, kp_proj_dim).
            enc_proj: (B, T, enc_proj_dim).
            geo_proj: (B, T, geo_proj_dim).

        Returns:
            Fused pose features (B, T, output_dim).
        """
        concat = torch.cat([kp_proj, enc_proj, geo_proj], dim=-1)
        gate_val = self.gate(concat)
        fused = self.fuse(concat)
        return self.norm(gate_val * fused)


class _PoseKeypointProjection(nn.Module):
    """Project raw keypoints via attention pooling (V3-compatible).

    Matches PoseKeypointProjection from V1 -- keypoint spatial branch only,
    without geometric features (those are in _PoseGeometricProjection).

    Input:  (B, T, 2, 8, 3) per-tool keypoints (x, y, confidence).
    Output: (B, T, proj_dim).
    """

    KP_TYPE_MAP = [0, 0, 0, 0, 1, 2, 3, 3]

    def __init__(
        self,
        n_keypoints: int = 8,
        kp_input_dim: int = 3,
        kp_embed_dim: int = 16,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.kp_mlp = nn.Sequential(
            nn.Linear(kp_input_dim, kp_embed_dim * 2),
            nn.GELU(),
            nn.Linear(kp_embed_dim * 2, kp_embed_dim),
        )
        self.kp_type_embed = nn.Embedding(4, kp_embed_dim)
        self.tool_embed = nn.Embedding(2, kp_embed_dim)
        self.attn_query = nn.Parameter(
            torch.randn(1, 1, 1, kp_embed_dim) * 0.02
        )
        self.attn_proj = nn.Linear(kp_embed_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(kp_embed_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Project keypoints.

        Args:
            keypoints: (B, T, 2, 8, 3).

        Returns:
            Projected features (B, T, proj_dim).
        """
        B, T, n_tools, n_kp, _ = keypoints.shape
        conf = keypoints[..., 2:3]
        kp_emb = self.kp_mlp(keypoints) * conf
        kp_types = torch.tensor(
            self.KP_TYPE_MAP, device=keypoints.device, dtype=torch.long,
        )
        kp_emb = kp_emb + self.kp_type_embed(kp_types).view(1, 1, 1, n_kp, -1)
        tool_ids = torch.arange(n_tools, device=keypoints.device)
        kp_emb = kp_emb + self.tool_embed(tool_ids).view(1, 1, n_tools, 1, -1)
        attn_scores = self.attn_proj(kp_emb).squeeze(-1)
        attn_scores = attn_scores.masked_fill(conf.squeeze(-1) < 0.1, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = (kp_emb * attn_weights.unsqueeze(-1)).sum(dim=3)
        tool1 = pooled[:, :, 0, :]
        tool2 = pooled[:, :, 1, :]
        return self.out_proj(torch.cat([tool1, tool2], dim=-1))


class _PoseGeometricProjection(nn.Module):
    """Project derived geometric features (V3-compatible).

    Input:  (B, T, input_dim) -- shaft_width(2) + midline_angle(2) + jaw_opening(2).
    Output: (B, T, proj_dim).
    """

    def __init__(self, input_dim: int = 6, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, geometric: torch.Tensor) -> torch.Tensor:
        """Project geometric features.

        Args:
            geometric: (B, T, input_dim).

        Returns:
            Projected features (B, T, proj_dim).
        """
        return self.proj(geometric)


# =============================================================================
# Depth Feature Projection
# =============================================================================


class DepthProjection(nn.Module):
    """Project Depth Anything V2 embeddings to model dimension.

    Takes concatenated depth embeddings [global(384) + ROI1(384) + ROI2(384)]
    from the frozen DA2 encoder and projects them to a compact representation.

    Paper: Section 3.2, "Depth Feature Projection"

    Args:
        depth_dim: Input depth embedding dimension (default 1152 for
            concatenated global + 2x per-tool ROI features, or 384 for
            global-only).
        proj_dim: Output projection dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        depth_dim: int = 1152,
        proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Attribute name 'proj' matches DepthFeatureProjection from V1
        self.proj = nn.Sequential(
            nn.Linear(depth_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
        )

    def forward(self, depth_features: torch.Tensor) -> torch.Tensor:
        """Project depth embeddings.

        Args:
            depth_features: (B, T, depth_dim) depth embeddings.

        Returns:
            Projected features (B, T, proj_dim).
        """
        return self.proj(depth_features)


# =============================================================================
# Keypoint Projection
# =============================================================================


class KeypointProjection(nn.Module):
    """Project per-tool 8-keypoint detections to learned embeddings.

    Merges the raw keypoint projection (spatial coordinates with confidence
    weighting, attention pooling across keypoints) with geometric feature
    projection (shaft width, midline angle, jaw opening).

    The keypoint encoder uses:
    - Per-keypoint MLP with visibility weighting by detection confidence
    - Learnable keypoint type embeddings (shaft/joint/tip/jaw)
    - Learnable tool identity embeddings
    - Attention pooling across keypoints per tool
    - Geometric feature branch for derived measurements

    Paper: Section 3.2, "Pose Keypoint Feature Projection"
    Figure 4: Keypoint encoding architecture

    Checkpoint compatibility:
        Internally uses attribute names from PoseKeypointProjection (V1)
        and PoseGeometricProjection (V1). Both are merged into this class.

    Args:
        n_keypoints: Number of keypoints per tool.
        kp_input_dim: Per-keypoint input dimension (x, y, confidence).
        kp_embed_dim: Intermediate keypoint embedding dimension.
        kp_proj_dim: Keypoint branch output dimension.
        geometric_dim: Geometric feature input dimension.
        geo_proj_dim: Geometric branch output dimension.
        dropout: Dropout rate.
    """

    # Keypoint type groups for learnable type embeddings.
    # 0=shaft (KP0-3), 1=joint (KP4), 2=tip (KP5), 3=jaw (KP6-7)
    KP_TYPE_MAP = [0, 0, 0, 0, 1, 2, 3, 3]

    def __init__(
        self,
        n_keypoints: int = 8,
        kp_input_dim: int = 3,
        kp_embed_dim: int = 16,
        kp_proj_dim: int = 64,
        geometric_dim: int = 6,
        geo_proj_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.kp_proj_dim = kp_proj_dim
        self.geo_proj_dim = geo_proj_dim

        # --- Keypoint spatial branch ---
        # Attribute names match PoseKeypointProjection from V1.
        self.kp_mlp = nn.Sequential(
            nn.Linear(kp_input_dim, kp_embed_dim * 2),
            nn.GELU(),
            nn.Linear(kp_embed_dim * 2, kp_embed_dim),
        )
        self.kp_type_embed = nn.Embedding(4, kp_embed_dim)
        self.tool_embed = nn.Embedding(2, kp_embed_dim)
        self.attn_query = nn.Parameter(
            torch.randn(1, 1, 1, kp_embed_dim) * 0.02
        )
        self.attn_proj = nn.Linear(kp_embed_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Linear(kp_embed_dim * 2, kp_proj_dim),
            nn.LayerNorm(kp_proj_dim),
            nn.Dropout(dropout),
        )

        # --- Geometric feature branch ---
        # Attribute names match PoseGeometricProjection from V1.
        self.geo_proj = nn.Sequential(
            nn.Linear(geometric_dim, geo_proj_dim),
            nn.LayerNorm(geo_proj_dim),
            nn.GELU(),
            nn.Linear(geo_proj_dim, geo_proj_dim),
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        geometric: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project keypoints and geometric features.

        Args:
            keypoints: (B, T, 2, 8, 3) per-tool keypoints with (x, y, conf).
            geometric: (B, T, 6) derived geometric features, optional.
                Contains shaft_width(2) + midline_angle(2) + jaw_opening(2).

        Returns:
            Tuple of:
                kp_features: (B, T, kp_proj_dim) keypoint projection.
                geo_features: (B, T, geo_proj_dim) geometric projection,
                    or None if geometric input is None.
        """
        B, T, n_tools, n_kp, _ = keypoints.shape

        # Visibility weighting
        conf = keypoints[..., 2:3]  # (B, T, 2, 8, 1)

        # Per-keypoint embedding
        kp_emb = self.kp_mlp(keypoints)  # (B, T, 2, 8, kp_embed_dim)
        kp_emb = kp_emb * conf  # visibility weighting

        # Add keypoint type embeddings
        kp_types = torch.tensor(
            self.KP_TYPE_MAP, device=keypoints.device, dtype=torch.long,
        )
        type_emb = self.kp_type_embed(kp_types)  # (8, kp_embed_dim)
        kp_emb = kp_emb + type_emb.view(1, 1, 1, n_kp, -1)

        # Add tool identity embeddings
        tool_ids = torch.arange(n_tools, device=keypoints.device)
        tool_emb = self.tool_embed(tool_ids)  # (2, kp_embed_dim)
        kp_emb = kp_emb + tool_emb.view(1, 1, n_tools, 1, -1)

        # Attention-pool across keypoints per tool
        attn_scores = self.attn_proj(kp_emb).squeeze(-1)  # (B, T, 2, 8)
        attn_mask = conf.squeeze(-1) < 0.1  # (B, T, 2, 8)
        attn_scores = attn_scores.masked_fill(attn_mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, 2, 8)

        pooled = (kp_emb * attn_weights.unsqueeze(-1)).sum(
            dim=3
        )  # (B, T, 2, embed)

        # Concat tools and project
        tool1 = pooled[:, :, 0, :]  # (B, T, embed)
        tool2 = pooled[:, :, 1, :]  # (B, T, embed)
        kp_features = self.out_proj(
            torch.cat([tool1, tool2], dim=-1)
        )  # (B, T, kp_proj_dim)

        # Geometric branch
        geo_features = None
        if geometric is not None:
            geo_features = self.geo_proj(geometric)  # (B, T, geo_proj_dim)

        return kp_features, geo_features


# =============================================================================
# Pose Backbone Projection
# =============================================================================


class PoseProjection(nn.Module):
    """Project per-tool pose backbone features from the YOLO pose encoder.

    Takes backbone features extracted by the YOLO pose estimation head and
    projects them to a compact representation. Tool identity is preserved
    through learnable embeddings.

    Paper: Section 3.2, "Pose Encoder Projection"

    Checkpoint compatibility:
        Uses attribute names from PoseEncoderProjection (V1):
        self.per_tool_proj, self.tool_embed, self.pool.

    Args:
        backbone_dim: Per-tool backbone feature dimension.
        proj_dim: Output projection dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        backbone_dim: int = 512,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Attribute names match PoseEncoderProjection from V1
        self.per_tool_proj = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.tool_embed = nn.Embedding(2, proj_dim)
        self.pool = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

    def forward(self, backbone: torch.Tensor) -> torch.Tensor:
        """Project pose backbone features.

        Args:
            backbone: (B, T, 2, backbone_dim) per-tool backbone features.

        Returns:
            Projected features (B, T, proj_dim).
        """
        B, T, n_tools, D = backbone.shape

        proj = self.per_tool_proj(backbone)  # (B, T, 2, proj_dim)

        tool_ids = torch.arange(n_tools, device=backbone.device)
        tool_emb = self.tool_embed(tool_ids)  # (2, proj_dim)
        proj = proj + tool_emb.view(1, 1, n_tools, -1)

        tool1 = proj[:, :, 0, :]
        tool2 = proj[:, :, 1, :]

        return self.pool(torch.cat([tool1, tool2], dim=-1))


# =============================================================================
# Scene-Level Visual Fusion
# =============================================================================


class SceneFusion(nn.Module):
    """Fuse segmentation and depth features into a unified scene representation.

    Concatenates the fused segmentation features (tool-specific + global)
    with depth features and projects to the model dimension.

    Paper: Section 3.2, "Scene-Level Fusion"

    Checkpoint compatibility:
        Uses self.proj matching SceneFusion from V2.

    Args:
        seg_dim: Segmentation fusion output dimension.
        depth_dim: Depth projection output dimension.
        output_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        seg_dim: int = 256,
        depth_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(seg_dim + depth_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        seg_fused: torch.Tensor,
        depth_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse segmentation and depth features.

        Args:
            seg_fused: (B, T, seg_dim) fused segmentation features.
            depth_proj: (B, T, depth_dim) projected depth features.

        Returns:
            Scene representation (B, T, output_dim).
        """
        return self.proj(torch.cat([seg_fused, depth_proj], dim=-1))


# =============================================================================
# Kinematic-Visual Cross-Attention Fusion
# =============================================================================


class KinematicVisualFusion(nn.Module):
    """Fuse kinematic and visual representations via cross-attention.

    The visual representation attends to the kinematic embedding via
    multi-head cross-attention, then a gated residual connection controls
    how much visual information blends into the kinematic prior.

    This is the bridge between the frozen kinematic model (BTPNv5) and
    the trainable visual branch. The gate learns to trust visual corrections
    only when they provide complementary information.

    Paper: Section 3.4, "Kinematic-Visual Fusion"

    Checkpoint compatibility:
        Uses attribute names from KinematicVisualFusion (V2):
        self.cross_attn, self.norm1, self.norm2, self.ffn, self.gate.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Gating: controls how much visual info blends into kinematic
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        kinematic_repr: torch.Tensor,
        visual_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse kinematic and visual representations.

        Args:
            kinematic_repr: (B, d_model) kinematic embedding from frozen
                KinematicFoundationModel.
            visual_repr: (B, d_model) visual embedding from
                ClinicalAttentionEncoder.

        Returns:
            Fused representation (B, d_model).
        """
        # Reshape for attention: (B, 1, d_model)
        kin_seq = kinematic_repr.unsqueeze(1)
        vis_seq = visual_repr.unsqueeze(1)

        # Cross-attention: visual queries, kinematic keys/values
        attn_out, _ = self.cross_attn(vis_seq, kin_seq, kin_seq)
        vis_attended = self.norm1(vis_seq + attn_out).squeeze(1)

        # Feed-forward
        vis_attended = vis_attended + self.ffn(self.norm2(vis_attended))

        # Gated fusion: kin + gate * visual_correction
        gate_val = self.gate(
            torch.cat([kinematic_repr, vis_attended], dim=-1)
        )
        fused = kinematic_repr + gate_val * vis_attended

        return fused


# =============================================================================
# Clinical Attention Encoder
# =============================================================================


class ClinicalAttentionEncoder(nn.Module):
    """Hierarchical visual encoder with clinically motivated attention spans.

    Applies windowed self-attention at three clinically meaningful timescales,
    matching the temporal structure of surgical actions:

    - Layers 0-1: Local window attention (default 8 frames, ~0.6s at 13 fps).
      Captures micro-gestures like grasp adjustments and wrist rotations.
    - Layers 2-3: Medium window attention (default 20 frames, ~1.5s).
      Captures atomic actions like grasp-lift-transfer sequences.
    - Layers 4-5: Global attention (unrestricted).
      Captures phase-level and trial-level context.

    A CLS token at position 0 always attends to all positions across all
    layers, providing a pooled output that integrates multi-scale information.

    Paper: Section 3.3, "Clinical Attention Encoder"
    Figure 5: Attention span visualization

    Checkpoint compatibility:
        Uses attribute names from ClinicalAttentionEncoder (V3):
        self.cls_token, self.pos_encoding, self.layers, self.norm.

    Args:
        d_model: Model dimension.
        n_layers: Total number of transformer layers (should be 6 for
            2 local + 2 medium + 2 global).
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        local_window: Local attention window size (frames).
        medium_window: Medium attention window size (frames).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
        local_window: int = 8,
        medium_window: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.local_window = local_window
        self.medium_window = medium_window

        # CLS token for pooled output
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (large enough for any reasonable sequence)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Individual layers with per-pair attention masks (not wrapped in
        # nn.TransformerEncoder, since each layer pair needs a different mask)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def _build_window_mask(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a windowed attention mask.

        Each position attends only to positions within a symmetric window.
        The CLS token (position 0) always attends to all positions and is
        attended to by all positions.

        Args:
            seq_len: Total sequence length including CLS token.
            window_size: Window radius (each position attends to
                window_size positions on each side).
            device: Device for the mask tensor.

        Returns:
            Boolean attention mask (seq_len, seq_len). True = blocked.
        """
        # Start with all-blocked mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # CLS token (row 0) attends to everything
        mask[0, :] = False
        # Everything attends to CLS token (column 0)
        mask[:, 0] = False

        # Windowed attention for non-CLS positions (1..seq_len-1)
        for i in range(1, seq_len):
            lo = max(1, i - window_size)
            hi = min(seq_len, i + window_size + 1)
            mask[i, lo:hi] = False

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual features with hierarchical clinical attention.

        Args:
            x: (B, T, d_model) visual feature sequence.

        Returns:
            Visual representation (B, d_model) from the CLS token.
        """
        B, T, _ = x.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T + 1, :]

        # Build attention masks for the three tiers
        device = x.device
        seq_len = T + 1
        local_mask = self._build_window_mask(
            seq_len, self.local_window, device
        )
        medium_mask = self._build_window_mask(
            seq_len, self.medium_window, device
        )

        # Apply layers with tier-appropriate masks
        for i, layer in enumerate(self.layers):
            if i < 2:
                # Layers 0-1: local window attention (~0.6s micro-gestures)
                x = layer(x, src_mask=local_mask)
            elif i < 4:
                # Layers 2-3: medium window attention (~1.5s actions)
                x = layer(x, src_mask=medium_mask)
            else:
                # Layers 4-5: global attention (phase/trial context)
                x = layer(x)

        x = self.norm(x)

        return x[:, 0, :]  # CLS token -> (B, d_model)


# =============================================================================
# Multi-Channel Confidence Gate
# =============================================================================


class ConfidenceGate(nn.Module):
    """Multi-channel confidence gate with per-component ceilings.

    Computes separate gate values for position, rotation, and jaw angle
    corrections, each bounded by a channel-specific ceiling. This prevents
    the failure mode observed in V2 where a single saturated gate caused
    rotation corruption from visual noise.

    Gate design:
    - High initial temperature produces flat sigmoid outputs, yielding
      conservative (low) initial gate values.
    - Per-channel ceilings (pos<=0.5, rot<=0.1, angle<=0.5) bound the
      maximum visual correction magnitude.
    - Gate entropy regularization (applied externally) prevents collapse
      to 0 or ceiling.

    Paper: Section 3.5, "Multi-Channel Confidence Gating"
    Table 3: Gate ceiling ablation

    Checkpoint compatibility:
        Renamed from MultiChannelConfidenceGate (V3). Internally uses
        attribute names: self.shared, self.pos_head, self.rot_head,
        self.angle_head, self.log_temperature (all from V3).

    Args:
        hidden_dim: MLP hidden dimension.
        max_gate_pos: Maximum position gate value.
        max_gate_rot: Maximum rotation gate value.
        max_gate_angle: Maximum angle gate value.
        init_temperature: Initial sigmoid temperature (high = conservative).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_gate_pos: float = 0.5,
        max_gate_rot: float = 0.1,
        max_gate_angle: float = 0.5,
        init_temperature: float = 5.0,
    ):
        super().__init__()
        self.max_gate_pos = max_gate_pos
        self.max_gate_rot = max_gate_rot
        self.max_gate_angle = max_gate_angle

        # Shared feature extraction from per-tool detection confidence
        self.shared = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for each correction channel
        self.pos_head = nn.Linear(hidden_dim, 1)
        self.rot_head = nn.Linear(hidden_dim, 1)
        self.angle_head = nn.Linear(hidden_dim, 1)

        # Learnable temperature (starts high = conservative)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(init_temperature)).log()
        )

    def forward(self, detection_conf: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute per-channel gate values.

        Args:
            detection_conf: (B, 2) per-tool detection confidence.

        Returns:
            Dict with gate values:
                pos_gate: (B, 1) in [0, max_gate_pos].
                rot_gate: (B, 1) in [0, max_gate_rot].
                angle_gate: (B, 1) in [0, max_gate_angle].
        """
        temperature = self.log_temperature.exp()
        h = self.shared(detection_conf)

        pos_gate = (
            torch.sigmoid(self.pos_head(h) / temperature) * self.max_gate_pos
        )
        rot_gate = (
            torch.sigmoid(self.rot_head(h) / temperature) * self.max_gate_rot
        )
        angle_gate = (
            torch.sigmoid(self.angle_head(h) / temperature)
            * self.max_gate_angle
        )

        return {
            "pos_gate": pos_gate,
            "rot_gate": rot_gate,
            "angle_gate": angle_gate,
        }


# =============================================================================
# Residual Pose Head
# =============================================================================


class ResidualPoseHead(nn.Module):
    """Predict gated residual corrections to kinematic prior predictions.

    Applies per-channel gated corrections to the frozen kinematic model's
    predictions:

        Position: final = kin_pos + pos_gate * delta_pos
        Rotation: final = normalize(kin_quat + rot_gate * delta_quat)
        Angle:    final = kin_angle + angle_gate * delta_angle

    Each correction is bounded by a maximum delta to prevent catastrophic
    deviations from the kinematic prior. Uncertainty parameters (sigma,
    kappa) are modulated multiplicatively.

    Paper: Section 3.5, "Residual Correction Architecture"

    Checkpoint compatibility:
        Renamed from ResidualPoseHeadV3 (V3). Internally uses attribute
        names: self.pos_head, self.quat_head, self.angle_head,
        self.jaw_head, self.displacement_head (all from V3).

    Args:
        d_model: Input dimension.
        use_relative_tracking: Enable displacement prediction head.
        max_pos_delta: Maximum position correction in normalized units.
        max_quat_delta: Maximum quaternion correction magnitude.
        max_angle_delta: Maximum angle correction.
        max_displacement_delta: Maximum displacement in normalized units.
        min_sigma: Minimum sigma for uncertainty.
        max_sigma: Maximum sigma for uncertainty.
        min_kappa: Minimum VMF concentration.
    """

    def __init__(
        self,
        d_model: int = 256,
        use_relative_tracking: bool = True,
        max_pos_delta: float = 5.0,
        max_quat_delta: float = 0.15,
        max_angle_delta: float = 1.0,
        max_displacement_delta: float = 5.0,
        min_sigma: float = 1e-3,
        max_sigma: float = 10.0,
        min_kappa: float = 1.0,
    ):
        super().__init__()
        self.use_relative_tracking = use_relative_tracking
        self.max_pos_delta = max_pos_delta
        self.max_quat_delta = max_quat_delta
        self.max_angle_delta = max_angle_delta
        self.max_displacement_delta = max_displacement_delta
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kappa = min_kappa

        # Position corrections: delta_pos(6) + delta_log_sigma(6)
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 12),  # 2 tools x (3 pos + 3 sigma)
        )

        # Quaternion corrections: delta_quat(8) + delta_log_kappa(2)
        self.quat_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 10),  # 2 tools x (4 quat + 1 kappa)
        )

        # Angle corrections: delta_angle(2) + delta_log_sigma_angle(2)
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),  # 2 tools x (1 angle + 1 sigma)
        )

        # Jaw state (binary) -- direct prediction, not residual
        self.jaw_head = nn.Linear(d_model, 2)

        # Displacement prediction head (relative tracking)
        if use_relative_tracking:
            self.displacement_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 6),  # 2 tools x 3D displacement
            )

    def forward(
        self,
        fused_repr: torch.Tensor,
        kin_outputs: dict[str, torch.Tensor],
        gates: dict[str, torch.Tensor],
        current_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict residual corrections with per-channel gating.

        Args:
            fused_repr: (B, d_model) fused kinematic-visual representation.
            kin_outputs: Dict from kinematic prior containing:
                mu_position (B,2,3), sigma_position (B,2,3),
                mu_quaternion (B,2,4), kappa_quaternion (B,2,1),
                mu_angle (B,2,1), sigma_angle (B,2,1).
            gates: Dict from ConfidenceGate containing:
                pos_gate (B,1), rot_gate (B,1), angle_gate (B,1).
            current_position: (B, 2, 3) current frame position for
                relative tracking. Required if use_relative_tracking
                is True.

        Returns:
            Dict with corrected predictions, raw deltas (for
            regularization), gate values (for logging), and optional
            displacement predictions.
        """
        B = fused_repr.shape[0]

        pos_gate = gates["pos_gate"]      # (B, 1)
        rot_gate = gates["rot_gate"]      # (B, 1)
        angle_gate = gates["angle_gate"]  # (B, 1)

        # --- Position ---
        pos_out = self.pos_head(fused_repr)  # (B, 12)
        delta_pos = pos_out[:, :6].reshape(B, 2, 3)
        delta_log_sigma = pos_out[:, 6:].reshape(B, 2, 3)

        delta_pos = torch.clamp(
            delta_pos, -self.max_pos_delta, self.max_pos_delta
        )

        # Gated correction: pos_gate is (B, 1) -> unsqueeze to (B, 1, 1)
        final_pos = (
            kin_outputs["mu_position"]
            + pos_gate.unsqueeze(-1) * delta_pos
        )

        # Sigma modulation: kin_sigma * (1 + softplus(delta_log_sigma))
        sigma_scale = 1.0 + F.softplus(delta_log_sigma)
        final_sigma = kin_outputs["sigma_position"] * sigma_scale
        final_sigma = torch.clamp(final_sigma, self.min_sigma, self.max_sigma)

        # --- Quaternion ---
        quat_out = self.quat_head(fused_repr)  # (B, 10)
        delta_quat = quat_out[:, :8].reshape(B, 2, 4)
        delta_log_kappa = quat_out[:, 8:].reshape(B, 2, 1)

        delta_quat = torch.clamp(
            delta_quat, -self.max_quat_delta, self.max_quat_delta
        )

        final_quat = (
            kin_outputs["mu_quaternion"]
            + rot_gate.unsqueeze(-1) * delta_quat
        )
        final_quat = F.normalize(final_quat, dim=-1)

        kappa_scale = 1.0 + F.softplus(delta_log_kappa)
        final_kappa = kin_outputs["kappa_quaternion"] * kappa_scale
        final_kappa = torch.clamp(final_kappa, min=self.min_kappa)

        # --- Angle ---
        angle_out = self.angle_head(fused_repr)  # (B, 4)
        delta_angle = angle_out[:, :2].reshape(B, 2, 1)
        delta_log_sigma_angle = angle_out[:, 2:].reshape(B, 2, 1)

        delta_angle = torch.clamp(
            delta_angle, -self.max_angle_delta, self.max_angle_delta
        )
        final_angle = (
            kin_outputs["mu_angle"]
            + angle_gate.unsqueeze(-1) * delta_angle
        )

        sigma_angle_scale = 1.0 + F.softplus(delta_log_sigma_angle)
        final_sigma_angle = kin_outputs["sigma_angle"] * sigma_angle_scale
        final_sigma_angle = torch.clamp(
            final_sigma_angle, self.min_sigma, self.max_sigma
        )

        # --- Jaw state ---
        jaw_logits = self.jaw_head(fused_repr)  # (B, 2)

        outputs: dict[str, torch.Tensor] = {
            "mu_position": final_pos,
            "sigma_position": final_sigma,
            "mu_quaternion": final_quat,
            "kappa_quaternion": final_kappa,
            "mu_angle": final_angle,
            "sigma_angle": final_sigma_angle,
            "jaw_state_logits": jaw_logits,
            # Deltas for regularization losses
            "delta_position": delta_pos,
            "delta_quaternion": delta_quat,
            "delta_angle": delta_angle,
            # Per-channel gates for logging
            "gates": gates,
        }

        # --- Displacement (relative tracking) ---
        if self.use_relative_tracking and current_position is not None:
            disp_out = self.displacement_head(fused_repr)  # (B, 6)
            visual_displacement = disp_out.reshape(B, 2, 3)
            visual_displacement = torch.clamp(
                visual_displacement,
                -self.max_displacement_delta,
                self.max_displacement_delta,
            )
            visual_predicted_pos = current_position + visual_displacement

            outputs["visual_displacement"] = visual_displacement
            outputs["visual_predicted_pos"] = visual_predicted_pos
            outputs["current_position"] = current_position

        return outputs


# =============================================================================
# Relative Displacement Head
# =============================================================================


class RelativeDisplacementHead(nn.Module):
    """Predict frame-to-frame displacement for relative tracking.

    This standalone displacement head can be used independently of the
    full ResidualPoseHead, for example in displacement-only training
    phases or for auxiliary displacement loss computation.

    The predicted displacement is added to the current frame's position
    to predict the next frame's position, providing a relative tracking
    signal that complements the absolute pose prediction.

    Paper: Section 3.5, "Relative Displacement Tracking"

    Args:
        d_model: Input dimension.
        max_displacement: Maximum displacement magnitude per axis.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_displacement: float = 5.0,
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),  # 2 tools x 3D
        )

    def forward(
        self,
        fused_repr: torch.Tensor,
        current_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict displacement from current position.

        Args:
            fused_repr: (B, d_model) fused representation.
            current_position: (B, 2, 3) current frame tool positions.

        Returns:
            Dict with:
                displacement: (B, 2, 3) predicted displacement vector.
                predicted_position: (B, 2, 3) current + displacement.
        """
        B = fused_repr.shape[0]
        disp = self.head(fused_repr).reshape(B, 2, 3)
        disp = torch.clamp(
            disp, -self.max_displacement, self.max_displacement
        )
        predicted = current_position + disp

        return {
            "displacement": disp,
            "predicted_position": predicted,
        }


# =============================================================================
# Gate Entropy Regularization
# =============================================================================


def gate_entropy_loss(gates: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute entropy regularization to prevent gate saturation.

    Penalizes gates that collapse to 0 or their ceiling value by
    encouraging high entropy in the gate distribution. Uses binary
    cross-entropy formulation on the normalized gate value.

    The loss is: -mean(g*log(g+eps) + (1-g)*log(1-g+eps))
    where g is the gate value normalized to [0, 1] by dividing by ceiling.

    A gate at midpoint has maximum entropy (loss = 0).
    A gate at 0 or ceiling (saturated) has minimum entropy (loss > 0).

    Paper: Section 3.5, "Gate Entropy Regularization"
    Equation 12: Anti-saturation loss

    Args:
        gates: Dict with pos_gate, rot_gate, angle_gate tensors,
            each (B, 1) in [0, ceiling].

    Returns:
        Scalar entropy loss (negative entropy, to be minimized).
    """
    eps = 1e-6
    total = torch.tensor(0.0, device=next(iter(gates.values())).device)

    ceilings = {
        "pos_gate": 0.5,
        "rot_gate": 0.3,
        "angle_gate": 0.5,
    }

    count = 0
    for key, gate in gates.items():
        if key not in ceilings:
            continue
        # Normalize to [0, 1] for entropy computation
        g = (gate / ceilings[key]).clamp(eps, 1.0 - eps)
        # Binary entropy (negated so minimizing this maximizes entropy)
        entropy = -(g * torch.log(g) + (1 - g) * torch.log(1 - g))
        total = total + entropy.mean()
        count += 1

    # Return negative entropy (minimize this to maximize entropy)
    return -total / max(count, 1)
