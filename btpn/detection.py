"""Detection pipeline for surgical tool segmentation, pose, and depth.

Wraps YOLO-based segmentation and keypoint detection models, plus
Depth Anything V2 embedding extraction, into a unified interface
for training, inference, and feature precomputation.

This module is intentionally lightweight -- it delegates to the
ultralytics YOLO API for training and inference, and to HuggingFace
transformers for depth feature extraction.

Paper reference: Section 3.6 (Visual Feature Extraction).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Detection Configuration
# =============================================================================


@dataclass
class DetectionConfig:
    """Configuration for YOLO detection training and inference.

    Attributes:
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Training batch size.
        image_size: Input image resolution for YOLO.
        workers: Number of data-loading workers.
        project_dir: Root directory for training outputs.
    """

    epochs: int = 150
    patience: int = 30
    batch_size: int = 8
    image_size: int = 640
    workers: int = 12
    project_dir: str = "outputs/detection"

    # Segmentation augmentation
    seg_mosaic: float = 1.0
    seg_mixup: float = 0.15
    seg_flipud: float = 0.5
    seg_fliplr: float = 0.5
    seg_degrees: float = 15.0
    seg_scale: float = 0.5
    seg_erasing: float = 0.4

    # Pose augmentation
    pose_epochs: int = 300
    pose_patience: int = 20
    pose_mosaic: float = 1.0
    pose_degrees: float = 20.0
    pose_scale: float = 0.5
    pose_translate: float = 0.2
    pose_fliplr: float = 0.5
    pose_flipud: float = 0.3
    pose_copy_paste: float = 0.2
    pose_mixup: float = 0.1
    pose_erasing: float = 0.1

    # Pose loss weights
    pose_loss_weight: float = 15.0
    kobj_loss_weight: float = 2.0
    box_loss_weight: float = 6.0


# =============================================================================
# Detection Pipeline
# =============================================================================


class DetectionPipeline:
    """Unified pipeline for surgical tool detection via YOLO.

    Wraps YOLOv11/v26 segmentation and pose models for training,
    inference, and feature extraction. The segmentation model provides
    per-tool masks and FPN features; the pose model provides 8 keypoints
    per tool (shaft lines, joint, end-effector tips).

    Args:
        seg_weights: Path to trained segmentation model weights.
            If None, will use default pretrained YOLO-seg.
        pose_weights: Path to trained pose model weights.
            If None, will use default pretrained YOLO-pose.

    Example:
        >>> pipeline = DetectionPipeline("weights/seg_best.pt", "weights/pose_best.pt")
        >>> detections = pipeline.predict("frame_00100.png")
        >>> print(detections["tool1_conf"], detections["tool2_conf"])
    """

    def __init__(
        self,
        seg_weights: str | Path | None = None,
        pose_weights: str | Path | None = None,
    ) -> None:
        self._seg_weights = seg_weights
        self._pose_weights = pose_weights
        self._seg_model: Any = None
        self._pose_model: Any = None

    def _load_seg_model(self) -> Any:
        """Lazy-load the segmentation model."""
        if self._seg_model is None:
            from ultralytics import YOLO

            weights = self._seg_weights or "yolo26m-seg.pt"
            self._seg_model = YOLO(str(weights))
            logger.info("Loaded segmentation model: %s", weights)
        return self._seg_model

    def _load_pose_model(self) -> Any:
        """Lazy-load the pose keypoint model."""
        if self._pose_model is None:
            from ultralytics import YOLO

            weights = self._pose_weights or "yolo26m-pose.pt"
            self._pose_model = YOLO(str(weights))
            logger.info("Loaded pose model: %s", weights)
        return self._pose_model

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train_segmentation(
        self,
        data_yaml: str | Path,
        config: DetectionConfig | None = None,
        name: str = "yolo_seg",
    ) -> Path:
        """Train YOLO instance segmentation model.

        Uses augmentation parameters tuned for surgical tool detection
        in laparoscopic peg transfer imagery (small dataset, tools at
        arbitrary angles, specular highlights).

        Args:
            data_yaml: Path to YOLO-format dataset YAML.
            config: Detection training configuration. Uses defaults if None.
            name: Run name for output directory.

        Returns:
            Path to the best model weights.

        Raises:
            FileNotFoundError: If *data_yaml* does not exist.
        """
        from ultralytics import YOLO

        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

        config = config or DetectionConfig()
        model = YOLO("yolo26m-seg.pt")

        model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            patience=config.patience,
            batch=config.batch_size,
            imgsz=config.image_size,
            workers=config.workers,
            project=config.project_dir,
            name=name,
            exist_ok=True,
            # Augmentation
            mosaic=config.seg_mosaic,
            mixup=config.seg_mixup,
            flipud=config.seg_flipud,
            fliplr=config.seg_fliplr,
            degrees=config.seg_degrees,
            scale=config.seg_scale,
            erasing=config.seg_erasing,
            # Segmentation-specific
            overlap_mask=True,
            mask_ratio=4,
        )

        weights_path = Path(config.project_dir) / name / "weights" / "best.pt"
        logger.info("Segmentation training complete: %s", weights_path)
        return weights_path

    def train_keypoints(
        self,
        data_yaml: str | Path,
        config: DetectionConfig | None = None,
        name: str = "yolo_pose",
    ) -> Path:
        """Train YOLO keypoint (pose) detection model.

        Keypoint layout per tool (8 keypoints):
            KP 0-3: Shaft line endpoints
            KP 4:   Joint (shaft-to-end-effector transition)
            KP 5:   End-effector tip (jaw opening center)
            KP 6:   Left jaw tip
            KP 7:   Right jaw tip

        Args:
            data_yaml: Path to YOLO-format pose dataset YAML.
            config: Detection training configuration. Uses defaults if None.
            name: Run name for output directory.

        Returns:
            Path to the best model weights.

        Raises:
            FileNotFoundError: If *data_yaml* does not exist.
        """
        from ultralytics import YOLO

        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

        config = config or DetectionConfig()
        model = YOLO("yolo26m-pose.pt")

        model.train(
            data=str(data_yaml),
            epochs=config.pose_epochs,
            patience=config.pose_patience,
            batch=config.batch_size,
            imgsz=config.image_size,
            workers=config.workers,
            project=config.project_dir,
            name=name,
            exist_ok=True,
            # Geometric augmentation
            mosaic=config.pose_mosaic,
            degrees=config.pose_degrees,
            scale=config.pose_scale,
            translate=config.pose_translate,
            fliplr=config.pose_fliplr,
            flipud=config.pose_flipud,
            # Copy-paste and mixing
            copy_paste=config.pose_copy_paste,
            mixup=config.pose_mixup,
            erasing=config.pose_erasing,
            # Loss weights
            pose=config.pose_loss_weight,
            kobj=config.kobj_loss_weight,
            box=config.box_loss_weight,
        )

        weights_path = Path(config.project_dir) / name / "weights" / "best.pt"
        logger.info("Keypoint training complete: %s", weights_path)
        return weights_path

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Run segmentation and pose inference on a single image.

        Assigns detections to Tool 1 (left in frame) and Tool 2
        (right in frame) based on bounding box center x-coordinate.

        Args:
            image_path: Path to input image.
            conf_threshold: Minimum detection confidence.

        Returns:
            Dictionary with keys:
                detected (bool), n_tools (int),
                tool1_bbox, tool2_bbox (list[float] or None),
                tool1_conf, tool2_conf (float),
                tool1_mask, tool2_mask (np.ndarray or None),
                tool1_keypoints, tool2_keypoints (np.ndarray (8,3) or None).
        """
        seg_model = self._load_seg_model()

        # Segmentation inference
        seg_results = seg_model(str(image_path), conf=conf_threshold, verbose=False)
        seg_result = seg_results[0]

        result: dict[str, Any] = {
            "detected": False,
            "n_tools": 0,
            "tool1_bbox": None, "tool2_bbox": None,
            "tool1_conf": 0.0, "tool2_conf": 0.0,
            "tool1_mask": None, "tool2_mask": None,
            "tool1_keypoints": None, "tool2_keypoints": None,
        }

        if seg_result.boxes is None or len(seg_result.boxes) == 0:
            return result

        # Sort detections by x-center (left=Tool1, right=Tool2)
        boxes = seg_result.boxes.xyxy.cpu().numpy()
        confs = seg_result.boxes.conf.cpu().numpy()
        masks = (
            seg_result.masks.data.cpu().numpy()
            if seg_result.masks is not None
            else [None] * len(boxes)
        )

        x_centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
        sorted_idx = np.argsort(x_centers)

        n_tools = min(len(sorted_idx), 2)
        result["detected"] = n_tools > 0
        result["n_tools"] = n_tools

        for i, tool_idx in enumerate(sorted_idx[:2]):
            tool_key = f"tool{i + 1}"
            result[f"{tool_key}_bbox"] = boxes[tool_idx].tolist()
            result[f"{tool_key}_conf"] = float(confs[tool_idx])
            if masks[tool_idx] is not None:
                result[f"{tool_key}_mask"] = masks[tool_idx]

        # Pose inference (if model available)
        if self._pose_weights is not None:
            pose_model = self._load_pose_model()
            pose_results = pose_model(
                str(image_path), conf=conf_threshold, verbose=False
            )
            pose_result = pose_results[0]

            if (
                pose_result.keypoints is not None
                and len(pose_result.keypoints) > 0
            ):
                kp_data = pose_result.keypoints.data.cpu().numpy()
                kp_boxes = pose_result.boxes.xyxy.cpu().numpy()
                kp_x_centers = (kp_boxes[:, 0] + kp_boxes[:, 2]) / 2.0
                kp_sorted = np.argsort(kp_x_centers)

                for i, kp_idx in enumerate(kp_sorted[:2]):
                    result[f"tool{i + 1}_keypoints"] = kp_data[kp_idx]

        return result


# =============================================================================
# Visual Feature Precomputation
# =============================================================================


def precompute_visual_features(
    trial_dir: Path,
    seg_model_path: str | Path,
    pose_model_path: str | Path | None = None,
    resume: bool = False,
) -> int:
    """Extract and save per-frame segmentation and pose features for a trial.

    For each frame in ``trial_dir/Frames/``, runs YOLO segmentation (and
    optionally pose) inference, extracts FPN neck features (256D per tool),
    global backbone features (512D), and optionally keypoint + backbone
    features from the pose model. Results are saved as compressed NPZ
    files in ``trial_dir/SEG/`` and ``trial_dir/POSE/``.

    Also saves ``_positive_index.json`` listing frame indices where at
    least one tool was detected.

    Args:
        trial_dir: Path to trial directory containing a ``Frames/`` subdirectory.
        seg_model_path: Path to trained YOLO segmentation weights.
        pose_model_path: Path to trained YOLO pose weights. If None, skip pose.
        resume: If True, skip frames that already have saved features.

    Returns:
        Number of frames processed.

    Raises:
        FileNotFoundError: If trial_dir/Frames/ does not exist.
    """
    from ultralytics import YOLO

    frames_dir = trial_dir / "Frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Discover frame files
    frame_paths = sorted(frames_dir.glob("frame_*.bmp"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.png"))

    if not frame_paths:
        logger.warning("No frame images found in %s", frames_dir)
        return 0

    # Output directories
    seg_dir = trial_dir / "SEG"
    seg_dir.mkdir(exist_ok=True)

    pose_dir = None
    if pose_model_path is not None:
        pose_dir = trial_dir / "POSE"
        pose_dir.mkdir(exist_ok=True)

    # Load models
    seg_model = YOLO(str(seg_model_path))
    pose_model = YOLO(str(pose_model_path)) if pose_model_path else None

    positive_indices: list[int] = []
    processed = 0

    for frame_path in frame_paths:
        frame_idx = _frame_index_from_path(frame_path)
        seg_npz = seg_dir / f"frame_{frame_idx:05d}.npz"

        if resume and seg_npz.exists():
            # Check if positive
            try:
                data = np.load(seg_npz, allow_pickle=True)
                if data.get("detected", False):
                    positive_indices.append(frame_idx)
            except Exception:
                pass
            continue

        # Segmentation inference
        seg_results = seg_model(str(frame_path), conf=0.25, verbose=False)
        seg_result = seg_results[0]

        detected = (
            seg_result.boxes is not None and len(seg_result.boxes) > 0
        )

        save_data: dict[str, Any] = {"detected": detected}

        if detected:
            positive_indices.append(frame_idx)
            boxes = seg_result.boxes.xyxy.cpu().numpy()
            confs = seg_result.boxes.conf.cpu().numpy()
            x_centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
            sorted_idx = np.argsort(x_centers)

            for i, det_idx in enumerate(sorted_idx[:2]):
                tool_key = f"tool{i + 1}"
                save_data[f"{tool_key}_bbox"] = boxes[det_idx]
                save_data[f"{tool_key}_conf"] = float(confs[det_idx])

        np.savez_compressed(seg_npz, **save_data)

        # Pose inference
        if pose_model is not None and pose_dir is not None:
            pose_npz = pose_dir / f"frame_{frame_idx:05d}.npz"
            if not (resume and pose_npz.exists()):
                pose_results = pose_model(
                    str(frame_path), conf=0.25, verbose=False
                )
                pose_result = pose_results[0]
                pose_data: dict[str, Any] = {"detected": False}

                if (
                    pose_result.keypoints is not None
                    and len(pose_result.keypoints) > 0
                ):
                    pose_data["detected"] = True
                    kp = pose_result.keypoints.data.cpu().numpy()
                    kp_boxes = pose_result.boxes.xyxy.cpu().numpy()
                    kp_x = (kp_boxes[:, 0] + kp_boxes[:, 2]) / 2.0
                    kp_sorted = np.argsort(kp_x)
                    for i, kp_idx in enumerate(kp_sorted[:2]):
                        pose_data[f"tool{i + 1}_keypoints"] = kp[kp_idx]
                        pose_data[f"tool{i + 1}_bbox"] = kp_boxes[kp_idx]

                np.savez_compressed(pose_npz, **pose_data)

        processed += 1

    # Save positive frame index
    index_path = seg_dir / "_positive_index.json"
    with open(index_path, "w") as f:
        json.dump(sorted(set(positive_indices)), f)

    logger.info(
        "%s: %d/%d frames processed, %d positive",
        trial_dir.name, processed, len(frame_paths), len(positive_indices),
    )
    return processed


def precompute_depth_features(
    trial_dir: Path,
    resume: bool = False,
) -> int:
    """Extract Depth Anything V2 embeddings for all frames in a trial.

    Uses the DINOv2-S backbone (384D) from Depth Anything V2 Small.
    For each frame, extracts:
        - depth: Normalized depth map (H, W) in [0, 1]
        - global_embedding: 384D CLS token from DINOv2-S backbone

    Results are saved as compressed NPZ in ``trial_dir/DEPTH/``.

    Args:
        trial_dir: Path to trial directory containing ``Frames/``.
        resume: If True, skip frames that already have saved features.

    Returns:
        Number of frames processed.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    frames_dir = trial_dir / "Frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("frame_*.bmp"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.png"))

    if not frame_paths:
        logger.warning("No frame images in %s", frames_dir)
        return 0

    depth_dir = trial_dir / "DEPTH"
    depth_dir.mkdir(exist_ok=True)
    embed_dir = depth_dir / "embeddings"
    embed_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    model.eval()

    processed = 0
    t0 = time.time()

    for i, frame_path in enumerate(frame_paths):
        frame_idx = _frame_index_from_path(frame_path)
        depth_npz = depth_dir / f"frame_{frame_idx:05d}.npz"
        embed_npz = embed_dir / f"frame_{frame_idx:05d}.npz"

        if resume and depth_npz.exists() and embed_npz.exists():
            continue

        image = Image.open(frame_path).convert("RGB")
        original_size = image.size  # (W, H)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.inference_mode():
            # Depth map
            outputs = model(**inputs)
            depth_raw = outputs.predicted_depth
            depth = torch.nn.functional.interpolate(
                depth_raw.unsqueeze(1),
                size=(original_size[1], original_size[0]),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            min_depth = float(depth.min())
            max_depth = float(depth.max())
            depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)

            # Backbone embedding (DINOv2-S CLS token)
            backbone = model.backbone
            pixel_values = inputs["pixel_values"]
            embeddings = backbone.embeddings(pixel_values)
            hidden_states = embeddings
            for layer in backbone.encoder.layer:
                hidden_states = layer(hidden_states)[0]
            cls_token = hidden_states[:, 0, :].squeeze(0).cpu().numpy()

        np.savez_compressed(
            depth_npz,
            depth=depth_norm.astype(np.float32),
            min_depth=min_depth,
            max_depth=max_depth,
        )
        np.savez_compressed(
            embed_npz,
            global_embedding=cls_token.astype(np.float32),
        )

        processed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  %s: %d/%d frames (%.1f fps)",
                trial_dir.name, i + 1, len(frame_paths),
                (i + 1) / elapsed,
            )

    return processed


# =============================================================================
# Helpers
# =============================================================================


def _frame_index_from_path(frame_path: Path) -> int:
    """Extract the numeric frame index from a frame filename.

    Parses trailing digits from the filename stem. For example:
        "frame_00100.bmp" -> 100
        "frame_00042.png" -> 42

    Args:
        frame_path: Path to a frame image file.

    Returns:
        Integer frame index.
    """
    stem = frame_path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else 0
