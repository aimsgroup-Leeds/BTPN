#!/usr/bin/env python3
"""Create COCO-format annotations for YOLO training.

Provides utilities to:
1. Extract frames from surgical videos
2. Create/convert annotations to YOLO format
3. Split into train/val sets
4. Verify annotation quality

Usage:
    python scripts/annotate.py extract --video path/to/video.mp4 --output frames/
    python scripts/annotate.py convert --input annotations.json --format coco --output yolo_labels/
    python scripts/annotate.py split --data-dir dataset/ --train-ratio 0.8
    python scripts/annotate.py verify --data-dir dataset/ --task segmentation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Keypoint definition — matches configs/detection.yaml
KEYPOINT_NAMES = [
    "tip_left", "tip_right", "jaw_hinge", "shaft_top",
    "shaft_mid", "shaft_bottom", "shaft_entry", "shaft_base",
]
SKELETON = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7]]
CATEGORIES = [
    {"id": 1, "name": "Tool", "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON},
]
NUM_KEYPOINTS = 8
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# -- extract ------------------------------------------------------------------

def extract_frames(video_path: Path, output_dir: Path,
                   fps: float = 1.0, prefix: str = "frame") -> int:
    """Extract frames from *video_path* at *fps* into *output_dir*."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(video_fps / fps)))
    output_dir.mkdir(parents=True, exist_ok=True)

    count, idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            cv2.imwrite(str(output_dir / f"{prefix}_{count:05d}.jpg"), frame)
            count += 1
        idx += 1
    cap.release()
    print(f"Extracted {count} frames to {output_dir}")
    return count


# -- convert ------------------------------------------------------------------

def _coco_bbox_to_yolo(bbox: list[float], w: int, h: int) -> tuple[float, ...]:
    """COCO [x,y,w,h] -> YOLO normalised [cx,cy,w,h]."""
    x, y, bw, bh = bbox
    return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)

def _polygon_to_yolo(seg: list[list[float]], w: int, h: int) -> list[float]:
    """Flatten COCO polygon and normalise for YOLO-seg format."""
    pts: list[float] = []
    for poly in seg:
        for i in range(0, len(poly), 2):
            pts.extend([poly[i] / w, poly[i + 1] / h])
    return pts

def _kps_to_yolo(kps: list[float], w: int, h: int) -> list[float]:
    """COCO keypoints [x,y,v,...] -> YOLO normalised."""
    out: list[float] = []
    for i in range(0, len(kps), 3):
        out.extend([kps[i] / w, kps[i + 1] / h, int(kps[i + 2])])
    return out

def convert_annotations(input_path: Path, output_dir: Path,
                        fmt: str = "coco", task: str = "detect") -> int:
    """Convert COCO JSON annotations to YOLO per-image .txt labels.

    Args:
        input_path: COCO-format JSON file.
        output_dir: Directory for YOLO .txt files.
        fmt: Source format (only ``coco`` supported).
        task: ``detect``, ``segment``, or ``pose``.
    """
    if fmt != "coco":
        raise ValueError(f"Unsupported format: {fmt}")
    if task not in ("detect", "segment", "pose"):
        raise ValueError(f"Unsupported task: {task}")

    with open(input_path, encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    cat_remap = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for img_id, info in images.items():
        iw, ih = info["width"], info["height"]
        lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            cls = cat_remap[ann["category_id"]]
            cx, cy, bw, bh = _coco_bbox_to_yolo(ann["bbox"], iw, ih)

            if task == "detect":
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif task == "segment":
                seg = ann.get("segmentation", [])
                if seg:
                    coords = " ".join(f"{v:.6f}" for v in _polygon_to_yolo(seg, iw, ih))
                    lines.append(f"{cls} {coords}")
            elif task == "pose":
                parts = [f"{cls}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                kps = ann.get("keypoints", [])
                if kps:
                    parts.extend(
                        f"{v:.6f}" if isinstance(v, float) else str(int(v))
                        for v in _kps_to_yolo(kps, iw, ih)
                    )
                else:
                    parts.extend(["0.000000", "0.000000", "0"] * NUM_KEYPOINTS)
                lines.append(" ".join(parts))

        (output_dir / f"{Path(info['file_name']).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        written += 1

    print(f"Wrote {written} YOLO labels to {output_dir} (task={task})")
    return written


# -- split --------------------------------------------------------------------

def split_dataset(data_dir: Path, train_ratio: float = 0.8,
                  seed: int = 42) -> dict[str, int]:
    """Move images/labels from flat dirs into train/val subdirectories."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images/ directory in {data_dir}")

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_train = int(len(files) * train_ratio)
    splits = {"train": files[:n_train], "val": files[n_train:]}

    for name, split_files in splits.items():
        for sub in ("images", "labels"):
            (data_dir / name / sub).mkdir(parents=True, exist_ok=True)
        for img in split_files:
            img.rename(data_dir / name / "images" / img.name)
            lbl = labels_dir / f"{img.stem}.txt"
            if lbl.exists():
                lbl.rename(data_dir / name / "labels" / lbl.name)

    counts = {k: len(v) for k, v in splits.items()}
    print(f"Split: {counts['train']} train, {counts['val']} val")
    return counts


# -- verify -------------------------------------------------------------------

def _draw_bbox(img: np.ndarray, cx: float, cy: float, w: float, h: float,
               iw: int, ih: int) -> None:
    x1, y1 = int((cx - w / 2) * iw), int((cy - h / 2) * ih)
    x2, y2 = int((cx + w / 2) * iw), int((cy + h / 2) * ih)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def _draw_keypoints(img: np.ndarray, kps: list[float],
                    iw: int, ih: int) -> None:
    points: list[tuple[int, int] | None] = []
    for i in range(0, len(kps), 3):
        vis = int(kps[i + 2])
        px, py = int(kps[i] * iw), int(kps[i + 1] * ih)
        if vis > 0:
            col = (0, 255, 0) if vis == 2 else (0, 165, 255)
            cv2.circle(img, (px, py), 4, col, -1)
            points.append((px, py))
        else:
            points.append(None)
    for a, b in SKELETON:
        if a < len(points) and b < len(points):
            pa, pb = points[a], points[b]
            if pa and pb:
                cv2.line(img, pa, pb, (255, 200, 0), 1)

def verify_dataset(data_dir: Path, task: str = "detect",
                   num_samples: int = 5, seed: int = 42) -> dict[str, Any]:
    """Check for missing/empty/invalid labels and visualise samples."""
    pairs: list[tuple[Path, Path]] = []
    for split in ("", "train", "val"):
        img_dir = data_dir / split / "images" if split else data_dir / "images"
        lbl_dir = data_dir / split / "labels" if split else data_dir / "labels"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in _IMG_EXTS:
                pairs.append((p, lbl_dir / f"{p.stem}.txt"))

    missing = empty = invalid = 0
    for _, lbl in pairs:
        if not lbl.exists():
            missing += 1; continue
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            empty += 1; continue
        for line in text.splitlines():
            vals = [float(v) for v in line.split()[1:]]
            if any(v < 0 or v > 1 for v in vals if not v.is_integer()):
                invalid += 1; break

    summary = {"total": len(pairs), "missing": missing,
               "empty": empty, "invalid": invalid}
    print(f"Verify ({task}): {len(pairs)} images, {missing} missing, "
          f"{empty} empty, {invalid} out-of-range")

    if num_samples > 0 and pairs:
        vis_dir = data_dir / "verify_samples"
        vis_dir.mkdir(parents=True, exist_ok=True)
        chosen = random.Random(seed).sample(pairs, min(num_samples, len(pairs)))
        for img_path, lbl_path in chosen:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    vals = line.split()
                    if task in ("detect", "pose") and len(vals) >= 5:
                        _draw_bbox(img, *(float(v) for v in vals[1:5]), iw, ih)
                    if task == "pose" and len(vals) > 5:
                        _draw_keypoints(img, [float(v) for v in vals[5:]], iw, ih)
                    if task == "segment" and len(vals) >= 5:
                        c = [float(v) for v in vals[1:]]
                        pts = np.array([(int(c[i]*iw), int(c[i+1]*ih))
                                        for i in range(0, len(c), 2)], dtype=np.int32)
                        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.imwrite(str(vis_dir / img_path.name), img)
        print(f"Saved {len(chosen)} visualisations to {vis_dir}")
    return summary


# -- CLI ----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotate",
        description="Create and manage COCO/YOLO annotations for YOLO training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/annotate.py extract --video vid.mp4 --output frames/\n"
            "  python scripts/annotate.py convert --input ann.json --output labels/ --task pose\n"
            "  python scripts/annotate.py split --data-dir dataset/ --train-ratio 0.8\n"
            "  python scripts/annotate.py verify --data-dir dataset/ --task pose\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract frames from video")
    e.add_argument("--video", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--fps", type=float, default=1.0)
    e.add_argument("--prefix", default="frame")

    c = sub.add_parser("convert", help="Convert COCO JSON to YOLO txt")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--format", default="coco", choices=["coco"])
    c.add_argument("--task", default="detect", choices=["detect", "segment", "pose"])

    s = sub.add_parser("split", help="Split dataset into train/val")
    s.add_argument("--data-dir", type=Path, required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=42)

    v = sub.add_parser("verify", help="Verify annotation quality")
    v.add_argument("--data-dir", type=Path, required=True)
    v.add_argument("--task", default="detect", choices=["detect", "segment", "pose"])
    v.add_argument("--num-samples", type=int, default=5)
    v.add_argument("--seed", type=int, default=42)

    return p


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    if args.command == "extract":
        extract_frames(args.video, args.output, fps=args.fps, prefix=args.prefix)
    elif args.command == "convert":
        convert_annotations(args.input, args.output, fmt=args.format, task=args.task)
    elif args.command == "split":
        split_dataset(args.data_dir, train_ratio=args.train_ratio, seed=args.seed)
    elif args.command == "verify":
        verify_dataset(args.data_dir, task=args.task,
                       num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
