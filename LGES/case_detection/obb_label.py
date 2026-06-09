"""Mask <-> YOLO-OBB label conversion + label sanity flags.

A SAM2 mask becomes a YOLO-OBB label by fitting a min-area rectangle and
writing its 4 corners (normalized). Pure numpy/opencv, no SAM dependency, so
it's unit-testable on its own and shared by sam2_autolabel.py and review.py.

YOLO-OBB label row:  <class> x1 y1 x2 y2 x3 y3 x4 y4   (corners normalized 0..1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def mask_to_obb_points(mask: np.ndarray) -> np.ndarray | None:
    """Largest blob of `mask` -> 4 corner points (px, float32 (4,2)), or None."""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < cfg.LABEL_MIN_AREA_FRAC * mask.size:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def points_to_yolo_line(points: np.ndarray, w: int, h: int, cls: int = 0) -> str:
    """4 corner points (px) -> normalized YOLO-OBB row."""
    p = points.astype(np.float64).copy()
    p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in p.reshape(-1))


def yolo_line_to_points(line: str, w: int, h: int) -> np.ndarray:
    """Normalized YOLO-OBB row -> 4 corner points (px, int (4,2)). For drawing."""
    vals = np.array(line.split()[1:9], dtype=np.float64).reshape(4, 2)
    vals[:, 0] *= w
    vals[:, 1] *= h
    return vals.round().astype(np.int32)


def mask_to_yolo_line(mask: np.ndarray, cls: int = 0) -> str | None:
    pts = mask_to_obb_points(mask)
    if pts is None:
        return None
    h, w = mask.shape[:2]
    return points_to_yolo_line(pts, w, h, cls)


def mask_to_bbox_line(mask: np.ndarray, cls: int = 0) -> str | None:
    """Largest blob -> axis-aligned YOLO detect row: cls cx cy w h (normalized).
    Used for the bin detector (regular detection, not oriented)."""
    pts = mask_to_obb_points(mask)
    if pts is None:
        return None
    h, w = mask.shape[:2]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def label_flags(points: np.ndarray, w: int, h: int) -> list[str]:
    """Heuristic problems with a label (drift / bad mask), for review highlighting."""
    flags = []
    area_frac = cv2.contourArea(points.astype(np.float32)) / float(w * h)
    if area_frac < cfg.LABEL_MIN_AREA_FRAC:
        flags.append("tiny")
    if area_frac > cfg.LABEL_MAX_AREA_FRAC:
        flags.append("huge")
    x, y = points[:, 0], points[:, 1]
    if x.min() <= 1 or y.min() <= 1 or x.max() >= w - 2 or y.max() >= h - 2:
        flags.append("border")  # touching the edge often means propagation drift
    return flags
