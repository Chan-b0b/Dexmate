"""Transparent-case detector: depth-hole footprint + known-shape scoring.

Pipeline (all in the camera image plane, then deproject):
    1. hole mask = depth invalid OR far from the known floor depth
    2. clean it (morphology), find connected blobs
    3. fit a min-area rectangle to each blob
    4. score each rect against the KNOWN footprint size at the known z
    5. keep the best-scoring blob above threshold -> pose (u, v, yaw_img, xyz)

The known footprint + known z are what make this work: they turn "find a clear
object" into "find the 3-DOF placement of a rectangle of known size", so a
partial / noisy hole still pins the pose and obvious junk is rejected by size.

This is the no-model baseline. If real frames show the hole/edges are too weak
(run run_test.py to find out), swap detect_case() for a trained transparent
-object segmenter that returns a mask, then reuse _rect_from_mask() downstream.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


@dataclass
class CaseDetection:
    found: bool
    center_px: tuple[float, float]          # (u, v)
    angle_deg: float                        # long-axis orientation, [0, 180)
    size_score: float                       # [0, 1] match to known footprint
    dims_px: tuple[float, float]            # (long, short) in pixels
    z_m: float                              # floor depth used
    center_cam_m: tuple[float, float, float]  # deprojected (x, y, z) in camera frame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def deproject(u: float, v: float, z: float) -> tuple[float, float, float]:
    """Pixel + depth -> camera-frame metres (matches perception.get_3d_zed_point)."""
    x = ((u - cfg.CX) / cfg.FX) * z
    y = ((v - cfg.CY) / cfg.FY) * z
    return (x, y, z)


def _long_axis_angle(rect) -> float:
    """Orientation of the rectangle's longer side, normalised to [0, 180)."""
    (w, h), ang = rect[1], rect[2]
    if w < h:
        ang += 90.0
    return ang % 180.0


def _expected_dims_px(z: float) -> tuple[float, float]:
    """Known footprint projected to (long, short) pixel lengths at depth z."""
    long_m, short_m = max(cfg.CASE_FOOTPRINT_M), min(cfg.CASE_FOOTPRINT_M)
    return (long_m * cfg.FX / z, short_m * cfg.FY / z)


def _size_score(dims_px: tuple[float, float], z: float) -> tuple[float, bool]:
    """Score blob dimensions against the expected footprint. -> (score, within_tol)."""
    exp_long, exp_short = _expected_dims_px(z)
    det_long, det_short = max(dims_px), min(dims_px)
    e_long = abs(det_long - exp_long) / exp_long
    e_short = abs(det_short - exp_short) / exp_short
    score = max(0.0, 1.0 - 0.5 * (e_long + e_short))
    return score, (e_long <= cfg.SIZE_TOL and e_short <= cfg.SIZE_TOL)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def _roi_slice(shape):
    if cfg.CASE_SEARCH_ROI is None:
        return (slice(0, shape[0]), slice(0, shape[1]))
    x, y, w, h = cfg.CASE_SEARCH_ROI
    return (slice(y, y + h), slice(x, x + w))


def floor_depth(depth: np.ndarray) -> float:
    """Known floor depth, or the median valid depth inside the ROI."""
    if cfg.FLOOR_Z_M is not None:
        return float(cfg.FLOOR_Z_M)
    roi = depth[_roi_slice(depth.shape)]
    valid = roi[np.isfinite(roi) & (roi >= cfg.DEPTH_MIN_M) & (roi <= cfg.DEPTH_MAX_M)]
    if valid.size == 0:
        return float(cfg.DEPTH_MAX_M)
    return float(np.median(valid))


def hole_mask(depth: np.ndarray, z_floor: float) -> np.ndarray:
    """Boolean mask of the case 'hole': invalid depth OR off the floor plane."""
    invalid = ~np.isfinite(depth) | (depth < cfg.DEPTH_MIN_M) | (depth > cfg.DEPTH_MAX_M)
    off_floor = np.isfinite(depth) & (np.abs(depth - z_floor) > cfg.FLOOR_DEV_TOL_M)
    mask = (invalid | off_floor).astype(np.uint8)

    # Limit to ROI.
    if cfg.CASE_SEARCH_ROI is not None:
        keep = np.zeros_like(mask)
        keep[_roi_slice(depth.shape)] = 1
        mask &= keep

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.MORPH_KERNEL_PX,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _rect_from_mask(mask: np.ndarray, z_floor: float):
    """Best (highest size-score) min-area rectangle in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exp_long, exp_short = _expected_dims_px(z_floor)
    min_area = cfg.MIN_AREA_FRAC * exp_long * exp_short

    best = None
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if min(w, h) < 1.0:
            continue
        score, _ = _size_score((w, h), z_floor)
        if best is None or score > best[0]:
            best = (score, rect)
    return best  # (score, rect) or None


def detect_case(depth: np.ndarray, rgb: np.ndarray | None = None) -> CaseDetection:
    """Detect the transparent case from a metric depth map. rgb is unused for
    now (reserved for an edge/learned backend)."""
    depth = np.asarray(depth, dtype=np.float32)
    z_floor = floor_depth(depth)
    mask = hole_mask(depth, z_floor)
    best = _rect_from_mask(mask, z_floor)

    if best is None:
        return CaseDetection(False, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0), z_floor, (0.0, 0.0, 0.0))

    score, rect = best
    (u, v), (w, h) = rect[0], rect[1]
    angle = _long_axis_angle(rect)
    return CaseDetection(
        found=score >= cfg.MIN_SCORE,
        center_px=(float(u), float(v)),
        angle_deg=float(angle),
        size_score=float(score),
        dims_px=(float(max(w, h)), float(min(w, h))),
        z_m=z_floor,
        center_cam_m=deproject(u, v, z_floor),
    )


if __name__ == "__main__":
    # Quick single-frame check: python detect_case.py data/frame_000.npz
    path = sys.argv[1]
    f = np.load(path)
    det = detect_case(f["depth"], f.get("rgb"))
    print(det)
