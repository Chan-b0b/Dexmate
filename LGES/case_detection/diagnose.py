"""Render a 4-panel diagnostic image for one frame.

Panels: RGB | depth (colormapped) | hole mask | detection overlay.
This is the thing you actually LOOK AT to judge whether the transparent case
leaves a usable signal under your lighting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from detect_case import CaseDetection, detect_case, floor_depth, hole_mask


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = cfg.DEPTH_MIN_M, cfg.DEPTH_MAX_M
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = (0, 0, 0)  # invalid -> black, so depth holes stand out
    return img


def _overlay(rgb: np.ndarray, det: CaseDetection) -> np.ndarray:
    img = rgb.copy()
    color = (0, 255, 0) if det.found else (0, 165, 255)
    if det.size_score > 0:
        rect = (det.center_px, (det.dims_px[0], det.dims_px[1]), det.angle_deg)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(img, [box], 0, color, 2)
        u, v = int(det.center_px[0]), int(det.center_px[1])
        cv2.circle(img, (u, v), 4, color, -1)
    txt = (f"found={det.found} score={det.size_score:.2f} "
           f"ang={det.angle_deg:.1f} z={det.z_m:.3f}")
    cv2.putText(img, txt, (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def render(rgb: np.ndarray | None, depth: np.ndarray, det: CaseDetection | None = None) -> np.ndarray:
    """Build the 4-panel diagnostic image (BGR)."""
    if det is None:
        det = detect_case(depth, rgb)
    h, w = depth.shape[:2]
    if rgb is None:
        rgb = np.zeros((h, w, 3), np.uint8)
    elif rgb.ndim == 3:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # camera RGB -> BGR for OpenCV
    else:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)

    z = floor_depth(depth)
    mask = hole_mask(depth, z)
    mask_bgr = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    panels = [
        _label(rgb.copy(), "rgb"),
        _label(_colorize_depth(depth), "depth (black=invalid)"),
        _label(mask_bgr, "hole mask"),
        _label(_overlay(rgb, det), "detection"),
    ]
    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    return np.vstack([top, bot])


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    out = render(f.get("rgb"), f["depth"])
    dst = sys.argv[2] if len(sys.argv) > 2 else "diagnose.png"
    cv2.imwrite(dst, out)
    print("wrote", dst)
