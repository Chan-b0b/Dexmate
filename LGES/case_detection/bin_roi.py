"""Locate the yellow source bin and crop to its interior.

The clear case always sits inside the opaque yellow bin, so finding the bin
first turns "detect a transparent object anywhere" into "search a small, clean
ROI". The bin is bright and opaque -> trivial HSV colour segmentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def find_bin(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the yellow bin, or None if not found.

    rgb is in camera RGB order (as stored by capture.py).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, cfg.BIN_HSV_LO, cfg.BIN_HSV_HI)
    # OPEN first to split the source bin from adjacent yellow objects + drop
    # speckle; then CLOSE to fill the bin interior.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = [c for c in contours if cv2.contourArea(c) >= cfg.BIN_MIN_AREA_FRAC * h * w]
    if not cand:
        return None
    # Several yellow bins can be in view; pick the one whose box center is
    # nearest the image center (the source bin the head is aimed at), not the
    # largest. If your source bin isn't centered, set cfg.BIN_ROI instead.
    cx, cy = w / 2.0, h / 2.0

    def center_dist(c):
        x, y, bw, bh = cv2.boundingRect(c)
        return abs(x + bw / 2 - cx) + abs(y + bh / 2 - cy)

    return cv2.boundingRect(min(cand, key=center_dist))


def inset_bbox(bbox: tuple[int, int, int, int], frac: float | None = None):
    """Shrink a bbox inward by `frac` on each side (default cfg.BIN_INSET_FRAC)."""
    if frac is None:
        frac = cfg.BIN_INSET_FRAC
    x, y, w, h = bbox
    ix, iy = int(w * frac), int(h * frac)
    return (x + ix, y + iy, w - 2 * ix, h - 2 * iy)


def crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y + h, x:x + w]


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    print("bin bbox:", find_bin(f["rgb"]))
