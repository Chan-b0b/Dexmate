"""Learned bin detector — finds the source bin ROI on the full frame.

A drop-in replacement for bin_roi.find_bin when the bin's image position varies
(HSV + nearest-center heuristic then break). Regular axis-aligned YOLO detection.

    train:  python train.py --target bin   -> runs/detect/bin/weights/best.pt
    enable: cfg.USE_BIN_MODEL = True   (then detect_case_obb uses this)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

_model = None


def load_model(weights: str | None = None):
    global _model
    if _model is None:
        from ultralytics import YOLO  # optional heavy dep
        path = Path(weights or cfg.BIN_MODEL_PATH)
        if not path.is_absolute():
            # cfg path is package-relative, not cwd-relative (as detect_case_bev)
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"Bin weights not found at {path}. Train: python train.py --target bin")
        _model = YOLO(str(path))
    return _model


def find_bin_model(rgb: np.ndarray, weights: str | None = None) -> tuple[int, int, int, int] | None:
    """Highest-confidence bin box as (x, y, w, h) in full-frame px, or None.
    rgb is camera RGB; ultralytics expects BGR."""
    res = load_model(weights).predict(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=cfg.BIN_MODEL_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    conf = res.boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = res.boxes.xyxy.cpu().numpy()[int(np.argmax(conf))]
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def find_bin_base_xy(rgb: np.ndarray, q_torso, q_head, plane_z: float,
                     weights: str | None = None) -> tuple[float, float] | None:
    """Bin bbox center projected to base_link (X, Y) on the z=plane_z plane,
    or None if no bin is detected. Inverts the bev plane homography for the
    live joints. plane_z only weakly affects Y near the robot center line —
    the projection-ray bias scales with the lateral offset — so a rough bin
    rim height is fine for Y-alignment uses."""
    bbox = find_bin_model(rgb, weights)
    if bbox is None:
        return None
    import bev  # sibling module (flat package imports, path set above)
    x, y, w, h = bbox
    H = bev._plane_to_img(q_torso, q_head, float(plane_z))  # base plane -> image
    p = np.linalg.inv(H) @ np.array([x + w / 2.0, y + h / 2.0, 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2]))


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    print("bin bbox:", find_bin_model(f["rgb"]))
