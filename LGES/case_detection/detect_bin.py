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
        path = weights or cfg.BIN_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Bin weights not found at {path}. Train: python train.py --target bin")
        _model = YOLO(path)
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


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    print("bin bbox:", find_bin_model(f["rgb"]))
