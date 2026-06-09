"""YOLO-OBB inference backend: detect the clear case inside the bin ROI.

Depth is useless for the transparent case and classical edge/texture isolation
is too entangled (bin walls, reflections, internal ribs), so we learn it. The
model runs on the bin-ROI crop (clean, consistent input) and returns an oriented
box; we map it back to full-image pixels and deproject to a camera-frame pose.

Reuses CaseDetection / deproject from detect_case.py so it drops straight into
run_test.py and diagnose.py.

    requires:  pip install ultralytics       (torch + CUDA already present)
    weights:   train with train.py -> runs/.../best.pt, then point MODEL_PATH here
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from bin_roi import crop, find_bin, inset_bbox
from detect_case import CaseDetection, deproject, floor_depth

_model = None  # lazy-loaded singleton


def load_model(weights: str | None = None):
    """Load (once) the trained YOLO-OBB model. ultralytics imported lazily."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)
        path = weights or cfg.OBB_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(
                f"OBB weights not found at {path}. Train first (see train.py) "
                f"and set cfg.OBB_MODEL_PATH."
            )
        _model = YOLO(path)
    return _model


def bin_roi_for(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Crop ROI (x, y, w, h) the case model runs on, or None. Same crop used at
    train/runtime. Priority: cfg.BIN_ROI > learned bin model > HSV find_bin."""
    if cfg.BIN_ROI is not None:
        return tuple(cfg.BIN_ROI)
    if cfg.USE_BIN_MODEL:
        from detect_bin import find_bin_model
        bbox = find_bin_model(rgb)
    else:
        bbox = find_bin(rgb)
    return inset_bbox(bbox) if bbox is not None else None


def _obb_to_long_axis(cx, cy, w, h, r_rad) -> float:
    """YOLO-OBB rotation -> long-axis orientation in degrees, [0, 180)."""
    deg = np.rad2deg(r_rad)
    if w < h:
        deg += 90.0
    return float(deg % 180.0)


def detect_case(depth: np.ndarray | None, rgb: np.ndarray,
                weights: str | None = None) -> CaseDetection:
    """Detect the case via YOLO-OBB. Signature mirrors detect_case.detect_case
    (rgb required here; depth only used for z)."""
    none = CaseDetection(False, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

    roi = bin_roi_for(rgb)
    if roi is None:
        return none
    ox, oy = roi[0], roi[1]
    roi_bgr = cv2.cvtColor(crop(rgb, roi), cv2.COLOR_RGB2BGR)

    model = load_model(weights)
    res = model.predict(roi_bgr, conf=cfg.OBB_CONF, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return none

    # Highest-confidence oriented box.
    i = int(np.argmax(res.obb.conf.cpu().numpy()))
    cx, cy, w, h, r = res.obb.xywhr.cpu().numpy()[i]
    conf = float(res.obb.conf.cpu().numpy()[i])

    u, v = float(cx + ox), float(cy + oy)               # crop -> full-image px
    angle = _obb_to_long_axis(cx, cy, w, h, r)
    z = cfg.FLOOR_Z_M if cfg.FLOOR_Z_M is not None else (
        floor_depth(depth) if depth is not None else 0.0)
    return CaseDetection(
        found=True,
        center_px=(u, v),
        angle_deg=angle,
        size_score=conf,                                 # model confidence
        dims_px=(float(max(w, h)), float(min(w, h))),
        z_m=float(z),
        center_cam_m=deproject(u, v, z),
    )


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    print(detect_case(f.get("depth"), f["rgb"]))
