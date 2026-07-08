"""Runtime case detector on the metric BEV image → base_link pose.

Replaces detect_case_obb.py for the BEV pipeline. Instead of cropping the raw
frame by a bin ROI and deprojecting a single pixel at a fixed z, we warp the
frame to the metric top-down canvas (on the current top-face plane) and run the
YOLO-OBB there. In BEV the OBB center maps linearly to base XY and the OBB angle
IS the base yaw — no perspective/parallax error, box-position-independent.

z is NOT taken from detection: the top face is top_face_z(layers_remaining) and
the real grab z comes from ik_demo's descend-to-contact.

    from detect_case_bev import detect_case_bev
    det = detect_case_bev(rgb, q_torso, q_head, layers_remaining=1)
    if det.found:
        X, Y, yaw = det.base_xy[0], det.base_xy[1], det.base_yaw_deg

    requires:  ultralytics + a trained BEV OBB model at cfg.OBB_MODEL_PATH.
    self-test: python detect_case_bev.py <frame_or_floor_measure>.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import bev

_model = None  # lazy-loaded singleton


@dataclass
class CaseBEV:
    found: bool
    base_xy: tuple[float, float]       # base_link (X, Y) of the case center, m
    base_yaw_deg: float                # case long-axis yaw about +z, [0,180)
    top_face_z: float                  # base z of the top face used for the warp, m
    bev_center_px: tuple[float, float] # OBB center in the BEV image (for viz)
    dims_px: tuple[float, float]       # (long, short) side in BEV px
    conf: float
    bev: np.ndarray | None = None      # the warped BEV image (for viz)


def _none(bev_img=None) -> CaseBEV:
    return CaseBEV(False, (0.0, 0.0), 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), 0.0, bev_img)


def load_model(weights: str | None = None):
    """Load (once) the trained BEV YOLO-OBB model."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)
        path = Path(weights or cfg.OBB_MODEL_PATH)
        # cfg.OBB_MODEL_PATH is relative to this package, not the caller's cwd
        # (chassis_sequence runs from the repo root) — resolve it here.
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"OBB weights not found at {path}. Train first "
                f"(python train.py --target case) and set cfg.OBB_MODEL_PATH.")
        _model = YOLO(str(path))
    return _model


def _obb_long_axis_deg(w: float, h: float, r_rad: float) -> float:
    """YOLO-OBB rotation -> long-axis orientation in degrees, [0,180)."""
    deg = np.rad2deg(r_rad)
    if w < h:
        deg += 90.0
    return float(deg % 180.0)


def detect_case_bev(rgb: np.ndarray, q_torso, q_head, layers_remaining: int = 1,
                    weights: str | None = None) -> CaseBEV:
    """Warp to BEV on the current top-face plane, run YOLO-OBB, map to base."""
    plane_z = bev.top_face_z(layers_remaining)
    mapper = bev.build_mapper(q_torso, q_head, plane_z)
    bev_img = mapper.warp(rgb)

    model = load_model(weights)
    res = model.predict(cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR),
                        conf=cfg.OBB_CONF, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return _none(bev_img)

    i = int(np.argmax(res.obb.conf.cpu().numpy()))            # highest-conf box
    cx, cy, w, h, r = res.obb.xywhr.cpu().numpy()[i]
    conf = float(res.obb.conf.cpu().numpy()[i])

    X, Y = mapper.bev_px_to_base(float(cx), float(cy))
    yaw = mapper.bev_yaw_to_base(_obb_long_axis_deg(w, h, r))
    return CaseBEV(
        found=True,
        base_xy=(X, Y),
        base_yaw_deg=yaw,
        top_face_z=plane_z,
        bev_center_px=(float(cx), float(cy)),
        dims_px=(float(max(w, h)), float(min(w, h))),
        conf=conf,
        bev=bev_img,
    )


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    d = detect_case_bev(f["rgb"], f["q_torso"], f["q_head"],
                        int(f["layers_remaining"]) if "layers_remaining" in f.files else 1)
    print(f"found={d.found}  base_xy=({d.base_xy[0]:.4f}, {d.base_xy[1]:+.4f}) m  "
          f"yaw={d.base_yaw_deg:.1f} deg  top_face_z={d.top_face_z:.4f}  conf={d.conf:.2f}")
