"""Runtime paper-box detector on the metric BEV image -> base_link pose + size.

Same recipe as detect_case_bev (warp the head frame to the metric top-down
canvas, run YOLO-OBB there, map the OBB back to base_link), with two box
specifics:

  * SIZE matters. The box is far wider than the gripper, so ik_demo grips a
    WALL — it needs the detected (long, short) in metres, not just the center.
    The BEV canvas is metric: px / BEV_PX_PER_M.
  * PLANE matters. The detector must warp at the box RIM height. A point above
    the warp plane is pushed away from the camera and enlarged, so a wrong
    plane biases the center by roughly tan(view angle) x height error (tens of
    mm) and the size by the same ratio. RGB alone does not give the rim
    height; plane_from_size() recovers it from the box's known long side by
    sweeping the plane until the detected size matches. (The training set,
    data/box_bev/20260903_161831_L1, was warped at top_face_z(1) = 0.6138.)

    from detect_box_bev import detect_box_bev, plane_from_size
    z, _ = plane_from_size(rgb, q_torso, q_head, known_long_m=0.62)  # optional
    det = detect_box_bev(rgb, q_torso, q_head, plane_z=z)
    det.base_xy, det.base_yaw_deg, det.dims_m, det.conf

self-test: python detect_box_bev.py data/box_bev/<run>/frame_000.npz [plane_z]
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
class BoxBEV:
    found: bool
    base_xy: tuple[float, float]        # base_link (X, Y) of the box center, m
    base_yaw_deg: float                 # long-axis yaw about +z, [0,180)
    top_face_z: float                   # plane z the frame was warped at, m
    dims_m: tuple[float, float]         # (long, short) side, m (metric BEV)
    conf: float
    bev: np.ndarray | None = None       # warped BEV image (RGB, for viz)
    obb_px: np.ndarray | None = None    # 4x2 OBB corners in BEV px (for viz)


def _none(plane_z: float, bev_img=None) -> BoxBEV:
    return BoxBEV(False, (0.0, 0.0), 0.0, plane_z, (0.0, 0.0), 0.0, bev_img)


def load_model(weights: str | None = None):
    global _model
    if _model is None:
        from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)
        path = Path(weights or cfg.BOX_MODEL_PATH)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(f"box OBB weights not found at {path} "
                                    f"(python train.py --target box, cfg.BOX_MODEL_PATH)")
        _model = YOLO(str(path))
    return _model


def _obb_long_axis_deg(w: float, h: float, r_rad: float) -> float:
    deg = np.rad2deg(r_rad)
    if w < h:
        deg += 90.0
    return float(deg % 180.0)


def detect_box_bev(rgb: np.ndarray, q_torso, q_head, plane_z: float,
                   weights: str | None = None,
                   cls_id: "int | None" = None) -> BoxBEV:
    """Warp at ``plane_z`` (the box rim height), run YOLO-OBB, map to base.

    ``cls_id``: keep only boxes of that class. The box set is 2-class
    (dataset_box/data.yaml: 0 = box, 1 = box_top / the paper LID), so an
    unfiltered pick can hand back the box body when the lid is what is wanted.
    None = any class (the original behaviour)."""
    mapper = bev.build_mapper(q_torso, q_head, float(plane_z))
    bev_img = mapper.warp(rgb)
    res = load_model(weights).predict(cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR),
                                      conf=cfg.BOX_OBB_CONF, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return _none(float(plane_z), bev_img)
    confs = res.obb.conf.cpu().numpy()
    xywhr = res.obb.xywhr.cpu().numpy()
    s = 1.0 / cfg.BEV_PX_PER_M
    keep = np.arange(len(confs))
    if cls_id is not None:
        keep = np.flatnonzero(res.obb.cls.cpu().numpy().astype(int) == int(cls_id))
        if keep.size == 0:
            return _none(float(plane_z), bev_img)
    i = None
    for k in keep[np.argsort(-confs[keep])]:   # highest confidence first, size-gated
        lw, sw = max(xywhr[k][2], xywhr[k][3]) * s, min(xywhr[k][2], xywhr[k][3]) * s
        if (abs(lw - cfg.BOX_BEV_SIZE_M[0]) <= cfg.BOX_SIZE_TOL * cfg.BOX_BEV_SIZE_M[0]
                and abs(sw - cfg.BOX_BEV_SIZE_M[1]) <= cfg.BOX_SIZE_TOL * cfg.BOX_BEV_SIZE_M[1]):
            i = int(k)
            break
    if i is None:
        return _none(float(plane_z), bev_img)
    cx, cy, w, h, r = xywhr[i]
    corners = res.obb.xyxyxyxy.cpu().numpy()[i].reshape(4, 2)
    X, Y = mapper.bev_px_to_base(float(cx), float(cy))
    yaw = mapper.bev_yaw_to_base(_obb_long_axis_deg(w, h, r))
    return BoxBEV(True, (X, Y), yaw, float(plane_z),
                  (float(max(w, h)) * s, float(min(w, h)) * s), float(confs[i]),
                  bev_img, corners)


def plane_from_size(rgb: np.ndarray, q_torso, q_head, known_long_m: float,
                    z_range: tuple[float, float] = (0.60, 1.05), step: float = 0.01,
                    weights: str | None = None) -> tuple[float | None, BoxBEV | None]:
    """Recover the rim height: sweep the warp plane and keep the one where the
    detected long side matches ``known_long_m`` (size scales with the plane).
    Returns (plane_z, detection) or (None, None) if nothing was detected."""
    best: tuple[float, float, BoxBEV] | None = None
    for z in np.arange(z_range[0], z_range[1] + 1e-9, step):
        det = detect_box_bev(rgb, q_torso, q_head, float(z), weights)
        if not det.found:
            continue
        err = abs(det.dims_m[0] - known_long_m)
        if best is None or err < best[0]:
            best = (err, float(z), det)
    if best is None:
        return None, None
    return best[1], best[2]


def draw(det: BoxBEV, extra_pts_base: list[tuple[float, float]] | None = None) -> np.ndarray:
    """BGR BEV image with the OBB, its center and optional base-frame points
    (e.g. the grasp point) drawn — for a debug PNG."""
    img = cv2.cvtColor(det.bev, cv2.COLOR_RGB2BGR).copy()
    if det.found and det.obb_px is not None:
        cv2.polylines(img, [det.obb_px.astype(np.int32)], True, (0, 255, 0), 2)
        x0, _ = cfg.BEV_X_RANGE
        y0, _ = cfg.BEV_Y_RANGE
        cu = int(round((det.base_xy[0] - x0) * cfg.BEV_PX_PER_M))
        cv = int(round((det.base_xy[1] - y0) * cfg.BEV_PX_PER_M))
        cv2.circle(img, (cu, cv), 5, (0, 255, 0), -1)
        cv2.putText(img, f"{det.dims_m[0]:.2f}x{det.dims_m[1]:.2f}m yaw {det.base_yaw_deg:.0f} "
                    f"conf {det.conf:.2f} z {det.top_face_z:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for (px, py) in extra_pts_base or []:
            u = int(round((px - x0) * cfg.BEV_PX_PER_M))
            v = int(round((py - y0) * cfg.BEV_PX_PER_M))
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    else:
        cv2.putText(img, f"no box (plane z {det.top_face_z:.3f})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    z = float(sys.argv[2]) if len(sys.argv) > 2 else bev.top_face_z(1)
    d = detect_box_bev(f["rgb"], f["q_torso"], f["q_head"], z)
    print(f"found={d.found} base_xy=({d.base_xy[0]:.3f},{d.base_xy[1]:+.3f}) yaw={d.base_yaw_deg:.1f}deg "
          f"size={d.dims_m[0]:.3f}x{d.dims_m[1]:.3f}m conf={d.conf:.2f} plane={z:.3f}")
