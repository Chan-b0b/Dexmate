"""Scene detection for the standing-object show: cylinders + the black bin.

One head-camera frame, warped ONCE to the metric BEV canvas at the plane the
two YOLO-OBB models were trained on (cfg.STAND_DET_PLANE_Z_M — the capture_bev
default of top_face_z(1) = 0.6138 that data/cylinder_bev and data/show_bev were
recorded with), run through both models, every centre then moved from the warp
plane to the height of the feature it really is (bev.reproject_plane: a point
above the warp plane appears pushed away from the camera nadir, tens of mm for
the 7 cm cylinder top / the bin rim).

    scene = detect_scene(rgb, q_torso, q_head)
    scene.cylinders     # [(x, y, conf)], base_link, nearest (smallest x) first
    scene.bin           # (x, y, yaw_deg, conf) or None
    save_debug(scene)   # BEV PNG with both sets of boxes -> case_detection/out/

self-test (no robot): python -m ik_demo.show_detect data/cylinder_bev/<run>/frame_000.npz
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from . import config as cfg

_CASE_DET = Path(__file__).resolve().parents[1] / "case_detection"
if str(_CASE_DET) not in sys.path:
    sys.path.insert(0, str(_CASE_DET))
import bev  # noqa: E402  (case_detection/bev.py)

_models: dict[str, object] = {}


@dataclass
class Scene:
    plane_z: float
    cylinders: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, conf)
    bin: "tuple[float, float, float, float] | None" = None                     # (x, y, yaw_deg, conf)
    bev: "np.ndarray | None" = None
    cyl_px: list[np.ndarray] = field(default_factory=list)                     # 4x2 OBB corners, BEV px
    bin_px: "np.ndarray | None" = None


def _model(path: str):
    if path not in _models:
        from ultralytics import YOLO  # noqa: PLC0415
        p = Path(path)
        if not p.is_absolute():
            p = _CASE_DET / p
        if not p.exists():
            raise FileNotFoundError(f"OBB weights not found at {p}")
        _models[path] = YOLO(str(p))
    return _models[path]


def _obb_long_axis_deg(w: float, h: float, r_rad: float) -> float:
    deg = np.rad2deg(r_rad)
    if w < h:
        deg += 90.0
    return float(deg % 180.0)


def detect_scene(rgb: np.ndarray, q_torso, q_head) -> Scene:
    plane_z = float(cfg.STAND_DET_PLANE_Z_M)
    mapper = bev.build_mapper(q_torso, q_head, plane_z)
    bev_img = mapper.warp(rgb)
    bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR)
    cam = bev.camera_centre(q_torso, q_head)
    scene = Scene(plane_z, bev=bev_img)

    def run(path: str):
        res = _model(path).predict(bgr, conf=float(cfg.STAND_DET_CONF), verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            return []
        confs = res.obb.conf.cpu().numpy()
        xywhr = res.obb.xywhr.cpu().numpy()
        corners = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        out = []
        for k in np.argsort(-confs):
            cx, cy, w, h, r = xywhr[k]
            X, Y = mapper.bev_px_to_base(float(cx), float(cy))
            yaw = mapper.bev_yaw_to_base(_obb_long_axis_deg(w, h, r))
            out.append(((X, Y), yaw, float(confs[k]), corners[k]))
        return out

    # bin first: the OBB is the rim -> its centre lives at the rim height
    bins = run(cfg.STAND_BIN_WEIGHTS)
    if bins:
        xy, yaw, conf, px = bins[0]
        X, Y = bev.reproject_plane(xy, plane_z, float(cfg.STAND_BIN_RIM_Z_M), cam) \
            + np.asarray(cfg.STAND_BIN_DET_OFFSET_XY, dtype=float)
        scene.bin = (float(X), float(Y), float(yaw), conf)
        scene.bin_px = px
    # cylinders: a STANDING cylinder smears in the BEV from its base to its top,
    # so the OBB centre sits at about mid-height -> reproject to desk + h/2.
    # Drop weak boxes (the lying one in the bin / one in a hand scored 0.65 in
    # the training frames, standing ones 0.89+) and anything inside the bin OBB.
    dz = float(cfg.STAND_DESK_Z_M) + 0.5 * float(cfg.STAND_OBJECTS["cylinder"]["height"])
    off = np.asarray(cfg.STAND_CYL_DET_OFFSET_XY, dtype=float)
    cyls = []
    for xy, _yaw, conf, px in run(cfg.STAND_CYL_WEIGHTS):
        if conf < float(cfg.STAND_CYL_CONF):
            continue
        if scene.bin_px is not None and cv2.pointPolygonTest(
                scene.bin_px.astype(np.float32), tuple(px.mean(axis=0).astype(float)), False) >= 0:
            logger.info("[detect] cylinder inside the bin ignored (conf {:.2f})", conf)
            continue
        X, Y = bev.reproject_plane(xy, plane_z, dz, cam) + off
        cyls.append(((float(X), float(Y), conf), px))
    cyls.sort(key=lambda c: c[0][0])
    scene.cylinders = [c[0] for c in cyls]
    scene.cyl_px = [c[1] for c in cyls]
    logger.info("[detect] plane {:.4f}: {} cylinder(s) {}; bin {}", plane_z, len(scene.cylinders),
                [(round(x, 3), round(y, 3), round(c, 2)) for x, y, c in scene.cylinders],
                "none" if scene.bin is None else
                f"({scene.bin[0]:.3f},{scene.bin[1]:+.3f}) yaw {scene.bin[2]:.0f} conf {scene.bin[3]:.2f}")
    return scene


def save_debug(scene: Scene, tag: str = "show") -> "Path | None":
    """BEV PNG: cylinders green, bin magenta, with base-frame coordinates."""
    if scene.bev is None:
        return None
    img = cv2.cvtColor(scene.bev, cv2.COLOR_RGB2BGR).copy()
    for (x, y, c), px in zip(scene.cylinders, scene.cyl_px):
        cv2.polylines(img, [px.astype(np.int32)], True, (0, 255, 0), 2)
        u, v = px.mean(axis=0).astype(int)
        cv2.putText(img, f"cyl ({x:.2f},{y:+.2f}) {c:.2f}", (int(u) + 6, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if scene.bin is not None and scene.bin_px is not None:
        cv2.polylines(img, [scene.bin_px.astype(np.int32)], True, (255, 0, 255), 2)
        u, v = scene.bin_px.mean(axis=0).astype(int)
        cv2.putText(img, f"bin ({scene.bin[0]:.2f},{scene.bin[1]:+.2f})", (int(u) + 6, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    out = _CASE_DET / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), img)
    return path


if __name__ == "__main__":
    f = np.load(sys.argv[1])
    s = detect_scene(f["rgb"], f["q_torso"], f["q_head"])
    print("cylinders:", s.cylinders)
    print("bin:", s.bin)
    print("png:", save_debug(s, "selftest"))
