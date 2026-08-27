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


_bev_model = None


def load_bev_model(weights: str | None = None):
    """Load (once) the trained BEV YOLO-OBB bin model."""
    global _bev_model
    if _bev_model is None:
        from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)
        path = Path(weights or cfg.BIN_OBB_MODEL_PATH)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"BEV bin OBB weights not found at {path}. Train first "
                f"(train.py, target bin on the BEV set) and set cfg.BIN_OBB_MODEL_PATH.")
        _bev_model = YOLO(str(path))
    return _bev_model


def find_bin_bev(rgb: np.ndarray, q_torso, q_head, plane_z: float,
                 weights: str | None = None) -> "tuple[float, float, float] | None":
    """Bin OBB on the metric BEV canvas -> (base X, base Y, yaw_deg), or None.

    Replaces find_bin_base_xy for runtime use: the raw-frame bbox center
    inverted through the homography carried a measured +47mm x bias (front
    wall + plane mismatch); in BEV the OBB center maps LINEARLY to base XY
    for anything ON the warp plane (pass the bin RIM height) and the OBB
    angle is the bin yaw for free. Highest-conf box wins; yaw is the
    long-axis convention, [0,180), mapped to base (as detect_case_bev)."""
    import bev  # sibling module (flat package imports, path set above)
    mapper = bev.build_mapper(q_torso, q_head, float(plane_z))
    bev_img = mapper.warp(rgb)
    model = load_bev_model(weights)
    res = model.predict(cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR),
                        conf=cfg.BIN_OBB_CONF, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return None
    i = int(np.argmax(res.obb.conf.cpu().numpy()))
    cx, cy, w, h, r = res.obb.xywhr.cpu().numpy()[i]
    X, Y = mapper.bev_px_to_base(float(cx), float(cy))
    deg = float(np.rad2deg(float(r)))
    if w < h:
        deg += 90.0
    return X, Y, float(mapper.bev_yaw_to_base(deg % 180.0))


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
    # self-test: python detect_bin.py <frame.npz> [plane_z]
    # npz with q_torso/q_head also runs the BEV OBB and saves the annotated
    # canvas (bin_bev_selftest.png) — use it to SEE what the model boxed when
    # the reported base xy looks wrong.
    f = np.load(sys.argv[1])
    print("bin bbox (raw frame):", find_bin_model(f["rgb"]))
    if "q_torso" in f.files:
        import bev  # sibling module
        plane = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
        mapper = bev.build_mapper(f["q_torso"], f["q_head"], plane)
        bev_img = mapper.warp(f["rgb"])
        res = load_bev_model().predict(cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR),
                                       conf=cfg.BIN_OBB_CONF, verbose=False)[0]
        vis = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR).copy()
        if res.obb is not None and len(res.obb):
            polys = res.obb.xyxyxyxy.cpu().numpy()
            confs = res.obb.conf.cpu().numpy()
            for k, (poly, c) in enumerate(zip(polys, confs)):
                cx, cy = poly.mean(axis=0)
                X, Y = mapper.bev_px_to_base(float(cx), float(cy))
                print(f"  obb[{k}] conf={c:.2f} base_xy=({X:.3f},{Y:+.3f})")
                cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 0, 255), 2)
        else:
            print("  no BEV OBB detection")
        cv2.imwrite("bin_bev_selftest.png", vis)
        print(f"BEV canvas (plane_z={plane}) -> bin_bev_selftest.png")
