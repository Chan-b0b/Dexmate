"""Bird-eye-view (BEV) rectification of the head-camera frame.

The head looks down at the box; a rectangle on the floor projects to a trapezoid,
so the OBB angle in the raw image is NOT the true floor yaw and pixel->metric is
non-linear. We warp the frame to a metric top-down image of a fixed base_link
floor ROI, on the plane at the current top-face height. In that image:

    * pixel <-> base XY is a plain affine (bev_px_to_base), and
    * the OBB long-axis angle IS the base-frame yaw (bev_yaw_to_base).

The homography comes straight from the robot kinematics (camera_geometry:
base<-camera from q_torso/q_head) + intrinsics — no separate calibration. With
head+torso held fixed the base<-camera transform is constant, so the mapper is
cached (keyed by rounded joints + plane_z); it still rebuilds correctly if the
joints move. Only the case top face lies exactly on the chosen plane, so warp on
the plane at FLOOR_Z_BASE_M + k*LAYER_PITCH_M for the current top layer.

    from bev import build_mapper
    m = build_mapper(q_torso, q_head, plane_z)      # plane_z = current top face
    bev = m.warp(rgb)                                # metric top-down image
    X, Y = m.bev_px_to_base(bu, bv)                  # OBB center -> base XY
    yaw  = m.bev_yaw_to_base(obb_angle_deg)          # OBB angle  -> base yaw

Self-test (verify geometry on a saved floor_measure_*.npz from measure_floor_z):
    python bev.py floor_measure_XXXX.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "case_battery_demo" / "dashboard"))

import config as cfg
import camera_geometry as cg


def intrinsic_matrix() -> np.ndarray:
    """ZED left 3x3 K (same intrinsics as camera_geometry / perception)."""
    return np.array([[cg.ZED_FX, 0.0, cg.ZED_CX],
                     [0.0, cg.ZED_FY, cg.ZED_CY],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def canvas_size() -> tuple[int, int]:
    """(w, h) of the BEV image from the base ROI and resolution."""
    x0, x1 = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    w = int(round((x1 - x0) * cfg.BEV_PX_PER_M))
    h = int(round((y1 - y0) * cfg.BEV_PX_PER_M))
    return w, h


def _canvas_to_base() -> np.ndarray:
    """3x3 affine A: BEV pixel [bu,bv,1] -> base plane coords [X,Y,1].
    bu -> +base_x (forward), bv -> +base_y (left); uniform scale 1/px_per_m."""
    x0, _ = cfg.BEV_X_RANGE
    y0, _ = cfg.BEV_Y_RANGE
    s = 1.0 / cfg.BEV_PX_PER_M
    return np.array([[s, 0.0, x0],
                     [0.0, s, y0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _plane_to_img(q_torso, q_head, plane_z: float) -> np.ndarray:
    """3x3 homography H: base plane point [X,Y,1] at z=plane_z -> image [u,v,1].

    For a fixed plane z=c the pinhole projection collapses to a homography whose
    columns are the camera-frame images of the base axes:
        p_cam = R_cb[:,0]*X + R_cb[:,1]*Y + (R_cb[:,2]*c + t_cb);  u = K p_cam.
    """
    T_base_cam = cg.zed_left_camera_pose_from_joints(q_torso, q_head)  # base<-cam
    T_cam_base = np.linalg.inv(T_base_cam)                             # cam<-base
    R_cb = T_cam_base[:3, :3]
    t_cb = T_cam_base[:3, 3]
    cols = np.column_stack([R_cb[:, 0], R_cb[:, 1], R_cb[:, 2] * plane_z + t_cb])
    return intrinsic_matrix() @ cols


@dataclass(frozen=True)
class BevMapper:
    """Metric top-down mapper for one (joints, plane_z). Warp + coordinate maps."""
    img_to_bev: np.ndarray   # 3x3 M for cv2.warpPerspective(img -> BEV)
    size: tuple[int, int]    # (w, h) of the BEV image
    plane_z: float

    def warp(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(rgb, self.img_to_bev, self.size)

    def bev_px_to_base(self, bu: float, bv: float) -> tuple[float, float]:
        """BEV pixel -> base_link (X, Y) on the plane. Linear."""
        x0, _ = cfg.BEV_X_RANGE
        y0, _ = cfg.BEV_Y_RANGE
        s = 1.0 / cfg.BEV_PX_PER_M
        return (x0 + bu * s, y0 + bv * s)

    def bev_yaw_to_base(self, angle_deg: float) -> float:
        """OBB long-axis angle (deg, in BEV pixel frame) -> base yaw about +z.
        BEV axes are bu||+X, bv||+Y with equal scale, so the angle carries over
        directly; returned in [0,180). Verify sign against a known-pose case."""
        return float(angle_deg % 180.0)


_cache: dict[tuple, BevMapper] = {}


def build_mapper(q_torso, q_head, plane_z: float) -> BevMapper:
    """Build (or reuse) the mapper for these joints + plane. Cached because a
    fixed head/torso gives a constant transform; rebuilds when anything moves."""
    key = (tuple(np.round(np.ravel(q_torso), 4)),
           tuple(np.round(np.ravel(q_head), 4)),
           round(float(plane_z), 4))
    m = _cache.get(key)
    if m is None:
        H = _plane_to_img(q_torso, q_head, plane_z) @ _canvas_to_base()  # BEV->img
        m = BevMapper(img_to_bev=np.linalg.inv(H), size=canvas_size(),
                      plane_z=float(plane_z))
        _cache[key] = m
    return m


def top_face_z(layers_remaining: int) -> float:
    """Base z of the current top face with `layers_remaining` layers still stacked."""
    return cfg.FLOOR_Z_BASE_M + layers_remaining * cfg.LAYER_PITCH_M


def frame_plane_z(npz, override: float | None = None) -> float:
    """The plane a saved capture frame belongs on, in one place.

    Precedence: an explicit ``override`` (a caller that knows better), else the
    ``plane_z`` the frame records, else top_face_z(layers_remaining).

    The recorded key matters because a run aimed at something that is NOT a
    stack top face (a lid on the floor, a pallet) sits at a height that
    top_face_z(k) cannot express — rewarp_bev.py writes plane_z for exactly
    that case. Re-deriving the plane from layers_remaining instead would put
    the labels on a different plane than the bev/*.png they trace, which is a
    silent scale-about-the-camera-nadir error (see the homothety note above)."""
    if override is not None:
        return float(override)
    keys = npz.files if hasattr(npz, "files") else npz
    if "plane_z" in keys:
        return float(npz["plane_z"])
    k = int(npz["layers_remaining"]) if "layers_remaining" in keys else 1
    return top_face_z(k if k >= 0 else 1)   # capture.py writes -1 for "unset"


# ---------------------------------------------------------------------------
# Wrong-plane recovery.
#
# Warping ONE image at two parallel planes cuts the SAME bundle of camera rays
# at two heights, so the two BEV results are related by a plain homothety about
# the camera centre C:
#
#     P_to = C_xy + (P_from - C_xy) * (z_to - C_z) / (z_from - C_z)
#
# Consequences, all verified on data/bin_bev (calib_bin_plane.py --verify):
#   * ANGLES ARE PLANE-INVARIANT — a wrong plane never costs yaw.
#   * LENGTHS scale by the same factor, so a feature of KNOWN physical size
#     pins its own height (plane_from_size) — this is the only height cue RGB
#     offers, and it makes the recovery self-calibrating per frame.
#   * A CENTER detected at the wrong plane needs no re-detection, no re-warp
#     and no re-training: rescaling it about C_xy is exact.
#
# The practical trap this closes: a detector trained on labels of one face
# (e.g. the bin's inner BOTTOM) but run with plane_z set to another (the bin
# RIM) reports a center pulled toward the camera nadir by the height ratio —
# tens of mm of pure x bias, with a clean picture and a confident box.
# ---------------------------------------------------------------------------
def camera_centre(q_torso, q_head) -> np.ndarray:
    """base_link xyz of the ZED left camera — the fixed point of the homothety."""
    return cg.zed_left_camera_pose_from_joints(q_torso, q_head)[:3, 3]


def reproject_plane(xy, z_from: float, z_to: float, cam_centre) -> np.ndarray:
    """Move a BEV measurement from the plane it was warped at onto another plane.

    ``xy`` is a base-frame (X, Y) read off a BEV warped at ``z_from``; the
    result is where that same image feature lands on the ``z_to`` plane."""
    C = np.asarray(cam_centre, dtype=np.float64)
    k = (float(z_to) - C[2]) / (float(z_from) - C[2])
    return C[:2] + (np.asarray(xy, dtype=np.float64) - C[:2]) * k


def plane_from_size(long_meas_m: float, long_true_m: float, z_warp: float,
                    cam_centre) -> float:
    """Height of the detected feature, from how much the warp magnified it.

    The homothety scale IS the size ratio, so a feature whose physical long
    side is known reports its own plane:

        (z_feat - C_z) = (z_warp - C_z) * long_true / long_meas

    ``long_meas_m`` is the detected long side on the metric BEV canvas
    (px / cfg.BEV_PX_PER_M), ``long_true_m`` the measured physical one."""
    Cz = float(np.asarray(cam_centre, dtype=np.float64)[2])
    return Cz + (float(z_warp) - Cz) * (float(long_true_m) / float(long_meas_m))


# ---------------------------------------------------------------------------
# Self-test: warp a saved floor_measure_*.npz and draw a base-frame grid so the
# geometry can be checked by eye (grid lines straight + evenly spaced = correct).
# ---------------------------------------------------------------------------
def _selftest(npz_path: str) -> None:
    f = np.load(npz_path)
    rgb, q_torso, q_head = f["rgb"], f["q_torso"], f["q_head"]
    plane_z = cfg.FLOOR_Z_BASE_M  # empty floor
    m = build_mapper(q_torso, q_head, plane_z)
    bev = m.warp(rgb)
    bev_bgr = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)

    # 10 cm base-frame grid, labeled — should be square and uniform if correct.
    x0, x1 = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    s = cfg.BEV_PX_PER_M
    for X in np.arange(np.ceil(x0 * 10) / 10, x1, 0.1):
        u = int((X - x0) * s)
        cv2.line(bev_bgr, (u, 0), (u, bev.shape[0]), (0, 180, 0), 1)
        cv2.putText(bev_bgr, f"x{X:.1f}", (u + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1)
    for Y in np.arange(np.ceil(y0 * 10) / 10, y1, 0.1):
        v = int((Y - y0) * s)
        cv2.line(bev_bgr, (0, v), (bev.shape[1], v), (0, 180, 0), 1)
        cv2.putText(bev_bgr, f"y{Y:.1f}", (2, v - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1)

    out = Path(npz_path).with_name(Path(npz_path).stem + "_bev.png")
    cv2.imwrite(str(out), bev_bgr)
    print(f"BEV {bev.shape[1]}x{bev.shape[0]} px @ {s} px/m, plane_z={plane_z:.4f}")
    print(f"saved {out}  (green = 10 cm base grid; check it's square + uniform)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python bev.py <floor_measure_*.npz>")
        sys.exit(1)
    _selftest(sys.argv[1])
