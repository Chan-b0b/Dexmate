"""Measure the height of the surface a BEV detection sits on, from ZED depth.

The BEV warp needs the plane the feature actually lies on: get it wrong and the
detected center is scaled about the camera nadir (see bev.py), tens of mm of
pure bias with a clean, confident box. bev.plane_from_size recovers it from a
known physical size; this recovers it by MEASURING, which needs no tape and no
assumption about which face the labels trace.

    z, n, spread = plane_from_depth(depth, rgb.shape, q_torso, q_head,
                                    base_xy=(0.86, 0.30), plane_guess=0.345)

The guess only picks the pixel to sample: the detection's own center, projected
through the guessed plane, lands within a few px of the true one and the window
covers the rest — then one iteration re-projects through the measured height and
re-samples. Depth is ZED's Z along the optical axis, in metres, registered to
the LEFT image, so its pixels map to the left-RGB intrinsics by a plain scale.

Offline-safe: pure geometry, no robot, no zenoh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "case_battery_demo" / "dashboard"))

import bev  # noqa: E402
import camera_geometry as cg  # noqa: E402


def base_to_pixel(base_xyz, q_torso, q_head) -> tuple[float, float]:
    """base_link (x, y, z) -> left-image pixel (u, v). The same projection the
    BEV homography is built from, without the plane collapse."""
    T = cg.zed_left_camera_pose_from_joints(q_torso, q_head)   # base <- cam
    p = np.linalg.inv(T) @ np.array([*(float(v) for v in base_xyz), 1.0])
    if p[2] <= 1e-6:
        raise ValueError("point is behind the camera")
    uv = bev.intrinsic_matrix() @ p[:3]
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def pixel_to_base(u: float, v: float, depth_m: float, q_torso, q_head) -> np.ndarray:
    """left-image pixel + ZED depth (Z along the optical axis, m) -> base xyz."""
    K = bev.intrinsic_matrix()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) / fx * depth_m, (v - cy) / fy * depth_m,
                      float(depth_m), 1.0])
    return (cg.zed_left_camera_pose_from_joints(q_torso, q_head) @ p_cam)[:3]


def sample_depth(depth: np.ndarray, rgb_shape, u: float, v: float,
                 half_win_px: int = 12, pct: float = 50.0):
    """(depth at percentile ``pct``, count, p95-p5 spread) around an
    intrinsics-frame pixel. Scales into the depth map's own resolution first —
    depth is registered to the left image but need not be published at the same
    size as the RGB the intrinsics describe.

    ``pct`` matters on a surface with relief. The MEDIAN (50) is the face's
    representative height, which is what the warp plane wants. A LOW percentile
    is the NEAREST point in the window — the tallest bump — and that is what a
    descending cup actually lands on, so the expected contact height wants
    something like 10, not 50. Measured lids carry ~20mm of embossing, so the
    two answers differ by about that much."""
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[0], rgb_shape[1]
    du, dv = u * (dw / float(rw)), v * (dh / float(rh))
    w = max(1, int(round(half_win_px * dw / float(rw))))
    i, j = int(round(dv)), int(round(du))
    if not (0 <= i < dh and 0 <= j < dw):
        return None, 0, 0.0
    patch = depth[max(0, i - w):i + w + 1, max(0, j - w):j + w + 1].astype(np.float64)
    ok = patch[np.isfinite(patch) & (patch > 0.05)]
    if ok.size < 8:
        return None, int(ok.size), 0.0
    return (float(np.percentile(ok, float(pct))), int(ok.size),
            float(np.percentile(ok, 95) - np.percentile(ok, 5)))


def expected_depth(base_xyz, q_torso, q_head) -> float:
    """What the depth map SHOULD read at that base point: Z along the optical
    axis, ZED's convention (camera_geometry.deproject_pixel inverts exactly
    this). The discriminator when a measurement disagrees with the configured
    plane — measured ~= expected means the plane is right and the pixel/scale
    mapping is wrong, measured far off means the surface really is elsewhere."""
    T = cg.zed_left_camera_pose_from_joints(q_torso, q_head)
    p = np.linalg.inv(T) @ np.array([*(float(v) for v in base_xyz), 1.0])
    return float(p[2])


def plane_from_depth(depth: np.ndarray, rgb_shape, q_torso, q_head, base_xy,
                     plane_guess: float, half_win_px: int = 12, iters: int = 2,
                     pct: float = 50.0):
    """Measured base-frame z of the surface under ``base_xy``.

    Returns (z, n_samples, depth_spread_m) or (None, n, spread) when too few
    valid depth pixels landed in the window. ``base_xy`` is where the detection
    says the feature is ON THE GUESSED PLANE; each iteration re-projects it
    through the height just measured, so the sampled pixel converges onto the
    real surface point. ``pct`` picks which depth in the window answers (see
    sample_depth): 50 for the face's height, ~10 for the tallest bump a cup
    would touch first."""
    z, n, spread = float(plane_guess), 0, 0.0
    C = bev.camera_centre(q_torso, q_head)
    for _ in range(max(1, iters)):
        xy = bev.reproject_plane(base_xy, float(plane_guess), z, C)
        u, v = base_to_pixel((xy[0], xy[1], z), q_torso, q_head)
        d, n, spread = sample_depth(depth, rgb_shape, u, v, half_win_px, pct)
        if d is None:
            return None, n, spread
        z_new = float(pixel_to_base(u, v, d, q_torso, q_head)[2])
        if abs(z_new - z) < 1e-4:
            z = z_new
            break
        z = z_new
    return z, n, spread
