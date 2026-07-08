"""Measure the base_link z of the (empty) box floor from head-camera depth.

Depth is reliable here because the box is EMPTY and opaque — the "stereo sees
through it" problem is specific to the clear case, not the box floor. So with
nothing in the bin we can read the floor height directly:

    grab one head frame -> sample a central grid of pixels -> deproject each at
    its own depth -> transform to base_link -> report the median z (+ flatness).

That base z anchors the layer-pitch model in ik_demo (SOURCE_CASE_CENTER z,
LAYER_PITCH_M): floor_z is the bottom, per-layer top faces sit above it.

    python measure_floor_z.py                 # head-aim + measure, print base z
    python measure_floor_z.py --no-align       # skip head motion
    python measure_floor_z.py --grid 9 --half 120

Robot-side (needs dexcontrol + ZED). Read-only apart from the head alignment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "perception"))
# camera_geometry: intrinsics + kinematic base<-camera transform (reused, not copied).
sys.path.insert(0, str(_HERE.parents[0] / "case_battery_demo" / "dashboard"))

import config as cfg
import camera_geometry as cg
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import align_head_to_forward


def _get_frame(robot):
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
    rgb = obs.get("left_rgb")
    depth = obs.get("depth")
    rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
    depth = depth.get("data") if isinstance(depth, dict) else depth
    return rgb, depth


def _joints(robot):
    return (np.asarray(robot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(robot.head.get_joint_pos(), dtype=np.float64))


def measure(depth, q_torso, q_head, grid: int, half: int):
    """Deproject a central grid of valid-depth pixels to base_link; return
    (base_pts Nx3, pixels Nx2, center_uv). invalid = <=0 / NaN / out of range."""
    h, w = depth.shape[:2]
    # Head is tilted down, so the box floor sits in the LOWER third of the frame;
    # the image center looks at the far wall/background. Sample the lower band.
    cu = w // 2
    v0, v1 = int(h * 0.5), int(h * 0.7)
    cv_ = (v0 + v1) // 2
    us = np.linspace(cu - half, cu + half, grid).astype(int)
    vs = np.linspace(v0, v1, grid).astype(int)
    pts, px = [], []
    for v in vs:
        for u in us:
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = float(depth[v, u])
            if not np.isfinite(d) or not (cfg.DEPTH_MIN_M < d < cfg.DEPTH_MAX_M):
                continue
            p_cam = cg.deproject_pixel(u, v, d)
            pts.append(cg.transform_zed_point_to_base(p_cam, q_torso, q_head))
            px.append((u, v))
    return np.asarray(pts, dtype=np.float64), np.asarray(px, dtype=int), (cu, cv_)


def robust_floor_z(z: np.ndarray, bin_m: float = 0.005, tol_m: float = 0.01):
    """Floor = densest z cluster (the dominant flat plane), robust to walls/rim/
    outside pixels that inflate a plain median. Returns (floor_z, inlier_mask)."""
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    hist, _ = np.histogram(z, edges)
    peak = edges[int(np.argmax(hist))] + bin_m / 2
    inliers = np.abs(z - peak) < tol_m
    return float(np.median(z[inliers])), inliers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-align", action="store_true", help="skip head alignment")
    ap.add_argument("--angle", type=float, default=30.0, help="head-down angle for align")
    ap.add_argument("--grid", type=int, default=9, help="NxN sample grid")
    ap.add_argument("--half", type=int, default=120, help="grid half-extent in px from center")
    ap.add_argument("--save", action="store_true", help="save the frame (npz+png) next to this script")
    args = ap.parse_args()

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        if not args.no_align:
            align_head_to_forward(robot, angle=args.angle)
            time.sleep(2)
            print('aligned head to forward; waiting 2s for camera to settle')
        rgb, depth = _get_frame(robot)
        q_torso, q_head = _joints(robot)

    if rgb is None or depth is None:
        print("No frame received.")
        return
    depth = np.asarray(depth, np.float32)

    pts, px, (cu, cv_) = measure(depth, q_torso, q_head, args.grid, args.half)
    print(f"depth range {np.nanmin(depth):.3f}..{np.nanmax(depth):.3f} m   "
          f"center depth {depth[cv_, cu]:.3f} m   valid samples {len(pts)}/{args.grid**2}")
    if len(pts) < 3:
        print("Too few valid depth samples — aim the head at the empty floor and retry.")
        return
    z = pts[:, 2]
    floor_z, inliers = robust_floor_z(z)
    print(f"\nall samples:   median {np.median(z):.4f} m   std {z.std()*1000:.1f} mm   "
          f"min {z.min():.4f}  max {z.max():.4f}")
    print(f"FLOOR (densest cluster):  z {floor_z:.4f} m   "
          f"std {z[inliers].std()*1000:.1f} mm   inliers {inliers.sum()}/{len(z)}")
    print(f"floor center (x,y,z): {np.median(pts[inliers], axis=0).round(4)}")
    if inliers.mean() < 0.6 or z[inliers].std() > 0.01:
        print("\n[!] cluster is weak / noisy — the grid likely straddles walls/rim. "
              "Re-run tighter, e.g.  --half 40  (and open the overlay PNG to check).")
    print("\n-> use FLOOR z as the anchor; top-layer faces sit above it "
          "(ik_demo LAYER_PITCH_M per layer).")

    # Always save an overlay so we can SEE what was sampled: green = floor inlier,
    # red = outlier (wall/rim/outside). Plus the raw frame for re-analysis.
    stem = _HERE / f"floor_measure_{time.strftime('%Y%m%d_%H%M%S')}"
    ov = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for (u, v), inl in zip(px, inliers):
        cv2.circle(ov, (int(u), int(v)), 3, (0, 255, 0) if inl else (0, 0, 255), -1)
    cv2.imwrite(str(stem) + "_overlay.png", ov)
    print(f"saved {stem.name}_overlay.png  (green=floor, red=outlier)")
    if args.save:
        np.savez_compressed(stem.with_suffix(".npz"), rgb=rgb, depth=depth,
                            q_torso=q_torso, q_head=q_head)
        print(f"saved {stem.name}.npz")


if __name__ == "__main__":
    main()
