"""Live accuracy check of the case BEV detector against hand-measured moves.

The absolute base_link pose of the case is hard to measure by hand to ~2 mm, so
this measures RELATIVE moves instead: put the case at a reference pose, move it
by a known (dx, dy, dyaw) measured with a ruler / jig / protractor, and compare
with the detected change. It exposes scale, plane-height and yaw errors; it
cannot see an absolute origin offset.

    ~/vla_venv/bin/python eval_accuracy.py --layer 1
    ~/vla_venv/bin/python eval_accuracy.py --layer 1 --samples 15 --out acc.csv

Flow: place the case, press Enter for the reference. Then for each move type
"dx_mm dy_mm dyaw_deg" (base_link axes: +x forward, +y left, +yaw CCW seen from
above), move the case by exactly that from the REFERENCE pose (not the previous
one), press Enter. Empty line / q ends. Each pose is the median of --samples
distinct camera frames.

The head pitch is set once at start (--angle, default 24 as chassis_sequence);
--keep-head skips it.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import bev
import config as cfg
from detect_case_bev import detect_case_bev


def _wrap90(d: float) -> float:
    """The case is 180-deg symmetric: wrap a yaw difference to [-90, 90)."""
    return (d + 90.0) % 180.0 - 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=1, help="layers_remaining for the warp plane")
    ap.add_argument("--plane-z", type=float, help="force the warp plane (m)")
    ap.add_argument("--samples", type=int, default=15, help="distinct frames per pose")
    ap.add_argument("--angle", type=float, default=24.0)
    ap.add_argument("--keep-head", action="store_true")
    ap.add_argument("--out", default="accuracy.csv")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot
    from utils import set_head_pitch

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    robot = Robot(configs=configs)
    for part in (robot.torso, robot.head):
        part.set_subscription_policy("always_on", recursive=True)
    robot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
    robot.sensors.head_camera.wait_for_active(timeout=5.0)
    if not args.keep_head:
        set_head_pitch(robot, angle=args.angle, tol_deg=2.0)
        time.sleep(2.0)   # let the head settle; frames taken while it moves scatter x

    def grab():
        obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
        rgb = obs.get("left_rgb")
        return rgb.get("data") if isinstance(rgb, dict) else rgb

    def measure():
        """Median (x, y, yaw) over --samples DISTINCT frames, plus the scatter."""
        out, prev, tries = [], None, 0
        while len(out) < args.samples and tries < args.samples * 20:
            tries += 1
            rgb = grab()
            if rgb is None or (prev is not None and np.array_equal(rgb, prev)):
                time.sleep(0.01)
                continue
            prev = rgb
            q_torso = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)
            q_head = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
            d = detect_case_bev(rgb, q_torso, q_head, layers_remaining=args.layer,
                                plane_z=args.plane_z)
            if d.found:
                out.append((d.base_xy[0], d.base_xy[1], d.base_yaw_deg, d.conf,
                            *d.dims_px))
        if len(out) < max(3, args.samples // 2):
            return None
        a = np.asarray(out)
        yaw = a[:, 2]
        # circular median about the first sample (yaw is [0,180) and wraps)
        yaw = yaw[0] + np.array([_wrap90(y - yaw[0]) for y in yaw])
        return dict(x=float(np.median(a[:, 0])), y=float(np.median(a[:, 1])),
                    yaw=float(np.median(yaw) % 180.0),
                    sx=float(a[:, 0].std()), sy=float(a[:, 1].std()), syaw=float(yaw.std()),
                    conf=float(a[:, 3].mean()), long_px=float(np.median(a[:, 4])),
                    short_px=float(np.median(a[:, 5])), n=len(out))

    rows = []
    input("Place the case at the REFERENCE pose, then Enter... ")
    ref = measure()
    if ref is None:
        raise SystemExit("reference: case not detected")
    print(f"reference: x={ref['x']*1000:.1f} y={ref['y']*1000:+.1f} mm yaw={ref['yaw']:.2f} deg "
          f"(scatter {ref['sx']*1000:.2f}/{ref['sy']*1000:.2f} mm, {ref['syaw']:.2f} deg)")

    while True:
        line = input("move from reference 'dx_mm dy_mm dyaw_deg' (empty=quit): ").strip()
        if not line or line.lower() == "q":
            break
        try:
            gx, gy, gyaw = (float(t) for t in line.replace(",", " ").split())
        except ValueError:
            print("  need three numbers")
            continue
        input("  move the case, then Enter... ")
        m = measure()
        if m is None:
            print("  not detected")
            continue
        ex = (m["x"] - ref["x"]) * 1000 - gx
        ey = (m["y"] - ref["y"]) * 1000 - gy
        eyaw = _wrap90(m["yaw"] - ref["yaw"] - gyaw)
        rows.append(dict(gt_dx=gx, gt_dy=gy, gt_dyaw=gyaw, err_x_mm=ex, err_y_mm=ey,
                         err_yaw_deg=eyaw, scatter_x_mm=m["sx"] * 1000,
                         scatter_y_mm=m["sy"] * 1000, scatter_yaw_deg=m["syaw"],
                         conf=m["conf"], long_px=m["long_px"], short_px=m["short_px"], n=m["n"]))
        print(f"  error: x {ex:+.2f} mm  y {ey:+.2f} mm  yaw {eyaw:+.2f} deg "
              f"(scatter {m['sx']*1000:.2f}/{m['sy']*1000:.2f} mm, {m['syaw']:.2f} deg)")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        e = np.array([[r["err_x_mm"], r["err_y_mm"], r["err_yaw_deg"]] for r in rows])
        rms = np.sqrt((e ** 2).mean(0))
        print(f"\n{len(rows)} moves -> {args.out}")
        print(f"  mean error (bias): x {e[:,0].mean():+.2f}  y {e[:,1].mean():+.2f} mm  "
              f"yaw {e[:,2].mean():+.2f} deg")
        print(f"  RMS error:         x {rms[0]:.2f}  y {rms[1]:.2f} mm  yaw {rms[2]:.2f} deg"
              f"   (target 5 mm / 1 deg)")
        print(f"  max |error|:       x {abs(e[:,0]).max():.2f}  y {abs(e[:,1]).max():.2f} mm  "
              f"yaw {abs(e[:,2]).max():.2f} deg")


if __name__ == "__main__":
    main()
