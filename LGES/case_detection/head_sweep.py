"""Head-pitch invariance test for the case BEV detector.

Leave the case where it is and sweep only the head pitch. The case does not
move, so its detected base_link XY / yaw must not depend on where the head
looks; any shift with head angle is a calibration / plane-height / scale error
(the size of the shift is the size of the error). No ground truth needed.

    ~/vla_venv/bin/python head_sweep.py --layer 3
    ~/vla_venv/bin/python head_sweep.py --layer 3 --angles 18 21 24 27 30 --samples 20

Each visit is the median of --samples DISTINCT camera frames after the head
settles. Every angle is visited --repeats times (shuffled order per round), so
the visit-to-visit std at one angle is the noise floor (head not repeating,
slow drift) and a straight-line fit over all visits says whether the shift with
angle is real or just that noise.
Angles use set_head_pitch's convention (chassis_sequence pins 24). Only the
head moves. Cannot see an error that is the same at every angle (e.g. an
origin offset).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from detect_case_bev import detect_case_bev


def _wrap90(d: float) -> float:
    """Case is 180-deg symmetric; wrap a yaw difference to [-90, 90)."""
    return (d + 90.0) % 180.0 - 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=1, help="layers_remaining for the warp plane")
    ap.add_argument("--plane-z", type=float, help="force the warp plane (m)")
    ap.add_argument("--angles", type=float, nargs="+", default=[24, 20, 22, 26, 28],
                    help="head pitch angles (deg); the first is the reference for the shifts")
    ap.add_argument("--samples", type=int, default=15, help="distinct frames per visit")
    ap.add_argument("--repeats", type=int, default=3,
                    help="visits per angle; each round is a shuffled pass over --angles, so "
                    "the visit-to-visit std separates noise from a real angle trend")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed (for the visit order)")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds after each head move")
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

    def measure():
        out, prev, tries = [], None, 0
        while len(out) < args.samples and tries < args.samples * 20:
            tries += 1
            o = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
            rgb = o.get("data") if isinstance(o, dict) else o
            if rgb is None or (prev is not None and np.array_equal(rgb, prev)):
                time.sleep(0.01)
                continue
            prev = rgb
            q_torso = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)
            q_head = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
            d = detect_case_bev(rgb, q_torso, q_head, layers_remaining=args.layer,
                                plane_z=args.plane_z)
            if d.found:
                out.append((d.base_xy[0], d.base_xy[1], d.base_yaw_deg, d.conf, *d.dims_px,
                            d.top_face_z, float(np.rad2deg(q_head[0]))))
        if len(out) < max(3, args.samples // 2):
            return None
        a = np.asarray(out)
        yaw = a[0, 2] + np.array([_wrap90(y - a[0, 2]) for y in a[:, 2]])
        return dict(x=np.median(a[:, 0]) * 1000, y=np.median(a[:, 1]) * 1000,
                    yaw=float(np.median(yaw) % 180.0),
                    sx=a[:, 0].std() * 1000, sy=a[:, 1].std() * 1000, syaw=float(yaw.std()),
                    conf=a[:, 3].mean(), long_px=np.median(a[:, 4]), short_px=np.median(a[:, 5]),
                    head=np.median(a[:, 7]), n=len(out))

    rng = np.random.default_rng(args.seed)
    plan = []                                   # each round visits every angle once, shuffled
    for _ in range(args.repeats):
        plan += [float(a) for a in rng.permutation(args.angles)]
    res, yaw0 = [], None                        # (angle, measurement) per visit
    for k, ang in enumerate(plan, 1):
        set_head_pitch(robot, angle=ang)          # blocks until settled (no tol_deg)
        time.sleep(args.settle)
        m = measure()
        if m is None:
            print(f"[{k}/{len(plan)}] angle {ang:5.1f}: case not detected")
            continue
        if yaw0 is None:
            yaw0 = m["yaw"]
        m["dyaw"] = _wrap90(m["yaw"] - yaw0)
        res.append((ang, m))
        print(f"[{k}/{len(plan)}] angle {ang:5.1f}  x={m['x']:8.2f} y={m['y']:7.2f} mm  "
              f"dyaw={m['dyaw']:+6.3f}  scatter {m['sx']:.2f}/{m['sy']:.2f} mm  "
              f"long {m['long_px']:.1f} short {m['short_px']:.1f}px conf {m['conf']:.2f}")

    if not res:
        raise SystemExit("case never detected")
    angs = sorted({a for a, _ in res})
    by = {a: [m for aa, m in res if aa == a] for a in angs}
    keys = ("x", "y", "dyaw", "long_px", "short_px")
    mean = {a: {k: float(np.mean([m[k] for m in by[a]])) for k in keys} for a in angs}
    vstd = {a: {k: float(np.std([m[k] for m in by[a]], ddof=1)) if len(by[a]) > 1 else float("nan")
                for k in keys} for a in angs}
    ref = args.angles[0] if args.angles[0] in mean else angs[0]
    print(f"\nper angle, mean over visits (n) and visit-to-visit std; shift vs ref {ref:.1f}:")
    print("  angle  n |    x mm  (std)  dx   |    y mm  (std)  dy   |  dyaw deg (std) | long px (std)  dlong | short px (std)  dshort")
    for a in angs:
        mm, sd, r = mean[a], vstd[a], mean[ref]
        print(f"  {a:5.1f} {len(by[a]):2d} | {mm['x']:8.2f} ({sd['x']:.2f}) {mm['x']-r['x']:+5.2f} |"
              f" {mm['y']:7.2f} ({sd['y']:.2f}) {mm['y']-r['y']:+5.2f} |"
              f" {mm['dyaw']:+7.3f} ({sd['dyaw']:.3f}) | {mm['long_px']:7.2f} ({sd['long_px']:.2f})"
              f" {mm['long_px']-r['long_px']:+5.2f} | {mm['short_px']:7.2f} ({sd['short_px']:.2f})"
              f" {mm['short_px']-r['short_px']:+5.2f}")

    # Straight-line fit over ALL visits: is there a trend with angle, and is it
    # bigger than the visit-to-visit noise?
    A = np.array([a for a, _ in res])
    span = float(A.max() - A.min())
    print(f"\ntrend with head angle over {span:.0f} deg ({len(res)} visits):")
    if len(angs) < 2 or span == 0:
        print("  need >= 2 angles")
    else:
        for k, unit in (("x", "mm"), ("y", "mm"), ("dyaw", "deg"), ("long_px", "px"),
                        ("short_px", "px")):
            v = np.array([m[k] for _, m in res])
            slope, icpt = np.polyfit(A, v, 1)
            resid = v - (slope * A + icpt)
            se = resid.std(ddof=2) / (A.std() * np.sqrt(len(A))) if len(A) > 2 else float("nan")
            print(f"  {k:8s} slope {slope:+.3f} {unit}/deg (+-{se:.3f})  ->  "
                  f"{slope * span:+.2f} {unit} over the range, residual std {resid.std():.2f}")
    print("(trend is real if |slope| is well above its +- ; target: total x,y change < 2 mm)")


if __name__ == "__main__":
    main()
