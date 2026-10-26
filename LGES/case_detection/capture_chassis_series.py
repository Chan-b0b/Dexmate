"""Capture a straight chassis series for offline detector-accuracy analysis.

The case stays put; the robot pre-aligns its rear caster with one move, then
takes --steps equal same-direction moves (closed-loop on the wheel encoders,
which matched the tape for aligned moves). At the start pose and after every
move it saves --frames distinct head-camera frames with the torso/head joints
and the cumulative per-wheel encoder travel, so the detector (or any other
method) can be re-run on them later against encoder truth.

    ~/vla_venv/bin/python capture_chassis_series.py --move b --layer 3
    ~/vla_venv/bin/python capture_chassis_series.py --move r --prealign-mm 100 --steps 3

Writes data/chassis_series/<stamp>_<move>/pose_<k>.npz with
    rgb (N,H,W,3)  q_torso (N,3)  q_head (N,3)  layers_remaining
    wheel_mm (2,)  cumulative |travel| [L, R] since the pre-align, mm
    move  ("f"|"b"|"l"|"r")
and prints the live YOLO case detection per pose. DRIVES THE ROBOT: E-stop in reach.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))           # .../Dexmate  -> package LGES.ik_demo
sys.path.insert(0, str(_HERE.parents[1] / "perception"))

from detect_case_bev import detect_case_bev          # noqa: E402
from LGES.ik_demo import move_chassis as mc          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move", choices=("f", "b", "l", "r"), required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--prealign-mm", type=float, default=50.0)
    ap.add_argument("--step-mm", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=5, help="distinct frames saved per pose")
    ap.add_argument("--speed", type=float, default=0.05)
    args = ap.parse_args()

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot
    from utils import set_head_pitch

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    robot = Robot(configs=configs)
    for part in (robot.torso, robot.head, robot.chassis):
        part.set_subscription_policy("always_on", recursive=True)
    robot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
    robot.sensors.head_camera.wait_for_active(timeout=5.0)
    set_head_pitch(robot, angle=24.0, tol_deg=0.3)
    time.sleep(2.0)

    vx, vy = {"f": (1, 0), "b": (-1, 0), "l": (0, 1), "r": (0, -1)}[args.move]

    # steer once to the exact angle (see eval_chassis.align: _pre_steer stops ~2 deg short)
    signed = mc._pre_steer(robot, vx * args.speed, vy * args.speed)
    if signed is None:
        raise SystemExit("pre-steer failed")
    unit = np.asarray(signed, dtype=float) / args.speed
    cur = np.asarray(robot.chassis.steering_angle, dtype=float)
    cand = -np.arctan2(vy, vx) + np.pi * np.array([-1.0, 0.0, 1.0])
    steer = np.array([cand[np.argmin(np.abs(cand - c))] for c in cur])
    robot.chassis.set_steering_angle(steer, wait_time=2.0)

    def enc() -> np.ndarray:
        return np.asarray(robot.chassis.wheel_encoder_pos, dtype=float)

    def drive(dist_m: float) -> None:
        e0 = enc()
        t_end = time.monotonic() + dist_m / 0.01 * 1.5 + 5.0
        while time.monotonic() < t_end:
            rem = dist_m - float(np.mean(np.abs(enc() - e0)))
            if rem <= 0.0005:
                break
            robot.chassis._send_single_command(steer, unit * float(np.clip(rem, 0.01, args.speed)))
            time.sleep(0.02)
        else:
            print("WARNING: encoder drive timed out")
        mc._stop(robot)

    def grab(n: int, skip: int = 3):
        rgbs, qts, qhs, prev, seen, t0 = [], [], [], None, 0, time.time()
        while len(rgbs) < n and time.time() - t0 < 30:
            o = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
            rgb = o.get("data") if isinstance(o, dict) else o
            if rgb is None or (prev is not None and np.array_equal(rgb, prev)):
                time.sleep(0.01)
                continue
            prev = rgb
            seen += 1
            if seen <= skip:
                continue
            rgbs.append(rgb.copy())
            qts.append(np.asarray(robot.torso.get_joint_pos(), dtype=float))
            qhs.append(np.asarray(robot.head.get_joint_pos(), dtype=float))
        return np.stack(rgbs), np.stack(qts), np.stack(qhs)

    out = _HERE / "data" / "chassis_series" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.move}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"pre-align: {args.move} {args.prealign_mm:.0f} mm", flush=True)
    drive(args.prealign_mm / 1000.0)
    time.sleep(2.0)
    e_start = enc()
    for k in range(args.steps + 1):
        if k:
            drive(args.step_mm / 1000.0)
            time.sleep(2.0)
        wheel_mm = np.abs(enc() - e_start) * 1000.0
        rgb, qt, qh = grab(args.frames)
        np.savez(out / f"pose_{k}.npz", rgb=rgb, q_torso=qt, q_head=qh,
                 layers_remaining=args.layer, wheel_mm=wheel_mm, move=args.move)
        dets = [detect_case_bev(rgb[i], qt[i], qh[i], layers_remaining=args.layer)
                for i in range(len(rgb))]
        xy = np.array([d.base_xy for d in dets if d.found]) * 1000.0
        dims = np.array([d.dims_px for d in dets if d.found])
        msg = (f"pose {k}: wheel L/R {wheel_mm[0]:.1f}/{wheel_mm[1]:.1f} mm | ")
        if len(xy):
            med = np.median(xy, 0)
            msg += (f"case x={med[0]:.1f} y={med[1]:.1f} mm  size {np.median(dims[:, 0]):.1f}x"
                    f"{np.median(dims[:, 1]):.1f}px  ({len(xy)}/{len(rgb)} found)")
        else:
            msg += "case not detected"
        print(msg, flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
