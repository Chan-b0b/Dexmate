"""Live accuracy check of the case BEV detector by driving the CHASSIS.

The case stays put; the robot is driven a little and the displacement is
measured with a tape measure. The case then must appear to move in base_link by
minus the robot's displacement, so

    detected robot displacement = -(change of detected case xy)
    error = detected - tape-measured

Straight moves are driven CLOSED-LOOP on the wheel encoders (stop when the
wheels have rolled the distance) and the encoder displacement is the truth by
default — no tape per step. The encoder reads metres of wheel travel
(--enc-scale 1.0; a 1001 tape check read 267 mm for 0.2655 units, within the
tape / 1 mm encoder-step error). Strafe uses the same wheel travel but check it
once with tape (type the tape value at the prompt). Wheel slip is invisible to
the encoders.
Each step is compared with the pose right before it.

    ~/vla_venv/bin/python eval_chassis.py --layer 1
    ~/vla_venv/bin/python eval_chassis.py --layer 1 --out chassis_acc.csv

Commands (mm / deg; base_link axes at the CURRENT pose: forward, left):
    f 100   b 100   l 100   r 100      drive forward / back / left / right
    ccw 3   cw 3                       turn in place (only yaw is scored)
    q / empty line                     finish

Before each move you confirm with Enter (Ctrl-C aborts). Afterwards Enter
accepts the encoder displacement, or type a tape-measured "fwd_mm left_mm" to
override. Turns stay open-loop: type the measured angle (deg, ccw +) or Enter
for the commanded one.

Do the straight moves first: after a turn the robot frame is rotated, so later
straight steps are only approximately comparable. Keep the case in view and the
path clear; one move is capped at --max-mm. This DRIVES the robot: E-stop in reach.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))           # .../Dexmate  -> package LGES.ik_demo
sys.path.insert(0, str(_HERE.parents[1] / "perception"))

from detect_case_bev import detect_case_bev          # noqa: E402  (flat case_detection modules)
from LGES.ik_demo import move_chassis as mc          # noqa: E402


def _wrap90(d: float) -> float:
    """The case is 180-deg symmetric: wrap a yaw difference to [-90, 90)."""
    return (d + 90.0) % 180.0 - 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=1, help="layers_remaining for the warp plane")
    ap.add_argument("--plane-z", type=float, help="force the warp plane (m)")
    ap.add_argument("--samples", type=int, default=15, help="distinct frames per pose")
    ap.add_argument("--max-mm", type=float, default=300.0, help="cap for one straight move")
    ap.add_argument("--max-deg", type=float, default=10.0, help="cap for one turn")
    ap.add_argument("--speed", type=float, default=0.05, help="max chassis speed m/s")
    ap.add_argument("--prealign-mm", type=float, default=20.0,
                    help="short move in the test direction before the reference pose, so "
                    "the rear caster is already trailing (0 = off)")
    ap.add_argument("--enc-scale", type=float, default=1.0,
                    help="metres of travel per encoder unit")
    ap.add_argument("--angle", type=float, default=24.0, help="head pitch (deg)")
    ap.add_argument("--out", default="chassis_accuracy.csv")
    args = ap.parse_args()

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot
    from utils import set_head_pitch

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    robot = Robot(configs=configs)
    # same as chassis_sequence: no idle pause on the state subscribers
    for part in (robot.torso, robot.head, robot.chassis):
        part.set_subscription_policy("always_on", recursive=True)
    robot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
    robot.sensors.head_camera.wait_for_active(timeout=5.0)
    set_head_pitch(robot, angle=args.angle, tol_deg=0.3)
    time.sleep(2.0)

    def grab():
        o = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
        return o.get("data") if isinstance(o, dict) else o

    def measure(skip: int = 3):
        """Median (x, y, yaw) over --samples DISTINCT frames, skipping the first
        `skip` distinct ones (they may predate the end of the move)."""
        out, prev, seen, tries = [], None, 0, 0
        while len(out) < args.samples and tries < (args.samples + skip) * 30:
            tries += 1
            rgb = grab()
            if rgb is None or (prev is not None and np.array_equal(rgb, prev)):
                time.sleep(0.01)
                continue
            prev = rgb
            seen += 1
            if seen <= skip:
                continue
            q_torso = np.asarray(robot.torso.get_joint_pos(), dtype=np.float64)
            q_head = np.asarray(robot.head.get_joint_pos(), dtype=np.float64)
            d = detect_case_bev(rgb, q_torso, q_head, layers_remaining=args.layer,
                                plane_z=args.plane_z)
            if d.found:
                out.append((d.base_xy[0] * 1000, d.base_xy[1] * 1000, d.base_yaw_deg))
        if len(out) < max(3, args.samples // 2):
            return None
        a = np.asarray(out)
        yaw = a[0, 2] + np.array([_wrap90(y - a[0, 2]) for y in a[:, 2]])
        return dict(x=float(np.median(a[:, 0])), y=float(np.median(a[:, 1])),
                    yaw=float(np.median(yaw)), sx=float(a[:, 0].std()),
                    sy=float(a[:, 1].std()))

    def align(vx: float, vy: float):
        """Steer both wheels to the move direction. Returns (steer, unit,
        changed): exact angles, +-1 wheel-speed signs, and whether the wheels
        had to pivot (pivoting rolls them ~2 mm and once preceded a 7 mm L/R
        travel split on the next forward move, 1001)."""
        cur0 = np.asarray(robot.chassis.steering_angle, dtype=float)
        signed = mc._pre_steer(robot, vx * args.speed, vy * args.speed)
        if signed is None:
            raise RuntimeError("pre-steer failed")
        unit = np.asarray(signed, dtype=float) / args.speed      # +-1 per wheel
        # _pre_steer accepts any angle within CHASSIS_PRESTEER_TOL_RAD (~2.9 deg):
        # a strafe left stopped at -87.7 deg and so also drove ~4 mm forward per
        # 100 mm (1001). Pin each wheel to the exact angle nearest where it is.
        cur = np.asarray(robot.chassis.steering_angle, dtype=float)
        base = -np.arctan2(vy, vx)
        cand = base + np.pi * np.array([-1.0, 0.0, 1.0])
        steer = np.array([cand[np.argmin(np.abs(cand - c))] for c in cur])
        robot.chassis.set_steering_angle(steer, wait_time=2.0)
        off = np.rad2deg(np.abs(np.asarray(robot.chassis.steering_angle) - steer)).max()
        if off > 0.3:
            print(f"  WARNING: steering {off:.2f} deg off the exact angle")
        changed = bool(np.rad2deg(np.abs(cur0 - steer)).max() > 0.3)
        return steer, unit, changed

    def drive_encoder(steer, unit, dist_m: float) -> float:
        """Closed-loop straight move on aligned wheels: stream wheel speed with
        a P slow-down until the mean wheel travel reaches dist_m, then the
        streamed stop. Returns the final per-wheel travel [L, R] (m, after the stop)."""
        e0 = np.asarray(robot.chassis.wheel_encoder_pos, dtype=float)

        def travel() -> np.ndarray:                              # per wheel [L, R], m
            e = np.asarray(robot.chassis.wheel_encoder_pos, dtype=float)
            return np.abs(e - e0) * args.enc_scale

        t_end = time.monotonic() + dist_m / 0.01 * 1.5 + 5.0      # timeout guard
        while time.monotonic() < t_end:
            rem = dist_m - float(np.mean(travel()))
            if rem <= 0.0005:
                break
            v = float(np.clip(1.0 * rem, 0.01, args.speed))       # 1/s slow-down, 1 cm/s floor
            robot.chassis._send_single_command(steer, unit * v)
            time.sleep(0.02)
        else:
            print("  WARNING: encoder drive timed out")
        mc._stop(robot)
        time.sleep(0.5)
        return travel()

    def ask_float(prompt: str, default: float) -> float:
        s = input(prompt).strip()
        return default if not s else float(s)

    prev = measure()
    if prev is None:
        raise SystemExit("case not detected at the start pose")
    print(f"start: case at x={prev['x']:.1f} y={prev['y']:+.1f} mm yaw={prev['yaw'] % 180:.2f} deg "
          f"(scatter {prev['sx']:.2f}/{prev['sy']:.2f} mm)")

    rows = []
    while True:
        line = input("\nmove ('f|b|l|r <mm>' or 'ccw|cw <deg>', empty=quit): ").strip().lower()
        if not line or line == "q":
            break
        try:
            kind, val = line.split()
            val = float(val)
        except ValueError:
            print("  format: f 100 | b 100 | l 100 | r 100 | ccw 3 | cw 3")
            continue
        turn = kind in ("ccw", "cw")
        if kind not in ("f", "b", "l", "r", "ccw", "cw") or val <= 0:
            print("  unknown command")
            continue
        if (turn and val > args.max_deg) or (not turn and val > args.max_mm):
            print(f"  too large (cap {args.max_deg if turn else args.max_mm})")
            continue
        input(f"  CLEAR THE PATH. Enter to drive '{line}' (Ctrl-C aborts)... ")
        if turn:
            (mc.turn_ccw if kind == "ccw" else mc.turn_cw)(robot, angle_deg=val)  # config turn speed
        else:
            vx, vy = {"f": (1, 0), "b": (-1, 0), "l": (0, 1), "r": (0, -1)}[kind]
            steer, unit, changed = align(vx, vy)
            if args.prealign_mm > 0:
                # A direction reversal swings the passive rear caster round and
                # drags the whole base ~11 mm further than the wheels rolled
                # (1001: b/f reversals read 62 mm for 51 mm of wheel travel,
                # same-direction repeats matched). Pre-move so the caster is
                # already trailing, then take the reference.
                drive_encoder(steer, unit, args.prealign_mm / 1000.0)
                print(f"  caster pre-aligned ({args.prealign_mm:.0f} mm) — re-taking the start pose")
                time.sleep(1.0)
                m = measure()
                if m is not None:
                    prev = m
            elif changed:
                # compare against the pose AFTER the pivot, not before it
                print("  wheels pivoted — re-taking the start pose")
                time.sleep(1.0)
                m = measure()
                if m is not None:
                    prev = m
            wheels = drive_encoder(steer, unit, val / 1000.0) * 1000.0
            enc_mm = float(wheels.mean())
            enc_f, enc_l = vx * enc_mm, vy * enc_mm
            # Straight f/b: the two wheels sit side by side and roll along x, so a
            # travel difference is a turn (1001: L/R 49/54 mm -> -0.64 deg, detected
            # -0.43). Strafe: they roll along the line joining them — no yaw info.
            enc_yaw = (vx * float(np.rad2deg((wheels[1] - wheels[0]) / 1000.0
                                             / robot.chassis._wheels_dist))
                       if vx else float("nan"))
            print(f"  encoder travel L/R {wheels[0]:.1f}/{wheels[1]:.1f} mean {enc_mm:.1f} mm "
                  f"(target {val:.0f})" + (f", turn {enc_yaw:+.2f} deg" if vx else ""))
        time.sleep(1.0)

        cmd_f = {"f": val, "b": -val}.get(kind, 0.0)
        cmd_l = {"l": val, "r": -val}.get(kind, 0.0)
        cmd_yaw = val if kind == "ccw" else -val if kind == "cw" else 0.0
        if turn:
            true_yaw = ask_float(f"  measured turn deg (ccw +, Enter = {cmd_yaw:+.1f}): ", cmd_yaw)
            true_f = true_l = 0.0
        else:
            s = input(f"  tape 'fwd_mm left_mm' (Enter = encoder {enc_f:+.1f} {enc_l:+.1f}): ").strip()
            true_f, true_l = (enc_f, enc_l) if not s else (float(t) for t in s.replace(",", " ").split())
            true_yaw = 0.0 if np.isnan(enc_yaw) else enc_yaw

        cur = measure()
        if cur is None:
            print("  case not detected — skipping this step")
            continue
        # case moves opposite to the robot; a robot yaw change rotates the case yaw the other way
        det_f, det_l = -(cur["x"] - prev["x"]), -(cur["y"] - prev["y"])
        det_yaw = -_wrap90(cur["yaw"] - prev["yaw"])
        if turn:
            err = (float("nan"), float("nan"), det_yaw - true_yaw)
            print(f"  yaw: detected {det_yaw:+.2f} deg vs measured {true_yaw:+.2f} -> error {err[2]:+.2f} deg"
                  f"   (commanded {cmd_yaw:+.1f})")
        else:
            # Predicted case position after the move: translate by the true move,
            # rotate by the encoder turn (taken about the base_link origin — the
            # real centre is on the wheel axis, ~2 mm difference per 0.6 deg).
            psi = np.deg2rad(true_yaw)
            rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
            pred = rot @ (np.array([prev["x"], prev["y"]]) - np.array([true_f, true_l]))
            err = (pred[0] - cur["x"], pred[1] - cur["y"], det_yaw - true_yaw)
            print(f"  detected robot move: fwd {det_f:+.1f} left {det_l:+.1f} mm, yaw change {det_yaw:+.2f} deg")
            print(f"  error vs truth: fwd {err[0]:+.2f}  left {err[1]:+.2f} mm  yaw {err[2]:+.2f} deg   "
                  f"(encoder-vs-truth: fwd {enc_f - true_f:+.1f} left {enc_l - true_l:+.1f} mm)")
        if turn:
            enc_f = enc_l = float("nan")
        rows.append(dict(cmd=line, cmd_fwd=cmd_f, cmd_left=cmd_l, cmd_yaw=cmd_yaw,
                         enc_fwd=enc_f, enc_left=enc_l,
                         enc_wheel_l=float("nan") if turn else wheels[0],
                         enc_wheel_r=float("nan") if turn else wheels[1],
                         tape_fwd=true_f, tape_left=true_l, tape_yaw=true_yaw,
                         det_fwd=det_f, det_left=det_l, det_yaw=det_yaw,
                         err_fwd_mm=err[0], err_left_mm=err[1], err_yaw_deg=err[2]))
        prev = cur

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} steps -> {args.out}")
        for name, key, unit in (("fwd", "err_fwd_mm", "mm"), ("left", "err_left_mm", "mm"),
                                ("yaw", "err_yaw_deg", "deg")):
            v = np.array([r[key] for r in rows], dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                print(f"  {name:5s} bias {v.mean():+.2f}  RMS {np.sqrt((v ** 2).mean()):.2f}  "
                      f"max {abs(v).max():.2f} {unit}   (n={v.size}; target 5 mm / 1 deg)")


if __name__ == "__main__":
    main()
