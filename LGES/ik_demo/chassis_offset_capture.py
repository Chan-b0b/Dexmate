"""Chassis-offset capture: head depth + right wrist photo at small chassis offsets
around one bin station, for the wrist-camera bin tracking study (bin_pose/).

Both arms stay at home (the right arm's home is its wrist-camera view pose); only
the CHASSIS moves, in small slow steps (make_steps: +-5 cm sideways, +-3 cm
(--x-cm) forward/back, +-2 deg turns), and comes back to the start. The wheels need not be
exact: at every stop the head depth camera measures the bin (0.8 mm repeatable
with the arm out of the way), which is the truth the wrist photo is scored against.

Run from LGES/ with the robot at a bin station, bin centre ~0.7-0.8 m ahead,
nothing held (E-stop in reach — moves the chassis):
    ~/vla_venv/bin/python -m ik_demo.chassis_offset_capture

Writes RUN_LOG_DIR/chassis_offset/<stamp>/<k>_<label>_<j>.npz with
    head_rgb  head_depth  right_wrist_camera_rgb  q_torso  q_head  q_left_arm
    q_right_arm  wheel_enc  step  cmd_offset (x m, y m, yaw deg; commanded, cumulative)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg

# (command, amount): l/r/f/b metres, tl/tr degrees. A capture follows every step.
def make_steps(x_m: float = 0.03) -> tuple:
    """Step list: sideways +1/+3/+5/-2/-5 cm, forward/back to +-x_m (two steps each
    way), turns +-2 deg, one combined stop, back to the start."""
    a = round(x_m - 0.02, 4)          # the second, smaller step of each x excursion
    return (
        ("start", 0.0),
        ("l", 0.01), ("l", 0.02), ("l", 0.02),          # y +1, +3, +5 cm
        ("r", 0.07), ("r", 0.03),                       # y -2, -5
        ("l", 0.05),                                    # y 0
        ("f", 0.02), ("f", a),                          # x +2, +x_m
        ("b", x_m + 0.02), ("b", a),                    # x -2, -x_m
        ("f", x_m),                                     # x 0
        ("tl", 2.0), ("tr", 4.0), ("tl", 2.0),          # yaw +2, -2, 0
        ("l", 0.03), ("f", 0.02), ("tr", 1.5),          # combined: y +3, x +2, yaw -1.5
        ("tl", 1.5), ("b", 0.02), ("r", 0.03),          # back to the start
    )


STEPS = make_steps()
SPEED_MS = 0.05
TURN_RADS = 0.10
SETTLE_S = 1.0
FRAMES = 2


def _offsets(steps=None):
    x = y = yaw = 0.0
    for cmd, a in (STEPS if steps is None else steps):
        x += {"f": a, "b": -a}.get(cmd, 0.0)
        y += {"l": a, "r": -a}.get(cmd, 0.0)
        yaw += {"tl": a, "tr": -a}.get(cmd, 0.0)
        yield cmd, a, (x, y, yaw)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x-cm", type=float, default=3.0,
                    help="forward/back excursion (cm, both ways) — check the table clearance first")
    steps = make_steps(ap.parse_args().x_cm / 100.0)
    # dexcontrol's import runs logger.configure() — before setup_logging (see held_view_capture)
    from dexcontrol.core.config import get_robot_config
    from . import chassis_sequence as cs
    from .arm import connect_robot
    from .go_home import both_arms_home
    from .move_chassis import (move_backward, move_forward, strafe_left, strafe_right,
                               turn_ccw, turn_cw)

    moves = {"l": strafe_left, "r": strafe_right, "f": move_forward, "b": move_backward}
    turns = {"tl": turn_ccw, "tr": turn_cw}
    span = np.array([o for _, _, o in _offsets(steps)])
    cs.setup_logging()
    out = Path(cfg.RUN_LOG_DIR) / "chassis_offset" / cs.run_stamp()
    out.mkdir(parents=True, exist_ok=True)
    logger.warning("CHASSIS OFFSET CAPTURE — arms home, the CHASSIS moves {} small steps "
                   "(x {:+.0f}..{:+.0f} cm, y {:+.0f}..{:+.0f} cm, yaw {:+.1f}..{:+.1f} deg) and returns. "
                   "E-stop in reach.", len(steps) - 1, span[:, 0].min() * 100, span[:, 0].max() * 100,
                   span[:, 1].min() * 100, span[:, 1].max() * 100, span[:, 2].min(), span[:, 2].max())
    if input("Connect + safe-home both arms? [y/N]: ").strip().lower() != "y":
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin_pose"))
    try:
        from fit_bin import fit, rim_points       # live truth print only
    except Exception:  # noqa: BLE001
        fit = rim_points = None

    configs = get_robot_config()
    for n in ("head_camera", "right_wrist_camera"):
        configs.enable_sensor(n)
        configs.sensors[n].transport = "zenoh"
    with connect_robot(configs) as bot:
        for part in (bot.left_arm, bot.right_arm, bot.torso, bot.head, bot.chassis):
            part.set_subscription_policy("always_on", recursive=True)
        head, wrist = bot.sensors.head_camera, bot.sensors.right_wrist_camera
        for st in ("left_rgb", "depth"):
            if head._streams.get(st) is not None:
                head._streams[st].set_subscription_policy("always_on")
        wrist._streams["rgb"].set_subscription_policy("always_on")
        head.wait_for_active(timeout=5.0)
        if not wrist.wait_for_active(timeout=5.0):
            raise SystemExit("right wrist camera not publishing (dexsensor on the nano)")
        both_arms_home(bot)
        cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
        if input("Robot at the bin station (bin ahead, nothing held)? Start? [y/N]: ").strip().lower() != "y":
            return

        def fresh_wrist(timeout_s=3.0):
            def g():
                o = wrist.get_obs(include_timestamp=True)
                return (o.get("data"), o.get("timestamp")) if isinstance(o, dict) else (o, None)
            img, ts = g()
            t_end, ticks = time.monotonic() + timeout_s, 0
            while ts is not None and ticks < 2 and time.monotonic() < t_end:
                time.sleep(0.02)
                img2, ts2 = g()
                if img2 is not None and ts2 != ts:
                    img, ts, ticks = img2, ts2, ticks + 1
            return img

        for k, (cmd, a, off) in enumerate(_offsets(steps)):
            if cmd in moves:
                moves[cmd](bot, distance_m=a, speed=SPEED_MS)
            elif cmd in turns:
                turns[cmd](bot, angle_deg=a, speed=TURN_RADS)
            time.sleep(SETTLE_S)
            label = "start" if cmd == "start" else f"{cmd}{a:g}"
            for j in range(FRAMES):
                rgb = cs._head_rgb(bot)
                depth = head.get_depth()
                w = fresh_wrist()
                q_torso, q_head = cs._joints(bot)
                np.savez_compressed(
                    out / f"{k:02d}_{label}_{j}.npz", head_rgb=rgb, head_depth=depth,
                    right_wrist_camera_rgb=w, q_torso=q_torso, q_head=q_head,
                    q_left_arm=np.asarray(bot.left_arm.get_joint_pos()),
                    q_right_arm=np.asarray(bot.right_arm.get_joint_pos()),
                    wheel_enc=np.asarray(bot.chassis.wheel_encoder_pos, dtype=float),
                    step=k, cmd_offset=np.asarray(off))
            msg = ""
            if fit is not None and rgb is not None and depth is not None:
                try:
                    P, _ = rim_points(dict(rgb=rgb, depth=depth, q_torso=q_torso, q_head=q_head), rim_z=0.891)
                    p = fit(P[:, :2], LW=(0.720, 0.413))
                    msg = f" | head-depth bin ({p[0]:.4f},{p[1]:+.4f}) yaw {np.degrees(p[2]):+.2f}"
                except Exception as e:  # noqa: BLE001
                    msg = f" | bin fit failed ({e})"
            logger.info("[offset] step {:2d} {:6s} cmd offset x {:+.0f} y {:+.0f} cm yaw {:+.1f}{}",
                        k, label, off[0] * 100, off[1] * 100, off[2], msg)
        logger.info("[offset] done — frames in {}", out)


if __name__ == "__main__":
    main()
