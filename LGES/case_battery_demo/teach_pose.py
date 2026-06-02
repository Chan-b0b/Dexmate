"""Teach-mode helper: jog the suction cup (held vertical) and record 6-DOF poses.

One process. On start the arm is reoriented to the vertical approach
(cfg.GRASP_ORIENTATION_RPY) at its current position. You then jog the
end-effector in base_link XYZ while the orientation is *held* at the target
(it does not drift to whatever the arm currently is). Orientation keys let you
fine-tune true vertical. Record prints the full pose to paste into
config.TAUGHT_POSES.

    python -m case_battery_demo.teach_pose --step-m 0.005 --move-time 4.0

Position keys (base frame):
    w / s : +x / -x      a / d : +y / -y      r / f : +z / -z
Orientation keys (tune the held target, degrees):
    u / o : roll +/-     i / k : pitch +/-     j / l : yaw +/-
Steps:
    + / - : position step      , / . : orientation step      ] / [ : faster / slower
Other:
    SPACE : record current pose      q : quit

Drive the cup tip onto the target surface (cup vertical), then record.
"""

from __future__ import annotations

import sys
import termios
import tty
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot

_POSE_FILE  = Path(__file__).parent / "taught_ee_poses.txt"
_JOINT_FILE = Path(__file__).parent / "taught_joint_poses.txt"

from .grasp import SuctionMover
from . import config as cfg

# key -> (vector index, sign); position in metres, orientation in radians
_POS_KEYS = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "r": (2, +1), "f": (2, -1)}
_RPY_KEYS = {"u": (0, +1), "o": (0, -1), "i": (1, +1), "k": (1, -1), "j": (2, +1), "l": (2, -1)}


def _get_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main(step_m: float = 0.01, move_time: float = 1.0, ostep_deg: float = 5.0) -> None:
    """Cartesian-jog teach tool that holds a vertical approach orientation.

    Args:
        step_m: Initial position jog distance per keypress (m).
        move_time: Seconds per move. Larger = slower (velocity ~= step / move_time).
        ostep_deg: Initial orientation jog per keypress (degrees).
    """
    logger.warning("Be ready to press the e-stop. Arm WILL reorient to vertical on start.")
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    def _status(p, rpy) -> str:
        return (f"pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
                f"rpy_deg=({np.rad2deg(rpy[0]):.1f}, {np.rad2deg(rpy[1]):.1f}, {np.rad2deg(rpy[2]):.1f})  "
                f"| step={step_m*100:.1f}cm ostep={ostep_deg:.1f}deg move={move_time:.1f}s")

    with Robot() as bot:
        mover = SuctionMover(bot)

        if mover.software_estop_active():
            logger.warning("Software E-Stop is ACTIVE — the arm cannot move until released.")
            if input("Release software E-Stop and enable the arm? [y/N]: ").strip().lower() != "y":
                logger.info("Leaving E-Stop engaged; exiting.")
                return
            release = True
        else:
            release = False
        if not mover.ensure_ready(release_estop=release):
            logger.error("Arm not ready (E-Stop still active?). Aborting.")
            return

        logger.info("ARM_SIDE={}, EE_FRAME={}", cfg.ARM_SIDE, cfg.EE_FRAME)
        print(__doc__)

        # Explicit commanded target: position = current; orientation = vertical
        # (roll/pitch from config) with yaw = current yaw + YAW_OFFSET_DEG so the
        # battery aligns with the case slot.
        target_pos, cur_rpy = mover.current_ee_pose()
        target_rpy = np.array([
            cfg.GRASP_ORIENTATION_RPY[0],
            cfg.GRASP_ORIENTATION_RPY[1],
            np.deg2rad(cfg.YAW_OFFSET_DEG),
        ], dtype=float)
        logger.info("Reorienting to vertical target rpy_deg={} ...", np.rad2deg(target_rpy).round(1))
        pos, rpy = mover.goto(target_pos, target_rpy, step_duration=max(move_time, 3.0))
        print(_status(target_pos, target_rpy))

        while True:
            key = _get_key()
            if key == "q" or ord(key) == 3:  # q or Ctrl+C
                break
            elif key in _POS_KEYS:
                axis, sign = _POS_KEYS[key]
                target_pos[axis] += sign * step_m
                pos, rpy = mover.goto(target_pos, target_rpy, move_time)
            elif key in _RPY_KEYS:
                axis, sign = _RPY_KEYS[key]
                target_rpy[axis] += sign * np.deg2rad(ostep_deg)
                pos, rpy = mover.goto(target_pos, target_rpy, move_time)
            elif key in ("+", "="):
                step_m = min(step_m * 2, 0.05)
            elif key == "-":
                step_m = max(step_m / 2, 0.001)
            elif key == ".":
                ostep_deg = min(ostep_deg * 2, 30.0)
            elif key == ",":
                ostep_deg = max(ostep_deg / 2, 0.5)
            elif key == "]":
                move_time = max(move_time / 2, 0.2)
            elif key == "[":
                move_time = min(move_time * 2, 8.0)
            elif key == " ":
                p, rp = mover.current_ee_pose()
                q = mover._arm.get_joint_pos()
                ee_line    = " ".join(f"{v:.6f}" for v in (*p, *rp))
                joint_line = " ".join(f"{v:.6f}" for v in q)
                with _POSE_FILE.open("a")  as f: f.write(ee_line    + "\n")
                with _JOINT_FILE.open("a") as f: f.write(joint_line + "\n")
                print(f"\n>>> EE   : {ee_line}")
                print(f"    joints: {joint_line}")
                print(f"    saved to {_POSE_FILE.name} / {_JOINT_FILE.name}")
                continue
            else:
                continue
            print(f"\r{_status(target_pos, target_rpy)}   ", end="", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
