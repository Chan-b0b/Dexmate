"""Teach-mode helper: lead the arm by hand and record joint positions.

Enables compliant (admittance) control on the left arm so you can physically
guide it to a pose, then press SPACE to record the 7 joint angles. Recorded
poses are appended to a JSON file on disk (default: taught_joint_poses.json).

    python -m case_battery_demo.teach_joint_pose                # default side (cfg.ARM_SIDE)
    python -m case_battery_demo.teach_joint_pose --side right   # e.g. the gripper's lower-right place pose

Controls:
    SPACE : record current joint positions
    q / Ctrl+C : quit

Tuning (top of file):
    CONTROL_HZ      — admittance loop rate
    KD_GAIN         — higher = stiffer (harder to push)
    FORCE_THRESHOLDS — per-joint deadband (A); lower = more sensitive
    BASELINE_TAU    — EMA time constant (s) for gravity-bias removal
"""

from __future__ import annotations

import json
import os
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import tyro
from dexcomm.utils import RateLimiter
from loguru import logger

from dexcontrol.robot import Robot

from . import config as cfg

CONTROL_HZ = 200.0
KD_GAIN = 0.3
#           index:  0     1     2     3     4     5     6
#                   shoulder    elbow       wrist
FORCE_THRESHOLDS = np.array([0.20, 0.20, 0.15, 0.15, 0.08, 0.08, 0.08])
BASELINE_TAU = 0.5

OUTPUT_FILE = Path(__file__).parent / "taught_joint_poses.json"


def _get_key_nonblock() -> str | None:
    """Return one keypress if one is pending, else None (non-blocking)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _load_poses(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_poses(path: Path, poses: list[dict]) -> None:
    path.write_text(json.dumps(poses, indent=2))


def _joint_force_from_current(
    current_cur: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    offset = current_cur - baseline
    return np.where(np.abs(offset) < FORCE_THRESHOLDS, 0.0, -offset)


def main(side: str = cfg.ARM_SIDE) -> None:
    logger.warning("=" * 70)
    logger.warning("SAFETY: {} arm will become COMPLIANT (lead-through).", side)
    logger.warning("It stays actively controlled — never goes limp.")
    logger.warning("Move slowly and keep the e-stop within reach!")
    logger.warning("=" * 70)
    if input("Start? [y/N]: ").strip().lower() != "y":
        return

    poses = _load_poses(OUTPUT_FILE)
    logger.info("Output file: {}", OUTPUT_FILE)
    logger.info("Existing poses in file: {}", len(poses))
    print("\nSPACE = record   q / Ctrl-C = quit\n")

    dt = 1.0 / CONTROL_HZ
    D_inv = np.linalg.inv(np.diag([2.0] * 7) * KD_GAIN)
    alpha = np.exp(-dt / BASELINE_TAU)

    with Robot() as bot:
        arm = getattr(bot, f"{side}_arm")

        baseline = np.array(arm.get_joint_current())
        rate = RateLimiter(rate_hz=CONTROL_HZ)

        try:
            while True:
                joint_pos = np.array(arm.get_joint_pos())
                joint_current = np.array(arm.get_joint_current())

                ext_force = _joint_force_from_current(joint_current, baseline)
                arm.set_joint_pos((joint_pos + D_inv @ ext_force * dt).tolist())

                baseline = alpha * baseline + (1.0 - alpha) * joint_current

                key = _get_key_nonblock()
                if key == " ":
                    q = [round(float(v), 5) for v in joint_pos]
                    entry = {"name": f"pose_{len(poses)}", "joints": q}
                    poses.append(entry)
                    _save_poses(OUTPUT_FILE, poses)
                    logger.info("Recorded pose_{}: {}", len(poses) - 1, q)
                elif key in ("q", "\x03"):
                    break

                rate.sleep()

        except KeyboardInterrupt:
            pass

    logger.info("Done. {} poses saved to {}", len(poses), OUTPUT_FILE)


if __name__ == "__main__":
    tyro.cli(main)
