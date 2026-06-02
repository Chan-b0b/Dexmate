"""Send both arms to the joint-space "Home" pose stored in default_pose.txt.

File format (comments / blank lines / leading whitespace allowed)::

     Home

     j1, j2, j3, j4, j5, j6, j7, JOINT     # left  arm
     j1, j2, j3, j4, j5, j6, j7, JOINT     # right arm

The trailing ``JOINT`` tag is ignored — only the 7 numeric joint values per
line are used. The first JOINT line is sent to the left arm, the second to
the right arm.
"""

from __future__ import annotations

import os
import time

import numpy as np
from loguru import logger

from . import config as cfg

DEFAULT_POSE_PATH = os.path.join(os.path.dirname(__file__), "default_pose.txt")


def _parse_joint_lines(path: str) -> list[np.ndarray]:
    """Return every line tagged ``JOINT`` as a 7-element numpy array."""
    poses: list[np.ndarray] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if not parts or parts[-1].upper() != "JOINT":
                continue
            nums = parts[:-1]
            if len(nums) != 7:
                raise ValueError(
                    f"{path}: expected 7 joint values per JOINT line, got {len(nums)}: {line!r}"
                )
            poses.append(np.array([float(x) for x in nums], dtype=float))
    if len(poses) < 2:
        raise ValueError(
            f"{path}: need at least two ', JOINT' lines (left then right arm); found {len(poses)}"
        )
    return poses


def go_to_default_pose(robot, duration: float = 4.0, path: str = DEFAULT_POSE_PATH) -> None:
    """Smoothly move both arms to the joint targets in *path*.

    Args:
        robot: an active ``dexcontrol.robot.Robot`` instance.
        duration: seconds to interpolate each arm.
        path: path to a default_pose.txt file.
    """
    poses = _parse_joint_lines(path)
    left_target, right_target = poses[0], poses[1]
    logger.info("[home_pose] left  target: {}", np.round(left_target, 4))
    logger.info("[home_pose] right target: {}", np.round(right_target, 4))

    # Arms must be in position mode (SuctionMover.ensure_ready already did this
    # for the suction arm; do the other one too so set_joint_pos is accepted).
    try:
        robot.left_arm.set_modes(["position"] * 7)
        robot.right_arm.set_modes(["position"] * 7)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[home_pose] set_modes skipped: {}", exc)

    left_start = robot.left_arm.get_joint_pos().astype(float)
    right_start = robot.right_arm.get_joint_pos().astype(float)

    n_steps = max(1, int(duration / cfg.CONTROL_DT))
    for step in range(n_steps):
        t = (step + 1) / n_steps
        alpha = t * t * (3 - 2 * t)  # smoothstep
        robot.set_joint_pos({
            "left_arm": left_start + alpha * (left_target - left_start),
            "right_arm": right_start + alpha * (right_target - right_start),
        })
        time.sleep(cfg.CONTROL_DT)
    logger.info("[home_pose] arms at default pose")
