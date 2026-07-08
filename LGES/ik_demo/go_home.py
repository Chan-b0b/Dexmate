"""Safe homing: lift a low EE straight up, THEN joint-move to the home pose.

A plain move_joints(home) from a low EE (down in/near a box after a failed
pick/place) can sweep the arm through the box walls. safe_home() checks the
live EE height first: if it is below cfg.HOME_LIFT_MIN_EE_Z, the EE is raised
STRAIGHT UP (same xy, same orientation) to cfg.HOME_LIFT_EE_Z, and only then
does the joint move home run.

    from .go_home import safe_home, both_arms_home
    safe_home(mover)                  # one arm (any ArmMover/SuctionMover)
    both_arms_home(bot, left=mover)   # left + right, left first

CLI (homes BOTH arms, behind a prompt):
    python -m LGES.ik_demo.go_home
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from . import config as cfg
from .arm import ArmMover


def safe_home(arm: ArmMover) -> None:
    """Lift-if-low, then home, for one arm."""
    pos, rpy = arm.current_ee_pose()
    if pos[2] < cfg.HOME_LIFT_MIN_EE_Z:
        lift = (float(pos[0]), float(pos[1]), float(cfg.HOME_LIFT_EE_Z))
        logger.info("[{}] EE z={:.3f} < {:.2f} — lifting straight up to z={:.2f} before homing",
                    arm._side, pos[2], cfg.HOME_LIFT_MIN_EE_Z, cfg.HOME_LIFT_EE_Z)
        if arm.move_ee(lift, tuple(rpy)) is None:
            logger.warning("[{}] lift unreachable — homing from the current pose", arm._side)
    logger.info("[{}] -> home", arm._side)
    arm.move_joints(arm._home_seed)


def both_arms_home(bot, left: ArmMover | None = None) -> None:
    """Safe-home BOTH arms (left first — it's the one usually low over the box).
    Pass the existing left mover to reuse it; the right arm gets a fresh ArmMover."""
    try:
        safe_home(left if left is not None else ArmMover(robot=bot, side="left"))
    except Exception as e:
        logger.warning("left arm home failed: {}", e)
    try:
        safe_home(ArmMover(robot=bot, side="right", ee_frame="R_gripper_base"))
    except Exception as e:
        logger.warning("right arm home failed: {}", e)


def _main() -> None:
    from dexcontrol.robot import Robot

    logger.warning("Safe-homes BOTH arms (lift-if-low, then home joints).")
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return
    with Robot() as bot:
        both_arms_home(bot)
    logger.info("done")


if __name__ == "__main__":
    _main()
