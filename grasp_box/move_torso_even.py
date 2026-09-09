# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating robot torso movement control.

This script shows how to safely move the robot torso through a sequence of positions
while enforcing velocity limits and safety checks.
"""

# import debugpy

# # Initialize debugpy for remote debugging
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached!")


import time

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p")
def main(
    target_angles_deg: tuple[float, float, float] = (30.0, 110.0, 15.0),
    velocity_scale: float = 0.3,
    wait_time: float = 5.0,
) -> None:
    """Move torso to the specified target joint angles.

    Args:
        target_angles_deg: Target torso joint angles in degrees (joint0, joint1, joint2).
        velocity_scale: Velocity scale in (0, 1], as a fraction of the hardware
            velocity ceiling, applied to the motion.
        wait_time: Time to wait after sending the command, in seconds, so the
            motion plugin has time to finish moving before disconnecting.

    Returns:
        None
    """
    # Safety warnings and confirmation
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around robot before proceeding.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    with Robot() as bot:
        current_angles = np.asarray(bot.torso.get_state()["pos"], dtype=float)
        target_angles = np.deg2rad(target_angles_deg)

        logger.info(f"Current angles (rad): {current_angles}")
        logger.info(f"Target angles (rad): {target_angles}")
        logger.info(f"Velocity scale: {velocity_scale}")

        bot.torso.move_joint_pos(
            target_angles, relative=False, velocity_scale=velocity_scale
        )
        time.sleep(wait_time)

if __name__ == "__main__":
    tyro.cli(main)

# 5, 5, -5
# 20, 20, -5
# 70, 120, 10
# 30, 110, 15