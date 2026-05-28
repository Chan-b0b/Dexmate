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


import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p")
def main() -> None:
    """Move torso through a predefined sequence of positions.

    Returns:
        None
    """
    # Safety warnings and confirmation
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around robot before proceeding.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    with Robot() as bot:
        # Move to intermediate crouching position
        # if not bot.torso.is_pose_reached("crouch45_medium"):
        #     bot.torso.go_to_pose("crouch45_medium", wait_time=4)
        current_angles = np.asarray(bot.torso._get_state()["pos"], dtype=float)
        # Move to target joint angles (60, 120, 30 degrees)
        
        target_angles = np.deg2rad([60,120,30])
        
        # Set velocity proportional to joint error with conservative bounds.
        error = target_angles - current_angles
        kp = 0.40
        min_vel = 0
        max_vel = 0.8
        joint_vel = np.clip(np.abs(error) * kp, min_vel, max_vel)
        print(f'moving in {joint_vel} rad/s')
        logger.info(f"Current angles (rad): {current_angles}")
        logger.info(f"Target angles (rad): {target_angles}")
        logger.info(f"Error (rad): {error}")
        logger.info(f"Commanded joint_vel (rad/s): {joint_vel}")

        bot.torso.set_joint_pos_vel(
            joint_pos=target_angles,
            joint_vel=joint_vel,
            wait_time=5.0,
            exit_on_reach=False,
        )

if __name__ == "__main__":
    tyro.cli(main)

# 5, 5, -5
# 20, 20, -5
# 70, 120, 10
