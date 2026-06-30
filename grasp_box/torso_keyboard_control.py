# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Keyboard-based torso joint control.

Controls the robot torso joints using keyboard input:
- Q/W: Joint 1 (decrease/increase)
- A/S: Joint 2 (decrease/increase)
- Z/X: Joint 3 (decrease/increase)
- SPACE: Read current state
- ESC or Q (twice): Exit
"""

import numpy as np
from pynput import keyboard
from loguru import logger

from dexcontrol.robot import Robot


class TorsoKeyboardController:
    """Keyboard controller for robot torso."""

    def __init__(self, step_size=0.05, max_velocity=0.5):
        """Initialize the controller.

        Args:
            step_size: Radians to move per keypress
            max_velocity: Maximum joint velocity (rad/s)
        """
        self.step_size = step_size
        self.max_velocity = max_velocity
        self.robot = None
        self.current_angles = None
        self.running = True
        self.joint_bounds = [
            (-np.pi, np.pi),      # Joint 0
            (-np.pi, np.pi),      # Joint 1
            (-np.pi, np.pi),      # Joint 2
        ]

    def start(self):
        """Start the keyboard control session."""
        logger.warning("Warning: Be ready to press e-stop if needed!")
        logger.warning("Please ensure adequate clearance around robot before proceeding.")
        if input("Continue? [y/N]: ").lower() != "y":
            return

        with Robot() as bot:
            self.robot = bot
            self._update_state()
            self._print_instructions()
            self._print_state()

            # Listen for keyboard input
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                listener.join()

    def _update_state(self):
        """Read current torso state."""
        state = self.robot.torso._get_state()
        self.current_angles = np.asarray(state["pos"], dtype=float)

    def _print_instructions(self):
        """Print keyboard control instructions."""
        logger.info("=== Torso Keyboard Control ===")
        logger.info("Q/W: Joint 0 (decrease/increase)")
        logger.info("A/S: Joint 1 (decrease/increase)")
        logger.info("Z/X: Joint 2 (decrease/increase)")
        logger.info("SPACE: Read current state")
        logger.info("ESC: Exit")
        logger.info(f"Step size: {np.rad2deg(self.step_size):.1f}°")

    def _print_state(self):
        """Print current joint angles."""
        logger.info(f"Current angles (deg): {np.rad2deg(self.current_angles)}")
        logger.info(f"Current angles (rad): {self.current_angles}")

    def _move_joint(self, joint_idx, direction):
        """Move a single joint.

        Args:
            joint_idx: Joint index (0, 1, or 2)
            direction: +1 to increase, -1 to decrease
        """
        target_angles = self.current_angles.copy()
        target_angles[joint_idx] += direction * self.step_size

        # Enforce bounds
        target_angles[joint_idx] = np.clip(
            target_angles[joint_idx],
            self.joint_bounds[joint_idx][0],
            self.joint_bounds[joint_idx][1]
        )

        # Calculate velocity proportional to error
        error = target_angles - self.current_angles
        joint_vel = np.clip(np.abs(error) * 2.0, 0.1, self.max_velocity)

        logger.info(f"Moving joint {joint_idx}: {np.rad2deg(self.current_angles[joint_idx]):.1f}° → {np.rad2deg(target_angles[joint_idx]):.1f}°")

        self.robot.torso.set_joint_pos_vel(
            joint_pos=target_angles,
            joint_vel=joint_vel,
            wait_time=1.0,
            exit_on_reach=False,
        )

        self.current_angles = target_angles

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.esc:
                logger.info("Exiting...")
                self.running = False
                return False

            if key == keyboard.Key.space:
                self._update_state()
                self._print_state()
                return

            char = key.char.lower() if hasattr(key, 'char') else None

            if char == 'q':
                self._move_joint(0, -1)
            elif char == 'w':
                self._move_joint(0, +1)
            elif char == 'a':
                self._move_joint(1, -1)
            elif char == 's':
                self._move_joint(1, +1)
            elif char == 'z':
                self._move_joint(2, -1)
            elif char == 'x':
                self._move_joint(2, +1)

        except Exception as e:
            logger.error(f"Error: {e}")


def main() -> None:
    """Main entry point."""
    controller = TorsoKeyboardController(step_size=0.1, max_velocity=0.5)
    controller.start()


if __name__ == "__main__":
    main()
