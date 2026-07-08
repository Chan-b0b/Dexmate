"""Right-arm Robotiq gripper for ik_demo (built on arm.ArmMover).

Used for the barcode divert: a matched battery held by the suction cup at
transport is gripped by the right-arm gripper (side approach), suction releases,
and the gripper carries it to a taught lower-right joint pose and opens.

Primitives only (grip_at, place_joints); the two-arm handoff choreography lives
in sequence.py, which coordinates the suction arm + this gripper.

Gripper poses (HANDOFF_GRIP_OFFSET, PLACE_LOWER_RIGHT_JOINTS) are from the old
demo / unset — RE-TEACH on the robot before trusting the divert.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
    from .arm import ArmMover
    from .drivers.robotiq import RobotiqGripper
except ImportError:  # allow `python gripper.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover
    from drivers.robotiq import RobotiqGripper


class GripperMover(ArmMover):
    """Right-arm mover that drives the Robotiq gripper."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
        self.gripper = RobotiqGripper(robot, side="right")

    def initialize(self) -> bool:
        """Reset + activate + open the gripper. Returns True on success."""
        if not self.gripper.initialize():
            logger.warning("[gripper] initialization failed — gripper disabled")
            return False
        self.gripper.open()
        logger.info("[gripper] initialized (reset + activated + opened)")
        return True

    def grip_at(self, pos, rpy=None) -> bool:
        """Side-approach grasp at base-frame *pos*; True if an object was gripped.

        Opens, backs off along the approach axis (gripper tool +z) by
        GRIPPER_PREGRASP_STANDOFF_M, moves there, moves straight in to *pos*,
        and closes.
        """
        grasp_pos = np.asarray(pos, dtype=float)
        rpy = np.asarray(cfg.GRIPPER_GRASP_RPY if rpy is None else rpy, dtype=float)
        self.gripper.open()
        approach = Rotation.from_euler("xyz", rpy).as_matrix()[:, 2]  # tool +z in base_link
        pre = grasp_pos - cfg.GRIPPER_PREGRASP_STANDOFF_M * approach
        logger.info("[gripper] pre-grasp standoff -> grasp")
        if self.move_ee(pre, rpy) is None or self.move_ee(grasp_pos, rpy) is None:
            logger.error("[gripper] grasp pose unreachable")
            return False
        self.gripper.close()
        gripped = self.gripper.is_object_grasped()
        logger.info("[gripper] grip_at -> {}", "GRIPPED" if gripped else "no object")
        return gripped

    def place_joints(self, joints=None) -> None:
        """Move the right arm to the taught lower-right joint pose and open."""
        joints = cfg.PLACE_LOWER_RIGHT_JOINTS if joints is None else joints
        if joints is None:
            raise ValueError("PLACE_LOWER_RIGHT_JOINTS not set — teach it first")
        logger.info("[gripper] place at taught lower-right joint pose")
        self.move_joints(np.asarray(joints, dtype=float))
        self.gripper.open()


# ---------------------------------------------------------------------------
# On-robot smoke test: python gripper.py   (init + open/close, no arm motion)
# ---------------------------------------------------------------------------
def _test_on_robot() -> None:
    from dexcontrol.robot import Robot

    logger.warning("Gripper smoke test: reset/activate/open, then close, then open. No arm motion.")
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return
    with Robot() as bot:
        g = GripperMover(bot)
        if not g.initialize():
            return
        import time
        time.sleep(0.5)
        g.gripper.close(); time.sleep(1.0)
        logger.info("object grasped? {}", g.gripper.is_object_grasped())
        g.gripper.open()


if __name__ == "__main__":
    _test_on_robot()
