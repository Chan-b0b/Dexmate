# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Send the robot to its default pose.

Torso target: [5, 5, 0] degrees.
Arms target:
  config=ik    — Cartesian IK pose (x=0.2, y=±0.2, z=1.0)
  config=joint — Direct joint-space pose (hardcoded joint values)
"""

import os
import sys
import time

import numpy as np
import pinocchio as pin
import qpsolvers
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

import config

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rot.as_matrix()


def solve_ik_for_pose(
    configuration: pink.Configuration,
    left_pos: np.ndarray,
    left_rpy: np.ndarray,
    right_pos: np.ndarray,
    right_rpy: np.ndarray,
    left_ee_task: FrameTask,
    right_ee_task: FrameTask,
    posture_task: PostureTask,
    solver: str,
    dt: float = 0.01,
    max_iters: int = 500,
) -> tuple[np.ndarray, bool]:
    model = configuration.model
    tasks = []

    left_ee_task.set_target(pin.SE3(rpy_to_rotation_matrix(*left_rpy), left_pos))
    right_ee_task.set_target(pin.SE3(rpy_to_rotation_matrix(*right_rpy), right_pos))
    posture_task.set_target(configuration.q.copy())
    tasks.extend([left_ee_task, right_ee_task, posture_task])

    # Freeze torso joints during arm IK solving
    torso_task = PostureTask(cost=5.0)
    torso_task.set_target(configuration.q.copy())
    torso_indices = [model.getJointId(f"torso_j{j+1}") for j in range(3)]
    mask = np.zeros(model.nv)
    for idx in torso_indices:
        mask[model.idx_vs[idx]] = 1.0
    torso_task.gain = mask
    tasks.append(torso_task)

    for iteration in range(max_iters):
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        q_next = pin.integrate(model, configuration.q, velocity * dt)
        configuration.update(q_next)

        left_err = np.linalg.norm(left_ee_task.compute_error(configuration))
        right_err = np.linalg.norm(right_ee_task.compute_error(configuration))
        if left_err + right_err < 1e-4:
            logger.debug(f"IK converged in {iteration} iterations")
            return configuration.q.copy(), True

    logger.warning(
        f"IK did not fully converge after {max_iters} iterations. "
        f"Left error: {left_err:.4f}, Right error: {right_err:.4f}"
    )
    return configuration.q.copy(), False


@supported_models("vega_1", "vega_1p")
def main(joint: bool = False) -> None:
    """Send robot to default pose.
    
    Args:
        joint: If True, send arms directly to joint-space targets (skip IK).
               If False (default), solve IK for Cartesian targets.
    """
    # ── Joint-space target (used when --joint) ─────────────────────────────
    JOINT_LEFT_Q  = np.array([ 1.7834, 0.0022, 0.0322, -1.6711, 0.1628, -1.3960, 0.1480])
    JOINT_RIGHT_Q = np.array([-1.7834, -0.0022, -0.0322, -1.6711, -0.1628, 1.3960, -0.1480])
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around the robot.")
    logger.info(f"Mode: {'joint-space' if joint else 'IK (Cartesian)'}")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    # ── Torso target ──────────────────────────────────────────────────────────
    torso_target = np.deg2rad([10, 30, 20])  # [rad] for torso_j1, torso_j2, torso_j3

    # ── Arm Cartesian targets (base frame, used when config=ik) ───────────────
    left_pos  = np.array([0.25,  0.2, 0.8])
    left_rpy  = np.array([0.0,  0.0, 0.0])
    right_pos = np.array([0.25, -0.2, 0.8])
    right_rpy = np.array([0.0,  0.0, 0.0])
    arm_duration = 5.0  # seconds for arm motion
    control_dt   = 0.02

    # ── Load URDF & set up Pink ───────────────────────────────────────────────
    logger.info("Loading robot model…")
    urdf_path = config.URDF_PATH
    robot_pin = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=[
            os.path.dirname(urdf_path),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf_path)))),
        ],
        root_joint=None,
    )
    logger.info(f"Loaded robot with {robot_pin.model.nq} DOFs")

    q_init = pin.neutral(robot_pin.model)
    configuration = pink.Configuration(robot_pin.model, robot_pin.data, q_init)

    left_ee_task = FrameTask("L_gripper_base", position_cost=2.0, orientation_cost=1.0)
    right_ee_task = FrameTask("R_gripper_base", position_cost=2.0, orientation_cost=1.0)
    posture_task = PostureTask(cost=1e-3)
    posture_task.set_target(q_init)

    solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
    logger.info(f"Using QP solver: {solver}")

    # ── Connect to robot ──────────────────────────────────────────────────────
    logger.info("Connecting to robot…")
    with Robot() as bot:
        # Read live joint state
        current_torso_q  = bot.torso.get_joint_pos().astype(float)
        current_left_q   = bot.left_arm.get_joint_pos().astype(float)
        current_right_q  = bot.right_arm.get_joint_pos().astype(float)

        # Seed IK configuration with live torso angles
        model = robot_pin.model
        torso_indices      = [model.getJointId(f"torso_j{j+1}") for j in range(3)]
        left_arm_indices   = [model.getJointId(f"L_arm_j{j+1}") for j in range(7)]
        right_arm_indices  = [model.getJointId(f"R_arm_j{j+1}") for j in range(7)]

        q_live = configuration.q.copy()
        for j, idx in enumerate(torso_indices):
            q_live[model.idx_qs[idx]] = current_torso_q[j]
        q_live = np.clip(q_live, model.lowerPositionLimit, model.upperPositionLimit)
        configuration.update(q_live)



        # ── Step 2: Solve IK or use joint targets ────────────────────────────
        if joint:
            # Direct joint-space mode: skip IK entirely
            logger.info("Using direct joint-space targets (--joint)…")
            logger.info(f"Target left arm joints:  {JOINT_LEFT_Q}")
            logger.info(f"Target right arm joints: {JOINT_RIGHT_Q}")
            target_left_q  = JOINT_LEFT_Q.copy()
            target_right_q = JOINT_RIGHT_Q.copy()
        else:
            # IK mode: solve Cartesian targets
            logger.info("Solving IK for arm default pose…")
            q_solution, success = solve_ik_for_pose(
                configuration=configuration,
                left_pos=left_pos,
                left_rpy=left_rpy,
                right_pos=right_pos,
                right_rpy=right_rpy,
                left_ee_task=left_ee_task,
                right_ee_task=right_ee_task,
                posture_task=posture_task,
                solver=solver,
            )
            if not success:
                logger.warning("IK did not fully converge — proceeding with best solution")
            target_left_q  = np.array([q_solution[model.idx_qs[idx]] for idx in left_arm_indices])
            target_right_q = np.array([q_solution[model.idx_qs[idx]] for idx in right_arm_indices])
            logger.info(f"Target left arm joints:  {target_left_q}")
            logger.info(f"Target right arm joints: {target_right_q}")

        # ── Step 3: Execute arm motion one arm at a time ──────────────────────
        n_steps = max(1, int(arm_duration / control_dt))

        logger.info(f"Moving left arm over {arm_duration:.1f}s…")
        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            bot.set_joint_pos({'left_arm': current_left_q + alpha * (target_left_q - current_left_q)})
            time.sleep(control_dt)

        logger.info(f"Moving right arm over {arm_duration:.1f}s…")
        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            bot.set_joint_pos({'right_arm': current_right_q + alpha * (target_right_q - current_right_q)})
            time.sleep(control_dt)
       
        # ── Step 1: Move torso ────────────────────────────────────────────────
        logger.info("Moving torso to default pose…")
        error = torso_target - current_torso_q
        kp = 0.35
        joint_vel = np.clip(np.abs(error) * kp, 0.001, 2.0)
        logger.info(f"Torso target (rad): {torso_target}")
        logger.info(f"Torso velocity (rad/s): {joint_vel}")

        bot.torso.set_joint_pos_vel(
            joint_pos=torso_target,
            joint_vel=joint_vel,
            wait_time=15.0,
            exit_on_reach=True,
        )
        current_torso_q = bot.torso.get_joint_pos().astype(float)


        logger.info("Default pose reached.")


if __name__ == "__main__":
    tyro.cli(main)
