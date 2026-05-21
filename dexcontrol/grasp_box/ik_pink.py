#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Pink IK solver for dual-arm manipulation with torso control.

This script uses the Pink IK library to solve inverse kinematics for the Vega robot.
It reads target poses from a trajectory file and computes joint configurations for
both arms and the torso, then executes them on the real robot.
"""

import os
import sys
import time
import copy
from dataclasses import dataclass
from typing import Any, Annotated
from read_force import get_force,tare_force

import numpy as np
import pinocchio as pin
import qpsolvers
import tyro
from loguru import logger

import pink
from pink import solve_ik
from pink.tasks import RelativeFrameTask, PostureTask, JointCouplingTask

from dexcontrol.robot import Robot

# Import box detection utilities
# from box_detection_utils import initialize_detection_model, get_box_coordinate

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dexcontrol", "examples", "custom_examples"))
from utils import align_head_to_forward, wait_for_space, parse_trajectory_file, rpy_to_rotation_matrix
import config


def solve_ik_for_waypoint(
    configuration: pink.Configuration,
    waypoint: dict[str, Any],
    left_ee_task: RelativeFrameTask,
    right_ee_task: RelativeFrameTask,
    posture_task: PostureTask,
    solver: str,
    dt: float = config.IK_DT,
    max_iters: int = config.IK_MAX_ITERS,
    use_shoulder_bias: bool = False,
) -> tuple[np.ndarray, bool]:
    """Solve IK for a single waypoint.

    Targets are expressed in the world frame but converted to chest-relative
    (arm_center) so that torso joints are outside the IK Jacobian and cannot
    drift during arm solving.

    Args:
        configuration: Current robot configuration.
        waypoint: Target waypoint with poses.
        left_ee_task: Left end-effector relative frame task.
        right_ee_task: Right end-effector relative frame task.
        posture_task: Posture task for regularization.
        solver: QP solver name.
        dt: Time step for IK iterations.
        max_iters: Maximum IK iterations.
        use_shoulder_bias: If True, bias L_arm_j2/R_arm_j2 toward their
            position limits to encourage maximum shoulder pitch extension.

    Returns:
        Tuple of (joint_configuration, success).
    """
    model = configuration.model
    tasks = []

    # Trajectory file coordinates are already chest-relative (arm_center frame).
    # RelativeFrameTask expects its target in the root (arm_center) frame directly.

    # Set left end-effector target (chest-relative)
    left_pos = waypoint['left_pose']['position'].copy()
    left_rot = rpy_to_rotation_matrix(*waypoint['left_pose']['rpy'])
    left_ee_task.set_target(pin.SE3(left_rot, left_pos))
    tasks.append(left_ee_task)

    # Set right end-effector target (chest-relative)
    right_pos = waypoint['right_pose']['position'].copy()
    right_rot = rpy_to_rotation_matrix(*waypoint['right_pose']['rpy'])
    right_ee_task.set_target(pin.SE3(right_rot, right_pos))
    tasks.append(right_ee_task)

    # Posture task: soft regularization toward current configuration.
    # When use_shoulder_bias=True, bias j2 (shoulder pitch) toward joint limits
    # to encourage maximum shoulder extension (e.g. for VegaTask4).
    if use_shoulder_bias:
        posture_target = configuration.q.copy()

        # Fixed shoulder bias targets for j1/j2/j3 (symmetric mirror):
        #   Left :  +1.92°, +51.69°, +84.92°
        #   Right:  -1.92°, -51.69°, -84.92°
        shoulder_bias_left  = np.deg2rad([ 1.92,  51.69,  84.92])
        shoulder_bias_right = np.deg2rad([-1.92, -51.69, -84.92])

        for j in range(3):
            left_idx  = model.getJointId(f"L_arm_j{j+1}")
            right_idx = model.getJointId(f"R_arm_j{j+1}")
            posture_target[model.idx_qs[left_idx]]  = shoulder_bias_left[j]
            posture_target[model.idx_qs[right_idx]] = shoulder_bias_right[j]
            logger.debug(
                f"Biasing L_arm_j{j+1} toward {shoulder_bias_left[j]:.3f} rad "
                f"({np.rad2deg(shoulder_bias_left[j]):.2f}°)"
            )
            logger.debug(
                f"Biasing R_arm_j{j+1} toward {shoulder_bias_right[j]:.3f} rad "
                f"({np.rad2deg(shoulder_bias_right[j]):.2f}°)"
            )

        posture_task.set_target(posture_target)
    else:
        # No shoulder bias: regularize toward the current configuration so the
        # posture task doesn't pull joints toward a stale or biased target.
        posture_task.set_target(configuration.q.copy())
    tasks.append(posture_task)

    for iteration in range(max_iters):
        # Compute velocity from tasks
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
        )
        
        # Integrate velocity
        q_next = pin.integrate(configuration.model, configuration.q, velocity * dt)
        configuration.update(q_next)
        
        left_error = np.linalg.norm(left_ee_task.compute_error(configuration))
        right_error = np.linalg.norm(right_ee_task.compute_error(configuration))
        total_error = left_error + right_error

        if total_error < config.IK_CONVERGENCE_THRESHOLD:
            logger.debug(f"IK converged in {iteration} iterations (error: {total_error:.6f})")
            return configuration.q.copy(), True
    
    left_error = np.linalg.norm(left_ee_task.compute_error(configuration))
    right_error = np.linalg.norm(right_ee_task.compute_error(configuration))
    logger.warning(
        f"IK did not fully converge after {max_iters} iterations. "
        f"Left error: {left_error:.4f}, Right error: {right_error:.4f}"
    )
    return configuration.q.copy(), False


# ---------------------------------------------------------------------------
# IK context — built once at startup and reused across trajectory calls
# ---------------------------------------------------------------------------

@dataclass
class IKContext:
    """Pre-built resources created once at startup to avoid per-call overhead."""
    robot_pin: Any
    configuration: Any          # pink.Configuration
    left_ee_task: Any           # RelativeFrameTask
    right_ee_task: Any          # RelativeFrameTask
    posture_task: Any           # PostureTask
    solver: str
    model: Any                  # pin.Model
    left_arm_indices: list
    right_arm_indices: list
    torso_indices: list
    bot: Any                    # Robot
    control_dt: float
    stop_event: Any             # threading.Event — set to abort trajectory execution


def build_ik_context(skip_confirmation: bool = False) -> IKContext:
    """Load URDF, build IK tasks, select QP solver, and connect to robot.

    This is the expensive one-time setup. Call once at startup and pass the
    returned IKContext to run_trajectory() for each execution.

    Args:
        skip_confirmation: If True, skip the interactive safety confirmation prompt.

    Returns:
        A fully initialised IKContext ready for trajectory execution.
    """
    # ── Load URDF ────────────────────────────────────────────────────────────
    logger.info("Loading robot model...")
    urdf_path = config.URDF_PATH
    vega_1p_dir = os.path.dirname(urdf_path)
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf_path))))

    robot_pin = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=[vega_1p_dir, package_dir],
        root_joint=None,
    )
    import threading as _threading
    logger.info(f"Loaded robot with {robot_pin.model.nq} DOFs")

    # ── Pink configuration & tasks ───────────────────────────────────────────
    q_init = pin.neutral(robot_pin.model)
    configuration = pink.Configuration(robot_pin.model, robot_pin.data, q_init)

    left_ee_task = RelativeFrameTask(
        "L_gripper_base",
        root="arm_center",
        position_cost=config.LEFT_EE_POSITION_COST,
        orientation_cost=config.LEFT_EE_ORIENTATION_COST,
    )
    right_ee_task = RelativeFrameTask(
        "R_gripper_base",
        root="arm_center",
        position_cost=config.RIGHT_EE_POSITION_COST,
        orientation_cost=config.RIGHT_EE_ORIENTATION_COST,
    )
    posture_task = PostureTask(cost=config.POSTURE_COST)
    posture_task.set_target(q_init)

    # ── QP solver ────────────────────────────────────────────────────────────
    solver = qpsolvers.available_solvers[0]
    if config.PREFERRED_QP_SOLVER in qpsolvers.available_solvers:
        solver = config.PREFERRED_QP_SOLVER
    logger.info(f"Using QP solver: {solver}")

    # ── Precompute joint indices ──────────────────────────────────────────────
    model = robot_pin.model
    left_arm_indices  = [model.getJointId(f"L_arm_j{j+1}") for j in range(7)]
    right_arm_indices = [model.getJointId(f"R_arm_j{j+1}") for j in range(7)]
    torso_indices     = [model.getJointId(f"torso_j{j+1}") for j in range(3)]

    # ── Safety confirmation ───────────────────────────────────────────────────
    logger.warning("=" * 60)
    logger.warning("WARNING: About to connect to real robot!")
    logger.warning("Make sure:")
    logger.warning("  1. The robot has enough space to move")
    logger.warning("  2. You are ready to press the e-stop if needed")
    logger.warning("=" * 60)

    if skip_confirmation:
        logger.info("Confirmation skipped (programmatic invocation)")
    else:
        response = input("Continue with execution? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Execution cancelled")
            raise SystemExit(0)

    # ── Connect to robot & camera ─────────────────────────────────────────────
    logger.info("Connecting to robot...")
    from dexcontrol.core.config import get_robot_config
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"

    bot = Robot(configs=configs)

    logger.info("Waiting for camera streams to become active...")
    if bot.sensors.head_camera.wait_for_active(timeout=config.CAMERA_TIMEOUT):
        logger.info("Camera streams active!")
    else:
        logger.warning("Warning: Some camera streams may not be active")

    # Robot.__init__ resets the head to the predefined home pose (≈90°).
    # Immediately re-align so the head looks forward at the correct angle.
    align_head_to_forward(bot)
    logger.info("IK context ready — URDF loaded, tasks built, robot connected.")
    return IKContext(
        robot_pin=robot_pin,
        configuration=configuration,
        left_ee_task=left_ee_task,
        right_ee_task=right_ee_task,
        posture_task=posture_task,
        solver=solver,
        model=model,
        left_arm_indices=left_arm_indices,
        right_arm_indices=right_arm_indices,
        torso_indices=torso_indices,
        bot=bot,
        control_dt=config.CONTROL_DT,
        stop_event=_threading.Event(),
    )


def run_trajectory(
    ctx: IKContext,
    trajectory_file: str,
    box_coord: dict | None = None,
    step_by_step: bool = False,
    use_shoulder_bias: bool = False,
) -> bool:
    """Sync robot state, parse trajectory file, and execute all waypoints.

    Uses the pre-built IKContext so URDF loading, task setup, and robot
    connection are NOT repeated between calls.

    Args:
        ctx: Pre-built context from build_ik_context().
        trajectory_file: Path to trajectory file.
        box_coord: Optional pre-detected box position dict with x, y, z keys.
        step_by_step: If True, wait for space bar press before each waypoint.
        use_shoulder_bias: If True, bias j2 shoulder pitch toward joint limits
            in every IK solve call (e.g. for VegaTask4).

    Returns:
        True on success, False on failure.
    """
    # ── Parse trajectory ──────────────────────────────────────────────────────
    waypoints = parse_trajectory_file(trajectory_file)
    if not waypoints:
        logger.error("No waypoints found in trajectory file")
        return False

    # ── Sync IK configuration from live robot state ───────────────────────────
    # Done before every execution so IK starts from the actual current pose,
    # not from where the previous trajectory ended in simulation.
    current_left_arm_q  = ctx.bot.left_arm.get_joint_pos().astype(float)
    current_right_arm_q = ctx.bot.right_arm.get_joint_pos().astype(float)
    current_torso_q     = ctx.bot.torso.get_joint_pos().astype(float)

    q_live = ctx.configuration.q.copy()
    for j, idx in enumerate(ctx.left_arm_indices):
        q_live[ctx.model.idx_qs[idx]] = current_left_arm_q[j]
    for j, idx in enumerate(ctx.right_arm_indices):
        q_live[ctx.model.idx_qs[idx]] = current_right_arm_q[j]
    for j, idx in enumerate(ctx.torso_indices):
        q_live[ctx.model.idx_qs[idx]] = current_torso_q[j]
    q_live = np.clip(q_live, ctx.model.lowerPositionLimit, ctx.model.upperPositionLimit)
    ctx.configuration.update(q_live)
    logger.info("IK configuration synced from live robot joint state")

    if step_by_step:
        logger.info("Step-by-step mode: Press SPACE to advance between waypoints")

    try:
        logger.info(f"Executing {len(waypoints)} waypoints with real-time IK solving...")
        for i, waypoint in enumerate(waypoints):
            # Re-sync full robot state into the IK configuration at the start of every
            # waypoint so that FK (used for REL/BOX coordinate conversion) always
            # reflects the actual robot pose, not a drifted IK simulation state.
            q_live = ctx.configuration.q.copy()
            for j, idx in enumerate(ctx.left_arm_indices):
                q_live[ctx.model.idx_qs[idx]] = current_left_arm_q[j]
            for j, idx in enumerate(ctx.right_arm_indices):
                q_live[ctx.model.idx_qs[idx]] = current_right_arm_q[j]
            for j, idx in enumerate(ctx.torso_indices):
                q_live[ctx.model.idx_qs[idx]] = current_torso_q[j]
            q_live = np.clip(q_live, ctx.model.lowerPositionLimit, ctx.model.upperPositionLimit)
            ctx.configuration.update(q_live)
            if ctx.stop_event.is_set():
                logger.warning("Trajectory aborted by stop_event")
                return False
            step_type = waypoint['type'].upper()
            logger.info(f"Waypoint {i+1}/{len(waypoints)} [{step_type}]:")
            if step_by_step:
                wait_for_space()

            # ── TORSO step ────────────────────────────────────────────────────
            if waypoint['type'] == 'torso':
                target_angles = np.deg2rad(waypoint['joint_angles_deg'])
                current_angles = ctx.bot.torso.get_joint_pos().astype(float)
                error = target_angles - current_angles
                joint_vel = np.clip(np.abs(error) * 0.30, 0.002, 2.0)

                # duration=0.0 means no timing constraint; fall back to 8s so the
                # _wait_for_position loop actually runs and exit_on_reach works.
                wait_secs = waypoint['duration']
                ctx.bot.torso.set_joint_pos_vel(
                    joint_pos=target_angles,
                    joint_vel=joint_vel,
                    wait_time=wait_secs,
                    exit_on_reach=True,
                )
                # Send zero-velocity hold so the firmware engages its lock.
                ctx.bot.torso.stop()
                current_torso_q = ctx.bot.torso.get_joint_pos().astype(float)
                align_head_to_forward(ctx.bot)

                logger.info(f"  Torso joints (deg): {waypoint['joint_angles_deg']}")
                logger.info(f"  Torso velocity (rad/s): {joint_vel}")
                continue

            # ── JOINT step ───────────────────────────────────────────────────── 
            if waypoint['type'] == 'joint':
                target_left_arm_q  = waypoint['left_arm_joints'].copy()
                target_right_arm_q = waypoint['right_arm_joints'].copy()
                # '-' in trajectory file → nan → keep current joint angle
                nan_left  = np.isnan(target_left_arm_q)
                nan_right = np.isnan(target_right_arm_q)
                if np.any(nan_left):
                    target_left_arm_q[nan_left]   = current_left_arm_q[nan_left]
                if np.any(nan_right):
                    target_right_arm_q[nan_right] = current_right_arm_q[nan_right]
                logger.info(f"  Left joints:  {target_left_arm_q}")
                logger.info(f"  Right joints: {target_right_arm_q}")
                n_steps = max(1, int(waypoint['duration'] / ctx.control_dt))
                for step in range(n_steps):
                    if ctx.stop_event.is_set():
                        logger.warning("JOINT motion aborted by stop_event")
                        return False
                    _t = (step + 1) / n_steps
                    alpha = _t * _t * (3 - 2 * _t)  # smoothstep: slow start, fast middle, slow end
                    ctx.bot.set_joint_pos({
                        'left_arm':  current_left_arm_q  + alpha * (target_left_arm_q  - current_left_arm_q),
                        'right_arm': current_right_arm_q + alpha * (target_right_arm_q - current_right_arm_q),
                    })
                    time.sleep(ctx.control_dt)
                current_left_arm_q  = target_left_arm_q
                current_right_arm_q = target_right_arm_q
                q_sync = ctx.configuration.q.copy()
                for j, idx in enumerate(ctx.left_arm_indices):
                    q_sync[ctx.model.idx_qs[idx]] = current_left_arm_q[j]
                for j, idx in enumerate(ctx.right_arm_indices):
                    q_sync[ctx.model.idx_qs[idx]] = current_right_arm_q[j]
                q_sync = np.clip(q_sync, ctx.model.lowerPositionLimit, ctx.model.upperPositionLimit)
                ctx.configuration.update(q_sync)
                align_head_to_forward(ctx.bot)
                continue

            # ── GRAB step ─────────────────────────────────────────────────────
            # Incrementally moves both arms inward along Y (left: -Y, right: +Y)
            # until either hand reports force ≥ threshold, then stops.
            # If the hand separation would drop below GRAB_MIN_HAND_SEPARATION_M,
            # the last step is reversed and the loop exits.
            if waypoint['type'] == 'grab':
                step_size   = waypoint['step_size']
                max_steps   = waypoint['max_steps']
                force_threshold   = config.GRAB_FORCE_THRESHOLD_N
                min_separation    = config.GRAB_MIN_HAND_SEPARATION_M
                tare_force("both", ctx.bot)

                logger.info(
                    f"  GRAB: step_size={step_size} m, max_steps={max_steps}, "
                    f"force_threshold={force_threshold} N, min_separation={min_separation} m"
                )

                # FK to find current EE positions in arm_center frame
                pin.framesForwardKinematics(ctx.robot_pin.model, ctx.robot_pin.data, ctx.configuration.q)
                arm_center_id   = ctx.robot_pin.model.getFrameId("arm_center")
                left_ee_id      = ctx.robot_pin.model.getFrameId("L_gripper_base")
                right_ee_id     = ctx.robot_pin.model.getFrameId("R_gripper_base")
                T_ac            = ctx.robot_pin.data.oMf[arm_center_id]
                left_pos_ac     = (T_ac.inverse() * ctx.robot_pin.data.oMf[left_ee_id]).translation.copy()
                right_pos_ac    = (T_ac.inverse() * ctx.robot_pin.data.oMf[right_ee_id]).translation.copy()
                left_rpy_ac     = pin.rpy.matrixToRpy((T_ac.inverse() * ctx.robot_pin.data.oMf[left_ee_id]).rotation)
                right_rpy_ac    = pin.rpy.matrixToRpy((T_ac.inverse() * ctx.robot_pin.data.oMf[right_ee_id]).rotation)

                grabbed = False
                for grab_step in range(max_steps):
                    if ctx.stop_event.is_set():
                        logger.warning("GRAB aborted by stop_event")
                        return False
                    # Compute new Y targets (left moves −Y, right moves +Y)
                    new_left_y  = left_pos_ac[1]  - step_size
                    new_right_y = right_pos_ac[1] + step_size
                    separation  = new_right_y - new_left_y

                    if np.abs(separation) < min_separation:
                        logger.warning(
                            f"  GRAB: hand separation {separation:.3f} m would drop below "
                            f"minimum {min_separation} m — stopping without reversing"
                        )
                        break

                    new_left_pos  = left_pos_ac.copy();  new_left_pos[1]  = new_left_y
                    new_right_pos = right_pos_ac.copy(); new_right_pos[1] = new_right_y

                    waypoint_grab = {
                        'type': 'ik',
                        'left_pose':  {'position': new_left_pos,  'rpy': left_rpy_ac,  'use_box': False, 'use_rel': False},
                        'right_pose': {'position': new_right_pos, 'rpy': right_rpy_ac, 'use_box': False, 'use_rel': False},
                        'duration': ctx.control_dt * 4,
                    }
                    q_solution, _ = solve_ik_for_waypoint(
                        configuration=ctx.configuration,
                        waypoint=waypoint_grab,
                        left_ee_task=ctx.left_ee_task,
                        right_ee_task=ctx.right_ee_task,
                        posture_task=ctx.posture_task,
                        solver=ctx.solver,
                        use_shoulder_bias=use_shoulder_bias,
                    )
                    target_left_arm_q  = np.array([q_solution[ctx.model.idx_qs[idx]] for idx in ctx.left_arm_indices])
                    target_right_arm_q = np.array([q_solution[ctx.model.idx_qs[idx]] for idx in ctx.right_arm_indices])
                    ctx.bot.set_joint_pos({'left_arm': target_left_arm_q, 'right_arm': target_right_arm_q})
                    time.sleep(ctx.control_dt * 4)

                    current_left_arm_q  = target_left_arm_q
                    current_right_arm_q = target_right_arm_q
                    left_pos_ac  = new_left_pos
                    right_pos_ac = new_right_pos

                    # Check force on both hands
                    force_left  = get_force("left",  ctx.bot)
                    force_right = get_force("right", ctx.bot)
                    fl_str = f"{force_left:.2f}" if force_left is not None else "N/A"
                    fr_str = f"{force_right:.2f}" if force_right is not None else "N/A"
                    logger.info(
                        f"  GRAB step {grab_step+1}: sep={separation:.3f} m  "
                        f"F_left={fl_str} N  F_right={fr_str} N"
                    )
                    # if (force_left  is not None and force_left  >= force_threshold) and \
                    #    (force_right is not None and force_right >= force_threshold):
                    if force_left  is not None and force_right  is not None:
                        if force_left + force_right >= force_threshold:
                            logger.info(f"  GRAB: force threshold reached — grasped!")
                            grabbed = True
                            break

                if not grabbed:
                    logger.warning("  GRAB: finished without reaching force threshold")
                else:
                    logger.info("  GRAB: box secured")

                # Sync IK config from actual joint positions
                q_sync = ctx.configuration.q.copy()
                for j, idx in enumerate(ctx.left_arm_indices):
                    q_sync[ctx.model.idx_qs[idx]] = current_left_arm_q[j]
                for j, idx in enumerate(ctx.right_arm_indices):
                    q_sync[ctx.model.idx_qs[idx]] = current_right_arm_q[j]
                q_sync = np.clip(q_sync, ctx.model.lowerPositionLimit, ctx.model.upperPositionLimit)
                ctx.configuration.update(q_sync)
                align_head_to_forward(ctx.bot)
                continue

            # ── IK step ───────────────────────────────────────────────────────
            logger.info(f"  Left:  pos={waypoint['left_pose']['position']}, rpy={waypoint['left_pose']['rpy']}")
            logger.info(f"  Right: pos={waypoint['right_pose']['position']}, rpy={waypoint['right_pose']['rpy']}")

            waypoint_to_solve = copy.deepcopy(waypoint)

            # Handle REL (relative) poses
            if waypoint['left_pose']['use_rel'] or waypoint['right_pose']['use_rel']:
                pin.framesForwardKinematics(ctx.robot_pin.model, ctx.robot_pin.data, ctx.configuration.q)
                arm_center_id = ctx.robot_pin.model.getFrameId("arm_center")
                T_world_arm_center = ctx.robot_pin.data.oMf[arm_center_id]

                if waypoint['left_pose']['use_rel']:
                    left_gripper_id = ctx.robot_pin.model.getFrameId("L_gripper_base")
                    T_armcenter_left = T_world_arm_center.inverse() * ctx.robot_pin.data.oMf[left_gripper_id]
                    offset = waypoint['left_pose']['position']
                    current_pos = T_armcenter_left.translation
                    waypoint_to_solve['left_pose']['position'] = current_pos + offset
                    logger.info(f"  Left arm: REL pos offset ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}) from current ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
                    rpy_offset = waypoint['left_pose']['rpy']
                    current_rpy = pin.rpy.matrixToRpy(T_armcenter_left.rotation)
                    waypoint_to_solve['left_pose']['rpy'] = current_rpy + rpy_offset
                    logger.info(f"  Left arm: REL rpy offset ({rpy_offset[0]:.3f}, {rpy_offset[1]:.3f}, {rpy_offset[2]:.3f}) from current ({current_rpy[0]:.3f}, {current_rpy[1]:.3f}, {current_rpy[2]:.3f})")

                if waypoint['right_pose']['use_rel']:
                    right_gripper_id = ctx.robot_pin.model.getFrameId("R_gripper_base")
                    T_armcenter_right = T_world_arm_center.inverse() * ctx.robot_pin.data.oMf[right_gripper_id]
                    offset = waypoint['right_pose']['position']
                    current_pos = T_armcenter_right.translation
                    waypoint_to_solve['right_pose']['position'] = current_pos + offset
                    logger.info(f"  Right arm: REL pos offset ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}) from current ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
                    rpy_offset = waypoint['right_pose']['rpy']
                    current_rpy = pin.rpy.matrixToRpy(T_armcenter_right.rotation)
                    waypoint_to_solve['right_pose']['rpy'] = current_rpy + rpy_offset
                    logger.info(f"  Right arm: REL rpy offset ({rpy_offset[0]:.3f}, {rpy_offset[1]:.3f}, {rpy_offset[2]:.3f}) from current ({current_rpy[0]:.3f}, {current_rpy[1]:.3f}, {current_rpy[2]:.3f})")

            # Handle BOX poses
            detected_box = None
            if waypoint['left_pose']['use_box'] or waypoint['right_pose']['use_box']:
                if box_coord is not None:
                    # Convert box_coord from world/base-link frame to arm_center frame.
                    # Trajectory offsets are arm_center-relative, so the box position must
                    # be expressed in the same frame before the offset is added.
                    pin.framesForwardKinematics(ctx.robot_pin.model, ctx.robot_pin.data, ctx.configuration.q)
                    _arm_center_id = ctx.robot_pin.model.getFrameId("arm_center")
                    T_world_arm_center = ctx.robot_pin.data.oMf[_arm_center_id]
                    box_world = np.array([box_coord['x'], box_coord['y'], box_coord['z']])
                    box_arm_center = T_world_arm_center.inverse().act(box_world)
                    detected_box = {'x': box_arm_center[0], 'y': box_arm_center[1], 'z': box_arm_center[2]}
                    logger.info(
                        f"  Box coord (base link):  x={box_coord['x']:.3f}, "
                        f"y={box_coord['y']:.3f}, z={box_coord['z']:.3f}"
                    )
                    logger.info(
                        f"  Box coord (arm_center): x={detected_box['x']:.3f}, "
                        f"y={detected_box['y']:.3f}, z={detected_box['z']:.3f}"
                    )
                else:
                    logger.error(f"Trajectory requires box coordinates but none were provided.")
                    return False

            if waypoint['left_pose']['use_box']:
                if detected_box is not None:
                    offset = waypoint['left_pose']['position']
                    waypoint_to_solve['left_pose']['position'] = np.array([
                        detected_box['x'] + offset[0],
                        detected_box['y'] + offset[1],
                        offset[2],  # Z from trajectory file, not from box coord
                    ])
                    logger.info(f"  Left arm: box xy + offset ({offset[0]:.3f}, {offset[1]:.3f}), z={offset[2]:.3f} (from file)")
                else:
                    logger.warning("  Left arm: Cannot apply box offset - using absolute position from file")

            if waypoint['right_pose']['use_box']:
                if detected_box is not None:
                    offset = waypoint['right_pose']['position']
                    waypoint_to_solve['right_pose']['position'] = np.array([
                        detected_box['x'] + offset[0],
                        detected_box['y'] + offset[1],
                        offset[2],  # Z from trajectory file, not from box coord
                    ])
                    logger.info(f"  Right arm: box xy + offset ({offset[0]:.3f}, {offset[1]:.3f}), z={offset[2]:.3f} (from file)")
                else:
                    logger.warning("  Right arm: Cannot apply box offset - using absolute position from file")

            # Handle CURRENT ('-') field substitution:
            # Any nan in position/rpy means "keep current EE value for that field".
            # This runs after REL/BOX so nan values produced by those (e.g. nan offset)
            # are also resolved to the current EE state.
            if waypoint['left_pose'].get('use_cur') or waypoint['right_pose'].get('use_cur'):
                pin.framesForwardKinematics(ctx.robot_pin.model, ctx.robot_pin.data, ctx.configuration.q)
                _ac_id = ctx.robot_pin.model.getFrameId("arm_center")
                _T_ac = ctx.robot_pin.data.oMf[_ac_id]

                if waypoint['left_pose'].get('use_cur'):
                    _lf_id = ctx.robot_pin.model.getFrameId("L_gripper_base")
                    _T_ac_left = _T_ac.inverse() * ctx.robot_pin.data.oMf[_lf_id]
                    _cur_lpos = _T_ac_left.translation
                    _cur_lrpy = pin.rpy.matrixToRpy(_T_ac_left.rotation)
                    _lpos = waypoint_to_solve['left_pose']['position'].copy()
                    _lrpy = waypoint_to_solve['left_pose']['rpy'].copy()
                    _lpos[np.isnan(_lpos)] = _cur_lpos[np.isnan(_lpos)]
                    _lrpy[np.isnan(_lrpy)] = _cur_lrpy[np.isnan(_lrpy)]
                    waypoint_to_solve['left_pose']['position'] = _lpos
                    waypoint_to_solve['left_pose']['rpy'] = _lrpy
                    logger.info(f"  Left arm: CUR substitution → pos={_lpos}, rpy={_lrpy}")

                if waypoint['right_pose'].get('use_cur'):
                    _rf_id = ctx.robot_pin.model.getFrameId("R_gripper_base")
                    _T_ac_right = _T_ac.inverse() * ctx.robot_pin.data.oMf[_rf_id]
                    _cur_rpos = _T_ac_right.translation
                    _cur_rrpy = pin.rpy.matrixToRpy(_T_ac_right.rotation)
                    _rpos = waypoint_to_solve['right_pose']['position'].copy()
                    _rrpy = waypoint_to_solve['right_pose']['rpy'].copy()
                    _rpos[np.isnan(_rpos)] = _cur_rpos[np.isnan(_rpos)]
                    _rrpy[np.isnan(_rrpy)] = _cur_rrpy[np.isnan(_rrpy)]
                    waypoint_to_solve['right_pose']['position'] = _rpos
                    waypoint_to_solve['right_pose']['rpy'] = _rrpy
                    logger.info(f"  Right arm: CUR substitution → pos={_rpos}, rpy={_rrpy}")

            logger.info("  Solving IK...")
            q_solution, success = solve_ik_for_waypoint(
                configuration=ctx.configuration,
                waypoint=waypoint_to_solve,
                left_ee_task=ctx.left_ee_task,
                right_ee_task=ctx.right_ee_task,
                posture_task=ctx.posture_task,
                solver=ctx.solver,
                use_shoulder_bias=use_shoulder_bias,
            )

            target_left_arm_q  = np.array([q_solution[ctx.model.idx_qs[idx]] for idx in ctx.left_arm_indices])
            target_right_arm_q = np.array([q_solution[ctx.model.idx_qs[idx]] for idx in ctx.right_arm_indices])

            logger.info(f"  Left arm joints:  {target_left_arm_q}")
            logger.info(f"  Right arm joints: {target_right_arm_q}")
            logger.info(f"  └─ L_arm_j2 (shoulder): {target_left_arm_q[1]:+.4f} rad ({np.rad2deg(target_left_arm_q[1]):+.2f}°)")
            logger.info(f"  └─ R_arm_j2 (shoulder): {target_right_arm_q[1]:+.4f} rad ({np.rad2deg(target_right_arm_q[1]):+.2f}°)")
            logger.info(f"  Status: {'✓ Converged' if success else '⚠ Did not fully converge'}")

            logger.info(f"  Executing motion (duration: {waypoint['duration']:.1f}s)...")
            n_steps = max(1, int(waypoint['duration'] / ctx.control_dt))
            for step in range(n_steps):
                if ctx.stop_event.is_set():
                    logger.warning("IK motion aborted by stop_event")
                    return False
                _force_hit = False
                for _side in ("left", "right"):
                    _f = get_force(_side, ctx.bot)
                    if _f is not None and _f >= config.IK_FORCE_PAUSE_THRESHOLD_N:
                        logger.warning(
                            f"  Force threshold reached on {_side} hand ({_f:.2f} N) "
                            f"at step {step+1}/{n_steps} — motion paused."
                        )
                        _force_hit = True
                        wait_for_space()

                _t = (step + 1) / n_steps
                alpha = _t * _t * (3 - 2 * _t)  # smoothstep: slow start, fast middle, slow end
                ctx.bot.set_joint_pos({
                    'left_arm':  current_left_arm_q  + alpha * (target_left_arm_q  - current_left_arm_q),
                    'right_arm': current_right_arm_q + alpha * (target_right_arm_q - current_right_arm_q),
                })
                time.sleep(ctx.control_dt)
            current_left_arm_q  = target_left_arm_q
            current_right_arm_q = target_right_arm_q
            align_head_to_forward(ctx.bot)

        logger.info("Trajectory execution completed!")
        return True

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return False


def main(
    trajectory_file: str = config.DEFAULT_TRAJECTORY_FILE,
    step_by_step: bool = False,
    skip_confirmation: bool = False,
    use_shoulder_bias: bool = False,
    box_coord: Annotated[dict | None, tyro.conf.Suppress] = None,
) -> bool:
    """Run IK solver and execute on robot in real-time.

    Args:
        trajectory_file: Path to trajectory file with target poses.
        step_by_step: If True, wait for space bar press before each waypoint.
        skip_confirmation: If True, skip interactive confirmation prompt.
        use_shoulder_bias: If True, bias j2 shoulder pitch toward joint limits
            in every IK solve to encourage maximum shoulder extension.
        box_coord: Pre-detected box position (programmatic use only).
    """
    ctx = build_ik_context(skip_confirmation=skip_confirmation)

    try:
        return run_trajectory(
            ctx,
            trajectory_file=trajectory_file,
            box_coord=box_coord,
            step_by_step=step_by_step,
            use_shoulder_bias=use_shoulder_bias,
        )
    finally:
        logger.info("Shutting down robot connection")
        ctx.bot.shutdown()


if __name__ == "__main__":
    success = tyro.cli(main)
    sys.exit(0 if success else 1)

