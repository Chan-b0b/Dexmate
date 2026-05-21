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

"""Interactive pose tuning tool for dual-arm IK.

Reads a target pose from a txt file (left arm + right arm, arm_center frame)
and lets you interactively adjust it with the keyboard.

Controls:
    SPACE         — toggle between default pose and target pose
                    (re-reads file each time you go to target, so edits pick up live)
    m / n         — move both grippers ±x (forward / back)
    Left / Right  — move both grippers ±y (left / right)
    Up / Down     — move both grippers ±z (up / down)
    q / Ctrl+C    — quit

Usage:
    python pose_tune.py
    python pose_tune.py --pose_file trajectories/test.txt
    python pose_tune.py --current   # start from current robot pose, no default step
    python pose_tune.py --step 0.05 # change step size (default 0.02 m)
"""

import os
import sys
import termios
import tty
import time

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import rpy_to_rotation_matrix
import config
from ik_pink import build_ik_context, solve_ik_for_waypoint

# ── Default arm joint-space pose (from default_pose.py --joint) ────────────
DEFAULT_LEFT_Q  = np.array([ 1.7834, 0.0022, 0.0322, -1.6711, 0.1628, -1.3960, 0.1480])
DEFAULT_RIGHT_Q = np.array([-1.7834, -0.0012, -0.0322, -1.6711, -0.1627, 1.3961, -0.1471])

# ── File I/O ────────────────────────────────────────────────────────────────

def read_pose_file(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read left and right arm poses from a txt file.

    Format (comments with # allowed):
        x, y, z, roll, pitch, yaw   <- left arm  (arm_center frame, radians)
        x, y, z, roll, pitch, yaw   <- right arm (arm_center frame, radians)

    Returns:
        (left_pos, left_rpy, right_pos, right_rpy)
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    if len(lines) < 2:
        raise ValueError(f"{filepath}: need at least 2 non-comment lines (left then right arm pose)")

    left  = [float(x.strip()) for x in lines[0].split(',')[:6]]
    right = [float(x.strip()) for x in lines[1].split(',')[:6]]

    return np.array(left[:3]), np.array(left[3:6]), np.array(right[:3]), np.array(right[3:6])


# ── IK ──────────────────────────────────────────────────────────────────────

def solve_ik_for_pose(
    ctx,
    left_pos: np.ndarray,
    left_rpy: np.ndarray,
    right_pos: np.ndarray,
    right_rpy: np.ndarray,
    use_shoulder_bias: bool = False,
) -> tuple[np.ndarray, bool]:
    """Thin wrapper: build a waypoint dict and call solve_ik_for_waypoint."""
    waypoint = {
        'left_pose':  {'position': left_pos.copy(),  'rpy': left_rpy.copy()},
        'right_pose': {'position': right_pos.copy(), 'rpy': right_rpy.copy()},
    }
    return solve_ik_for_waypoint(
        configuration=ctx.configuration,
        waypoint=waypoint,
        left_ee_task=ctx.left_ee_task,
        right_ee_task=ctx.right_ee_task,
        posture_task=ctx.posture_task,
        solver=ctx.solver,
        use_shoulder_bias=use_shoulder_bias,
    )


# ── Motion execution ─────────────────────────────────────────────────────────

def move_to_joints(
    bot,
    current_left_q: np.ndarray,
    current_right_q: np.ndarray,
    target_left_q: np.ndarray,
    target_right_q: np.ndarray,
    duration: float,
    control_dt: float = config.CONTROL_DT,
) -> None:
    """Linearly interpolate to target joint positions and execute."""
    n_steps = max(1, int(duration / control_dt))
    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        bot.set_joint_pos({
            'left_arm':  current_left_q  + alpha * (target_left_q  - current_left_q),
            'right_arm': current_right_q + alpha * (target_right_q - current_right_q),
        })
        time.sleep(control_dt)


# ── Keyboard ─────────────────────────────────────────────────────────────────

def get_key() -> str:
    """Read one keypress. Arrow keys are returned as escape sequences."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return '\x1b[' + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Display ──────────────────────────────────────────────────────────────────

def print_status(
    left_pos: np.ndarray,
    left_rpy: np.ndarray,
    right_pos: np.ndarray,
    right_rpy: np.ndarray,
    state: str,
    step: float,
) -> None:
    l = left_pos
    r = right_pos
    print(f"\n{'─'*64}")
    print(f" State : {state}")
    print(f" Step  : {step:.3f} m")
    print(f" Left  pos : ({l[0]:+.4f},  {l[1]:+.4f},  {l[2]:+.4f})  "
          f"rpy: ({left_rpy[0]:+.4f}, {left_rpy[1]:+.4f}, {left_rpy[2]:+.4f})")
    print(f" Right pos : ({r[0]:+.4f},  {r[1]:+.4f},  {r[2]:+.4f})  "
          f"rpy: ({right_rpy[0]:+.4f}, {right_rpy[1]:+.4f}, {right_rpy[2]:+.4f})")
    print(f"{'─'*64}")
    print(f" [SPACE] toggle default↔target  |  [m/n] ±x  [←/→] widen/narrow  [↑/↓] ±z")
    print(f" [q] quit")
    print(f"{'─'*64}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    pose_file: str = "trajectories/test.txt",
    current: bool = False,
    step: float = 0.02,
    move_duration: float = 3.0,
    adjust_duration: float = 0.4,
    use_shoulder_bias: bool = False,
) -> None:
    """Interactive pose tuning tool.

    Args:
        pose_file: Path to pose file with two lines: left arm, right arm (arm_center frame).
        current: If True, start from the robot's current live pose (skips default step).
        step: Position adjustment per keypress in metres.
        move_duration: Duration (s) for default ↔ target transitions.
        adjust_duration: Duration (s) for incremental key adjustments.
        use_shoulder_bias: If True, bias j2 shoulder abduction toward joint limits in IK.
    """
    # ── Build IK context (URDF, tasks, robot connection) ───────────────────
    ctx = build_ik_context(skip_confirmation=False)
    bot = ctx.bot
    model = ctx.model
    left_arm_indices  = ctx.left_arm_indices
    right_arm_indices = ctx.right_arm_indices
    torso_indices     = ctx.torso_indices
    configuration     = ctx.configuration

    try:
        # Read live joint state
        live_left_q  = bot.left_arm.get_joint_pos().astype(float)
        live_right_q = bot.right_arm.get_joint_pos().astype(float)
        live_torso_q = bot.torso.get_joint_pos().astype(float)

        # Sync IK config to live state
        q_live = configuration.q.copy()
        for j, idx in enumerate(left_arm_indices):
            q_live[model.idx_qs[idx]] = live_left_q[j]
        for j, idx in enumerate(right_arm_indices):
            q_live[model.idx_qs[idx]] = live_right_q[j]
        for j, idx in enumerate(torso_indices):
            q_live[model.idx_qs[idx]] = live_torso_q[j]
        q_live = np.clip(q_live, model.lowerPositionLimit, model.upperPositionLimit)
        configuration.update(q_live)

        current_left_q  = live_left_q.copy()
        current_right_q = live_right_q.copy()

        # ── Initialise state ─────────────────────────────────────────────────
        if current:
            # Compute FK to get current EE poses in arm_center frame
            logger.info("--current: reading live end-effector pose from FK...")
            pin.framesForwardKinematics(ctx.robot_pin.model, ctx.robot_pin.data, configuration.q)
            arm_center_id = ctx.robot_pin.model.getFrameId("arm_center")
            T_world_ac    = ctx.robot_pin.data.oMf[arm_center_id]

            T_ac_left  = T_world_ac.inverse() * ctx.robot_pin.data.oMf[ctx.robot_pin.model.getFrameId("L_gripper_base")]
            T_ac_right = T_world_ac.inverse() * ctx.robot_pin.data.oMf[ctx.robot_pin.model.getFrameId("R_gripper_base")]

            left_pos  = T_ac_left.translation.copy()
            right_pos = T_ac_right.translation.copy()
            left_rpy  = Rotation.from_matrix(T_ac_left.rotation).as_euler('xyz')
            right_rpy = Rotation.from_matrix(T_ac_right.rotation).as_euler('xyz')

            state = 'AT_TARGET'
            logger.info(f"Live left  pose: pos={left_pos}, rpy={left_rpy}")
            logger.info(f"Live right pose: pos={right_pos}, rpy={right_rpy}")

        else:
            # Go to default pose first
            logger.info("Moving to default pose...")
            move_to_joints(bot, current_left_q, current_right_q,
                           DEFAULT_LEFT_Q, DEFAULT_RIGHT_Q, move_duration)
            current_left_q  = DEFAULT_LEFT_Q.copy()
            current_right_q = DEFAULT_RIGHT_Q.copy()

            # Load initial pose from file (but don't move yet)
            left_pos, left_rpy, right_pos, right_rpy = read_pose_file(pose_file)
            state = 'AT_DEFAULT'
            logger.info("At default pose. Press SPACE to move to target.")

        print_status(left_pos, left_rpy, right_pos, right_rpy, state, step)

        # ── Main key loop ────────────────────────────────────────────────────
        while True:
            key = get_key()

            if key in ('q', '\x03'):
                logger.info("Quitting...")
                break

            # ── SPACE: toggle between default and target ──────────────────
            elif key == ' ':
                if state == 'AT_DEFAULT':
                    # Re-read file so edits are picked up
                    left_pos, left_rpy, right_pos, right_rpy = read_pose_file(pose_file)
                    logger.info(f"Read pose from {pose_file}")
                    logger.info("  Solving IK and moving to target...")

                    q_sol, ok = solve_ik_for_pose(
                        ctx, left_pos, left_rpy, right_pos, right_rpy,
                        use_shoulder_bias=use_shoulder_bias,
                    )
                    if not ok:
                        logger.warning("IK did not fully converge, executing anyway")

                    target_left_q  = np.array([q_sol[model.idx_qs[idx]] for idx in left_arm_indices])
                    target_right_q = np.array([q_sol[model.idx_qs[idx]] for idx in right_arm_indices])

                    move_to_joints(bot, current_left_q, current_right_q,
                                   target_left_q, target_right_q, move_duration)
                    current_left_q  = target_left_q.copy()
                    current_right_q = target_right_q.copy()
                    state = 'AT_TARGET'

                else:  # AT_TARGET → back to default
                    logger.info("Returning to default pose...")
                    move_to_joints(bot, current_left_q, current_right_q,
                                   DEFAULT_LEFT_Q, DEFAULT_RIGHT_Q, move_duration)
                    current_left_q  = DEFAULT_LEFT_Q.copy()
                    current_right_q = DEFAULT_RIGHT_Q.copy()
                    state = 'AT_DEFAULT'

                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step)

            # ── Adjustment keys (only when at target) ─────────────────────
            else:
                if state != 'AT_TARGET':
                    logger.info("Go to target pose first (press SPACE)")
                    continue

                left_delta  = np.zeros(3)
                right_delta = np.zeros(3)
                #if   key == 'm':       left_delta[0] = right_delta[0] = +step   # +x  forward
                #elif key == 'n':       left_delta[0] = right_delta[0] = -step   # -x  back

                if   key == 'm':       left_delta[0] = +step   # +x  forward
                elif key == 'n':       left_delta[0] = -step   # -x  back

                elif key == '\x1b[D':  left_delta[1] = +step; right_delta[1] = -step  # ←  widen
                elif key == '\x1b[C':  left_delta[1] = -step; right_delta[1] = +step  # →  narrow
                #elif key == '\x1b[A':  left_delta[2] = right_delta[2] = +step   # ↑   +z  up
                elif key == '\x1b[A':  left_delta[2]  = +step   # ↑   +z  up
                #elif key == '\x1b[B':  left_delta[2] = right_delta[2] = -step   # ↓   -z  down
                elif key == '\x1b[B':  left_delta[2] = -step   # ↓   -z  down
                else:
                    continue

                left_pos  = left_pos  + left_delta
                right_pos = right_pos + right_delta

                logger.info(
                    f"  left_delta ({left_delta[0]:+.3f}, {left_delta[1]:+.3f}, {left_delta[2]:+.3f})  "
                    f"right_delta ({right_delta[0]:+.3f}, {right_delta[1]:+.3f}, {right_delta[2]:+.3f})  →  "
                    f"left ({left_pos[0]:+.4f}, {left_pos[1]:+.4f}, {left_pos[2]:+.4f})  "
                    f"right ({right_pos[0]:+.4f}, {right_pos[1]:+.4f}, {right_pos[2]:+.4f})"
                )

                q_sol, ok = solve_ik_for_pose(
                    ctx, left_pos, left_rpy, right_pos, right_rpy,
                    use_shoulder_bias=use_shoulder_bias,
                )
                if not ok:
                    logger.warning("IK did not fully converge")

                target_left_q  = np.array([q_sol[model.idx_qs[idx]] for idx in left_arm_indices])
                target_right_q = np.array([q_sol[model.idx_qs[idx]] for idx in right_arm_indices])

                move_to_joints(bot, current_left_q, current_right_q,
                               target_left_q, target_right_q, adjust_duration)
                current_left_q  = target_left_q.copy()
                current_right_q = target_right_q.copy()

                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down robot connection")
        ctx.bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
