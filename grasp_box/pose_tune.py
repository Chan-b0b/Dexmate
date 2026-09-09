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
    1 / 2 / 3     — select which arm the adjustment keys act on: left / right / both
    m / n         — move ±x (forward / back)
    Left / Right  — move ±y (left / right)
    Up / Down     — move ±z (up / down)
    r / f         — rotate ±roll
    t / g         — rotate ±pitch
    y / h         — rotate ±yaw
    [ / ]         — decrease / increase position step
    - / =         — decrease / increase rotation step
    . / ,         — right-hand Robotiq gripper: close / open by one grip step
    p / o         — right-hand Robotiq gripper: close on contact / fully open
                    (p stops at first finger contact unless --no-soft-grip)
    q / Ctrl+C    — quit

In "both" mode every delta is applied in the same direction to both arms.

Usage:
    python pose_tune.py
    python pose_tune.py --pose_file trajectories/test.txt
    python pose_tune.py --current      # start from current robot pose, no default step
    python pose_tune.py --step 0.05    # position step size (default 0.02 m)
    python pose_tune.py --arm both     # initial arm selection (default left)
    python pose_tune.py --rot_step_deg 10 # rotation step in degrees (default 5)
    python pose_tune.py --no-gripper   # skip the Robotiq gripper entirely
    python pose_tune.py --cu_stop 2    # gentler contact-stop threshold (default 3)
    python pose_tune.py --no-soft-grip # let 'p' squeeze to the force-controller floor
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

# Right-hand Robotiq gripper — reuse the ik_demo driver (Modbus RTU over the
# arm EE pass-through). Its RS485 cable sits on the side named by
# ik_demo's ROBOTIQ_EE_SIDE, which is not necessarily the mounting arm.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LGES.ik_demo.drivers.robotiq import RobotiqGripper
from LGES.ik_demo import config as gripper_cfg

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


# ── Gripper ──────────────────────────────────────────────────────────────────

def setup_gripper(bot) -> tuple[object | None, int]:
    """Reset + activate the right-hand Robotiq gripper.

    Returns (driver_or_None, current_position). Activation runs the gripper's
    own open/close calibration sweep, so the fingers will move once here.
    """
    side = gripper_cfg.ROBOTIQ_EE_SIDE or "right"
    gripper = RobotiqGripper(bot, side=side)
    if not gripper.initialize():
        logger.warning("Robotiq gripper did not answer — gripper keys disabled")
        return None, gripper_cfg.ROBOTIQ_OPEN_POS

    status = gripper.read_status()
    pos = status.gPO if status is not None else gripper_cfg.ROBOTIQ_OPEN_POS
    logger.info(f"Robotiq gripper ready on the {side} arm's EE bus (position {pos})")
    return gripper, pos


def soft_close(grip, cu_stop: int) -> int:
    """Close until the fingers touch, then stop — no hard squeeze.

    A plain close() lets the gripper's own force controller push until it
    stalls, which is ~20 N even at force 0. This streams a slow close instead,
    polls the motor current, and the moment the current has been above
    ``cu_stop`` for SOFT_GRIP_CU_CONSECUTIVE polls it re-commands the position
    the fingers are at right now (+SOFT_GRIP_SQUEEZE counts). Holding a
    position applies only the elastic squeeze of that extra travel.

    Returns the resting finger position (0=open .. 255=closed).
    """
    speed = config.SOFT_GRIP_SPEED
    force = config.SOFT_GRIP_FORCE
    grip.write_control(gripper_cfg.ROBOTIQ_CLOSE_POS, speed=speed, force=force)

    t0 = time.time()
    cu_hits = 0
    while time.time() - t0 < config.SOFT_GRIP_TIMEOUT_S:
        status = grip.read_status()
        if status is None:
            continue

        if status.gOBJ == 3:
            logger.info(f"  Soft close: fully closed, nothing between the fingers (gPO={status.gPO})")
            return status.gPO
        if status.gOBJ == 2:
            logger.info(f"  Soft close: force controller stalled first at gPO={status.gPO} "
                        f"(gCU={status.gCU} ~{status.gCU * 10} mA) — contact was missed, "
                        f"lower --cu_stop for a gentler grip")
            return status.gPO

        # Consecutive polls only: a single noise spike or a one-finger graze
        # must not freeze the close — the object keeps self-centring until
        # BOTH fingers load up.
        if time.time() - t0 > config.SOFT_GRIP_WARMUP_S and status.gCU >= cu_stop:
            cu_hits += 1
        else:
            cu_hits = 0

        if cu_hits >= config.SOFT_GRIP_CU_CONSECUTIVE:
            target = min(255, status.gPO + config.SOFT_GRIP_SQUEEZE)
            grip.write_control(target, speed=speed, force=force)
            logger.info(f"  Soft close: contact at gPO={status.gPO} "
                        f"(gCU={status.gCU} ~{status.gCU * 10} mA) — holding {target}")
            time.sleep(0.3)  # let the fingers settle before reading back
            final = grip.read_status()
            return final.gPO if final is not None else target

    logger.warning(f"  Soft close: no contact within {config.SOFT_GRIP_TIMEOUT_S:.1f} s")
    final = grip.read_status()
    return final.gPO if final is not None else gripper_cfg.ROBOTIQ_CLOSE_POS


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
    arm: str,
    rot_step: float,
    grip_pos: int | None,
    grip_step: int,
) -> None:
    l = left_pos
    r = right_pos
    print(f"\n{'─'*64}")
    print(f" State : {state}")
    print(f" Arm   : {arm.upper()}")
    print(f" Step  : {step:.3f} m  |  {np.rad2deg(rot_step):.1f}° rot")
    if grip_pos is None:
        print(f" Grip  : (disabled)")
    else:
        print(f" Grip  : {grip_pos:3d} / 255  (0=open, 255=closed)  |  step {grip_step}")
    print(f" Left  pos : ({l[0]:+.4f},  {l[1]:+.4f},  {l[2]:+.4f})  "
          f"rpy: ({left_rpy[0]:+.4f}, {left_rpy[1]:+.4f}, {left_rpy[2]:+.4f})")
    print(f" Right pos : ({r[0]:+.4f},  {r[1]:+.4f},  {r[2]:+.4f})  "
          f"rpy: ({right_rpy[0]:+.4f}, {right_rpy[1]:+.4f}, {right_rpy[2]:+.4f})")
    print(f"{'─'*64}")
    print(f" [SPACE] toggle default↔target  |  [1] left  [2] right  [3] both")
    print(f" [m/n] ±x  [←/→] ±y  [↑/↓] ±z  |  [r/f] ±roll  [t/g] ±pitch  [y/h] ±yaw")
    print(f" [[/]] pos step  [-/=] rot step  |  [./,] grip close/open  [p/o] full close/open")
    print(f" [q] quit")
    print(f"{'─'*64}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    pose_file: str = "trajectories/test.txt",
    current: bool = False,
    step: float = 0.02,
    arm: str = "left",
    rot_step_deg: float = 5.0,
    gripper: bool = True,
    grip_step: int = 25,
    soft_grip: bool = True,
    cu_stop: int = config.SOFT_GRIP_CU_STOP,
    move_duration: float = 3.0,
    adjust_duration: float = 0.4,
    use_shoulder_bias: bool = False,
) -> None:
    """Interactive pose tuning tool.

    Args:
        pose_file: Path to pose file with two lines: left arm, right arm (arm_center frame).
        current: If True, start from the robot's current live pose (skips default step).
        step: Position adjustment per keypress in metres.
        arm: Initial arm selection for adjustment keys: "left", "right" or "both".
        rot_step_deg: Rotation adjustment per keypress in degrees.
        gripper: If True, activate the right-hand Robotiq gripper and enable its keys.
        grip_step: Gripper adjustment per keypress in raw counts (0=open .. 255=closed).
        soft_grip: If True, the full-close key (p) stops at finger contact instead of
            letting the gripper's force controller squeeze to its ~20 N floor.
        cu_stop: Motor-current threshold (gCU counts, ~10 mA each) that counts as
            contact during a soft close. Lower = gentler, more prone to false stops.
        move_duration: Duration (s) for default ↔ target transitions.
        adjust_duration: Duration (s) for incremental key adjustments.
        use_shoulder_bias: If True, bias j2 shoulder abduction toward joint limits in IK.
    """
    if arm not in ('left', 'right', 'both'):
        raise ValueError(f"--arm must be left, right or both (got {arm!r})")
    rot_step = np.deg2rad(rot_step_deg)
    grip_step = max(1, min(255, int(grip_step)))

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

        # ── Right-hand Robotiq gripper ───────────────────────────────────────
        grip = None
        grip_pos = None
        if gripper:
            grip, grip_pos = setup_gripper(bot)
            if grip is None:
                grip_pos = None

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

        print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

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

                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

            # ── Arm selection ─────────────────────────────────────────────
            elif key in ('1', '2', '3'):
                arm = {'1': 'left', '2': 'right', '3': 'both'}[key]
                logger.info(f"Arm selection: {arm.upper()}")
                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

            # ── Gripper ───────────────────────────────────────────────────
            elif key in ('.', ',', 'p', 'o'):
                if grip is None:
                    logger.info("Gripper is disabled")
                    continue

                if key == 'p' and soft_grip:
                    # Full close, but stop at contact instead of squeezing
                    grip_pos = soft_close(grip, cu_stop)
                else:
                    if   key == '.':  target_pos = grip_pos + grip_step   # squeeze a bit
                    elif key == ',':  target_pos = grip_pos - grip_step   # release a bit
                    elif key == 'p':  target_pos = gripper_cfg.ROBOTIQ_CLOSE_POS
                    else:             target_pos = gripper_cfg.ROBOTIQ_OPEN_POS
                    target_pos = int(np.clip(target_pos, 0, 255))

                    grip.goto(target_pos)
                    status = grip.read_status()
                    grip_pos = status.gPO if status is not None else target_pos
                    logger.info(f"  Grip → commanded {target_pos}, actual {grip_pos}")

                if grip.is_object_grasped():
                    logger.info("  Object gripped")
                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

            # ── Step size ─────────────────────────────────────────────────
            elif key in ('[', ']', '-', '='):
                if   key == '[':  step = max(0.001, step / 2)
                elif key == ']':  step = min(0.200, step * 2)
                elif key == '-':  rot_step = max(np.deg2rad(0.5), rot_step / 2)
                elif key == '=':  rot_step = min(np.deg2rad(45.0), rot_step * 2)
                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

            # ── Adjustment keys (only when at target) ─────────────────────
            else:
                if state != 'AT_TARGET':
                    logger.info("Go to target pose first (press SPACE)")
                    continue

                pos_delta = np.zeros(3)
                rpy_delta = np.zeros(3)

                if   key == 'm':       pos_delta[0] = +step       # +x  forward
                elif key == 'n':       pos_delta[0] = -step       # -x  back
                elif key == '\x1b[D':  pos_delta[1] = +step       # ←   +y  left
                elif key == '\x1b[C':  pos_delta[1] = -step       # →   -y  right
                elif key == '\x1b[A':  pos_delta[2] = +step       # ↑   +z  up
                elif key == '\x1b[B':  pos_delta[2] = -step       # ↓   -z  down
                elif key == 'r':       rpy_delta[0] = +rot_step   # +roll
                elif key == 'f':       rpy_delta[0] = -rot_step   # -roll
                elif key == 't':       rpy_delta[1] = +rot_step   # +pitch
                elif key == 'g':       rpy_delta[1] = -rot_step   # -pitch
                elif key == 'y':       rpy_delta[2] = +rot_step   # +yaw
                elif key == 'h':       rpy_delta[2] = -rot_step   # -yaw
                else:
                    continue

                # Same-direction delta applied to whichever arm(s) are selected
                if arm in ('left', 'both'):
                    left_pos  = left_pos  + pos_delta
                    left_rpy  = left_rpy  + rpy_delta
                if arm in ('right', 'both'):
                    right_pos = right_pos + pos_delta
                    right_rpy = right_rpy + rpy_delta

                logger.info(
                    f"  [{arm.upper()}] "
                    f"pos_delta ({pos_delta[0]:+.3f}, {pos_delta[1]:+.3f}, {pos_delta[2]:+.3f})  "
                    f"rpy_delta ({rpy_delta[0]:+.3f}, {rpy_delta[1]:+.3f}, {rpy_delta[2]:+.3f})  →  "
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

                print_status(left_pos, left_rpy, right_pos, right_rpy, state, step, arm,
                             rot_step, grip_pos, grip_step)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down robot connection")
        ctx.bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
