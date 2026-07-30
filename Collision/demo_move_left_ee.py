#!/usr/bin/env python3
"""Random-motion collision demo for the left arm.

Continuously moves the left EE to random targets inside a small box around
its start pose (±X forward/back, ±Y left/right, ±Z up/down) while the
model-based ``CollisionMonitor`` (calibration_left.json) watches for contact.

On collision:
  - motion stops immediately (the monitor freezes both arms),
  - the sensed contact is printed: per-joint banded residual [A], its
    torque equivalent [Nm], and the wrist wrench force [N],
  - the arm holds still for ``--pause`` seconds (default 1 s), then resumes
    with a new random target. If the obstacle is still in the way, it simply
    triggers again.

The process keeps running until 'q'+Enter or Ctrl+C. Press 'r'+Enter after
grasping or releasing an object to re-estimate the payload
(``monitor.retare_payload()``) — the demo then executes one slow stroke and
samples during it, because at standstill stiction carries the extra load and
hides it from the motor current. Keep the path clear during that stroke.

Requires a calibration file — run ``calibrate_gravity_model.py`` first.

Usage:
    python demo_move_left_ee.py                          # default ranges
    python demo_move_left_ee.py --amp-x 0.08 --amp-z 0.1 --seed 7
"""

import os
import select
import sys
import threading
import time

import numpy as np
import pinocchio as pin
import qpsolvers
import tyro
from loguru import logger

import pink
from pink import solve_ik
from pink.tasks import PostureTask, RelativeFrameTask

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

URDF_PATH = "/home/dexmate/miniconda3/lib/python3.13/site-packages/dexmate_urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"
CONTROL_DT = 0.02  # 50 Hz command rate
IK_DT = 0.01
IK_MAX_ITERS = 200
IK_CONVERGENCE_THRESHOLD = 1e-4
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class LeftArmIK:
    """Minimal Pink IK for one arm (chest-relative, torso stays put).

    Historical name: with ``side="right"`` the class solves/monitors the
    RIGHT arm — the ``left_*`` attribute and method names then refer to that
    arm (kept so existing callers work unchanged)."""

    def __init__(self, side: str = "left") -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        prefix = "L" if side == "left" else "R"
        self.ee_frame = f"{prefix}_gripper_base"
        vega_dir = os.path.dirname(URDF_PATH)
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(vega_dir)))
        robot_pin = pin.RobotWrapper.BuildFromURDF(
            filename=URDF_PATH,
            package_dirs=[vega_dir, package_dir],
            root_joint=None,
        )
        self.model = robot_pin.model
        self.data = robot_pin.data
        self.configuration = pink.Configuration(self.model, self.data, pin.neutral(self.model))

        # EE frame relative to arm_center: the kinematic path contains only
        # the 7 monitored-arm joints, so torso/other-arm joints stay out of
        # the Jacobian.
        self.left_task = RelativeFrameTask(
            self.ee_frame, root="arm_center",
            position_cost=2.0, orientation_cost=1.0,
        )
        self.posture_task = PostureTask(cost=1e-3)

        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

        self._L_idx = [self.model.getJointId(f"L_arm_j{j+1}") for j in range(7)]
        self._R_idx = [self.model.getJointId(f"R_arm_j{j+1}") for j in range(7)]
        # Monitored arm (historical names — see class docstring).
        self.left_idx = self._L_idx if side == "left" else self._R_idx
        self.right_idx = self._R_idx
        self.torso_idx = [self.model.getJointId(f"torso_j{j+1}") for j in range(3)]

    def sync_from_robot(self, bot: Robot) -> None:
        """Overwrite the IK configuration with the live robot joint state."""
        q = self.configuration.q.copy()
        for j, idx in enumerate(self._L_idx):
            q[self.model.idx_qs[idx]] = bot.left_arm.get_joint_pos().astype(float)[j]
        for j, idx in enumerate(self._R_idx):
            q[self.model.idx_qs[idx]] = bot.right_arm.get_joint_pos().astype(float)[j]
        for j, idx in enumerate(self.torso_idx):
            q[self.model.idx_qs[idx]] = bot.torso.get_joint_pos().astype(float)[j]
        q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        self.configuration.update(q)

    def left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rotation) of the monitored arm's gripper base in
        the arm_center frame."""
        pin.framesForwardKinematics(self.model, self.data, self.configuration.q)
        T_ac = self.data.oMf[self.model.getFrameId("arm_center")]
        T_left = T_ac.inverse() * self.data.oMf[self.model.getFrameId(self.ee_frame)]
        return T_left.translation.copy(), T_left.rotation.copy()

    def solve_left(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """Solve IK for a left-EE target pose; returns the 7 left-arm joint angles."""
        self.left_task.set_target(pin.SE3(target_rot, target_pos))
        self.posture_task.set_target(self.configuration.q.copy())
        tasks = [self.left_task, self.posture_task]
        error = np.inf
        for _ in range(IK_MAX_ITERS):
            velocity = solve_ik(self.configuration, tasks, IK_DT, solver=self.solver)
            q_next = pin.integrate(self.model, self.configuration.q, velocity * IK_DT)
            self.configuration.update(q_next)
            error = np.linalg.norm(self.left_task.compute_error(self.configuration))
            if error < IK_CONVERGENCE_THRESHOLD:
                break
        else:
            logger.warning(f"IK did not fully converge (error: {error:.5f})")
        return np.array(
            [self.configuration.q[self.model.idx_qs[idx]] for idx in self.left_idx]
        )


def check_environment() -> None:
    """Verify ZENOH_CONFIG and ROBOT_NAME are set."""
    if not os.environ.get("ZENOH_CONFIG"):
        logger.error("ZENOH_CONFIG not set. Please set it in ~/.bashrc")
        sys.exit(1)
    if not os.environ.get("ROBOT_NAME"):
        logger.error("ROBOT_NAME not set. Please set it in ~/.bashrc")
        sys.exit(1)


def tare_wrench(arm, samples: int = 10) -> np.ndarray | None:
    """Average the resting wrist wrench (fx, fy, fz); None if sensor unavailable."""
    ws = arm.wrench_sensor
    if ws is None:
        return None
    readings = []
    for _ in range(samples):
        try:
            state = ws.get_state()
        except ServiceUnavailableError:
            return None
        readings.append(np.asarray(state["wrench"], dtype=float)[:3])
        time.sleep(0.02)
    return np.mean(readings, axis=0)


def read_force(arm, wrench_baseline: np.ndarray | None) -> float | None:
    """Tared wrist force magnitude [N]; None if sensor unavailable."""
    if wrench_baseline is None:
        return None
    ws = arm.wrench_sensor
    if ws is None:
        return None
    try:
        state = ws.get_state()
    except ServiceUnavailableError:
        return None
    f = np.asarray(state["wrench"], dtype=float)[:3] - wrench_baseline
    return float(np.linalg.norm(f))


def read_command() -> str | None:
    """Non-blocking read of a one-letter command + Enter from stdin.

    Returns 'q' (quit), 'r' (retare payload), or None. Commands take effect
    at the next stroke boundary.
    """
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.readline().strip().lower() or None
    return None


def run_stroke(bot: Robot, monitor, q_from: np.ndarray, q_to: np.ndarray, duration: float) -> bool:
    """Smoothstep the left arm toward q_to; abort as soon as the monitor triggers.

    Returns True if the stroke completed, False if aborted on collision.
    """
    n_steps = max(1, int(duration / CONTROL_DT))
    for step in range(n_steps):
        if monitor.triggered:
            return False  # monitor has already frozen the arms
        t = (step + 1) / n_steps
        alpha = t * t * (3 - 2 * t)
        bot.set_joint_pos({"left_arm": q_from + alpha * (q_to - q_from)})
        time.sleep(CONTROL_DT)
    return not monitor.triggered


def report_collision(bot: Robot, monitor, wrench_baseline: np.ndarray | None) -> None:
    """Print how much the detected contact registered on each sensor."""
    info = monitor.trigger_info or {}
    layer = info.get("layer", "?")
    excess = info.get("excess", monitor.last_excess)
    k = monitor._k
    torque_eq = np.where(np.abs(k) > 1e-6, excess / np.abs(k), 0.0)
    force = read_force(bot.left_arm, wrench_baseline)
    logger.warning("=" * 60)
    logger.warning(f"COLLISION ({layer} layer) — sensed contact:")
    logger.warning(f"  residual excess per joint [A]:    {np.round(excess, 3)}")
    logger.warning(f"  torque equivalent per joint [Nm]: {np.round(torque_eq, 2)}")
    if force is not None:
        logger.warning(f"  wrist wrench force:               {force:.2f} N")
    else:
        logger.warning("  wrist wrench force:               (sensor unavailable)")
    logger.warning("=" * 60)


def main(
    amp_x: float = 0.10,
    amp_y: float = 0.15,
    amp_z: float = 0.10,
    joint_speed: float = 0.5,
    pause: float = 1.0,
    seed: int | None = None,
) -> None:
    """Move the left EE to random targets until 'q'+Enter / Ctrl+C; stop on contact.

    Args:
        amp_x: Random target range [m] forward/back (arm_center X).
        amp_y: Random target range [m] left/right (arm_center Y).
        amp_z: Random target range [m] up/down (arm_center Z).
        joint_speed: Peak joint speed [rad/s]; stroke duration scales with
            distance so speed stays constant. Keep this at or below the
            calibration's validation_joint_speed — the collision thresholds
            were measured at that speed.
        pause: Seconds to hold still after a collision before resuming.
        seed: RNG seed for a reproducible target sequence.
    """
    check_environment()

    logger.info("Loading robot model for IK...")
    ik = LeftArmIK()

    logger.warning("=" * 60)
    logger.warning(f"The LEFT arm will move to RANDOM targets within ±{amp_x*100:.0f}/"
                   f"±{amp_y*100:.0f}/±{amp_z*100:.0f} cm (X/Y/Z) of its start pose,")
    logger.warning("continuously, until 'q'+Enter or Ctrl+C. On contact it stops and")
    logger.warning("prints the sensed force. Keep the e-stop within reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").lower() != "y":
        logger.info("Cancelled by user")
        sys.exit(0)

    logger.info(f"Connecting to robot: {os.environ['ROBOT_NAME']}")
    bot = Robot()

    # Imported here (not at module top) because collision_monitor imports this
    # module for LeftArmIK — a top-level import would be circular.
    from collision_monitor import CollisionMonitor
    monitor = CollisionMonitor(bot)

    try:
        ik.sync_from_robot(bot)
        pos0, rot0 = ik.left_ee_pose()
        q_home = bot.left_arm.get_joint_pos().astype(float)
        logger.info(f"Left EE start pos (arm_center frame): {np.round(pos0, 3)}")

        logger.info("Taring wrist wrench sensor (keep the gripper free)...")
        wrench_baseline = tare_wrench(bot.left_arm)
        if wrench_baseline is None:
            logger.warning("Wrench sensor unavailable — wrench force will not be printed.")

        monitor.start()
        rng = np.random.default_rng(seed)
        q_prev = q_home
        stroke_i = 0
        logger.info("Random motion running — 'q'+Enter: quit, 'r'+Enter: retare payload "
                    "(after grasping/releasing an object), Ctrl+C anytime.")

        while True:
            cmd = read_command()
            if cmd == "q":
                logger.info("Quit requested.")
                break
            if cmd == "r":
                # Retare needs the arm MOVING (at rest, stiction carries the
                # extra load and hides it from the current), so sample during
                # one slow stroke to a random target.
                logger.info("Retaring payload during one slow stroke — keep the path clear...")
                offset = rng.uniform(-1.0, 1.0, 3) * np.array([amp_x, amp_y, amp_z])
                ik.sync_from_robot(bot)
                q_target = ik.solve_left(pos0 + offset, rot0)
                duration = float(np.clip(
                    1.5 * np.max(np.abs(q_target - q_prev)) / joint_speed, 2.0, 6.0))
                retare_thread = threading.Thread(
                    target=monitor.retare_payload, kwargs={"duration": duration})
                retare_thread.start()
                completed = run_stroke(bot, monitor, q_prev, q_target, duration)
                retare_thread.join()
                if completed:
                    q_prev = q_target
                else:
                    report_collision(bot, monitor, wrench_baseline)
                    time.sleep(pause)
                    monitor.reset()
                    q_prev = bot.left_arm.get_joint_pos().astype(float)
                continue

            offset = rng.uniform(-1.0, 1.0, 3) * np.array([amp_x, amp_y, amp_z])
            ik.sync_from_robot(bot)
            q_target = ik.solve_left(pos0 + offset, rot0)
            # Constant peak speed: smoothstep peak velocity = 1.5 * dq / T.
            duration = float(np.clip(
                1.5 * np.max(np.abs(q_target - q_prev)) / joint_speed, 0.8, 6.0))
            stroke_i += 1
            logger.info(f"Stroke {stroke_i}: target offset {np.round(offset, 3)} m, {duration:.1f} s")

            completed = run_stroke(bot, monitor, q_prev, q_target, duration)

            if not completed:
                report_collision(bot, monitor, wrench_baseline)
                logger.info(f"Holding for {pause:.1f} s, then resuming...")
                time.sleep(pause)  # arms are frozen by the monitor; just wait
                monitor.reset()
                q_prev = bot.left_arm.get_joint_pos().astype(float)
            else:
                q_prev = q_target

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        monitor.stop()
        logger.info("Shutting down; arm holds its last commanded position.")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
