"""Module 3 — Suction Grasp Execution.

Moves the arm to a hover position above a detected battery, pre-activates
suction, then descends until a vacuum seal is confirmed (primary) or
contact force is detected (secondary).

Contact detection strategy
--------------------------
Activating suction *before* touching the battery is more reliable than
waiting for a force spike, because:
- The suction cup rubber is compliant and seals before significant force builds.
- DI0 = 1 (vacuum sealed) is an unambiguous "I'm gripping" signal.
- Force is used as a safety fallback and hard-stop guard.

Priority:
  1. DI0 = 1  → vacuum sealed, stop descent, return success.
  2. force > FORCE_CONTACT_THRESHOLD_N → hold, wait up to VACUUM_SEAL_TIMEOUT_S
     for vacuum to seal; succeed if it seals, fail otherwise.
  3. force > FORCE_HARD_LIMIT_N  → emergency retract, return failure.
  4. MAX_DESCENT_M reached without signal → failure.

Usage::

    from battery_pick.suction_grasp import SuctionGrasper
    from battery_pick.perception import BatteryDetector

    with Robot() as bot:
        detector  = BatteryDetector(bot)
        grasper   = SuctionGrasper(bot)

        obs = detector.detect()
        result = grasper.grasp(obs[0])
        if result.success:
            # transport battery to place pose …
            grasper.release()
        grasper.close()
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import requests
from loguru import logger
from scipy.spatial.transform import Rotation

# Pink / Pinocchio
import pinocchio as pin
import pink
import qpsolvers
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

# Resolve sibling imports
_LGES_DIR = os.path.join(os.path.dirname(__file__), "..")
if _LGES_DIR not in sys.path:
    sys.path.insert(0, _LGES_DIR)

import battery_pick.config as cfg  # noqa: E402
from read_force import get_force, tare_force  # noqa: E402
from battery_pick.perception import BatteryObservation  # noqa: E402


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class GraspResult:
    """Outcome of a single grasp attempt.

    Attributes:
        success: True when the battery was securely gripped.
        contact_position_base: [x, y, z] where contact was detected in the
            robot base frame (None on max-descent failure).
        trigger: What stopped the descent — ``"vacuum"``, ``"force+vacuum"``,
            ``"vacuum_timeout"``, ``"force_limit"``, or ``"max_descent"``.
    """

    __slots__ = ("success", "contact_position_base", "trigger")

    def __init__(
        self,
        success: bool,
        contact_position_base: np.ndarray | None,
        trigger: str,
    ) -> None:
        self.success = success
        self.contact_position_base = contact_position_base
        self.trigger = trigger

    def __repr__(self) -> str:
        pos = (
            f"({self.contact_position_base[0]:.3f}, "
            f"{self.contact_position_base[1]:.3f}, "
            f"{self.contact_position_base[2]:.3f})"
            if self.contact_position_base is not None
            else "None"
        )
        return f"GraspResult(success={self.success}, pos={pos}, trigger={self.trigger!r})"


# ---------------------------------------------------------------------------
# Vacuum monitor (socketio background thread)
# ---------------------------------------------------------------------------

class _VacuumMonitor:
    """Listens for DI0 events from the suction controller via socketio.

    Runs in a daemon thread so it doesn't block the grasp loop.
    Call ``start()`` before descending and ``stop()`` afterwards.
    """

    def __init__(self, host: str) -> None:
        import socketio as _sio_module  # lazy import to keep startup fast

        self._seal_event = threading.Event()
        self._sio = _sio_module.Client()
        self._thread: threading.Thread | None = None

        @self._sio.on("*")
        def _on_data(event, data) -> None:
            try:
                di0 = data["computebox"]["variable"]["dInput"][0]
                if di0 == 1:
                    self._seal_event.set()
                elif di0 == 0:
                    self._seal_event.clear()
            except (KeyError, TypeError):
                pass

        self._host = host

    def start(self) -> None:
        self._seal_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="VacuumMonitor")
        self._thread.start()

    def _run(self) -> None:
        try:
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            self._sio.wait()
        except Exception as exc:
            logger.debug("[VacuumMonitor] connection ended: {}", exc)

    def is_sealed(self) -> bool:
        return self._seal_event.is_set()

    def wait_for_seal(self, timeout: float) -> bool:
        """Block up to *timeout* seconds for DI0 = 1.  Returns True if sealed."""
        return self._seal_event.wait(timeout=timeout)

    def stop(self) -> None:
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Suction API helpers
# ---------------------------------------------------------------------------

def _stop_processes() -> None:
    requests.post(f"{cfg.SUCTION_BASE_URL}/stop", timeout=5.0)


def _suction_on() -> None:
    _stop_processes()
    time.sleep(0.5)
    requests.post(f"{cfg.SUCTION_BASE_URL}/run/3587", timeout=5.0)
    logger.info("[Grasp] Suction ON")


def _suction_off() -> None:
    _stop_processes()
    time.sleep(0.5)
    requests.post(f"{cfg.SUCTION_BASE_URL}/run/763", timeout=5.0)
    logger.info("[Grasp] Suction OFF")


def _blow_on() -> None:
    _stop_processes()
    time.sleep(0.5)
    requests.post(f"{cfg.SUCTION_BASE_URL}/run/7381", timeout=5.0)


def _blow_off() -> None:
    _stop_processes()
    time.sleep(0.5)
    requests.post(f"{cfg.SUCTION_BASE_URL}/run/5484", timeout=5.0)


# ---------------------------------------------------------------------------
# IK utility
# ---------------------------------------------------------------------------

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


# ---------------------------------------------------------------------------
# Grasper
# ---------------------------------------------------------------------------

class SuctionGrasper:
    """Execute a suction grasp for a single BatteryObservation.

    The robot model (URDF + pinocchio) is loaded once in ``__init__``.
    Call ``grasp()`` for each pick, ``release()`` at the place location,
    and ``close()`` (or use as a context manager) for cleanup.

    Args:
        robot: Live Robot instance already connected to hardware.
    """

    def __init__(self, robot) -> None:
        self._robot = robot
        self._setup_ik()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SuctionGrasper":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Ensure suction is off on teardown."""
        try:
            _suction_off()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grasp(self, obs: BatteryObservation) -> GraspResult:
        """Grasp the battery described by *obs*.

        Steps:
            1. Compute hover pose above battery (suction_length + hover_height).
            2. Move arm to hover pose via IK interpolation.
            3. Activate suction + start vacuum monitor.
            4. Tare wrist force sensor.
            5. Descend in small steps, monitoring vacuum and force.
            6. Return GraspResult.

        Args:
            obs: BatteryObservation from the perception module.

        Returns:
            GraspResult with success status and contact pose.
        """
        # 1. Hover position: battery top + suction length + hover clearance
        hover_pos = obs.position_base.copy()
        hover_pos[2] += cfg.SUCTION_LENGTH_M + cfg.HOVER_HEIGHT_M
        target_rpy = np.array(cfg.GRASP_ORIENTATION_RPY)

        # 2. Seed IK from live robot state
        q_live = self._seed_from_robot()
        configuration = pink.Configuration(self._model, self._robot_pin.data, q_live)

        # 3. Solve and move to hover
        logger.info("[Grasp] Moving to hover pos ({:.3f}, {:.3f}, {:.3f})", *hover_pos)
        q_hover, ok = self._solve_ik(configuration, hover_pos, target_rpy)
        if not ok:
            logger.warning("[Grasp] IK for hover did not fully converge — proceeding")
        self._move_to_joints(q_hover, duration=cfg.MOVE_DURATION_S)

        # 4. Activate suction + start vacuum monitor
        _suction_on()
        vac = _VacuumMonitor(cfg.SUCTION_HOST)
        vac.start()

        # 5. Tare force sensor at hover (hands free of battery)
        tare_force(cfg.ARM_SIDE, self._robot)

        # 6. Descent loop
        result = self._descent_loop(hover_pos, target_rpy, configuration, vac)

        vac.stop()
        if not result.success:
            _suction_off()
        return result

    def release(self) -> None:
        """Release the gripped battery with a short blow pulse."""
        _suction_off()
        time.sleep(0.1)
        _blow_on()
        time.sleep(0.3)
        _blow_off()
        logger.info("[Grasp] Battery released")

    # ------------------------------------------------------------------
    # IK setup
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        urdf = cfg.URDF_PATH
        self._robot_pin = pin.RobotWrapper.BuildFromURDF(
            filename=urdf,
            package_dirs=[
                os.path.dirname(urdf),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf)))),
            ],
            root_joint=None,
        )
        self._model = self._robot_pin.model

        # Joint index maps
        self._torso_indices = [
            self._model.getJointId(f"torso_j{j + 1}") for j in range(3)
        ]
        arm_prefix = "R" if cfg.ARM_SIDE == "right" else "L"
        self._arm_indices = [
            self._model.getJointId(f"{arm_prefix}_arm_j{j + 1}") for j in range(7)
        ]

        # IK tasks (reused across calls)
        self._ee_task = FrameTask(cfg.EE_FRAME, position_cost=2.0, orientation_cost=1.0)
        self._posture_task = PostureTask(cost=1e-3)

        import qpsolvers as _qp
        preferred = cfg.PREFERRED_QP_SOLVER
        self._solver = preferred if preferred in _qp.available_solvers else _qp.available_solvers[0]
        logger.info("[Grasp] IK ready — EE={}, solver={}", cfg.EE_FRAME, self._solver)

    # ------------------------------------------------------------------
    # IK helpers
    # ------------------------------------------------------------------

    def _seed_from_robot(self) -> np.ndarray:
        """Build a pinocchio q vector seeded from the live robot state."""
        q = pin.neutral(self._model)
        torso_q = self._robot.torso.get_joint_pos().astype(float)
        arm = self._robot.right_arm if cfg.ARM_SIDE == "right" else self._robot.left_arm
        arm_q = arm.get_joint_pos().astype(float)

        for j, idx in enumerate(self._torso_indices):
            q[self._model.idx_qs[idx]] = torso_q[j]
        for j, idx in enumerate(self._arm_indices):
            q[self._model.idx_qs[idx]] = arm_q[j]

        return np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)

    def _solve_ik(
        self,
        configuration: pink.Configuration,
        target_pos: np.ndarray,
        target_rpy: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Solve IK for *target_pos* / *target_rpy* (warm-start from *configuration*)."""
        rotation = _rpy_to_matrix(*target_rpy)
        self._ee_task.set_target(pin.SE3(rotation, target_pos))
        self._posture_task.set_target(configuration.q.copy())
        tasks = [self._ee_task, self._posture_task]

        for _ in range(cfg.IK_MAX_ITERS):
            velocity = solve_ik(configuration, tasks, cfg.IK_DT, solver=self._solver)
            q_next = pin.integrate(self._model, configuration.q, velocity * cfg.IK_DT)
            configuration.update(q_next)
            if np.linalg.norm(self._ee_task.compute_error(configuration)) < cfg.IK_CONVERGENCE_THRESHOLD:
                return configuration.q.copy(), True

        logger.warning(
            "[Grasp] IK did not converge (err={:.4f})",
            float(np.linalg.norm(self._ee_task.compute_error(configuration))),
        )
        return configuration.q.copy(), False

    def _arm_joints_from_q(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[self._model.idx_qs[i]] for i in self._arm_indices])

    def _move_to_joints(self, target_q: np.ndarray, duration: float) -> None:
        """Linearly interpolate the arm from its current position to *target_q*."""
        arm = self._robot.right_arm if cfg.ARM_SIDE == "right" else self._robot.left_arm
        current_q = arm.get_joint_pos().astype(float)
        target_arm_q = self._arm_joints_from_q(target_q)

        n_steps = max(1, int(duration / cfg.CONTROL_DT))
        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            arm.set_joint_pos(current_q + alpha * (target_arm_q - current_q))
            time.sleep(cfg.CONTROL_DT)

    # ------------------------------------------------------------------
    # Descent loop
    # ------------------------------------------------------------------

    def _descent_loop(
        self,
        hover_pos: np.ndarray,
        target_rpy: np.ndarray,
        configuration: pink.Configuration,
        vac: _VacuumMonitor,
    ) -> GraspResult:
        arm = self._robot.right_arm if cfg.ARM_SIDE == "right" else self._robot.left_arm
        n_steps = int(cfg.MAX_DESCENT_M / cfg.DESCENT_STEP_M)
        target_pos = hover_pos.copy()

        for step in range(n_steps):
            target_pos[2] = hover_pos[2] - cfg.DESCENT_STEP_M * (step + 1)

            # Solve IK for this descent step (warm-started from previous)
            q_step, _ = self._solve_ik(configuration, target_pos, target_rpy)
            arm.set_joint_pos(self._arm_joints_from_q(q_step))
            time.sleep(cfg.DESCENT_DT_S)

            # — Primary: vacuum seal —
            if vac.is_sealed():
                logger.info(
                    "[Grasp] Vacuum sealed at step {} / z={:.4f}m",
                    step, float(target_pos[2]),
                )
                return GraspResult(
                    success=True,
                    contact_position_base=target_pos.copy(),
                    trigger="vacuum",
                )

            # — Force checks —
            force = get_force(cfg.ARM_SIDE, self._robot)
            if force is None:
                continue

            if force > cfg.FORCE_HARD_LIMIT_N:
                logger.warning(
                    "[Grasp] Hard force limit {:.1f}N > {:.1f}N — aborting",
                    force, cfg.FORCE_HARD_LIMIT_N,
                )
                return GraspResult(
                    success=False,
                    contact_position_base=target_pos.copy(),
                    trigger="force_limit",
                )

            if force > cfg.FORCE_CONTACT_THRESHOLD_N:
                logger.info(
                    "[Grasp] Contact force {:.1f}N at z={:.4f}m — waiting for vacuum seal",
                    force, float(target_pos[2]),
                )
                sealed = vac.wait_for_seal(timeout=cfg.VACUUM_SEAL_TIMEOUT_S)
                return GraspResult(
                    success=sealed,
                    contact_position_base=target_pos.copy(),
                    trigger="force+vacuum" if sealed else "vacuum_timeout",
                )

        logger.warning("[Grasp] Max descent ({:.3f}m) reached without contact", cfg.MAX_DESCENT_M)
        return GraspResult(success=False, contact_position_base=None, trigger="max_descent")
