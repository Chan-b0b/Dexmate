"""Single-arm suction pick-and-place mover (base_link frame).

Adapts the proven vacuum-seal descent loop from battery_pick/suction_grasp.py
into a pose-driven primitive:

    mover.pick(pose)      # hover -> suction on -> descend until seal/contact
    mover.lift()          # raise straight up to SAFE_TRANSPORT_Z
    mover.move_to(pose)   # travel sideways to above a place pose
    mover.place(pose)     # controlled descent to taught z -> release -> retract

All poses are cup-tip [x, y, z] targets in the robot base_link frame; the
arm's down-pointing approach orientation comes from cfg.GRASP_ORIENTATION_RPY.
The taught z only needs to be approximately right for picks — the vacuum seal
stops the descent.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
import pink
from loguru import logger
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation

from . import config as cfg
from . import suction_io

# read_force lives in grasp_box/
_GRASP_BOX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "grasp_box")
if _GRASP_BOX_DIR not in sys.path:
    sys.path.insert(0, _GRASP_BOX_DIR)
from read_force import get_force, tare_force  # noqa: E402


# ---------------------------------------------------------------------------
# Pose + result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """A cup-tip target in base_link frame."""

    pos: np.ndarray                                  # [x, y, z] (m)
    rpy: np.ndarray = field(                         # approach orientation (rad)
        default_factory=lambda: np.array(cfg.GRASP_ORIENTATION_RPY)
    )

    @classmethod
    def from_xyz(cls, xyz) -> "Pose":
        return cls(pos=np.asarray(xyz, dtype=float))


@dataclass
class PickResult:
    success: bool
    contact_position_base: np.ndarray | None
    trigger: str  # "vacuum" | "force+vacuum" | "vacuum_timeout" | "force_limit" | "max_descent"


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


# ---------------------------------------------------------------------------
# Mover
# ---------------------------------------------------------------------------

class SuctionMover:
    """IK-driven suction pick-and-place for a single arm.

    The URDF/pinocchio model is loaded once in ``__init__``. Use as a context
    manager so suction is always turned off on teardown.
    """

    def __init__(self, robot) -> None:
        self._robot = robot
        self._setup_ik()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SuctionMover":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        try:
            suction_io.suction_off()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Readiness (arm must be in position mode to accept set_joint_pos)
    # ------------------------------------------------------------------

    def software_estop_active(self) -> bool:
        """True if the software E-Stop is engaged (arms won't move)."""
        estop = getattr(self._robot, "estop", None)
        return bool(estop is not None and estop.is_software_estop_enabled())

    def enable_position_mode(self) -> None:
        """Put the suction arm into position control mode.

        Robot() init only does this if the software E-Stop was clear at
        startup, so call it after releasing the E-Stop (or any time the arm
        may have been left disabled)."""
        self._arm.set_modes(["position"] * 7)
        logger.info("[Mover] {} arm set to position mode", cfg.ARM_SIDE)

    def ensure_ready(self, release_estop: bool = False) -> bool:
        """Make the suction arm ready to accept motion commands.

        Returns True if ready. If the software E-Stop is active the arm cannot
        move; with release_estop=True it is deactivated first (caller is
        responsible for confirming this is safe)."""
        if self.software_estop_active():
            logger.warning("[Mover] Software E-Stop is ACTIVE — arm will not move.")
            if not release_estop:
                logger.warning(
                    "[Mover] Release it (physical reset or bot.estop.deactivate()) and retry."
                )
                return False
            logger.warning("[Mover] Deactivating software E-Stop...")
            self._robot.estop.deactivate()
            import time
            time.sleep(0.5)
        self.enable_position_mode()
        return not self.software_estop_active()

    # ------------------------------------------------------------------
    # IK setup
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        urdf = cfg.URDF_PATH
        robot_pin = pin.RobotWrapper.BuildFromURDF(
            filename=urdf,
            package_dirs=[
                os.path.dirname(urdf),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf)))),
            ],
            root_joint=None,
        )
        full_model = robot_pin.model

        # Reduce the model to ONLY the 7 suction-arm joints: lock every other
        # joint (torso, head, other arm, hands). The torso is locked at its live
        # (held-fixed) angles since it moves the arm base; the rest don't affect
        # this arm's EE so they're locked at neutral (harmless). Result: every IK
        # solution is exactly achievable by the arm alone — the solver can't
        # reduce EE error using DOFs we never command (which previously let the
        # cup drift off vertical).
        arm_prefix = "R" if cfg.ARM_SIDE == "right" else "L"
        arm_names = {f"{arm_prefix}_arm_j{j + 1}" for j in range(7)}
        q_ref = pin.neutral(full_model)
        torso_q = self._robot.torso.get_joint_pos().astype(float)
        for j in range(3):
            jid = full_model.getJointId(f"torso_j{j + 1}")
            q_ref[full_model.idx_qs[jid]] = torso_q[j]
        lock_ids = [jid for jid in range(1, full_model.njoints)
                    if full_model.names[jid] not in arm_names]
        self._model = pin.buildReducedModel(full_model, lock_ids, q_ref)
        self._data = self._model.createData()

        self._arm_indices = [self._model.getJointId(f"{arm_prefix}_arm_j{j + 1}") for j in range(7)]

        # FrameTask with no root => target in world (base_link) frame.
        self._ee_task = FrameTask(
            cfg.EE_FRAME, position_cost=2.0, orientation_cost=1.0, lm_damping=cfg.IK_LM_DAMPING
        )
        # Nullspace centering: pull the redundant DOF toward joint mid-ranges
        # (auto-computed from URDF limits) so joints stay away from motor stops.
        self._posture_task = PostureTask(cost=cfg.POSTURE_COST)
        mid = 0.5 * (self._model.lowerPositionLimit + self._model.upperPositionLimit)
        self._posture_target = np.where(np.isfinite(mid), mid, 0.0)
        self._posture_task.set_target(self._posture_target)

        # Keep every solve inside joint position and velocity limits.
        self._limits = [ConfigurationLimit(self._model), VelocityLimit(self._model)]

        import qpsolvers as _qp
        preferred = cfg.PREFERRED_QP_SOLVER
        self._solver = preferred if preferred in _qp.available_solvers else _qp.available_solvers[0]
        logger.info(
            "[Mover] IK ready — EE={}, solver={}, arm-only DOFs={} (torso locked)",
            cfg.EE_FRAME, self._solver, self._model.nq,
        )

    @property
    def _arm(self):
        return self._robot.right_arm if cfg.ARM_SIDE == "right" else self._robot.left_arm

    def _fresh_configuration(self) -> pink.Configuration:
        """Build a pink.Configuration seeded from the live arm state (torso locked)."""
        q = pin.neutral(self._model)
        arm_q = self._arm.get_joint_pos().astype(float)
        for j, idx in enumerate(self._arm_indices):
            q[self._model.idx_qs[idx]] = arm_q[j]
        q = np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        return pink.Configuration(self._model, self._data, q)

    def _solve_ik(self, configuration, target_pos, target_rpy) -> tuple[np.ndarray, bool]:
        rotation = _rpy_to_matrix(*target_rpy)
        self._ee_task.set_target(pin.SE3(rotation, np.asarray(target_pos, dtype=float)))
        # Posture target stays at the joint mid-ranges (set in _setup_ik) for
        # limit avoidance — do NOT retarget it to the current config here.
        tasks = [self._ee_task, self._posture_task]
        for _ in range(cfg.IK_MAX_ITERS):
            velocity = solve_ik(
                configuration, tasks, cfg.IK_DT, solver=self._solver, limits=self._limits
            )
            q_next = pin.integrate(self._model, configuration.q, velocity * cfg.IK_DT)
            configuration.update(q_next)
            if np.linalg.norm(self._ee_task.compute_error(configuration)) < cfg.IK_CONVERGENCE_THRESHOLD:
                return configuration.q.copy(), True
        err = float(np.linalg.norm(self._ee_task.compute_error(configuration)))
        # logger.warning("[Mover] IK did not converge (err={:.4f})", err)
        return configuration.q.copy(), False

    def _arm_joints_from_q(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[self._model.idx_qs[i]] for i in self._arm_indices])

    def current_ee_position(self) -> np.ndarray:
        """Public accessor for the live EE-frame position in base_link (m)."""
        return self._current_ee_pos()

    def current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Live EE pose in base_link: (position [x,y,z], rpy [r,p,y])."""
        cfg_now = self._fresh_configuration()
        pin.framesForwardKinematics(self._model, self._data, cfg_now.q)
        T = self._data.oMf[self._model.getFrameId(cfg.EE_FRAME)]
        rpy = Rotation.from_matrix(T.rotation).as_euler("xyz")
        return T.translation.copy(), rpy

    def goto(self, pos, rpy, step_duration: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Move the EE to an absolute base_link pose (pos [m], rpy [rad]).

        Teaching/helper: the caller tracks the commanded target so orientation
        is always held to an explicit value (e.g. the vertical approach), not
        whatever the arm happens to be at. Returns the achieved (pos, rpy)."""
        self._move_ee_to(np.asarray(pos, dtype=float), np.asarray(rpy, dtype=float), step_duration)
        return self.current_ee_pose()

    def _current_ee_pos(self) -> np.ndarray:
        """Cup-tip... actually EE-frame position in base_link, from live FK."""
        cfg_now = self._fresh_configuration()
        pin.framesForwardKinematics(self._model, self._data, cfg_now.q)
        fid = self._model.getFrameId(cfg.EE_FRAME)
        return self._data.oMf[fid].translation.copy()

    def _move_to_joints(self, target_q: np.ndarray, duration: float) -> None:
        """Smoothly interpolate the arm to *target_q* (arm-joint vector)."""
        current_q = self._arm.get_joint_pos().astype(float)
        target_arm_q = self._arm_joints_from_q(target_q)
        n_steps = max(1, int(duration / cfg.CONTROL_DT))
        import time
        for step in range(n_steps):
            t = (step + 1) / n_steps
            alpha = t * t * (3 - 2 * t)  # smoothstep
            self._arm.set_joint_pos(current_q + alpha * (target_arm_q - current_q))
            time.sleep(cfg.CONTROL_DT)

    def _move_ee_to(self, target_pos, target_rpy, duration: float) -> bool:
        """Solve IK for an absolute base-frame target and move there smoothly."""
        configuration = self._fresh_configuration()
        q_target, ok = self._solve_ik(configuration, target_pos, target_rpy)
        if not ok:
            logger.warning("[Mover] proceeding with non-converged IK to {}", np.round(target_pos, 3))
        self._move_to_joints(q_target, duration)
        return ok

    def _cartesian_z_to(self, target_z: float, rpy) -> None:
        """Step the EE straight to *target_z* in Cartesian space (up or down).

        Unlike ``_move_ee_to`` (which interpolates in joint space and can let
        the EE dip then rise on a long descent because the elbow swings), this
        steps the cup tip monotonically along Z while keeping x, y and the
        commanded rpy fixed. Use it for hover-descent legs so the user never
        sees an unintended pre-descent lift caused by joint-space interp.
        """
        import time
        cur_pos, _ = self.current_ee_pose()
        target_z = float(target_z)
        if abs(target_z - cur_pos[2]) < 1e-4:
            return
        direction = 1.0 if target_z > cur_pos[2] else -1.0
        configuration = self._fresh_configuration()
        step_pos = cur_pos.copy()
        while (direction > 0 and step_pos[2] < target_z) or \
              (direction < 0 and step_pos[2] > target_z):
            remaining = target_z - step_pos[2]
            step = direction * min(cfg.LIFT_STEP_M, abs(remaining))
            step_pos[2] += step
            q_step, _ = self._solve_ik(configuration, step_pos, rpy)
            self._arm.set_joint_pos(self._arm_joints_from_q(q_step))
            time.sleep(cfg.DESCENT_DT_S)

    # ------------------------------------------------------------------
    # Public primitives
    # ------------------------------------------------------------------

    def pick(self, pose: Pose) -> PickResult:
        """Hover above *pose*, activate suction, descend until seal/contact."""
        hover_pos = pose.pos.copy()
        hover_pos[2] += cfg.HOVER_HEIGHT_M
        rpy = pose.rpy

        # Two-leg approach: first travel sideways at SAFE_TRANSPORT_Z, then
        # drop straight down to hover_z. Avoids the cup sweeping diagonally
        # through the box rim or an adjacent stack.
        if cfg.SAFE_TRANSPORT_Z is not None:
            transit = np.array([hover_pos[0], hover_pos[1], cfg.SAFE_TRANSPORT_Z])
            logger.info(
                "[Mover] pick: transit ({:.3f}, {:.3f}) @ z={:.3f}",
                transit[0], transit[1], transit[2],
            )
            self._move_ee_to(transit, rpy, cfg.MOVE_DURATION_S)

        logger.info("[Mover] pick: hover at ({:.3f}, {:.3f}, {:.3f})", *hover_pos)
        # Cartesian Z descent so the EE moves monotonically down to hover_z.
        # Joint-space interp here can let the elbow swing and the tip dip
        # below hover_z then rise back up, which looks like a "lift" right
        # before the descent loop starts.
        self._cartesian_z_to(hover_pos[2], rpy)

        suction_io.suction_on()
        vac = suction_io.VacuumMonitor()
        vac.start()
        tare_force(cfg.ARM_SIDE, self._robot)  # hands free at hover

        result = self._descent_loop(hover_pos, pose.pos[2], rpy, vac)
        vac.stop()
        if not result.success:
            suction_io.suction_off()
        return result

    def lift(self, z: float | None = None) -> None:
        """Raise the EE straight up in Cartesian Z steps, holding x, y and orientation fixed."""
        import time
        z = cfg.SAFE_TRANSPORT_Z if z is None else z
        if z is None:
            raise ValueError("SAFE_TRANSPORT_Z is not set — teach it before running")
        cur_pos, cur_rpy = self.current_ee_pose()
        target_z = float(z)
        logger.info("[Mover] lift from z={:.3f} to z={:.3f}", cur_pos[2], target_z)

        configuration = self._fresh_configuration()
        step_pos = cur_pos.copy()
        while step_pos[2] < target_z:
            step_pos[2] = min(step_pos[2] + cfg.LIFT_STEP_M, target_z)
            q_step, _ = self._solve_ik(configuration, step_pos, cur_rpy)
            self._arm.set_joint_pos(self._arm_joints_from_q(q_step))
            time.sleep(cfg.DESCENT_DT_S)

    def move_to(self, pose: Pose) -> None:
        """Travel sideways to hover above *pose* at SAFE_TRANSPORT_Z."""
        if cfg.SAFE_TRANSPORT_Z is None:
            raise ValueError("SAFE_TRANSPORT_Z is not set — teach it before running")
        target = np.array([pose.pos[0], pose.pos[1], cfg.SAFE_TRANSPORT_Z])
        logger.info("[Mover] move_to ({:.3f}, {:.3f}) @ transport z", target[0], target[1])
        self._move_ee_to(target, pose.rpy, cfg.MOVE_DURATION_S)
        self._wait_until_arrived(target, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

    def _wait_until_arrived(self, target_pos, tol_m: float, timeout_s: float) -> None:
        """Block until the live EE position is within *tol_m* of *target_pos*.

        Position-mode commands are accepted asynchronously, so the arm can
        still be settling toward the target after our smoothstep loop
        finishes. Subsequent legs (e.g. the vertical descent in pick/place)
        must not start until the sideways travel has actually completed,
        otherwise the cup descends from the wrong (x, y).
        """
        import time
        target = np.asarray(target_pos, dtype=float)
        deadline = time.time() + float(timeout_s)
        last_err = None
        while time.time() < deadline:
            err = float(np.linalg.norm(self._current_ee_pos() - target))
            last_err = err
            if err <= tol_m:
                return
            time.sleep(0.02)
        logger.warning(
            "[Mover] move_to: arrival not confirmed within {:.1f}s "
            "(remaining error {:.4f}m, tol {:.4f}m)",
            timeout_s, last_err if last_err is not None else float("nan"), tol_m,
        )

    def place(self, pose: Pose) -> None:
        """Descend to the taught place z, release, then retract to transport z."""
        # Approach hover just above the place target.
        hover = pose.pos.copy()
        hover[2] += cfg.HOVER_HEIGHT_M
        # The orchestrator's preceding move_to() already left us at
        # (x, y, SAFE_TRANSPORT_Z). This is the vertical-only descent leg
        # to hover_z — same two-leg pattern used in pick().
        logger.info("[Mover] place: hover at ({:.3f}, {:.3f}, {:.3f})", *hover)
        self._move_ee_to(hover, pose.rpy, cfg.APPROACH_DESCENT_S)

        # Re-tare the wrench while hovering with the battery still attached.
        # Without this, the force baseline left over from the empty-cup tare
        # at pick() time means the battery's own weight reads as a large
        # apparent force and instantly trips FORCE_HARD_LIMIT_N during the
        # place descent (especially noticeable on undo moves where the arm
        # has been carrying a battery the whole sequence).
        import time
        time.sleep(0.5)  # let the arm settle before sampling baseline
        tare_force(cfg.ARM_SIDE, self._robot)
        # Sanity-check: immediately after tare with no contact, force should
        # read close to zero. If it doesn't, the tare didn't take and the
        # next descent step will trip the hard-force limit on the first
        # sample. Surface that explicitly so it's diagnosable from logs.
        f_after_tare = get_force(cfg.ARM_SIDE, self._robot)
        if f_after_tare is None:
            logger.warning("[Mover] place: wrench sensor unavailable after tare")
        elif f_after_tare > cfg.FORCE_CONTACT_THRESHOLD_N:
            logger.warning(
                "[Mover] place: post-tare force is {:.1f}N (expected ~0). "
                "Wrench baseline may be unstable; descent may abort early.",
                f_after_tare,
            )
        else:
            logger.info("[Mover] place: post-tare force {:.2f}N (baseline ok)", f_after_tare)

        # Controlled descent to the taught seat height with a hard-force guard.
        self._descend_to(pose.pos, pose.rpy)

        # Small pre-release lift so the cup is clear of the placed object
        # before the blow pulse fires — prevents sticking and avoids
        # blowback pushing the part sideways.
        if cfg.RELEASE_PRELIFT_M and cfg.RELEASE_PRELIFT_M > 0.0:
            cur_pos, _ = self.current_ee_pose()
            prelift_z = float(cur_pos[2]) + float(cfg.RELEASE_PRELIFT_M)
            logger.info("[Mover] pre-release lift +{:.0f}mm", cfg.RELEASE_PRELIFT_M * 1000)
            self.lift(prelift_z)

        suction_io.release()

        if cfg.SAFE_TRANSPORT_Z is not None:
            self.lift(cfg.SAFE_TRANSPORT_Z)

    # ------------------------------------------------------------------
    # Descent helpers
    # ------------------------------------------------------------------

    def _descent_loop(self, hover_pos, target_z: float, rpy, vac: suction_io.VacuumMonitor) -> PickResult:
        """Step down from hover until vacuum seal (primary) or force (fallback).

        Empirical toolA profile (from monitor_current.py traces):
            Pump ON, unsealed (free air OR pressing without seal): ~0.012 A
            Pump ON, sealed (pump unloads):                         ~0.006 A
            Pump OFF:                                               ~0.003-0.009 A
        So the seal signal is the DROP from ~0.012 → <0.008 A while the pump
        was running. There is no "rising approach" phase to detect — the cup
        either seals (current drops) or it doesn't (current stays high while
        we keep pushing). This loop watches the latched seal event AND the
        live current drop, and on any stop event re-commands the current pose
        to halt the arm immediately (otherwise it would keep tracking the
        last commanded step which is below the contact point).
        """
        import time
        configuration = self._fresh_configuration()
        current_pos = np.asarray(hover_pos, dtype=float).copy()
        descended = 0.0

        def _halt() -> None:
            """Re-command the live EE pose so the arm stops where it is."""
            live_pos = self._current_ee_pos()
            hold_q, _ = self._solve_ik(self._fresh_configuration(), live_pos, rpy)
            self._arm.set_joint_pos(self._arm_joints_from_q(hold_q))

        while descended < cfg.MAX_DESCENT_M:
            step = float(np.clip((current_pos[2] - target_z) * cfg.DESCENT_KP,
                                 cfg.DESCENT_MIN_STEP_M, cfg.DESCENT_MAX_STEP_M))
            current_pos[2] -= step
            descended += step
            q_step, _ = self._solve_ik(configuration, current_pos, rpy)
            self._arm.set_joint_pos(self._arm_joints_from_q(q_step))
            time.sleep(cfg.DESCENT_DT_S)

            tool_a = vac.get_tool_current()

            # PRIMARY: vacuum seal detected by VacuumMonitor (DI0 == T while
            # suction commanded ON). Halt the arm immediately on detection.
            if vac.is_sealed():
                _halt()
                live_pos = self._current_ee_pos()
                logger.info(
                    "[Mover] vacuum sealed at z={:.4f}m (toolA={:.4f}A) — HALT",
                    float(live_pos[2]), tool_a,
                )
                return PickResult(True, live_pos, "vacuum")

            force = get_force(cfg.ARM_SIDE, self._robot)
            if force is None:
                continue
            if force > cfg.FORCE_HARD_LIMIT_N:
                _halt()
                logger.warning("[Mover] hard force limit {:.1f}N — aborting", force)
                return PickResult(False, current_pos.copy(), "force_limit")
            if force > cfg.FORCE_CONTACT_THRESHOLD_N:
                # Cup is touching but not yet sealed — STOP descent and hold
                # position while the pump pulls vacuum. Without this halt the
                # arm keeps tracking the last commanded step and over-drives
                # the battery while we wait for seal.
                _halt()
                logger.info("[Mover] contact {:.1f}N at z={:.4f} — HALT, waiting for seal",
                            force, float(current_pos[2]))
                # Keep the arm pinned to the live pose for the entire wait so
                # it can't drift further into the battery while the seal forms.
                deadline = time.time() + cfg.VACUUM_SEAL_TIMEOUT_S
                sealed = False
                while time.time() < deadline:
                    _halt()
                    if vac.is_sealed():
                        sealed = True
                        break
                    f = get_force(cfg.ARM_SIDE, self._robot)
                    if f is not None and f > cfg.FORCE_HARD_LIMIT_N:
                        logger.warning("[Mover] hard force {:.1f}N during seal wait — aborting", f)
                        return PickResult(False, current_pos.copy(), "force_limit")
                    time.sleep(0.05)

                if sealed:
                    live_pos = self._current_ee_pos()
                    logger.info(
                        "[Mover] seal confirmed after contact at z={:.4f}m (toolA={:.4f}A)",
                        float(live_pos[2]), vac.get_tool_current(),
                    )
                    return PickResult(True, live_pos, "force+vacuum")

                # No real seal formed — DO NOT lift the battery. Force alone
                # is not proof of vacuum (cup can be pressed on a leaky surface
                # and still feel solid). Treat as a failed pick.
                logger.error(
                    "[Mover] seal NOT confirmed within {:.1f}s (toolA={:.4f}A) — pick FAILED",
                    cfg.VACUUM_SEAL_TIMEOUT_S,
                    vac.get_tool_current(),
                )
                return PickResult(False, current_pos.copy(), "vacuum_timeout")

        logger.warning("[Mover] max descent reached without contact")
        return PickResult(False, None, "max_descent")

    def _descend_to(self, target_pos, rpy) -> None:
        """Constant-velocity descent that stops at the taught goal_z.

        Stop conditions, in priority order:
          1. FORCE_HARD_LIMIT_PLACE_N exceeded -> emergency stop (safety).
          2. current_z <= goal_z                -> normal end-of-descent
             (we reached the taught seat; release the battery here).
          3. force > FORCE_CONTACT_THRESHOLD_N  AND
             current_z <= goal_z + PLACE_Z_BUFFER_M
                                                 -> early contact within
             buffer; treat as good seat and release.
          4. descended > MAX_DESCENT_M           -> never reached goal_z;
             abort.

        Previous behaviour kept descending past goal_z while waiting for a
        contact force >= 2 N. With a battery already in the cup, the wrench
        reading typically jumps from ~0 N straight to 20-30 N in a single
        sample when the battery hits the slot floor (no gradual ramp), so
        we'd routinely overshoot goal_z and trip the hard limit on impact.
        Trusting the taught goal_z avoids that.
        """
        import time
        configuration = self._fresh_configuration()
        current_z = self._current_ee_pos()[2]
        goal_z = float(target_pos[2])
        base_x, base_y = float(target_pos[0]), float(target_pos[1])
        step_pos = np.array([base_x, base_y, current_z])
        descended = 0.0
        step_count = 0
        f_max_seen = 0.0  # diagnostic: peak force across the descent
        while descended < cfg.MAX_DESCENT_M:
            # Reached the taught seat -> stop here (primary stop condition).
            if current_z <= goal_z:
                logger.info(
                    "[Mover] place: reached goal_z={:.4f} (descended {:.3f}m, "
                    "peak force {:.1f}N) — releasing",
                    goal_z, descended, f_max_seen,
                )
                return

            # Clamp the next step so we never go below goal_z in one move.
            raw_step = float(np.clip((current_z - goal_z) * cfg.PLACE_DESCENT_KP,
                                     cfg.PLACE_DESCENT_MIN_STEP_M,
                                     cfg.PLACE_DESCENT_MAX_STEP_M))
            step = min(raw_step, current_z - goal_z)
            phase = 2 * np.pi * step_count * cfg.PLACE_DESCENT_DT_S * cfg.JITTER_FREQ_HZ
            # Ramp jitter amplitude in from 0 over the first JITTER_RAMP_S
            # seconds of descent so the Y/roll/pitch terms (which use cos /
            # sin(+π/2) and start at full amplitude) don't snap on step 0.
            # Without this ramp the cup visibly flicks at the start of every
            # place descent because the commanded orientation jumps
            # discontinuously from the hover pose.
            ramp_s = max(cfg.JITTER_RAMP_S, 1e-6)
            ramp = min(1.0, step_count * cfg.PLACE_DESCENT_DT_S / ramp_s)
            step_pos[0] = base_x + ramp * cfg.JITTER_AMPLITUDE_M * np.sin(phase)
            step_pos[1] = base_y + ramp * cfg.JITTER_AMPLITUDE_M * np.cos(phase)
            step_pos[2] = current_z - step
            # Same-frequency yaw wobble so the cup wiggles slightly while it
            # descends, helping the battery seat into the slot if X/Y jitter
            # alone isn't enough to clear a tight fit. Roll/pitch are added
            # at phase offsets so the cup nutates instead of just twisting.
            step_rpy = np.array(rpy, dtype=float, copy=True)
            step_rpy[0] += ramp * cfg.JITTER_ROLL_AMPLITUDE_RAD * np.cos(phase)
            step_rpy[1] += ramp * cfg.JITTER_PITCH_AMPLITUDE_RAD * np.sin(phase + np.pi / 2)
            step_rpy[2] += ramp * cfg.JITTER_YAW_AMPLITUDE_RAD * np.sin(phase)
            q_step, _ = self._solve_ik(configuration, step_pos, step_rpy)
            self._arm.set_joint_pos(self._arm_joints_from_q(q_step))
            current_z = step_pos[2]
            descended += step
            step_count += 1
            time.sleep(cfg.PLACE_DESCENT_DT_S)

            force = get_force(cfg.ARM_SIDE, self._robot)
            if force is None:
                continue
            if force > f_max_seen:
                f_max_seen = force
            if force > cfg.FORCE_HARD_LIMIT_PLACE_N:
                logger.warning(
                    "[Mover] place: hard force {:.1f}N at z={:.4f} "
                    "(descended {:.3f}m, peak {:.1f}N) — emergency stop",
                    force, current_z, descended, f_max_seen,
                )
                return
            if force > cfg.FORCE_CONTACT_THRESHOLD_N and \
                    current_z <= goal_z + cfg.PLACE_Z_BUFFER_M:
                logger.info(
                    "[Mover] place: early contact {:.1f}N at z={:.4f} within buffer — releasing",
                    force, current_z,
                )
                return

        logger.warning(
            "[Mover] place: max descent reached without reaching goal_z={:.4f} "
            "(current z={:.4f}, peak force seen {:.1f}N)",
            goal_z, current_z, f_max_seen,
        )
