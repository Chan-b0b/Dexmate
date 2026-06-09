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


def _ease(t: float, profile: str = "quintic") -> float:
    """Return the smoothstep-eased parameter for t in [0, 1].

    Profiles:
      cubic     : 3t² - 2t³                (C¹, zero vel at both endpoints)
      quintic   : 6t⁵ - 15t⁴ + 10t³        (C², zero vel + accel at both endpoints)
      ease_in   : t²                       (zero vel at start, full slope at end)
      ease_out  : 2t - t²                  (full slope at start, zero vel at end)

    Quintic is the safe default for a complete A→B move (gentlest on the
    motors). Use ease_in when the move is followed immediately by another
    moving leg (e.g. pick's hover descent feeding into the contact-search
    descent loop) so the EE doesn't decelerate to a stop and re-accelerate
    — that boundary is what's felt as a stutter. Use ease_out for the
    inverse: a moving leg coming to rest at a target.
    """
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    if profile == "cubic":
        return t * t * (3.0 - 2.0 * t)
    if profile == "ease_in":
        return t * t
    if profile == "ease_out":
        return t * (2.0 - t)
    # default: quintic
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _shortest_rpy_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component-wise shortest angular delta from a to b, each in (-pi, pi].

    Used by the Cartesian-interpolated travel so a small yaw change doesn't
    accidentally interpolate the long way around the circle.
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return (d + np.pi) % (2.0 * np.pi) - np.pi


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
        self._trace_file = None
        if getattr(cfg, "TRACE_ENABLED", False):
            self._open_trace()

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
        if getattr(self, "_trace_file", None) is not None:
            try:
                self._trace_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._trace_file = None

    # ------------------------------------------------------------------
    # Per-tick motion trace (commanded vs actual joints) — debug aid
    # ------------------------------------------------------------------

    def _open_trace(self) -> None:
        path = getattr(cfg, "TRACE_PATH", "/tmp/cns_trace.csv")
        self._trace_file = open(path, "w")
        header = (["t", "leg"]
                  + [f"cmd_j{i + 1}" for i in range(7)]
                  + [f"act_j{i + 1}" for i in range(7)]
                  + ["ee_x", "ee_y", "ee_z"])  # true base-frame EE from actual joints
        self._trace_file.write(",".join(header) + "\n")
        logger.info("[Mover] motion trace -> {}", path)

    def _fk_arm(self, arm_q) -> np.ndarray:
        """True base-frame EE position for a 7-vector of arm joints.

        Uses the mover's reduced model (torso locked at its real live angle),
        so unlike a neutral-torso FK this is the correct base_link position."""
        q = pin.neutral(self._model)
        for k, i in enumerate(self._arm_indices):
            q[self._model.idx_qs[i]] = arm_q[k]
        pin.framesForwardKinematics(self._model, self._data, q)
        return self._data.oMf[self._model.getFrameId(cfg.EE_FRAME)].translation.copy()

    def _trace(self, leg: str, commanded_arm_q) -> None:
        """Append one row: time, leg, commanded 7 joints, actual 7 joints, and
        the true base-frame EE position (FK of the actual joints).

        Called right after each set_joint_pos so the actual reading is the
        live (lagging) state at that command instant. No-op when disabled."""
        if getattr(self, "_trace_file", None) is None:
            return
        import time
        actual = np.asarray(self._arm.get_joint_pos(), dtype=float)
        cmd = np.asarray(commanded_arm_q, dtype=float)
        ee = self._fk_arm(actual)
        row = [f"{time.time():.4f}", leg]
        row += [f"{v:.6f}" for v in cmd]
        row += [f"{v:.6f}" for v in actual]
        row += [f"{v:.6f}" for v in ee]
        self._trace_file.write(",".join(row) + "\n")

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
            alpha = _ease(t, getattr(cfg, "SMOOTH_PROFILE", "cubic"))
            cmd = current_q + alpha * (target_arm_q - current_q)
            self._arm.set_joint_pos(cmd)
            self._trace("move_joints", cmd)
            time.sleep(cfg.CONTROL_DT)

    def _move_ee_cartesian(self, target_pos, target_rpy, duration: float) -> bool:
        """Interpolate the EE pose linearly in Cartesian space over *duration*.

        Each substep advances the commanded EE pose along a straight line
        (with shortest-angle slerp on rpy) using a smoothstep time profile,
        then solves IK warm-started from the previous solution. This avoids
        the EE arcs / elbow swings that joint-space interpolation between two
        IK solutions can produce when the start and end joints differ by a
        lot.
        """
        import time
        start_pos, start_rpy = self.current_ee_pose()
        target_pos = np.asarray(target_pos, dtype=float)
        target_rpy = np.asarray(target_rpy, dtype=float)
        rpy_delta = _shortest_rpy_delta(start_rpy, target_rpy)

        dt = float(getattr(cfg, "EE_TRAVEL_DT_S", cfg.CONTROL_DT))
        n_steps = max(1, int(round(duration / dt)))
        profile = getattr(cfg, "SMOOTH_PROFILE", "quintic")

        # Warm-start IK from the live configuration; reuse the same
        # configuration across substeps so each solve seeds from the previous
        # arm pose (small per-step deltas -> tight convergence, no jumps).
        configuration = self._fresh_configuration()
        ok_all = True
        for step in range(1, n_steps + 1):
            alpha = _ease(step / n_steps, profile)
            step_pos = start_pos + alpha * (target_pos - start_pos)
            step_rpy = start_rpy + alpha * rpy_delta
            q_step, ok = self._solve_ik(configuration, step_pos, step_rpy)
            if not ok:
                ok_all = False
            cmd = self._arm_joints_from_q(q_step)
            self._arm.set_joint_pos(cmd)
            self._trace("move_cart", cmd)
            time.sleep(dt)
        # Real reachability check: FK of the *last commanded* q vs the target.
        # This is independent of position-mode lag (it uses the commanded
        # config, not the live one), so it only fires when IK genuinely could
        # not place the EE on target — i.e. the pose is unreachable. Plain
        # settle lag does NOT trip this.
        pin.framesForwardKinematics(self._model, self._data, q_step)
        cmd_pos = self._data.oMf[self._model.getFrameId(cfg.EE_FRAME)].translation
        cmd_err = float(np.linalg.norm(cmd_pos - target_pos))
        if cmd_err > 0.01:
            logger.warning(
                "[Mover] move_ee: IK could not reach target — commanded EE {:.4f}m short (unreachable?)",
                cmd_err,
            )
        return ok_all

    def _move_ee_to(self, target_pos, target_rpy, duration: float) -> bool:
        """Solve IK for an absolute base-frame target and move there smoothly.

        With cfg.USE_CARTESIAN_INTERP=True (default) the EE follows a straight
        line in base_link space with a smoothstep time profile (see
        _move_ee_cartesian). Otherwise the original behaviour applies: solve
        IK once to the target and joint-space smoothstep into it.
        """
        live = self._current_ee_pos()
        logger.info(
            "[Mover] move_ee start: live=({:.4f}, {:.4f}, {:.4f}) -> target=({:.4f}, {:.4f}, {:.4f})",
            live[0], live[1], live[2],
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
        )
        if getattr(cfg, "USE_CARTESIAN_INTERP", False):
            return self._move_ee_cartesian(target_pos, target_rpy, duration)
        configuration = self._fresh_configuration()
        q_target, ok = self._solve_ik(configuration, target_pos, target_rpy)
        if not ok:
            logger.warning("[Mover] proceeding with non-converged IK to {}", np.round(target_pos, 3))
        self._move_to_joints(q_target, duration)
        return ok

    def _cartesian_z_to(
        self,
        target_z: float,
        rpy,
        profile: str | None = None,
        avg_speed: float | None = None,
    ) -> None:
        """Move the EE straight to *target_z* in Cartesian space (up or down).

        Unlike ``_move_ee_to`` (which can interpolate in joint space and let
        the EE dip then rise because the elbow swings), this keeps x, y and
        the commanded rpy fixed and only varies z, so the cup tip moves
        monotonically along Z. Velocity follows the smoothstep profile in
        _smooth_z_to so there's no constant-rate jolt at start/stop.

        ``profile`` overrides cfg.SMOOTH_PROFILE for this single move — use
        "ease_in" when the next leg is itself moving (e.g. feeding the pick
        descent loop) so we don't brake to a stop only to re-accelerate.
        ``avg_speed`` overrides cfg.LIFT_AVG_SPEED_M_S for this single move
        — used when the next leg has a much lower velocity cap so we don't
        hand off into a hard deceleration.
        """
        cur_pos, _ = self.current_ee_pose()
        target_z = float(target_z)
        distance = abs(target_z - cur_pos[2])
        if distance < 1e-4:
            return
        self._smooth_z_to(cur_pos, target_z, rpy, distance,
                          profile=profile, avg_speed=avg_speed)

    def _smooth_z_to(
        self,
        start_pos,
        target_z: float,
        rpy,
        distance: float,
        profile: str | None = None,
        avg_speed: float | None = None,
    ) -> None:
        """Smoothstep-profiled Cartesian Z move with warm-started IK.

        Used by both _cartesian_z_to and lift(). Keeps x, y and the
        commanded rpy fixed; only z varies. ``profile`` overrides
        cfg.SMOOTH_PROFILE for this call (see _ease for options).
        ``avg_speed`` overrides cfg.LIFT_AVG_SPEED_M_S for this call.
        """
        import time
        speed = avg_speed if avg_speed is not None else getattr(cfg, "LIFT_AVG_SPEED_M_S", 0.08)
        avg_speed = max(float(speed), 1e-3)
        duration = max(distance / avg_speed, float(getattr(cfg, "LIFT_MIN_DURATION_S", 0.3)))
        dt = float(getattr(cfg, "EE_TRAVEL_DT_S", cfg.DESCENT_DT_S))
        n_steps = max(1, int(round(duration / dt)))
        prof = profile if profile is not None else getattr(cfg, "SMOOTH_PROFILE", "quintic")

        z_start = float(start_pos[2])
        live = self._current_ee_pos()
        logger.info(
            "[Mover] z_move start: live=({:.4f}, {:.4f}, {:.4f}) -> target_z={:.4f}",
            live[0], live[1], live[2], target_z,
        )
        configuration = self._fresh_configuration()
        # Pure-z translation: hold the arm's current joint configuration. The
        # default posture target (joint mid-ranges) centres the redundant DOF,
        # which walks it across an IK branch boundary mid-lift — the solver then
        # returns a discontinuous joint solution and the arm slews across joint
        # space, dragging the EE off the vertical line. Pinning posture to the
        # start config keeps the solve on one branch (minimal joint motion).
        prev_posture = self._posture_target
        self._posture_task.set_target(configuration.q.copy())
        try:
            step_pos = np.asarray(start_pos, dtype=float).copy()
            for step in range(1, n_steps + 1):
                alpha = _ease(step / n_steps, prof)
                step_pos[2] = z_start + alpha * (target_z - z_start)
                q_step, _ = self._solve_ik(configuration, step_pos, rpy)
                cmd = self._arm_joints_from_q(q_step)
                self._arm.set_joint_pos(cmd)
                self._trace("z_move", cmd)
                time.sleep(dt)
        finally:
            self._posture_task.set_target(prev_posture)

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
            # Position-mode commands are async: wait for the sideways travel to
            # actually reach transit before descending, so the hover drop starts
            # from the right (x, y) at SAFE_TRANSPORT_Z.
            self._wait_until_arrived(transit, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

        logger.info("[Mover] pick: hover at ({:.3f}, {:.3f}, {:.3f})", *hover_pos)
        # Z-only descent from SAFE_TRANSPORT_Z down to hover_z. Use ease_in
        # so the EE accelerates from rest at transport z and is still moving
        # at hover_z, blending into the contact-search descent loop without
        # decelerating to a full stop and re-accelerating.
        #
        # Match the velocity at handoff to what _descent_loop is willing to
        # command on its very first tick: DESCENT_MAX_STEP_M / DESCENT_DT_S.
        # Ease_in's terminal velocity is 2× the average, so size the average
        # at half the descent loop's velocity cap. Without this match the
        # hover descent arrives ~5× faster than the descent loop allows and
        # the first descent tick is felt as a hard deceleration.
        descent_v_cap = cfg.DESCENT_MAX_STEP_M / max(cfg.DESCENT_DT_S, 1e-3)
        hover_avg_speed = 0.5 * descent_v_cap
        self._cartesian_z_to(
            hover_pos[2], rpy,
            profile="ease_in", avg_speed=hover_avg_speed,
        )

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
        """Raise the EE straight up to *z*, holding x, y and orientation fixed.

        Uses the same smoothstep position profile as the rest of the EE
        travel (see _smooth_z_to) so velocity ramps in and out instead of
        stepping at a constant rate.
        """
        z = cfg.SAFE_TRANSPORT_Z if z is None else z
        if z is None:
            raise ValueError("SAFE_TRANSPORT_Z is not set — teach it before running")
        cur_pos, cur_rpy = self.current_ee_pose()
        target_z = float(z)
        logger.info("[Mover] lift from z={:.3f} to z={:.3f}", cur_pos[2], target_z)
        if target_z <= cur_pos[2] + 1e-4:
            return
        self._smooth_z_to(cur_pos, target_z, cur_rpy, target_z - cur_pos[2])
        self._wait_until_arrived(
            np.array([cur_pos[0], cur_pos[1], target_z]),
            cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S,
        )
        
    def move_to(self, pose: Pose) -> None:
        """Travel sideways to hover above *pose* at SAFE_TRANSPORT_Z."""
        if cfg.SAFE_TRANSPORT_Z is None:
            raise ValueError("SAFE_TRANSPORT_Z is not set — teach it before running")
        target = np.array([pose.pos[0], pose.pos[1], cfg.SAFE_TRANSPORT_Z])
        logger.info("[Mover] move_to ({:.3f}, {:.3f}) @ transport z", target[0], target[1])
        self._move_ee_to(target, pose.rpy, cfg.MOVE_DURATION_S)
        self._wait_until_arrived(target, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

    def _wait_until_arrived(
        self,
        target_pos,
        tol_m: float,
        timeout_s: float,
        settle_speed_m_s: float = 0.01,
    ) -> None:
        """Block until the EE is within *tol_m* of *target_pos* AND has settled
        (live speed below *settle_speed_m_s*).

        Position-mode commands are accepted asynchronously, so the arm can
        still be settling toward the target after our smoothstep loop
        finishes. The next leg in pick/place is a *perpendicular* move (the
        sideways travel hands off to a vertical descent), and both legs use a
        zero-velocity-endpoint smoothstep — so the corner is only smooth if the
        arm has actually come to rest. Returning as soon as we're within tol,
        while the arm still carries velocity from the sideways leg, makes the
        direction change happen under load: that's felt as a jerk. Gating on
        near-zero speed as well as position makes it a true rest-to-rest corner.
        """
        import time
        target = np.asarray(target_pos, dtype=float)
        deadline = time.time() + float(timeout_s)
        prev_pos = self._current_ee_pos()
        prev_t = time.time()
        last_err = float(np.linalg.norm(prev_pos - target))
        while time.time() < deadline:
            time.sleep(0.02)
            cur_pos = self._current_ee_pos()
            now = time.time()
            last_err = float(np.linalg.norm(cur_pos - target))
            speed = float(np.linalg.norm(cur_pos - prev_pos)) / max(now - prev_t, 1e-3)
            prev_pos, prev_t = cur_pos, now
            if last_err <= tol_m and speed <= settle_speed_m_s:
                return
        logger.warning(
            "[Mover] arrival not confirmed within {:.1f}s "
            "(remaining error {:.4f}m, tol {:.4f}m)",
            timeout_s, last_err, tol_m,
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
        # Position-mode commands are async; block until the arm has actually
        # reached hover before _descend_to starts pushing down. Otherwise the
        # descent loop's first step issues a downward IK target while the
        # arm is still settling into hover, which is felt as a jerk at the
        # _move_ee_to -> _descend_to boundary.
        self._wait_until_arrived(hover, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

        # Re-tare the wrench while hovering with the battery still attached.
        # Without this, the force baseline left over from the empty-cup tare
        # at pick() time means the battery's own weight reads as a large
        # apparent force and instantly trips FORCE_HARD_LIMIT_N during the
        # place descent (especially noticeable on undo moves where the arm
        # has been carrying a battery the whole sequence).
        import time

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
        # current_pos: the next commanded EE target. It marches down as an
        # open-loop ramp (one step per tick) and is deliberately NOT reset to
        # the live cup pose — keeping the commanded target ahead of the
        # position-mode-lagging arm is what lets the descent reach its full
        # step/dt velocity. x, y stay pinned at hover so the descent is purely
        # vertical.
        current_pos = np.asarray(hover_pos, dtype=float).copy()
        descended = 0.0
        _live0 = self._current_ee_pos()
        logger.info(
            "[Mover] descent_loop start: live=({:.4f}, {:.4f}, {:.4f}) -> target_z={:.4f}",
            _live0[0], _live0[1], _live0[2], float(target_z),
        )

        def _halt() -> None:
            """Re-command the live EE pose so the arm stops where it is."""
            live_pos = self._current_ee_pos()
            hold_q, _ = self._solve_ik(self._fresh_configuration(), live_pos, rpy)
            cmd = self._arm_joints_from_q(hold_q)
            self._arm.set_joint_pos(cmd)
            self._trace("descent_halt", cmd)

        while descended < cfg.MAX_DESCENT_M:
            # Warm-start IK from the live arm state so each solve seeds from
            # where the arm actually is. The commanded current_pos is NOT
            # re-grounded to live: it marches down as an open-loop ramp so the
            # commanded target stays ahead of the (position-mode-lagging) arm,
            # letting the servo reach the full step/dt descent velocity. The
            # lead-below-contact is intentional and handled by _halt() on a
            # stop event; re-grounding current_pos to live collapses the speed
            # because the target then never leads the arm by more than a step.
            configuration = self._fresh_configuration()
            step = float(np.clip((current_pos[2] - target_z) * cfg.DESCENT_KP,
                                 cfg.DESCENT_MIN_STEP_M, cfg.DESCENT_MAX_STEP_M))
            current_pos[2] -= step
            descended += step
            q_step, _ = self._solve_ik(configuration, current_pos, rpy)
            cmd = self._arm_joints_from_q(q_step)
            self._arm.set_joint_pos(cmd)
            self._trace("descent", cmd)
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
        """Constant-velocity descent that stops on contact within the buffer.

        Stop conditions, in priority order:
          1. FORCE_HARD_LIMIT_PLACE_N exceeded -> emergency stop (safety).
          2. force > FORCE_CONTACT_THRESHOLD_N  AND
             current_z <= goal_z + PLACE_Z_BUFFER_M
                                                 -> contact within buffer;
             treat as good seat and release.
          3. descended > MAX_DESCENT_M           -> never made contact; abort.

        The taught goal_z is no longer used as an auto-stop on its own — we
        keep descending past it (while remaining bounded by MAX_DESCENT_M)
        until the wrench reports contact within the buffer window, so that
        slight teach errors or stack-height variation don't release the part
        in mid-air.
        """
        import time
        goal_z = float(target_pos[2])
        base_x, base_y = float(target_pos[0]), float(target_pos[1])
        _live0 = self._current_ee_pos()
        logger.info(
            "[Mover] descend_to start: live=({:.4f}, {:.4f}, {:.4f}) -> goal=({:.4f}, {:.4f}, {:.4f})",
            _live0[0], _live0[1], _live0[2], base_x, base_y, goal_z,
        )
        descended = 0.0
        step_count = 0
        f_max_seen = 0.0  # diagnostic: peak force across the descent
        # z_cmd is the commanded EE z. Like _descent_loop, it marches down as
        # an open-loop ramp (one step per tick) and is NOT re-grounded to the
        # live cup z, so the commanded target leads the position-mode-lagging
        # arm and the descent reaches its full step/dt velocity. IK is still
        # warm-started from the live arm each tick (configuration below). The
        # cumulative *actual* descent (prev_z) bounds MAX_DESCENT_M.
        z_cmd = self._current_ee_pos()[2]
        z_cmd_start = z_cmd
        prev_z = z_cmd

        def _halt() -> None:
            """Re-command the live EE pose so the arm stops where it is.

            On a stop event z_cmd sits below the contact point (it leads the
            arm); without this halt the arm would keep driving into the
            battery until place() issues its next command. Halting bounds the
            over-press to one tick of servo lag.
            """
            live = self._current_ee_pos()
            hold_q, _ = self._solve_ik(self._fresh_configuration(), live, rpy)
            cmd = self._arm_joints_from_q(hold_q)
            self._arm.set_joint_pos(cmd)
            self._trace("place_halt", cmd)

        while descended < cfg.MAX_DESCENT_M:
            configuration = self._fresh_configuration()
            current_z = self._current_ee_pos()[2]
            descended += max(0.0, prev_z - current_z)
            prev_z = current_z
            step_pos = np.array([base_x, base_y, current_z])
            # Step size ramps linearly with descent progress: PLACE_DESCENT_MAX_STEP_M
            # at the start (z_cmd == z_cmd_start) down to PLACE_DESCENT_MIN_STEP_M
            # at goal_z. Past goal_z the progress clamps at 1, so the step holds
            # at the min until contact stops the loop.
            progress = float(np.clip(
                (z_cmd_start - z_cmd) / max(z_cmd_start - goal_z, 1e-6), 0.0, 1.0))
            step = float(cfg.PLACE_DESCENT_MAX_STEP_M
                         - progress * (cfg.PLACE_DESCENT_MAX_STEP_M - cfg.PLACE_DESCENT_MIN_STEP_M))
            phase = 2 * np.pi * step_count * cfg.PLACE_DESCENT_DT_S * cfg.JITTER_FREQ_HZ
            # Ramp jitter amplitude in from 0 over the first JITTER_RAMP_S
            # seconds of descent so the Y/roll/pitch terms (which use cos /
            # sin(+π/2) and start at full amplitude) don't snap on step 0.
            # Without this ramp the cup visibly flicks at the start of every
            # place descent because the commanded orientation jumps
            # discontinuously from the hover pose.
            ramp_s = max(cfg.JITTER_RAMP_S, 1e-6)
            ramp = min(1.0, step_count * cfg.PLACE_DESCENT_DT_S / ramp_s)
            # Smoothly ease the descent velocity in from 0 over the same
            # ramp window. Without this the very first step jumps to the
            # clamped max step size, so the EE goes from a clean stop at
            # hover to full descent velocity in one tick — that's felt as a
            # jerk between _move_ee_to(hover) and the start of this loop.
            # Quintic profile matches the rest of the smoothstep moves.
            vel_ramp = _ease(ramp, getattr(cfg, "SMOOTH_PROFILE", "quintic"))
            step *= vel_ramp
            z_cmd -= step
            step_pos[0] = base_x + ramp * cfg.JITTER_AMPLITUDE_M * np.sin(phase)
            step_pos[1] = base_y + ramp * cfg.JITTER_AMPLITUDE_M * np.cos(phase)
            step_pos[2] = z_cmd
            # Same-frequency yaw wobble so the cup wiggles slightly while it
            # descends, helping the battery seat into the slot if X/Y jitter
            # alone isn't enough to clear a tight fit. Roll/pitch are added
            # at phase offsets so the cup nutates instead of just twisting.
            step_rpy = np.array(rpy, dtype=float, copy=True)
            step_rpy[0] += ramp * cfg.JITTER_ROLL_AMPLITUDE_RAD * np.cos(phase)
            step_rpy[1] += ramp * cfg.JITTER_PITCH_AMPLITUDE_RAD * np.sin(phase + np.pi / 2)
            step_rpy[2] += ramp * cfg.JITTER_YAW_AMPLITUDE_RAD * np.sin(phase)
            q_step, _ = self._solve_ik(configuration, step_pos, step_rpy)
            cmd = self._arm_joints_from_q(q_step)
            self._arm.set_joint_pos(cmd)
            self._trace("place_descend", cmd)
            step_count += 1
            time.sleep(cfg.PLACE_DESCENT_DT_S)

            force = get_force(cfg.ARM_SIDE, self._robot)
            if force is None:
                continue
            if force > f_max_seen:
                f_max_seen = force
            if force > cfg.FORCE_HARD_LIMIT_PLACE_N:
                _halt()
                logger.warning(
                    "[Mover] place: hard force {:.1f}N at z={:.4f} "
                    "(descended {:.3f}m, peak {:.1f}N) — emergency stop",
                    force, current_z, descended, f_max_seen,
                )
                return
            if force > cfg.FORCE_CONTACT_THRESHOLD_N and \
                    current_z <= goal_z + cfg.PLACE_Z_BUFFER_M:
                _halt()
                logger.info(
                    "[Mover] place: contact {:.1f}N at z={:.4f} (goal_z={:.4f}, "
                    "dz={:+.4f}m) within buffer — releasing",
                    force, current_z, goal_z, current_z - goal_z,
                )
                return

        _halt()
        logger.warning(
            "[Mover] place: max descent reached without reaching goal_z={:.4f} "
            "(current z={:.4f}, peak force seen {:.1f}N)",
            goal_z, current_z, f_max_seen,
        )
