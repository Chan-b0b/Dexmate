"""IK-driven single-arm motion core for ik_demo.

Self-contained: pinocchio (IK + self-collision), pink (differential IK QP),
ruckig (jerk-limited trajectories), dexcontrol (streaming). No dexmotion at
runtime — see PLAN.md for why and what was harvested from it (the SRDF path,
the real per-joint limits, and the "park joint 2" tip: L_arm_j2 has the
tightest range, [-0.45, +1.55] rad).

Two motion primitives only (PLAN.md):
  - move_joints(q)      : Ruckig joint-space profile between cached configs,
                          streamed via arm.set_joint_pos_vel at CONTROL_HZ.
  - move_ee(pos, rpy)   : Cartesian, live warm-started IK per tick (sensing
                          legs). Endpoint-only: the joint-space trajectory arcs
                          sideways in between — use move_ee_vertical for lifts.

Fixed taught poses are solved to joints once (cache_taught_poses), validated
(converged + in-limits + collision-free), and thereafter reached with
move_joints — deterministic, no branch-flip lottery. Live IK (min-motion,
seeded from the current config) is confined to the sensing legs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import pink
from loguru import logger
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from ruckig import InputParameter, Ruckig, Trajectory
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
except ImportError:  # allow `python arm.py` from inside ik_demo/
    import config as cfg

_ARM_DOF = 7


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


@dataclass
class PoseSolution:
    """Result of solving a Cartesian target to arm joints."""

    q: np.ndarray            # (7,) arm-joint vector
    converged: bool          # EE error < IK_CONVERGENCE_THRESHOLD
    pos_err_m: float         # FK position error at the solution
    in_collision: bool       # self-collision at the solution
    in_limits: bool          # within joint position limits

    @property
    def valid(self) -> bool:
        return self.converged and not self.in_collision and self.in_limits


class ArmMover:
    """Single-arm IK + trajectory motion for the suction/gripper arm.

    Args:
        robot: connected dexcontrol Robot, or None for headless use (pose
            caching / validation / benchmarks). Without a robot the torso is
            assumed at ``torso_deg`` since its live angle is unknown.
        side / ee_frame: default to cfg.ARM_SIDE / cfg.EE_FRAME.
        torso_deg: nominal torso joint angles (rad) when robot is None.
    """

    def __init__(
        self,
        robot=None,
        side: str | None = None,
        ee_frame: str | None = None,
        torso_q: np.ndarray | None = None,
    ) -> None:
        self._robot = robot
        self._side = side or cfg.ARM_SIDE
        self._ee_frame = ee_frame or cfg.EE_FRAME
        self._setup_model(torso_q)
        self._setup_ik()
        self._setup_collision()
        self._setup_ruckig()

    # ------------------------------------------------------------------
    # Model (base_link-rooted; arm-only reduced model, torso locked)
    # ------------------------------------------------------------------
    def _setup_model(self, torso_q: np.ndarray | None) -> None:
        urdf = cfg.URDF_PATH
        pkg_dirs = [
            os.path.dirname(urdf),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf)))),
        ]
        self._urdf = urdf
        self._pkg_dirs = pkg_dirs
        full = pin.RobotWrapper.BuildFromURDF(
            filename=urdf, package_dirs=pkg_dirs, root_joint=None
        ).model
        self._full_model = full

        # Torso angle: live if a robot is attached, else the configured stance
        # (taught poses are only reachable at the torso pose they were taught at).
        if torso_q is None:
            if self._robot is not None:
                torso_q = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
            else:
                torso_q = np.asarray(cfg.TORSO_JOINTS, dtype=float)
        self._torso_q = np.asarray(torso_q, dtype=float)
        # Home config for this arm — the default IK seed (a near-workspace seed;
        # differential IK stalls from a zero seed).
        self._home_seed = np.asarray(
            cfg.HOME_JOINTS_RIGHT if self._side == "right" else cfg.HOME_JOINTS_LEFT,
            dtype=float,
        )

        prefix = "R" if self._side == "right" else "L"
        arm_names = {f"{prefix}_arm_j{j + 1}" for j in range(_ARM_DOF)}
        q_ref = pin.neutral(full)
        for j in range(len(self._torso_q)):
            jid = full.getJointId(f"torso_j{j + 1}")
            if jid < full.njoints:
                q_ref[full.idx_qs[jid]] = self._torso_q[j]
        self._q_ref_full = q_ref.copy()

        lock_ids = [
            jid for jid in range(1, full.njoints) if full.names[jid] not in arm_names
        ]
        self._model = pin.buildReducedModel(full, lock_ids, q_ref)
        self._data = self._model.createData()
        self._arm_joint_ids = [
            self._model.getJointId(f"{prefix}_arm_j{j + 1}") for j in range(_ARM_DOF)
        ]
        self._ee_frame_id = self._model.getFrameId(self._ee_frame)
        # Per-joint limits straight from the URDF (via the reduced model).
        self._q_lo = self._model.lowerPositionLimit.copy()
        self._q_hi = self._model.upperPositionLimit.copy()
        self._v_max = self._model.velocityLimit.copy()

        # IK-side-only tightening (this class's model/config_limit_gain only —
        # URDF and dexcontrol's hardware command clamping are untouched): keep
        # L_arm_j4 (elbow) from swinging above -0.5 rad, well inside its URDF
        # range of [-3.071, 0.244].
        if self._side == "left":
            j4_idx = self._model.idx_qs[self._model.getJointId("L_arm_j4")]
            self._q_hi[j4_idx] = min(float(self._q_hi[j4_idx]), -0.5)
            self._model.upperPositionLimit[j4_idx] = self._q_hi[j4_idx]

    def _setup_ik(self) -> None:
        # Position weighted 2x over orientation (grasp.py's proven ratio).
        self._ee_task = FrameTask(
            self._ee_frame, position_cost=2.0, orientation_cost=1.0,
            lm_damping=cfg.IK_LM_DAMPING,
        )
        self._posture_task = PostureTask(cost=cfg.POSTURE_COST)
        mid = 0.5 * (self._q_lo + self._q_hi)
        self._posture_mid = np.where(np.isfinite(mid), mid, 0.0)
        self._posture_task.set_target(self._posture_mid)
        self._limits = [ConfigurationLimit(self._model), VelocityLimit(self._model)]
        import qpsolvers
        pref = cfg.PREFERRED_QP_SOLVER
        self._solver = pref if pref in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
        logger.info(
            "[arm] IK ready — side={} EE={} DOF={} solver={}",
            self._side, self._ee_frame, self._model.nq, self._solver,
        )

    def _setup_collision(self) -> None:
        """Self-collision on the FULL model, filtered by the dexmate_urdf SRDF."""
        srdf = os.path.splitext(self._urdf)[0] + ".srdf"
        self._collision_ok_setup = False
        try:
            geom = pin.buildGeomFromUrdf(
                self._full_model, self._urdf, pin.GeometryType.COLLISION, self._pkg_dirs
            )
            geom.addAllCollisionPairs()
            if os.path.exists(srdf):
                pin.removeCollisionPairs(self._full_model, geom, srdf)
            else:
                logger.warning("[arm] SRDF not found at {} — collision pairs unfiltered", srdf)
            self._geom = geom
            self._geom_data = pin.GeometryData(geom)
            self._full_data = self._full_model.createData()
            self._collision_ok_setup = True
            logger.info("[arm] self-collision ready — {} pairs", len(geom.collisionPairs))
        except Exception as e:  # noqa: BLE001
            logger.warning("[arm] collision setup failed ({}); checks will pass-through", e)

    def _setup_ruckig(self) -> None:
        self._otg = Ruckig(_ARM_DOF)
        s = float(cfg.SPEED_SCALE_RIGHT if self._side == "right" else cfg.SPEED_SCALE_LEFT)
        # Clamp the configured velocity cap to the arm's real per-joint limit.
        self._ruckig_vmax = np.minimum(self._v_max, cfg.MAX_JOINT_VEL) * s
        self._ruckig_amax = np.full(_ARM_DOF, cfg.MAX_JOINT_ACCEL * s)
        self._ruckig_jmax = np.full(_ARM_DOF, cfg.MAX_JOINT_JERK * s)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    @property
    def _arm(self):
        return getattr(self._robot, f"{self._side}_arm")

    def _live_arm_q(self) -> np.ndarray:
        if self._robot is None:
            raise RuntimeError("no robot attached — cannot read live joints")
        return np.asarray(self._arm.get_joint_pos(), dtype=float)

    def _configuration(self, arm_q: np.ndarray) -> pink.Configuration:
        q = pin.neutral(self._model)
        for k, jid in enumerate(self._arm_joint_ids):
            q[self._model.idx_qs[jid]] = arm_q[k]
        q = np.clip(q, self._q_lo, self._q_hi)
        return pink.Configuration(self._model, self._data, q)

    def _arm_q_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.array([q_full[self._model.idx_qs[j]] for j in self._arm_joint_ids])

    def fk(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pos [x,y,z], rpy) of the EE frame in base_link for an arm-joint vector."""
        cfg_ = self._configuration(arm_q)
        pin.framesForwardKinematics(self._model, self._data, cfg_.q)
        T = self._data.oMf[self._ee_frame_id]
        return T.translation.copy(), Rotation.from_matrix(T.rotation).as_euler("xyz")

    def current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.fk(self._live_arm_q())

    def current_ee_rotation(self) -> np.ndarray:
        """Live rotation matrix R (base <- EE), for projecting the wrist wrench
        onto the base-frame vertical."""
        cfg_ = self._configuration(self._live_arm_q())
        pin.framesForwardKinematics(self._model, self._data, cfg_.q)
        return self._data.oMf[self._ee_frame_id].rotation.copy()

    # ------------------------------------------------------------------
    # IK: solve a Cartesian target to arm joints (own convergence loop)
    # ------------------------------------------------------------------
    def solve_pose(
        self, pos, rpy, seed: np.ndarray | None = None, min_motion: bool = False
    ) -> PoseSolution:
        """Iterate differential IK to a base_link target; validate the result.

        seed: initial arm config (defaults to joint mid-ranges for a robust
            cold solve; pass the current config for a warm/live solve).
        min_motion: pin the posture target to the seed (stay on one branch,
            minimal joint travel) — use for live moves. Cold offline solves
            leave it at the mid-ranges (curate away from limits).

        A QP iteration can diverge to a non-finite (NaN/Inf) configuration
        near a singularity or an infeasible limit box — fk()'s
        Rotation.from_matrix does an SVD that raises LinAlgError on non-finite
        input, which would crash the caller's streamed loop outright instead
        of just failing to converge. On that, retry from the SAME seed with
        escalating LM damping rather than giving up immediately: retrying from
        a *different* seed (e.g. home) is NOT safe here, since a live per-tick
        caller (move_ee_vertical, the suction descent loops) computes its
        velocity feedforward as (new_q - prev_q)/dt — landing on an unrelated
        branch from a different seed would command a large, sudden jump. Only
        after every damping level still diverges is it reported unreachable
        (pos_err_m=inf trips every caller's existing REACH_TOL_M check) so it
        halts gracefully like any other unreachable target.
        """
        seed = self._home_seed if seed is None else np.asarray(seed, dtype=float)
        target = pin.SE3(_rpy_to_matrix(*rpy), np.asarray(pos, dtype=float))
        self._ee_task.set_target(target)
        orig_damping = self._ee_task.lm_damping
        try:
            for damping in (orig_damping, orig_damping * 1e3, orig_damping * 1e6):
                self._ee_task.lm_damping = damping
                configuration = self._configuration(seed)
                if min_motion:
                    self._posture_task.set_target(configuration.q.copy())
                else:
                    self._posture_task.set_target(self._posture_mid)
                tasks = [self._ee_task, self._posture_task]
                converged = False
                for _ in range(cfg.IK_MAX_ITERS):
                    v = solve_ik(configuration, tasks, cfg.IK_DT, solver=self._solver, limits=self._limits)
                    configuration.update(pin.integrate(self._model, configuration.q, v * cfg.IK_DT))
                    if np.linalg.norm(self._ee_task.compute_error(configuration)) < cfg.IK_CONVERGENCE_THRESHOLD:
                        converged = True
                        break
                arm_q = self._arm_q_from_full(configuration.q)
                if np.all(np.isfinite(arm_q)):
                    if damping != orig_damping:
                        logger.warning("[arm] IK diverged at damping={:.1e} — recovered at damping={:.1e}",
                                        orig_damping, damping)
                    fk_pos, _ = self.fk(arm_q)
                    return PoseSolution(
                        q=arm_q,
                        converged=converged,
                        pos_err_m=float(np.linalg.norm(fk_pos - np.asarray(pos, dtype=float))),
                        in_collision=self.in_collision(arm_q),
                        in_limits=self.in_limits(arm_q),
                    )
            logger.warning("[arm] IK diverged to a non-finite configuration even after damping "
                            "retries — treating as unreachable")
            return PoseSolution(q=seed, converged=False, pos_err_m=float("inf"),
                                 in_collision=True, in_limits=False)
        finally:
            self._ee_task.lm_damping = orig_damping

    def _posture_mid_arm(self) -> np.ndarray:
        return np.array([self._posture_mid[self._model.idx_qs[j]] for j in self._arm_joint_ids])

    def in_limits(self, arm_q: np.ndarray, margin: float = 1e-3) -> bool:
        lo = np.array([self._q_lo[self._model.idx_qs[j]] for j in self._arm_joint_ids])
        hi = np.array([self._q_hi[self._model.idx_qs[j]] for j in self._arm_joint_ids])
        return bool(np.all(arm_q >= lo - margin) and np.all(arm_q <= hi + margin))

    def in_collision(self, arm_q: np.ndarray) -> bool:
        if not self._collision_ok_setup:
            return False
        q_full = self._q_ref_full.copy()
        prefix = "R" if self._side == "right" else "L"
        for k in range(_ARM_DOF):
            jid = self._full_model.getJointId(f"{prefix}_arm_j{k + 1}")
            q_full[self._full_model.idx_qs[jid]] = arm_q[k]
        pin.computeCollisions(
            self._full_model, self._full_data, self._geom, self._geom_data, q_full, True
        )
        return any(self._geom_data.collisionResults[i].isCollision()
                   for i in range(len(self._geom.collisionPairs)))

    # ------------------------------------------------------------------
    # Pose cache: solve + validate all taught poses to joints
    # ------------------------------------------------------------------
    def taught_target(self, pose) -> tuple[np.ndarray, np.ndarray]:
        """Return the EE-frame IK target for a taught pose.

        The taught poses were recorded as the EE (L_gripper_base) pose — the
        same frame the IK solves for — so they are used directly. (We do NOT
        add SUCTION_LENGTH_M: that would double-count, since the taught z is
        already L_gripper_base, not the cup tip.)
        """
        return np.array(pose[:3], dtype=float), np.array(pose[3:6], dtype=float)

    def cache_taught_poses(self) -> dict[str, PoseSolution]:
        out: dict[str, PoseSolution] = {}
        seed = self._home_seed.copy()  # warm-chain from home
        for name, pose in cfg.TAUGHT_POSES.items():
            pos, rpy = self.taught_target(pose)
            sol = self.solve_pose(pos, rpy, seed=seed, min_motion=True)
            out[name] = sol
            if sol.converged:
                seed = sol.q
            flag = "OK " if sol.valid else "BAD"
            logger.info(
                "[arm] cache {} {:12s} err={:.2f}mm converged={} collision={} in_limits={}",
                flag, name, sol.pos_err_m * 1000, sol.converged, sol.in_collision, sol.in_limits,
            )
        return out

    # ------------------------------------------------------------------
    # Trajectory generation + streaming
    # ------------------------------------------------------------------
    def plan_joint_traj(self, q_start: np.ndarray, q_goal: np.ndarray) -> Trajectory:
        """Jerk-limited joint-space trajectory (Ruckig) under the arm's limits."""
        inp = InputParameter(_ARM_DOF)
        inp.current_position = list(map(float, q_start))
        inp.current_velocity = [0.0] * _ARM_DOF
        inp.current_acceleration = [0.0] * _ARM_DOF
        inp.target_position = list(map(float, q_goal))
        inp.target_velocity = [0.0] * _ARM_DOF
        inp.target_acceleration = [0.0] * _ARM_DOF
        inp.max_velocity = list(map(float, self._ruckig_vmax))
        inp.max_acceleration = list(map(float, self._ruckig_amax))
        inp.max_jerk = list(map(float, self._ruckig_jmax))
        traj = Trajectory(_ARM_DOF)
        self._otg.calculate(inp, traj)
        return traj

    def move_joints(self, q_goal: np.ndarray) -> None:
        """Stream a Ruckig joint-space trajectory to q_goal at CONTROL_HZ."""
        q_start = self._live_arm_q()
        traj = self.plan_joint_traj(q_start, np.asarray(q_goal, dtype=float))
        dt = 1.0 / float(cfg.CONTROL_HZ)
        n = max(1, int(np.ceil(traj.duration / dt)))
        logger.info("[arm] move_joints: {:.2f}s, {} steps", traj.duration, n)
        for i in range(1, n + 1):
            pos, vel, _acc = traj.at_time(min(i * dt, traj.duration))
            self._arm.set_joint_pos_vel(np.asarray(pos), np.asarray(vel))
            time.sleep(dt)

    def move_ee(self, pos, rpy, quiet: bool = True) -> np.ndarray | None:
        """Move the EE frame to an absolute base_link pose (solve IK from the live
        config, min-motion, then move_joints). Returns the commanded target joints
        (so a caller can continue a stream from exactly there), or None if the
        target is beyond REACH_TOL_M. ``quiet`` demotes the within-tol shortfall
        log to debug (for transport-height legs where a few mm is expected and
        recovered lower down — not the alignment-critical descent/sweep legs)."""
        sol = self.solve_pose(pos, rpy, seed=self._live_arm_q(), min_motion=True)
        if sol.pos_err_m > cfg.REACH_TOL_M:
            logger.error("[arm] target unreachable ({:.1f}mm short) — not moving", sol.pos_err_m * 1000)
            return None
        if not sol.converged:
            log = logger.debug if quiet else logger.warning
            log("[arm] target {:.1f}mm short (within reach tol) — moving", sol.pos_err_m * 1000)
        self.move_joints(sol.q)
        return sol.q

    def move_ee_vertical(self, z_target: float, rpy) -> np.ndarray | None:
        """Straight vertical EE move to base-frame z_target, x,y,rpy held EVERY
        tick (per-tick warm IK stream) — the free-air analog of the suction
        descent legs. move_ee only constrains the endpoints; its joint-space
        trajectory arcs sideways in between, so lifts holding a part must use
        this instead. Speed reuses the descent budget: smoothstep ramp-in from
        rest, cruise at DESCENT_APPROACH_SPEED_M_S, blend to creep into the
        target. Returns the last commanded joints, or None if a tick's IK falls
        beyond REACH_TOL_M (halts in place — partial motion, logged)."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = self._live_arm_q()
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        z_target = float(z_target)
        direction = 1.0 if z_target >= z else -1.0
        fast, creep = cfg.DESCENT_APPROACH_SPEED_M_S, cfg.DESCENT_CREEP_SPEED_M_S
        band = max(float(cfg.DESCENT_CREEP_BLEND_M), 1e-6)
        elapsed = 0.0
        logger.info("[arm] move_ee_vertical: z {:.4f} -> {:.4f} (xy held at {:.4f},{:.4f})",
                    z, z_target, x, y)
        while abs(z_target - z) > 1e-4:
            dist = abs(z_target - z)
            if dist >= band:
                base = fast
            else:
                f = dist / band
                f = f * f * (3.0 - 2.0 * f)                # smoothstep decel into target
                base = creep + (fast - creep) * f
            r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
            speed = base * (r * r * (3.0 - 2.0 * r))       # smoothstep ramp-in
            z_next = z + direction * min(speed * dt, dist)
            sol = self.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                self._arm.set_joint_pos_vel(prev_q, np.zeros(_ARM_DOF))
                logger.warning("[arm] vertical move stalled {:.1f}mm short at z={:.4f} — halting",
                               sol.pos_err_m * 1000, z)
                return None
            self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
            z, prev_q = z_next, sol.q
            elapsed += dt
            time.sleep(dt)
        self._arm.set_joint_pos_vel(prev_q, np.zeros(_ARM_DOF))
        return prev_q

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------
    def software_estop_active(self) -> bool:
        estop = getattr(self._robot, "estop", None)
        return bool(estop is not None and estop.is_software_estop_enabled())

    def pin_torso(self, wait_time: float = 6.0) -> None:
        """Command the torso to cfg.TORSO_JOINTS and hold it there, then rebuild
        the reduced model at that stance so the model matches reality.

        The taught poses were validated at TORSO_JOINTS; live-torso drift shifts
        the arm base and shrinks reach at the transport height (a horizontal
        shortfall that offsets the descent). Pinning removes that drift."""
        if self._robot is None:
            return
        target = np.asarray(cfg.TORSO_JOINTS, dtype=float)
        live = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
        logger.info("[arm] pinning torso {} -> {} (rad)", np.round(live, 3), np.round(target, 3))
        self._robot.torso.set_joint_pos(target, wait_time=wait_time, exit_on_reach=True)
        # Rebuild the reduced model at the pinned stance (was built from the live
        # torso at construction, which may have drifted).
        self._setup_model(target)
        self._setup_ik()
        self._setup_collision()

    def ensure_ready(self, release_estop: bool = False) -> bool:
        if self.software_estop_active():
            if not release_estop:
                logger.warning("[arm] software E-Stop active — release it and retry")
                return False
            self._robot.estop.deactivate()
            time.sleep(0.5)
        self._arm.set_modes(["position"] * _ARM_DOF)
        if self.software_estop_active():
            return False
        self.pin_torso()
        return True


# ---------------------------------------------------------------------------
# Headless self-test: model + IK + cache + collision + Ruckig + benchmark.
# Run from LGES/:  python -m ik_demo.arm
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # Uses cfg.TORSO_JOINTS (the demo's stance); reachability is torso-dependent.
    logger.info("=== ik_demo.arm headless self-test (no robot, torso={}) ===",
                np.round(cfg.TORSO_JOINTS, 3))
    mover = ArmMover(robot=None)
    sols = mover.cache_taught_poses()
    n_valid = sum(s.valid for s in sols.values())
    logger.info("cache: {}/{} poses valid", n_valid, len(sols))

    # warm-solve benchmark (100 Hz budget = 10 ms)
    p = cfg.TAUGHT_POSES["CASE_PICK"]
    seed = sols["CASE_PICK"].q
    t0 = time.perf_counter()
    N = 300
    for i in range(N):
        z = p[2] + 0.02 * np.sin(i / 8.0)
        mover.solve_pose([p[0], p[1], z], p[3:6], seed=seed, min_motion=True)
    ms = (time.perf_counter() - t0) / N * 1000
    logger.info("warm solve: {:.2f} ms/solve -> {} at {}Hz",
                ms, "OK" if ms < 1000.0 / cfg.CONTROL_HZ else "TOO SLOW", cfg.CONTROL_HZ)

    # Ruckig trajectory between two cached configs
    if sols["CASE_PICK"].valid and sols["CASE_PLACE_R"].valid:
        traj = mover.plan_joint_traj(sols["CASE_PICK"].q, sols["CASE_PLACE_R"].q)
        logger.info("Ruckig CASE_PICK->CASE_PLACE_R: {:.2f}s", traj.duration)


def _verify_on_robot() -> None:
    """On-robot verification: validate the pose cache at the LIVE torso (no
    motion), then optionally stream move_joints home -> each pose -> home.

    Run from LGES/:  python -m ik_demo.arm --robot
    """
    from dexcontrol.robot import Robot

    with Robot() as bot:
        mover = ArmMover(robot=bot)  # reads the live torso
        live_torso = np.asarray(bot.torso.get_joint_pos(), dtype=float)
        logger.info("live torso (rad): {}  | cfg.TORSO_JOINTS: {}",
                    np.round(live_torso, 3), np.round(cfg.TORSO_JOINTS, 3))
        if np.max(np.abs(live_torso - np.asarray(cfg.TORSO_JOINTS))) > 0.05:
            logger.warning("live torso differs from cfg.TORSO_JOINTS — poses were "
                           "taught at a different stance; expect solve failures.")

        # --- Step 1: validate the cache at the real torso (NO motion) ---
        sols = mover.cache_taught_poses()
        n_valid = sum(s.valid for s in sols.values())
        logger.info("cache at live torso: {}/{} valid", n_valid, len(sols))
        if n_valid < len(sols):
            logger.error("Not all poses valid — NOT moving. Fix torso/teach first.")
            return

        # --- Step 2: real motion, behind a safety prompt ---
        logger.warning("=" * 60)
        logger.warning("NEXT STEP MOVES THE REAL ARM: home -> each taught pose -> home.")
        logger.warning("Clear the workspace. Keep the e-stop within reach.")
        logger.warning("=" * 60)
        if input("Stream move_joints through the poses? [y/N]: ").strip().lower() != "y":
            logger.info("Validation only — no motion. Done.")
            return

        release = mover.software_estop_active()
        if release and input("Software E-Stop is active. Release it? [y/N]: ").strip().lower() != "y":
            logger.info("Leaving E-Stop engaged; aborting.")
            return
        if not mover.ensure_ready(release_estop=release):
            logger.error("Arm not ready. Aborting.")
            return

        logger.info("-> home")
        mover.move_joints(mover._home_seed)
        for name, sol in sols.items():
            input(f"[enter] move to {name} (Ctrl-C to stop) ")
            logger.info("-> {}", name)
            mover.move_joints(sol.q)
        input("[enter] return home ")
        mover.move_joints(mover._home_seed)
        logger.info("verification sequence complete.")


if __name__ == "__main__":
    import sys
    if "--robot" in sys.argv:
        _verify_on_robot()
    else:
        _selftest()
