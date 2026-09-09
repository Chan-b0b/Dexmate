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
from ruckig import InputParameter, Result, Ruckig, Trajectory
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
except ImportError:  # allow `python arm.py` from inside ik_demo/
    import config as cfg

_ARM_DOF = 7


class TickPacer:
    """Wall-clock pacing for the streaming loops.

    ``time.sleep(dt)`` alone stretches every period by the tick's own compute
    time (IK solve + publish + force read), so the finite-difference velocity
    feedforward (dq / dt) runs 15-30% hot and the arm leads-then-corrects on
    every tick. Pacing against absolute tick boundaries keeps the mean period
    at dt. A tick that overruns by more than a full period re-bases the
    schedule instead of bursting through the missed ticks.
    """

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.t0 = time.perf_counter()
        self.n = 0

    def wait(self) -> float:
        """Block until the next tick boundary; return the time since start (s)."""
        self.n += 1
        target = self.t0 + self.n * self.dt
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        elif now - target > self.dt:
            self.t0 = now - self.n * self.dt   # long stall: re-base, don't burst
        return time.perf_counter() - self.t0


def _preset_head_init_pitch(configs):
    """Set the head's "home" pose in ``configs`` to cfg.HEAD_INIT_PITCH_DEG, so
    Robot() construction takes the head straight to the working angle instead of
    raising it to the horizon for the demo to pull back down (see the constant).

    Takes the config the caller is ALREADY passing and edits it, building a
    default one only when there is none. It has to be this way round: every real
    entry point passes a config (chassis_sequence builds one to enable the head
    camera), so a version that only filled in a missing config — the first cut
    here — silently did nothing on exactly the runs it was written for, with no
    warning to show for it.

    Mutating the caller's object is safe: get_robot_config() returns a fresh
    dataclass per call and pose_pool is a default_factory field, so the dict
    belongs to this instance alone.

    Returns ``configs`` UNCHANGED when the override is off or the config layout
    is not what we expect. dexcontrol and dexbot_utils are both pinned
    dependencies (dexcontrol is a submodule), so the shape read here can move
    under us on an update; a head that starts at the wrong angle is a cosmetic
    regression, while a raise from in here would stop the run connecting at
    all."""
    pitch = cfg.HEAD_INIT_PITCH_DEG
    if pitch is None:
        return configs
    try:
        conf = configs
        if conf is None:
            from dexbot_utils import RobotInfo

            conf = RobotInfo().config      # same resolution dexcontrol would do
        home = [float(np.deg2rad(90.0 - float(pitch))), 0.0, 0.0]
        conf.components["head"].pose_pool["home"] = home
        logger.debug("[robot] head init pose preset to {:.1f}deg "
                     "(head_j1 home {:+.3f} rad)", float(pitch), home[0])
        return conf
    except Exception as e:  # noqa: BLE001 — see the docstring
        logger.warning("[robot] could not preset the head init pitch ({}: {}) — "
                       "connecting with the stock config; the head will rise at "
                       "startup", type(e).__name__, e)
        return configs


def connect_robot(configs=None, attempts: "int | None" = None,
                  delay: "float | None" = None):
    """Construct a dexcontrol Robot, retrying a transient startup failure.

    Robot.__init__ ends with _set_default_state, which reads the TORSO state
    (torso.pitch_angle) to compensate the head's home pose. When the torso's
    state subscriber has not delivered a parsed sample by then, that read blows
    up INSIDE the constructor — 0904, repeatedly but not every time:

        File ".../dexcontrol/core/component.py", line 326, in get_joint_pos
          if "pos" not in state:

    It is a startup race, not a robot fault. There is nothing to pre-wait on
    from out here either: the constructor's own _wait_for_components only waits
    for the CRITICAL components, and it has already returned by then. So the
    whole construction is retried, after dropping the half-built robot (its
    zenoh session closes with it, which is what lets the next attempt bind).

    The LAST exception is re-raised if every attempt fails: a persistent
    failure is a real problem (subsystem down, robot-server not up, client /
    server version mismatch) and should not be papered over.

    The head's "home" pose is preset to cfg.HEAD_INIT_PITCH_DEG on the way
    through, whether ``configs`` was passed or not — see _preset_head_init_pitch."""
    import gc

    from dexcontrol.robot import Robot

    configs = _preset_head_init_pitch(configs)
    n = int(cfg.ROBOT_CONNECT_ATTEMPTS if attempts is None else attempts)
    wait = float(cfg.ROBOT_CONNECT_DELAY_S if delay is None else delay)
    for k in range(1, max(1, n) + 1):
        try:
            return Robot(configs=configs)
        except Exception as e:  # noqa: BLE001 — any startup failure is worth one retry
            if k >= n:
                logger.error("[robot] connect failed {}/{} — giving up: {}: {}",
                             k, n, type(e).__name__, e)
                raise
            logger.warning("[robot] connect failed {}/{} ({}: {}) — dropping the "
                           "half-built robot and retrying in {:.0f}s",
                           k, n, type(e).__name__, e, wait)
        gc.collect()          # release the partial Robot -> closes its session
        time.sleep(wait)


def move_torso(torso, target, vel_scale: float, timeout: float) -> str:
    """Move the torso to ``target`` (rad) at ``vel_scale`` of its velocity
    ceiling, blocking until it arrives or ``timeout``. Returns a state string
    ("finished" when it got there).

    TWO dexcontrol generations, because the installed client and the source
    checked out next to this repo disagree — site-packages has 0.4.9, the
    dexcontrol/ tree is 0.5.0:

      * >= 0.5: ``move_to_joint_pos`` hands the motion to the robot-server
        motion plugin (trajectory smoothing + gravity comp) and returns a
        MotionHandle to wait on. This is also where ``set_joint_pos``'s
        wait_time got deprecated, in favour of exactly this call.
      * 0.4.9: there is no motion plugin and no deprecation. ``set_joint_pos``
        publishes ONE (pos, vel) target whose velocity _process_joint_velocities
        fills in as (|target-live| / ||target-live||) * the FULL joint velocity
        ceiling — that full-speed step IS the lurch. Same message, same
        direction vector, scaled down by vel_scale.

    Keep both until the client is upgraded: calling the 0.5 API on 0.4.9 is an
    AttributeError at the first ensure_ready()."""
    target = np.asarray(target, dtype=float)
    if hasattr(torso, "move_to_joint_pos"):
        handle = torso.move_to_joint_pos(target, relative=False,
                                         velocity_scale=float(vel_scale))
        return str(handle.wait(timeout=float(timeout)))
    live = np.asarray(torso.get_joint_pos(), dtype=float)
    lim = getattr(torso, "_joint_vel_limit", None)
    ceiling = 0.6 if lim is None else float(np.min(np.abs(np.asarray(lim, dtype=float))))
    d = np.abs(target - live)
    n = float(np.linalg.norm(d))
    vel = (np.zeros_like(d) if n < 1e-6 else (d / n) * (ceiling * float(vel_scale)))
    logger.info("[arm] torso (dexcontrol 0.4.x path): |vel|={:.3f} rad/s of the {:.3f} "
                "ceiling", float(np.linalg.norm(vel)), ceiling)
    torso.set_joint_pos_vel(target, joint_vel=vel, wait_time=float(timeout),
                            exit_on_reach=True)
    reached = bool(torso.is_joint_pos_reached(target, tolerance=0.05))
    return "finished" if reached else "timeout"


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
        self._q_cmd: np.ndarray | None = None   # last joints sent (see _send / _start_q)
        self._step_log_t = 0.0                   # solve_step slowdown log rate limit
        self._clamp_streak = 0                   # consecutive solve_step ticks spent re-configuring
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
        # LOG IT. Reachability is torso-dependent and the torso is read ONCE,
        # here, from the live robot — so every reach verdict in a run traces
        # back to this number, and it was never recorded. 0903 left an
        # unresolvable contradiction because of that: the view-park pose solves
        # offline at cfg.TORSO_JOINTS yet failed 26/26 times on the robot.
        logger.info("[arm] torso {} rad = {} deg ({})",
                    np.round(self._torso_q, 4),
                    np.round(np.rad2deg(self._torso_q), 1),
                    "LIVE from the robot" if self._robot is not None
                    else "cfg.TORSO_JOINTS, no robot attached")
        if self._robot is not None:
            d = np.abs(self._torso_q - np.asarray(cfg.TORSO_JOINTS, dtype=float))
            if float(d.max()) > 0.02:
                logger.warning("[arm] live torso differs from cfg.TORSO_JOINTS by "
                               "up to {:.3f} rad ({:.1f} deg) — the taught poses "
                               "were validated at the cfg value, so reach checks "
                               "and every taught column shift with this",
                               float(d.max()), float(np.rad2deg(d.max())))
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
        # The OTHER arm stays at q_ref in the full model — and that is what the
        # self-collision check sees. pin.neutral leaves it at ZEROS, a pose the
        # robot is never in: at torso stances that swing the arms in toward the
        # base (the lid place, torso j3 = -60 deg) the zeroed RIGHT arm
        # intersects base_0, so in_collision() came back True for every LEFT arm
        # config and every column pre-check refused a perfectly good column. Use
        # its live joints (its configured home when headless).
        other = "right" if self._side == "left" else "left"
        other_q = np.asarray(cfg.HOME_JOINTS_RIGHT if other == "right"
                             else cfg.HOME_JOINTS_LEFT, dtype=float)
        if self._robot is not None:
            try:
                other_q = np.asarray(getattr(self._robot, f"{other}_arm").get_joint_pos(),
                                     dtype=float)
            except Exception as e:  # noqa: BLE001
                logger.warning("[arm] could not read the {} arm ({}) — collision model "
                               "uses its home config instead", other, e)
        op = "R" if other == "right" else "L"
        for k in range(_ARM_DOF):
            jid = full.getJointId(f"{op}_arm_j{k + 1}")
            if jid < full.njoints:
                q_ref[full.idx_qs[jid]] = float(other_q[k])
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

        # IK-side joint-range margin: solve/validate inside JOINT_RANGE_FRAC of
        # each joint's URDF range (centered) so solutions keep off the hard
        # stops. IK-side only — URDF and dexcontrol's hardware clamps untouched.
        finite = np.isfinite(self._q_lo) & np.isfinite(self._q_hi)
        mid = 0.5 * (self._q_lo + self._q_hi)
        half = 0.5 * (self._q_hi - self._q_lo) * float(cfg.JOINT_RANGE_FRAC)
        self._q_lo = np.where(finite, mid - half, self._q_lo)
        self._q_hi = np.where(finite, mid + half, self._q_hi)
        self._model.lowerPositionLimit = np.asarray(self._q_lo, dtype=float)
        self._model.upperPositionLimit = np.asarray(self._q_hi, dtype=float)

        # IK-side-only tightening (this class's model/config_limit_gain only —
        # URDF and dexcontrol's hardware command clamping are untouched): keep
        # L_arm_j4 (elbow) from swinging above -0.5 rad, well inside its URDF
        # range of [-3.071, 0.244] (and inside the 95% band [-2.988, +0.161],
        # so this stays the binding constraint for j4).
        if self._side == "left":
            j4_idx = self._model.idx_qs[self._model.getJointId("L_arm_j4")]
            self._q_hi[j4_idx] = min(float(self._q_hi[j4_idx]), -0.5)
            self._model.upperPositionLimit[j4_idx] = self._q_hi[j4_idx]
        if not self.in_limits(self._home_seed):
            logger.warning("[arm] HOME_JOINTS_{} lies OUTSIDE the IK joint band — FK and "
                           "IK seeding from home get clipped; move it inside the band",
                           self._side.upper())

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

    def _send(self, q, v) -> None:
        """The ONE path to arm.set_joint_pos_vel — records the commanded joints
        so the next primitive can continue from them (_start_q).

        NOTE an accel limiter was tried here (0903) and REVERTED: clamping dv to
        _ruckig_amax did bound the commanded accel (84 -> 2.5 rad/s^2) but made
        the arm track WORSE, because the descent loops carry prev_q = sol.q (the
        unclamped IK solution) while this path sends the clamped command — the
        two chains diverge with nothing to pull them back, so the clamp lag
        stacks on top of the servo lag. Measured on the battery place column:
        cup deviation 33.8 -> 40.7mm, pitch 4.92 -> 5.88 deg, contact impact
        47.8 -> 60.5N. Any retry has to close that loop (seed the next solve
        from what was actually SENT) instead of limiting open-loop.
        """
        q = np.asarray(q, dtype=float)
        self._arm.set_joint_pos_vel(q, np.asarray(v, dtype=float))
        self._q_cmd = q

    def _start_q(self) -> np.ndarray:
        """Start config for the next primitive: the last COMMANDED joints while
        the live joints agree with them within CMD_CARRY_TOL_RAD (the measured
        q lags the command by the tracking error — planning from it would first
        step BACK to the lagging position at every primitive boundary); the
        live q otherwise (E-stop, hand-guiding at an operator gate, another
        controller: the command is stale)."""
        live = self._live_arm_q()
        q = self._q_cmd
        if q is None:
            return live
        gap = float(np.max(np.abs(live - q)))
        if gap <= float(cfg.CMD_CARRY_TOL_RAD):
            return q.copy()
        logger.debug("[arm] commanded q stale (live differs by {:.3f} rad) — starting from live", gap)
        return live

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

    def _track_trace(self, tag: str, target, prev_q, tick_ms: float,
                     extra: str = "") -> None:
        """TEMP diagnostic for "the descent isn't vertical": every per-tick
        vertical stream holds xy fixed, so the commanded path is vertical BY
        CONSTRUCTION — the deviation is one of two things, and this splits them.

        ik    target -> FK(prev_q): what the SOLVER gave up. Passes silently up
              to REACH_TOL_M (10mm) and near the reach ceiling it is almost all
              HORIZONTAL and shrinks as z drops, which walks the path sideways
              (0903 offline replay of the taught source column: +8.4mm x at
              z=0.78 -> +0.6mm at z=0.76). Zero on a centered column.
        track FK(prev_q) -> FK(live q): what the ROBOT did with the command.
              Covers the un-settled hover handoff (the descents continue from
              the approach's COMMANDED joints, with no arrival wait), the
              (sol.q-prev_q)/dt feedforward computed on a NOMINAL dt, and a
              stale torso model (read once at construction).
        tilt  live-vs-commanded EE rpy. The cup tip is SUCTION_LENGTH_M below
              the EE frame, so 1 deg here is ~2.7mm of visible tip swing —
              tip<= bounds the tip deviation as |track_xy| + tilt*L.
        tick  the REAL mean loop period against 1000/CONTROL_HZ, which is what
              the velocity feedforward assumes.

        Never raises: a headless/validation run has no live joints."""
        try:
            p_cmd, r_cmd = self.fk(np.asarray(prev_q, dtype=float))
            p_live, r_live = self.fk(self._live_arm_q())
        except Exception:  # noqa: BLE001 — diagnostics must never kill a stream
            return
        # target None (a planned joint-space leg): no Cartesian line to miss
        tgt = None if target is None else np.asarray(target, dtype=float)
        ik = None if tgt is None else (p_cmd - tgt) * 1000.0
        tr = (p_live - p_cmd) * 1000.0
        dr = np.rad2deg((np.asarray(r_live) - np.asarray(r_cmd) + np.pi)
                        % (2.0 * np.pi) - np.pi)
        tr_xy = float(np.hypot(tr[0], tr[1]))
        logger.info("[arm] {} track: z={:.4f} ik={} "
                    "track=({:+.1f},{:+.1f},{:+.1f})mm |track_xy|={:.1f}mm "
                    "tilt=({:+.2f},{:+.2f},{:+.2f})deg tip<={:.1f}mm "
                    "tick={:.1f}ms (nominal {:.1f}ms){}",
                    tag, float(p_cmd[2] if tgt is None else tgt[2]),
                    "-" if ik is None else
                    "({:+.1f},{:+.1f},{:+.1f})mm".format(ik[0], ik[1], ik[2]),
                    tr[0], tr[1], tr[2],
                    tr_xy, dr[0], dr[1], dr[2],
                    tr_xy + float(np.deg2rad(np.hypot(dr[0], dr[1]))
                                  * cfg.SUCTION_LENGTH_M * 1000.0),
                    tick_ms, 1000.0 / float(cfg.CONTROL_HZ), extra)

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
        escalating LM damping rather than giving up immediately; if every
        damping level diverges, run the ladder once more from a NEARBY seed
        (_retry_seed: the live measured joints, or the seed nudged 1 mrad).
        Nearby is the key: a far seed (e.g. home) is NOT safe here, since a
        live per-tick caller (move_ee_vertical, the suction descent loops)
        computes its velocity feedforward as (new_q - prev_q)/dt — landing on
        an unrelated branch would command a large, sudden jump, while a nearby
        seed stays on the same branch yet can escape a numeric blow-up. Only
        after the reseed retry also diverges is it reported unreachable
        (pos_err_m=inf trips every caller's existing REACH_TOL_M check) so it
        halts gracefully like any other unreachable target.
        """
        seed = self._home_seed if seed is None else np.asarray(seed, dtype=float)
        target = pin.SE3(_rpy_to_matrix(*rpy), np.asarray(pos, dtype=float))
        self._ee_task.set_target(target)
        orig_damping = self._ee_task.lm_damping
        try:
            for attempt in range(2):
                s = seed if attempt == 0 else self._retry_seed(seed)
                if attempt:
                    logger.warning("[arm] IK diverged from the given seed at every damping — "
                                    "retrying once from a nearby seed")
                for damping in (orig_damping, orig_damping * 1e3, orig_damping * 1e6):
                    self._ee_task.lm_damping = damping
                    configuration = self._configuration(s)
                    if min_motion:
                        self._posture_task.set_target(self._min_motion_target(configuration.q))
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
                        if damping != orig_damping or attempt:
                            logger.warning("[arm] IK diverged — recovered at damping={:.1e}{}",
                                            damping, " on the reseed retry" if attempt else "")
                        fk_pos, _ = self.fk(arm_q)
                        return PoseSolution(
                            q=arm_q,
                            converged=converged,
                            pos_err_m=float(np.linalg.norm(fk_pos - np.asarray(pos, dtype=float))),
                            in_collision=self.in_collision(arm_q),
                            in_limits=self.in_limits(arm_q),
                        )
            logger.warning("[arm] IK diverged to a non-finite configuration even after damping "
                            "retries and a reseed retry — treating as unreachable")
            return PoseSolution(q=seed, converged=False, pos_err_m=float("inf"),
                                 in_collision=True, in_limits=False)
        finally:
            self._ee_task.lm_damping = orig_damping

    def _retry_seed(self, seed: np.ndarray) -> np.ndarray:
        """A second IK seed near a failed one: the live measured joints when
        they differ from the seed (per-tick streams seed from the COMMANDED q;
        the measured q lags it by tracking error — close enough to stay on the
        same branch, different enough to escape a numeric blow-up), else the
        seed nudged by 1 mrad."""
        try:
            live = self._live_arm_q()
            if np.all(np.isfinite(live)) and not np.allclose(live, seed, atol=1e-6):
                return live
        except Exception:  # noqa: BLE001 — no live robot (offline solve)
            pass
        return seed + 1e-3

    def _min_motion_target(self, q_seed: np.ndarray) -> np.ndarray:
        """Nullspace posture target for a live (min-motion) solve: the seed
        itself, except joints within LIMIT_AVOID_MARGIN_RAD of their band edge,
        whose target is moved back inside by up to LIMIT_AVOID_STEP_RAD (see
        config). ``q_seed`` is a full reduced-model q, already band-clipped."""
        q_t = np.asarray(q_seed, dtype=float).copy()
        margin, step = float(cfg.LIMIT_AVOID_MARGIN_RAD), float(cfg.LIMIT_AVOID_STEP_RAD)
        if margin <= 0.0 or step <= 0.0:
            return q_t
        finite = np.isfinite(self._q_lo) & np.isfinite(self._q_hi)
        d_lo = np.where(finite, q_t - self._q_lo, np.inf)
        d_hi = np.where(finite, self._q_hi - q_t, np.inf)
        q_t += np.where(d_lo < margin, np.minimum(step, margin - d_lo), 0.0)
        q_t -= np.where(d_hi < margin, np.minimum(step, margin - d_hi), 0.0)
        return q_t

    def solve_step(self, prev_q: np.ndarray, p_from, p_to, rpy, dt: float
                   ) -> tuple[PoseSolution, np.ndarray]:
        """One tick of a streamed Cartesian leg under the joint-speed budget.

        Solves ``p_to`` min-motion from ``prev_q``. If the joint step exceeds
        the per-tick cap (ruckig vmax * dt — a joint hitting its band and the QP
        re-routing the motion through the others), the Cartesian step
        p_from -> p_to is shortened by the excess ratio and re-solved: the leg
        slows down where the joints can't keep up instead of kinking. A step
        still over STREAM_JOINT_STEP_MAX_X times the cap after shortening means
        the joints have to move even though the EE barely does — typically the
        stream starting from a not-quite-converged config (a hover a few mm /
        deg off its target near the reach edge) whose residual the first tick
        would otherwise fix in one jump. That tick is rate-limited instead:
        the joint step is clamped to the cap and the schedule holds at
        ``p_from``, so the arm converges over a few ticks and the leg then
        proceeds. Only a hold that lasts STREAM_CLAMP_MAX_S (the IK keeps
        demanding big steps: a real singular reshuffle) comes back as
        unreachable (pos_err inf) so the caller halts like any other
        unreachable tick. Returns (solution, pose actually scheduled) — callers
        advance their schedule to the returned pose, not to ``p_to``.
        """
        p_from = np.asarray(p_from, dtype=float)
        p_to = np.asarray(p_to, dtype=float)
        cap = self._ruckig_vmax * float(dt)
        sol = self.solve_pose(p_to, rpy, seed=prev_q, min_motion=True)
        if sol.pos_err_m > cfg.REACH_TOL_M:
            return sol, p_to
        excess = np.abs(sol.q - prev_q) / cap
        ratio = float(np.max(excess))
        if ratio <= 1.0:
            self._clamp_streak = 0
            return sol, p_to
        p_mid = p_from + (p_to - p_from) / ratio
        sol2 = self.solve_pose(p_mid, rpy, seed=prev_q, min_motion=True)
        ratio2 = float(np.max(np.abs(sol2.q - prev_q) / cap))
        if sol2.pos_err_m > cfg.REACH_TOL_M or ratio2 > float(cfg.STREAM_JOINT_STEP_MAX_X):
            # joints must move a lot for (almost) no EE motion: re-configure at the
            # capped joint speed with the schedule held, instead of jumping
            self._clamp_streak += 1
            max_ticks = int(float(cfg.STREAM_CLAMP_MAX_S) / float(dt))
            if sol2.pos_err_m > cfg.REACH_TOL_M or self._clamp_streak > max_ticks:
                logger.warning("[arm] joint step {:.1f}x the per-tick cap for a {:.2f}mm Cartesian "
                               "step, {} ticks in a row (err {:.1f}mm) — singular reshuffle, "
                               "halting the leg", ratio2,
                               float(np.linalg.norm(p_mid - p_from)) * 1000.0,
                               self._clamp_streak, sol2.pos_err_m * 1000.0)
                self._clamp_streak = 0
                return (PoseSolution(q=np.asarray(prev_q, dtype=float), converged=False,
                                     pos_err_m=float("inf"), in_collision=False, in_limits=True),
                        p_from)
            if self._clamp_streak == 1:
                logger.info("[arm] joint step {:.1f}x the per-tick cap for a {:.2f}mm Cartesian "
                            "step — re-configuring at the capped joint speed, schedule held",
                            ratio2, float(np.linalg.norm(p_mid - p_from)) * 1000.0)
            q_cmd = np.asarray(prev_q, dtype=float) + (sol2.q - prev_q) / ratio2
            return (PoseSolution(q=q_cmd, converged=False, pos_err_m=sol2.pos_err_m,
                                 in_collision=sol2.in_collision, in_limits=sol2.in_limits),
                    p_from)
        self._clamp_streak = 0
        now = time.monotonic()
        if now - self._step_log_t > 1.0:
            self._step_log_t = now
            j = int(np.argmax(excess))
            logger.info("[arm] joint-speed cap: Cartesian step shortened {:.2f}x "
                        "(j{} would run {:.2f} rad/s, cap {:.2f})", ratio, j + 1,
                        float(np.abs(sol.q[j] - prev_q[j])) / float(dt),
                        float(cap[j]) / float(dt))
        return sol2, p_mid

    def _posture_mid_arm(self) -> np.ndarray:
        return np.array([self._posture_mid[self._model.idx_qs[j]] for j in self._arm_joint_ids])

    def clip_to_band(self, arm_q, label: str = "config") -> np.ndarray:
        """Clip a joint vector into this model's IK band and say so if it moved.

        The band is the URDF range narrowed by cfg.JOINT_RANGE_FRAC (plus the
        left j4 cap), i.e. the range every solve and every reach verdict here is
        answered inside. A TAUGHT config read off the robot can sit just
        outside it — the robot's own limits are wider — and then it is a config
        the model calls invalid while move_joints would happily command it.
        Clipping keeps the two consistent; the log line is there because it
        silently edits a measurement.
        """
        q = np.asarray(arm_q, dtype=float)
        out = np.clip(q, self._q_lo, self._q_hi)
        d = out - q
        if np.any(np.abs(d) > 1e-9):
            for k in np.flatnonzero(np.abs(d) > 1e-9):
                logger.warning("[arm] {} j{} {:+.4f} is outside the IK band "
                               "[{:+.4f}, {:+.4f}] — clipped by {:+.4f} rad "
                               "({:+.2f} deg)", label, int(k) + 1, q[k],
                               self._q_lo[k], self._q_hi[k], d[k],
                               float(np.rad2deg(d[k])))
        return out

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

    def column_reachable(self, x: float, y: float, rpy, z_top: float, z_bottom: float,
                         seed: np.ndarray | None = None, quiet: bool = False) -> bool:
        """IK pre-check of a straight vertical column at (x, y), z_top down to
        z_bottom in DESCENT_CHECK_STEP_M steps, warm-chained so the solves stay
        on one branch: every step must reach (REACH_TOL_M), sit inside the joint
        band and be self-collision free. Command-side only — it says nothing
        about how the arm will TRACK the column."""
        zs = np.arange(float(z_top), float(z_bottom) - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
        if zs[-1] > z_bottom + 1e-9:
            zs = np.append(zs, float(z_bottom))
        q = seed
        for z in zs:
            sol = self.solve_pose((x, y, float(z)), rpy, seed=q, min_motion=q is not None)
            if not (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision):
                if quiet:                 # a search over many candidates, not a pre-flight
                    return False
                logger.warning("[arm] column pre-check FAILED at z={:.3f} (err={:.1f}mm, in_limits={}, "
                               "collision={}) for xy=({:.3f},{:+.3f})", z, sol.pos_err_m * 1000.0,
                               sol.in_limits, sol.in_collision, x, y)
                return False
            q = sol.q
        return True

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
        q_start = self._start_q()
        traj = self.plan_joint_traj(q_start, np.asarray(q_goal, dtype=float))
        dt = 1.0 / float(cfg.CONTROL_HZ)
        logger.info("[arm] move_joints: {:.2f}s, {} steps", traj.duration,
                    max(1, int(np.ceil(traj.duration / dt))))
        self._stream_traj(traj, dt)

    def _stream_traj(self, traj: Trajectory, dt: float, tick_cb=None,
                     trace_tag: "str | None" = None) -> bool:
        """Stream one Ruckig trajectory, sampled at the WALL-CLOCK time since the
        start (TickPacer) so the (pos, vel) pairs stay consistent when a tick
        overruns; the last sample lands exactly on traj.duration (at rest).

        ``tick_cb(ee_z, q)`` is an optional per-tick guard; a truthy return
        halts the stream in place and returns False (see
        suction._approach_and_hover's to_creep_z, where a planned stretch
        replaces one the descent loop used to force-monitor). True = ran to
        completion."""
        # trace the PLANNED stream too: the per-tick streams were instrumented
        # first, which left the joint-space legs as the one blind spot — and
        # they now carry the whole above-creep stretch, so "the cup pitches
        # before the descent" had no measurement behind it (0903).
        trace_s = float(cfg.DESCENT_TRACE_S) if trace_tag else 0.0
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        pace = TickPacer(dt)
        while True:
            t = pace.wait()
            pos, vel, _acc = traj.at_time(min(t, traj.duration))
            self._send(pos, vel)
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                # target=None: a planned leg has no straight-line Cartesian
                # target to be short of (the joint-space arc IS the path), so
                # the ik= field would read a meaningless 0.0. Only track= and
                # tilt= mean anything here.
                self._track_trace(trace_tag, None,
                                  np.asarray(pos, dtype=float),
                                  tick_acc / tick_n * 1000.0)
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0
            if tick_cb is not None:
                q = np.asarray(pos, dtype=float)
                if tick_cb(float(self.fk(q)[0][2]), q):
                    self._send(q, np.zeros(_ARM_DOF))
                    return False
            if t >= traj.duration:
                break
        return True

    def _plan_seg(self, q0, v0, q1, v1) -> "Trajectory | None":
        """One jerk-limited segment with explicit boundary velocities (Ruckig).
        None if Ruckig rejects the boundary conditions (only possible with
        nonzero velocities — the caller falls back to zero-velocity junctions)."""
        inp = InputParameter(_ARM_DOF)
        inp.current_position = list(map(float, q0))
        inp.current_velocity = list(map(float, v0))
        inp.current_acceleration = [0.0] * _ARM_DOF
        inp.target_position = list(map(float, q1))
        inp.target_velocity = list(map(float, v1))
        inp.target_acceleration = [0.0] * _ARM_DOF
        inp.max_velocity = list(map(float, self._ruckig_vmax))
        inp.max_acceleration = list(map(float, self._ruckig_amax))
        inp.max_jerk = list(map(float, self._ruckig_jmax))
        traj = Trajectory(_ARM_DOF)
        if self._otg.calculate(inp, traj) not in (Result.Working, Result.Finished):
            return None
        return traj

    def move_joints_through(self, qs, stops=(), at_waypoint=None,
                            tick_cb=None) -> None:
        """Stream ONE continuous trajectory through the joint waypoints `qs` —
        the multi-waypoint form of move_joints, without the full stop at every
        waypoint that chained move_joints calls produce.

        Junctions are crossed at cfg.JOINT_BLEND_FRAC of vmax on the joints
        that keep direction across the junction (direction-reversing joints
        cross at 0), scaled down for hops shorter than
        JOINT_BLEND_FULL_DIST_RAD. Zero-velocity (full-stop) waypoints: the
        last one, indices in `stops`, and indices with an `at_waypoint`
        callback — the stream pauses to run the callback, and pausing the
        command stream mid-flight would step the velocity.

        Blending rounds corners off the waypoint-to-waypoint path, so every
        blended segment is sampled and collision-checked BEFORE any motion;
        a segment that fails to plan or collides drops its junction
        velocities to zero (the old stop-at-waypoint behavior) and is
        re-checked. A segment that collides even unblended executes anyway —
        move_joints never path-checked, and inventing a new failure mode
        here would strand sequences that ran fine before."""
        qs = [np.asarray(q, dtype=float) for q in qs]
        at_waypoint = at_waypoint or {}
        stops = set(stops) | set(at_waypoint)
        q_live = self._start_q()
        n = len(qs)
        alpha = float(cfg.JOINT_BLEND_FRAC)
        full = max(float(cfg.JOINT_BLEND_FULL_DIST_RAD), 1e-6)
        vs: list[np.ndarray] = []
        for k in range(n):
            if k == n - 1 or k in stops or alpha <= 0.0:
                vs.append(np.zeros(_ARM_DOF))
                continue
            d_in = qs[k] - (qs[k - 1] if k else q_live)
            d_out = qs[k + 1] - qs[k]
            keep = ((np.sign(d_in) == np.sign(d_out))
                    & (np.abs(d_in) > 1e-6) & (np.abs(d_out) > 1e-6))
            scale = np.minimum(1.0, np.minimum(np.abs(d_in), np.abs(d_out)) / full)
            vs.append(np.where(keep, alpha * self._ruckig_vmax * np.sign(d_out) * scale, 0.0))
        # plan + pre-flight: zeroing a junction changes BOTH adjacent segments,
        # so a fallback at segment k re-plans k-1 too
        trajs: list = [None] * n
        chk_dt = 0.05
        k = 0
        while k < n:
            v0 = vs[k - 1] if k else np.zeros(_ARM_DOF)
            q0 = qs[k - 1] if k else q_live
            traj = self._plan_seg(q0, v0, qs[k], vs[k])
            blended = bool(np.any(v0)) or bool(np.any(vs[k]))
            bad = traj is None
            if not bad and blended:
                for t in np.arange(chk_dt, traj.duration, chk_dt):
                    if self.in_collision(np.asarray(traj.at_time(t)[0])):
                        bad = True
                        break
            if bad and blended:
                logger.warning("[arm] blended segment {} {} — falling back to a "
                               "full stop at its junctions", k,
                               "infeasible" if traj is None else "clips a collision")
                vs[k] = np.zeros(_ARM_DOF)
                if k and np.any(vs[k - 1]):
                    vs[k - 1] = np.zeros(_ARM_DOF)
                    k -= 1
                continue
            if traj is None:
                raise RuntimeError(f"joint segment {k} unplannable at zero boundary velocity")
            trajs[k] = traj
            k += 1
        dt = 1.0 / float(cfg.CONTROL_HZ)
        logger.info("[arm] move_joints_through: {} waypoints, {:.2f}s total",
                    n, sum(t.duration for t in trajs))
        for k, traj in enumerate(trajs):
            if not self._stream_traj(traj, dt, tick_cb=tick_cb,
                                     trace_tag=f"planned[{k + 1}/{len(trajs)}]"):
                logger.warning("[arm] planned stream halted by its guard at "
                               "segment {}/{}", k + 1, len(trajs))
                return
            cb = at_waypoint.get(k)
            if cb is not None:
                cb()

    def move_ee(self, pos, rpy, quiet: bool = True) -> np.ndarray | None:
        """Move the EE frame to an absolute base_link pose (solve IK from the live
        config, min-motion, then move_joints). Returns the commanded target joints
        (so a caller can continue a stream from exactly there), or None if the
        target is beyond REACH_TOL_M. ``quiet`` demotes the within-tol shortfall
        log to debug (for transport-height legs where a few mm is expected and
        recovered lower down — not the alignment-critical descent/sweep legs)."""
        sol = self.solve_pose(pos, rpy, seed=self._start_q(), min_motion=True)
        if sol.pos_err_m > cfg.REACH_TOL_M:
            logger.error("[arm] target unreachable ({:.1f}mm short) — not moving", sol.pos_err_m * 1000)
            return None
        if not sol.converged:
            log = logger.debug if quiet else logger.warning
            log("[arm] target {:.1f}mm short (within reach tol) — moving", sol.pos_err_m * 1000)
        self.move_joints(sol.q)
        return sol.q

    def move_ee_line(self, pos, rpy, speed: float | None = None, stop_fn=None,
                     trace_tag: str = "line") -> "np.ndarray | None":
        """Straight-line EE move to an absolute base_link ``pos``, orientation
        held EVERY tick (per-tick warm IK stream) — the 3-D generalisation of
        move_ee_vertical, for a leg whose direction is not vertical.

        move_ee only constrains the endpoints; its joint-space path arcs
        sideways in between, which is fine at transport height and wrong when
        the fingers have to enter beside a held part. Speed defaults to the
        descent creep (``DESCENT_CREEP_SPEED_M_S``) with the same smoothstep
        ramp-in from rest and decel into the target the descents use, so there
        is no velocity step at either end.

        Returns the last commanded joints, or None if a tick's IK falls beyond
        REACH_TOL_M (halts in place — partial motion, logged). ``stop_fn`` is
        polled once per tick AFTER the tick's command; True halts the stream.
        """
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = self._start_q()
        p_now = np.asarray(self.fk(prev_q)[0], dtype=float)
        p_goal = np.asarray(pos, dtype=float)
        total = float(np.linalg.norm(p_goal - p_now))
        if total < 1e-4:
            return prev_q
        v_max = float(cfg.DESCENT_CREEP_SPEED_M_S if speed is None else speed)
        ramp = max(float(cfg.DESCENT_RAMP_S), 1e-6)
        band = max(v_max * ramp, 1e-6)
        logger.info("[arm] move_ee_line: {} -> {} ({:.1f}mm at {:.3f} m/s)",
                    np.round(p_now, 4), np.round(p_goal, 4), total * 1000, v_max)
        elapsed, deadline = 0.0, time.perf_counter() + total / v_max + 4.0 * ramp + 3.0
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        pace = TickPacer(dt)
        while True:
            d = p_goal - p_now
            dist = float(np.linalg.norm(d))
            if dist <= 5e-4:
                break
            if time.perf_counter() > deadline:
                self._send(prev_q, np.zeros(_ARM_DOF))
                logger.warning("[arm] line move timed out {:.1f}mm short at {} — "
                               "halting", dist * 1000, np.round(p_now, 4))
                return None
            r_in = min(1.0, elapsed / ramp)
            r_out = min(1.0, dist / band)
            f = min(r_in, r_out)
            # floor: a pure smoothstep reaches zero speed and never arrives
            v = v_max * max(f * f * (3.0 - 2.0 * f), 0.05)
            p_next = p_now + d / dist * min(v * dt, dist)
            sol, p_sched = self.solve_step(prev_q, p_now, p_next, rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                self._send(prev_q, np.zeros(_ARM_DOF))
                logger.warning("[arm] line move stalled {:.1f}mm short at {} — "
                               "halting", sol.pos_err_m * 1000, np.round(p_now, 4))
                return None
            self._send(sol.q, (sol.q - prev_q) / dt)
            p_now, prev_q = np.asarray(p_sched, dtype=float), sol.q
            if stop_fn is not None and stop_fn():
                self._send(prev_q, np.zeros(_ARM_DOF))
                logger.info("[arm] move_ee_line stopped by the caller at {}",
                            np.round(p_now, 4))
                return prev_q
            elapsed += dt
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                self._track_trace(trace_tag, p_now, prev_q,
                                  tick_acc / tick_n * 1000.0)
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0
        self._send(prev_q, np.zeros(_ARM_DOF))
        return prev_q

    def move_ee_vertical(self, z_target: float, rpy, stop_fn=None,
                         creep_out_m: float = 0.0) -> np.ndarray | None:
        """Straight vertical EE move to base-frame z_target, x,y,rpy held EVERY
        tick (per-tick warm IK stream) — the free-air analog of the suction
        descent legs. move_ee only constrains the endpoints; its joint-space
        trajectory arcs sideways in between, so lifts holding a part must use
        this instead. Speed reuses the descent budget: smoothstep ramp-in from
        rest, cruise at DESCENT_APPROACH_SPEED_M_S, blend to creep into the
        target. Returns the last commanded joints, or None if a tick's IK falls
        beyond REACH_TOL_M (halts in place — partial motion, logged).

        ``creep_out_m``: hold CREEP speed for the first this-many metres of
        travel, then blend up to cruise — the MIRROR of the decel-into-target
        band, for a leg that starts somewhere it must leave carefully (a lift
        out of a slot). The two directions were not symmetric before: a descent
        ends with DESCENT_CREEP_GAP_M of creep, which is exactly where the
        tracking deviation it built up at cruise decays away (0903: 27-55mm at
        0.2-0.3 m/s, 0.3-3mm once creeping), while a lift started at cruise
        immediately and never shed it — 59mm of lateral and 13.7 deg of pitch
        with the part in the cup, all of it above the walls but enough to shift
        the part on the cup. Pass DESCENT_CREEP_GAP_M to come back up the way
        the descent came down.
        ``stop_fn``: optional zero-arg callable polled once per tick AFTER the
        tick's command; True halts the stream in place (zero velocity at the
        last commanded joints) and returns them — a contact guard hook. The
        caller tells a stopped leg from a finished one by its own flag / FK."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = self._start_q()
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        z_target = float(z_target)
        direction = 1.0 if z_target >= z else -1.0
        z_start = z
        creep_out = max(0.0, float(creep_out_m))
        fast, creep = cfg.DESCENT_APPROACH_SPEED_M_S, cfg.DESCENT_CREEP_SPEED_M_S
        band = max(float(cfg.DESCENT_CREEP_BLEND_M), 1e-6)
        elapsed = 0.0
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        logger.info("[arm] move_ee_vertical: z {:.4f} -> {:.4f} (xy held at {:.4f},{:.4f})",
                    z, z_target, x, y)
        pace = TickPacer(dt)
        while abs(z_target - z) > 1e-4:
            dist = abs(z_target - z)
            gone = abs(z - z_start)
            if dist >= band:
                base = fast
            else:
                f = dist / band
                f = f * f * (3.0 - 2.0 * f)                # smoothstep decel into target
                base = creep + (fast - creep) * f
            if creep_out > 0.0 and gone < creep_out + band:
                # mirror of the above, measured from the START: creep out of the
                # first creep_out metres, then smoothstep up to cruise over the
                # same band. min() so a short leg never speeds up past the
                # decel-into-target shape.
                g = max(0.0, (gone - creep_out) / band)
                g = min(1.0, g)
                g = g * g * (3.0 - 2.0 * g)
                base = min(base, creep + (fast - creep) * g)
            r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
            speed = base * (r * r * (3.0 - 2.0 * r))       # smoothstep ramp-in
            z_next = z + direction * min(speed * dt, dist)
            sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                self._send(prev_q, np.zeros(_ARM_DOF))
                logger.warning("[arm] vertical move stalled {:.1f}mm short at z={:.4f} — halting",
                               sol.pos_err_m * 1000, z)
                return None
            z_next = float(p[2])
            self._send(sol.q, (sol.q - prev_q) / dt)
            z, prev_q = z_next, sol.q
            if stop_fn is not None and stop_fn():
                self._send(prev_q, np.zeros(_ARM_DOF))
                logger.info("[arm] move_ee_vertical stopped by the caller at z={:.4f}", z)
                return prev_q
            elapsed += dt
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                self._track_trace("vertical", [x, y, z], prev_q,
                                  tick_acc / tick_n * 1000.0)
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0
        self._send(prev_q, np.zeros(_ARM_DOF))
        return prev_q

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------
    def software_estop_active(self) -> bool:
        estop = getattr(self._robot, "estop", None)
        return bool(estop is not None and estop.is_software_estop_enabled())

    def pin_torso(self, target=None, vel_scale: float | None = None,
                  timeout: float | None = None) -> None:
        """Move the torso to ``target`` (default cfg.TORSO_JOINTS) and hold it
        there, then rebuild the reduced model at the stance actually REACHED so
        the model matches reality.

        The taught poses were validated at TORSO_JOINTS; live-torso drift shifts
        the arm base and shrinks reach at the transport height (a horizontal
        shortfall that offsets the descent). Pinning removes that drift.
        ``target`` is a parameter (rad) because the demo will need to CHANGE the
        stance mid-run, not only correct drift.

        Speed is cfg.TORSO_VEL_SCALE of the torso's ceiling, via ``move_torso``
        (which picks the right call for the installed dexcontrol). It used a
        bare set_joint_pos(wait_time=...) until 0904 — that ships the target at
        the FULL velocity ceiling, fine for the mm of drift it was written for,
        a lurch for a real stance change. The model is rebuilt from the live
        reading afterwards, not from ``target``, so a motion that stops short
        (limit, obstruction, timeout) still leaves the model matching the
        robot."""
        if self._robot is None:
            return
        target = np.asarray(cfg.TORSO_JOINTS if target is None else target, dtype=float)
        vel = float(cfg.TORSO_VEL_SCALE if vel_scale is None else vel_scale)
        live = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
        delta = float(np.max(np.abs(target - live)))
        if timeout is None:
            # generous: a finished motion returns from wait() immediately, so
            # only a genuinely stuck torso ever pays this (the hardware
            # velocity ceiling the scale multiplies is not exposed here).
            timeout = max(10.0, 10.0 * delta / max(vel, 1e-3))
        logger.info("[arm] pinning torso {} -> {} (rad, {:.3f} rad move at "
                    "velocity_scale={:.2f}, {:.0f}s timeout)",
                    np.round(live, 3), np.round(target, 3), delta, vel, timeout)
        state = move_torso(self._robot.torso, target, vel, timeout)
        reached = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
        off = float(np.max(np.abs(reached - target)))
        if state != "finished" or off > 0.02:
            logger.warning("[arm] torso motion ended '{}' {:.3f} rad ({:.1f} deg) from "
                           "the target — modelling at where it IS, so reach verdicts "
                           "stay honest", state, off, float(np.rad2deg(off)))
        # Rebuild the reduced model at the stance reached (it was built from the
        # live torso at construction, which may have drifted or since moved).
        self._setup_model(reached)
        self._setup_ik()
        self._setup_collision()

    def ensure_ready_live_torso(self, release_estop: bool = False,
                                vel_scale: float | None = None) -> bool:
        """Arm ready at WHATEVER torso stance is live, unlike ensure_ready which
        moves it to cfg.TORSO_JOINTS. For tools that work in a non-demo stance
        (the EE jog, the lid probe): the live torso IS the stance, the model was
        already built from it at construction, and it is commanded to its own
        reading so it holds while the arm moves."""
        if self.software_estop_active():
            if not release_estop:
                logger.warning("[arm] software E-Stop active — release it and retry")
                return False
            self._robot.estop.deactivate()
            time.sleep(0.5)
        self._arm.set_modes(["position"] * _ARM_DOF)
        if self.software_estop_active():
            return False
        live = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
        logger.info("[arm] holding the torso at its live stance {} rad",
                    np.round(live, 4))
        move_torso(self._robot.torso, live,
                   float(cfg.TORSO_VEL_SCALE if vel_scale is None else vel_scale),
                   5.0)
        return True

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
        # Always take the taught stance, however far away the torso is. This
        # used to REFUSE past 0.05 rad, from when the only way there was a
        # full-speed position step; pin_torso is velocity-scaled now, and --lid
        # ends a run at a different stance on purpose, so refusing just meant a
        # run could not be started after one.
        live = np.asarray(self._robot.torso.get_joint_pos(), dtype=float)
        gap = float(np.max(np.abs(live - np.asarray(cfg.TORSO_JOINTS, dtype=float))))
        if gap > 0.05:
            logger.warning("[arm] live torso is {:.3f} rad ({:.1f} deg) off the taught "
                           "stance — moving it there at velocity_scale={:.2f}; CLEAR "
                           "THE TORSO", gap, float(np.rad2deg(gap)), cfg.TORSO_VEL_SCALE)
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
