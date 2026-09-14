"""Right-arm stand pick + lay-down place: standing cylinders -> the black bin.

For each cylinder (one cycle per --slots entry):

    detect the CYLINDERS only — the bin is not looked at yet
         [chassis: if the nearest cylinder is outside the pick window in y, the
          base strafes it to the nearest spot inside. The base only ever
          strafes; x is the arm's reach, driven only if a plan fails outright]
         -> home -> just above the standoff -> short straight descent to pinch
            height
         -> straight-line entry from behind (+x), fingers open left/right,
            wrench guard stops an early bump
         -> soft close (position-checked, one corrective re-grip)
         -> straight up to transport height -> carry upright over the slot
            [chassis: the base drives BLIND toward the bin meanwhile, by the
             learned total displacement minus the pre-pick move]
         -> turn in the air, the same path every cycle (tool ends pointing down,
            cylinder lying along base x)
         [chassis: NOW detect the bin — the only place it is measurable — and
          send the HAND to it, base unmoved, spinning the WRIST by the bin's
          yaw so the cylinder lands parallel to its walls — everything up to
          the hover pose is identical every cycle and only the lay-down turns.
          That same reading teaches the next cycle its carry and place centre
          (but never the yaw: the base loses heading when it strafes). Bin not
          measurable, or further off than STAND_CARRY_CORRECT_MAX_M? the
          cylinder stays held and the run stops]
         -> straight down until the cylinder rests on the bin floor (touchdown
            guard) -> open 4 mm -> straight up -> open -> home
            [chassis: the base strafes back by the y it moved this cycle, while
             the arm turns back and homes. x is not undone]

Positions come from the head camera by default (show_detect: both OBB models,
nearest cylinder first); --no-detect uses the typed --cyl-x/--cyl-y and
--bin-x/--bin-y instead.

Run from LGES/:
    python -m ik_demo.stand_place                             # DEFAULT (= the old --detect --chassis): camera
                                                              # finds the cylinders + bin, the chassis brings the
                                                              # bin into reach, four cylinders 5 cm apart, the
                                                              # base returns to the cycle start after each place
    python -m ik_demo.stand_place --dry                       # camera -> plan only; nothing moves but the head
    python -m ik_demo.stand_place --slots 0                   # one cylinder into the bin centre
    python -m ik_demo.stand_place --pick-only                 # grip, lift 10 cm, set back down, release
    python -m ik_demo.stand_place --no-chassis                # place at the detected bin, arm reach only
    python -m ik_demo.stand_place --no-detect                 # typed positions on the robot
    python -m ik_demo.stand_place --no-detect --dry           # offline planner, no robot at all
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

import threading

from . import config as cfg
from .arm import ArmMover
from .go_home import safe_home
from .gripper import GripperMover


@dataclass
class Args:
    # cylinder default = where the user hand-flew the grasp on 0906 (pose_tune, arm_center
    # frame -> base_link at TORSO_JOINTS; pinch EE x 0.860 + reach 0.073)
    cyl_x: float = 0.933          # standing cylinder centre, base_link (m) — used only with --no-detect
    cyl_y: float = -0.28
    bin_x: float = 0.95           # black bin centre, base_link (m) — used only with --no-detect
    bin_y: float = -0.05
    slots: str = "-7.5,-2.5,2.5,7.5"   # one cycle per entry: slot y offsets from the bin centre, cm, left is +
                                       # (default: four cylinders 5 cm apart, user's choice 0906); "0" = one, centred
    dry: bool = False             # plan + report only, no robot
    pick_only: bool = False       # grip, lift STAND_LIFT_TEST_M, set back down, release (no bin)
    keep: bool = False            # skip the final home
    yaw_align: bool = True        # turn the slot row (and the laid-down cylinder) to follow the bin's
                                  # long axis, instead of always lying along base x
    detect: bool = True           # find the cylinders + bin with the head camera (show_detect), every cycle;
                                  # --no-detect uses the typed cyl_*/bin_* above instead
    chassis: bool = True          # with detection: drive the base so the bin sits at STAND_BIN_PLACE_XY
                                  # (slowly, while the arm carries + turns), then place there;
                                  # --no-chassis places at the detected bin, arm reach only
    speed: float = cfg.STAND_SPEED_SCALE   # right-arm joint-speed scale for this run; the default is
                                           # this job's own STAND_SPEED_SCALE, not robot.py's
                                           # SPEED_SCALE_RIGHT (which the session keeps)


# ---------------------------------------------------------------------------
# Planning (headless-capable)
# ---------------------------------------------------------------------------

@dataclass
class ObjectPlan:
    label: str
    rpy_pick: tuple[float, float, float]
    rpy_place: tuple[float, float, float]
    approach_dir: np.ndarray              # unit vector of the entry, base xy-plane
    p_standoff: np.ndarray                # EE origin, entry start
    p_pinch: np.ndarray                   # EE origin, fingers around the object
    z_transport: float
    p_place_hi: np.ndarray                # EE origin above the slot at z_transport
    z_place: float                        # EE z with the object resting on the bin floor
    q_rotate: list[np.ndarray] = field(default_factory=list)   # in-air turn waypoints
    ok: bool = False
    problems: list[str] = field(default_factory=list)


def _or(q, fallback: np.ndarray) -> np.ndarray:
    return fallback if q is None else q


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in Rotation.from_matrix(R).as_euler("xyz"))


def plan_object(g: GripperMover, label: str, x: float, y: float, yaw: float,
                slot_xy: tuple[float, float], seed: np.ndarray) -> ObjectPlan:
    """Geometry + IK pre-flight for one object. ``yaw`` rotates the entry
    direction about base z (0 = straight along +x). ``seed`` warm-starts the
    solves; pass the config the arm will actually be in (the task home).

    Plans the lay-down at STAND_PLACE_RPY, the orientation the in-air turn was
    proven into. Lining the row up with the bin is a WRIST SPIN applied later,
    at the place point, from the yaw measured in that same cycle."""
    obj = cfg.STAND_OBJECTS[label]
    z_grasp = float(cfg.STAND_DESK_Z_M) + float(obj["grasp_h"])
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    R_pick = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler(
        "xyz", cfg.STAND_APPROACH_RPY).as_matrix()
    reach = float(cfg.STAND_FINGER_LENGTH_M) - float(cfg.STAND_PAD_DEPTH_M)
    p_pinch = np.array([x, y, z_grasp]) - d * reach
    p_standoff = p_pinch - d * float(cfg.STAND_STANDOFF_M)
    z_transport = float(cfg.STAND_TRANSPORT_Z_M)
    z_place = float(cfg.STAND_BIN_FLOOR_Z_M) + float(obj["lying_half"]) + reach
    plan = ObjectPlan(label, _rpy(R_pick), tuple(cfg.STAND_PLACE_RPY), d, p_standoff, p_pinch,
                      z_transport, np.array([slot_xy[0], slot_xy[1], z_transport]), z_place)

    def check(name: str, pos, rpy, q_seed):
        """Warm solve from q_seed; if that branch falls short, retry from the
        task home (the arm may then hop branches at this waypoint — logged)."""
        sol = g.solve_pose(pos, rpy, seed=q_seed, min_motion=True)
        good = sol.converged and sol.in_limits and not sol.in_collision
        if not good and q_seed is not seed:
            alt = g.solve_pose(pos, rpy, seed=seed, min_motion=True)
            if alt.converged and alt.in_limits and not alt.in_collision:
                logger.warning("[{}] {}: warm branch {:.1f}mm short — re-seeded from home "
                               "(joint hop {:.2f} rad)", label, name, sol.pos_err_m * 1000,
                               float(np.max(np.abs(alt.q - q_seed))))
                return alt.q
        if not good:
            plan.problems.append(f"{name}: err {sol.pos_err_m * 1000:.1f}mm converged={sol.converged} "
                                 f"in_limits={sol.in_limits} collision={sol.in_collision}")
        return sol.q if good else None

    # entry: joint-space to just above the standoff, a short column down, then
    # the straight line in. (A column from transport height was tried first and
    # warm-chained into a self-colliding branch near the body, 0906 dry run.)
    z_pre = p_standoff[2] + float(cfg.STAND_PRE_LIFT_M)
    q = check("pre_standoff", [p_standoff[0], p_standoff[1], z_pre], plan.rpy_pick, seed)
    if q is not None and not g.column_reachable(p_standoff[0], p_standoff[1], plan.rpy_pick,
                                                z_pre, p_standoff[2], seed=q):
        plan.problems.append("standoff column: see [arm] column pre-check")
    q = check("standoff", p_standoff, plan.rpy_pick, _or(q, seed))
    n = int(np.ceil(float(cfg.STAND_STANDOFF_M) / 0.01))
    for k in range(1, n + 1):
        p = p_standoff + (p_pinch - p_standoff) * k / n
        q = check(f"entry[{k}/{n}]", p, plan.rpy_pick, _or(q, seed))
        if q is None:
            break
    q_pinch = q
    # lift column with the object (tool still forward), carry over the slot
    if q_pinch is not None and not g.column_reachable(p_pinch[0], p_pinch[1], plan.rpy_pick,
                                                      z_transport, p_pinch[2], seed=q_pinch):
        plan.problems.append("lift column: see [arm] column pre-check")
    q = check("lift_top", [p_pinch[0], p_pinch[1], z_transport], plan.rpy_pick, _or(q_pinch, seed))
    q = check("carry_hi", plan.p_place_hi, plan.rpy_pick, _or(q, seed))
    # the in-air turn, above the slot: shortest rotation approach -> place
    R_place = Rotation.from_euler("xyz", plan.rpy_place).as_matrix()
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_pick, R_place])))
    steps = int(cfg.STAND_ROTATE_STEPS)
    for k in range(1, steps + 1):
        rpy_k = _rpy(slerp([k / steps]).as_matrix()[0])
        q = check(f"turn[{k}/{steps}]", plan.p_place_hi, rpy_k, _or(q, seed))
        if q is None:
            break
        plan.q_rotate.append(q)
    # down to the floor
    if q is not None and not g.column_reachable(plan.p_place_hi[0], plan.p_place_hi[1], plan.rpy_place,
                                                z_transport, z_place - float(cfg.STAND_PLACE_OVERSHOOT_M),
                                                seed=q):
        plan.problems.append("place column: see [arm] column pre-check")
    plan.ok = not plan.problems
    return plan


def describe(plan: ObjectPlan) -> None:
    obj = cfg.STAND_OBJECTS[plan.label]
    logger.info("[{}] pinch at EE ({:.3f},{:+.3f},{:.3f}) -> object centre z {:.3f} ({:.0f}mm above the desk); "
                "entry from ({:.3f},{:+.3f}) along ({:+.2f},{:+.2f}); transport z {:.3f}; "
                "slot ({:.3f},{:+.3f}) resting EE z {:.3f}; turn waypoints {}",
                plan.label, *plan.p_pinch, plan.p_pinch[2], obj["grasp_h"] * 1000,
                plan.p_standoff[0], plan.p_standoff[1], plan.approach_dir[0], plan.approach_dir[1],
                plan.z_transport, plan.p_place_hi[0], plan.p_place_hi[1], plan.z_place,
                len(plan.q_rotate))
    if plan.ok:
        logger.info("[{}] plan OK", plan.label)
    else:
        for p in plan.problems:
            logger.error("[{}] {}", plan.label, p)


def build_plans(g: GripperMover, a: Args) -> list[ObjectPlan]:
    """Typed-position plans: one cylinder per slot (the user re-stands it)."""
    seed = np.asarray(cfg.STAND_HOME_JOINTS_RIGHT, dtype=float)
    plans = []
    for dy in _slot_offsets(a):
        plans.append(plan_object(g, "cylinder", a.cyl_x, a.cyl_y, 0.0, (a.bin_x, a.bin_y + dy), seed))
        describe(plans[-1])
    return plans


# ---------------------------------------------------------------------------
# Execution (robot)
# ---------------------------------------------------------------------------

def _guard(g: GripperMover, axis, limit: "float | None" = None, tare: bool = True) -> "callable | None":
    """stop_fn: True once the tared force along ``axis`` exceeds ``limit``
    (STAND_CONTACT_N by default). ``tare=False`` reuses the baseline the last
    guard measured, so two thresholds can watch one tare."""
    if tare and not g.tare_wrench():
        return None
    lim = float(cfg.STAND_CONTACT_N if limit is None else limit)
    hit: list[float] = []

    def stop() -> bool:
        f = g.axis_force(axis)
        if f is not None and abs(f) > lim:
            hit.append(f)
            return True
        return False

    stop.hit = hit  # type: ignore[attr-defined]
    return stop


def soft_close(grip) -> int:
    """Close until the fingers touch, then hold — no hard squeeze (pose_tune's
    contact-stop grip). Streams a slow, force-0 close and polls the gripper.
    Contact = EITHER the motor current has stayed above STAND_SOFT_GRIP_CU_STOP
    for STAND_SOFT_GRIP_CU_HOLD_S OR the fingers have stopped advancing for
    STAND_SOFT_GRIP_STALL_S — in both cases only after the fingers have moved
    STAND_SOFT_GRIP_MIN_TRAVEL counts from where they started (the start-up
    inrush and a stale status from the previous open cannot end the grip). Then
    the position the fingers are at (+SQUEEZE) is re-commanded, so only the
    elastic squeeze of that extra travel remains. Every 0.1 s the finger
    position and current are logged, to tune CU_STOP from a real trace.
    Returns (contact position, resting position) (0=open .. 255=closed): the
    contact position says WHERE in the fingers the object sits (a cylinder deep
    against the palm stops the fingers much later than one between the pads),
    the resting one is the hold after the squeeze. Falls back to a plain
    close() on a driver without the non-blocking write."""
    write = getattr(grip, "write_control", None)
    if write is None:
        logger.warning("[stand] gripper driver has no write_control — plain (hard) close")
        grip.close()
        st = grip.read_status()
        pos = st.gPO if st else int(cfg.ROBOTIQ_CLOSE_POS)
        return pos, pos
    st0 = grip.read_status()
    pos0 = st0.gPO if st0 else int(cfg.ROBOTIQ_OPEN_POS)
    speed, force = int(cfg.STAND_SOFT_GRIP_SPEED), int(cfg.STAND_SOFT_GRIP_FORCE)
    write(int(cfg.ROBOTIQ_CLOSE_POS), speed=speed, force=force)
    t0 = time.time()
    cu_since = None                 # when the current first stayed above CU_STOP
    last_pos, pos_since = pos0, t0  # last finger position and when it last changed
    t_log = t0
    while time.time() - t0 < float(cfg.STAND_SOFT_GRIP_TIMEOUT_S):
        time.sleep(0.02)
        st = grip.read_status()
        if st is None:
            continue
        now = time.time()
        if now - t_log >= 0.1:
            logger.info("[stand] soft close trace t={:.2f}s gPO={} gCU={} gOBJ={}", now - t0, st.gPO, st.gCU, st.gOBJ)
            t_log = now
        if st.gPO != last_pos:
            last_pos, pos_since = st.gPO, now
        moved = st.gPO - pos0 >= int(cfg.STAND_SOFT_GRIP_MIN_TRAVEL)
        if st.gOBJ == 3 and st.gPO >= int(cfg.ROBOTIQ_CLOSE_POS) - int(cfg.ROBOTIQ_GRIP_MIN_GAP):
            logger.info("[stand] soft close: fully closed, nothing between the fingers (gPO={})", st.gPO)
            return st.gPO, st.gPO
        if moved and st.gOBJ == 2:
            # the gripper's own force-0 controller stopped on the object (its
            # ~20 N minimum) — the usual contact signal on this gripper (0906
            # trace); commanding further travel would add nothing
            logger.info("[stand] soft close: contact by the gripper's own stop at gPO={} (gCU={} ~{} mA, "
                        "{:.2f}s) — holding there", st.gPO, st.gCU, st.gCU * 10, now - t0)
            return st.gPO, st.gPO
        if st.gCU >= int(cfg.STAND_SOFT_GRIP_CU_STOP):
            cu_since = cu_since or now
        else:
            cu_since = None
        by_current = moved and cu_since is not None and now - cu_since >= float(cfg.STAND_SOFT_GRIP_CU_HOLD_S)
        by_stall = moved and now - pos_since >= float(cfg.STAND_SOFT_GRIP_STALL_S)
        if by_current or by_stall:
            target = min(255, st.gPO + int(cfg.STAND_SOFT_GRIP_SQUEEZE))
            write(target, speed=speed, force=force)
            logger.info("[stand] soft close: contact by {} at gPO={} (gCU={} ~{} mA, {:.2f}s) — holding {}",
                        "current" if by_current else "finger stall", st.gPO, st.gCU, st.gCU * 10,
                        now - t0, target)
            # wait for the squeeze to finish (gOBJ 2 = stopped on the object,
            # 3 = reached the target): judging while the status still says
            # "moving" read as "no object" on the robot (0906 15:45)
            grip.wait_until_done(timeout=2.0)
            final = grip.read_status()
            return st.gPO, (final.gPO if final else target)
    logger.warning("[stand] soft close: no contact within {:.1f} s", cfg.STAND_SOFT_GRIP_TIMEOUT_S)
    final = grip.read_status()
    pos = final.gPO if final else int(cfg.ROBOTIQ_CLOSE_POS)
    return pos, pos


def release(grip, contact_pos: int) -> None:
    """Open to STAND_RELEASE_OPEN_COUNTS past the CONTACT position (where the
    fingers first met the object — not the squeezed hold, which is tighter) so
    the object drops free without the fingers sweeping the bin wall."""
    target = max(int(cfg.ROBOTIQ_OPEN_POS), int(contact_pos) - int(cfg.STAND_RELEASE_OPEN_COUNTS))
    logger.info("[stand] release: fingers -> {} ({} counts wider than the {} contact, about {:.0f} mm)",
                target, int(contact_pos) - target, contact_pos, (int(contact_pos) - target) / 3.0)
    grip.goto(target)


def grasp_with_check(g: GripperMover, plan: ObjectPlan) -> tuple[bool, int]:
    """Soft-close and judge the grasp by WHERE the fingers met the object.

    A cylinder sitting between the pads stops the fingers near the object's
    reference position (STAND_OBJECTS grip_pos); one that slid deep toward the
    palm stops them much later, nothing at all lets them close fully. Outside
    the +/- STAND_GRIP_POS_TOL band the grasp is retried ONCE after a
    STAND_REGRIP_SHIFT_M correction along the entry line: back out when the
    object is deep, further in when the fingers closed on nothing. Returns
    (grasped, contact position)."""
    expected = float(cfg.STAND_OBJECTS[plan.label]["grip_pos"])
    tol = float(cfg.STAND_GRIP_POS_TOL)
    rpy = plan.rpy_pick
    for attempt in (1, 2):
        contact, hold = soft_close(g.gripper)
        st = g.gripper.read_status()
        in_band = abs(contact - expected) <= tol
        # the gripper's own verdict: 2 = stopped BY the object (held); 3 =
        # reached the commanded position, i.e. nothing stopped the fingers
        # (0906 16:24: a graze at 166 then a free close to 191 read as
        # "grasped" under a position-only test). Only without a readable
        # status fall back to "stopped short of fully closed".
        if st is not None and st.gOBJ in (2, 3):
            grasped = st.gOBJ == 2
        else:
            grasped = hold < int(cfg.ROBOTIQ_CLOSE_POS) - int(cfg.ROBOTIQ_GRIP_MIN_GAP)
        logger.info("[{}] soft close -> {} (contact {} / hold {}, gOBJ {}, expected {:.0f}+/-{:.0f} -> {})",
                    plan.label, "GRASPED" if grasped else "no object", contact, hold,
                    st.gOBJ if st else "?", expected, tol, "OK" if in_band else "OFF")
        if grasped and in_band:
            return True, contact
        if attempt == 2:
            return False, contact
        # Correct along the entry line and try once more. "Deep" needs the
        # gripper to have actually STOPPED on something: with gOBJ 3 the fingers
        # met nothing, and a late contact position then means they closed past
        # the object, which has to be approached FURTHER IN, not backed out.
        # (0911 20:56: contact 227 with gOBJ 3 was read as deep and the arm
        # backed out 28 mm, spending the retry in the wrong direction.)
        deep = grasped and contact > expected + tol
        shift = -float(cfg.STAND_REGRIP_SHIFT_M) if deep else float(cfg.STAND_REGRIP_SHIFT_M)
        pos_now = np.asarray(g.fk(g._q_cmd)[0], dtype=float)
        target = pos_now + plan.approach_dir * shift
        logger.warning("[{}] re-grip: object {} (contact {}, gOBJ {}) — shifting {:+.0f}mm along the "
                       "entry and closing again", plan.label,
                       "deep in the fingers" if deep else "not between the pads", contact,
                       st.gOBJ if st else "?", shift * 1000)
        g.gripper.open()
        if g.move_ee_line(target, rpy, speed=float(cfg.STAND_APPROACH_SPEED_M_S)) is None:
            logger.error("[{}] re-grip shift stalled", plan.label)
            return False, contact
        time.sleep(0.2)
    return False, contact


def held_weight(g: GripperMover, n: int = 30) -> "float | None":
    """Vertical force change since the last tare, averaged over ``n`` readings
    at rest, in N (positive = pulling down on the wrist). Taken after the lift,
    with the tare from the empty gripper at the standoff in the SAME
    orientation, this is the object's weight — if it stands out of the ~0.2 N
    sensor noise. Logged only: 0906 loaded-vs-empty came out at 9 g, i.e. the
    cylinder is not measurable here, so nothing decides on it."""
    vals = []
    for _ in range(n):
        f = g.axis_force((0.0, 0.0, -1.0))
        if f is not None:
            vals.append(f)
        time.sleep(0.01)
    return float(np.mean(vals)) if vals else None


def snapshot(g: GripperMover, tag: str) -> None:
    """Log the LIVE torso and right-arm joints plus the model's EE pose of the
    live arm — to see whether anything the model holds fixed (the torso) or the
    arm's branch differs between cycles that should be identical (0906: the
    first-placed cylinder lands ~1 cm toward the robot, every run)."""
    if g._robot is None:
        return
    try:
        torso = np.asarray(g._robot.torso.get_joint_pos(), dtype=float)
        q = g._live_arm_q()
        pos, rpy = g.fk(q)
        model_t = np.asarray(g._torso_q, dtype=float) if hasattr(g, "_torso_q") else None
        logger.info("[snap:{}] torso live {} (model {}); arm q {}; live-arm EE ({:.4f},{:+.4f},{:.4f}) "
                    "rpy ({:+.3f},{:+.3f},{:+.3f})", tag, np.round(torso, 4),
                    "?" if model_t is None else np.round(model_t, 4), np.round(q, 3), *pos, *rpy)
    except Exception as e:  # noqa: BLE001 — diagnostics must never stop the run
        logger.warning("[snap:{}] failed: {}", tag, e)


class _Background(threading.Thread):
    """Run ``fn`` in a thread; the exception (if any) lands in .error."""

    def __init__(self, fn, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._fn, self.error = fn, None

    def run(self) -> None:
        try:
            self._fn()
        except Exception as e:  # noqa: BLE001
            self.error = e


_warm: "_Background | None" = None   # the detector warmup, overlapped with boot


def warmup_start() -> None:
    """Kick off the OBB model warmup in the background, to be joined by the
    first detect_scene. Boot spends ~25 s waiting on hardware (pin_torso, the
    Robotiq activation, three arm-model builds, the camera) with the GPU idle,
    and the warmup is 2.8 s, so it lands for free."""
    global _warm
    from .show_detect import warmup  # noqa: PLC0415
    _warm = _Background(warmup, "detect-warmup")
    _warm.start()


def detect_scene(bot, save: bool = True):
    """Head to the training angle, one fresh frame, both detectors."""
    from .chassis_sequence import _head_rgb, _joints, set_head_pitch  # noqa: PLC0415
    from .show_detect import detect_scene as _detect, save_debug       # noqa: PLC0415
    global _warm
    if _warm is not None:
        # Two threads inside show_detect._model would load the same weights
        # twice, so the warmup has to be done before the first real detect.
        # Boot is ~25 s against its 2.8 s, so this does not actually wait.
        _warm.join()
        if _warm.error is not None:
            logger.warning("[detect] warmup failed ({}) — the first detection pays for it",
                           _warm.error)
        _warm = None
    # tol_deg: skip the move (and its flat 5 s wait) when the head is already
    # aimed. This is the only head command in the run and the angle never
    # changes, so every detection after the first was paying the full 5 s on a
    # head that did not move. _head_rgb below still waits for two fresh frames,
    # so the settle is not lost.
    set_head_pitch(bot, angle=float(cfg.STAND_HEAD_ANGLE_DEG), tol_deg=2.0)
    rgb = _head_rgb(bot)
    if rgb is None:
        logger.error("[detect] no head-camera frame")
        return None
    scene = _detect(rgb, *_joints(bot))
    if save:
        logger.info("[detect] BEV debug PNG -> {}", save_debug(scene))
    return scene


def chassis_drive(bot, dx: float, dy: float, what: str) -> tuple[float, float]:
    """Forward by ``dx`` then left by ``dy`` (m), each clamped to
    STAND_CHASSIS_MAX_M and skipped inside the STAND_CHASSIS_MIN_M deadband, at
    STAND_CHASSIS_SPEED_MS. Open-loop distance legs (move_chassis). Returns the
    displacement actually commanded."""
    from .move_chassis import move_backward, move_forward, strafe_left, strafe_right  # noqa: PLC0415
    dx = float(np.clip(dx, -cfg.STAND_CHASSIS_MAX_M, cfg.STAND_CHASSIS_MAX_M))
    dy = float(np.clip(dy, -cfg.STAND_CHASSIS_MAX_M, cfg.STAND_CHASSIS_MAX_M))
    v = float(cfg.STAND_CHASSIS_SPEED_MS)
    go_x, go_y = abs(dx) >= float(cfg.STAND_CHASSIS_MIN_M), abs(dy) >= float(cfg.STAND_CHASSIS_MIN_M)
    logger.info("[chassis] {}: forward {:+.3f} m, left {:+.3f} m at {:.2f} m/s{}", what, dx, dy, v,
                "" if (go_x or go_y) else " — within the deadband, staying")
    if go_x:
        (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx), speed=v)
    if go_y:
        (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy), speed=v)
    if go_x or go_y:
        logger.info("[chassis] done")
    return (dx if go_x else 0.0, dy if go_y else 0.0)


def chassis_align(bot, from_xy, to_xy, what: str = "bin") -> tuple[float, float]:
    """Drive so a feature now at ``from_xy`` (base_link) ends up at ``to_xy``.
    Returns the displacement commanded (forward, left)."""
    return chassis_drive(bot, from_xy[0] - to_xy[0], from_xy[1] - to_xy[1],
                         f"{what} ({from_xy[0]:.3f},{from_xy[1]:+.3f}) -> ({to_xy[0]:.3f},{to_xy[1]:+.3f})")


def _row_alpha_deg(bin_yaw_deg: "float | None") -> float:
    """Wrist spin at the place point that lines the slot row up with the bin's
    long axis. 0 = the row along base y, the cylinder along base x.

    Applied AFTER the proven in-air turn, as a spin about the tool axis with the
    tool already down — that is what makes the whole range usable
    (STAND_WRIST_SPIN_RANGE_DEG, measured 0911 over the full circle with the
    real IK: final pose and descent column, all four slots). A cylinder is
    symmetric end to end, so alpha and alpha+-180 lay it the same way; take the
    smallest one that is in range."""
    if bin_yaw_deg is None:
        return 0.0
    a = float(bin_yaw_deg) - 90.0
    if abs(a) < float(cfg.STAND_ROW_ALPHA_MIN_DEG):
        return 0.0
    lo, hi = (float(v) for v in cfg.STAND_WRIST_SPIN_RANGE_DEG)
    for cand in sorted((a, a - 180.0, a + 180.0), key=abs):
        if lo <= cand <= hi:
            return float(cand)
    logger.warning("[row] the bin wants the row turned {:+.1f} deg, and neither that nor +-180 is inside "
                   "{:+.0f}..{:+.0f} — leaving the row along base y", a, lo, hi)
    return 0.0


def _row_dir(alpha_deg: float) -> np.ndarray:
    """Unit vector the slot row runs along, base xy. alpha 0 -> base +y."""
    a = np.deg2rad(alpha_deg)
    return np.array([-np.sin(a), np.cos(a)])


def _place_rpy(alpha_deg: float) -> tuple[float, float, float]:
    """STAND_PLACE_RPY turned by ``alpha_deg`` about base z."""
    if alpha_deg == 0.0:
        return tuple(cfg.STAND_PLACE_RPY)
    R = Rotation.from_euler("z", np.deg2rad(alpha_deg)).as_matrix() @ \
        Rotation.from_euler("xyz", cfg.STAND_PLACE_RPY).as_matrix()
    return _rpy(R)


def detect_scene_unclipped(bot, net: "list[float] | None" = None, move: bool = True):
    """detect_scene, but when show_detect DROPS the bin for running off the BEV
    edge, step the chassis toward the clipped side and look again.

    Worth the moves because the dropped reading is the one that used to poison
    the whole cycle: 0911 18:25, the pre-pick move pushed the bin off the +y
    edge, the re-detect read its centre 17 cm short, the carry-time chassis move
    stopped that much shy of the place spot and the cylinder was laid on the
    desk beside the bin. Returns the last scene either way — with the bin still
    None the caller stops, which beats placing into thin air."""
    tries = int(cfg.STAND_BIN_UNCLIP_TRIES) if move else 0
    for k in range(tries + 1):
        scene = detect_scene(bot)
        if scene is None or scene.bin_clip is None:
            return scene
        if k == tries:
            break
        step = float(cfg.STAND_BIN_UNCLIP_STEP_M)
        dx, dy = (step * c for c in scene.bin_clip)
        moved = chassis_drive(bot, dx, dy, f"bin off the BEV edge {scene.bin_clip} — step toward it")
        if net is not None:
            net[0] += moved[0]; net[1] += moved[1]
        if not any(moved):
            break
    logger.error("[detect] bin still off the BEV edge after {} chassis step(s)", tries)
    return scene


def _in_pick_window(xy) -> bool:
    (x0, x1), (y0, y1) = cfg.STAND_CYL_PICK_WINDOW
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def _pick_window_fetch(xy) -> tuple[float, float]:
    """Where to bring a cylinder that lies outside the pick window: the nearest
    spot inside it with STAND_PICK_FETCH_MARGIN_M to spare, not the middle."""
    (x0, x1), (y0, y1) = cfg.STAND_CYL_PICK_WINDOW
    m = float(cfg.STAND_PICK_FETCH_MARGIN_M)
    return (float(np.clip(xy[0], x0 + m, x1 - m)), float(np.clip(xy[1], y0 + m, y1 - m)))


def _slot_y_band(x: float) -> "tuple[float, float] | None":
    """Reachable slot-y band at place x, from the measured STAND_SLOT_Y_BAND_BY_X
    table. None = this x has no measured band (outside the table), so nothing
    places there at all.

    An x BETWEEN two table rows gets the INTERSECTION of the two that bracket
    it: the band edges move by several cm per 25mm of x, so interpolating would
    invent a band wider than anything that was measured. An x that lands ON a
    row gets that row as measured (both brackets are the same row) — widening
    the intersection to the neighbours as well would throw away the measurement
    that is exactly right."""
    rows = [(float(r[0]), float(r[1]), float(r[2])) for r in cfg.STAND_SLOT_Y_BAND_BY_X]
    step = float(cfg.STAND_SLOT_BAND_X_STEP_M)
    below = [r for r in rows if r[0] <= float(x) + 1e-9]
    above = [r for r in rows if r[0] >= float(x) - 1e-9]
    if not below or not above:
        return None                     # off the end of the measured range
    lo_row, hi_row = max(below), min(above)
    if hi_row[0] - lo_row[0] > step + 1e-9:
        return None                     # straddles a gap in the table (0.725/0.750)
    return (max(lo_row[1], hi_row[1]), min(lo_row[2], hi_row[2]))


def _row_fits(band, offsets) -> "tuple[float, float] | None":
    """The band of place-centre y that keeps the WHOLE slot row inside ``band``
    with STAND_PLACE_MARGIN_M to spare, or None if the row is too wide for it."""
    if band is None:
        return None
    m = float(cfg.STAND_PLACE_MARGIN_M)
    lo = band[0] + m - min(offsets)
    hi = band[1] - m - max(offsets)
    return (lo, hi) if lo <= hi else None


def _place_target(bin_xy, offsets) -> tuple[tuple[float, float], "tuple[float, float] | None"]:
    """The place centre for this whole run, and the base displacement that puts
    the bin on it (None = the bin was not seen, so the carry has to start from
    the STAND_CARRY_LEFT_M guess).

    y: the centre is clamped so the whole slot row stays inside the band that x
    actually has (_slot_y_band), minus STAND_PLACE_MARGIN_M, so the bin only
    travels to the nearest edge of that band.

    x: taken from the measurement whenever the row fits there with
    STAND_PLACE_CENTRE_SLACK_M of centre room to spare — the pick window is
    23 cm deep and the arm reaches for x, which is cheaper than a forward leg.
    Otherwise the base drives x after all, to the NEAREST table x that hosts the
    row WITH that slack (0914: a bin at 0.773 has no measured band at all, and
    the nearest fitting x, 0.800, leaves only 20mm of centre room, so it goes to
    0.825 instead). Slack matters because the plan is built at the clamped centre
    while the hand goes to the bin as measured: a roomy centre is what keeps a
    slightly-off carry inside the band the plan verified. Nothing else can fix a
    bin that is simply too near — the row span is fixed and the wrist runs out of
    range mid-flip. The drive is never strafed back (see the return leg), so it
    is paid once per run."""
    if bin_xy is None:
        return tuple(cfg.STAND_BIN_PLACE_XY), None
    x_m = float(bin_xy[0])
    slack = float(cfg.STAND_PLACE_CENTRE_SLACK_M)

    def room(x) -> "tuple[tuple[float, float], float] | None":
        """(allowed centre band, its width) at place x, or None if the row cannot
        fit there at all."""
        b = _row_fits(_slot_y_band(x), offsets)
        return None if b is None else (b, b[1] - b[0])

    here = room(x_m)
    if here is not None and here[1] >= slack:
        y_t = float(np.clip(bin_xy[1], *here[0]))
        return (x_m, y_t), (0.0, bin_xy[1] - y_t)
    # not enough centre room where the bin is — move x to the nearest place that
    # has it, and fall back to the roomiest x at all if nothing clears the slack
    fits = [(float(r[0]), room(float(r[0]))) for r in cfg.STAND_SLOT_Y_BAND_BY_X]
    fits = [(x, r) for x, r in fits if r is not None]
    if not fits:
        logger.warning("[place] {} slots spanning {:.0f} cm do not fit at ANY measured place x with a "
                       "{:.0f} cm margin — using the nominal centre", len(offsets),
                       (max(offsets) - min(offsets)) * 100, float(cfg.STAND_PLACE_MARGIN_M) * 100)
        return tuple(cfg.STAND_BIN_PLACE_XY), (0.0, bin_xy[1] - cfg.STAND_BIN_PLACE_XY[1])
    roomy = [(abs(x - x_m), x) for x, r in fits if r[1] >= slack]
    if roomy:
        x_t = min(roomy)[1]
    else:
        x_t = max(fits, key=lambda e: e[1][1])[0]
        logger.warning("[place] no place x leaves {:.0f} mm of centre room for these slots — taking the "
                       "roomiest ({:.3f}) instead", slack * 1000, x_t)
    band, width = room(x_t)
    y_t = float(np.clip(bin_xy[1], *band))
    logger.warning("[place] bin x {:.3f}: {} for the {:.0f} cm slot row — place x moves to {:.3f} "
                   "(band {:+.3f}..{:+.3f}, centre room {:.0f} mm), so the base drives {:+.0f} mm "
                   "forward on the carry", x_m,
                   "no measured band" if _slot_y_band(x_m) is None else
                   ("band {:+.3f}..{:+.3f} is too narrow".format(*_slot_y_band(x_m)) if here is None
                    else "only {:.0f} mm of centre room".format(here[1] * 1000)),
                   (max(offsets) - min(offsets)) * 100, x_t, *_slot_y_band(x_t), width * 1000,
                   (x_m - x_t) * 1000)
    return (x_t, y_t), (x_m - x_t, bin_xy[1] - y_t)


def go_home(g: GripperMover) -> None:
    pos, rpy = g.current_ee_pose()
    if pos[2] < float(cfg.STAND_TRANSPORT_Z_M) - 0.02:
        logger.info("[stand] low (z {:.3f}) — straight up before homing", pos[2])
        g.move_ee_vertical(float(cfg.STAND_TRANSPORT_Z_M), tuple(rpy))
    logger.info("[stand] -> task home")
    g.move_joints(np.asarray(cfg.STAND_HOME_JOINTS_RIGHT, dtype=float))


def run_object(g: GripperMover, plan: ObjectPlan, pick_only: bool,
               chassis_move: "callable | None" = None,
               place_at: "callable | None" = None) -> "bool | None":
    """One object, start and end at transport height. False = gave up before
    lifting (logged). ``chassis_move`` (no-arg callable) is started in a thread
    right after the lift and runs while the arm carries the upright object over
    the slot and turns it; the descent waits for it. If it raises, the run
    stops with the object still held at transport height -> None.
    Returns with the gripper still at the release width, tool clear above the
    bin: the caller opens it fully, which overlaps the drive back.
    ``place_at`` (no-arg callable -> (x, y, alpha_deg) or None) is asked after
    the chassis has stopped and the turn is done — everything up to the hover
    pose is identical every cycle, and only the LAY-DOWN changes. It measures
    the bin from wherever the base ended up and answers with the point to place
    at plus the wrist spin that lines the row up with the bin, so the chassis is
    out of the final error chain (the arm, not the base, closes the loop on the
    camera). The arm slides and spins there at transport height first. None, or
    a point it cannot reach, leaves the object held -> None. It never falls back
    to the planned point: that fallback is what laid a cylinder on the desk
    beside the bin on 0911. An unreachable SPIN does fall back to the unturned
    lay-down, which places correctly, just not parallel to the bin."""
    rpy = plan.rpy_pick
    g.gripper.open()
    snapshot(g, f"{plan.label} start")
    # Going STRAIGHT to the standoff was tried 0911 and scraped the desk: the
    # offline check that cleared it sampled the straight joint-space
    # interpolation, but Ruckig gives every joint its own time profile, so the
    # real path sags well below that line. The joint leg therefore still lands
    # STAND_PRE_LIFT_M above the standoff and a tracked column takes it down.
    # Still two legs — the joint leg lands STAND_PRE_LIFT_M above the standoff
    # and a TRACKED column takes it down, because going straight to the standoff
    # scraped the desk (0911: the offline check that cleared it sampled the
    # straight joint-space interpolation, but Ruckig gives every joint its own
    # time profile, so the real path sags well below that line).
    #
    # What changed is the HANDOVER. The joint leg used to end at rest and the
    # column used to ramp in from zero, which is the pause at the waypoint;
    # blending the junction in joint space cannot fix it, since 5 of the 7
    # joints reverse direction there and a reversing joint crosses at zero
    # whatever you do. So the joint leg now ARRIVES already moving down at
    # STAND_HANDOVER_SPEED_M_S and the column starts from that speed. The path
    # is unchanged; only the velocity at the junction is.
    z_pre = plan.p_standoff[2] + float(cfg.STAND_PRE_LIFT_M)
    logger.info("[{}] -> {:.0f}mm above the standoff, arriving in motion", plan.label,
                cfg.STAND_PRE_LIFT_M * 1000)
    g._handover_speed = 0.0
    if g.move_ee([plan.p_standoff[0], plan.p_standoff[1], z_pre], rpy,
                 v_out=(0.0, 0.0, -float(cfg.STAND_HANDOVER_SPEED_M_S))) is None:
        return False
    logger.info("[{}] straight down to pinch height z={:.3f}", plan.label, plan.p_standoff[2])
    if g.move_ee_vertical(plan.p_standoff[2], rpy, v_in=g._handover_speed) is None:
        return False
    time.sleep(0.3)
    stop = _guard(g, plan.approach_dir)
    logger.info("[{}] entry {:.0f}mm along ({:+.2f},{:+.2f}) at {:.3f} m/s{}", plan.label,
                float(cfg.STAND_STANDOFF_M) * 1000, *plan.approach_dir[:2],
                float(cfg.STAND_APPROACH_SPEED_M_S),
                f" (bump guard {cfg.STAND_CONTACT_N:.0f}N)" if stop else " (NO force guard)")
    q = g.move_ee_line(plan.p_pinch, rpy, speed=float(cfg.STAND_APPROACH_SPEED_M_S), stop_fn=stop,
                       trace_tag="entry")
    if q is None:
        logger.error("[{}] entry stalled — backing out", plan.label)
        g.move_ee_line(plan.p_standoff, rpy)
        return False
    if stop is not None and stop.hit:  # type: ignore[attr-defined]
        logger.warning("[{}] BUMP {:.1f}N during the entry — not closing, backing out", plan.label, stop.hit[0])
        g.move_ee_line(plan.p_standoff, rpy)
        return False
    grasped, contact = grasp_with_check(g, plan)
    if not grasped:
        g.gripper.open()
        g.move_ee_line(plan.p_standoff, rpy)
        return False

    if pick_only:
        z_up = plan.p_pinch[2] + float(cfg.STAND_LIFT_TEST_M)
        logger.info("[{}] lift test +{:.0f}mm, hold, set back down", plan.label, cfg.STAND_LIFT_TEST_M * 1000)
        g.move_ee_vertical(z_up, rpy)
        time.sleep(float(cfg.BOX_GRIP_HOLD_S))
        w = held_weight(g)
        logger.info("[{}] held weight ~{} (vs the empty tare at the standoff; noise ~0.2 N)", plan.label,
                    "n/a (no wrench tare)" if w is None else f"{w:+.2f} N = {w / 9.81 * 1000:+.0f} g")
        touch = _guard(g, (0.0, 0.0, 1.0))
        g.move_ee_vertical(plan.p_pinch[2], rpy, stop_fn=touch)
        release(g.gripper, contact)
        g.gripper.open()                     # fully, before the fingers back out past the object
        g.move_ee_line(plan.p_standoff, rpy)
        return True

    logger.info("[{}] lift to transport z={:.3f}", plan.label, plan.z_transport)
    if g.move_ee_vertical(plan.z_transport, rpy) is None:
        logger.error("[{}] lift stalled — holding here", plan.label)
        return False
    time.sleep(0.5)
    w = held_weight(g)
    logger.info("[{}] held weight ~{} (vs the empty tare at the standoff; noise ~0.2 N)", plan.label,
                "n/a (no wrench tare)" if w is None else f"{w:+.2f} N = {w / 9.81 * 1000:+.0f} g")
    mover = None
    if chassis_move is not None:
        mover = _Background(chassis_move, "chassis")
        mover.start()
    logger.info("[{}] carry (upright) over the slot ({:.3f},{:+.3f}){}", plan.label, *plan.p_place_hi[:2],
                " while the chassis drives" if mover else "")
    if g.move_ee(plan.p_place_hi, rpy) is None:
        if mover:
            mover.join()
        return False
    logger.info("[{}] in-air turn ({} waypoints) at speed scale {:.2f}: object ends lying along x, tool down",
                plan.label, len(plan.q_rotate), cfg.STAND_TURN_SPEED_SCALE or cfg.SPEED_SCALE_RIGHT)
    # the turn gets its own (slower) joint-speed budget: the cylinder is held
    # along the axis the turn's decelerations act on, and it crept ~1 cm in the
    # pads at 0.5 / less at 0.3 (0906) — rebuild the Ruckig limits around it
    run_scale = cfg.SPEED_SCALE_RIGHT
    if cfg.STAND_TURN_SPEED_SCALE:
        cfg.SPEED_SCALE_RIGHT = float(cfg.STAND_TURN_SPEED_SCALE)
        g._setup_ruckig()
    try:
        g.move_joints_through(plan.q_rotate)
    finally:
        cfg.SPEED_SCALE_RIGHT = run_scale
        g._setup_ruckig()
    rpy = plan.rpy_place
    if mover:
        mover.join()
        if mover.error is not None:
            logger.error("[{}] chassis move failed ({}) — holding the object at transport height, "
                         "NOT placing", plan.label, mover.error)
            return None
    place_xy = np.asarray(plan.p_place_hi[:2], dtype=float)
    if place_at is not None:
        answer = place_at()
        if answer is None:
            logger.error("[{}] no trusted place point — holding the object at transport height, "
                         "NOT placing", plan.label)
            return None
        place_xy = np.asarray(answer[:2], dtype=float)
        alpha = float(answer[2])
        moved = not np.allclose(place_xy, plan.p_place_hi[:2], atol=1e-4)
        if moved or alpha != 0.0:
            rpy = _place_rpy(alpha)
            logger.info("[{}] lay-down from the camera: ({:.3f},{:+.3f}) {:.0f} mm off the planned point, "
                        "wrist spun {:+.1f} deg — sliding there at transport height", plan.label,
                        *place_xy, float(np.linalg.norm(place_xy - plan.p_place_hi[:2])) * 1000, alpha)
            if g.move_ee([place_xy[0], place_xy[1], plan.z_transport], rpy) is None:
                if alpha == 0.0:
                    logger.error("[{}] that point is not reachable from here — holding the object, "
                                 "NOT placing (no fallback to the planned point)", plan.label)
                    return None
                logger.warning("[{}] not reachable with the wrist spun {:+.1f} deg — retrying unturned "
                               "(the cylinder will not lie parallel to the bin)", plan.label, alpha)
                rpy = tuple(plan.rpy_place)
                if g.move_ee([place_xy[0], place_xy[1], plan.z_transport], rpy) is None:
                    logger.error("[{}] not reachable unturned either — holding the object, NOT placing",
                                 plan.label)
                    return None
    time.sleep(0.3)
    touch = _guard(g, (0.0, 0.0, 1.0))                       # tared WITH the object hanging
    # Only the CREEP decides touchdown. The fast leg stops STAND_PLACE_SLOW_FROM_M
    # above the resting height, so a trip there cannot be the floor: 0911 19:22
    # cycle 2 tripped at 5.2N 164 ms in, 169 mm up with nothing under the
    # cylinder, and the old code read that as a touchdown and released in mid
    # air. But the fast leg is not free air either — by its end the cylinder is
    # already below the rim — so it keeps a COARSE guard for a real obstruction
    # (a wall, another cylinder), and that one aborts the place instead of
    # releasing. STAND_DESCENT_ABORT_N = 0 turns it off.
    abort = _guard(g, (0.0, 0.0, 1.0), limit=cfg.STAND_DESCENT_ABORT_N, tare=False) \
        if (touch and float(cfg.STAND_DESCENT_ABORT_N) > 0) else None
    z_aim = plan.z_place - (float(cfg.STAND_PLACE_OVERSHOOT_M) if touch else 0.0)
    z_slow = plan.z_place + float(cfg.STAND_PLACE_SLOW_FROM_M)
    logger.info("[{}] lower to the bin floor (resting EE z {:.3f}): normal to z {:.3f} (abort guard {}), "
                "then creep to {:.3f}{}", plan.label, plan.z_place, z_slow,
                f"{cfg.STAND_DESCENT_ABORT_N:.0f}N" if abort else "off", z_aim,
                f" (touchdown guard {cfg.STAND_CONTACT_N:.0f}N)" if touch else " (NO force guard)")
    # two legs: the second is a constant CREEP so the arm is never more than a
    # millimetre or two behind its command when the floor arrives. (0906: a
    # single leg aimed 3 cm below the floor hit at 20 N; a second
    # move_ee_vertical leg re-accelerated to cruise over its first 2 cm and
    # the arm, 17 mm behind the command at contact, kept pressing after the
    # guard froze it.)
    g.move_ee_vertical(z_slow, rpy, stop_fn=abort)
    if abort is not None and abort.hit:  # type: ignore[attr-defined]
        logger.error("[{}] {:.1f}N during the FAST descent at z={:.4f} — something is under the tool "
                     "(bin wall? another cylinder?). Holding the object at that height, NOT placing",
                     plan.label, abort.hit[0], g.fk(g._q_cmd)[0][2])
        return None
    g.move_ee_line([place_xy[0], place_xy[1], z_aim], rpy,
                   speed=float(cfg.STAND_PLACE_CREEP_M_S), stop_fn=touch, trace_tag="place")
    if touch is not None and touch.hit:  # type: ignore[attr-defined]
        logger.info("[{}] touchdown {:.1f}N at EE z={:.4f} ({:+.0f}mm vs resting)", plan.label,
                    touch.hit[0], g.fk(g._q_cmd)[0][2], (g.fk(g._q_cmd)[0][2] - plan.z_place) * 1000)
    time.sleep(0.3)
    snapshot(g, f"{plan.label} touchdown")
    release(g.gripper, contact)
    time.sleep(0.3)
    logger.info("[{}] straight up, empty", plan.label)
    g.move_ee_vertical(plan.z_transport, rpy)
    # The fingers are clear of the bin walls here, but the FULL open is left to
    # the caller: it takes ~0.8 s that the base can spend driving back.
    return True


def run_cylinders(bot, g: GripperMover, args: "Args | None" = None) -> bool:
    """The whole cylinder job as ONE STEP of a bigger session (chassis_sequence
    task 5), instead of `python -m ik_demo.stand_place`.

    The caller owns the robot: it has already connected, pinned the subscribers,
    safe-homed both arms and initialised the Robotiq, and it does its own go /
    no-go gate and run log. So this only borrows the right arm, sets its speed
    for the job — the mover's motion limits were built at the session's
    SPEED_SCALE_RIGHT, hence the rebuild — and runs the cycles."""
    a = args or Args()
    if g.gripper is None:
        logger.error("[stand] the cylinder job needs the Robotiq — none available")
        return False
    was = float(cfg.SPEED_SCALE_RIGHT)
    cfg.SPEED_SCALE_RIGHT = float(a.speed)
    g._setup_ruckig()
    logger.warning("[stand] right-arm speed scale {:.2f} for the cylinder job (STAND_SPEED_SCALE {:.2f}; "
                   "the session's SPEED_SCALE_RIGHT {:.2f} is restored afterwards)",
                   a.speed, cfg.STAND_SPEED_SCALE, was)
    try:
        go_home(g)
        return _run_detected(bot, g, a)
    finally:
        cfg.SPEED_SCALE_RIGHT = was
        g._setup_ruckig()


def _main(a: Args) -> None:
    if a.dry and not a.detect:
        plans = build_plans(GripperMover(robot=None), a)
        logger.info("dry run -> {}", ", ".join(f"{p.label}: {'OK' if p.ok else 'FAIL'}" for p in plans))
        return

    from dexcontrol.robot import Robot

    logger.warning("=" * 60)
    if a.detect and a.dry:
        logger.warning("DETECT ONLY (head moves, arms and chassis do not): cylinders + bin -> plan, no motion")
    else:
        logger.warning("MOVES THE REAL RIGHT ARM + ROBOTIQ GRIPPER{}:",
                       " + THE CHASSIS" if (a.chassis and a.detect) else "")
    if a.detect:
        logger.warning("  cylinders + bin from the head camera (angle {:.0f}){}", cfg.STAND_HEAD_ANGLE_DEG,
                       "; chassis drives the bin to ({:.2f},{:+.2f}) at {:.2f} m/s".format(
                           *cfg.STAND_BIN_PLACE_XY, cfg.STAND_CHASSIS_SPEED_MS) if a.chassis else "")
    else:
        logger.warning("  cylinder at ({:.2f},{:+.2f}); bin at ({:.2f},{:+.2f}); {} cycle(s)",
                       a.cyl_x, a.cyl_y, a.bin_x, a.bin_y, len(_slot_offsets(a)))
    logger.warning("  {}", "grip, lift 10 cm, set back down, RELEASE (--pick-only)" if a.pick_only
                   else "grip -> lift -> turn -> lay into the bin -> release")
    logger.warning("Clear the right arm's workspace. Keep the e-stop within reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if cfg.RUN_LOG_DIR is not None:
        Path(cfg.RUN_LOG_DIR).mkdir(parents=True, exist_ok=True)
        logger.add(f"{cfg.RUN_LOG_DIR}/stand_{stamp}.log", level="INFO")
        logger.info("run log -> {}/stand_{}.log", cfg.RUN_LOG_DIR, stamp)
    cfg.SPEED_SCALE_RIGHT = float(a.speed)      # read once, when the mover builds its motion limits
    logger.warning("right-arm speed scale {:.2f} for this run", a.speed)
    robot_configs = None
    if a.detect:
        from dexcontrol.core.config import get_robot_config  # noqa: PLC0415
        robot_configs = get_robot_config()
        robot_configs.enable_sensor("head_camera")
        robot_configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=robot_configs) as bot:
        if a.detect and not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        # The steering angles and the right wrench are each read in a burst
        # (pre-steer polls, the descent guard) with a whole cycle of silence
        # in between, so dexcontrol's 5 s idle policy pauses them every time
        # and the next burst pays the resume. Measured 0911: 4.8 s and 2.5 s
        # over a 202 s run. Pinning them ON is free — both are small state
        # topics. NOT done globally: that would also pin right_rgb and depth,
        # which this run never reads.
        bot.chassis.set_subscription_policy("always_on")
        if bot.right_arm.wrench_sensor is not None:
            bot.right_arm.wrench_sensor.set_subscription_policy("always_on")
        if a.detect:
            warmup_start()   # runs while the arm/gripper/torso boot below
        g = GripperMover(bot)
        if a.detect and a.dry:
            _detect_dry(bot, g, a)
            return
        release = g.software_estop_active()
        if release and input("Software E-Stop is active. Release it? [y/N]: ").strip().lower() != "y":
            return
        if not g.ensure_ready(release_estop=release):
            logger.error("right arm not ready — aborting")
            return
        plans = [] if a.detect else build_plans(g, a)
        if not all(p.ok for p in plans):
            logger.error("plan failed — arm not moved")
            return
        if not g.initialize():
            logger.error("gripper not available — aborting (arm not moved)")
            return
        # LEFT arm to the chassis-sequence home first (both_arms_home does the
        # same at the start of every chassis run): it otherwise sits in the
        # head-camera view and in the right arm's swing.
        safe_home(ArmMover(robot=bot, side="left", ee_frame=cfg.EE_FRAME))
        go_home(g)
        if a.detect:
            _run_detected(bot, g, a)
        for plan in plans:
            ok = run_object(g, plan, a.pick_only)
            g.gripper.open()
            logger.info("[{}] -> {} (slot y {:+.3f})", plan.label, "done" if ok else "SKIPPED",
                        plan.p_place_hi[1])
        if not a.keep:
            go_home(g)


def _slot_offsets(a: Args) -> list[float]:
    offsets = [float(v) / 100.0 for v in a.slots.split(",") if v.strip()]
    return offsets or [0.0]


def _plan_detected(g: GripperMover, scene, dy: float, place_c) -> "ObjectPlan | None":
    """Plan the NEAREST detected cylinder into the slot ``dy`` along base y from
    the place centre. ``place_c`` is the centre the base will bring the bin to;
    None (--no-chassis) means place at the bin wherever it was detected. The row
    is turned to the bin later, by the wrist, at the place point."""
    if not scene.cylinders:
        logger.warning("[detect] no cylinder in view")
        return None
    if place_c is None and scene.bin is None:
        logger.warning("[detect] no bin in view — needed to place with --no-chassis")
        return None
    bx, by = place_c if place_c is not None else scene.bin[:2]
    x, y, conf = scene.cylinders[0]
    plan = plan_object(g, "cylinder", x, y, 0.0, (bx, by + dy),
                       np.asarray(cfg.STAND_HOME_JOINTS_RIGHT, dtype=float))
    describe(plan)
    return plan


def _detect_dry(bot, g: GripperMover, a: Args) -> None:
    scene = detect_scene(bot)
    if scene is None:
        return
    offsets = _slot_offsets(a)
    target, need = _place_target(scene.bin[:2] if scene.bin else None, offsets) if a.chassis else (None, None)
    if a.chassis and a.yaw_align:
        logger.info("[row] bin long axis {} -> the wrist would spin {:+.1f} deg at the lay-down",
                    f"{scene.bin[2]:.0f} deg" if scene.bin else "not in view from here (normal)",
                    _row_alpha_deg(scene.bin[2]) if scene.bin else 0.0)
    plan = _plan_detected(g, scene, offsets[0], target)
    if a.chassis:
        # the sequence the real run would follow, in the order it decides things
        if need is None:
            logger.info("[chassis] bin not in view from here, which is normal — cycle 1 runs on the "
                        "nominal centre ({:.3f},{:+.3f}) and the {:.2f} m carry guess, then learns the "
                        "real ones when it comes over the bin", target[0], target[1],
                        cfg.STAND_CARRY_LEFT_M)
        else:
            logger.info("[chassis] bin IS in view from here: place centre would be ({:.3f},{:+.3f}) for "
                        "slots {:+.3f}..{:+.3f}, carry {:+.3f} m left", target[0], target[1],
                        target[1] + min(offsets), target[1] + max(offsets), need[1])
        total = need or (0.0, cfg.STAND_CARRY_LEFT_M)
        pre = (0.0, 0.0)
        if scene.cylinders:
            cyl = np.asarray(scene.cylinders[0][:2])
            if not _in_pick_window(cyl):
                fetch = _pick_window_fetch(cyl)
                pre = (0.0, cyl[1] - fetch[1])      # y only
            logger.info("[chassis] 1. cylinder ({:.3f},{:+.3f}) {} -> pre-pick move forward {:+.3f} m, "
                        "left {:+.3f} m", cyl[0], cyl[1],
                        "in the pick window" if _in_pick_window(cyl) else "outside the pick window", *pre)
        logger.info("[chassis] 2. loaded carry leg: forward {:+.3f} m, left {:+.3f} m (the total minus "
                    "the pre-pick)", total[0] - pre[0], total[1] - pre[1])
        logger.info("[chassis] 3. bin measured again from there; the correction folds into the total. "
                    "Per cycle the base drives {:.2f} m out and the same back",
                    abs(total[0] - pre[0]) + abs(total[1] - pre[1]))
    logger.info("detect dry run -> {}", "no plan" if plan is None else ("OK" if plan.ok else "FAIL"))


def _run_detected(bot, g: GripperMover, a: Args) -> bool:
    """One cycle per slot: detect the CYLINDERS (the bin is not looked at yet)
    -> the base fetches the nearest one into the pick window -> pick -> the base
    carries toward the bin by the learned displacement while the arm lifts,
    carries and turns -> only NOW detect the bin, correct the base onto the
    place spot and keep that correction for the next cycle -> place -> home ->
    the base returns to where the cycle started.

    ``carry`` is the total LEFT displacement FROM THE CYCLE START that lands the
    bin on the place centre, not the length of the carry leg: the pre-pick move
    differs per cylinder (each is strafed to the pick window from a different
    place on the desk), so the leg is this total minus what the pre-pick already
    moved. The total is the thing that stays constant while the bin sits still,
    which is what makes it learnable.

    The base normally only strafes: x is left to the arm, since the pick window
    is 23 cm deep and the place solves over 20 cm of x. Two things do drive x,
    both on demand and neither strafed back — the STAND_PICK_RETRY nudge when a
    plan fails outright (below), and the place-x fit when the bin is measured
    somewhere the slot row cannot fit in y at all (_place_target).

    True only if every slot got its cylinder."""
    net = [0.0, 0.0]                                 # base displacement so far THIS cycle (forward, left)
    offsets = _slot_offsets(a)
    carry = [float(cfg.STAND_CARRY_LEFT_M)]          # learned LEFT displacement
    # x the base still owes to put the bin where the slot row fits (_place_target).
    # Usually 0 — only a bin measured too near/far to host the row asks for it.
    pend_x = [0.0]
    target = [float(v) for v in cfg.STAND_BIN_PLACE_XY]   # place centre; replaced at the first sighting
    seen = [False]                                   # has the bin been measured yet this run
    # Slot index, NOT a for-loop: a cycle that gave up before lifting (entry
    # stalled, bump, or the grasp failed even after grasp_with_check's backoff
    # re-grip) does not consume its slot. The end of the body already opens the
    # gripper, strafes back and homes the arm, and the top re-detects, so the
    # retry is exactly a fresh attempt at the same slot. Unlimited by request:
    # the cylinder may need standing back up by hand, and the operator watches
    # the attempt counter and takes over rather than the code giving up.
    k, tries = 0, 0
    while k < len(offsets):
        dy = offsets[k]
        scene = detect_scene(bot)
        if scene is None:
            break
        if a.chassis and scene.cylinders:
            cyl = np.asarray(scene.cylinders[0][:2])
            fetch = _pick_window_fetch(cyl)
            if abs(cyl[1] - fetch[1]) >= float(cfg.STAND_CHASSIS_MIN_M):
                logger.info("[cycle {}] cylinder ({:.3f},{:+.3f}) outside the pick window {} in y -> "
                            "strafe it to {:+.3f} (y only; x is the arm's job unless a plan fails "
                            "or the slot row needs a different place x)",
                            k + 1, cyl[0], cyl[1], cfg.STAND_CYL_PICK_WINDOW, fetch[1])
                moved = chassis_drive(bot, 0.0, float(cyl[1] - fetch[1]), "cylinder y -> pick window")
                net[0] += moved[0]; net[1] += moved[1]
                if any(moved):
                    scene = detect_scene(bot)
                    if scene is None:
                        break
        plan = _plan_detected(g, scene, dy, target if a.chassis else None)
        # Last resort, after the row-angle fallback above: a plan can fail with
        # the cylinder well INSIDE the pick window. Offline sweep 0911 at the
        # torso the robot actually holds (see STAND_PICK_RETRY_STEP_M) found
        # 1-2 cm islands where lift_top stalls a few mm short with the joints in
        # limits and nothing colliding, solid ground on every side. Nudge the
        # cylinder deeper into the window (+x, the direction that measured
        # solid) and look again rather than ending the run.
        for t in range(int(cfg.STAND_PICK_RETRY_TRIES)):
            if plan is not None and plan.ok:
                break
            if not a.chassis or not scene.cylinders:
                break
            cyl = np.asarray(scene.cylinders[0][:2])
            far_x = cfg.STAND_CYL_PICK_WINDOW[0][1] - float(cfg.STAND_PICK_FETCH_MARGIN_M)
            to_x = min(cyl[0] + float(cfg.STAND_PICK_RETRY_STEP_M), far_x)
            if to_x - cyl[0] < float(cfg.STAND_CHASSIS_MIN_M):
                logger.error("[cycle {}] no feasible plan and the cylinder is already at the far edge "
                             "of the pick window ({:.3f}) — nowhere left to nudge it", k + 1, far_x)
                break
            logger.warning("[cycle {}] no feasible plan for the cylinder at ({:.3f},{:+.3f}) — nudging it "
                           "{:.0f}mm deeper into the pick window and re-detecting (retry {}/{})",
                           k + 1, cyl[0], cyl[1], (to_x - cyl[0]) * 1000, t + 1,
                           int(cfg.STAND_PICK_RETRY_TRIES))
            moved = chassis_align(bot, tuple(cyl), (to_x, float(cyl[1])), "pick retry")
            net[0] += moved[0]; net[1] += moved[1]
            if not any(moved):
                logger.error("[cycle {}] the nudge fell inside the chassis deadband — stopping", k + 1)
                break
            scene = detect_scene(bot)
            if scene is None:
                break
            plan = _plan_detected(g, scene, dy, target if a.chassis else None)
        if plan is None or not plan.ok:
            logger.error("[cycle {}] no feasible plan — stopping", k + 1)
            break

        def carry_base(n=k + 1):
            """The blind leg, run while the arm carries and turns. Left is the
            learned carry; forward is normally 0 — the bin's x is reached for,
            not driven to — and non-zero only when _place_target had to move the
            place x to fit the slot row (paid once, never strafed back)."""
            leg = carry[0] - net[1]
            leg_x = pend_x[0]
            logger.info("[cycle {}] loaded carry: total left from the cycle start is {:+.3f}, {:+.3f} "
                        "already strafed -> leg {:+.3f} m left{}", n, carry[0], net[1], leg,
                        ", {:+.3f} m forward (place-x fit)".format(leg_x) if leg_x else "")
            d = chassis_drive(bot, leg_x, leg, "loaded carry toward the bin (learned, bin not seen yet)")
            net[0] += d[0]; net[1] += d[1]
            pend_x[0] -= d[0]

        def place_at(n=k + 1, slot_dy=dy):
            """The one look at the bin this cycle, taken from wherever the blind
            carry stopped. The HAND goes to the bin as measured, so the chassis
            is out of the final error chain (it only has to get the bin roughly
            under the arm, and its error shows up as a hand offset the camera
            sees). Everything the next cycle needs is learned right here — the
            bin is never in view from where a cycle starts:

              * the carry, so the next blind leg lands closer,
              * the place centre (y clamped into the band that x actually has,
                x as measured unless the row needs a different one), so the next
                plan is made where the bin really is,
              * the x the next carry owes to get there (normally 0),
            The row angle is NOT among them: it is measured and used here, in
            this same cycle, every time. The base loses heading when it strafes,
            so a yaw read one cycle ago would be wrong by the next."""
            sc = detect_scene_unclipped(bot, net)
            if sc is None or sc.bin is None:
                logger.error("[cycle {}] bin not measurable from here — not placing", n)
                return None
            off = np.asarray(sc.bin[:2]) - np.asarray(target)
            if np.hypot(*off) > float(cfg.STAND_CARRY_CORRECT_MAX_M):
                logger.error("[cycle {}] bin ({:.3f},{:+.3f}) is {:.3f} m off the planned centre — beyond "
                             "STAND_CARRY_CORRECT_MAX_M ({:.2f}), so this is a bad detection, not an "
                             "offset: not placing", n, sc.bin[0], sc.bin[1], float(np.hypot(*off)),
                             cfg.STAND_CARRY_CORRECT_MAX_M)
                return None
            alpha = _row_alpha_deg(sc.bin[2]) if a.yaw_align else 0.0
            logger.info("[cycle {}] bin measured at ({:.3f},{:+.3f}) yaw {:.0f}, {:.0f} mm off the planned "
                        "centre -> the hand goes there, the base stays; row spun {:+.1f} deg",
                        n, sc.bin[0], sc.bin[1], sc.bin[2], float(np.hypot(*off)) * 1000, alpha)
            hand = np.asarray(sc.bin[:2], dtype=float) + slot_dy * _row_dir(alpha)
            # ...and everything after this point is for the NEXT cycle
            # The bin as it would read from where this cycle started. Only y is
            # added: the carry strafes, and any x nudge happened BEFORE the plan
            # was made, so the measured x is already in the planning frame.
            p_bin = (sc.bin[0], sc.bin[1] + net[1])
            t, need_next = _place_target(p_bin, offsets)
            was_t, was_c = tuple(target), carry[0]
            target[0], target[1] = t
            carry[0] = p_bin[1] - target[1]
            # x is not a "total from the cycle start" like the carry: it is never
            # strafed back, so it is a one-shot debt the next carry pays off.
            pend_x[0] = need_next[0] if need_next is not None else 0.0
            logger.info("[cycle {}] learned for the next cycle: place centre ({:.3f},{:+.3f}) -> "
                        "({:.3f},{:+.3f}) [slots {:+.3f}..{:+.3f}], carry left {:+.3f} -> {:+.3f}",
                        n, *was_t, *target, target[1] + min(offsets), target[1] + max(offsets),
                        was_c, carry[0])
            if not seen[0]:
                seen[0] = True
                logger.info("[cycle {}] (first sighting of the bin this run — cycle 1 ran on the "
                            "STAND_CARRY_LEFT_M guess)", n)
            return (float(hand[0]), float(hand[1]), float(alpha))

        ok = run_object(g, plan, a.pick_only,
                        chassis_move=carry_base if a.chassis else None,
                        place_at=place_at if a.chassis else None)
        if ok is None:
            logger.error("[cycle {}] stopped with the cylinder held — take over", k + 1)
            return False
        logger.info("[cycle {}] -> {} (slot y {:+.3f})", k + 1,
                    "done" if ok else "no lift — retrying this slot", plan.p_place_hi[1])
        # The return leg runs WHILE the arm turns back and homes, the mirror of
        # the outbound carry — the arm is empty here, and go_home's first move
        # is straight up, clear of the bin. Both must finish before the next
        # detection: the base has to be settled and the arm out of the head view.
        back = None
        if a.chassis and abs(net[1]) > 1e-6:
            # Strafe back to where this cycle started, so the learned carry is
            # measured from the same pose every cycle. y ONLY: whatever x the
            # on-demand nudges and the place-x fit gave the base, it keeps.
            # Undoing x would walk
            # straight back into the pose whose plan had just failed, and
            # nothing needs it — the cylinders are re-detected every cycle and
            # the carry is a strafe distance.
            dy_back = -net[1]
            back = _Background(lambda: chassis_drive(bot, 0.0, dy_back,
                                                     "strafe back to the cycle start (y only)"),
                               "chassis-back")
            back.start()
        g.gripper.open()    # fully — alongside the drive back, not before it
        go_home(g)          # clear the head view before the next detection
        if back is not None:
            back.join()
            if back.error is not None:
                logger.error("[cycle {}] the return strafe failed ({}) — stopping; the base is part way "
                             "back, so the next cycle's carry would be measured from the wrong pose",
                             k + 1, back.error)
                return False
            net[1] = 0.0
        if ok:
            k += 1
            tries = 0
        else:
            tries += 1
            logger.warning("[cycle {}] gave up before lifting — home and re-detected, trying this slot "
                           "again (attempt {}). Nothing was placed and the gripper is empty; stand the "
                           "cylinder back up if it fell, or e-stop to take over.", k + 1, tries + 1)
    logger.info("[carry] final learned total: {:+.3f} m left (STAND_CARRY_LEFT_M is {:.2f}); place centre "
                "({:.3f},{:+.3f}); the base also ended {:+.3f} m forward of where the run started, from "
                "the on-demand x nudges and any place-x fit (never strafed back)", carry[0], cfg.STAND_CARRY_LEFT_M,
                target[0], target[1], net[0])
    return k >= len(offsets)


if __name__ == "__main__":
    _main(tyro.cli(Args))
