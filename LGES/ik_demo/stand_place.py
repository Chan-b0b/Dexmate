"""Right-arm stand pick + lay-down place: standing cylinders -> the black bin.

For each cylinder (one cycle per --slots entry):

    home -> just above the standoff -> short straight descent to pinch height
         -> straight-line entry from behind (+x), fingers open left/right,
            wrench guard stops an early bump
         -> soft close (position-checked, one corrective re-grip)
         -> straight up to transport height -> carry upright over the slot
            [--chassis: the base drives the bin to STAND_BIN_PLACE_XY meanwhile]
         -> turn in the air (tool ends pointing down, cylinder lies along x)
         -> straight down until the cylinder rests on the bin floor (touchdown
            guard) -> open 4 mm -> straight up -> open -> home
            [--chassis: the base drives back by everything it moved this cycle]

Positions come from --detect (show_detect: head camera, both OBB models,
nearest cylinder first) or from the typed --cyl-x/--cyl-y and --bin-x/--bin-y.

Run from LGES/:
    python -m ik_demo.stand_place --dry                       # plan only, no robot (typed positions)
    python -m ik_demo.stand_place --pick-only                 # grip, lift 10 cm, set back down, release
    python -m ik_demo.stand_place                             # four cylinders, 5 cm apart (re-stand between cycles)
    python -m ik_demo.stand_place --slots 0                   # one cylinder into the bin centre
    python -m ik_demo.stand_place --measure-fingers           # fingertips touch the desk: finger length
    python -m ik_demo.stand_place --detect --dry              # camera: cylinders + bin -> plan only (calibrate)
    python -m ik_demo.stand_place --detect                    # detected cylinder -> detected bin (arm reach only)
    python -m ik_demo.stand_place --detect --chassis          # chassis brings the bin (and, if needed, the
                                                              # cylinder) into reach, places, drives back
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
    cyl_x: float = 0.933          # standing cylinder centre, base_link (m) — ignored with --detect
    cyl_y: float = -0.28
    bin_x: float = 0.95           # black bin centre, base_link (m) — ignored with --detect
    bin_y: float = -0.05
    slots: str = "-7.5,-2.5,2.5,7.5"   # one cycle per entry: slot y offsets from the bin centre, cm, left is +
                                       # (default: four cylinders 5 cm apart, user's choice 0906); "0" = one, centred
    dry: bool = False             # plan + report only, no robot
    pick_only: bool = False       # grip, lift STAND_LIFT_TEST_M, set back down, release (no bin)
    keep: bool = False            # skip the final home
    detect: bool = False          # find the cylinders + bin with the head camera (show_detect), every cycle
    chassis: bool = False         # with --detect: drive the base so the bin sits at STAND_BIN_PLACE_XY
                                  # (slowly, while the arm carries + turns), then place there
    measure_fingers: bool = False # no pick: fingers closed + tool down, touch the desk at (--probe-x, --probe-y),
                                  # report flange-to-fingertip length = contact z - STAND_DESK_Z_M
    probe_x: float = 0.85         # desk spot for --measure-fingers (must be clear of objects and the bin)
    probe_y: float = -0.15
    speed: float = 0.1            # right-arm speed scale for this run (cfg.SPEED_SCALE_RIGHT is 0.2)


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
    solves; pass the config the arm will actually be in (the task home)."""
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

def _guard(g: GripperMover, axis) -> "callable | None":
    """stop_fn: True once the tared force along ``axis`` exceeds STAND_CONTACT_N."""
    if not g.tare_wrench():
        return None
    hit: list[float] = []

    def stop() -> bool:
        f = g.axis_force(axis)
        if f is not None and abs(f) > float(cfg.STAND_CONTACT_N):
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
        # correct along the entry line and try once more
        deep = contact > expected + tol and contact < int(cfg.ROBOTIQ_CLOSE_POS) - int(cfg.ROBOTIQ_GRIP_MIN_GAP)
        shift = -float(cfg.STAND_REGRIP_SHIFT_M) if deep else float(cfg.STAND_REGRIP_SHIFT_M)
        pos_now = np.asarray(g.fk(g._q_cmd)[0], dtype=float)
        target = pos_now + plan.approach_dir * shift
        logger.warning("[{}] re-grip: object {} — shifting {:+.0f}mm along the entry and closing again",
                       plan.label, "deep in the fingers" if deep else "not between the pads", shift * 1000)
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


def measure_fingers(g: GripperMover, x: float, y: float) -> None:
    """Touch the desk with the closed fingertips, tool down, and report the
    flange-to-fingertip length. That constant (STAND_FINGER_LENGTH_M; 0.14 was the
    guess) sets both the pinch depth and the place height; 0906 the touchdown
    heights implied ~0.11, which would also mean the fingertips reach the bin
    floor before the cylinder does."""
    rpy = tuple(cfg.STAND_PLACE_RPY)
    z_top = float(cfg.STAND_TRANSPORT_Z_M)
    z_aim = float(cfg.STAND_DESK_Z_M) + 0.06          # well below any plausible fingertip contact
    logger.info("[measure] -> ({:.3f},{:+.3f}) at z {:.3f}, tool down, fingers CLOSED", x, y, z_top)
    if g.move_ee([x, y, z_top], rpy) is None:
        return
    g.gripper.close()
    time.sleep(0.5)
    touch = _guard(g, (0.0, 0.0, 1.0))
    if touch is None:
        logger.error("[measure] no wrench guard — not descending")
        return
    logger.info("[measure] descending to the desk (guard {:.0f}N)", cfg.STAND_CONTACT_N)
    g.move_ee_vertical(z_aim, rpy, stop_fn=touch)
    if touch.hit:  # type: ignore[attr-defined]
        z = float(g.fk(g._q_cmd)[0][2])
        logger.info("[measure] fingertips touched at EE z={:.4f} ({:.1f}N). If this was the DESK ({:.3f}): "
                    "flange->fingertip = {:.4f} m (config {:.3f}). If it was another surface (bin floor): "
                    "surface z = {:.4f} using the config finger length", z, touch.hit[0], cfg.STAND_DESK_Z_M,
                    z - float(cfg.STAND_DESK_Z_M), cfg.STAND_FINGER_LENGTH_M, z - float(cfg.STAND_FINGER_LENGTH_M))
    else:
        logger.error("[measure] reached z {:.3f} without contact — desk lower than assumed or fingers shorter "
                     "than 6 cm?!", z_aim)
    g.move_ee_vertical(z_top, rpy)
    g.gripper.open()


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


def detect_scene(bot, save: bool = True):
    """Head to the training angle, one fresh frame, both detectors."""
    from .chassis_sequence import _head_rgb, _joints, set_head_pitch  # noqa: PLC0415
    from .show_detect import detect_scene as _detect, save_debug       # noqa: PLC0415
    set_head_pitch(bot, angle=float(cfg.STAND_HEAD_ANGLE_DEG))
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


def _in_pick_window(xy) -> bool:
    (x0, x1), (y0, y1) = cfg.STAND_CYL_PICK_WINDOW
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def _pick_window_mid() -> tuple[float, float]:
    (x0, x1), (y0, y1) = cfg.STAND_CYL_PICK_WINDOW
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def go_home(g: GripperMover) -> None:
    pos, rpy = g.current_ee_pose()
    if pos[2] < float(cfg.STAND_TRANSPORT_Z_M) - 0.02:
        logger.info("[stand] low (z {:.3f}) — straight up before homing", pos[2])
        g.move_ee_vertical(float(cfg.STAND_TRANSPORT_Z_M), tuple(rpy))
    logger.info("[stand] -> task home")
    g.move_joints(np.asarray(cfg.STAND_HOME_JOINTS_RIGHT, dtype=float))


def run_object(g: GripperMover, plan: ObjectPlan, pick_only: bool,
               chassis_move: "callable | None" = None,
               refine_place: "callable | None" = None) -> "bool | None":
    """One object, start and end at transport height. False = gave up before
    lifting (logged). ``chassis_move`` (no-arg callable) is started in a thread
    right after the lift and runs while the arm carries the upright object over
    the slot and turns it; the descent waits for it. If it raises, the run
    stops with the object still held at transport height -> None.
    ``refine_place`` (no-arg callable -> (dx, dy) or None) is asked after the
    chassis has stopped and the turn is done; a non-None answer shifts the
    place point by that much (a re-detection of the bin from where it ended
    up) before the descent."""
    rpy = plan.rpy_pick
    g.gripper.open()
    snapshot(g, f"{plan.label} start")
    z_pre = plan.p_standoff[2] + float(cfg.STAND_PRE_LIFT_M)
    logger.info("[{}] -> {:.0f}mm above the standoff (joint-space)", plan.label, cfg.STAND_PRE_LIFT_M * 1000)
    if g.move_ee([plan.p_standoff[0], plan.p_standoff[1], z_pre], rpy) is None:
        return False
    logger.info("[{}] straight down to pinch height z={:.3f}", plan.label, plan.p_standoff[2])
    if g.move_ee_vertical(plan.p_standoff[2], rpy) is None:
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
    place_xy = plan.p_place_hi[:2].copy()
    if refine_place is not None:
        d = refine_place()
        if d is not None:
            place_xy = place_xy + np.asarray(d, dtype=float)
            logger.info("[{}] place point refined by ({:+.3f},{:+.3f}) -> ({:.3f},{:+.3f})", plan.label,
                        d[0], d[1], *place_xy)
            if g.move_ee([place_xy[0], place_xy[1], plan.z_transport], rpy) is None:
                logger.warning("[{}] refined point unreachable — using the nominal one", plan.label)
                place_xy = plan.p_place_hi[:2].copy()
                g.move_ee(plan.p_place_hi, rpy)
    time.sleep(0.3)
    touch = _guard(g, (0.0, 0.0, 1.0))                       # tared WITH the object hanging
    z_aim = plan.z_place - (float(cfg.STAND_PLACE_OVERSHOOT_M) if touch else 0.0)
    z_slow = plan.z_place + float(cfg.STAND_PLACE_SLOW_FROM_M)
    logger.info("[{}] lower to the bin floor (resting EE z {:.3f}): normal to z {:.3f}, then slow to {:.3f}{}",
                plan.label, plan.z_place, z_slow, z_aim,
                f" (touchdown guard {cfg.STAND_CONTACT_N:.0f}N)" if touch else " (NO force guard)")
    # two legs: the second is a constant CREEP so the arm is never more than a
    # millimetre or two behind its command when the floor arrives. (0906: a
    # single leg aimed 3 cm below the floor hit at 20 N; a second
    # move_ee_vertical leg re-accelerated to cruise over its first 2 cm and
    # the arm, 17 mm behind the command at contact, kept pressing after the
    # guard froze it.)
    g.move_ee_vertical(z_slow, rpy, stop_fn=touch)
    if not touch or not touch.hit:  # type: ignore[attr-defined]
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
    g.gripper.open()                         # fully, now that the fingers are clear of the bin walls
    return True


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
        logger.warning("MOVES THE REAL RIGHT ARM + ROBOTIQ GRIPPER{}:", " + THE CHASSIS" if a.chassis else "")
    if a.detect:
        logger.warning("  cylinders + bin from the head camera (angle {:.0f}){}", cfg.STAND_HEAD_ANGLE_DEG,
                       "; chassis drives the bin to ({:.2f},{:+.2f}) at {:.2f} m/s".format(
                           *cfg.STAND_BIN_PLACE_XY, cfg.STAND_CHASSIS_SPEED_MS) if a.chassis else "")
    else:
        logger.warning("  cylinder at ({:.2f},{:+.2f}); bin at ({:.2f},{:+.2f}); {} cycle(s)",
                       a.cyl_x, a.cyl_y, a.bin_x, a.bin_y, len(_slot_offsets(a)))
    logger.warning("  {}", "touch the desk with the closed fingertips at ({:.2f},{:+.2f}), report the finger "
                   "length (--measure-fingers)".format(a.probe_x, a.probe_y) if a.measure_fingers
                   else "grip, lift 10 cm, set back down, RELEASE (--pick-only)" if a.pick_only
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
        plans = [] if (a.measure_fingers or a.detect) else build_plans(g, a)
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
        if a.measure_fingers:
            measure_fingers(g, a.probe_x, a.probe_y)
        if a.detect:
            _run_detected(bot, g, a)
        for plan in plans:
            ok = run_object(g, plan, a.pick_only)
            logger.info("[{}] -> {} (slot y {:+.3f})", plan.label, "done" if ok else "SKIPPED",
                        plan.p_place_hi[1])
        if not a.keep:
            go_home(g)


def _slot_offsets(a: Args) -> list[float]:
    offsets = [float(v) / 100.0 for v in a.slots.split(",") if v.strip()]
    return offsets or [0.0]


def _plan_detected(g: GripperMover, scene, dy: float, chassis: bool) -> "ObjectPlan | None":
    """Plan the NEAREST detected cylinder into the slot ``dy`` left of the bin
    centre. With --chassis the bin will be driven to STAND_BIN_PLACE_XY, so the
    place is planned there; otherwise at the detected bin."""
    if not scene.cylinders:
        logger.warning("[detect] no cylinder in view")
        return None
    if scene.bin is None:
        logger.warning("[detect] no bin in view")
        return None
    bx, by = cfg.STAND_BIN_PLACE_XY if chassis else scene.bin[:2]
    x, y, conf = scene.cylinders[0]
    plan = plan_object(g, "cylinder", x, y, 0.0, (bx, by + dy),
                       np.asarray(cfg.STAND_HOME_JOINTS_RIGHT, dtype=float))
    describe(plan)
    return plan


def _detect_dry(bot, g: GripperMover, a: Args) -> None:
    scene = detect_scene(bot)
    if scene is None:
        return
    plan = _plan_detected(g, scene, _slot_offsets(a)[0], a.chassis)
    if a.chassis and scene.bin is not None:
        tx, ty = cfg.STAND_BIN_PLACE_XY
        logger.info("[chassis] would drive forward {:+.3f} m, left {:+.3f} m (bin ({:.3f},{:+.3f}) -> "
                    "({:.2f},{:+.2f}))", scene.bin[0] - tx, scene.bin[1] - ty, scene.bin[0], scene.bin[1], tx, ty)
        if scene.cylinders:
            cyl = np.asarray(scene.cylinders[0][:2])
            logger.info("[chassis] cylinder in the pick window now: {}; after that move: {}",
                        _in_pick_window(cyl), _in_pick_window(cyl - np.array([scene.bin[0] - tx, scene.bin[1] - ty])))
    logger.info("detect dry run -> {}", "no plan" if plan is None else ("OK" if plan.ok else "FAIL"))


def _run_detected(bot, g: GripperMover, a: Args) -> None:
    """Cycle: detect (arm at home, out of the view) -> pick the nearest
    cylinder -> lift -> [chassis drives the bin to the place spot while the arm
    carries + turns] -> place -> home -> [chassis drives back by everything it
    moved this cycle]. Re-detects every cycle."""
    net = [0.0, 0.0]        # chassis displacement (forward, left) accumulated within a cycle
    for k, dy in enumerate(_slot_offsets(a)):
        scene = detect_scene(bot)
        if scene is None:
            break
        if a.chassis and scene.cylinders and scene.bin is not None:
            # Pre-pick chassis move, only when needed. Preferred: the ONE move
            # that will be needed for the bin anyway, if it also brings the
            # nearest cylinder into the pick window. Otherwise, a cylinder
            # outside the window is brought to the proven pick spot. Either
            # way re-detect afterwards (the arm is at home, the view is clear).
            cyl = np.asarray(scene.cylinders[0][:2])
            d_bin = np.asarray(scene.bin[:2]) - np.asarray(cfg.STAND_BIN_PLACE_XY)
            moved = (0.0, 0.0)
            if _in_pick_window(cyl - d_bin) and not (_in_pick_window(cyl) and np.hypot(*d_bin) < 0.02):
                logger.info("[cycle {}] one chassis move serves both: bin -> place spot brings the cylinder to "
                            "({:.3f},{:+.3f})", k + 1, *(cyl - d_bin))
                moved = chassis_align(bot, scene.bin[:2], cfg.STAND_BIN_PLACE_XY, "bin")
            elif not _in_pick_window(cyl):
                logger.info("[cycle {}] cylinder ({:.3f},{:+.3f}) outside the pick window {} -> bring it to "
                            "the window centre ({:.3f},{:+.3f})", k + 1, cyl[0], cyl[1], cfg.STAND_CYL_PICK_WINDOW,
                            *_pick_window_mid())
                moved = chassis_align(bot, tuple(cyl), _pick_window_mid(), "cylinder")
            net[0] += moved[0]; net[1] += moved[1]
            if any(moved):
                scene = detect_scene(bot)
                if scene is None:
                    break
        plan = _plan_detected(g, scene, dy, a.chassis)
        if plan is None or not plan.ok:
            logger.error("[cycle {}] no feasible plan — stopping", k + 1)
            break

        def move(xy=scene.bin[:2] if scene.bin else None):
            d = chassis_align(bot, xy, cfg.STAND_BIN_PLACE_XY, "bin")
            net[0] += d[0]; net[1] += d[1]

        def refine():
            """Bin re-detected from where the chassis stopped (the arm holds the
            cylinder above the target, so the bin may be partly hidden): the
            residual vs the place target, if small enough to trust."""
            sc = detect_scene(bot, save=True)
            if sc is None or sc.bin is None:
                logger.warning("[refine] bin not re-detected after the chassis move — placing at the nominal point")
                return None
            d = np.asarray(sc.bin[:2]) - np.asarray(cfg.STAND_BIN_PLACE_XY)
            if np.hypot(*d) > float(cfg.STAND_PLACE_REFINE_MAX_M):
                logger.warning("[refine] bin residual ({:+.3f},{:+.3f}) too large to trust — nominal point",
                               d[0], d[1])
                return None
            logger.info("[refine] bin residual after the chassis move: ({:+.3f},{:+.3f}) m", d[0], d[1])
            return d

        ok = run_object(g, plan, a.pick_only, chassis_move=move if a.chassis else None,
                        refine_place=refine if a.chassis else None)
        if ok is None:
            logger.error("[cycle {}] stopped with the cylinder held — take over", k + 1)
            return
        logger.info("[cycle {}] -> {} (slot y {:+.3f})", k + 1, "done" if ok else "SKIPPED", plan.p_place_hi[1])
        go_home(g)          # clear the head view before the next detection
        if a.chassis and (abs(net[0]) > 1e-6 or abs(net[1]) > 1e-6):
            # back to where this cycle started (arm at home), so the remaining
            # cylinders are where the first detection saw them
            chassis_drive(bot, -net[0], -net[1], "return to the cycle start")
            net[0] = net[1] = 0.0


if __name__ == "__main__":
    _main(tyro.cli(Args))
