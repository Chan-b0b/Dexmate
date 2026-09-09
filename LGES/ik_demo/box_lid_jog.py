"""Rehearse the paper-lid handoff: the cup picks the lid, then the GRIPPER is
jogged in by hand.

Everything up to the handoff is the sequence's own code (chassis_sequence's
_box_lid_pick — detect the box-OBB lid class, depth-refine it, pick its centre
with the cardboard force pair, lift to cfg.BOX_LID_HANDOFF_Z_M), so what is
rehearsed here is what --box-lid flies. Then, instead of computing a grasp and
entering under force, the right arm just takes a STANDOFF pose with its yaw
matched to the lid's and stops, and the keyboard takes over. The left arm holds
the lid on the cup the whole time.

Every press prints the fingertips in the LID's own frame — how far past the
edge, how far off centre along the long axis, how high above the sheet — which
is exactly the number cfg.BOX_LID_GRASP_INSET_M wants.

    python -m LGES.ik_demo.box_lid_jog
    python -m LGES.ik_demo.box_lid_jog --standoff 0.05   # ask to stand closer in
    python -m LGES.ik_demo.box_lid_jog --above 0.15      # higher over the sheet

WHERE THE GRIPPER CAN STAND — the largest fingertip clearance OUTSIDE the -y
edge that solves (offline, demo torso stance, lid at x 0.85, cup at z 1.10 so
the sheet is at 0.945, EE 100mm above it; + is outside the edge, - is past it):

    lid centre y | +0.05  +0.10  +0.15  +0.20  +0.25  +0.30
    lid yaw   0  |   -40     -0    +60   +100   +120   +120
    lid yaw  +8  |  none   -120    -80    -40     -0    +40
    lid yaw  -8  |   -40    +20    +60   +120   +120   +120

That table is why the lid is CARRIED cfg.BOX_LID_HANDOFF_SHIFT_Y_M to +y after
the lift (this tool flies the same carry, so it rehearses the spot the run
uses): where the lid is picked the fingertips do not reach outside its edge at
all, and a lid yawed +8 deg has nowhere that solves. Raising the EE (+50..200mm)
or pulling it back in x does not unlock it — the wall is the -y reach with the
wrist pinned pointing +y. The standoff is still taken ABOVE the sheet rather
than outside it (walking inward from --standoff in 20mm steps to the first
clearance that solves, EE --above the plane so the open fingers are never around
the sheet until you jog down), because above is always clear and outside is not.

    w / s   +x / -x                        r / f   +z / -z
    a / d   +y / -y                        , / .   wrist spin -/+ (--yaw-step)
    - / +   step /2 or x2                  g       type an absolute `x y z`
    o / c   gripper open / close           x       suction off (hand it over)
    p       pose + the lid-frame numbers   q / ESC quit (leaves everything held)

E-stop in reach: this picks a real lid and flies both arms.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from dexcontrol.core.config import get_robot_config

from . import config as cfg
from .arm import connect_robot
from .chassis_sequence import (_box_lid_pick, _lid_frame, _manual_strafe,
                               _side_rpy, set_head_pitch, setup_logging)
from .drivers import suction_io
from .gripper import GripperMover
from .jog_ee import _KEYS, _fmt, _getch, _parse_xyz, _resolve, _step_to
from .suction import SuctionMover


def _tip(right, rpy):
    """(fingertip point, EE point) for the COMMANDED pose. The fingertips are
    cfg.BOX_FINGER_LENGTH_M out along the gripper's own tool z, so this follows
    the wrist wherever it is spun."""
    pos = np.asarray(right.fk(right._start_q())[0], dtype=float)
    tz = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).apply([0.0, 0.0, 1.0])
    return pos + tz * float(cfg.BOX_FINGER_LENGTH_M), pos


def _report(right, rpy, frame, step: float | None = None) -> None:
    """The fingertips in the LID's frame — the three numbers a config wants."""
    edge, approach, long_axis = frame
    tip, pos = _tip(right, rpy)
    rel = tip - edge
    logger.info("fingertips: {:+.0f}mm past the edge (= GRASP_INSET), {:+.0f}mm "
                "off centre, {:+.0f}mm above the sheet", float(rel @ approach) * 1000,
                float(rel @ long_axis) * 1000, float(rel[2]) * 1000)
    logger.info("ee {}  spin {:+.1f}deg{}", _fmt(pos),
                float(np.rad2deg(rpy[0] - cfg.BOX_LID_SIDE_RPY[0])),
                "" if step is None else "   step {:.0f}mm".format(step * 1000))


def _standoff(right, frame, want: float, above: float, lid_yaw_deg: float):
    """Somewhere reachable to stand before jogging in: fingertips ``want`` clear
    of the edge, EE ``above`` the sheet, yaw matched to the lid.

    Walked INWARD in 20mm steps because outward is what this arm cannot do (see
    the module docstring). Returns (pos, rpy, clearance) or (None, None, None).
    """
    edge, approach, _ = frame
    rpy = _side_rpy(lid_yaw_deg)
    floor = -(float(cfg.BOX_LID_GRASP_INSET_M) + 0.05)
    d = float(want)
    while d >= floor - 1e-9:
        pos = (edge - approach * (d + float(cfg.BOX_FINGER_LENGTH_M))
               + np.array([0.0, 0.0, float(above)]))
        use, sol = _resolve(right, tuple(pos), rpy, False)
        if use is not None:
            if abs(d - want) > 1e-9:
                logger.warning("the requested {:.0f}mm clearance is out of reach — "
                               "standing at {:+.0f}mm instead ({} the edge)",
                               want * 1000, d * 1000,
                               "outside" if d >= 0 else "PAST")
            return pos, rpy, d
        d -= 0.02
        logger.info("clearance {:+.0f}mm: {:.0f}mm short (limits={} collision={})"
                    "{}", (d + 0.02) * 1000, sol.pos_err_m * 1000, sol.in_limits,
                    sol.in_collision,
                    " — trying 20mm further in" if d >= floor - 1e-9 else "")
    return None, None, None


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standoff", type=float, default=0.10,
                    help="fingertip clearance OUTSIDE the lid edge to stand at, m "
                         "(default 0.10; searched inward until it solves)")
    ap.add_argument("--above", type=float, default=0.10,
                    help="EE height above the sheet at the standoff, m (default "
                         "0.10 — keeps the open fingers off the lid until you jog down)")
    ap.add_argument("--step", type=float, default=0.01,
                    help="metres per keypress (default 0.01; -/+ halves/doubles live)")
    ap.add_argument("--yaw-step", type=float, default=5.0,
                    help="degrees of wrist spin per , / . press (default 5)")
    ap.add_argument("--speed", type=float, default=cfg.DESCENT_CREEP_SPEED_M_S,
                    help=f"jog speed m/s (default {cfg.DESCENT_CREEP_SPEED_M_S})")
    ap.add_argument("--no-head", action="store_true", help="do not move the head")
    args = ap.parse_args()

    stamp = setup_logging()
    logger.warning("paper-lid handoff rehearsal (run {}) — picks a REAL lid with "
                   "the cup and then flies the RIGHT arm at {:.3f} m/s", stamp,
                   args.speed)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        # Same subscriber pinning the sequence needs: every prompt below sits
        # longer than dexcontrol's 5s idle timeout, and a resumed subscriber
        # hands back a cleared buffer to the next IK / wrench read.
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        if not args.no_head:
            set_head_pitch(bot, angle=24.0)

        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("left arm not ready — aborting")
                return
            logger.info("-> both arms safe home")
            from .go_home import both_arms_home
            both_arms_home(bot, left=m)
            # Gripper hardware FIRST (reset + activate + open), so a dead bus is
            # found before a lid is picked and left hanging on the cup.
            right = GripperMover(bot)
            if not (right.ensure_ready() and right.initialize()):
                logger.error("right gripper not ready — nothing to jog")
                return

            logger.info("position the chassis at the paper box, then `d`")
            if not _manual_strafe(bot, "start"):
                logger.error("start positioning aborted (`q`)")
                return
            got = _box_lid_pick(bot, m)
            if got is None:
                return
            det, short = got

            cup, _ = m.current_ee_pose()
            if cup[2] < float(cfg.BOX_LID_HANDOFF_Z_M) - 1e-3:
                _, rpy_live = m.current_ee_pose()
                if m.move_ee_vertical(float(cfg.BOX_LID_HANDOFF_Z_M),
                                      tuple(float(v) for v in rpy_live)) is None:
                    logger.error("could not lift to the handoff height — lid HELD, "
                                 "stopping")
                    return
                cup, _ = m.current_ee_pose()
            # The same carry the sequence flies (cfg.BOX_LID_HANDOFF_SHIFT_Y_M),
            # or this would rehearse a spot the run never visits.
            shift = float(cfg.BOX_LID_HANDOFF_SHIFT_Y_M)
            if shift > 1e-4:
                logger.info("carrying the lid {:.0f}mm to +y (y {:+.3f} -> {:+.3f})",
                            shift * 1000, float(cup[1]), float(cup[1]) + shift)
                if m.move_ee_line((float(cup[0]), float(cup[1]) + shift,
                                   float(cup[2])),
                                  tuple(float(v) for v in m.current_ee_pose()[1]),
                                  trace_tag="handoff carry") is None:
                    logger.error("the carry stalled — lid HELD, stopping")
                    return
                cup, _ = m.current_ee_pose()
            frame = _lid_frame(cup, det[2], short)
            edge, approach, _ = frame
            logger.info("=== lid at ({:.3f},{:+.3f}) yaw {:+.1f}deg, sheet at z "
                        "{:.3f}; -y edge midpoint ({:.3f},{:+.3f})",
                        cup[0], cup[1], det[2], cup[2] - float(cfg.SUCTION_LENGTH_M),
                        edge[0], edge[1])
            # The right arm's collision model holds the OTHER arm at the config it
            # read when it was BUILT — that was the left arm at home, and it is
            # now up holding the lid. Re-read it, or every reach verdict below is
            # answered against an arm that has moved.
            right._setup_model(None)
            right._setup_ik()
            right._setup_collision()

            pos, rpy, clear = _standoff(right, frame, float(args.standoff),
                                        float(args.above), det[2])
            if pos is None:
                logger.error("nothing on the approach line solves, from {:+.0f}mm "
                             "outside the edge to {:+.0f}mm past it — the lid is "
                             "too far right for this arm. Lid still HELD; hand it "
                             "over further LEFT (chassis) and try again.",
                             args.standoff * 1000,
                             -(cfg.BOX_LID_GRASP_INSET_M + 0.05) * 1000)
                return
            logger.warning("standing at {:+.0f}mm clearance (= inset {:+.0f}mm), "
                           "{:.0f}mm above the sheet: ee {} — the lid is NOT in the "
                           "collision model, watch it", clear * 1000, -clear * 1000,
                           args.above * 1000, _fmt(pos))
            if input("Move the gripper there? [y/N]: ").strip().lower() != "y":
                return
            if right.move_ee(tuple(pos), rpy, quiet=False) is None:
                logger.error("the standoff went unreachable between the check and "
                             "the move — not jogging")
                return

            jog = SimpleNamespace(free_yaw=False, speed=float(args.speed))
            step, spin = float(args.step), float(det[2])
            _report(right, rpy, frame, step)
            logger.info("keys: w/s=+-x  a/d=+-y  r/f=+-z  ,/.=spin  -/+=step  "
                        "o/c=open/close  x=suction off  p=pose  g=goto  q=quit")
            logger.info("`a` (+y) moves TOWARD the lid; at yaw {:+.1f}deg the "
                        "approach axis is ({:+.2f},{:+.2f})", det[2], *approach[:2])
            while True:
                try:
                    k = _getch()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if k in ("q", "\x1b", "\x03", ""):
                    break
                if k in _KEYS:
                    axis, sign = _KEYS[k]
                    target = [float(v) for v in right.fk(right._start_q())[0]]
                    target[axis] += sign * step
                    _step_to(right, tuple(target), rpy, jog)
                    _report(right, rpy, frame, step)
                elif k in (",", "."):
                    want = spin + (args.yaw_step if k == "." else -args.yaw_step)
                    want_rpy = _side_rpy(want)
                    if _step_to(right, tuple(float(v) for v in
                                             right.fk(right._start_q())[0]),
                                want_rpy, jog):
                        rpy, spin = want_rpy, want
                    _report(right, rpy, frame, step)
                elif k in ("-", "_"):
                    step = max(0.0005, step / 2.0)
                    logger.info("step {:.1f}mm", step * 1000)
                elif k in ("+", "="):
                    step = min(0.20, step * 2.0)
                    logger.info("step {:.1f}mm", step * 1000)
                elif k == "o":
                    right.gripper.open()
                    logger.info("gripper OPEN")
                elif k == "c":
                    right.gripper.close()
                    logger.info("gripper CLOSED")
                elif k == "x":
                    logger.warning("suction off — if the gripper is not holding the "
                                   "lid it DROPS")
                    if input("Release the cup? [y/N]: ").strip().lower() == "y":
                        suction_io.release()
                        logger.info("cup released — the lid is the gripper's now")
                elif k == "p":
                    _report(right, rpy, frame, step)
                elif k == "g":
                    target = _parse_xyz(input("goto x y z: "))
                    if target is None:
                        logger.warning("need three numbers, e.g. 0.85 -0.26 1.04")
                    else:
                        _step_to(right, target, rpy, jog)
                        _report(right, rpy, frame, step)
            _report(right, rpy, frame)
            logger.warning("left as it is — the cup may still be holding the lid "
                           "and the gripper may still be closed on it")


if __name__ == "__main__":
    _main()
