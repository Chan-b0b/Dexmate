"""One case from the box in front -> chassis strafe LEFT -> the empty bin there.

A single round trip, no layer loop: the shortest path through the pieces
chassis_sequence already uses, for rehearsing / demoing the move on its own.

    SOURCE (the box in front, where the operator parks the chassis)
      -> BEV-detect the case (median-of-N refine), reach pre-check the WHOLE
         descent column, suction-pick it (descend-to-contact finds the real
         grab height; the lift creeps out of the box walls first)
    strafe LEFT --left m, OPEN-LOOP (speed*time, no odometry), with the arm's
         view park running IN PARALLEL and joined before the camera is needed
    BIN (empty, in front)
      -> closed-loop bin align (absorbs the open-loop leg's drift), then the
         place center comes from a BIN detection + SEED_BIN_CENTER_OFFSET —
         the same closed-loop seed place the sequence's first case uses, so
         there is NO blind default-pose place: a failed detection walks
         SEED_BIN_SEARCH_STRAFES_M and then hands the operator the keyboard.
         The case is corner-seated (driven into the bin corner while
         descending) at the yaw it was PICKED at, so it mirrors the source
         stack's orientation instead of the chassis heading on arrival.
    strafe RIGHT --right m, then home.

Nothing here is remembered across runs (no ZTracker anchors, no learned leg
distances): one case means one measurement, so the expected z comes from the
detected warp plane and the seat z from the taught TARGET_DEFAULT_CASE_CENTER,
both refined by descend-to-contact.

Run as a PACKAGE so the ik config and the case_detection config don't collide
on the shared name `config`:

    python -m LGES.ik_demo.case_to_bin                  # 1.5 m left, 0.5 m back
    python -m LGES.ik_demo.case_to_bin --layers 1       # single case in the box
    python -m LGES.ik_demo.case_to_bin --left 1.2 --right 0.0

--layers is the SOURCE stack height (cases in the box): it sets the BEV warp
plane, so a wrong value biases the detected xy along the camera ray.

Needs a trained BEV detector (case_detection cfg.OBB_MODEL_PATH) for both the
case and the bin class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger

from . import config as cfg
from .arm import connect_robot
from .config import resolve_poses
from .drivers import suction_io
from .suction import SuctionMover

LABEL = "case"          # log prefix, and the pose key is CASE_PICK throughout


@dataclass
class Args:
    layers: int = 1   # cases in the source box (BEV warp plane)
    left: float = 1.8       # source -> bin leg (m, open-loop)
    right: float = 0.5       # leg back after the place (m); 0.0 = stay at the bin


def _pick_source_case(bot, mover: SuctionMover, layers: int):
    """Detect + pick the top case of the box in front. Returns the pick's
    source center (x, y, z_EE, yaw_rad) once the case is on the cup, else None.

    The detection/pre-check/prompt loop is chassis_sequence.run_item's source
    flow: a miss or an out-of-reach pose hands the operator the keyboard
    (`f/b/l/r`, `d` re-detects because the base frame moved, `q` gives up)
    rather than reaching blind."""
    from .chassis_sequence import (_arms_home, _center_from_det, _manual_strafe,
                                   _refine_det, descent_reachable, detect)  # noqa: PLC0415
    while True:
        det = detect(bot, layers)
        if det is not None and det.found:
            det = _refine_det(bot, layers, det)     # median-of-N for the pose
            logger.info("[{}] source case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg "
                        "conf={:.2f}", LABEL, det.base_xy[0], det.base_xy[1],
                        det.base_yaw_deg, det.conf)
            center = _center_from_det(det)
            pick_pose = resolve_poses(center)["CASE_PICK"]
            if descent_reachable(mover, pick_pose):
                break
            logger.warning("[{}] pick pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to give up", LABEL)
        else:
            logger.warning("[{}] source case NOT detected — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to give up", LABEL)
        if not _manual_strafe(bot, "adjust"):
            logger.error("[{}] giving up at the source — arms home", LABEL)
            _arms_home(bot, mover)
            return None
    # z is NOT taken from the detection: descend-to-contact finds the real grab
    # height, the detected plane only says where to expect it.
    exp_z = det.top_face_z + cfg.SUCTION_LENGTH_M
    # lift_creep_out_m: the case comes up from BETWEEN the box walls, where a
    # fast start throws the cup tens of mm off line (chassis_sequence does the
    # same for its case picks).
    res = mover.pick(pick_pose, expected_z=exp_z, lift_to_clear=True,
                     lift_creep_out_m=cfg.DESCENT_CREEP_GAP_M)
    if not res.success:
        logger.error("[{}] pick failed: {} — arms home", LABEL, res.reason)
        _arms_home(bot, mover)
        return None
    if res.contact_ee_z is None:
        logger.info("[{}] picked (no contact z reported)", LABEL)
    else:
        logger.info("[{}] picked: contact ee_z {:.4f} ({:+.1f}mm vs the expected {:.4f})",
                    LABEL, res.contact_ee_z, (res.contact_ee_z - exp_z) * 1000.0, exp_z)
    return center


def _bin_place_center(bot, source_yaw_rad: float, auto_search: bool):
    """Place center for the empty bin: BEV bin detection (median over
    SEED_BIN_DETECT_N frames, class-filtered to BIN_CLS_ID) + the measured
    SEED_BIN_CENTER_OFFSET bias, at the height taught in
    TARGET_DEFAULT_CASE_CENTER and at the yaw the case was PICKED at (the wrist
    then does no de-rotation in transit). Returns (x, y, z_EE, yaw_rad), or None
    if the operator gives up with the case still held.

    Detection is REQUIRED — the bin is empty, so nothing else says where its
    floor is, and a base-fixed default pose would land on a wall if the
    open-loop leg drifted. A miss walks SEED_BIN_SEARCH_STRAFES_M to change the
    view, then hands over the keyboard."""
    from .chassis_sequence import _detect_bin_xy, _manual_strafe  # noqa: PLC0415
    from .move_chassis import strafe_left, strafe_right  # noqa: PLC0415
    search = [float(s) for s in cfg.SEED_BIN_SEARCH_STRAFES_M] if auto_search else []
    while True:
        bxy = _detect_bin_xy(bot)
        if bxy is not None:
            x = bxy[0] + cfg.SEED_BIN_CENTER_OFFSET[0]
            y = bxy[1] + cfg.SEED_BIN_CENTER_OFFSET[1]
            logger.info("[{}] bin center ({:.3f},{:+.3f}) -> place ({:.3f},{:+.3f}) "
                        "at the source yaw {:+.1f} deg", LABEL, bxy[0], bxy[1], x, y,
                        float(np.rad2deg(source_yaw_rad)))
            return (x, y, cfg.TARGET_DEFAULT_CASE_CENTER[2], source_yaw_rad)
        logger.warning("[{}] bin NOT detected", LABEL)
        if search:
            move = search.pop(0)
            logger.warning("[{}] bin search: strafe {} {:.2f} m, re-detect "
                           "({} step(s) left)", LABEL,
                           "LEFT" if move > 0 else "RIGHT", abs(move), len(search))
            (strafe_left if move > 0 else strafe_right)(bot, distance_m=abs(move))
            continue
        logger.warning("[{}] adjust the chassis (f/b/l/r), `d` to re-detect, "
                       "`q` to stop (case still held)", LABEL)
        if not _manual_strafe(bot, "adjust"):
            logger.error("[{}] giving up at the bin (case still held)", LABEL)
            return None


def _place_in_bin(bot, mover: SuctionMover, source_yaw_rad: float) -> bool:
    """Align to the detected bin, then corner-seat the held case into it."""
    from .chassis_sequence import _align_to_bin, _manual_strafe, descent_reachable  # noqa: PLC0415
    # Closed-loop y align first: the long open-loop leg drifts, and the y gate
    # rejects a detection far off target (the SOURCE-side box is one leg away
    # and in view). fallback_right_m=0.0 = stay put when nothing is detected.
    _align_to_bin(bot, LABEL, target_y=cfg.TARGET_DEFAULT_CASE_CENTER[1],
                  fallback_right_m=0.0, max_err_m=cfg.CHASSIS_DETECT_Y_GATE_M)
    auto_search = True
    while True:
        center = _bin_place_center(bot, source_yaw_rad, auto_search)
        if center is None:
            return False
        auto_search = False        # the search strafes are a one-shot recovery
        place_pose = resolve_poses(center)["CASE_PICK"]
        # Corner-seat aim bias applied HERE, not in suction.place, so the reach
        # pre-check below checks the pose that is actually FLOWN.
        bias = float(cfg.CASE_CORNER_AIM_BIAS_M)
        place_pose = (place_pose[0] - np.sign(cfg.CASE_CORNER_DIR[0]) * bias,
                      place_pose[1] - np.sign(cfg.CASE_CORNER_DIR[1]) * bias,
                      *place_pose[2:])
        logger.info("[{}] corner-seat aim bias ({:+.1f},{:+.1f})mm (away from the "
                    "datum corner)", LABEL,
                    -np.sign(cfg.CASE_CORNER_DIR[0]) * bias * 1000,
                    -np.sign(cfg.CASE_CORNER_DIR[1]) * bias * 1000)
        if not descent_reachable(mover, place_pose):
            logger.warning("[{}] place pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to stop (case still held)", LABEL)
            if not _manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", LABEL)
                return False
            continue
        # expected_z stays None -> place() gates the misseat off the TAUGHT seat
        # z: a rim/wall landing contacts several cm above the bin floor and is
        # HELD for the operator instead of released.
        # corner_touch_first: this bin is EMPTY and this case is the only one,
        # i.e. the sequence's seed place — straight down to the first vertical
        # contact, drive only then, so no airborne lateral spike can latch an
        # axis before the case is actually down (see place()).
        pres = mover.place(place_pose, misseat_tol_m=cfg.SEED_MISSEAT_TOL_M,
                           lift_to_clear=True, corner_seat="case",
                           corner_touch_first=True)
        if (pres is not None and not getattr(pres, "success", True)
                and pres.reason == "unreachable" and pres.contact_ee_z is None):
            # The hover leg failed BEFORE the release gate: the case is still on
            # the cup and nothing moved, so the operator can reposition and the
            # bin-anchored pose gets re-detected (`d`) from the new base frame.
            logger.warning("[{}] place unreachable (case still held) — adjust the "
                           "chassis (f/b/l/r), `d` to re-detect + retry, `q` to stop", LABEL)
            if _manual_strafe(bot, "adjust"):
                continue
            logger.error("[{}] stopping at the bin (case still held)", LABEL)
            return False
        if pres is not None and not getattr(pres, "success", True):
            # Every other failure descended far enough to release (at the
            # operator gate or automatically), so the case IS in the bin.
            logger.warning("[{}] place failed ({}) — released at the bin anyway",
                           LABEL, pres.reason)
        elif pres is not None and pres.contact_ee_z is not None:
            logger.info("[{}] seated in the bin (contact ee_z {:.4f})", LABEL,
                        pres.contact_ee_z)
        else:
            logger.info("[{}] seated in the bin", LABEL)
        return True


def run_case_to_bin(bot, mover: SuctionMover, layers: int,
                    left_m: float, right_m: float) -> bool:
    """source detect+pick -> LEFT leg -> bin align+place -> RIGHT leg.
    False leaves the robot where the failure happened (the log says where)."""
    from .chassis_sequence import _park_during_legs, _view_park  # noqa: PLC0415
    from .move_chassis import strafe_left, strafe_right  # noqa: PLC0415

    center = _pick_source_case(bot, mover, layers)
    if center is None:
        return False

    # The park clears the arm from the head-camera view for the bin detection;
    # arm targets are base-frame, so running it DURING the leg only drags the
    # world-frame EE path sideways (the case is already lifted wall-clear).
    logger.info("[{}] strafe LEFT {:.2f} m to the bin", LABEL, left_m)
    _park_during_legs(LABEL, [lambda: _view_park(mover, LABEL)],
                      lambda: strafe_left(bot, distance_m=left_m,
                                          speed=cfg.CHASSIS_LEG_SPEED_MS))

    if not _place_in_bin(bot, mover, center[3]):
        return False

    if right_m > 0.0:
        logger.info("[{}] strafe back RIGHT {:.2f} m", LABEL, right_m)
        _park_during_legs(LABEL, [lambda: _view_park(mover, LABEL)],
                          lambda: strafe_right(bot, distance_m=right_m,
                                               speed=cfg.CHASSIS_LEG_SPEED_MS))
    return True


def _main(a: Args) -> None:
    from dexcontrol.core.config import get_robot_config

    from .chassis_sequence import _manual_strafe, set_head_pitch, setup_logging

    setup_logging()
    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION + CHASSIS: one case out of the box "
                   "in front, {:.2f} m LEFT, into the empty bin there, {:.2f} m back.",
                   a.left, a.right)
    logger.warning("Source stack height {} layer(s) (BEV warp plane); the bin place is "
                   "closed-loop on a BIN detection + offset ({:+.0f},{:+.0f})mm, "
                   "corner-seated.", a.layers,
                   cfg.SEED_BIN_CENTER_OFFSET[0] * 1000, cfg.SEED_BIN_CENTER_OFFSET[1] * 1000)
    logger.warning("Clear the strafe path. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        # Pin the state subscribers: the default AUTO policy undeclares the
        # zenoh liveliness token after 5 s without a read, which every
        # interactive prompt in this run sits through, and resume() then clears
        # the buffer — the first read after a prompt (joint pos for IK, wrench
        # for descend-to-contact) would wait on fresh data or come back None.
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        # Tilt the head down so the box is in view; the BEV homography uses the
        # live joints, so ~24 deg matches the training data.
        set_head_pitch(bot, angle=24.0)
        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            # Start clean: both arms safe-homed (lift-if-low first, so an arm
            # left down in a box by a previous run doesn't sweep the walls).
            logger.info("-> both arms safe home")
            from .go_home import both_arms_home
            both_arms_home(bot, left=m)
            logger.info("position the chassis at the BOX (case in front), then `d`")
            if not _manual_strafe(bot, "start"):
                logger.error("start positioning aborted (`q`) — run cancelled")
                return
            ok = run_case_to_bin(bot, m, a.layers, a.left, a.right)
            logger.info("-> home")
            m.move_joints(m._home_seed)
            logger.info("case -> bin {}", "OK" if ok else "FAILED")


if __name__ == "__main__":
    _main(tyro.cli(Args))
