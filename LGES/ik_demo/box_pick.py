"""Right-arm box pick: straight-down wall pinch on a detected (or hand-typed) box.

The box is far wider than the Robotiq opening, so the pick grips a WALL from
above — the midpoint of the long wall on the robot's RIGHT (base -y), one
finger inside the box, one outside — then lifts. Same choreography as the
suction pick: hover above the grasp point, descend vertically, close, lift.

Two ways to say where the box is:
  * hand-typed: --x --y --top-z --yaw-deg = the box top-face center + long-axis
    yaw (the grasp point is then that xy itself — air-test mode)
  * --detect: head camera -> BEV YOLO-OBB (case_detection/detect_box_bev) ->
    box center / size / yaw in base_link -> right-wall grasp point. The BEV
    warp needs the box RIM height. By default --top-z (cfg.BOX_RIM_Z_M) is only
    the first guess: the box FLOOR is measured from head-camera depth and the
    rim rebuilt as floor + cfg.BOX_WALL_HEIGHT_M, and the frame is re-warped
    there — the WARP PLANE only, not the grasp height (see detect_box for why
    they are deliberately different). --box-long-m recovers the plane from the
    box's tape-measured long side instead, and moves both.

``run_box_pick`` is the whole step (home -> detect -> grip -> home) and is
the SAME routine chassis_sequence runs at the end of a `--box` run, so that
last step can be rehearsed on its own here without the layer loop.

Run from LGES/:
    python -m ik_demo.box_pick --dry                          # plan only, no robot
    python -m ik_demo.box_pick                                # air test at the typed pose
    python -m ik_demo.box_pick --detect --box-long-m 0.62 --dry   # detect + plan, no arm motion
    python -m ik_demo.box_pick --detect                        # the sequence's last step, alone
    python -m ik_demo.box_pick --detect --home-left            # ... left arm homed first, as in the sequence
    python -m ik_demo.box_pick --keep                         # end at the hover, gripper as-is
    python -m ik_demo.box_pick --detect --carry                # actually hoist the box
    python -m ik_demo.box_pick --detect --seat-probe           # DIAGNOSIS: force vs height, never grips
By default the pick CLOSES on the wall, lifts the box BOX_LIFT_TEST_M, sets it
back down at the grasp height, RELEASES and retreats empty; --carry keeps
holding through the lift to the hover instead.
--detect moves the head (look-down angle) and, unless --dry, pins the torso
before the frame is taken. A debug PNG of the BEV detection + grasp point is
written to case_detection/out/.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from . import config as cfg
from .gripper import BoxPickResult, BoxPose, GripperMover
from .go_home import safe_home


@dataclass
class Args:
    x: float = 0.70          # box top center, base_link x (m) — ignored with --detect
    y: float = -0.35         # base_link y (m); the robot's right is negative
    top_z: float = cfg.BOX_RIM_Z_M   # box RIM height, base_link z (m); the warp plane unless --box-long-m
    yaw_deg: float = 0.0     # yaw of the box's long axis (deg, CCW+) — ignored with --detect
    detect: bool = False     # find the box with the head camera + BEV OBB model
    box_long_m: float | None = None   # tape-measured long side (m): recovers the rim height
    head_angle: float = 24.0 # head look-down angle for the detection frame (capture_bev default)
    dry: bool = False        # plan and log only — no arm motion (no robot at all unless --detect)
    keep: bool = False       # end at the hover with the gripper as-is (inspect the grasp)
    home_left: bool = False  # safe-home the LEFT arm first (the sequence does; it can block the view)
    carry: bool = False      # keep holding through the lift; default = grasp, lift 10 cm, set down, RELEASE
    seat_probe: bool = False # DIAGNOSIS: creep past the grasp height logging force vs height, never grip


def box_pose_from_detection(det, top_z: float, inset_m: float) -> BoxPose:
    """Grasp point for a detected box: midpoint of the LONG wall on the robot's
    right (base -y), ``inset_m`` in from the wall line toward the center, then
    shifted cfg.BOX_GRASP_Y_OFFSET_M in BASE y. The pose yaw is the box's
    long-axis yaw, so pick_box's EE yaw (+ pi/2) closes the fingers ACROSS
    that wall."""
    yaw = float(np.deg2rad(det.base_yaw_deg))
    u_short = np.array([-np.sin(yaw), np.cos(yaw)])   # across the box, unit
    if u_short[1] > 0.0:                               # point to the robot's right
        u_short = -u_short
    half = float(det.dims_m[1]) / 2.0 - float(inset_m)
    gx = float(det.base_xy[0]) + float(u_short[0]) * half
    gy = float(det.base_xy[1]) + float(u_short[1]) * half + float(cfg.BOX_GRASP_Y_OFFSET_M)
    return BoxPose(gx, gy, float(top_z), yaw)


def detect_box(bot, top_z: float, box_long_m: "float | None" = None,
               head_angle: float = 24.0):
    """Head frame -> BEV box detection -> (right-wall BoxPose, detection), or
    (None, None). Moves the head to ``head_angle``; the BEV plane is ``top_z``
    (the rim height) unless ``box_long_m`` recovers it by a size sweep. Writes
    a debug PNG (BEV + OBB + grasp cross) to case_detection/out/."""
    from .chassis_sequence import _head_rgb, _joints, set_head_pitch   # noqa: PLC0415
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "case_detection"))
    import detect_box_bev as dbb   # noqa: PLC0415

    # tol_deg: skip the move (and its flat 5 s wait) when the head is already
    # aimed — it usually is, from whatever task ran before. _head_rgb below
    # still waits for two fresh frames, so the settle is not lost.
    set_head_pitch(bot, angle=head_angle, tol_deg=2.0)
    rgb = _head_rgb(bot)
    if rgb is None:
        logger.error("no head-camera frame — is the head camera enabled?")
        return None, None
    q_torso, q_head = _joints(bot)
    if box_long_m:
        z, det = dbb.plane_from_size(rgb, q_torso, q_head, float(box_long_m))
        if det is None:
            logger.error("box not detected at any plane 0.60-1.05 m")
            return None, None
        # --box-long-m is an explicit operator choice: the recovered plane is
        # used for BOTH the warp and the grasp height (unlike the automatic
        # refinement below, which only moves the warp). Note it may well land
        # near the true rim and therefore out of the arm's reach — that is the
        # operator's call to make.
        z_grasp_ref = z
        logger.info("rim height recovered from the {:.3f} m long side: plane z = {:.3f} "
                    "(used for the warp AND the grasp height)", box_long_m, z)
    else:
        z = z_grasp_ref = float(top_z)
        logger.info("rim height first guess: warping at z = {:.3f} (--top-z / cfg.BOX_RIM_Z_M); "
                    "the floor depth below refines the WARP PLANE", z)
        det = dbb.detect_box_bev(rgb, q_torso, q_head, z)
        z_rim = _rim_from_floor_depth(bot, det, z, q_torso, q_head, rgb.shape)
        if z_rim is not None:
            # DETECTION ONLY (0911). Re-warping at the measured rim fixes the xy
            # and the SIZE: warping 81mm low made the box read 11% too big
            # (homothety about the camera nadir), and since the grasp point is
            # the box centre plus half the short side, that put the wall
            # midpoint ~20mm OUTSIDE the wall the fingers have to straddle.
            #
            # The GRASP height follows it too, since 0911: the TILTED wall
            # pinch (cfg.BOX_GRASP_TILT_DEG) reaches the true rim comfortably,
            # where the old vertical grasp could not reach it at all. The
            # operator's hand-tuned grip independently landed 35mm under this
            # measured rim, against BOX_GRASP_DEPTH_M's intended 30mm.
            logger.info("re-warping at the measured rim {:.4f} "
                        "(warp was {:.4f}, {:+.1f}mm) — used for the warp AND the "
                        "grasp height, so the fingers land {:.0f}mm below the rim",
                        z_rim, z, (z_rim - z) * 1000.0, cfg.BOX_GRASP_DEPTH_M * 1000.0)
            z = z_grasp_ref = z_rim
            det = dbb.detect_box_bev(rgb, q_torso, q_head, z)
    out_dir = Path(__file__).resolve().parents[1] / "case_detection" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"box_detect_{time.strftime('%Y%m%d_%H%M%S')}.png"
    if not det.found:
        import cv2  # noqa: PLC0415
        cv2.imwrite(str(png), dbb.draw(det))
        logger.error("box NOT detected (plane z {:.3f}) — BEV saved to {}", z, png)
        return None, None
    # xy/size from the WARP plane z (the real rim, when depth measured it);
    # top_z from z_grasp_ref, which is what sets the descent depth.
    pose = box_pose_from_detection(det, z_grasp_ref, cfg.BOX_GRASP_EDGE_INSET_M)
    import cv2  # noqa: PLC0415
    cv2.imwrite(str(png), dbb.draw(det, [(pose.x, pose.y)]))
    logger.info("box: center ({:.3f},{:+.3f}) size {:.3f}x{:.3f} m yaw {:.1f} deg conf {:.2f} "
                "rim z {:.3f}", det.base_xy[0], det.base_xy[1], det.dims_m[0], det.dims_m[1],
                det.base_yaw_deg, det.conf, z)
    logger.info("grasp: right long-wall midpoint ({:.3f},{:+.3f}), fingers close across the "
                "wall (EE yaw {:+.2f} rad) — BEV + grasp point saved to {}",
                pose.x, pose.y, pose.yaw + cfg.BOX_GRASP_YAW_OFFSET_RAD, png)
    _log_rim_depth(bot, det, pose, z, q_torso, q_head, rgb.shape)
    return pose, det


def _rim_from_floor_depth(bot, det, plane_guess: float, q_torso, q_head,
                          rgb_shape) -> "float | None":
    """Box rim z measured as (depth-measured FLOOR) + cfg.BOX_WALL_HEIGHT_M, or
    None to keep ``plane_guess`` — see the cfg.BOX_WALL_HEIGHT_M comment for why
    the floor is measured instead of the rim.

    The floor window sits at the box CENTRE, which comes from a detection warped
    at ``plane_guess``; that is fine even when the guess is off, because the
    floor is wide and flat there (plane_from_depth also re-projects the sample
    point through each height it measures, so the pixel converges onto the real
    surface). Gated on the window's own spread and on agreeing with
    cfg.FLOOR_Z_BASE_M, which the case stacking model already pins for this same
    box. Never raises: a missing depth stream must not stop a box pick."""
    if det is None or not det.found:
        return None
    try:
        import depth_plane as dp   # case_detection sibling (path set by detect_box)

        depth = bot.sensors.head_camera.get_depth()
        if depth is None:
            logger.warning("[box] floor depth: no depth frame (is the dexsensor publishing "
                           "depth?) — keeping the assumed rim {:.4f}", plane_guess)
            return None
        xy = (float(det.base_xy[0]), float(det.base_xy[1]))
        z_floor, n_px, spread = dp.plane_from_depth(depth, rgb_shape, q_torso, q_head,
                                                    xy, float(plane_guess), pct=50.0)
        if z_floor is None:
            logger.warning("[box] floor depth @ centre ({:.3f},{:+.3f}): only {} valid px "
                           "— keeping the assumed rim {:.4f}", xy[0], xy[1], n_px, plane_guess)
            return None
        dev = z_floor - float(cfg.FLOOR_Z_BASE_M)
        logger.info("[box] DEPTH floor @ centre ({:.3f},{:+.3f}): z={:.4f} [{} px, spread "
                    "{:.0f}mm] vs cfg.FLOOR_Z_BASE_M {:.3f} ({:+.1f}mm)", xy[0], xy[1],
                    z_floor, n_px, spread * 1000.0, cfg.FLOOR_Z_BASE_M, dev * 1000.0)
        if spread > float(cfg.BOX_FLOOR_DEPTH_MAX_SPREAD_M):
            logger.warning("[box] floor depth rejected: spread {:.0f}mm > {:.0f}mm — the window "
                           "is not on the floor. Keeping the assumed rim {:.4f}",
                           spread * 1000.0, cfg.BOX_FLOOR_DEPTH_MAX_SPREAD_M * 1000.0,
                           plane_guess)
            return None
        if abs(dev) > float(cfg.BOX_FLOOR_DEPTH_MAX_DEV_M):
            logger.warning("[box] floor depth rejected: {:+.0f}mm from FLOOR_Z_BASE_M (limit "
                           "{:.0f}mm) — contents or the rim, not the floor. Keeping the "
                           "assumed rim {:.4f}", dev * 1000.0,
                           cfg.BOX_FLOOR_DEPTH_MAX_DEV_M * 1000.0, plane_guess)
            return None
        return float(z_floor) + float(cfg.BOX_WALL_HEIGHT_M)
    except Exception as e:  # noqa: BLE001 — a measurement must not stop a box pick
        logger.warning("[box] floor depth failed ({}) — keeping the assumed rim {:.4f}",
                       e, plane_guess)
        return None


def _log_rim_depth(bot, det, pose, plane_z: float, q_torso, q_head, rgb_shape) -> None:
    """DIAGNOSIS ONLY: what the ZED depth reads for the box rim, next to the
    ``plane_z`` the detection was warped at (cfg.BOX_RIM_Z_M unless --top-z /
    --box-long-m). Nothing acts on it — this exists to find out whether the
    hand-set 0.65 is right before anything depends on the measurement.

    Two windows, because they answer different questions (depth_plane.
    sample_depth): at the GRASP point (the wall midpoint) a LOW percentile —
    the nearest point, i.e. the tallest thing in the window — is the rim, since
    that window also catches the outside wall face falling away and the box
    interior, both of which drag a median down. At the box CENTRE the MEDIAN is
    the contents/floor, which says how far the rim stands above what is inside.
    ``spread`` (p95-p5) flags a window straddling an edge: the lid work threw
    out a read with 59mm of it for exactly that reason.

    plane_z feeds BOTH the BEV warp (a wrong plane biases the detected x by
    ~0.7*dz, a homothety about the camera nadir) and the grasp height
    (z_grasp = plane_z - BOX_GRASP_DEPTH_M + BOX_FINGER_LENGTH_M), so an error
    here lands in two places at once.

    The depth frame is grabbed separately from the RGB above — fine while the
    chassis and head are parked for the detection, which is always the case
    here. Never raises: a missing depth stream must not stop a box pick."""
    try:
        import depth_plane as dp   # case_detection sibling (path set by detect_box)

        depth = bot.sensors.head_camera.get_depth()
        if depth is None:
            logger.warning("[box] rim depth: no depth frame (is the dexsensor "
                           "publishing depth?) — plane stays at the assumed {:.3f}", plane_z)
            return
        rim_xy = (float(pose.x), float(pose.y))
        z_rim, n_rim, sp_rim = dp.plane_from_depth(depth, rgb_shape, q_torso, q_head,
                                                  rim_xy, float(plane_z), pct=10.0)
        ctr_xy = (float(det.base_xy[0]), float(det.base_xy[1]))
        z_ctr, n_ctr, sp_ctr = dp.plane_from_depth(depth, rgb_shape, q_torso, q_head,
                                                   ctr_xy, float(plane_z), pct=50.0)
        if z_rim is None:
            logger.warning("[box] rim depth @ grasp ({:.3f},{:+.3f}): only {} valid px "
                           "— rim NOT seen by depth", rim_xy[0], rim_xy[1], n_rim)
        else:
            logger.info("[box] DEPTH rim @ grasp ({:.3f},{:+.3f}): z={:.4f} vs assumed "
                        "{:.4f} ({:+.1f}mm) [{} px, spread {:.0f}mm] -> would move the "
                        "grasp height by {:+.1f}mm", rim_xy[0], rim_xy[1], z_rim,
                        plane_z, (z_rim - plane_z) * 1000.0, n_rim, sp_rim * 1000.0,
                        (z_rim - plane_z) * 1000.0)
        if z_ctr is None:
            logger.info("[box] DEPTH interior @ centre ({:.3f},{:+.3f}): only {} valid px",
                        ctr_xy[0], ctr_xy[1], n_ctr)
        else:
            logger.info("[box] DEPTH interior @ centre ({:.3f},{:+.3f}): z={:.4f} "
                        "[{} px, spread {:.0f}mm]{}", ctr_xy[0], ctr_xy[1], z_ctr, n_ctr,
                        sp_ctr * 1000.0,
                        "" if z_rim is None else
                        f" -> rim stands {(z_rim - z_ctr) * 1000.0:+.0f}mm above it")
    except Exception as e:  # noqa: BLE001 — a log line must not stop a box pick
        logger.warning("[box] rim depth failed ({}) — plane stays at the assumed {:.3f}",
                       e, plane_z)


def run_box_pick(bot, gripper: GripperMover, left=None, pose: "BoxPose | None" = None,
                 mode: str = "lift_test", top_z: "float | None" = None,
                 box_long_m: "float | None" = None, head_angle: float = 24.0,
                 home_after: bool = True) -> BoxPickResult:
    """The whole box-pick step: home the arms, find the box, grip it, home.

    ONE implementation for both entry points — this module's CLI and the
    ``--box`` ending of chassis_sequence.run — so the sequence's last step can
    be rehearsed alone with ``python -m ik_demo.box_pick --detect``.

    ``left``: an existing left-arm mover to safe-home first (it otherwise sits
    in the head-camera view and the right arm's swing); None leaves the left
    arm alone. ``pose``: skip detection and grip there (hand-typed air test).
    ``top_z`` defaults to the fixed cfg.BOX_RIM_Z_M; ``box_long_m`` recovers
    the rim height from the box size instead. ``mode`` is pick_box's
    (lift_test / carry / release). ``home_after``: open the gripper and
    safe-home the right arm at the end.
    """
    logger.info("=== box pick (right arm): home -> {} -> grip ({}) -> {} ===",
                "detect" if pose is None else "typed pose", mode,
                "release + home" if home_after else "stay at the hover")
    if left is not None:
        safe_home(left)                      # clear of the head view and the right arm
    safe_home(gripper)                       # lift-if-low, then the right arm's home pose
    if pose is None:
        pose, _det = detect_box(bot, cfg.BOX_RIM_Z_M if top_z is None else float(top_z),
                                box_long_m, head_angle)
        if pose is None:
            logger.warning("[box] not detected — skipping the pick (arms left at home)")
            return BoxPickResult(False, "not_detected")
    res = gripper.pick_box(pose, mode=mode)
    res.box = pose                            # for the caller's reach-driven chassis adjust
    logger.info("[box] pick_box -> {} ({})", "GRASPED" if res.success else "no grasp", res.reason)
    if home_after:
        if gripper.gripper is not None and mode == "carry":
            # "carry" is the only mode that ends still holding; lift_test /
            # release / an empty grasp already opened inside pick_box, and a
            # redundant Robotiq command costs a blocking Modbus round-trip
            gripper.gripper.open()           # let go at the hover, before the joint-space home
        safe_home(gripper)
    else:
        logger.info("[box] staying at the hover, gripper as-is")
    return res


def _main(a: Args) -> None:
    if a.dry and not a.detect:
        box = BoxPose(a.x, a.y, a.top_z, float(np.deg2rad(a.yaw_deg)))
        res = GripperMover(robot=None).pick_box(box, dry=True)
        logger.info("dry run -> {} (yaw {})", res.reason,
                    "-" if res.yaw_used is None else f"{res.yaw_used:+.2f} rad")
        return

    from dexcontrol.robot import Robot

    logger.warning("=" * 60)
    if a.detect:
        logger.warning("DETECT the box with the head camera (head moves), then {}",
                       "PLAN ONLY (--dry)" if a.dry else "MOVE THE RIGHT ARM + ROBOTIQ:")
    else:
        logger.warning("MOVES THE REAL RIGHT ARM + ROBOTIQ GRIPPER:")
    if a.seat_probe:
        logger.warning("  SEAT PROBE: home -> hover -> creep PAST the grasp height while logging "
                       "wrist force vs height -> lift. The gripper never closes and nothing is "
                       "picked up; the palm is pressed onto the box RIM at {:.0f}N at most.",
                       cfg.BOX_SEAT_PROBE_BACKSTOP_N)
    elif not a.dry:
        logger.warning("  home -> hover over the grasp point -> straight down -> close -> {} -> lift -> {}",
                       "carry the box (--carry)" if a.carry
                       else f"lift {cfg.BOX_LIFT_TEST_M * 100:.0f}cm, set down, RELEASE",
                       "STAY at the hover (--keep)" if a.keep else "open + home")
    if not a.detect:
        logger.warning("  box top ({:.3f}, {:+.3f}, z {:.3f}) yaw {:+.1f} deg — nothing needs to be there.",
                       a.x, a.y, a.top_z, a.yaw_deg)
    logger.warning("Clear the right arm's workspace. Keep the e-stop within reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

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
        if not a.dry:
            release = g.software_estop_active()
            if release and input("Software E-Stop is active. Release it? [y/N]: ").strip().lower() != "y":
                return
            if not g.ensure_ready(release_estop=release):   # pins the torso (head pitch depends on it)
                logger.error("right arm not ready — aborting")
                return
            if not g.initialize():
                logger.error("gripper not available — aborting (arm not moved)")
                return

        box = None if a.detect else BoxPose(a.x, a.y, a.top_z, float(np.deg2rad(a.yaw_deg)))
        if a.dry:
            if a.detect:
                box, _det = detect_box(bot, a.top_z, a.box_long_m, a.head_angle)
                if box is None:
                    return
            res = g.pick_box(box, dry=True)
            logger.info("dry run -> {} (yaw {})", res.reason,
                        "-" if res.yaw_used is None else f"{res.yaw_used:+.2f} rad")
            return
        if a.seat_probe:
            if box is None:
                box, _det = detect_box(bot, a.top_z, a.box_long_m, a.head_angle)
                if box is None:
                    return
            safe_home(g)                      # the probe starts from the right arm's home
            g.probe_seat(box)
            safe_home(g)
            return
        left = None
        if a.home_left:
            from .arm import ArmMover  # noqa: PLC0415
            left = ArmMover(robot=bot, side="left", ee_frame=cfg.EE_FRAME)
        res = run_box_pick(bot, g, left=left, pose=box,
                           mode="carry" if a.carry else "lift_test",
                           top_z=a.top_z, box_long_m=a.box_long_m,
                           head_angle=a.head_angle, home_after=not a.keep)
        logger.info("box pick -> {} ({})", "OK" if res.success else "FAILED", res.reason)


if __name__ == "__main__":
    _main(tyro.cli(Args))
