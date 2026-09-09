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
    warp needs the box RIM height: give --box-long-m (tape-measure the box's
    long side) and it is recovered by a plane sweep; otherwise --top-z is used
    as the rim height, which biases the box position by tens of mm if wrong.

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


def box_pose_from_detection(det, top_z: float, inset_m: float) -> BoxPose:
    """Grasp point for a detected box: midpoint of the LONG wall on the robot's
    right (base -y), ``inset_m`` in from the wall line toward the center. The
    pose yaw is the box's long-axis yaw, so pick_box's EE yaw (+ pi/2) closes
    the fingers ACROSS that wall."""
    yaw = float(np.deg2rad(det.base_yaw_deg))
    u_short = np.array([-np.sin(yaw), np.cos(yaw)])   # across the box, unit
    if u_short[1] > 0.0:                               # point to the robot's right
        u_short = -u_short
    half = float(det.dims_m[1]) / 2.0 - float(inset_m)
    gx = float(det.base_xy[0]) + float(u_short[0]) * half
    gy = float(det.base_xy[1]) + float(u_short[1]) * half
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

    set_head_pitch(bot, angle=head_angle)
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
        logger.info("rim height recovered from the {:.3f} m long side: plane z = {:.3f}",
                    box_long_m, z)
    else:
        z = float(top_z)
        logger.info("fixed rim height: warping at z = {:.3f} (--top-z / cfg.BOX_RIM_Z_M; pass "
                    "--box-long-m to recover it from the box size instead)", z)
        det = dbb.detect_box_bev(rgb, q_torso, q_head, z)
    out_dir = Path(__file__).resolve().parents[1] / "case_detection" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"box_detect_{time.strftime('%Y%m%d_%H%M%S')}.png"
    if not det.found:
        import cv2  # noqa: PLC0415
        cv2.imwrite(str(png), dbb.draw(det))
        logger.error("box NOT detected (plane z {:.3f}) — BEV saved to {}", z, png)
        return None, None
    pose = box_pose_from_detection(det, z, cfg.BOX_GRASP_EDGE_INSET_M)
    import cv2  # noqa: PLC0415
    cv2.imwrite(str(png), dbb.draw(det, [(pose.x, pose.y)]))
    logger.info("box: center ({:.3f},{:+.3f}) size {:.3f}x{:.3f} m yaw {:.1f} deg conf {:.2f} "
                "rim z {:.3f}", det.base_xy[0], det.base_xy[1], det.dims_m[0], det.dims_m[1],
                det.base_yaw_deg, det.conf, z)
    logger.info("grasp: right long-wall midpoint ({:.3f},{:+.3f}), fingers close across the "
                "wall (EE yaw {:+.2f} rad) — BEV + grasp point saved to {}",
                pose.x, pose.y, pose.yaw + cfg.BOX_GRASP_YAW_OFFSET_RAD, png)
    return pose, det


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
    if not a.dry:
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
