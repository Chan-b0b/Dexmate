"""Chassis-based, detection-driven pick & place.

Loops over LAYERS until the source stack is exhausted: each layer runs the full
item set (case -> battery_1 -> battery_2), then the stack heights step
automatically (source -1, target +1) so the BEV warp plane stays on the true
top face. cfg.SRC/TGT_LAYERS_REMAINING are only the STARTING heights.

Per item, one chassis round trip:

    strafe LEFT  (source ~ robot center, open-loop)
      -> detect the case (BEV) -> resolve_poses(detected) -> pick this item
      -> park the arm (clear the head-camera view)
    strafe RIGHT (target ~ robot center)
      -> detect the case at the target:
             found     -> place aligned to it (same case-frame offset)
             not found -> place at the default front pose (first case seeds it)
    strafe LEFT  (back, for the next item)

The chassis strafe is OPEN-LOOP (move_sideways = speed*time, no odometry); a
fresh BEV detection recenters the case in base_link at every visit, so the
imprecise strafe is fine. z is not taken from detection — descend-to-contact
finds the real grab/seat height.

Run as a PACKAGE so the ik config and the case_detection config don't collide
on the shared name `config`:

    python -m LGES.ik_demo.chassis_sequence          # (from the repo root)

Needs a trained BEV detector (case_detection cfg.OBB_MODEL_PATH).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg
from .config import resolve_poses
from .suction import SuctionMover
from .drivers import suction_io

# Detection lives in the sibling case_detection package (flat imports). We run in
# package mode so ik uses `from . import config` (-> LGES.ik_demo.config); adding
# case_detection to the path lets its flat `import config`/`import bev` resolve to
# ITS OWN modules without clashing.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
sys.path.insert(0, str(_HERE.parents[1] / "perception"))
import detect_case_bev as dcb  # noqa: E402
from utils import set_head_pitch  # perception/utils (path added above)  # noqa: E402


# item label -> pose key in resolve_poses (case-frame offset, same for pick@source
# and place@target — only the detected center differs).
ITEMS: tuple[tuple[str, str], ...] = (
    ("case", "CASE_PICK"),
    ("battery_1", "BAT_SRC_1"),
    ("battery_2", "BAT_SRC_2"),
)


def _head_rgb(bot):
    obs = bot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
    rgb = obs.get("left_rgb")
    return rgb.get("data") if isinstance(rgb, dict) else rgb


def _joints(bot):
    return (np.asarray(bot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(bot.head.get_joint_pos(), dtype=np.float64))


def detect(bot, layers_remaining: int):
    """One BEV case detection at the current chassis position, warped at the
    plane for `layers_remaining` (source = full stack, target = built-up)."""
    rgb = _head_rgb(bot)
    if rgb is None:
        return None
    return dcb.detect_case_bev(rgb, *_joints(bot), layers_remaining=layers_remaining)


def _center_from_det(det) -> tuple[float, float, float, float]:
    """CaseBEV -> resolve_poses source center (x, y, z_EE, yaw_rad).
    z_EE = top-face base z + EE->cup-tip offset.

    Resolve the OBB long-axis 180-deg ambiguity: a top-down suction grasp is
    symmetric under a 180-deg case flip, so wrap the detected yaw to [-90, 90)
    around the taught reference (case yaw 0). A ~180-deg-flipped detection
    otherwise sends the grasp wrist to an unreachable branch (confirmed via
    reach_sweep: at yaw~5.0 rad the reachable x window collapses to ~[0.96,1.01]).
    """
    x, y = det.base_xy
    z_ee = det.top_face_z + cfg.SUCTION_LENGTH_M
    yaw_deg = det.base_yaw_deg
    if yaw_deg >= 90.0:
        yaw_deg -= 180.0
    return (x, y, z_ee, float(np.deg2rad(yaw_deg)))


def _manual_strafe(bot, direction: str) -> bool:
    """Interactive chassis leg (cfg.CHASSIS_MANUAL): drive with `l/r/f/b [dist_m]
    [speed]` / `tl/tr [deg] [rad_s]` commands (move_chassis.py grammar), `d` done.
    Returns False if the user gives up with `q`."""
    from .move_chassis import (strafe_left, strafe_right, move_forward,
                               move_backward, turn_ccw, turn_cw)
    moves = {"l": strafe_left, "r": strafe_right, "f": move_forward, "b": move_backward}
    turns = {"tl": turn_ccw, "tr": turn_cw}
    logger.info("MANUAL chassis leg -> go {} : `l/r/f/b [dist_m] [speed]`, "
                "`tl/tr [deg] [rad_s]`, `d` = in position, `q` = give up", direction.upper())
    while True:
        parts = input(f"chassis[{direction}]> ").strip().lower().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "d":
            return True
        if cmd == "q":
            return False
        try:
            a = float(parts[1]) if len(parts) > 1 else None
            speed = float(parts[2]) if len(parts) > 2 else None
            if cmd in moves:
                moves[cmd](bot, distance_m=a, speed=speed)
            elif cmd in turns:
                turns[cmd](bot, angle_deg=a, speed=speed)
            else:
                logger.warning("commands: l/r/f/b [dist_m] [speed], tl/tr [deg] [rad_s], "
                               "d = done, q = give up")
        except (ValueError, IndexError) as e:
            logger.warning("parse error: {} — l/r/f/b [dist] [speed] | tl/tr [deg] [rad_s]", e)


def strafe(bot, direction: str) -> bool:
    """One chassis leg toward `direction`: manual (interactive) when
    cfg.CHASSIS_MANUAL, else the fixed open-loop speed*time strafe.
    Returns False only when the user gives up a manual leg (`q`)."""
    if cfg.CHASSIS_MANUAL:
        return _manual_strafe(bot, direction)
    v = cfg.CHASSIS_STRAFE_SPEED_MS if direction == "left" else -cfg.CHASSIS_STRAFE_SPEED_MS
    logger.info("chassis strafe {} ({:.2f} m/s x {:.1f}s)", direction, v, cfg.CHASSIS_STRAFE_TIME_S)
    bot.chassis.move_sideways(v, wait_time=cfg.CHASSIS_STRAFE_TIME_S)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    return True


def descent_reachable(mover: SuctionMover, pose) -> bool:
    """Pre-flight: solve IK for the WHOLE descent column at the pose's xy —
    from the current EE height down to DESCENT_CHECK_BOTTOM_EE_Z (box floor +
    suction length), regardless of the expected layer — before moving at all.
    Warm-chained downward so successive solves stay on one branch. False (with
    the failing z logged) if any step misses REACH_TOL / limits / collision."""
    x, y = float(pose[0]), float(pose[1])
    rpy = tuple(pose[3:6])
    z0 = float(mover.current_ee_pose()[0][2])
    bottom = float(cfg.DESCENT_CHECK_BOTTOM_EE_Z)
    zs = np.arange(max(z0, bottom), bottom - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
    if zs[-1] > bottom + 1e-9:
        zs = np.append(zs, bottom)
    seed = None
    for z in zs:
        sol = mover.solve_pose((x, y, float(z)), rpy,
                               seed=seed, min_motion=seed is not None)
        ok = (sol.pos_err_m <= cfg.REACH_TOL_M) and sol.in_limits and not sol.in_collision
        if not ok:
            logger.error("descent pre-check FAILED at z={:.3f} (err={:.1f}mm, in_lim={}, col={}) "
                         "for xy=({:.3f},{:+.3f})", z, sol.pos_err_m * 1000,
                         sol.in_limits, sol.in_collision, x, y)
            return False
        seed = sol.q
    logger.info("descent pre-check OK: xy=({:.3f},{:+.3f}) z {:.3f} -> {:.3f}", x, y, z0, bottom)
    return True


def _view_park(mover: SuctionMover, label: str) -> None:
    """Move the arm out of the head-camera view: Cartesian park (keeps the
    current EE orientation — safe with or without a held item, tunable via
    config), joint park fallback if unset/unreachable."""
    parked = False
    if cfg.ARM_VIEW_PARK_EE_POS is not None:
        _, rpy = mover.current_ee_pose()
        parked = mover.move_ee(tuple(cfg.ARM_VIEW_PARK_EE_POS), tuple(rpy)) is not None
        if not parked:
            logger.warning("[{}] Cartesian view-park unreachable — joint park fallback", label)
    if not parked:
        mover.move_joints(np.asarray(cfg.ARM_VIEW_PARK_JOINTS, dtype=np.float64))


def _arms_home(bot, mover: SuctionMover) -> None:
    """Safe-home BOTH arms before a failure strafe (lift-if-low first, so a low
    EE doesn't sweep through the box walls on the way home)."""
    logger.info("failure recovery: both arms -> safe home")
    from .go_home import both_arms_home
    both_arms_home(bot, left=mover)


def run_item(bot, mover: SuctionMover, label: str, pose_key: str,
             src_layers: int, tgt_layers: int) -> bool:
    """One item's full left->pick->right->place->left cycle.

    `src_layers` / `tgt_layers` are the CURRENT stack heights for this layer
    (the layer loop in run() steps them; cfg.SRC/TGT_LAYERS_REMAINING are only
    the starting values)."""
    logger.info("=== item: {} ({}) src_layers={} tgt_layers={} ===",
                label, pose_key, src_layers, tgt_layers)

    # --- SOURCE (already here from the initial move / previous item's return):
    #     detect, pick ---
    while True:
        det = detect(bot, src_layers)
        if det is None or not det.found:
            logger.error("[{}] source detect failed — arms home, then strafe right", label)
            _arms_home(bot, mover)
            strafe(bot, "right")   # don't leave it stuck on the source side
            return False
        logger.info("[{}] source case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f}",
                    label, det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf)
        pick_pose = resolve_poses(_center_from_det(det))[pose_key]
        if descent_reachable(mover, pick_pose):
            break
        # Out of reach: let the operator reposition the chassis, then re-detect
        # (the base frame moved, so the pose must be recomputed) and retry.
        logger.warning("[{}] pick pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to give up", label)
        if not cfg.CHASSIS_MANUAL or not strafe(bot, "adjust"):
            logger.error("[{}] giving up — arms home, then strafe right", label)
            _arms_home(bot, mover)
            strafe(bot, "right")
            return False
    res = mover.pick(pick_pose, expected_z=det.top_face_z + cfg.SUCTION_LENGTH_M)
    if not res.success:
        logger.error("[{}] pick failed: {} — arms home, then strafe right", label, res.reason)
        _arms_home(bot, mover)
        strafe(bot, "right")   # not on the target side yet -> move off the source side
        return False

    # --- park the arm so it doesn't block the head view during transport ---
    _view_park(mover, label)

    # --- TARGET: strafe right, detect, place (aligned if a case is there) ---
    strafe(bot, "right")
    while True:
        tdet = detect(bot, tgt_layers)
        if tdet is not None and tdet.found:
            logger.info("[{}] target case found -> aligned place", label)
            place_pose = resolve_poses(_center_from_det(tdet))[pose_key]
            exp_z = tdet.top_face_z + cfg.SUCTION_LENGTH_M
        else:
            logger.info("[{}] no target case -> default front place", label)
            place_pose = resolve_poses(cfg.TARGET_DEFAULT_CASE_CENTER)[pose_key]
            exp_z = None
        if descent_reachable(mover, place_pose):
            break
        # holding the item — no auto-recovery; reposition + retry, or stop here
        logger.warning("[{}] place pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to stop (item still held)", label)
        if not cfg.CHASSIS_MANUAL or not strafe(bot, "adjust"):
            logger.error("[{}] stopping on the right side (item still held)", label)
            return False
    pres = mover.place(place_pose, expected_z=exp_z)
    if pres is not None and not getattr(pres, "success", True):
        # already on the target (right) side -> reachable; stop here, don't return left
        logger.error("[{}] place failed: {} — stopping on the right side", label, pres.reason)
        return False

    # --- park again (clear the head view for the SOURCE detect), then return LEFT ---
    _view_park(mover, label)
    strafe(bot, "left")
    return True


def run(bot, mover: SuctionMover) -> bool:
    """Layer loop: each layer runs the full item set (case + 2 batteries), then
    the stacks step (source -1, target +1) so the BEV warp plane tracks the
    shrinking source / growing target. Loops until the source is exhausted."""
    # Position at the source ONCE; each item returns here at its end, so a failed
    # pick just stops (no strafe-right, robot left where it is).
    strafe(bot, "left")
    src, tgt = cfg.SRC_LAYERS_REMAINING, cfg.TGT_LAYERS_REMAINING
    layer = 0
    while src >= 1:
        layer += 1
        logger.info("=== layer {}: source stack {}, target stack {} ===", layer, src, tgt)
        for label, key in ITEMS:
            if not run_item(bot, mover, label, key, src, tgt):
                logger.error("stopping at layer {} item {} (robot left where it is) — "
                             "to resume, set SRC_LAYERS_REMAINING={} TGT_LAYERS_REMAINING={}",
                             layer, label, src, tgt)
                return False
        src -= 1
        tgt += 1
    logger.info("=== all {} layers moved ===", layer)
    return True


def _main() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION + CHASSIS (strafe L/R per item):")
    for label, key in ITEMS:
        logger.warning("   {} ({})", label, key)
    logger.warning("Clear the strafe path. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        # Tilt the head down so the box is in view (same as live_bev/capture); the
        # BEV homography uses the live joints, so ~30 deg matches the training data.
        set_head_pitch(bot, angle=30.0)
        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            # Start clean: BOTH arms safe-homed (lift-if-low first, so an arm
            # left down in a box by a previous run doesn't sweep the walls).
            logger.info("-> both arms safe home")
            from .go_home import both_arms_home
            both_arms_home(bot, left=m)
            ok = run(bot, m)
            logger.info("-> home")
            m.move_joints(m._home_seed)
            logger.info("sequence {}", "OK" if ok else "FAILED")


if __name__ == "__main__":
    _main()
