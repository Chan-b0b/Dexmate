"""Run ONLY the lid PLACE leg, with the lid put on the cup BY HAND.

The --lid sequence has to pick a lid off a box before it can place one, so every
attempt at debugging the place costs a full pick + a hand-driven chassis leg.
This skips all of it: take the place stance, turn the vacuum ON, let the operator
press a lid onto the cup, and on ENTER run the detect-and-stack leg.

    python -m LGES.ik_demo.lid_place
    python -m LGES.ik_demo.lid_place --yaw-delta 15    # held lid is 15 deg off
    python -m LGES.ik_demo.lid_place --no-strafe       # already in position

What it does, in order:

  1. arm ready + BOTH arms safe-homed (same start as the --lid run)
  2. hand-drive the chassis to the unload spot (`d` when in position; skip the
     leg entirely with --no-strafe)
  3. torso to LID_PLACE_TORSO_DEG and BOTH arms to their unload stow joints,
     together — chassis_sequence.lid_place_stance, the same call the run makes
  4. vacuum ON, and it prints WHERE to put the lid: the cup sits
     LID_GRAB_OFFSET_M from the lid's centre in the LID's frame, and the lid's
     long axis has to end up at a stated base-frame angle (see below). Press
     ENTER once it is on and DI0 has latched.
  5. chassis_sequence.run_lid_place — the run's own place leg, not a copy

The place leg is SHARED with --lid (run_lid ends by calling it), so anything
fixed while debugging here lands in the real sequence too.

WHERE TO PUT THE LID (step 4 prints the numbers for the live pose):
The place aims the cup over the same point of the floor lid that the pick
grabbed, so the held lid's offset and yaw have to match what a pick would have
produced. run_lid solves the pick wrist as w = L_pick + GRASP_YAW + yaw_delta,
so a lid held at the canonical yaw (--yaw-delta 0) means its long axis lies at
wrist_yaw - GRASP_YAW in the base frame. If the placed lid comes out rotated
against the floor lid, the held lid was not at the canonical yaw: measure the
error and pass it as --yaw-delta.
"""

from __future__ import annotations

import argparse

import numpy as np
from loguru import logger

from dexcontrol.core.config import get_robot_config

from . import config as cfg
from .arm import connect_robot
from .chassis_sequence import (_manual_strafe, lid_place_stance, run_lid_place,
                               setup_logging)
from .drivers import suction_io
from .suction import SuctionMover


def _hand_off_instructions(mover, yaw_delta: float) -> None:
    """Print where the lid has to sit on the cup, from the LIVE wrist pose.

    Inverts _pick_yaw_search's wrist solve, w = L + GRASP_YAW + yaw_delta, for
    the lid axis L — so the printed angle is the one that makes the place's
    re-added ``yaw_delta`` come out right, not just the canonical one."""
    pos, rpy = mover.current_ee_pose()
    wrist = float(rpy[2])
    lid_yaw = float((wrist - float(cfg.GRASP_YAW) - float(yaw_delta) + np.pi)
                    % (2.0 * np.pi) - np.pi)
    ox, oy = cfg.LID_GRAB_OFFSET_M
    logger.warning("=" * 68)
    logger.warning("VACUUM IS ON — put the lid on the cup now.")
    logger.warning("  cup is at ({:.3f},{:+.3f},{:.3f}), wrist yaw {:+.1f}deg",
                   pos[0], pos[1], pos[2], float(np.rad2deg(wrist)))
    logger.warning("  1) the cup must land {:+.0f},{:+.0f}mm from the lid CENTRE, "
                   "measured in the LID's own frame", ox * 1000.0, oy * 1000.0)
    logger.warning("  2) the lid's LONG AXIS should lie at {:+.1f}deg in the base "
                   "frame (= wrist {:+.1f} - GRASP_YAW {:+.1f} - yaw_delta "
                   "{:+.1f}); 180deg off is the same lid",
                   float(np.rad2deg(lid_yaw)), float(np.rad2deg(wrist)),
                   float(np.rad2deg(cfg.GRASP_YAW)),
                   float(np.rad2deg(yaw_delta)))
    logger.warning("  a lid held anywhere else lands rotated / offset against the "
                   "floor lid — pass the error as --yaw-delta")
    logger.warning("=" * 68)


def _seal_state() -> "bool | None":
    """DI0 sealed? None if the monitor never connected."""
    vac = suction_io.VacuumMonitor()
    vac.start()
    try:
        if not vac.is_connected():
            return None
        return bool(vac.is_sealed())
    finally:
        vac.stop()


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Run only the lid place leg, lid handed to the cup by hand.")
    ap.add_argument("--yaw-delta", type=float, default=0.0, metavar="DEG",
                    help="extra wrist rotation to re-add at the place, i.e. how "
                         "far the HELD lid is off the canonical yaw (default 0)")
    ap.add_argument("--no-strafe", action="store_true",
                    help="skip the hand-driven chassis leg — the robot is already "
                         "at the unload spot")
    args = ap.parse_args()

    setup_logging()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        logger.warning("=" * 68)
        logger.warning("LID PLACE ONLY. Moves BOTH arms and the TORSO (to {} deg). "
                       "Clear the robot. E-stop in reach.", cfg.LID_PLACE_TORSO_DEG)
        logger.warning("=" * 68)
        if input("Continue? [y/N]: ").strip().lower() != "y":
            return

        m = SuctionMover(bot)
        release = m.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        # ensure_ready (demo stance), not ensure_ready_live_torso: this has to
        # start where the --lid run starts, because lid_place_stance's taught
        # stow joints and its clip_to_band were checked from that stance.
        if not m.ensure_ready(release_estop=release):
            logger.error("arm not ready — aborting")
            return
        from .go_home import both_arms_home
        both_arms_home(bot, left=m)

        if not args.no_strafe and not _manual_strafe(bot, "unload"):
            logger.error("unload positioning aborted (`q`) — nothing moved further")
            return

        lid_place_stance(bot, m)

        yaw_delta = float(np.deg2rad(args.yaw_delta))
        suction_io.suction_on()
        _hand_off_instructions(m, yaw_delta)
        input("lid on the cup? ENTER to detect and place (ctrl-c to abort): ")

        sealed = _seal_state()
        if sealed is None:
            logger.warning("DI0 monitor did not connect — the seal is UNVERIFIED")
        elif not sealed:
            logger.warning("DI0 says the cup is NOT sealed — the place would "
                           "descend and release nothing")
            if input("Place anyway? [y/N]: ").strip().lower() != "y":
                suction_io.suction_off()
                logger.info("aborted — vacuum off, robot left at the place stance")
                return
        else:
            logger.info("DI0 sealed — lid is held")

        ok = run_lid_place(bot, m, yaw_delta)
        logger.info("lid place {}", "OK" if ok else "FAILED")


if __name__ == "__main__":
    _main()
