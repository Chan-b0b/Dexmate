"""Hover check: send the suction cup ABOVE the detected pick points, no contact.

For re-verifying the pick offsets (CASE_GRAB_OFFSET / SLOT_OFFSETS) after a
detection change. Same start as chassis_sequence (torso stance + both arms
safe-home, head 24 deg, camera-info intrinsics), then one BEV case detection;
for each pose key: SAFE_TRANSPORT_Z above it, a straight vertical leg down to
--hover m above the pick z, hold --hold s so the cup position over the case can
be checked by eye / tape, straight back up. Ends with a safe home. The arm
never goes below the hover height and never switches suction on.

Run from LGES/:
    ~/vla_venv/bin/python -m ik_demo.hover_check --layers 3
    ~/vla_venv/bin/python -m ik_demo.hover_check --layers 3 --keys CASE_PICK --hover 0.10 --hold 60
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from loguru import logger

from . import config as cfg
from .arm import connect_robot
from .config import resolve_poses
from .go_home import both_arms_home, safe_home
from .suction import SuctionMover
from . import chassis_sequence as cs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, required=True, help="layers remaining in the source stack")
    ap.add_argument("--keys", nargs="+", default=["CASE_PICK", "BAT_SRC_1", "BAT_SRC_2"])
    ap.add_argument("--hover", type=float, default=0.10, help="cup height above the pick z (m)")
    ap.add_argument("--hold", type=float, default=60.0, help="seconds to stay at each hover")
    args = ap.parse_args()
    if args.hover < 0.05:
        ap.error("--hover below 5 cm is not a hover check")

    from dexcontrol.core.config import get_robot_config
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        bot.sensors.head_camera.wait_for_active(timeout=5.0)
        cg = cs.dcb.bev.cg
        logger.info("intrinsics from {}: cx={:.2f} cy={:.2f}",
                    "camera info" if cg.load_intrinsics(bot) else "FALLBACK", cg.ZED_CX, cg.ZED_CY)
        with SuctionMover(bot) as m:
            if m.software_estop_active() or not m.ensure_ready(release_estop=False):
                raise SystemExit("arm not ready (software E-Stop?)")
            both_arms_home(bot, left=m)
            cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
            time.sleep(2.0)
            det = cs.detect(bot, args.layers)
            if det is None or not det.found:
                raise SystemExit("case not detected")
            center = cs._center_from_det(det)
            logger.info("case: x={:.4f} y={:+.4f} yaw={:+.2f} deg z_ee={:.3f}",
                        center[0], center[1], np.rad2deg(center[3]), center[2])
            poses = resolve_poses(center)
            try:
                for key in args.keys:
                    x, y, z, r, p, yw = poses[key]
                    rpy = (r, p, yw)
                    logger.info("[{}] pick ({:.4f}, {:+.4f}, {:.3f}) -> hover z {:.3f}",
                                key, x, y, z, z + args.hover)
                    if m.move_ee((x, y, cfg.SAFE_TRANSPORT_Z), rpy, quiet=False) is None:
                        logger.error("[{}] transport point unreachable — skipped", key)
                        continue
                    if m.move_ee_line((x, y, z + args.hover), rpy) is None:
                        logger.error("[{}] hover unreachable — back up", key)
                    else:
                        logger.warning("[{}] HOLDING {:.0f} s at {:.0f} cm above the pick point "
                                       "— check where the cup is over the case", key, args.hold,
                                       args.hover * 100)
                        time.sleep(args.hold)
                    m.move_ee_line((x, y, cfg.SAFE_TRANSPORT_Z), rpy)
            finally:
                safe_home(m)


if __name__ == "__main__":
    main()
