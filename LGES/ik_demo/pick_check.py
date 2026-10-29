"""Real pick check: pick the detected source case and put it straight back.

For re-verifying the pick offsets on-robot after a detection change (see
hover_check.py for the no-contact version). Same start as chassis_sequence
(both arms safe-home, head 24 deg, camera-info intrinsics), then task 3's
source flow for a case sitting ON BATTERIES (chassis_sequence.run_item, no
chassis moves): model-plane detect -> battery-top depth plane re-detect ->
median-of-N -> descent pre-check -> pick(lift_to_clear) -> view park,
then the target place's flow back on the SAME pose (straight, release
PLACE_RELEASE_ABOVE_M above the measured seal height), safe home.

``--layers`` = task 3's source case+battery layer count (its first prompt).

Run it from a TERMINAL: place() holds a part for an operator prompt when a
descent fails, and an automatic blow-off there would drop the case from height.

    cd ~/LGES/Dexmate/LGES && ~/vla_venv/bin/python -m ik_demo.pick_check --layers 3
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
    ap.add_argument("--layers", type=int, required=True,
                    help="source case+battery layers (task 3's first prompt)")
    args = ap.parse_args()
    cs.setup_logging()

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
            # the case is transparent: its plane comes from the batteries under it
            det, bat_top = cs._redetect_on_battery_depth(bot, "case", "source", args.layers,
                                                         det, 1, level=args.layers)
            if det is None:
                raise SystemExit("battery-top depth / re-detect failed")
            det = cs._refine_det(bot, args.layers, det, det.top_face_z)
            center = cs._center_from_det(det)
            pose = resolve_poses(center)["CASE_PICK"]
            exp_z = bat_top + cfg.SUCTION_LENGTH_M   # the cup meets the case on the batteries
            logger.info("case: x={:.4f} y={:+.4f} yaw={:+.2f} deg -> CASE_PICK ({:.4f}, {:+.4f}), "
                        "expected ee_z {:.4f}",
                        center[0], center[1], np.rad2deg(center[3]), pose[0], pose[1], exp_z)
            if not cs.descent_reachable(m, pose):
                safe_home(m)
                raise SystemExit("pick pose out of reach")
            if input("Pick the case and put it back? [y/N]: ").strip().lower() != "y":
                safe_home(m)
                return
            res = m.pick(pose, expected_z=exp_z, lift_to_clear=True,
                         lift_creep_out_m=cfg.DESCENT_CREEP_GAP_M)
            logger.info("pick: success={} reason={} contact_ee_z={}", res.success, res.reason,
                        None if res.contact_ee_z is None else round(res.contact_ee_z, 4))
            if res.success:
                cs._view_park(m, "case")   # task 3 runs this during the strafe
                back = m.place(pose, expected_z=res.contact_ee_z, lift_to_clear=True,
                               straight=True, release_above_m=cfg.PLACE_RELEASE_ABOVE_M)
                logger.info("place back: success={} reason={} contact_ee_z={}", back.success,
                            back.reason,
                            None if back.contact_ee_z is None else round(back.contact_ee_z, 4))
            safe_home(m)


if __name__ == "__main__":
    main()
