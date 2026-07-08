"""In-place OBB reach test — NO chassis motion.

Isolates the detect -> resolve_poses -> IK -> reach chain from the chassis
strafe. Detects the case at the current position, logs the resolved pick pose
side-by-side with the taught (known-reachable) pose so a big delta explains an
"unreachable" IK, then (unless --dry) attempts the pick.

    python -m LGES.ik_demo.reach_in_place                 # item=case, source layer
    python -m LGES.ik_demo.reach_in_place --item battery_1
    python -m LGES.ik_demo.reach_in_place --dry           # detect + log poses only

Put the case where the arm should reach it; E-stop in reach.
"""

from __future__ import annotations

import argparse

from loguru import logger

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

from . import config as cfg
from .config import resolve_poses
from .suction import SuctionMover
from .drivers import suction_io
from .chassis_sequence import (ITEMS, detect, _center_from_det,
                               align_head_to_forward, descent_reachable)

_KEYS = dict(ITEMS)  # label -> resolve_poses key


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:+.4f}" for v in p) + ")"


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="case", choices=[l for l, _ in ITEMS])
    ap.add_argument("--layer", type=int, default=cfg.SRC_LAYERS_REMAINING,
                    help="layers_remaining for the warp plane (default source stack)")
    ap.add_argument("--dry", action="store_true", help="detect + log poses only, no motion")
    args = ap.parse_args()
    key = _KEYS[args.item]

    logger.warning("In-place reach test: item={} layer={} dry={}", args.item, args.layer, args.dry)
    if not args.dry and input("Moves the real arm. Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        align_head_to_forward(bot, angle=30.0)

        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if not args.dry and release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            if not args.dry:
                logger.info("-> home")
                m.move_joints(m._home_seed)

            det = detect(bot, args.layer)
            if det is None or not det.found:
                logger.error("detect failed (found={})", None if det is None else det.found)
                return
            center = _center_from_det(det)
            pose = resolve_poses(center)[key]
            taught = resolve_poses()[key]  # default SOURCE_CASE_CENTER

            logger.info("detected case: base xy=({:.3f},{:+.3f}) yaw={:.1f}deg top_z={:.4f} conf={:.2f}",
                        det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.top_face_z, det.conf)
            logger.info("resolve center (x,y,z_ee,yaw) = {}", _fmt(center))
            logger.info("[{}] pick pose  DETECTED = {}", args.item, _fmt(pose))
            logger.info("[{}] pick pose  TAUGHT   = {}", args.item, _fmt(taught))
            d = [pose[i] - taught[i] for i in range(3)]
            logger.info("[{}] delta xyz vs taught  = ({:+.3f}, {:+.3f}, {:+.3f}) m", args.item, *d)

            if args.dry:
                return
            if not descent_reachable(m, pose):
                logger.error("descent pre-check failed — not moving")
                return
            res = m.pick(pose, expected_z=det.top_face_z + cfg.SUCTION_LENGTH_M)
            # pick reaches -> descends -> suction -> lifts to transport; stop at the
            # lift (no place / no home) so we can inspect the reach+grab result.
            logger.info("pick {} — left lifted at transport", "OK" if res.success else f"FAILED: {res.reason}")


if __name__ == "__main__":
    _main()
