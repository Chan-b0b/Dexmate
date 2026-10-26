"""Depth vs touch check: how far the ZED depth's surface z is from the z the
cup actually stops at when it touches.

Robot in front of an OPAQUE, EMPTY, open box; one point, base xy given on the
CLI — the cup goes down onto the box FLOOR there, so keep it clear of the walls.

    1. arms home, head 24 deg; depth reads the floor height straight under --xy
    2. the cup goes there the way pick_place picks (planned stream to the creep
       line DESCENT_CREEP_GAP_M above the depth floor, then the per-tick creep),
       suction OFF, and stops at FORCE_CONTACT_THRESHOLD_N
    3. logs touch EE z (commanded + live) vs depth floor z + SUCTION_LENGTH_M,
       lifts straight up, safe home

    cd ~/LGES/Dexmate/LGES && ~/vla_venv/bin/python -m ik_demo.pick_place.depth_check --xy 0.75 0.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from loguru import logger

from .place_config import cfg
from ..arm import connect_robot
from ..drivers import suction_io
from ..go_home import both_arms_home, safe_home
from .place_mover import PlaceMover
from .. import chassis_sequence as cs
import depth_plane as dp  # case_detection (path added by chassis_sequence)  # noqa: E402


def depth_column_z(bot, xy, z_guess: float, iters: int = 3):
    """Depth surface z straight UNDER base `xy` (not along the camera ray through
    a guessed plane, as plane_from_depth does): sample the pixel of (x, y, z),
    move z to the measured height, re-sample. Returns (z, n_px, spread, xyz_hit)
    with z None when too few depth pixels landed."""
    rgb = cs._head_rgb(bot, fresh=False)
    depth = bot.sensors.head_camera.get_depth()
    if rgb is None or depth is None:
        logger.warning("depth: no frame (rgb={}, depth={})", rgb is not None, depth is not None)
        return None, 0, 0.0, None
    q_torso, q_head = cs._joints(bot)
    z, n, spread, hit = float(z_guess), 0, 0.0, None
    for _ in range(iters):
        u, v = dp.base_to_pixel((xy[0], xy[1], z), q_torso, q_head)
        d, n, spread = dp.sample_depth(depth, rgb.shape, u, v)
        if d is None:
            return None, n, spread, None
        hit = dp.pixel_to_base(u, v, d, q_torso, q_head)
        z_new = float(hit[2])
        done = abs(z_new - z) < 1e-4
        z = z_new
        if done:
            break
    return z, n, spread, hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy", type=float, nargs=2, required=True, metavar=("X", "Y"),
                    help="base-frame cup xy over the box floor (m)")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="wrist yaw in the case frame (deg, 0 = the taught pick yaw)")
    ap.add_argument("--z-guess", type=float, default=cfg.FLOOR_Z_BASE_M,
                    help="starting floor z for the depth read (m)")
    args = ap.parse_args()
    x, y = args.xy

    # import dexcontrol BEFORE the log sinks: its __init__ calls
    # logger.configure(handlers=...), which drops every sink added earlier
    # (1012: the run log file stopped after its first line)
    from dexcontrol.core.config import get_robot_config
    cs.setup_logging()
    if input(f"MOVES THE REAL RIGHT ARM (suction stays OFF): touch the box floor at "
             f"({x:.3f},{y:+.3f}). Connect + home? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
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
        with PlaceMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                raise SystemExit("arm not ready")
            both_arms_home(bot, left=m)
            cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
            time.sleep(2.0)

            fz, n_px, spread, hit = depth_column_z(bot, (x, y), args.z_guess)
            if fz is None:
                logger.error("depth: only {} valid px under ({:.3f},{:+.3f}) — nothing to "
                             "compare", n_px, x, y)
                return
            depth_ee = fz + cfg.SUCTION_LENGTH_M
            logger.info("depth: floor z={:.4f} at ({:.3f},{:+.3f}) ({} px, relief {:.0f}mm) "
                        "-> expected touch ee_z {:.4f}", fz, hit[0], hit[1], n_px,
                        spread * 1000.0, depth_ee)
            if input("Descend to touch? [y/N]: ").strip().lower() != "y":
                return

            rpy = np.array([cfg.GRASP_ORIENTATION_RPY[0], cfg.GRASP_ORIENTATION_RPY[1],
                            np.deg2rad(args.yaw) + cfg.GRASP_YAW])
            hard = float(cfg.FORCE_HARD_LIMIT_N)
            q = m._approach_and_hover(np.array([x, y, depth_ee]), rpy, depth_ee,
                                      to_creep_z=True,
                                      tick_cb=m._approach_force_guard(hard),
                                      creep_gap=float(cfg.DESCENT_CREEP_GAP_M))
            if q is None:
                logger.error("column unreachable — not touched")
                return
            try:
                res = m._descend_to_contact(depth_ee, rpy, hard, q)
                time.sleep(0.5)                       # let the arm catch up to the halt
                live_z = float(m.current_ee_pose()[0][2])
                if res.reason not in ("contact", "force_limit") or res.contact_ee_z is None:
                    logger.error("no touch ({}) — nothing to compare", res.reason)
                    return
                cmd_z = float(res.contact_ee_z)
                logger.info("TOUCH ({}): ee_z cmd {:.4f} / live {:.4f} vs depth {:.4f} -> "
                            "cmd-depth {:+.1f}mm, live-depth {:+.1f}mm (live-cmd {:+.1f}mm)",
                            res.reason, cmd_z, live_z, depth_ee,
                            (cmd_z - depth_ee) * 1000.0, (live_z - depth_ee) * 1000.0,
                            (live_z - cmd_z) * 1000.0)
                logger.info("  surface z: touch(live) {:.4f} vs depth {:.4f}",
                            live_z - cfg.SUCTION_LENGTH_M, fz)
            finally:
                m._lift_to_transport(rpy)
                safe_home(m)


if __name__ == "__main__":
    main()
