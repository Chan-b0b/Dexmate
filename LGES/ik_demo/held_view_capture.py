"""Held-view capture: what the cameras see of the destination bin WITHOUT the view park.

Data for the "no view park" study (bin_pose/): does the fixed bin's rim stay
visible enough to locate it while the arm still holds the part over it?

Runs the REAL chassis_sequence.run_item for the chosen item(s) — pick, legs,
target detection and place are all unchanged. Only the OUTBOUND leg (source ->
destination, part held) is swapped: instead of parking the arm during the leg,

  1. the arm only rises to SAFE_TRANSPORT_Z during the leg (as _view_park
     does first), and stays over the pick xy — i.e. over the destination bin;
  2. chassis stopped: capture "carry" (head RGB + depth, right wrist camera, joints);
  3. each --heights z (UP only, part held, straight vertical): capture "carry_z<z>";
  4. the normal view park (same as every run), capture "parked" — the clean
     reference from the SAME chassis pose, for both cameras;
  5. run_item continues as normal (detect + place).

The RETURN leg (cup empty, 1026 AprilTag study) is swapped the same way:

  6. right after the place the arm only rises to SAFE_TRANSPORT_Z: capture
     "placed" (the just-placed part under the empty arm, chassis still);
  7. the leg runs with the arm left there: capture "return" at the next station;
  8. the normal view park, capture "return_parked".

Plus "start": once at the source after the positioning `d`, arms at home.

The right arm never moves here beyond the start-up home: its home
(cfg.HOME_JOINTS_RIGHT, 1015) IS the wrist-camera view pose, aimed at the bin in
front of the robot.
Chassis legs use the learned distances but do NOT write chassis_legs.json.

Run from LGES/ (E-stop in reach — moves arms, suction, chassis):
    ~/vla_venv/bin/python -m ik_demo.held_view_capture --src 2 --tgt 2
    ~/vla_venv/bin/python -m ik_demo.held_view_capture --src 2 --tgt 2 --items case battery_2 --heights 1.22

--src/--tgt are the stack counts as in the task-3 menu. All items run at that
one layer (no stepping), in the order given. A case at --tgt 1 is the run's
"first case" and goes to the bin LEFT of the source, as in task 3.

Writes RUN_LOG_DIR/held_view/<stamp>/<HHMMSS>_<label>_<tag>_<k>.npz with
    head_rgb (H,W,3)  head_depth (H,W) m  q_torso  q_head  q_left_arm  q_right_arm
    ee_pos  ee_rpy  (suction arm, base_link)  label  tag  src_layers  tgt_layers
    <cam>_rgb  for every wrist camera that delivered a frame
plus camera_info.json (intrinsics of every camera that answered).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import chassis_sequence as cs
from . import config as cfg
from .arm import connect_robot
from .go_home import both_arms_home, safe_home
from .suction import SuctionMover

WRIST_CAMS = ("right_wrist_camera",)   # the robot has no left wrist camera
LABEL_KEY = dict(cs.ITEMS)
MAX_HEIGHT_Z = 1.30          # cap for the extra holds (part held, arm near the head)


def _wrist_rgb(cam):
    obs = cam.get_obs(include_timestamp=True)
    return obs.get("data") if isinstance(obs, dict) else obs


class Capture:
    def __init__(self, bot, mover: SuctionMover, out: Path, frames: int,
                 src: int, tgt: int) -> None:
        self.bot, self.mover, self.out, self.frames = bot, mover, out, frames
        self.src, self.tgt = src, tgt
        self.wrist = {n: getattr(bot.sensors, n) for n in WRIST_CAMS
                      if bot.has_sensor(n)}

    def save_camera_info(self) -> None:
        info = {}
        for n in ("head_camera", *self.wrist):
            try:
                info[n] = getattr(self.bot.sensors, n).get_camera_info()
            except Exception as e:  # noqa: BLE001 — diagnostic only
                info[n] = f"unavailable: {e}"
        (self.out / "camera_info.json").write_text(json.dumps(info, indent=1, default=str))

    def __call__(self, label: str, tag: str) -> None:
        """`frames` fresh head frames (each after two new frames, see _head_rgb)
        with depth + wrist + joints. Never raises — a capture must not stop a run."""
        for k in range(self.frames):
            try:
                rgb = cs._head_rgb(self.bot)
                depth = self.bot.sensors.head_camera.get_depth()
                q_torso, q_head = cs._joints(self.bot)
                pos, rpy = self.mover.current_ee_pose()
                d = dict(head_rgb=rgb, head_depth=depth, q_torso=q_torso, q_head=q_head,
                         q_left_arm=np.asarray(self.bot.left_arm.get_joint_pos()),
                         q_right_arm=np.asarray(self.bot.right_arm.get_joint_pos()),
                         ee_pos=np.asarray(pos), ee_rpy=np.asarray(rpy),
                         label=label, tag=tag, src_layers=self.src, tgt_layers=self.tgt)
                for n, cam in self.wrist.items():
                    w = _wrist_rgb(cam)
                    if w is not None:
                        d[f"{n}_rgb"] = w
                f = self.out / f"{time.strftime('%H%M%S')}_{label}_{tag}_{k}.npz"
                np.savez_compressed(f, **d)
                logger.info("[capture] {} {} {}/{}: ee z {:.3f} -> {} (wrist: {})",
                            label, tag, k + 1, self.frames, pos[2], f.name,
                            [n for n in self.wrist if f"{n}_rgb" in d] or "none")
            except Exception as e:  # noqa: BLE001
                logger.warning("[capture] {} {} frame {} failed ({})", label, tag, k, e)


def _rise_to_transport(mover: SuctionMover) -> None:
    """The rise half of chassis_sequence._view_park (keep in step with it): from
    the lift_to_clear stop, up to SAFE_TRANSPORT_Z — but no park move after."""
    pos, rpy = mover.current_ee_pose()
    if cfg.LIFT_VERTICAL_TO_Z - 1e-3 <= pos[2] < cfg.LIFT_CLEAR_EE_Z - 1e-3:
        if mover.move_ee([float(pos[0]), float(pos[1]), cfg.LIFT_CLEAR_EE_Z],
                         tuple(rpy), quiet=True) is None:
            mover._best_effort_ascent(rpy)
        pos, rpy = mover.current_ee_pose()
    if pos[2] < cfg.SAFE_TRANSPORT_Z - 0.01:
        mover.move_ee_vertical(cfg.SAFE_TRANSPORT_Z, rpy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, required=True, help="source case+battery layers")
    ap.add_argument("--tgt", type=int, required=True, help="target stack count")
    ap.add_argument("--items", nargs="+", default=["case"], choices=list(LABEL_KEY))
    ap.add_argument("--heights", nargs="*", type=float, default=[],
                    help=f"extra hold heights, EE z (m), {cfg.SAFE_TRANSPORT_Z}..{MAX_HEIGHT_Z}")
    ap.add_argument("--frames", type=int, default=3, help="frames saved per capture point")
    ap.add_argument("--no-wrist", action="store_true", help="head camera only")
    args = ap.parse_args()
    bad = [z for z in args.heights if not cfg.SAFE_TRANSPORT_Z <= z <= MAX_HEIGHT_Z]
    if bad:
        ap.error(f"--heights {bad}: holds go UP only, "
                 f"{cfg.SAFE_TRANSPORT_Z}..{MAX_HEIGHT_Z} m (part held over the bin)")

    # dexcontrol's import runs logger.configure(), which drops every sink added
    # before it — import it BEFORE setup_logging or the run log stays empty
    from dexcontrol.core.config import get_robot_config
    cs.setup_logging()
    out = Path(cfg.RUN_LOG_DIR) / "held_view" / cs.run_stamp()
    out.mkdir(parents=True, exist_ok=True)
    logger.warning("HELD-VIEW CAPTURE — moves the real arms + suction + chassis. "
                   "Items {} at src {} / tgt {}, extra heights {}. E-stop in reach.",
                   args.items, args.src, args.tgt, args.heights or "none")
    if input("Connect + safe-home both arms? [y/N]: ").strip().lower() != "y":
        return

    from .drivers import suction_io
    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    if not args.no_wrist:
        for n in WRIST_CAMS:
            configs.enable_sensor(n)
    with connect_robot(configs) as bot:
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        if not args.no_wrist:
            for n in WRIST_CAMS:
                if not getattr(bot.sensors, n).wait_for_active(timeout=3.0):
                    logger.warning("{} not publishing — captures go without it "
                                   "(it is started by dexsensor on the nano)", n)
        cg = cs.dcb.bev.cg
        logger.info("intrinsics from {}", "camera info" if cg.load_intrinsics(bot) else "FALLBACK")
        cs._preload_detectors()
        with SuctionMover(bot) as m:
            if m.software_estop_active() or not m.ensure_ready(release_estop=False):
                raise SystemExit("arm not ready (software E-Stop?)")
            both_arms_home(bot, left=m, right_first=True)
            cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
            cap = Capture(bot, m, out, args.frames, args.src, args.tgt)
            cap.save_camera_info()

            # outbound leg of each item = its FIRST _park_during_legs call, the
            # return leg = its SECOND (scan=False: no divert excursion in between)
            orig_park_during_legs = cs._park_during_legs
            state = {"calls": 2}

            def park_during_legs(label, arm_moves, chassis_legs):
                state["calls"] += 1
                if state["calls"] == 2:       # return leg, cup empty
                    _rise_to_transport(m)
                    cap(label, "placed")
                    orig_park_during_legs(label, [], chassis_legs)
                    cap(label, "return")
                    for fn in arm_moves:      # the normal view park, chassis still
                        fn()
                    cap(label, "return_parked")
                    return
                if state["calls"] > 2:
                    return orig_park_during_legs(label, arm_moves, chassis_legs)
                orig_park_during_legs(label, [lambda: _rise_to_transport(m)], chassis_legs)
                cap(label, "carry")
                for z in args.heights:
                    _, rpy = m.current_ee_pose()
                    if m.move_ee_vertical(z, rpy) is None:
                        logger.warning("[capture] {} hold z {:.2f} not reached — "
                                       "capturing where it stopped", label, z)
                    cap(label, f"carry_z{z:.2f}")
                for fn in arm_moves:          # the normal view park, chassis still
                    fn()
                cap(label, "parked")

            cs._park_during_legs = park_during_legs
            saved = cs._load_legs()
            leg = cs.ChassisNav(start_dist_m=cs._start_dist(saved, "main", cfg.BUFFER_STRAFE_DIST_M),
                                start_dx_m=cs._start_dx(saved, "main", cfg.BUFFER_STRAFE_DIST_M))
            divert_leg = cs.ChassisNav(cfg.TARGET_STRAFE_DIST_M, skip_first=False,
                                       start_dist_m=cs._start_dist(saved, "divert",
                                                                   cfg.TARGET_STRAFE_DIST_M),
                                       start_dx_m=cs._start_dx(saved, "divert",
                                                               cfg.TARGET_STRAFE_DIST_M))
            try:
                logger.info("position the chassis at the SOURCE station (case in front), then `d`")
                if not cs._manual_strafe(bot, "start"):
                    return
                cap("start", "start")
                for label in args.items:
                    state["calls"] = 0
                    ok = cs.run_item(bot, m, label, LABEL_KEY[label], args.src, args.tgt,
                                     scan=False, auto=True, leg=leg, divert_leg=divert_leg)
                    logger.info("[capture] item {} {}", label, "OK" if ok else "FAILED")
                    if not ok:
                        break
            finally:
                cs._park_during_legs = orig_park_during_legs
                safe_home(m)
                logger.info("[capture] frames in {}", out)


if __name__ == "__main__":
    main()
