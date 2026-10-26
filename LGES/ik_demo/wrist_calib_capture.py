"""Wrist-camera calibration capture: the right arm visits views of a still AprilTag.

Put the printed tag (36h11 id 0, bin_pose/apriltag_36h11_id0_150mm_A4.pdf, taped
flat to a board) on the battery tops in the bin in front of the robot, and park
the chassis at that station as for a place (bin centre ~0.72 m ahead). Only the
RIGHT arm moves (40% speed); the left arm stays home, the chassis stays still.

For each pose of --poses (bin_pose/calib_poses.py; "via_home" legs go through
home where the direct joint move was not clear), first re-checked here
against the LIVE left arm: every joint-space step self-collision free and every
right-arm point + finger tip above the bins over the table. Then: move, settle,
save FRAMES frames of the right wrist camera + head camera (RGB + depth) +
joints, and log whether the tag was seen.

Run from LGES/ (E-stop in reach):
    ~/vla_venv/bin/python -m ik_demo.wrist_calib_capture
then solve:
    cd bin_pose && ~/vla_venv/bin/python solve_wrist_calib.py ../ik_demo/run_logs/wrist_calib/<stamp> --tag-mm <measured>

Writes RUN_LOG_DIR/wrist_calib/<stamp>/<HHMMSS>_view<i>_<k>.npz with
    right_wrist_camera_rgb  head_rgb  head_depth  q_right_arm  q_left_arm  q_torso  q_head
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg

HOME = tuple(cfg.HOME_JOINTS_RIGHT)
# Pose list from bin_pose/calib_poses.py (--poses). The default is the 1015 second
# plan: 10 natural views (every joint within 1 rad of home, image roll within
# +-45 deg) of the tag where it lay at 16:23 — ~(0.79, +0.03, 0.77) in base, bin
# at (0.81, +0.06) — with bin-wall occlusion checked. They only see the tag if the
# robot and the tag are where they were then; otherwise re-plan with calib_poses.py.
DEFAULT_POSES = Path(__file__).resolve().parents[1] / "bin_pose" / "calib_poses_1015b.npy"
VEL_SCALE = 0.4
SETTLE_S = 1.0
FRAMES = 2
FINGER_LEN_M = 0.18          # R_gripper_base -> Robotiq finger tips (conservative)
TABLE_X0_M = 0.33            # base x where the table + bins begin ...
ABOVE_BINS_Z = 0.95          # ... above which every right-arm point stays (rim 0.89)


def _leg_ok(arm, q0, q1) -> bool:
    import pinocchio as pin
    fids = [arm._model.getFrameId(n) for n in
            [f"R_arm_l{i}" for i in range(1, 8)] + [cfg.GRIPPER_EE_FRAME]]
    for t in np.linspace(0.0, 1.0, 21):
        q = q0 + t * (q1 - q0)
        if arm.in_collision(q):
            logger.warning("[calib] leg: self-collision at {:.0%}", t)
            return False
        conf = arm._configuration(q)
        pin.framesForwardKinematics(arm._model, arm._data, conf.q)
        pts = [arm._data.oMf[f].translation.copy() for f in fids]
        T = arm._data.oMf[arm._ee_frame_id]
        pts.append(T.translation + T.rotation[:, 2] * FINGER_LEN_M)
        P = np.concatenate([np.linspace(a, b, 6) for a, b in zip(pts[:-1], pts[1:])])
        low = (P[:, 0] > TABLE_X0_M) & (P[:, 2] < ABOVE_BINS_Z)
        if low.any():
            logger.warning("[calib] leg: below z {:.2f} over the table at {:.0%} (lowest {:.3f})",
                           ABOVE_BINS_Z, t, P[low, 2].min())
            return False
    return True


def _tag_seen(rgb) -> bool:
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return ids is not None and 0 in ids.ravel()


def _load_views(path) -> list:
    d = np.load(path, allow_pickle=True).item()
    return [(k, tuple(q)) for k, q in zip(d["kinds"], d["seq"])]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default=str(DEFAULT_POSES), help="calib_poses.py output (.npy)")
    VIEWS = _load_views(ap.parse_args().poses)
    # dexcontrol's import runs logger.configure() — before setup_logging (see held_view_capture)
    from dexcontrol.core.config import get_robot_config
    from . import chassis_sequence as cs
    from .arm import ArmMover, connect_robot
    from .go_home import both_arms_home

    cs.setup_logging()
    out = Path(cfg.RUN_LOG_DIR) / "wrist_calib" / cs.run_stamp()
    out.mkdir(parents=True, exist_ok=True)
    logger.warning("WRIST CAMERA CALIBRATION — moves the RIGHT arm through {} views (left arm "
                   "home, chassis still). Tag on the battery tops in the bin ahead. E-stop in reach.",
                   sum(k == "view" for k, _ in VIEWS))
    if input("Connect + safe-home both arms? [y/N]: ").strip().lower() != "y":
        return
    configs = get_robot_config()
    for n in ("head_camera", "right_wrist_camera"):
        configs.enable_sensor(n)
        configs.sensors[n].transport = "zenoh"
    with connect_robot(configs) as bot:
        for part in (bot.left_arm, bot.right_arm, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        bot.sensors.head_camera.wait_for_active(timeout=5.0)
        if not bot.sensors.right_wrist_camera.wait_for_active(timeout=5.0):
            raise SystemExit("right wrist camera not publishing (dexsensor on the nano)")
        both_arms_home(bot)
        cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
        arm = ArmMover(robot=bot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)  # model sees the live left arm

        def wrist_rgb():
            o = bot.sensors.right_wrist_camera.get_obs(include_timestamp=True)
            return o.get("data") if isinstance(o, dict) else o

        if not _tag_seen(wrist_rgb()):
            logger.warning("[calib] the tag is NOT visible from home — check it lies in the bin ahead")
        if input("Tag placed, robot at the bin station. Start the views? [y/N]: ").strip().lower() != "y":
            return
        seen = 0
        try:
            vi = 0
            for kind, q in VIEWS:
                q1 = np.asarray(q, dtype=float)
                q0 = np.asarray(bot.right_arm.get_joint_pos(), dtype=float)
                if not _leg_ok(arm, q0, q1):
                    logger.error("[calib] leg to {} {} not clear — stopping here", kind, vi)
                    break
                arm.move_joints(q1, vel_scale=VEL_SCALE)
                if kind != "view":
                    continue
                time.sleep(SETTLE_S)
                for k in range(FRAMES):
                    w = wrist_rgb()
                    rgb = cs._head_rgb(bot)
                    q_torso, q_head = cs._joints(bot)
                    np.savez_compressed(
                        out / f"{time.strftime('%H%M%S')}_view{vi}_{k}.npz",
                        right_wrist_camera_rgb=w, head_rgb=rgb,
                        head_depth=bot.sensors.head_camera.get_depth(),
                        q_right_arm=np.asarray(bot.right_arm.get_joint_pos()),
                        q_left_arm=np.asarray(bot.left_arm.get_joint_pos()),
                        q_torso=q_torso, q_head=q_head)
                ok = _tag_seen(w)
                seen += ok
                logger.info("[calib] view {}: saved {} frame(s), tag {}", vi, FRAMES,
                            "SEEN" if ok else "not seen")
                vi += 1
        finally:
            q0 = np.asarray(bot.right_arm.get_joint_pos(), dtype=float)
            if _leg_ok(arm, q0, np.asarray(HOME)):
                arm.move_joints(np.asarray(HOME), vel_scale=VEL_SCALE)
            else:
                logger.error("[calib] leg back home not clear — right arm left where it is")
            logger.info("[calib] tag seen in {} view(s); frames in {}", seen, out)


if __name__ == "__main__":
    main()
