"""Record a head-camera sequence at camera pace for testing the tracker
offline (live_detect_vlm.py --replay).

capture_bev.py saves frames too, but one every 0.3 s at best (it warps and
compresses each one inline) and it does not enable the head; the recorded
sequences of 2026-09-06 came out at 0.7 fps — 1.3 s of motion between frames,
useless for judging a tracker that will see ~5 fps. This keeps every frame in
memory (~1.7 MB each) and writes them all at the end, so the rate is the one
asked for. Same per-frame format as capture_bev (frame_<idx>.npz with rgb,
q_torso, q_head, timestamp), so the other tools read it as well.

    python record_track.py --seconds 30                  # 5 fps, data/track_seq/<timestamp>/
    python record_track.py --seconds 60 --fps 8 --serve 8088   # with a browser preview

Move things on the table while it records: slide objects, lift one and put it
down elsewhere, take one away, add one, reach across with a hand. Then:

    python live_detect_vlm.py --replay data/track_seq/<timestamp> --serve 8088
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import live_detect_bev as ld          # HTTP preview; puts ../perception on sys.path
import config as cfg
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--angle", type=float, default=30.0, help="head-down align angle")
    ap.add_argument("--out", default=str(_HERE / cfg.DATA_DIR / "track_seq"))
    ap.add_argument("--serve", type=int, metavar="PORT", default=None,
                    help="live preview over HTTP while recording")
    args = ap.parse_args()

    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        # Same head handling as live_detect_vlm: the head is disabled whenever
        # no client is connected and the enable takes ~1 s to bite.
        target = np.deg2rad(np.rad2deg(float(robot.torso.pitch_angle)) - args.angle)
        for _ in range(3):
            robot.head.set_mode("enable")
            time.sleep(1.0)
            set_head_pitch(robot, angle=args.angle)
            pitch = float(np.asarray(robot.head.get_state()["pos"], float)[0])
            if abs(pitch - target) < np.deg2rad(3.0):
                break
        else:
            print("WARNING: head did not reach the target pitch")

        ctl = None
        if args.serve:
            ctl = ld._Control()
            ld.serve(args.serve, ctl)
            print(f"preview on http://0.0.0.0:{args.serve}")

        def show(bgr, text):
            if ctl is None:
                return
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                ctl.publish(jpg.tobytes())

        for k in range(3, 0, -1):
            rgb = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
            rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
            if rgb is not None:
                show(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"recording in {k}...")
            print(f"recording in {k}...")
            time.sleep(1.0)

        frames = []
        period = 1.0 / args.fps
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < args.seconds:
            if time.time() < next_t:
                time.sleep(0.005)
                continue
            next_t += period
            rgb = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
            rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
            if rgb is None:
                continue
            q_torso, q_head = ld._joints(robot)
            frames.append((rgb.copy(), np.asarray(q_torso, np.float64),
                           np.asarray(q_head, np.float64), time.time()))
            if len(frames) % int(args.fps) == 0:
                el = time.time() - t0
                show(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                     f"REC {el:.0f}/{args.seconds:.0f}s  {len(frames)} frames")
                print(f"\r{el:5.1f}s  {len(frames)} frames", end="", flush=True)
        print()

    out.mkdir(parents=True, exist_ok=True)
    for i, (rgb, qt, qh, ts) in enumerate(frames):
        np.savez_compressed(out / f"frame_{i:03d}.npz", rgb=rgb, q_torso=qt, q_head=qh,
                            timestamp=ts)
    dur = frames[-1][3] - frames[0][3] if len(frames) > 1 else 0.0
    print(f"{len(frames)} frames over {dur:.1f}s ({(len(frames) - 1) / max(dur, 1e-9):.1f} fps) "
          f"-> {out}")
    print(f"replay:  python live_detect_vlm.py --replay {out} --serve 8088")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
