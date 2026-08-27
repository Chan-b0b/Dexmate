"""Robot-side BEV capture: save metric top-down (bird-eye-view) frames for
labeling / training the BEV YOLO-OBB case detector.

Like capture.py, but every frame is also warped to the metric BEV canvas
(bev.build_mapper on the plane at top_face_z(--layer)) — the exact image
detect_case_bev sees at runtime, so train/run framing is identical. Saves:

    frame_<idx>.npz       raw rgb + q_torso/q_head + layers_remaining
                          (re-warpable at any plane; feeds detect_case_bev's self-test)
    bev/frame_<idx>.png   the metric BEV image  (label / train on these)

Depth is not grabbed: the BEV pipeline never uses it and requesting it doubles
transport cost (same reasoning as live_bev.py). Runs land in
data/<target>_bev/<timestamp>_L<layer>/ (--target case|bin, like capture.py).
Aim the head at the target, then:

    python capture_bev.py --layer 3 --n 20 --interval 0.3
    python capture_bev.py --layer 3 --keyboard     # SPACE saves, q/Esc quits
    python capture_bev.py --target bin --layer 0   # bin-detector data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the proven head-aim helper from the perception package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))

import config as cfg
import bev
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch


def _get_frame(robot):
    """One RGB frame from the head ZED, or None."""
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
    rgb = obs.get("left_rgb")
    return rgb.get("data") if isinstance(rgb, dict) else rgb


def _joints(robot):
    return (np.asarray(robot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(robot.head.get_joint_pos(), dtype=np.float64))


def _grid(disp: np.ndarray) -> None:
    """Draw a 10 cm base-frame grid on the BEV image (dark green)."""
    x0, x1 = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    s = cfg.BEV_PX_PER_M
    for X in np.arange(np.ceil(x0 * 10) / 10, x1, 0.1):
        u = int((X - x0) * s)
        cv2.line(disp, (u, 0), (u, disp.shape[0]), (0, 90, 0), 1)
    for Y in np.arange(np.ceil(y0 * 10) / 10, y1, 0.1):
        v = int((Y - y0) * s)
        cv2.line(disp, (0, v), (disp.shape[1], v), (0, 90, 0), 1)


def _save_frame(out, bev_dir, idx, rgb, bev_img, q_torso, q_head, layer) -> None:
    """Write frame_<idx>.npz (raw, re-warpable) + bev/frame_<idx>.png (training image)."""
    np.savez_compressed(
        out / f"frame_{idx:03d}.npz",
        rgb=rgb, q_torso=q_torso, q_head=q_head, timestamp=time.time(),
        layers_remaining=np.int32(layer),
    )
    # Stream is RGB (base_camera bgr=False); cv2 writes BGR -> convert.
    cv2.imwrite(str(bev_dir / f"frame_{idx:03d}.png"),
                cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))


def _timed_loop(robot, out, bev_dir, args) -> int:
    """Capture args.n frames, one every args.interval seconds."""
    plane_z = bev.top_face_z(args.layer)
    saved = 0
    for i in range(args.n):
        rgb = _get_frame(robot)
        if rgb is None:
            print(f"[{i}] no frame, retrying")
            time.sleep(args.interval)
            continue
        q_torso, q_head = _joints(robot)
        bev_img = bev.build_mapper(q_torso, q_head, plane_z).warp(rgb)
        _save_frame(out, bev_dir, saved, rgb, bev_img, q_torso, q_head, args.layer)
        saved += 1
        print(f"[{saved}/{args.n}] saved frame_{saved-1:03d}  "
              f"bev {bev_img.shape[1]}x{bev_img.shape[0]} px  plane_z={plane_z:.4f}")
        time.sleep(args.interval)
    return saved


def _keyboard_loop(robot, out, bev_dir, args) -> int:
    """Live BEV preview window; SPACE saves the current frame, q/Esc quits."""
    plane_z = bev.top_face_z(args.layer)
    win = "capture_bev"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("BEV preview focused: SPACE = save, q/Esc = quit")
    saved = 0
    while saved < args.n:
        rgb = _get_frame(robot)
        if rgb is None:
            continue
        q_torso, q_head = _joints(robot)
        bev_img = bev.build_mapper(q_torso, q_head, plane_z).warp(rgb)

        disp = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR)
        _grid(disp)
        cv2.putText(disp, f"SPACE=save  q=quit   saved {saved}/{args.n}  "
                    f"L{args.layer} plane_z={plane_z:.3f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            _save_frame(out, bev_dir, saved, rgb, bev_img, q_torso, q_head, args.layer)
            saved += 1
            print(f"[{saved}/{args.n}] saved frame_{saved-1:03d}")
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="frames to capture (max, in --keyboard)")
    ap.add_argument("--interval", type=float, default=0.3,
                    help="seconds between frames (ignored with --keyboard)")
    ap.add_argument("--layer", type=int, default=1,
                    help="layers remaining in the stack (sets the warp plane "
                         "top_face_z(layer) = FLOOR_Z_BASE_M + layer*LAYER_PITCH_M)")
    ap.add_argument("--target", choices=["case", "bin"], default="case",
                    help="which detector this data is for -> data/<target>_bev/<timestamp>/")
    ap.add_argument("--out", default=cfg.DATA_DIR, help="data root (default 'data')")
    ap.add_argument("--angle", type=float, default=24.0, help="head-down align angle")
    ap.add_argument("--no-align", action="store_true", help="skip head alignment")
    ap.add_argument("--keyboard", action="store_true",
                    help="capture on SPACE keypress in a BEV preview window, not by time")
    args = ap.parse_args()

    # data/<target>_bev/<timestamp>_L<layer>/ so bin vs case captures stay
    # separate (mirrors capture.py), BEV runs don't mix with raw capture.py
    # runs, and each sequence's stack layer is obvious.
    tag = time.strftime("%Y%m%d_%H%M%S") + f"_L{args.layer}"
    out = Path(__file__).resolve().parent / args.out / f"{args.target}_bev" / tag
    out.mkdir(parents=True, exist_ok=True)
    bev_dir = out / "bev"
    bev_dir.mkdir(exist_ok=True)

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        if not args.no_align:
            set_head_pitch(robot, angle=args.angle)

        loop = _keyboard_loop if args.keyboard else _timed_loop
        saved = loop(robot, out, bev_dir, args)

    print(f"\nDone. {saved} frames in {out}  (BEV images in {bev_dir})")


if __name__ == "__main__":
    main()
