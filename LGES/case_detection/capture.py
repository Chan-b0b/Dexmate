"""Robot-side capture: grab RGB + depth frames of the case for the detection test.

Mirrors perception/perception.py's camera setup. Saves one .npz per frame
(rgb, depth, q_torso, q_head, timestamp) plus a .png of the RGB for quick
browsing. Aim the head at the case, then:

    python capture.py --n 20 --interval 0.3
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
from bin_roi import crop, find_bin, inset_bbox
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch


def _get_frame(robot):
    """One (rgb, depth) pair from the head ZED. Returns (rgb, depth) or (None, None)."""
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
    rgb = obs.get("left_rgb")
    depth = obs.get("depth")
    rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
    depth = depth.get("data") if isinstance(depth, dict) else depth
    return rgb, depth


def _joints(robot):
    return (np.asarray(robot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(robot.head.get_joint_pos(), dtype=np.float64))


def _save_frame(out, crops, idx, rgb, depth, q_torso, q_head, crop_bin, layer=None):
    """Write frame_<idx>.{npz,png} (+ bin crop). Returns the bin bbox or None.

    layer = layers remaining in the stack; stored so prewarp/BEV warps at the
    right plane (top face = FLOOR_Z_BASE_M + layer*LAYER_PITCH_M)."""
    bbox = find_bin(rgb)  # yellow-bin ROI (x, y, w, h) or None
    stem = out / f"frame_{idx:03d}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        rgb=rgb, depth=np.asarray(depth, np.float32),
        q_torso=q_torso, q_head=q_head, timestamp=time.time(),
        bin_bbox=np.array(bbox if bbox is not None else (-1, -1, -1, -1)),
        layers_remaining=np.int32(layer if layer is not None else -1),
    )
    # Stream is RGB (base_camera bgr=False); cv2 writes BGR -> convert.
    cv2.imwrite(str(stem.with_suffix(".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if crop_bin and bbox is not None:
        bgr = cv2.cvtColor(crop(rgb, inset_bbox(bbox)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(crops / f"frame_{idx:03d}.png"), bgr)
    return bbox


def _timed_loop(robot, out, crops, args) -> int:
    """Capture args.n frames, one every args.interval seconds."""
    saved = 0
    for i in range(args.n):
        rgb, depth = _get_frame(robot)
        if rgb is None or depth is None:
            print(f"[{i}] no frame, retrying")
            time.sleep(args.interval)
            continue
        bbox = _save_frame(out, crops, saved, rgb, depth, *_joints(robot), args.crop_bin, args.layer)
        saved += 1
        print(f"[{saved}/{args.n}] saved frame_{saved-1:03d}  bin={'ok' if bbox is not None else 'MISSING'}"
              f"  depth {np.nanmin(depth):.2f}..{np.nanmax(depth):.2f} m")
        time.sleep(args.interval)
    return saved


def _keyboard_loop(robot, out, crops, args) -> int:
    """Live preview window; SPACE saves the current frame, q/Esc quits.

    The bin box shows the trained detector when cfg.USE_BIN_MODEL (the same one
    used at runtime), falling back to HSV find_bin if the model is unavailable.
    """
    win = "capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    use_model = cfg.USE_BIN_MODEL
    src = "model" if use_model else "HSV"
    print(f"Preview window focused: SPACE = save, q/Esc = quit   (bin box: {src})")
    saved = 0
    while saved < args.n:
        rgb, depth = _get_frame(robot)
        if rgb is None or depth is None:
            continue
        if use_model:
            try:
                from detect_bin import find_bin_model
                bbox = find_bin_model(rgb)
            except Exception as e:  # missing weights / ultralytics -> HSV
                print(f"bin model unavailable ({e}); using HSV")
                use_model, src = False, "HSV"
                bbox = find_bin(rgb)
        else:
            bbox = find_bin(rgb)

        disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(disp, f"SPACE=save  q=quit   saved {saved}/{args.n}  "
                    f"bin[{src}]={'ok' if bbox is not None else 'MISSING'}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            _save_frame(out, crops, saved, rgb, depth, *_joints(robot), args.crop_bin, args.layer)
            saved += 1
            print(f"[{saved}/{args.n}] saved frame_{saved-1:03d}  "
                  f"bin[{src}]={'ok' if bbox is not None else 'MISSING'}")
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="frames to capture (max, in --keyboard)")
    ap.add_argument("--interval", type=float, default=0.3,
                    help="seconds between frames (ignored with --keyboard)")
    ap.add_argument("--target", choices=["case", "bin"], default="case",
                    help="which detector this data is for -> data/<target>/<timestamp>/")
    ap.add_argument("--out", default=cfg.DATA_DIR, help="data root (default 'data')")
    ap.add_argument("--no-align", action="store_true", help="skip head alignment")
    ap.add_argument("--crop-bin", action="store_true",
                    help="also save bin-ROI RGB crops (for labeling / training)")
    ap.add_argument("--keyboard", action="store_true",
                    help="capture on SPACE keypress in a preview window, not by time")
    ap.add_argument("--layer", type=int, default=None,
                    help="layers remaining in the stack for this sequence (top case "
                         "height = FLOOR_Z_BASE_M + layer*LAYER_PITCH_M); recorded per "
                         "frame so BEV warps at the right plane")
    args = ap.parse_args()

    # data/<target>/<timestamp>[_L<layer>]/ so bin vs case captures stay separate,
    # repeated runs don't clash, and each sequence's stack layer is obvious.
    tag = time.strftime("%Y%m%d_%H%M%S") + (f"_L{args.layer}" if args.layer is not None else "")
    out = (Path(__file__).resolve().parent / args.out / args.target / tag)
    out.mkdir(parents=True, exist_ok=True)
    crops = out / "crops"
    if args.crop_bin:
        crops.mkdir(exist_ok=True)

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        if not args.no_align:
            set_head_pitch(robot, angle=30.0)

        loop = _keyboard_loop if args.keyboard else _timed_loop
        saved = loop(robot, out, crops, args)

    print(f"\nDone. {saved} frames in {out}"
          + (f"  (+ crops in {crops})" if args.crop_bin else ""))


if __name__ == "__main__":
    main()

#ssh dexmate-nano@192.168.50.22
# dexsensor launch -s head_camera --config /home/dexmate-nano/.dexmate/sensors/depth.toml