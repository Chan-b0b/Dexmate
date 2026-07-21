"""Live case-detector viewer.

Runs the full pipeline on the head camera and draws, per frame:
  - the bin ROI (orange) the case model is cropped to,
  - the detected case oriented box (green) + center,
  - the pose readout (score, yaw, camera-frame xyz).

Robot-side (needs the camera + the trained models). Keys: q/Esc quit, s snapshot.

    python live.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))

import config as cfg
from detect_case_obb import bin_roi_for, detect_case
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch


def _get_frame(robot):
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
    rgb, depth = obs.get("left_rgb"), obs.get("depth")
    rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
    depth = depth.get("data") if isinstance(depth, dict) else depth
    return rgb, depth


def draw(rgb, det, roi, fps) -> np.ndarray:
    """Overlay ROI + case OBB + pose on a BGR copy of the frame."""
    disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 165, 255), 2)  # bin ROI
    if det.found:
        (cx, cy), (lo, sh) = det.center_px, det.dims_px
        box = cv2.boxPoints(((cx, cy), (lo, sh), det.angle_deg)).astype(np.int32)
        cv2.drawContours(disp, [box], 0, (0, 255, 0), 2)               # case OBB
        cv2.circle(disp, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        x_, y_, z_ = det.center_cam_m
        cv2.putText(disp, f"case score={det.size_score:.2f}  yaw={det.angle_deg:.0f}deg",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, f"cam xyz=({x_:+.3f}, {y_:+.3f}, {z_:.3f}) m",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        msg = "case: not found" + ("" if roi is not None else "  (no bin ROI)")
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(disp, f"{fps:.1f} fps   q/Esc quit  s snapshot",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return disp


def main() -> None:
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    out = Path(__file__).resolve().parent / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        set_head_pitch(robot, angle=30.0)

        win = "case detector"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("Live case detector — q/Esc quit, s snapshot")
        fps, t = 0.0, time.time()
        while True:
            rgb, depth = _get_frame(robot)
            if rgb is None or depth is None:
                continue
            det = detect_case(depth, rgb)
            roi = bin_roi_for(rgb)             # for drawing the crop region
            disp = draw(rgb, det, roi, fps)
            cv2.imshow(win, disp)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t, 1e-6)
            t = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                p = out / f"live_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), disp)
                print("saved", p)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
