"""Live BEV case-detector viewer.

Warps the head frame to the metric top-down canvas every frame and runs the
YOLO-OBB there, drawing on the BEV image:
  - the detected case oriented box (green) + center,
  - the base_link pose readout (X, Y, yaw) + top-face z,
  - a 10 cm base-frame grid so the metric scale is visible.

Because the warp is rebuilt per frame from the live (q_torso, q_head), the box
stays put in BEV even if the head jitters — only the case moves. z is not
sensed here; the top-face plane is top_face_z(--layer).

Robot-side (needs the camera + trained cfg.OBB_MODEL_PATH). Keys: q/Esc quit, s snapshot.

    python live_bev.py                 # layers_remaining = 1
    python live_bev.py --layer 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))

import config as cfg
import bev
from detect_case_bev import detect_case_bev
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch


def _get_frame(robot):
    # rgb only — live_bev never uses depth; requesting it doubles transport cost.
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
    rgb = obs.get("left_rgb")
    rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
    return rgb


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


def draw(det, fps: float, grab_ms: float = 0.0, det_ms: float = 0.0) -> np.ndarray:
    """Overlay OBB + base pose on a BGR copy of the BEV image."""
    disp = cv2.cvtColor(det.bev, cv2.COLOR_RGB2BGR)
    _grid(disp)
    if det.found:
        (cx, cy) = det.bev_center_px
        box = cv2.boxPoints(((cx, cy), det.dims_px, det.base_yaw_deg)).astype(np.int32)
        cv2.drawContours(disp, [box], 0, (0, 255, 0), 2)
        cv2.circle(disp, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        X, Y = det.base_xy
        cv2.putText(disp, f"case conf={det.conf:.2f}  yaw={det.base_yaw_deg:.0f}deg",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, f"base X={X:+.3f} Y={Y:+.3f}  ztop={det.top_face_z:.3f} m",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(disp, "case: not found", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(disp, f"{fps:.1f} fps  grab {grab_ms:.0f}ms  det {det_ms:.0f}ms   q/Esc quit  s snap",
                (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return disp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1,
                    help="layers remaining (sets the warp plane top_face_z(layer))")
    ap.add_argument("--angle", type=float, default=15.0, help="head-down align angle")
    ap.add_argument("--render-every", type=int, default=3,
                    help="draw/imshow only every N frames (detection still runs every "
                         "frame); raise it if the display (esp. remote VNC/X11) is the bottleneck")
    args = ap.parse_args()

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    out = Path(__file__).resolve().parent / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        set_head_pitch(robot, angle=args.angle)

        win = "BEV case detector"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"Live BEV detector (layer={args.layer}) — q/Esc quit, s snapshot")
        fps, t, i = 0.0, time.time(), 0
        disp = None
        while True:
            t0 = time.time()
            rgb = _get_frame(robot)
            if rgb is None:
                continue
            t1 = time.time()
            det = detect_case_bev(rgb, *_joints(robot), layers_remaining=args.layer)
            t2 = time.time()

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t, 1e-6)
            t = now

            # Detection runs every frame; only render every N (imshow over remote
            # X11/VNC is the real cost). waitKey still runs each loop for keys.
            if i % args.render_every == 0:
                disp = draw(det, fps, (t1 - t0) * 1e3, (t2 - t1) * 1e3)
                cv2.imshow(win, disp)
            i += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                p = out / f"live_bev_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), disp)
                print("saved", p)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
