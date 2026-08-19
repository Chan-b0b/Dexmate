"""Case-detection (BEV) overlay for the dashboard — a separate, isolated process.

Reads the RGB frame the publisher already spools (``frame.jpg``) and the
joints from ``state.json``, warps the frame to the metric top-down (BEV)
plane and runs the trained YOLO-OBB case detector there
(case_detection/detect_case_bev.py — same pipeline as case_detection/live_bev.py),
draws the case oriented box + base pose on the BEV image, and writes it back
to the spool as ``detect.jpg`` (+ ``detect.json`` metadata). The dashboard
server serves those and the page shows the overlay under the depth image.

Why a separate process (not part of the demo):
  * The demo process drives the real-time arm control loop. torch/ultralytics
    is a heavy import and uses the GPU; keeping it out of that process means
    detection cannot jitter the motion. Inference itself is only a few ms.
  * Reading the spooled frame (rather than its own camera subscriber) keeps it
    fully decoupled and lets it run unchanged over a recorded session.

    python -m case_battery_demo.dashboard.detector                 # live spool
    python -m case_battery_demo.dashboard.detector --spool DIR --layer 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

# detect_case_bev.py (case_detection/) does bare `import config`/`import bev`,
# so it must be imported with that package dir on sys.path — same as
# case_detection/live_bev.py does for itself.
_LGES_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CASE_DETECTION_DIR = os.path.join(_LGES_DIR, "case_detection")
sys.path.insert(0, _CASE_DETECTION_DIR)
from detect_case_bev import detect_case_bev, load_model  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(_CASE_DETECTION_DIR, "runs", "obb", "case", "weights", "best.pt")
DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"
DEFAULT_LAYER = 5

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _grid(disp: np.ndarray, cfg) -> None:
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


class CaseDetector:
    """Polls the spooled frame, runs the BEV case detector, writes the overlay."""

    def __init__(
        self,
        spool_dir: str = DEFAULT_SPOOL_DIR,
        weights: str = DEFAULT_WEIGHTS,
        layer: int = DEFAULT_LAYER,
        hz: float = 15.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.spool_dir = spool_dir
        self._weights = weights
        self._layer = int(layer)
        self._period = 1.0 / max(hz, 1.0)
        self._jpeg_quality = int(jpeg_quality)
        self._frame_path = os.path.join(spool_dir, "frame.jpg")
        self._state_path = os.path.join(spool_dir, "state.json")
        self._detect_path = os.path.join(spool_dir, "detect.jpg")
        self._detect_json = os.path.join(spool_dir, "detect.json")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._last_mtime = 0.0
        os.makedirs(spool_dir, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "CaseDetector":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="case-detector", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "CaseDetector":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        try:
            load_model(self._weights)
            print("[detector] model ready")
        except Exception as e:  # noqa: BLE001
            print(f"[detector] cannot start: {e}")
            return
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - never die on a bad frame
                print(f"[detector] tick error: {e}")
            self._stop.wait(max(0.0, self._period - (time.time() - t0)))

    def _tick(self) -> None:
        # Only run when a new frame has been spooled.
        try:
            mtime = os.path.getmtime(self._frame_path)
        except OSError:
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime

        with open(self._frame_path, "rb") as f:
            buf = f.read()
        bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return

        joints = self._read_joints()
        if joints is None:
            return
        q_torso, q_head = joints

        import config as cfg  # case_detection/config.py, on sys.path since module import
        # frame.jpg is downscaled by ik_demo.dashboard_publish for browser bandwidth;
        # the BEV homography's intrinsics (camera_geometry.ZED_*) are calibrated at
        # cfg.IMG_W x cfg.IMG_H (the native stream size), so resize back up before
        # warping or the projected geometry samples outside the smaller array.
        h, w = bgr.shape[:2]
        if (w, h) != (cfg.IMG_W, cfg.IMG_H):
            bgr = cv2.resize(bgr, (cfg.IMG_W, cfg.IMG_H))

        # frame.jpg is BGR (publisher encoded a BGR array); detect_case_bev wants RGB.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = detect_case_bev(rgb, q_torso, q_head, layers_remaining=self._layer,
                              weights=self._weights)

        disp = cv2.cvtColor(det.bev, cv2.COLOR_RGB2BGR)
        _grid(disp, cfg)
        if det.found:
            (cx, cy) = det.bev_center_px
            box = cv2.boxPoints(((cx, cy), det.dims_px, det.base_yaw_deg)).astype(np.int32)
            cv2.drawContours(disp, [box], 0, _GREEN, 2)
            cv2.circle(disp, (int(cx), int(cy)), 4, _GREEN, -1)
            cv2.putText(disp, f"case conf={det.conf:.2f}  yaw={det.base_yaw_deg:.0f}deg",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2)
        else:
            cv2.putText(disp, "case: not found", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 2)

        ok, out = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        self._seq += 1
        _atomic_write(self._detect_path, out.tobytes())
        _atomic_write(self._detect_json, json.dumps({
            "seq": self._seq,
            "stamp": time.time(),
            "found": det.found,
            "conf": round(float(det.conf), 3) if det.found else None,
            "base_x_m": round(float(det.base_xy[0]), 4) if det.found else None,
            "base_y_m": round(float(det.base_xy[1]), 4) if det.found else None,
            "base_yaw_deg": round(float(det.base_yaw_deg), 1) if det.found else None,
            "top_face_z_m": round(float(det.top_face_z), 4),
        }).encode("utf-8"))

    def _read_joints(self):
        """(q_torso, q_head), joint-name-ordered, from the spooled state.json, or None."""
        try:
            with open(self._state_path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return None
        j = st.get("joints", {})
        torso, head = j.get("torso"), j.get("head")
        if not torso or not head:
            return None
        return [torso[k] for k in sorted(torso)], [head[k] for k in sorted(head)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard case-detection (BEV) overlay")
    parser.add_argument("--spool", default=DEFAULT_SPOOL_DIR,
                        help="spool dir holding frame.jpg (where detect.jpg is written)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO-OBB case detector weights")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="layers remaining in the stack (sets the BEV warp plane)")
    parser.add_argument("--hz", type=float, default=15.0, help="max detection rate")
    args = parser.parse_args()

    det = CaseDetector(spool_dir=os.path.abspath(args.spool), weights=args.weights,
                       layer=args.layer, hz=args.hz)
    print(f"[detector] watching {det._frame_path}  (Ctrl-C to stop)")
    det.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[detector] stopping")
    finally:
        det.stop()


if __name__ == "__main__":
    main()
