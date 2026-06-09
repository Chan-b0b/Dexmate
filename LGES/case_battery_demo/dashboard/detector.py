"""Bin-detection overlay for the dashboard — a separate, isolated process.

Reads the RGB frame the publisher already spools (``frame.jpg``), runs the
trained YOLO bin detector on it, draws the highest-confidence bin box, and
writes an annotated ``detect.jpg`` (+ ``detect.json`` metadata) back to the
spool. The dashboard server serves those and the page shows the overlay under
the depth image.

Why a separate process (not part of the demo):
  * The demo process drives the real-time arm control loop. torch/ultralytics
    is a heavy import and uses the GPU; keeping it out of that process means
    detection cannot jitter the motion. Inference itself is only a few ms.
  * Reading the spooled frame (rather than its own camera subscriber) keeps it
    fully decoupled and lets it run unchanged over a recorded session.

    python -m case_battery_demo.dashboard.detector                 # live spool
    python -m case_battery_demo.dashboard.detector --spool DIR --weights X.pt
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

# Default to the trained bin detector shipped in case_detection/.
_LGES_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(
    _LGES_DIR, "case_detection", "runs", "detect", "bin", "weights", "bin_detector.pt"
)
DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class BinDetector:
    """Polls the spooled frame, runs the bin detector, writes the overlay."""

    def __init__(
        self,
        spool_dir: str = DEFAULT_SPOOL_DIR,
        weights: str = DEFAULT_WEIGHTS,
        conf: float = 0.40,
        hz: float = 15.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.spool_dir = spool_dir
        self._weights = weights
        self._conf = float(conf)
        self._period = 1.0 / max(hz, 1.0)
        self._jpeg_quality = int(jpeg_quality)
        self._frame_path = os.path.join(spool_dir, "frame.jpg")
        self._detect_path = os.path.join(spool_dir, "detect.jpg")
        self._detect_json = os.path.join(spool_dir, "detect.json")
        self._model = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._last_mtime = 0.0
        os.makedirs(spool_dir, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def _load_model(self):
        from ultralytics import YOLO  # heavy import, kept in this process only
        if not os.path.exists(self._weights):
            raise FileNotFoundError(f"bin weights not found: {self._weights}")
        print(f"[detector] loading {self._weights} …")
        self._model = YOLO(self._weights)
        print("[detector] model ready")

    def start(self) -> "BinDetector":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="bin-detector", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "BinDetector":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._load_model()
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

        # frame.jpg is BGR (publisher encoded a BGR array); YOLO wants BGR too.
        res = self._model.predict(bgr, conf=self._conf, verbose=False)[0]
        found, box, conf = self._best_box(res)

        disp = bgr.copy()
        if found:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(disp, (x, y), (x + w, y + h), _GREEN, 2)
            cv2.putText(disp, f"bin {conf:.2f}", (x, max(14, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2)
        else:
            cv2.putText(disp, "bin: not found", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2)

        ok, out = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        self._seq += 1
        _atomic_write(self._detect_path, out.tobytes())
        _atomic_write(self._detect_json, json.dumps({
            "seq": self._seq,
            "stamp": time.time(),
            "found": found,
            "conf": None if conf is None else round(float(conf), 3),
            "box": None if box is None else [int(v) for v in box],
        }).encode("utf-8"))

    @staticmethod
    def _best_box(res):
        """Return (found, (x,y,w,h) | None, conf | None) for the top box."""
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return False, None, None
        confs = boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[i]
        return True, (x1, y1, x2 - x1, y2 - y1), float(confs[i])


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard bin-detection overlay")
    parser.add_argument("--spool", default=DEFAULT_SPOOL_DIR,
                        help="spool dir holding frame.jpg (where detect.jpg is written)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO bin detector weights")
    parser.add_argument("--conf", type=float, default=0.40, help="detection confidence threshold")
    parser.add_argument("--hz", type=float, default=15.0, help="max detection rate")
    args = parser.parse_args()

    det = BinDetector(spool_dir=os.path.abspath(args.spool), weights=args.weights,
                      conf=args.conf, hz=args.hz)
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
