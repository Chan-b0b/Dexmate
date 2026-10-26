"""Low-latency live view of the head camera and the right wrist camera.

Browser dashboard (MJPEG over plain HTTP, stdlib only). Runs as its OWN process
and subscribes to the camera streams only — it builds dexcontrol `Sensors`,
NOT a `Robot`, because Robot.__init__ applies the default head pose and must not
be created while another script is driving the robot.

Latency design: each camera has a grabber thread that polls the zenoh
subscriber's latest frame (the stream keeps only the newest; nothing queues),
and encodes JPEG ONLY when someone is watching and the frame is new. Each
browser connection sends only the newest encoded frame and skips any it missed,
so a slow client lags by frames, never by seconds. TCP_NODELAY on the socket.
Cost knobs: --width (downscale before JPEG), --quality.

    cd ~/LGES/Dexmate/LGES && ~/vla_venv/bin/python -m live_view
    open http://<robot-ip>:8765 in a browser on the same network
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from loguru import logger

CAMERAS = {  # url name -> (sensor attr, obs key or None for single-stream sensors)
    "head": ("head_camera", "left_rgb"),
    "right_hand": ("right_wrist_camera", None),
}

PAGE = """<!doctype html><meta charset=utf-8><title>robot cameras</title>
<style>body{margin:0;background:#111;color:#ccc;font:14px sans-serif}
.g{display:flex;flex-wrap:wrap;gap:6px;padding:6px}
.c{flex:1 1 480px;min-width:0}.c img{width:100%;display:block;background:#000}
.c div{padding:2px 0}</style>
<div class=g>
<div class=c><div>head</div><img src=/head.mjpg></div>
<div class=c><div>right hand</div><img src=/right_hand.mjpg></div>
</div>"""


class Feed:
    """Newest JPEG of one camera; encodes lazily while at least one client waits."""

    def __init__(self, name: str, sensor, key, width: int, quality: int) -> None:
        self.name, self.sensor, self.key = name, sensor, key
        self.width, self.quality = width, quality
        self.cond = threading.Condition()
        self.jpeg: "bytes | None" = None
        self.seq = 0
        self.viewers = 0
        threading.Thread(target=self._run, daemon=True, name=f"feed-{name}").start()

    def _latest(self):
        if self.key is None:
            obs = self.sensor.get_obs(include_timestamp=True)
        else:
            obs = self.sensor.get_obs(obs_keys=[self.key], include_timestamp=True).get(self.key)
        if isinstance(obs, dict):
            return obs.get("data"), obs.get("timestamp")
        return obs, None

    def _run(self) -> None:
        last_ts, last_id = None, None
        enc = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while True:
            if self.viewers == 0:
                time.sleep(0.1)
                continue
            try:
                rgb, ts = self._latest()
            except Exception as e:  # sensor hiccup: keep the feed alive
                logger.warning("[{}] read failed: {}", self.name, e)
                time.sleep(0.2)
                continue
            # new-frame test: timestamp when the transport has one, else array identity
            if rgb is None or (ts is not None and ts == last_ts) or (ts is None and id(rgb) == last_id):
                time.sleep(0.002)
                continue
            last_ts, last_id = ts, id(rgb)
            if self.width and rgb.shape[1] > self.width:
                h = round(rgb.shape[0] * self.width / rgb.shape[1])
                rgb = cv2.resize(rgb, (self.width, h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), enc)
            if ok:
                with self.cond:
                    self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                    self.cond.notify_all()

    def wait_newer(self, seq: int, timeout: float = 2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout)
            return self.jpeg, self.seq


def make_handler(feeds: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # no per-request spam
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            feed = feeds.get(path.removeprefix("/").removesuffix(".mjpg"))
            if feed is None:
                self.send_error(404)
                return
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            feed.viewers += 1
            seq = -1
            try:
                while True:
                    jpg, seq2 = feed.wait_newer(seq)
                    if jpg is None or seq2 == seq:
                        continue
                    seq = seq2   # frames produced while we were sending are skipped
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                feed.viewers -= 1
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--width", type=int, default=480,
                    help="downscale to this width before JPEG (0 = full size)")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100")
    args = ap.parse_args()

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.sensors.manager import Sensors
    configs = get_robot_config()
    for attr, _ in CAMERAS.values():
        configs.enable_sensor(attr)
        configs.sensors[attr].transport = "zenoh"
    sensors = Sensors(configs.sensors)

    feeds = {}
    for name, (attr, key) in CAMERAS.items():
        sensor = getattr(sensors, attr)
        # always_on: the default AUTO policy drops the subscription after 5 s
        # without a read, i.e. whenever nobody is watching
        streams = sensor._streams
        for k, s in streams.items():
            if s is not None and (key is None or k == key):
                s.set_subscription_policy("always_on")
        feeds[name] = Feed(name, sensor, key, args.width, args.quality)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(feeds))
    srv.daemon_threads = True
    logger.info("serving on http://0.0.0.0:{}  (width {}, quality {})",
                args.port, args.width, args.quality)
    try:
        srv.serve_forever()
    finally:
        sensors.shutdown() if hasattr(sensors, "shutdown") else None


if __name__ == "__main__":
    main()
