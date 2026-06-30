"""Cognex barcode-reader image feed for the dashboard — a separate process.

Pulls the reader's last captured frame over DMCC (``||>IMAGE.SEND``) and writes
it to the spool as ``barcode.jpg`` (+ ``barcode.json`` metadata). The dashboard
server serves those and the page shows the panel under the head-camera feed.

Image-only by design: it NEVER sends the ``T`` trigger, so it cannot fight the
demo's own scanning. ``IMAGE.SEND`` returns whatever the reader last captured,
so the panel refreshes whenever a read is triggered (by the demo or operator);
otherwise it keeps showing the last frame.

A fresh, short-lived connection per tick frees the reader immediately (it allows
only a few concurrent clients), and the low 1 Hz default keeps telnet traffic
light (~30 KB/frame).

    python -m case_battery_demo.dashboard.barcode                  # live spool
    python -m case_battery_demo.dashboard.barcode --host 192.168.50.101 --hz 1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time

from .. import config as cfg

DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"

# If the demo's scan lock was touched within this window, the image feed yields
# the reader. Must exceed the scanner's per-trigger period (~BCR_SCAN_TIMEOUT_S
# + 0.05s) so a still-scanning demo never looks idle between triggers.
SCAN_LOCK_FRESH_S = 2.5


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def pull_image(host: str, port: int, timeout: float = 4.0) -> bytes | None:
    """Fetch the reader's last image via DMCC ``IMAGE.SEND``.

    Response framing (confirmed against a DataMan): a successful reply is an
    ASCII byte-count + ``\\r\\n`` followed by exactly that many JPEG bytes, e.g.
    ``29085\\r\\n<...jpeg...>``. When no image is buffered the reader instead
    returns an STX..ETX error frame (e.g. ``\\x02NG,\\x03``). Returns the JPEG
    bytes on success, else ``None``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(b"||>IMAGE.SEND\r\n")

        first = s.recv(1)
        if not first:
            return None
        if first == b"\x02":  # STX error frame (e.g. no image buffered) -> drain & bail
            drained = first
            while b"\x03" not in drained:
                b = s.recv(1)
                if not b:
                    break
                drained += b
            return None

        # otherwise it's the leading digit of the "<size>\r\n" header
        head = first
        while b"\r\n" not in head:
            b = s.recv(1)
            if not b:
                return None
            head += b
        try:
            size = int(head.strip())
        except ValueError:
            return None

        buf = bytearray()
        while len(buf) < size:
            chunk = s.recv(min(size - len(buf), 65536))
            if not chunk:
                break
            buf.extend(chunk)
        if len(buf) != size or buf[:3] != b"\xff\xd8\xff":  # must be a complete JPEG
            return None
        return bytes(buf)


class BarcodeImagePublisher:
    """Polls the reader for its last image and spools it for the viewer."""

    def __init__(
        self,
        spool_dir: str = DEFAULT_SPOOL_DIR,
        host: str = cfg.BCR_HOST,
        port: int = cfg.BCR_PORT,
        hz: float = 1.0,
    ) -> None:
        self.spool_dir = spool_dir
        self._host = host
        self._port = int(port)
        self._period = 1.0 / max(hz, 0.1)
        self._img_path = os.path.join(spool_dir, "barcode.jpg")
        self._json_path = os.path.join(spool_dir, "barcode.json")
        # The demo's scanner touches this while actively triggering reads; the
        # Cognex serves one client at a time, so we skip IMAGE.SEND while it is
        # fresh and let the demo have the reader (panel keeps its last frame).
        self._lock_path = os.path.join(spool_dir, "bcr_scanning.lock")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        os.makedirs(spool_dir, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "BarcodeImagePublisher":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="bcr-image", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "BarcodeImagePublisher":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        # Event-driven: the reader's image only changes when a read is taken, so
        # there's no point polling IMAGE.SEND continuously (it also fights the
        # demo's triggers — one client at a time). Instead we pull a single
        # frame at startup, then exactly once each time the demo's scan finishes
        # (the scan lock goes from fresh -> stale). Between scans the reader is
        # left alone. We poll the lock file (cheap, local) — not the reader.
        poll = min(self._period, 0.5)
        was_scanning = self._scan_in_progress()
        if not was_scanning:
            self._safe_tick()  # one frame up front so the panel isn't blank
        while not self._stop.is_set():
            scanning = self._scan_in_progress()
            if was_scanning and not scanning:
                # Scan just completed: the reader is free and still holds the
                # frame it captured during the scan — grab it once.
                self._safe_tick()
            was_scanning = scanning
            self._stop.wait(poll)

    def _scan_in_progress(self) -> bool:
        """True if the demo's scanner is actively holding the reader."""
        try:
            age = time.time() - os.path.getmtime(self._lock_path)
        except OSError:
            return False
        return age < SCAN_LOCK_FRESH_S

    def _safe_tick(self) -> None:
        try:
            self._tick()
        except Exception as e:  # noqa: BLE001 - never die on a bad read
            self._write_meta(False, str(e))

    def _tick(self) -> None:
        jpg = pull_image(self._host, self._port)
        if jpg:
            self._seq += 1
            _atomic_write(self._img_path, jpg)
            self._write_meta(True, None, nbytes=len(jpg))
        else:
            self._write_meta(False, "no image")

    def _write_meta(self, ok: bool, error: str | None, nbytes: int = 0) -> None:
        meta = {"seq": self._seq, "stamp": time.time(), "ok": ok}
        if ok:
            meta["bytes"] = nbytes
        else:
            meta["error"] = error
        _atomic_write(self._json_path, json.dumps(meta).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard Cognex barcode-reader image feed")
    parser.add_argument("--spool", default=DEFAULT_SPOOL_DIR,
                        help="spool dir to write barcode.jpg + barcode.json into")
    parser.add_argument("--host", default=cfg.BCR_HOST, help="reader IP")
    parser.add_argument("--port", type=int, default=cfg.BCR_PORT, help="reader telnet/DMCC port")
    parser.add_argument("--hz", type=float, default=1.0,
                        help="scan-lock poll rate (default: 1 Hz; capped at 2 Hz internally). "
                             "Images are pulled only when a demo scan finishes, not on this cadence.")
    args = parser.parse_args()

    pub = BarcodeImagePublisher(spool_dir=os.path.abspath(args.spool),
                                host=args.host, port=args.port, hz=args.hz)
    print(f"[barcode] pulling images from {args.host}:{args.port} -> {pub._img_path}  (Ctrl-C to stop)")
    pub.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[barcode] stopping")
    finally:
        pub.stop()


if __name__ == "__main__":
    main()
