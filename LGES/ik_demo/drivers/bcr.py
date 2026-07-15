"""Cognex barcode reader (DataMan) — trigger a read and collect the result.

Sends ``T`` to fire a read and returns the decoded string.

``BackgroundScanner`` fires reads in a daemon thread so the demo can scan
"while descending" during a pick without stalling the descent loop, then
reports a value only if enough reads agree.

The scan-lock (touched while actively triggering, cleared on stop) is back:
ik_demo.dashboard_publish now runs alongside case_battery_demo.dashboard.barcode
(same spool dir), and that image feed only ever pulls ONE frame at startup and
then waits for this lock to go fresh -> stale before pulling again. Without
it, the dashboard's barcode panel silently freezes after its first frame.
"""

from __future__ import annotations

import os
import random
import socket
import threading
from collections import Counter

from loguru import logger

try:
    from .. import config as cfg
except ImportError:  # allow running a module directly from ik_demo/
    import config as cfg

# Reader replies that mean "no code read" rather than a real barcode.
_NO_CODE = {"", "NOREAD", "NO READ", "NG"}

# Matches dashboard_publish.DEFAULT_SPOOL_DIR / case_battery_demo.dashboard.barcode's
# DEFAULT_SPOOL_DIR — the same lock path both sides agree on.
DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"
SCAN_LOCK_PATH = os.path.join(DEFAULT_SPOOL_DIR, "bcr_scanning.lock")


def _touch_scan_lock() -> None:
    """Refresh the scan lock (best-effort) so the dashboard image feed backs off."""
    try:
        os.makedirs(DEFAULT_SPOOL_DIR, exist_ok=True)
        with open(SCAN_LOCK_PATH, "w") as f:
            f.write("1")
    except OSError:
        pass


def _clear_scan_lock() -> None:
    try:
        os.remove(SCAN_LOCK_PATH)
    except OSError:
        pass


def scan_once(
    host: str = cfg.BCR_HOST,
    port: int = cfg.BCR_PORT,
    timeout: float | None = None,
) -> str | None:
    """Trigger one read and return the decoded barcode, or None on no-read/error.

    Uses a short-lived connection per call (the reader allows only a few
    concurrent clients).
    """
    timeout = cfg.BCR_SCAN_TIMEOUT_S if timeout is None else timeout
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            s.sendall(b"T\r\n")
            data = s.recv(1024)
    except OSError as exc:
        logger.debug("[BCR] scan error: {}", exc)
        return None

    if not data:
        return None
    # Strip STX/ETX framing and whitespace, then take only the first
    # comma-delimited field. The DataMan appends bounding-box coordinates
    # after the barcode ID (e.g. "UDCG7B0307,1005,257,...") and those vary
    # per read, so we must discard them before the consistency check.
    text = data.decode("utf-8", "ignore").strip().strip("\x02\x03").strip()
    text = text.split(",")[0].strip()
    if not text or text.upper() in _NO_CODE:
        return None
    return text


class BackgroundScanner:
    """Fires reads in a daemon thread and reports the agreed value.

    Start before a pick, stop after. ``result()`` returns the majority code
    among the collected reads, provided its count reaches ``cfg.BCR_MIN_READS``.
    A true tie is broken randomly rather than treated as no-read.
    """

    def __init__(
        self,
        host: str = cfg.BCR_HOST,
        port: int = cfg.BCR_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._reads: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "BackgroundScanner":
        self._reads = []
        self._stop.clear()
        _touch_scan_lock()  # claim the reader before the first trigger
        self._thread = threading.Thread(target=self._run, name="bcr-scan", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            _touch_scan_lock()  # keep the lock fresh so the image feed stays backed off
            code = scan_once(self._host, self._port)
            if code is not None:
                with self._lock:
                    self._reads.append(code)
                    n = len(self._reads)
                logger.info("[BCR] read: {!r}  (total reads: {})", code, n)
                # Enough reads in hand — stop triggering rather than keep
                # hammering the reader. result() still applies the agreement
                # check over what we collected.
                if n >= cfg.BCR_MAX_READS:
                    logger.info("[BCR] reached {} reads — stopping scan", n)
                    break
            # Small breather so we don't hammer the reader (and so stop() is
            # responsive between triggers).
            self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=cfg.BCR_SCAN_TIMEOUT_S + 1.0)
            self._thread = None
        _clear_scan_lock()  # release the reader so the image feed resumes

    def __enter__(self) -> "BackgroundScanner":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def result(self) -> str | None:
        """Majority-agreed barcode, or None if too few reads even for the winner.

        The winning code's OWN count (not the total read count) must reach
        cfg.BCR_MIN_READS — e.g. 2 agreeing reads out of 5 noisy ones still wins.
        A true tie for first place is broken randomly (never yields None just
        because of a tie).
        """
        with self._lock:
            reads = list(self._reads)
        if not reads:
            return None
        counts = Counter(reads)
        top_n = max(counts.values())
        tied = [code for code, n in counts.items() if n == top_n]
        if len(tied) > 1:
            code = random.choice(tied)
            logger.warning("[BCR] tie among {} ({} reads each) — randomly picked {!r}",
                           sorted(tied), top_n, code)
        else:
            code = tied[0]
            if len(counts) > 1:
                logger.warning("[BCR] inconsistent reads {} — going with majority {!r} ({}/{})",
                               dict(counts), code, top_n, len(reads))
        if top_n < cfg.BCR_MIN_READS:
            logger.info("[BCR] only {} read(s) for {!r} (< {}), ignoring", top_n, code, cfg.BCR_MIN_READS)
            return None
        logger.info("[BCR] agreed: {!r}  ({}/{} reads)", code, top_n, len(reads))
        return code
