"""Cognex barcode reader (DataMan) — trigger a read and collect the result.

Sends ``T`` to fire a read and returns the decoded string.

``BackgroundScanner`` fires reads in a daemon thread so the demo can scan
"while descending" during a pick without stalling the descent loop, then
reports a value only if enough reads agree.

Reused as-is from case_battery_demo, minus the dashboard scan-lock (the lock
only told the dashboard image feed to yield the Cognex reader; with no
dashboard there is no competing client, so it is gone).
"""

from __future__ import annotations

import socket
import threading

from loguru import logger

try:
    from .. import config as cfg
except ImportError:  # allow running a module directly from ik_demo/
    import config as cfg

# Reader replies that mean "no code read" rather than a real barcode.
_NO_CODE = {"", "NOREAD", "NO READ", "NG"}


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

    Start before a pick, stop after. ``result()`` returns the barcode iff at
    least ``cfg.BCR_MIN_READS`` successful reads were collected and they all
    agree; any disagreement (two different codes seen) yields None.
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
        self._thread = threading.Thread(target=self._run, name="bcr-scan", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
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

    def __enter__(self) -> "BackgroundScanner":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def result(self) -> str | None:
        """Agreed barcode, or None if too few reads / disagreement."""
        with self._lock:
            reads = list(self._reads)
        unique = set(reads)
        if len(reads) >= cfg.BCR_MIN_READS and len(unique) == 1:
            logger.info("[BCR] agreed: {!r}  ({} reads)", reads[0], len(reads))
            return reads[0]
        if len(unique) > 1:
            logger.warning("[BCR] inconsistent reads, ignoring: {}", sorted(unique))
        elif reads:
            logger.info("[BCR] only {} read(s) (< {}), ignoring", len(reads), cfg.BCR_MIN_READS)
        return None
