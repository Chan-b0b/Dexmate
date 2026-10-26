"""SPACE-to-pause for the barcode-gated pick (BCR aim tuning).

While pick_gated's scan-descent / sweep streams per tick, a daemon thread
watches stdin: SPACE toggles a pause. The streaming loop calls ``hold()`` on
its next tick — the arm is already halted by the caller — and blocks there
with the BackgroundScanner reading continuously (BCR_MAX_READS cap lifted,
every read / no-read logged) until SPACE is pressed again.

Needs a tty on stdin (cbreak mode, Ctrl+C still works); otherwise it logs a
warning and never pauses.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

from loguru import logger


class SpacePause:
    def __init__(self, scanner=None) -> None:
        self._scanner = scanner
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old = None

    def start(self) -> "SpacePause":
        if not sys.stdin.isatty():
            logger.warning("[pause] stdin is not a tty — SPACE-to-pause disabled")
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="space-pause", daemon=True)
        self._thread.start()
        logger.info("[pause] SPACE pauses the arm (BCR keeps reading); SPACE again resumes")
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([self._fd], [], [], 0.05)
            if r and os.read(self._fd, 1) == b" ":
                if self._paused.is_set():
                    self._paused.clear()
                else:
                    self._paused.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None
        self._paused.clear()

    def paused(self) -> bool:
        return self._paused.is_set()

    def hold(self) -> None:
        """Block until SPACE is pressed again; the scanner reads uncapped meanwhile."""
        logger.warning("[pause] PAUSED — arm held, BCR reading continuously; SPACE to resume")
        if self._scanner is not None:
            self._scanner.set_unlimited(True)
        try:
            while self._paused.is_set():
                time.sleep(0.02)
        finally:
            if self._scanner is not None:
                self._scanner.set_unlimited(False)
        logger.info("[pause] resumed")
