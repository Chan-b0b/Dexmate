"""Suction controller I/O for the case + battery demo.

Thin HTTP wrappers around the weblogic suction controller plus a background
vacuum-seal monitor (DI0 over socketio).  Lifted from
battery_pick/suction_grasp.py and pointed at the demo config.
"""

from __future__ import annotations

import threading
import time

import requests
from loguru import logger

from . import config as cfg


# ---------------------------------------------------------------------------
# Suction / blow commands
# ---------------------------------------------------------------------------

def _run(program_id: int) -> None:
    """Stop running programs, then trigger weblogic program *program_id*."""
    r_stop = requests.post(f"{cfg.SUCTION_BASE_URL}/stop", timeout=5.0)
    logger.debug("[Suction] stop → {} {}", r_stop.status_code, r_stop.text[:80])
    time.sleep(0.5)
    r_run = requests.post(f"{cfg.SUCTION_BASE_URL}/run/{program_id}", timeout=5.0)
    logger.debug("[Suction] run/{} → {} {}", program_id, r_run.status_code, r_run.text[:80])


# Tracks whether *we* have commanded suction ON. The toolA reading alone can't
# tell us this — the controller reports a 0.012 A idle baseline when OFF,
# which is higher than the actual pump-running current. So the seal logic
# must gate on what we commanded, not on what toolA looks like.
_suction_commanded_on: bool = False


def is_suction_commanded_on() -> bool:
    return _suction_commanded_on


def suction_on() -> None:
    global _suction_commanded_on
    _run(cfg.SUCTION_ON_ID)
    _suction_commanded_on = True
    logger.info("[Suction] ON (id={})", cfg.SUCTION_ON_ID)


def suction_off() -> None:
    global _suction_commanded_on
    _run(cfg.SUCTION_OFF_ID)
    _suction_commanded_on = False
    logger.info("[Suction] OFF (id={})", cfg.SUCTION_OFF_ID)


def blow_on() -> None:
    logger.info("[Suction] BLOW ON (id={})", cfg.BLOW_ON_ID)
    _run(cfg.BLOW_ON_ID)


def blow_off() -> None:
    logger.info("[Suction] BLOW OFF (id={})", cfg.BLOW_OFF_ID)
    _run(cfg.BLOW_OFF_ID)


def release() -> None:
    """Release a held object: suction off, then a short blow pulse."""
    suction_off()
    blow_on()
    time.sleep(2)
    logger.info("[Suction] released")


# ---------------------------------------------------------------------------
# Vacuum seal monitor (DI0 only)
# ---------------------------------------------------------------------------
#
# Empirical hardware behaviour (see traces from test_di0.py):
#
#     dInput[0] (DI0): goes T the moment vacuum seal is achieved while
#                      suction is commanded ON. Fastest, most reliable
#                      seal indicator (~500 ms ahead of any toolA change).
#                      Stays T after suction OFF until cup releases.
#
# toolA is intentionally NOT used as a seal signal: its OFF idle baseline
# (~0.012 A) is higher than the running-pump current (~0.006 A), so it
# can't be reasoned about without extra state and was the source of
# repeated false-positive seal detections in earlier revisions. toolA is
# still surfaced via ``get_tool_current()`` for diagnostics/logging only.
# ---------------------------------------------------------------------------

class VacuumMonitor:
    """Watches DI0 over socketio and reports vacuum seal.

    Runs in a daemon thread. Call ``start()`` before descending and
    ``stop()`` after. ``is_sealed()`` / ``wait_for_seal()`` report seal
    state, gated on ``is_suction_commanded_on()`` so a latched DI0 from a
    previous cycle can never be mistaken for a fresh seal.
    """

    def __init__(self, host: str = cfg.SUCTION_HOST) -> None:
        import socketio as _sio_module  # lazy import keeps startup fast

        self._seal_event = threading.Event()
        self._sio = _sio_module.Client()
        self._thread: threading.Thread | None = None
        self._host = host
        self._tool_current = 0.0  # diagnostic only
        self._di0 = False

        @self._sio.on("*")
        def _on_data(event, data) -> None:  # noqa: ARG001
            try:
                var = data["computebox"]["variable"]
            except (KeyError, TypeError):
                return

            try:
                self._tool_current = float(var["toolA"])
            except (KeyError, TypeError, ValueError):
                pass

            try:
                self._di0 = bool(var["dInput"][0])
            except (KeyError, TypeError, IndexError):
                self._di0 = False

            # Seal is only meaningful while we have commanded the pump ON.
            if not is_suction_commanded_on():
                self._seal_event.clear()
                return

            if self._di0:
                if not self._seal_event.is_set():
                    logger.debug(
                        "[VacuumMonitor] seal via DI0 (toolA={:.4f}A)",
                        self._tool_current,
                    )
                self._seal_event.set()
            else:
                self._seal_event.clear()

    def start(self) -> None:
        self._seal_event.clear()
        self._di0 = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="VacuumMonitor")
        self._thread.start()

    def _run(self) -> None:
        try:
            logger.info("[VacuumMonitor] Connecting to http://{}/socket.io", self._host)
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            logger.info("[VacuumMonitor] ✓ Connected! Watching DI0 for seal...")
            self._sio.wait()
        except Exception as exc:  # noqa: BLE001
            logger.error("[VacuumMonitor] ✗ Connection failed: {}", exc)

    def is_sealed(self) -> bool:
        return self._seal_event.is_set()

    def is_connected(self) -> bool:
        return self._sio.connected

    def get_tool_current(self) -> float:
        """Last toolA reading (Amps). Diagnostic only — not used for seal."""
        return self._tool_current

    def wait_for_seal(self, timeout: float) -> bool:
        """Block up to *timeout* s for seal. Returns True if sealed."""
        return self._seal_event.wait(timeout=timeout)

    def stop(self) -> None:
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
