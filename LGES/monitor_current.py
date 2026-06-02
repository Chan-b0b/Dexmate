#!/usr/bin/env python3
"""Auto-cycle suction ON/OFF and rapidly sample tool current + extra signals.

Hold an object near the cup at varying distances to characterize the
toolA current profile while the pump fights to build vacuum.

This version attaches its own socketio listener so we can observe MANY
fields from ``data["computebox"]["variable"]`` side by side, not just
``toolA`` / ``dInput[0]``. The first event received dumps every key in
the variable dict so you can see what's available.
"""

import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, 'case_battery_demo')

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import socketio as _sio_module

from case_battery_demo import config as cfg
from case_battery_demo import suction_io

# Cycle parameters
ON_DURATION_S = 3.0      # how long suction stays ON
OFF_DURATION_S = 1.5     # how long suction stays OFF
SAMPLE_INTERVAL_S = 0.05 # 20 Hz sampling

# Fields to display each tick. Anything missing is shown as "-".
# Add/remove keys here after seeing the first-event dump.
NUMERIC_FIELDS = ["toolA", "toolV", "robotV", "robotA", "robotTemp", "tcpForce"]
LIST_FIELDS = [("dInput", 8), ("dOutput", 8)]  # (key, num_indices_to_show)


# ---------------------------------------------------------------------------
# Raw payload watcher — captures the full "variable" dict from socketio
# ---------------------------------------------------------------------------

class RawWatcher:
    """Mirrors VacuumMonitor's connection but stores the full latest payload."""

    def __init__(self, host: str = cfg.SUCTION_HOST) -> None:
        self._sio = _sio_module.Client()
        self._host = host
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._first_seen = False
        self._thread: threading.Thread | None = None

        @self._sio.on("*")
        def _on_data(event, data) -> None:  # noqa: ARG001
            try:
                var = data["computebox"]["variable"]
            except (KeyError, TypeError):
                return
            with self._lock:
                self._latest = var
                if not self._first_seen:
                    self._first_seen = True
                    self._dump_keys(var)

    @staticmethod
    def _dump_keys(var: dict) -> None:
        print()
        print("=" * 78)
        print("FIRST EVENT — available keys in computebox.variable:")
        print("=" * 78)
        for k, v in var.items():
            if isinstance(v, list):
                preview = f"list[{len(v)}] head={v[: min(4, len(v))]}"
            elif isinstance(v, dict):
                preview = f"dict keys={list(v.keys())[:6]}"
            else:
                preview = repr(v)
            print(f"  {k:20s} = {preview}")
        print("=" * 78)
        print()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="RawWatcher")
        self._thread.start()

    def _run(self) -> None:
        try:
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            self._sio.wait()
        except Exception as exc:  # noqa: BLE001
            print(f"[RawWatcher] connection failed: {exc}")

    def snapshot(self) -> dict | None:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self) -> None:
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:  # noqa: BLE001
            pass


def _fmt_value(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int, float)):
        return f"{float(v):8.4f}"
    if isinstance(v, list):
        return repr(v)[:20]
    if isinstance(v, dict):
        return f"<dict:{len(v)}>"
    return str(v)[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 78)
print("AUTO-CYCLE SUCTION + MULTI-FIELD MONITOR")
print("=" * 78)
print(f"Cycle: {ON_DURATION_S}s ON / {OFF_DURATION_S}s OFF, sampling at {1/SAMPLE_INTERVAL_S:.0f} Hz")
print("Hold the battery near the cup. Press Ctrl+C to stop.")
print()

vac = suction_io.VacuumMonitor()
vac.start()
raw = RawWatcher()
raw.start()

print("Connecting...")
time.sleep(2.5)

if not vac.is_connected():
    print("VacuumMonitor connection failed.")
    sys.exit(1)

print("Connected.\n")

# Build header dynamically from the configured fields.
header_cols = [f"{'Time':14}", f"{'Phase':5}", f"{'Sealed':6}"]
header_cols += [f"{name:>10}" for name in NUMERIC_FIELDS]
for name, n in LIST_FIELDS:
    header_cols += [f"{name + str(i):>4}" for i in range(n)]
header = " | ".join(header_cols)
print(header)
print("-" * len(header))

state = "OFF"
suction_io.suction_off()
phase_start = time.time()

try:
    while True:
        now = time.time()

        # Toggle suction based on phase duration.
        if state == "OFF" and now - phase_start >= OFF_DURATION_S:
            suction_io.suction_on()
            state = "ON"
            phase_start = now
            print(f"{'':14}   {'─' * (len(header) - 18)}")
        elif state == "ON" and now - phase_start >= ON_DURATION_S:
            suction_io.suction_off()
            state = "OFF"
            phase_start = now
            print(f"{'':14}   {'─' * (len(header) - 18)}")

        snap = raw.snapshot() or {}
        sealed = vac.is_sealed()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        cols = [f"{ts:14}", f"{state:5}", f"{('T' if sealed else 'F'):6}"]
        for name in NUMERIC_FIELDS:
            cols.append(f"{_fmt_value(snap.get(name)):>10}")
        for name, n in LIST_FIELDS:
            arr = snap.get(name) or []
            for i in range(n):
                v = arr[i] if i < len(arr) else None
                cols.append(f"{_fmt_value(v):>4}")
        print(" | ".join(cols))

        time.sleep(SAMPLE_INTERVAL_S)

except KeyboardInterrupt:
    pass
finally:
    suction_io.suction_off()
    vac.stop()
    raw.stop()
    print("\nStopped.")
