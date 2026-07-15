#!/usr/bin/env python3
"""Standalone DI0 (digital input 0) probe for the suction controller.

Connects to the controller's socketio stream and prints every DI/DO change
plus the live toolA reading. Cycles suction ON/OFF in the background so you
can press a battery against the cup during ON phases and watch whether DI0
ever transitions T while sealed.

Run:
    cd Dexmate/LGES
    python test_di0.py
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, "case_battery_demo")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import socketio as _sio_module

from case_battery_demo import config as cfg
from case_battery_demo import suction_io


# Cycle parameters
ON_DURATION_S = 10.0
OFF_DURATION_S = 2.0
PRINT_INTERVAL_S = 0.1   # heartbeat row every 100 ms even if nothing changed
NUM_DI = 16              # how many digital inputs to track
NUM_DO = 16              # how many digital outputs to track


class IOWatcher:
    """Subscribe to socketio and report DI/DO transitions + toolA."""

    def __init__(self, host: str = cfg.SUCTION_HOST) -> None:
        self._sio = _sio_module.Client()
        self._host = host
        self._lock = threading.Lock()
        self._di: list[bool] = [False] * NUM_DI
        self._do: list[bool] = [False] * NUM_DO
        self._toolA: float = 0.0
        self._connected = False

        self._di0_high_count = 0   # how many events saw DI0 = T
        self._di0_high_while_on = 0
        self._first_event_dumped = False

        @self._sio.on("*")
        def _on_data(event, data) -> None:  # noqa: ARG001
            try:
                var = data["computebox"]["variable"]
            except (KeyError, TypeError):
                return

            di = list(var.get("dInput") or [])
            do = list(var.get("dOutput") or [])
            tool_a = float(var.get("toolA", 0.0))

            with self._lock:
                # Pad/truncate to fixed length so indexing is always safe.
                di_padded = (di + [False] * NUM_DI)[:NUM_DI]
                do_padded = (do + [False] * NUM_DO)[:NUM_DO]

                # Detect transitions on every DI / DO bit.
                for i in range(NUM_DI):
                    new = bool(di_padded[i])
                    if new != self._di[i]:
                        self._log_transition("DI", i, new)
                        self._di[i] = new
                        if i == 0 and new:
                            self._di0_high_count += 1
                            if suction_io.is_suction_commanded_on():
                                self._di0_high_while_on += 1
                for i in range(NUM_DO):
                    new = bool(do_padded[i])
                    if new != self._do[i]:
                        self._log_transition("DO", i, new)
                        self._do[i] = new

                self._toolA = tool_a

                if not self._first_event_dumped:
                    self._first_event_dumped = True
                    self._dump_first_event(var)

    @staticmethod
    def _log_transition(kind: str, index: int, value: bool) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        suction = "ON" if suction_io.is_suction_commanded_on() else "OFF"
        arrow = "T" if value else "F"
        marker = " ★" if (kind == "DI" and index == 0) else ""
        print(f"{ts}  [suction={suction:3}]  {kind}{index:<2} -> {arrow}{marker}")

    @staticmethod
    def _dump_first_event(var: dict) -> None:
        print()
        print("=" * 78)
        print("FIRST EVENT — keys in computebox.variable:")
        print("=" * 78)
        for k, v in var.items():
            if isinstance(v, list):
                preview = f"list[{len(v)}] head={v[: min(6, len(v))]}"
            elif isinstance(v, dict):
                preview = f"dict keys={list(v.keys())[:8]}"
            else:
                preview = repr(v)
            print(f"  {k:20s} = {preview}")
        print("=" * 78)
        print()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="IOWatcher").start()

    def _run(self) -> None:
        try:
            print(f"Connecting to http://{self._host}/socket.io ...")
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            self._connected = True
            print("✓ Connected. Watching DI/DO transitions + toolA.\n")
            self._sio.wait()
        except Exception as exc:  # noqa: BLE001
            print(f"✗ Connection failed: {exc}")

    def is_connected(self) -> bool:
        return self._connected

    def snapshot(self) -> tuple[list[bool], list[bool], float]:
        with self._lock:
            return list(self._di), list(self._do), self._toolA

    def stats(self) -> tuple[int, int]:
        return self._di0_high_count, self._di0_high_while_on

    def stop(self) -> None:
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:  # noqa: BLE001
            pass


def _fmt_bits(bits: list[bool]) -> str:
    return "".join("1" if b else "0" for b in bits)


def main() -> None:
    print("=" * 78)
    print("DI0 SEAL-SENSOR PROBE")
    print("=" * 78)
    print(f"Host: {cfg.SUCTION_HOST}")
    print(f"Cycle: {ON_DURATION_S}s ON / {OFF_DURATION_S}s OFF")
    print()
    print("During each ON phase, press the cup firmly onto a battery and")
    print("hold it. Watch whether 'DI0 -> T' ever appears WHILE suction=ON.")
    print("Press Ctrl+C to stop and see a summary.")
    print()

    watcher = IOWatcher()
    watcher.start()
    time.sleep(2.0)

    if not watcher.is_connected():
        print("Could not connect; aborting.")
        sys.exit(1)

    suction_io.suction_off()
    state = "OFF"
    phase_start = time.time()
    last_heartbeat = 0.0

    try:
        while True:
            now = time.time()

            if state == "OFF" and now - phase_start >= OFF_DURATION_S:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')}  SUCTION ON ---")
                suction_io.suction_on()
                state = "ON"
                phase_start = now
            elif state == "ON" and now - phase_start >= ON_DURATION_S:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')}  SUCTION OFF ---")
                suction_io.suction_off()
                state = "OFF"
                phase_start = now

            # Periodic heartbeat row so you can see live state even when no
            # transition is happening.
            if now - last_heartbeat >= PRINT_INTERVAL_S:
                last_heartbeat = now
                di, do, tool_a = watcher.snapshot()
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(
                    f"{ts}  [{state:3}]  toolA={tool_a:7.4f}A  "
                    f"DI={_fmt_bits(di)}  DO={_fmt_bits(do)}"
                )

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        suction_io.suction_off()
        di0_total, di0_while_on = watcher.stats()
        watcher.stop()
        print()
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"DI0 high transitions total:        {di0_total}")
        print(f"DI0 high transitions while ON:     {di0_while_on}")
        if di0_while_on > 0:
            print("→ DI0 DOES fire as a seal sensor on this hardware.")
        elif di0_total > 0:
            print("→ DI0 fires, but only OUTSIDE suction-ON windows. Not a seal sensor.")
        else:
            print("→ DI0 never went high. Either not wired, or wrong index.")
        print()


if __name__ == "__main__":
    main()
