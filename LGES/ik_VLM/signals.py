"""Background signal tap: tared wrench features at SIGNAL_HZ, phase-labeled.

Runs in a daemon thread beside the main IK stream. Reads ONLY the raw wrench
sensor + the taring baselines the mover already keeps — never FK/IK (the
pinocchio model is shared with the main thread and not thread-safe), so all
features are sensor-frame magnitudes (see config.py docstring).

Phases: the supervisor labels coarse phases (set_phase); the suction.place
tick_cb marks fine "descend" ticks (note_descent) which decay back to the
enclosing phase after DESCEND_PHASE_DECAY_S. Each tick is pushed to an
optional on_tick callback (the monitor) and, with SIGNAL_LOG_DIR set, appended
to a per-run jsonl — the envelope-build source.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg


@dataclass
class Tick:
    t: float
    phase: str
    f_ax: float
    f_lat: float
    t_mag: float
    df_mag: float           # N/s
    q_err_max: float = 0.0  # max |joint tracking error| (rad) — whole-arm
                            # contact/stall coverage the wrist wrench can't see
    ee_z: float | None = None   # only during descents (fed by the tick_cb)


def wrench_features(tared_now: np.ndarray, tared_prev: np.ndarray | None,
                    dt: float) -> tuple[float, float, float, float]:
    """(f_ax, f_lat, t_mag, df_mag) from a tared 6-vec sensor-frame wrench."""
    f, m = tared_now[:3], tared_now[3:6]
    f_ax = float(abs(f[2]))
    f_lat = float(np.hypot(f[0], f[1]))
    t_mag = float(np.linalg.norm(m))
    df = 0.0 if tared_prev is None else float(np.linalg.norm(f - tared_prev[:3]) / max(dt, 1e-6))
    return f_ax, f_lat, t_mag, df


class SignalTap:
    """Samples the mover's wrench sensor at SIGNAL_HZ in a daemon thread."""

    def __init__(self, mover, on_tick=None, log_dir: str | None = cfg.SIGNAL_LOG_DIR,
                 hz: float = cfg.SIGNAL_HZ) -> None:
        self._mover = mover
        self._on_tick = on_tick
        self._hz = float(hz)
        self._q_err_ok = True   # disabled on the first failed read (warn once)
        self._buf: deque[Tick] = deque(maxlen=int(cfg.SIGNAL_BUFFER_S * hz))
        self._phase = "idle"
        self._descend_z: float | None = None
        self._descend_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = None
        if log_dir is not None:
            try:
                p = Path(log_dir)
                p.mkdir(parents=True, exist_ok=True)
                path = p / f"signals_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                self._log = path.open("w", buffering=1)
                logger.info("[ik_VLM] signal log: {}", path)
            except OSError as e:
                logger.warning("[ik_VLM] signal log disabled ({})", e)

    # -- phase labeling (main thread) -----------------------------------
    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def note_descent(self, ee_z: float) -> None:
        """Called from suction.place's tick_cb (main thread, per descent tick)."""
        with self._lock:
            self._descend_z = float(ee_z)
            self._descend_t = time.monotonic()

    def current_phase(self) -> str:
        with self._lock:
            if time.monotonic() - self._descend_t < cfg.DESCEND_PHASE_DECAY_S:
                return "descend"
            return self._phase

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if getattr(self._mover, "_wrench", None) is None:
            logger.warning("[ik_VLM] no wrench sensor — signal tap OFF")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="SignalTap")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._log is not None:
            self._log.close()
            self._log = None

    def recent(self, seconds: float = 2.0) -> list[Tick]:
        cut = time.time() - seconds
        with self._lock:
            return [t for t in self._buf if t.t >= cut]

    # -- sampling thread ---------------------------------------------------
    def _tared(self) -> np.ndarray | None:
        s = np.asarray(self._mover._wrench.get_wrench_state(), dtype=float).ravel()
        if s.size < 6:
            s = np.concatenate([s[:3], np.zeros(3)])
        base = np.concatenate([self._mover._force_baseline, self._mover._torque_baseline])
        return s[:6] - base

    def _run(self) -> None:
        dt = 1.0 / self._hz
        prev: np.ndarray | None = None
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                w = self._tared()
            except Exception as e:  # noqa: BLE001 — the tap must never kill the run
                logger.warning("[ik_VLM] wrench read failed ({}) — tap paused 1s", e)
                time.sleep(1.0)
                prev = None
                continue
            f_ax, f_lat, t_mag, df = wrench_features(w, prev, dt)
            prev = w
            q_err = 0.0
            if self._q_err_ok:
                try:
                    e = self._mover._arm.get_joint_err()
                    if e is not None:
                        q_err = float(np.max(np.abs(np.asarray(e, dtype=float))))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[ik_VLM] get_joint_err unavailable ({}) — "
                                   "q_err_max feature disabled", exc)
                    self._q_err_ok = False
            with self._lock:
                descend = time.monotonic() - self._descend_t < cfg.DESCEND_PHASE_DECAY_S
                phase = "descend" if descend else self._phase
                ee_z = self._descend_z if descend else None
            tick = Tick(time.time(), phase, f_ax, f_lat, t_mag, df, q_err, ee_z)
            with self._lock:
                self._buf.append(tick)
            if self._log is not None:
                self._log.write(json.dumps(asdict(tick)) + "\n")
            if self._on_tick is not None:
                try:
                    self._on_tick(tick)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[ik_VLM] on_tick raised: {}", e)
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
