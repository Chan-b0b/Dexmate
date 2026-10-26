"""Background watcher that logs every chassis move, whoever drives it.

The manual legs are driven by the external navigation, which tells us nothing:
the operator presses `d` by eye, so the `d` time carries the operator's wait.
The wheel motors report their state whoever commands them, so this samples
wheel speed + steering angle on a daemon thread and logs one line per move,
after the chassis has stood still for WATCH_STILL_S:

    [chassis watch] moved 0.632 m in 4.21 s: 16:07:43.913 -> 16:07:48.123

The two times are when the wheels started / stopped moving (the line itself is
logged WATCH_STILL_S later). Steering counts as moving, so the pre-steer before
a strafe is inside the move. Log only — it commands nothing.

Same lifecycle as StatePublisher (start()/stop(), context manager).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
from loguru import logger

WATCH_HZ = 50.0
WATCH_WHEEL_MOVE_MPS = 0.01     # either wheel faster than this = moving
WATCH_STEER_MOVE_RAD = 0.03     # steering this far from where it last rested = moving
WATCH_STILL_S = 0.5             # still this long = the move is over
WATCH_MIN_TRAVEL_M = 0.005      # shorter moves without a steer change are noise, not logged


def _hms(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


class ChassisWatch:
    """Samples the chassis wheels on a daemon thread and logs each move."""

    def __init__(self, robot, hz: float = WATCH_HZ) -> None:
        self._chassis = robot.chassis
        self._period = 1.0 / max(hz, 1.0)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._move: dict | None = None   # the move in progress

    def start(self) -> "ChassisWatch":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="chassis-watch", daemon=True)
        self._thread.start()
        logger.info("[chassis watch] logging every chassis move (wheels > {:.3f} m/s or steer "
                    "> {:.2f} rad, over after {:.1f} s still)",
                    WATCH_WHEEL_MOVE_MPS, WATCH_STEER_MOVE_RAD, WATCH_STILL_S)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._move is not None:
            self._log(self._move, ended=False)
            self._move = None

    def __enter__(self) -> "ChassisWatch":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _read(self):
        try:
            v = np.abs(np.asarray(self._chassis.wheel_velocity, dtype=float))
            s = np.asarray(self._chassis.steering_angle, dtype=float)
        except Exception:
            return None, None
        if v.shape != (2,) or s.shape != (2,):
            return None, None
        return v, s

    def _log(self, mv: dict, ended: bool = True) -> None:
        steered = float(np.max(np.abs(mv["steer_end"] - mv["steer_start"])))
        if mv["travel"] < WATCH_MIN_TRAVEL_M and steered < WATCH_STEER_MOVE_RAD:
            return
        logger.info("[chassis watch] moved {:.3f} m in {:.2f} s: {} -> {}{} (peak {:.2f} m/s, "
                    "steer {} -> {} rad)",
                    mv["travel"], mv["last"] - mv["t0"], _hms(mv["t0"]), _hms(mv["last"]),
                    "" if ended else " (still moving at shutdown)", mv["peak"],
                    np.round(mv["steer_start"], 2).tolist(), np.round(mv["steer_end"], 2).tolist())

    def _run(self) -> None:
        rest = None     # steering where the chassis last stood still
        t_prev = None
        while not self._stop.is_set():
            t = time.time()
            v, s = self._read()
            if v is not None:
                if rest is None:
                    rest = s
                active = (v.max() > WATCH_WHEEL_MOVE_MPS
                          or np.max(np.abs(s - (self._move["steer_end"] if self._move else rest)))
                          > WATCH_STEER_MOVE_RAD)
                mv = self._move
                if mv is None and active:
                    mv = self._move = dict(t0=t, last=t, travel=0.0, peak=0.0,
                                           steer_start=rest, steer_end=s)
                if mv is not None:
                    if t_prev is not None:
                        mv["travel"] += float(v.mean()) * (t - t_prev)
                    mv["peak"] = max(mv["peak"], float(v.max()))
                    if active:
                        mv["last"] = t
                        mv["steer_end"] = s
                    elif t - mv["last"] > WATCH_STILL_S:
                        self._log(mv)
                        self._move = None
                        rest = s
                t_prev = t
            self._stop.wait(max(0.0, self._period - (time.time() - t)))
