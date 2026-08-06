"""Open-set anomaly monitor: per-(phase, feature) force envelope.

The envelope is built from NOMINAL runs only (envelope_build.py) — mean +
k*sigma UPPER bands per phase and feature, floored by ENVELOPE_MIN_BAND. No
failure enumeration: anything whose force signature leaves the nominal band
trips it, whatever the cause. TRIGGER_CONSECUTIVE consecutive out-of-band
ticks are required, so single-tick sensor spikes don't trip.

Thread contract: observe() is called from the SignalTap thread; tripped() is
polled from the main thread (suction.place tick_cb / supervisor phase
boundaries). State transitions are simple enough for the GIL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from . import config as cfg
from .signals import Tick

FEATURES = ("f_ax", "f_lat", "t_mag", "df_mag", "q_err_max")


class EnvelopeModel:
    """{phase: {feature: {"mean": m, "sigma": s, "n": n}}} + derived bounds."""

    def __init__(self, stats: dict, k: float = cfg.ENVELOPE_K_SIGMA) -> None:
        self.stats = stats
        self.k = float(k)
        self.bounds: dict[str, dict[str, float]] = {}
        for phase, feats in stats.items():
            self.bounds[phase] = {}
            for name, st in feats.items():
                band = max(self.k * float(st["sigma"]),
                           cfg.ENVELOPE_MIN_BAND.get(name, 0.0))
                bound = float(st["mean"]) + band
                if "qmax" in st:  # heavy-tail guard: never trip inside the
                    bound = max(bound,  # range nominal data actually reached
                                float(st["qmax"]) * cfg.ENVELOPE_QMAX_MARGIN)
                self.bounds[phase][name] = bound

    @classmethod
    def load(cls, path: str = cfg.ENVELOPE_PATH,
             k: float = cfg.ENVELOPE_K_SIGMA) -> "EnvelopeModel | None":
        p = Path(path)
        if not p.exists():
            return None
        model = cls(json.loads(p.read_text()), k=k)
        logger.info("[ik_VLM] envelope loaded: {} ({} phases, k={})",
                    p, len(model.stats), k)
        return model

    def save(self, path: str = cfg.ENVELOPE_PATH) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=2))
        logger.info("[ik_VLM] envelope saved: {}", path)


@dataclass
class TripInfo:
    t: float
    phase: str
    feature: str
    value: float
    bound: float
    ticks: int

    def describe(self) -> str:
        return (f"phase={self.phase} {self.feature}={self.value:.2f} "
                f"exceeded the nominal bound {self.bound:.2f} "
                f"for {self.ticks} consecutive ticks")


@dataclass
class Monitor:
    model: EnvelopeModel
    consecutive: int = cfg.TRIGGER_CONSECUTIVE
    armed_phases: tuple = cfg.MONITORED_PHASES
    _count: int = 0
    _tripped: TripInfo | None = None
    _last_exceed: tuple | None = field(default=None)  # (feature, value, bound)

    def observe(self, tick: Tick) -> bool:
        """Feed one tick (SignalTap thread). Returns True on the trip edge."""
        if self._tripped is not None:
            return False
        bounds = self.model.bounds.get(tick.phase)
        if bounds is None or tick.phase not in self.armed_phases:
            self._count = 0
            return False
        worst = None
        for name in FEATURES:
            b = bounds.get(name)
            v = getattr(tick, name)
            if b is not None and v > b:
                over = v - b
                if worst is None or over > worst[0]:
                    worst = (over, name, v, b)
        if worst is None:
            self._count = 0
            return False
        self._count += 1
        self._last_exceed = worst[1:]
        if self._count >= self.consecutive:
            self._tripped = TripInfo(tick.t, tick.phase, worst[1], worst[2],
                                     worst[3], self._count)
            logger.warning("[ik_VLM] MONITOR TRIPPED: {}", self._tripped.describe())
            return True
        return False

    def tripped(self) -> bool:
        return self._tripped is not None

    def trip_info(self) -> TripInfo | None:
        return self._tripped

    def reset(self) -> None:
        self._count = 0
        self._tripped = None
