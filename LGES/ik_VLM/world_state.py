"""Tier-1 world re-reading: seal + fresh BEV detection -> WorldState.

After a hold, the ONLY reliable way to handle a cause we didn't enumerate is
to re-read the world and re-enter the script at the right point. This module
does the reading; resume_matrix.py does the deciding.

Imports of the detection stack are kept inside classify() so the offline
tools (envelope_build / replay_test) never pull YOLO/camera dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from . import config as cfg


@dataclass
class WorldState:
    phase: str                    # phase at the trigger ("descend", "transport", ...)
    label: str                    # item ("case" / "battery_1" / "battery_2")
    station: str                  # "source" / "target"
    holding_expected: bool        # the script believed a part was on the cup
    sealed: bool | None           # DI0 (None = monitor unreachable)
    suction_commanded: bool
    case_found: bool
    case_center: tuple | None     # (x, y, z_ee, yaw_rad) — resolve_poses convention
    xy_err_m: float | None        # |detected - expected| xy, when both known

    def summary(self) -> str:
        parts = [f"phase={self.phase}", f"item={self.label}@{self.station}",
                 f"holding_expected={self.holding_expected}", f"sealed={self.sealed}"]
        if self.case_found and self.case_center is not None:
            parts.append("case@(%.3f,%+.3f) yaw=%.1fdeg" % (
                self.case_center[0], self.case_center[1],
                float(np.rad2deg(self.case_center[3]))))
            if self.xy_err_m is not None:
                parts.append(f"xy_err={self.xy_err_m * 1000:.0f}mm")
        else:
            parts.append("case NOT found")
        return " ".join(parts)


def check_seal() -> bool | None:
    """One-shot DI0 read (own VacuumMonitor; ~SEAL_CHECK_TIMEOUT_S + the
    ~1.3 s socketio disconnect — recovery-path cost, never in a control loop)."""
    from ..ik_demo.drivers import suction_io
    vac = suction_io.VacuumMonitor()
    vac.start()
    try:
        deadline = time.time() + cfg.SEAL_CHECK_TIMEOUT_S
        while time.time() < deadline:
            if vac.is_sealed():
                return True
            time.sleep(0.05)
        return False if vac.is_connected() else None
    finally:
        import threading
        threading.Thread(target=vac.stop, daemon=True).start()


def classify(bot, *, phase: str, label: str, station: str, holding_expected: bool,
             layers: int, plane_z: float | None = None,
             expected_xy: tuple[float, float] | None = None) -> WorldState:
    """Re-read the world at the CURRENT chassis position. The arm must already
    be clear of the head-camera view (recovery lifts to transport first)."""
    from ..ik_demo import chassis_sequence as cs
    from ..ik_demo.drivers import suction_io

    sealed = check_seal()
    det = None
    for i in range(cfg.REDETECT_TRIES):
        det = cs.detect(bot, layers, plane_z)
        if det is not None and det.found:
            break
        logger.warning("[ik_VLM] re-detect {}/{}: no case", i + 1, cfg.REDETECT_TRIES)
    found = det is not None and det.found
    center = cs._center_from_det(det) if found else None
    xy_err = None
    if found and expected_xy is not None:
        xy_err = float(np.hypot(center[0] - expected_xy[0], center[1] - expected_xy[1]))
    w = WorldState(phase=phase, label=label, station=station,
                   holding_expected=holding_expected, sealed=sealed,
                   suction_commanded=suction_io.is_suction_commanded_on(),
                   case_found=found, case_center=center, xy_err_m=xy_err)
    logger.info("[ik_VLM] world state: {}", w.summary())
    return w
