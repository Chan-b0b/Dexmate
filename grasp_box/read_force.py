"""Wrist force reading utilities.

Provides ``get_force(side, robot)`` which returns the scalar force magnitude
(sqrt(fx² + fy² + fz²)) for a given hand. Intended for use in grasp control
loops — e.g. move the arm inward until the measured force exceeds a threshold.

Usage as a standalone script:
    python read_force.py               # print once for both hands
    python read_force.py --loop        # print at ~10 Hz until Ctrl+C
    python read_force.py --side left
"""

from __future__ import annotations

import math
import time

import numpy as np
import tyro
from loguru import logger

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

# Per-side baseline wrench (fx, fy, fz) set by tare_force().
# Subtracted before computing force magnitude in get_force().
_baseline: dict[str, np.ndarray] = {}


def tare_force(side: str, robot: Robot, samples: int = 10) -> bool:
    """Record the current resting wrench as a baseline to zero out gravity/offset.

    Call this once while the hands are free (not touching anything).
    Subsequent calls to ``get_force()`` will subtract this baseline.

    Args:
        side: ``'left'``, ``'right'``, or ``'both'``.
        robot: An already-connected :class:`Robot` instance.
        samples: Number of readings to average for the baseline.

    Returns:
        True if baseline was recorded for all requested sides, False otherwise.
    """
    sides = ["left", "right"] if side == "both" else [side]
    success = True
    for s in sides:
        arm = robot.left_arm if s == "left" else robot.right_arm
        ws = arm.wrench_sensor
        if ws is None:
            logger.warning("[{}] wrench sensor unavailable — cannot tare", s)
            success = False
            continue
        readings = []
        for _ in range(samples):
            try:
                state = ws.get_state()
            except ServiceUnavailableError:
                logger.warning("[{}] wrench sensor unavailable — cannot tare", s)
                success = False
                break
            w = np.asarray(state["wrench"], dtype=np.float32)
            readings.append(w[:3])
            time.sleep(0.02)
        if readings:
            _baseline[s] = np.mean(readings, axis=0)
            logger.info(
                "[{}] tared: baseline fx={:.3f} fy={:.3f} fz={:.3f} N",
                s, *_baseline[s].tolist(),
            )
    print(f'Baseline after taring: {_baseline}')
    return success


def get_force(side: str, robot: Robot) -> float | None:
    """Return the tared scalar force magnitude (N) applied to the specified hand.

    Computes sqrt(fx² + fy² + fz²) after subtracting the baseline recorded by
    ``tare_force()``. If ``tare_force()`` has not been called, raw values are used.

    Args:
        side: ``'left'`` or ``'right'``.
        robot: An already-connected :class:`Robot` instance.

    Returns:
        Force magnitude in Newtons, or ``None`` if the sensor is unavailable.

    Example::

        with Robot() as bot:
            tare_force("both", bot)   # zero while hands are free
            while True:
                force = get_force("left", bot)
                if force is not None and force > 5.0:
                    print("Grasped!")
                    break
                # move arm closer …
    """
    arm = robot.left_arm if side == "left" else robot.right_arm
    ws = arm.wrench_sensor
    if ws is None:
        return None

    try:
        state = ws.get_state()
    except ServiceUnavailableError:
        return None

    w = np.asarray(state["wrench"], dtype=np.float32)
    raw = w[:3]
    baseline = _baseline.get(side, np.zeros(3, dtype=np.float32))
    fx, fy, fz = np.maximum(np.abs(raw) - np.abs(baseline), 0).tolist()
    # print(f'Raw wrench: {raw}, baseline: {baseline}, tared: {(fx, fy, fz)}')
    return math.sqrt(fx**2 + fy**2 + fz**2)


def _get_arm_row(side: str, robot: Robot) -> dict | None:
    """Return full wrench state dict (tared), or None if unavailable."""
    arm = robot.left_arm if side == "left" else robot.right_arm
    ws = arm.wrench_sensor
    if ws is None:
        return None

    try:
        state = ws.get_state()
    except ServiceUnavailableError:
        return None

    w = np.asarray(state["wrench"], dtype=np.float32)
    baseline = _baseline.get(side, np.zeros(3, dtype=np.float32))
    tared = np.maximum(np.abs(w[:3]) - np.abs(baseline), 0)
    fx, fy, fz = float(tared[0]), float(tared[1]), float(tared[2])
    _, _, _, tx, ty, tz = (float(x) for x in w.tolist())

    return {
        "timestamp": time.time(),
        "side": side,
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "force": math.sqrt(fx**2 + fy**2 + fz**2),
        "blue_button": bool(state["blue_button"]),
        "green_button": bool(state["green_button"]),
    }


def _print_row(row: dict) -> None:
    logger.info(
        "[{side}] force={force:.3f} N  "
        "(fx={fx:.3f}, fy={fy:.3f}, fz={fz:.3f})  "
        "torque=(tx={tx:.3f}, ty={ty:.3f}, tz={tz:.3f})  "
        "buttons=blue:{blue_button} green:{green_button}",
        **row,
    )


def main(
    side: str = "both",
    loop: bool = False,
    hz: float = 10.0,
) -> None:
    """Read tared wrist force from one or both hands.

    The sensor is automatically tared at startup (hands must be free).

    Args:
        side: Which hand to read — 'left', 'right', or 'both'.
        loop: If True, print continuously until Ctrl+C.
        hz: Polling rate in Hz when --loop is set.
    """
    if side not in ("left", "right", "both"):
        raise ValueError(f"side must be 'left', 'right', or 'both', got {side!r}")

    sides = ["left", "right"] if side == "both" else [side]
    period = 1.0 / max(hz, 0.1)

    with Robot() as bot:
        logger.info("Taring force sensors (keep hands free)...")
        tare_force(side, bot)
        while True:
            for s in sides:
                row = _get_arm_row(s, bot)
                if row is None:
                    logger.warning("[{}] wrench sensor unavailable", s)
                else:
                    _print_row(row)

            if not loop:
                break
            time.sleep(period)


if __name__ == "__main__":
    tyro.cli(main)



#ROBOT_NAME=dm/vg71b3858845-1p