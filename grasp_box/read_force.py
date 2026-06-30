"""Wrist force reading utilities.

Provides ``get_force(side, robot)`` which returns the scalar force magnitude
(sqrt(fx² + fy² + fz²)) for a given hand. Intended for use in grasp control
loops — e.g. move the arm inward until the measured force exceeds a threshold.

For vertical descents, ``get_push_force(side, robot)`` returns the median-
filtered *signed* force along the sensor approach axis (positive = pressing into
the surface), which rejects lateral noise and single-sample spikes that the
magnitude does not.

``get_vertical_force(side, robot, rotation)`` goes one step further: it rotates
the wrench into the base frame and returns the median-filtered vertical contact
force. Since gravity is a constant offset in the base frame, one tare stays
valid as the wrist tilts — so it is robust to the orientation changes a fixed
sensor axis is not (e.g. the place-descent jitter).

Usage as a standalone script:
    python read_force.py               # print once for both hands
    python read_force.py --loop        # print at ~10 Hz until Ctrl+C
    python read_force.py --side left
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import tyro
from loguru import logger

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

# Per-side baseline wrench (fx, fy, fz) set by tare_force().
# Subtracted before computing force magnitude in get_force().
_baseline: dict[str, np.ndarray] = {}

# Signed push-force support (see get_push_force).
# A vertical descent presses the tool INTO the surface along the wrench z-axis;
# empirically (suction arm, cup pointing down) that press DROPS fz below its
# resting/gravity baseline, so the press magnitude is `baseline_fz - fz_now`.
_PUSH_AXIS = 2  # wrench index for the approach/descent axis (z)
# Median-filter window for get_push_force. At the 50 ms descent tick a window
# of 3 is a 2-of-3 majority vote (~100 ms to confirm a sustained change): it
# still rejects single-sample sensor spikes, but keeps detection latency low so
# the arm halts before it over-presses a stiff contact into the hard-force
# limit. Larger windows reject more noise at the cost of more contact overshoot.
_FILTER_WINDOW = 3
# Per-side ring buffer of recent signed push samples; cleared on tare_force().
_push_filter: dict[str, deque] = {}

# Base-frame-vertical support (see get_vertical_force). The wrist wrench is
# reported in the sensor frame, which rotates with the arm, so a fixed sensor
# axis is only valid at one orientation. Rotating the force into the base frame
# (where gravity is a constant vertical offset) lets a single tare stay valid as
# the wrist wobbles. _baseline_base_z holds the resting base-frame vertical
# force captured by tare_force(rotation=...); _vertical_filter is its median buffer.
_baseline_base_z: dict[str, float] = {}
_vertical_filter: dict[str, deque] = {}


def tare_force(side: str, robot: Robot, rotation=None, samples: int = 10) -> bool:
    """Record the current resting wrench as a baseline to zero out gravity/offset.

    Call this once while the hands are free (not touching anything).
    Subsequent calls to ``get_force()`` will subtract this baseline.

    Args:
        side: ``'left'``, ``'right'``, or ``'both'``.
        robot: An already-connected :class:`Robot` instance.
        rotation: Optional 3x3 ``R_base_sensor`` matrix (sensor-frame -> base).
            If given, the resting base-frame vertical force is also stored so
            :func:`get_vertical_force` can subtract it. Pass the live EE rotation
            for the side being tared (single side only).
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
            _push_filter.pop(s, None)  # drop stale pre-tare samples
            _vertical_filter.pop(s, None)
            if rotation is not None:
                _baseline_base_z[s] = float(
                    (np.asarray(rotation, dtype=float) @ _baseline[s])[2]
                )
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
    tared = raw - baseline
    fx, fy, fz = float(tared[0]), float(tared[1]), float(tared[2])
    return math.sqrt(fx**2 + fy**2 + fz**2)


def get_push_force(side: str, robot: Robot, window: int | None = None) -> float | None:
    """Return the median-filtered signed *push* force (N) along the approach axis.

    Unlike :func:`get_force` (direction-agnostic magnitude), this isolates the
    component that matters during a vertical descent: how hard the tool is
    pressing INTO the surface. Positive means pressing; values near zero or
    negative mean no contact (or being pulled away).

    The approach-axis reading is baselined by :func:`tare_force` and
    sign-corrected so a press reads positive (a press drops ``fz`` below its
    resting/gravity baseline, hence ``push = baseline_fz - fz_now``). The last
    ``window`` samples are median-filtered to reject single-sample sensor
    spikes that would otherwise trip a contact/hard-force threshold on noise.

    Args:
        side: ``'left'`` or ``'right'``.
        robot: An already-connected :class:`Robot` instance.
        window: Median filter length in samples. Defaults to ``_FILTER_WINDOW``.

    Returns:
        Signed push force in Newtons, or ``None`` if the sensor is unavailable.
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
    baseline = _baseline.get(side, np.zeros(3, dtype=np.float32))
    push = float(baseline[_PUSH_AXIS] - w[_PUSH_AXIS])

    win = _FILTER_WINDOW if window is None else max(1, int(window))
    buf = _push_filter.get(side)
    if buf is None or buf.maxlen != win:
        buf = deque(buf or (), maxlen=win)
        _push_filter[side] = buf
    buf.append(push)
    return float(np.median(buf))


def get_vertical_force(side: str, robot: Robot, rotation, window: int | None = None) -> float | None:
    """Return the median-filtered contact force along the base-frame vertical.

    This is the orientation-robust version of :func:`get_push_force`. The raw
    wrist force (sensor frame) is rotated into the base frame via ``rotation``
    (``R_base_sensor``, the live EE rotation), then the resting base-frame
    vertical force captured by ``tare_force(rotation=...)`` is subtracted.
    Because gravity/tool weight is a *constant* vertical offset in the base
    frame, that single baseline stays valid even as the wrist tilts (e.g. the
    place-descent jitter) — unlike a fixed sensor-axis baseline. Positive means
    pressing into the surface (the contact pushes the tool up).

    Args:
        side: ``'left'`` or ``'right'``.
        robot: An already-connected :class:`Robot` instance.
        rotation: 3x3 ``R_base_sensor`` matrix (sensor-frame -> base).
        window: Median filter length in samples. Defaults to ``_FILTER_WINDOW``.

    Returns:
        Signed vertical contact force in Newtons, or ``None`` if unavailable.
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
    f_base_z = float((np.asarray(rotation, dtype=float) @ w[:3].astype(float))[2])
    push = f_base_z - _baseline_base_z.get(side, 0.0)

    win = _FILTER_WINDOW if window is None else max(1, int(window))
    buf = _vertical_filter.get(side)
    if buf is None or buf.maxlen != win:
        buf = deque(buf or (), maxlen=win)
        _vertical_filter[side] = buf
    buf.append(push)
    return float(np.median(buf))


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
    tared = w[:3] - baseline
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
        "push": get_push_force(side, robot),
        "blue_button": bool(state["blue_button"]),
        "green_button": bool(state["green_button"]),
    }


def _print_row(row: dict) -> None:
    logger.info(
        "[{side}] push={push:.3f} N  force={force:.3f} N  "
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