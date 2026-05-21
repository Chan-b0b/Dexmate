#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Rotate the chassis by a fixed angle.

Usage:
    python rotate.py --rotate left    # 90° counter-clockwise
    python rotate.py --rotate right   # 90° clockwise
    python rotate.py --rotate back    # 180°
    python rotate.py --rotate left --speed 0.5
"""

import math
from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    rotate: Literal["left", "right", "back"] = "left",
    speed: float = 0.3,
) -> None:
    """Rotate the chassis by a preset angle.

    Args:
        rotate: Direction to rotate — 'left' (90° CCW), 'right' (90° CW),
            or 'back' (180°).
        speed: Angular velocity in rad/s. Defaults to 0.3.
    """
    angle_map = {
        "left":  +math.pi / 2,   # +90° counter-clockwise
        "right": -math.pi / 2,   # −90° clockwise
        "back":  +math.pi,       # 180°
    }

    angle = angle_map[rotate] * 1.15
    duration = abs(angle) / abs(speed)
    actual_speed = math.copysign(speed, angle)

    logger.info(
        f"Rotating {rotate}: {math.degrees(angle):.0f}° "
        f"at {actual_speed:.2f} rad/s for {duration:.2f}s"
    )

    with Robot() as bot:
        bot.chassis.turn(actual_speed, wait_time=duration)

    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
