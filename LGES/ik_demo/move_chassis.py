"""Chassis left/right strafe movement control for ik_demo.

Simple interface for testing and controlling chassis sideways (strafe) movement.
Uses open-loop distance/time control: distance_m = speed * time.

Examples:
    Interactive mode (speeds up to 0.5 m/s):
        $ python -m ik_demo.move_chassis

    Move left 1 meter (at configured speed):
        $ python -m ik_demo.move_chassis --left 1.0

    Move right 0.5 meter:
        $ python -m ik_demo.move_chassis --right 0.5

    Custom speed + distance (speed: m/s, will override time calculation):
        $ python -m ik_demo.move_chassis --left 1.0 --speed 0.15
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import tyro
from loguru import logger

try:
    from . import config as cfg
except ImportError:
    import config as cfg


def strafe_left(bot, distance_m: float = None, speed: float = None) -> None:
    """Move chassis to the left (positive sideways velocity).

    Args:
        bot: Robot instance (from Robot() context).
        distance_m: Distance in meters. If provided, calculates time = distance / speed.
                   Overrides cfg.CHASSIS_STRAFE_TIME_S.
        speed: Speed in m/s. If None, uses cfg.CHASSIS_STRAFE_SPEED_MS.
    """
    if speed is None:
        speed = cfg.CHASSIS_STRAFE_SPEED_MS

    if distance_m is not None:
        wait_time = distance_m / speed
        logger.info("Strafe LEFT: {} m @ {:.3f} m/s = {:.1f} s", distance_m, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        distance_m = speed * wait_time
        logger.info("Strafe LEFT: {:.3f} m/s x {:.1f} s = {} m", speed, wait_time, distance_m)

    bot.chassis.move_sideways(speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def strafe_right(bot, distance_m: float = None, speed: float = None) -> None:
    """Move chassis to the right (negative sideways velocity).

    Args:
        bot: Robot instance (from Robot() context).
        distance_m: Distance in meters. If provided, calculates time = distance / speed.
                   Overrides cfg.CHASSIS_STRAFE_TIME_S.
        speed: Speed in m/s. If None, uses cfg.CHASSIS_STRAFE_SPEED_MS.
    """
    if speed is None:
        speed = cfg.CHASSIS_STRAFE_SPEED_MS

    if distance_m is not None:
        wait_time = distance_m / speed
        logger.info("Strafe RIGHT: {} m @ {:.3f} m/s = {:.1f} s", distance_m, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        distance_m = speed * wait_time
        logger.info("Strafe RIGHT: {:.3f} m/s x {:.1f} s = {} m", speed, wait_time, distance_m)

    bot.chassis.move_sideways(-speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def move_forward(bot, distance_m: float = None, speed: float = None) -> None:
    """Move chassis forward (positive straight velocity).

    Args:
        bot: Robot instance (from Robot() context).
        distance_m: Distance in meters. If provided, calculates time = distance / speed.
                   Overrides cfg.CHASSIS_STRAFE_TIME_S.
        speed: Speed in m/s. If None, uses cfg.CHASSIS_STRAFE_SPEED_MS.
    """
    if speed is None:
        speed = cfg.CHASSIS_STRAFE_SPEED_MS

    if distance_m is not None:
        wait_time = distance_m / speed
        logger.info("Move FORWARD: {} m @ {:.3f} m/s = {:.1f} s", distance_m, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        distance_m = speed * wait_time
        logger.info("Move FORWARD: {:.3f} m/s x {:.1f} s = {} m", speed, wait_time, distance_m)

    bot.chassis.move_straight(speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def move_backward(bot, distance_m: float = None, speed: float = None) -> None:
    """Move chassis backward (negative straight velocity).

    Args:
        bot: Robot instance (from Robot() context).
        distance_m: Distance in meters. If provided, calculates time = distance / speed.
                   Overrides cfg.CHASSIS_STRAFE_TIME_S.
        speed: Speed in m/s. If None, uses cfg.CHASSIS_STRAFE_SPEED_MS.
    """
    if speed is None:
        speed = cfg.CHASSIS_STRAFE_SPEED_MS

    if distance_m is not None:
        wait_time = distance_m / speed
        logger.info("Move BACKWARD: {} m @ {:.3f} m/s = {:.1f} s", distance_m, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        distance_m = speed * wait_time
        logger.info("Move BACKWARD: {:.3f} m/s x {:.1f} s = {} m", speed, wait_time, distance_m)

    bot.chassis.move_straight(-speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def turn_ccw(bot, angle_deg: float = None, speed: float = None) -> None:
    """Turn the chassis in place counter-clockwise (yaw left).

    Args:
        bot: Robot instance (from Robot() context).
        angle_deg: Turn angle in DEGREES. If provided, time = angle_rad / speed.
        speed: Angular speed in rad/s. If None, uses cfg.CHASSIS_TURN_SPEED_RADS.
    """
    import numpy as np
    if speed is None:
        speed = cfg.CHASSIS_TURN_SPEED_RADS
    if angle_deg is not None:
        wait_time = abs(np.deg2rad(angle_deg)) / speed
        logger.info("Turn CCW: {} deg @ {:.2f} rad/s = {:.1f} s", angle_deg, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        logger.info("Turn CCW: {:.2f} rad/s x {:.1f} s = {:.1f} deg",
                    speed, wait_time, np.rad2deg(speed * wait_time))
    bot.chassis.turn(speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def turn_cw(bot, angle_deg: float = None, speed: float = None) -> None:
    """Turn the chassis in place clockwise (yaw right). Args as turn_ccw."""
    import numpy as np
    if speed is None:
        speed = cfg.CHASSIS_TURN_SPEED_RADS
    if angle_deg is not None:
        wait_time = abs(np.deg2rad(angle_deg)) / speed
        logger.info("Turn CW: {} deg @ {:.2f} rad/s = {:.1f} s", angle_deg, speed, wait_time)
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        logger.info("Turn CW: {:.2f} rad/s x {:.1f} s = {:.1f} deg",
                    speed, wait_time, np.rad2deg(speed * wait_time))
    bot.chassis.turn(-speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def interactive_mode(bot) -> None:
    """Interactive CLI for chassis movement testing.

    User can input commands to move left/right with various speeds.
    """
    logger.info("=" * 60)
    logger.info("INTERACTIVE CHASSIS MOVEMENT")
    logger.info("=" * 60)
    logger.info("Available speeds: 0.1, 0.15, 0.2, 0.3, 0.5 m/s")
    logger.info("Default (config): {:.3f} m/s x {:.1f} s",
                cfg.CHASSIS_STRAFE_SPEED_MS, cfg.CHASSIS_STRAFE_TIME_S)
    logger.info("")
    logger.info("Commands:")
    logger.info("  l              - strafe LEFT (default speed/time)")
    logger.info("  r              - strafe RIGHT (default speed/time)")
    logger.info("  f              - move FORWARD (default speed/time)")
    logger.info("  b              - move BACKWARD (default speed/time)")
    logger.info("  l|r|f|b <distance>          - move <distance_m> at default speed")
    logger.info("  l|r|f|b <distance> <speed>  - move <distance_m> at <speed> m/s")
    logger.info("  tl|tr <angle_deg> [rad_s]   - turn in place CCW|CW by <angle_deg>")
    logger.info("  q              - quit")
    logger.info("=" * 60)
    logger.warning("Make sure the strafe path is CLEAR before moving!")
    logger.warning("Be ready to e-stop if needed.")
    logger.warning("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip().lower().split()

            if not user_input:
                continue

            cmd = user_input[0]

            if cmd == "q":
                logger.info("Exiting interactive mode")
                break

            elif cmd == "l":
                # Parse: l [distance] [speed]
                if len(user_input) == 1:
                    strafe_left(bot)
                elif len(user_input) == 2:
                    distance = float(user_input[1])
                    strafe_left(bot, distance_m=distance)
                elif len(user_input) == 3:
                    distance = float(user_input[1])
                    speed = float(user_input[2])
                    strafe_left(bot, distance_m=distance, speed=speed)
                else:
                    logger.warning("Invalid format: l [distance] [speed]")

            elif cmd == "r":
                # Parse: r [distance] [speed]
                if len(user_input) == 1:
                    strafe_right(bot)
                elif len(user_input) == 2:
                    distance = float(user_input[1])
                    strafe_right(bot, distance_m=distance)
                elif len(user_input) == 3:
                    distance = float(user_input[1])
                    speed = float(user_input[2])
                    strafe_right(bot, distance_m=distance, speed=speed)
                else:
                    logger.warning("Invalid format: r [distance] [speed]")

            elif cmd in ("f", "b"):
                # Parse: f|b [distance] [speed]
                fn = move_forward if cmd == "f" else move_backward
                if len(user_input) == 1:
                    fn(bot)
                elif len(user_input) == 2:
                    fn(bot, distance_m=float(user_input[1]))
                elif len(user_input) == 3:
                    fn(bot, distance_m=float(user_input[1]), speed=float(user_input[2]))
                else:
                    logger.warning("Invalid format: {} [distance] [speed]", cmd)

            elif cmd in ("tl", "tr"):
                # Parse: tl|tr [angle_deg] [speed_rad_s]
                fn = turn_ccw if cmd == "tl" else turn_cw
                if len(user_input) == 1:
                    fn(bot)
                elif len(user_input) == 2:
                    fn(bot, angle_deg=float(user_input[1]))
                elif len(user_input) == 3:
                    fn(bot, angle_deg=float(user_input[1]), speed=float(user_input[2]))
                else:
                    logger.warning("Invalid format: {} [angle_deg] [speed_rad_s]", cmd)

            else:
                logger.warning("Unknown command: {}. Try 'l', 'r', 'f', 'b', 'tl', 'tr', or 'q'.", cmd)

        except ValueError as e:
            logger.warning("Parse error: {}. Expected: l|r [distance] [speed]", e)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.warning("Error: {}", e)


def main(
    left: float = None,
    right: float = None,
    speed: float = None,
    interactive: bool = False,
) -> None:
    """Chassis left/right strafe movement control.

    Args:
        left: Distance in meters to strafe LEFT. Mutually exclusive with --right.
        right: Distance in meters to strafe RIGHT. Mutually exclusive with --left.
        speed: Custom speed in m/s (overrides config default).
               Applies to whatever movement is requested.
        interactive: Enter interactive mode for repeated commands.
    """
    from dexcontrol.robot import Robot

    # Validate args
    if left is not None and right is not None:
        logger.error("Cannot specify both --left and --right")
        sys.exit(1)

    if interactive and (left is not None or right is not None):
        logger.warning("Ignoring --left/--right; entering interactive mode")

    logger.warning("=" * 60)
    logger.warning("CHASSIS STRAFE (LEFT/RIGHT MOVEMENT)")
    logger.warning("=" * 60)
    logger.warning("Ensure the strafe path is CLEAR.")
    logger.warning("Be ready to press E-STOP if needed.")
    logger.warning("=" * 60)

    if input("Continue? [y/N]: ").strip().lower() != "y":
        logger.info("Aborted")
        return

    with Robot() as bot:
        if interactive or (left is None and right is None):
            # Interactive mode
            interactive_mode(bot)

        elif left is not None:
            strafe_left(bot, distance_m=left, speed=speed)
            logger.info("Left strafe complete: {} m", left)

        elif right is not None:
            strafe_right(bot, distance_m=right, speed=speed)
            logger.info("Right strafe complete: {} m", right)


if __name__ == "__main__":
    tyro.cli(main)
