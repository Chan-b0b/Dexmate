"""Chassis left/right strafe movement control for ik_demo.

Simple interface for testing and controlling chassis sideways (strafe) movement.
Uses open-loop distance/time control: distance_m = speed * time.

DISTANCE-based legs (distance_m / angle_deg given) are calibrated against two
dexcontrol timing quirks (see _pre_steer): the wheels are pre-steered to the
leg's direction before the timed drive starts, the -1.0 s the library clips
off every streamed command is added back (cfg.CHASSIS_CMD_DEAD_TIME_S), and an
explicit zero-velocity stop is sent after the leg. The legacy speed*time legs
(no distance given) are left untouched — their tuned times already absorb
those quirks.

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

import numpy as np
import tyro
from loguru import logger

try:
    from . import config as cfg
except ImportError:
    import config as cfg

# dexcontrol clips every timed command's streaming window by this FIXED amount
# (chassis.py _execute_timed_command: duration = max(wait_time - 1.0, 0)).
_LIB_CMD_CLAMP_S = 1.0


def _stop(bot) -> None:
    """Actively stop the chassis: STREAM zero velocity (current steering kept)
    for CHASSIS_STOP_STREAM_S instead of a single shot. A single stop command
    was observed unreliable — the chassis coasted ~0.7 s past the leg
    (over-traveling ~7 cm at 0.1 m/s), i.e. the firmware kept the latched
    velocity until its own watchdog. Streaming zeros brakes deterministically."""
    bot.chassis.set_wheel_velocity(
        0.0, wait_time=_LIB_CMD_CLAMP_S + cfg.CHASSIS_STOP_STREAM_S)


def _pre_steer(bot, vx: float, vy: float, wz: float = 0.0) -> np.ndarray | None:
    """Point both wheels at the leg's steering angles BEFORE the timed drive,
    and return the per-wheel signed speed that matches the angle we picked.

    dexcontrol's own sequential steering is supposed to hold a pure-steer
    command for 1.0 s first, but that hold passes through the same
    `max(wait_time - 1.0, 0)` clamp as every timed command and resolves to
    0 s — the drive command lands while the wheels are still pivoting, so part
    of the leg is spent off-axis (the observed strafe undershoot).

    Angles replicate chassis._compute_wheel_control exactly (note the NEGATED
    atan2): base angle or its pi-flip (the wheel-speed sign absorbs the flip),
    whichever is closer to the wheel's current angle. Any two legs 90 deg
    apart (e.g. strafe -> straight) leave a wheel sitting exactly on the
    bisector between that next leg's base/alt solutions — a dead tie that a
    hair of encoder noise can break either way. Calling the leg's own
    move_straight/move_sideways/turn afterwards re-derives base/alt from
    scratch and can land on the opposite side of that tie from what we just
    steered to, snapping the wheel 180 deg the instant the drive command
    lands. Returning our own choice here lets the caller drive with
    set_wheel_velocity (holds steering, no angle recompute) instead, so
    there's only ever one tie-break decision, not two that can disagree.
    Reads private chassis geometry attrs; if a dexcontrol update renames
    them, pre-steering is skipped with a warning (the time compensation
    still applies) rather than failing the leg.

    Returns:
        Per-wheel signed speed [left, right] for set_wheel_velocity, or None
        if pre-steering was skipped — callers should fall back to their
        normal move_*/turn call in that case.
    """
    ch = bot.chassis
    try:
        if wz != 0.0:
            scale = wz * ch._center_to_wheel_dist
            vecs = (scale * ch._left_wheel_ang_vel_vector, scale * ch._right_wheel_ang_vel_vector)
        else:
            vecs = (np.array([vx, vy]), np.array([vx, vy]))
        max_ang = ch._max_steering_angle
    except AttributeError as e:
        logger.warning("pre-steer skipped (dexcontrol chassis internals changed?): {}", e)
        return None
    target = []
    signed_speed = []
    for vec, cur in zip(vecs, ch.steering_angle):
        speed = float(np.linalg.norm(vec))
        base = float(-np.arctan2(vec[1], vec[0]))
        alt = base + np.pi if base < 0 else base - np.pi
        use_base = abs(base - cur) <= abs(alt - cur)
        target.append(float(np.clip(base if use_base else alt, -max_ang, max_ang)))
        signed_speed.append(speed if use_base else -speed)
    target = np.asarray(target)
    signed_speed = np.asarray(signed_speed)
    deadline = time.monotonic() + cfg.CHASSIS_PRESTEER_TIMEOUT_S
    while True:
        ch.set_steering_angle(target)   # re-send each poll: latched-target insurance
        err = float(np.max(np.abs(ch.steering_angle - target)))
        if err <= cfg.CHASSIS_PRESTEER_TOL_RAD:
            logger.info("pre-steer done: [{:+.2f}, {:+.2f}] rad", target[0], target[1])
            return signed_speed
        if time.monotonic() >= deadline:
            logger.warning("pre-steer timeout (err {:.3f} rad) — driving anyway", err)
            return signed_speed
        time.sleep(0.05)


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
        wait_time = distance_m / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Strafe LEFT: {} m @ {:.3f} m/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    distance_m, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, 0.0, speed)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.move_sideways(speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
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
        wait_time = distance_m / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Strafe RIGHT: {} m @ {:.3f} m/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    distance_m, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, 0.0, -speed)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.move_sideways(-speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
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
        wait_time = distance_m / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Move FORWARD: {} m @ {:.3f} m/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    distance_m, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, speed, 0.0)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.move_straight(speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
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
        wait_time = distance_m / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Move BACKWARD: {} m @ {:.3f} m/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    distance_m, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, -speed, 0.0)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.move_straight(-speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
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
    if speed is None:
        speed = cfg.CHASSIS_TURN_SPEED_RADS
    if angle_deg is not None:
        wait_time = abs(np.deg2rad(angle_deg)) / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Turn CCW: {} deg @ {:.2f} rad/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    angle_deg, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, 0.0, 0.0, wz=speed)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.turn(speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
    else:
        wait_time = cfg.CHASSIS_STRAFE_TIME_S
        logger.info("Turn CCW: {:.2f} rad/s x {:.1f} s = {:.1f} deg",
                    speed, wait_time, np.rad2deg(speed * wait_time))
        bot.chassis.turn(speed, wait_time=wait_time)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    logger.info("Settle complete")


def turn_cw(bot, angle_deg: float = None, speed: float = None) -> None:
    """Turn the chassis in place clockwise (yaw right). Args as turn_ccw."""
    if speed is None:
        speed = cfg.CHASSIS_TURN_SPEED_RADS
    if angle_deg is not None:
        wait_time = abs(np.deg2rad(angle_deg)) / speed + cfg.CHASSIS_CMD_DEAD_TIME_S
        logger.info("Turn CW: {} deg @ {:.2f} rad/s = {:.1f} s (incl. {:.1f} s cmd dead time)",
                    angle_deg, speed, wait_time, cfg.CHASSIS_CMD_DEAD_TIME_S)
        signed_speed = _pre_steer(bot, 0.0, 0.0, wz=-speed)
        if signed_speed is not None:
            bot.chassis.set_wheel_velocity(signed_speed, wait_time=wait_time)
        else:
            bot.chassis.turn(-speed, wait_time=wait_time)
        _stop(bot)   # streamed stop, steering kept
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
