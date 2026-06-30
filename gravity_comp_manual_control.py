#!/usr/bin/env python3
"""
Compliant manual control of the left arm via current-based admittance control.

The Vega arms have no zero-torque/gravity-compensation mode: setting a joint to
"disable" engages its brake (locks it), and releasing the brake drops the arm.
Instead, this script keeps the arm in position mode (so it actively holds against
gravity) and makes it *feel* compliant: it reads each joint's current, treats the
offset from the resting current as the external force you apply by pushing, and
nudges the position setpoint accordingly at high rate. The arm follows your hand
(free lead-through) while never going limp.

While running, joint positions and velocities are printed at 10 Hz.

Safety: User must confirm before starting. The arm stays actively controlled, but
move slowly and keep the e-stop within reach.
"""

import os
import sys
import time

import numpy as np
from dexcomm.utils import RateLimiter
from loguru import logger

from dexcontrol.robot import Robot

CONTROL_HZ = 200.0
KD_GAIN = 0.3
# Per-joint current deadband (A): offset below this is treated as noise, not a push.
# Proximal joints have large motors (higher idle current noise → higher threshold);
# distal/wrist joints have small motors (even a gentle push = big current → lower).
# Tune each joint individually if it still feels sticky or drifts on its own.
#   index:  0     1     2     3     4     5     6
#           shoulder    elbow       wrist
FORCE_THRESHOLDS = np.array([0.20, 0.20, 0.15, 0.15, 0.08, 0.08, 0.08])
PRINT_HZ = 10.0
# Time constant (s) for the adaptive current baseline. The baseline tracks the
# pose-dependent gravity-hold current so only the fast push from your hand is
# detected, keeping push effort symmetric in every direction. Larger = the
# gravity bias is removed more slowly (asymmetry returns); smaller = sustained
# pushes fade faster as the baseline catches up. ~0.5 s is a good start.
BASELINE_TAU = 0.5


def check_environment():
    """Verify ZENOH_CONFIG and ROBOT_NAME are set."""
    if not os.environ.get('ZENOH_CONFIG'):
        logger.error("ZENOH_CONFIG not set. Please set it in ~/.bashrc")
        sys.exit(1)
    if not os.environ.get('ROBOT_NAME'):
        logger.error("ROBOT_NAME not set. Please set it in ~/.bashrc")
        sys.exit(1)


def prompt_safety_confirmation():
    """Prompt user for safety confirmation before enabling compliant control."""
    logger.warning("=" * 70)
    logger.warning("SAFETY CHECK: The left arm will become COMPLIANT (lead-through).")
    logger.warning("It stays actively controlled, but will follow your hand when pushed.")
    logger.warning("Move slowly and keep the e-stop within reach!")
    logger.warning("=" * 70)

    response = input("Start compliant manual control? (yes/no): ").strip().lower()
    if response != "yes":
        logger.info("Cancelled by user")
        sys.exit(0)


def joint_force_from_current(current_cur, current_baseline):
    """Estimate external joint force from the current offset, with deadband.

    Returns a 7-element array; joints whose |offset| is below FORCE_THRESHOLDS
    contribute zero force. Sign is inverted so the arm yields in the push direction.
    """
    offset = current_cur - current_baseline
    force = np.where(np.abs(offset) < FORCE_THRESHOLDS, 0.0, -offset)
    return force


def manual_control_loop(arm):
    """Run current-based admittance control on the left arm until Ctrl+C."""
    dt = 1.0 / CONTROL_HZ
    # Damping matrix; force is integrated through D^-1 to a velocity correction.
    D = np.diag([2.0] * 7) * KD_GAIN
    D_inv = np.linalg.inv(D)

    # Adaptive baseline that tracks the pose-dependent gravity-hold current.
    baseline_current = np.array(arm.get_joint_current())
    baseline_alpha = np.exp(-dt / BASELINE_TAU)
    rate_limiter = RateLimiter(rate_hz=CONTROL_HZ)
    print_every = max(1, int(round(CONTROL_HZ / PRINT_HZ)))

    logger.info(f"Compliant control started at {CONTROL_HZ:.0f} Hz. Press Ctrl+C to stop.")
    print("\nTime(s) | Joint_1 | Joint_2 | Joint_3 | Joint_4 | Joint_5 | Joint_6 | Joint_7 | "
          "Vel_1 | Vel_2 | Vel_3 | Vel_4 | Vel_5 | Vel_6 | Vel_7")
    print("-" * 150)

    start_time = time.time()
    iteration = 0

    try:
        while True:
            joint_pos = np.array(arm.get_joint_pos())
            joint_vel = np.array(arm.get_joint_vel())
            joint_current = np.array(arm.get_joint_current())

            # Zero-force lead-through: setpoint is driven purely by external force.
            ext_force = joint_force_from_current(joint_current, baseline_current)
            pos_correction = (D_inv @ ext_force) * dt
            arm.set_joint_pos((joint_pos + pos_correction).tolist())

            # Slowly track the gravity-hold current so the push stays symmetric.
            baseline_current = (
                baseline_alpha * baseline_current + (1.0 - baseline_alpha) * joint_current
            )

            if iteration % print_every == 0:
                elapsed = time.time() - start_time
                positions = [f"{p:7.3f}" for p in joint_pos]
                velocities = [f"{v:5.2f}" for v in joint_vel]
                print(f"{elapsed:7.2f} | {' | '.join(positions)} | {' | '.join(velocities)}")

            iteration += 1
            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info(f"\nManual control stopped after {iteration} iterations")


def main():
    check_environment()

    logger.info(f"Connecting to robot: {os.environ['ROBOT_NAME']}")
    bot = Robot()
    logger.info("Robot connected successfully")

    try:
        prompt_safety_confirmation()
        manual_control_loop(bot.left_arm)

    except Exception as e:
        logger.error(f"Error during operation: {type(e).__name__}: {e}")
        sys.exit(1)

    finally:
        logger.info("Stopping commands; arm holds its current position.")
        bot.shutdown()
        logger.info("Script completed")


if __name__ == "__main__":
    main()
