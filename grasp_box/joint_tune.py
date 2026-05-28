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

"""Keyboard controller for all 7 joints of both arms in symmetric mirror mode.

Both arms move together: left gets +delta, right gets −delta (mirrored).

Controls:
    q / a  — j1  + / −    (shoulder pitch,      Y-axis)
    w / s  — j2  + / −    (shoulder abduction,  Z-axis)
    e / d  — j3  + / −    (shoulder roll,        X-axis)
    t / g  — j4  + / −    (elbow)
    y / h  — j5  + / −    (wrist 1)
    u / j  — j6  + / −    (wrist 2)
    i / k  — j7  + / −    (wrist 3)
    [ / ]  — decrease / increase step size
    r      — reset to starting pose
    Ctrl+C — quit

Usage:
    python joint_tune.py
    python joint_tune.py --step 0.05
    python joint_tune.py --adjust-duration 0.2
"""

import os
import sys
import termios
import tty
import time

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ik_pink import build_ik_context


# ── Keyboard ─────────────────────────────────────────────────────────────────

def get_key() -> str:
    """Read one keypress (raw). Escape sequences returned as-is."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return '\x1b[' + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Motion execution ─────────────────────────────────────────────────────────

def move_to_joints(
    bot,
    current_left_q: np.ndarray,
    current_right_q: np.ndarray,
    target_left_q: np.ndarray,
    target_right_q: np.ndarray,
    duration: float,
    control_dt: float = config.CONTROL_DT,
) -> None:
    """Smoothstep interpolation to target joint positions."""
    n_steps = max(1, int(duration / control_dt))
    for step in range(n_steps):
        _t = (step + 1) / n_steps
        alpha = _t * _t * (3 - 2 * _t)  # smoothstep
        bot.set_joint_pos({
            'left_arm':  current_left_q  + alpha * (target_left_q  - current_left_q),
            'right_arm': current_right_q + alpha * (target_right_q - current_right_q),
        })
        time.sleep(control_dt)


# ── Display ──────────────────────────────────────────────────────────────────

def print_status(
    left_q: np.ndarray,
    right_q: np.ndarray,
    step_rad: float,
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> None:
    def limit_marker(val, lo, hi):
        if val <= lo + 0.01:
            return "▼"
        if val >= hi - 0.01:
            return "▲"
        return " "

    def fmt_row(q, lower, upper, indices):
        return ",  ".join(
            f"{np.rad2deg(q[i]):>+9.2f}°{limit_marker(q[i], lower[i], upper[i])}"
            for i in indices
        )

    W = 82
    print(f"\n{'─'*W}")
    print(f" Step : {np.rad2deg(step_rad):.1f}°  ({step_rad:.4f} rad)")
    # j1-j4
    print(f"        {'j1(pitch)':>11}  {'j2(abduct)':>11}  {'j3(roll)':>11}  {'j4(elbow)':>11}")
    print(f" Left : {fmt_row(left_q,  left_lower,  left_upper,  [0, 1, 2, 3])}")
    print(f" Right: {fmt_row(right_q, right_lower, right_upper, [0, 1, 2, 3])}")
    # j5-j7
    print(f"        {'j5(wrist1)':>11}  {'j6(wrist2)':>11}  {'j7(wrist3)':>11}")
    print(f" Left : {fmt_row(left_q,  left_lower,  left_upper,  [4, 5, 6])}")
    print(f" Right: {fmt_row(right_q, right_lower, right_upper, [4, 5, 6])}")
    print(f" (▲/▼ = at joint limit)")
    # JOINT format for copy-paste into trajectory files
    def fmt_joint(q):
        return ", ".join(f"{v:+.4f}" for v in q) + ", JOINT"
    print(f" {fmt_joint(left_q)}")
    print(f" {fmt_joint(right_q)}")
    print(f"{'─'*W}")
    print(f" [q/a] j1±  [w/s] j2±  [e/d] j3±  [t/g] j4±  |  [[/]] step  [r] reset  [Ctrl+C] quit")
    print(f" [y/h] j5±  [u/j] j6±  [i/k] j7±")
    print(f"{'─'*W}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    step: float = 0.05,
    adjust_duration: float = 0.5,
) -> None:
    """Keyboard controller for all 7 joints of both arms (symmetric mirror mode).

    Args:
        step: Joint angle step per keypress in radians.
        adjust_duration: Duration (s) for each incremental movement.
    """
    ctx = build_ik_context(skip_confirmation=False)
    bot = ctx.bot
    model = ctx.model

    # Precompute per-joint position limits for left and right arm j1-j7
    left_lower  = np.array([model.lowerPositionLimit[model.idx_qs[ctx.left_arm_indices[j]]]  for j in range(7)])
    left_upper  = np.array([model.upperPositionLimit[model.idx_qs[ctx.left_arm_indices[j]]]  for j in range(7)])
    right_lower = np.array([model.lowerPositionLimit[model.idx_qs[ctx.right_arm_indices[j]]] for j in range(7)])
    right_upper = np.array([model.upperPositionLimit[model.idx_qs[ctx.right_arm_indices[j]]] for j in range(7)])

    try:
        # Read live joint state
        current_left_q  = bot.left_arm.get_joint_pos().astype(float)
        current_right_q = bot.right_arm.get_joint_pos().astype(float)
        start_left_q    = current_left_q.copy()
        start_right_q   = current_right_q.copy()

        logger.info("Starting from live robot joint state.")
        logger.info(
            f"Joint limits (left arm)  "
            + "  ".join(
                f"j{j+1}: [{np.rad2deg(left_lower[j]):.1f}°, {np.rad2deg(left_upper[j]):.1f}°]"
                for j in range(7)
            )
        )
        print_status(current_left_q, current_right_q, step, left_lower, left_upper, right_lower, right_upper)

        while True:
            key = get_key()

            if key == '\x03':  # Ctrl+C
                logger.info("Quitting...")
                break

            # ── Step size ─────────────────────────────────────────────────
            elif key == '[':
                step = max(0.01, round(step - 0.01, 4))
                print_status(current_left_q, current_right_q, step, left_lower, left_upper, right_lower, right_upper)
                continue
            elif key == ']':
                step = min(0.50, round(step + 0.01, 4))
                print_status(current_left_q, current_right_q, step, left_lower, left_upper, right_lower, right_upper)
                continue

            # ── Reset ─────────────────────────────────────────────────────
            elif key == 'r':
                logger.info("Resetting to starting pose...")
                move_to_joints(bot, current_left_q, current_right_q,
                               start_left_q, start_right_q, adjust_duration * 4)
                current_left_q  = start_left_q.copy()
                current_right_q = start_right_q.copy()
                print_status(current_left_q, current_right_q, step, left_lower, left_upper, right_lower, right_upper)
                continue

            # ── Joint adjustments (mirrored: left +delta, right −delta) ──
            else:
                delta = np.zeros(7)
                if   key == 'q':  delta[0] = +step   # j1 +
                elif key == 'a':  delta[0] = -step   # j1 −
                elif key == 'w':  delta[1] = +step   # j2 +
                elif key == 's':  delta[1] = -step   # j2 −
                elif key == 'e':  delta[2] = +step   # j3 +
                elif key == 'd':  delta[2] = -step   # j3 −
                elif key == 't':  delta[3] = +step   # j4 +
                elif key == 'g':  delta[3] = -step   # j4 −
                elif key == 'y':  delta[4] = +step   # j5 +
                elif key == 'h':  delta[4] = -step   # j5 −
                elif key == 'u':  delta[5] = +step   # j6 +
                elif key == 'j':  delta[5] = -step   # j6 −
                elif key == 'i':  delta[6] = +step   # j7 +
                elif key == 'k':  delta[6] = -step   # j7 −
                else:
                    continue

                target_left_q  = current_left_q.copy()
                target_right_q = current_right_q.copy()
                # j4 (index 3) is not mirrored — both arms move in the same direction
                mirror = np.array([-1.0 if i != 3 else 1.0 for i in range(7)])
                target_left_q[:7]  = np.clip(current_left_q[:7]  + delta[:7],          left_lower,  left_upper)
                target_right_q[:7] = np.clip(current_right_q[:7] + mirror * delta[:7], right_lower, right_upper)

                move_to_joints(bot, current_left_q, current_right_q,
                               target_left_q, target_right_q, adjust_duration)
                current_left_q  = target_left_q.copy()
                current_right_q = target_right_q.copy()

                print_status(current_left_q, current_right_q, step, left_lower, left_upper, right_lower, right_upper)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down robot connection")
        ctx.bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
