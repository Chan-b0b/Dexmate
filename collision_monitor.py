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

"""Background collision monitor for Dexmate Vega arms.

Detects collisions by requiring BOTH conditions to hold simultaneously for any joint:
  1. Current spike   — |current[j] - baseline[j]| > current_threshold[j]
  2. Velocity change — |vel[j] - prev_vel[j]|     > vel_change_threshold

This conjunction avoids false positives:
  - Smooth motion    : low current delta, low vel change  → no trigger
  - Collision impact : current spike + sudden vel change  → TRIGGER
  - At rest          : low current delta, no vel change   → no trigger

Each signal also has an independent hard threshold that triggers a stop on its own.

Usage::

    from collision_monitor import CollisionMonitor

    bot = Robot()
    monitor = CollisionMonitor(bot)
    monitor.start()           # starts background daemon thread

    # ... run your motion code ...

    monitor.stop()            # stop monitoring
"""

import threading
import time
from collections.abc import Callable

import numpy as np
from loguru import logger

import config


class CollisionMonitor:
    """Background daemon that stops the arms on collision detection.

    Parameters
    ----------
    bot:
        Live ``Robot`` instance already connected to the hardware.
    current_thresholds:
        Per-joint current-delta thresholds [A] (shape ``(7,)``).
        A joint is considered "overloaded" when
        ``|current - baseline| > threshold``.
        Defaults to ``0.8 A`` for every joint.
    current_hard_thresholds:
        Per-joint absolute current-delta thresholds [A] (shape ``(7,)``).
        If ``|current - baseline| > hard_threshold`` on *any* joint, a
        collision is declared immediately regardless of velocity.
        Defaults to ``None`` (disabled).
    vel_change_thresholds:
        Per-joint soft velocity-change thresholds [rad/s] (shape ``(7,)``).
        Used in the combined condition: ``|vel[t] - vel[t-1]| > threshold``.
        Defaults to ``0.3 rad/s`` for every joint.
    vel_change_hard_thresholds:
        Per-joint hard velocity-change thresholds [rad/s] (shape ``(7,)``).
        If exceeded on any joint, triggers a stop regardless of current.
        Defaults to ``None`` (disabled).
    n_joints_required:
        How many joints must simultaneously satisfy both combined conditions
        before a collision is declared.  Default ``1`` (any single joint).
    poll_hz:
        How often the background thread polls the robot state.  Default 50 Hz.
    on_collision:
        Optional callback ``fn(side: str, joint_mask: np.ndarray) -> None``
        called (from the monitor thread) right before the freeze command is
        sent.  Use to log, raise a flag, or trigger custom behaviour.
    """

    def __init__(
        self,
        bot,
        current_thresholds: np.ndarray | float = 0.8,
        current_hard_thresholds: np.ndarray | float | None = None,
        vel_change_thresholds: np.ndarray | float = 0.3,
        vel_change_hard_thresholds: np.ndarray | float | None = None,
        n_joints_required: int = 1,
        poll_hz: float = 50.0,
        on_collision: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        self._bot = bot
        self._current_thresholds = np.full(7, current_thresholds, dtype=float) \
            if np.isscalar(current_thresholds) else np.asarray(current_thresholds, dtype=float)
        self._current_hard_thresholds: np.ndarray | None = None
        if current_hard_thresholds is not None:
            self._current_hard_thresholds = (
                np.full(7, current_hard_thresholds, dtype=float)
                if np.isscalar(current_hard_thresholds)
                else np.asarray(current_hard_thresholds, dtype=float)
            )
        self._vel_change_thresholds = np.full(7, vel_change_thresholds, dtype=float) \
            if np.isscalar(vel_change_thresholds) else np.asarray(vel_change_thresholds, dtype=float)
        self._vel_change_hard_thresholds: np.ndarray | None = None
        if vel_change_hard_thresholds is not None:
            self._vel_change_hard_thresholds = (
                np.full(7, vel_change_hard_thresholds, dtype=float)
                if np.isscalar(vel_change_hard_thresholds)
                else np.asarray(vel_change_hard_thresholds, dtype=float)
            )
        self._n_required = n_joints_required
        self._poll_dt = 1.0 / poll_hz
        self._on_collision = on_collision

        self._running = False
        self._triggered = False
        self._thread: threading.Thread | None = None

        # Baseline currents measured at start()
        self._baseline: dict[str, np.ndarray] = {}
        # Previous velocities for delta computation (initialised in start())
        self._prev_vel: dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Measure baseline currents and start the background monitor thread."""
        if self._running:
            logger.warning("CollisionMonitor is already running.")
            return

        logger.info("CollisionMonitor: measuring baseline currents …")
        self._baseline = {
            "left":  self._bot.left_arm.get_joint_current().astype(float),
            "right": self._bot.right_arm.get_joint_current().astype(float),
        }
        logger.info(
            f"  left  baseline  [A]: {np.round(self._baseline['left'],  3)}"
        )
        logger.info(
            f"  right baseline  [A]: {np.round(self._baseline['right'], 3)}"
        )
        self._prev_vel = {
            "left":  self._bot.left_arm.get_joint_vel().astype(float),
            "right": self._bot.right_arm.get_joint_vel().astype(float),
        }

        self._triggered = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CollisionMonitor")
        self._thread.start()
        cur_hard_str = (
            f", current_hard={self._current_hard_thresholds}"
            if self._current_hard_thresholds is not None else ""
        )
        vel_hard_str = (
            f", vel_change_hard={self._vel_change_hard_thresholds}"
            if self._vel_change_hard_thresholds is not None else ""
        )
        logger.info(
            f"CollisionMonitor started  "
            f"(current={self._current_thresholds}{cur_hard_str}, "
            f"vel_change={self._vel_change_thresholds}{vel_hard_str}, "
            f"n_joints={self._n_required}, "
            f"poll={1/self._poll_dt:.0f} Hz)"
        )

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("CollisionMonitor stopped.")

    def reset(self) -> None:
        """Re-measure baseline and clear the triggered flag (re-arms the monitor)."""
        was_running = self._running
        self.stop()
        self._triggered = False
        if was_running:
            self.start()

    def update_baseline(self) -> None:
        """Re-sample baseline currents from the current arm pose.

        Call this at the start of each motion so thresholds are relative to
        the gravity-compensation load at the new configuration.
        """
        self._baseline = {
            "left":  self._bot.left_arm.get_joint_current().astype(float),
            "right": self._bot.right_arm.get_joint_current().astype(float),
        }
        self._prev_vel = {
            "left":  self._bot.left_arm.get_joint_vel().astype(float),
            "right": self._bot.right_arm.get_joint_vel().astype(float),
        }

    @property
    def triggered(self) -> bool:
        """True if a collision has been detected since the last reset."""
        return self._triggered

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._check()
            except Exception as exc:
                logger.warning(f"CollisionMonitor: read error — {exc}")
            time.sleep(self._poll_dt)

    def _check(self) -> None:
        if self._triggered:
            return  # already stopped; wait for reset()

        for side in ("left", "right"):
            arm = self._bot.left_arm if side == "left" else self._bot.right_arm

            current  = arm.get_joint_current().astype(float)
            vel      = arm.get_joint_vel().astype(float)
            baseline = self._baseline[side]
            prev_vel = self._prev_vel[side]

            cur_delta = np.abs(current - baseline)      # current offset from baseline
            vel_delta = np.abs(vel - prev_vel)          # velocity change since last poll

            self._prev_vel[side] = vel.copy()           # update for next cycle

            current_spike = cur_delta > self._current_thresholds       # (7,) bool
            vel_changed   = vel_delta > self._vel_change_thresholds    # (7,) bool

            collision_joints = current_spike & vel_changed  # BOTH combined conditions

            # Hard current threshold: triggers on its own
            if self._current_hard_thresholds is not None:
                hard_cur = cur_delta > self._current_hard_thresholds
                if np.any(hard_cur):
                    self._triggered = True
                    logger.warning(
                        f"CollisionMonitor: HARD current threshold exceeded on {side} arm! "
                        f"joints={np.where(hard_cur)[0] + 1}  "
                        f"Δcurrent={np.round(current - baseline, 3)}  "
                        f"Δvel={np.round(vel_delta, 3)}"
                    )
                    if self._on_collision is not None:
                        try:
                            self._on_collision(side, hard_cur)
                        except Exception as cb_exc:
                            logger.warning(f"CollisionMonitor: on_collision callback raised {cb_exc}")
                    self._freeze_arms()
                    return

            # Hard velocity-change threshold: triggers on its own
            if self._vel_change_hard_thresholds is not None:
                hard_vel = vel_delta > self._vel_change_hard_thresholds
                if np.any(hard_vel):
                    self._triggered = True
                    logger.warning(
                        f"CollisionMonitor: HARD velocity change threshold exceeded on {side} arm! "
                        f"joints={np.where(hard_vel)[0] + 1}  "
                        f"Δcurrent={np.round(current - baseline, 3)}  "
                        f"Δvel={np.round(vel_delta, 3)}"
                    )
                    if self._on_collision is not None:
                        try:
                            self._on_collision(side, hard_vel)
                        except Exception as cb_exc:
                            logger.warning(f"CollisionMonitor: on_collision callback raised {cb_exc}")
                    self._freeze_arms()
                    return

            if np.sum(collision_joints) >= self._n_required:
                self._triggered = True
                logger.warning(
                    f"CollisionMonitor: COLLISION detected on {side} arm! "
                    f"joints={np.where(collision_joints)[0] + 1}  "
                    f"Δcurrent={np.round(current - baseline, 3)}  "
                    f"Δvel={np.round(vel_delta, 3)}"
                )
                if self._on_collision is not None:
                    try:
                        self._on_collision(side, collision_joints)
                    except Exception as cb_exc:
                        logger.warning(f"CollisionMonitor: on_collision callback raised {cb_exc}")
                self._freeze_arms()
                return  # stop checking after first collision

    def _freeze_arms(self) -> None:
        """Hold both arms at their current joint positions."""
        try:
            left_pos  = self._bot.left_arm.get_joint_pos().astype(float)
            right_pos = self._bot.right_arm.get_joint_pos().astype(float)
            self._bot.set_joint_pos({
                "left_arm":  left_pos,
                "right_arm": right_pos,
            })
            logger.warning("CollisionMonitor: arms frozen at current position.")
        except Exception as exc:
            logger.error(f"CollisionMonitor: freeze failed — {exc}")
