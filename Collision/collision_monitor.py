#!/usr/bin/env python3
"""Two-layer model-based collision monitor for the Dexmate Vega left arm.

Compares the measured joint signal (current [A] or torque [Nm]) against a
calibrated gravity + Coulomb-friction model:

    pred_j = k_j * tau_j(q; m) + c_j * sign(v_j) + b_j

(calibration from ``calibrate_gravity_model.py`` / ``calibration_left.json``)
and watches two complementary residuals:

Layer A — CHANGE of residual (impact detector, payload-robust):
    excess_A = | res(t) - res(t - window) |  with  res = signal - pred
    Slow model errors — a grasped object, friction drift, calibration aging —
    cancel in the difference (a +1 kg held object shifts this signal by
    <0.03 A), so this layer keeps working while carrying things. If either
    endpoint of the window is stopped/reversing, ±c is tolerated (friction
    sign flips are legitimate jumps). Blind spot: slowly building contact.

Layer B — ABSOLUTE residual (sustained-press detector):
    excess_B = max(0, |res| - band),  band = ±c while the joint is stopped.
    Catches slow pushes and stalls that Layer A cannot see, since it measures
    accumulated magnitude, not rate. This layer IS sensitive to payload —
    after grasping an object call :meth:`retare_payload` (or
    :meth:`set_extra_payload`) to re-baseline it.

A collision is declared when either layer exceeds its thresholds on
``n_joints_required`` joints for ``n_consecutive`` consecutive polls; both
arms are then frozen. ``trigger_info`` records which layer fired.

This monitor covers the LEFT arm only (that is what the calibration file
describes). Run the calibration for the right arm to extend it.

Usage as a library::

    from collision_monitor import CollisionMonitor

    bot = Robot()
    monitor = CollisionMonitor(bot)
    monitor.start()            # background daemon thread
    ...                        # motion code; poll monitor.triggered
    monitor.retare_payload()   # after grasping an object (arm MOVING slowly)
    ...
    monitor.stop()

Standalone::

    python collision_monitor.py             # active monitor (freezes on trigger)
    python collision_monitor.py --dry-run   # print live residuals, never freezes
"""

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger

from demo_move_left_ee import LeftArmIK, check_environment

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_left.json")


def banded_excess(
    y: np.ndarray, pred: np.ndarray, v: np.ndarray, c: np.ndarray, v_eps: float
) -> np.ndarray:
    """Layer B residual: |signal - pred| beyond the stiction band.

    Moving joints get no band (friction is already in pred via sign(v));
    stopped joints tolerate ±c around the gravity prediction.
    """
    band = np.where(np.abs(v) <= v_eps, np.abs(c), 0.0)
    return np.maximum(0.0, np.abs(y - pred) - band)


def change_excess(
    res_now: np.ndarray,
    res_old: np.ndarray,
    v_now: np.ndarray,
    v_old: np.ndarray,
    c: np.ndarray,
    v_eps: float,
) -> np.ndarray:
    """Layer A residual: |Δres| over the window, beyond the reversal band.

    Windows whose endpoints are both moving get no band; windows that cross a
    stop or reversal tolerate ±c (the friction sign flip is a legitimate jump).
    """
    moving_both = (np.abs(v_now) > v_eps) & (np.abs(v_old) > v_eps)
    band = np.where(moving_both, 0.0, np.abs(c))
    return np.maximum(0.0, np.abs(res_now - res_old) - band)


class CollisionMonitor:
    """Background daemon that stops the arms when the left arm hits something.

    Parameters
    ----------
    bot:
        Live ``Robot`` instance already connected to the hardware.
    calib_path:
        Path to the calibration JSON from ``calibrate_gravity_model.py``.
    thresholds:
        Layer B per-joint thresholds on the absolute banded residual (shape
        ``(7,)`` or scalar). Defaults to ``banded_suggested_thresholds`` from
        the calibration file.
    change_thresholds:
        Layer A per-joint thresholds on the residual change over the window.
        Defaults to ``change_suggested_thresholds`` from the calibration file.
    change_window:
        Layer A window length in seconds. Default: calibration's
        ``change_window_s`` (0.1 s). Longer windows catch slower impacts but
        react later.
    n_joints_required:
        How many joints must exceed a layer's threshold simultaneously. Default 1.
    n_consecutive:
        How many consecutive polls the condition must hold before triggering
        (rejects single-sample sensor spikes). Default 2.
    poll_hz:
        Poll rate of the background thread. Default 50 Hz.
    warmup:
        Seconds after start() during which no trigger is raised — zenoh
        subscribers wake from idle-pause with a brief data gap that would
        otherwise cause a spurious residual spike. Default 0.5 s.
    enable_change / enable_absolute:
        Enable Layer A / Layer B. Both on by default.
    freeze_on_trigger:
        If True (default) both arms are frozen when a collision is declared.
    on_collision:
        Optional callback ``fn(excess: np.ndarray) -> None`` invoked right
        before the freeze command.
    """

    def __init__(
        self,
        bot,
        calib_path: str = CALIB_PATH,
        thresholds: np.ndarray | float | None = None,
        change_thresholds: np.ndarray | float | None = None,
        change_window: float | None = None,
        n_joints_required: int = 1,
        n_consecutive: int = 2,
        poll_hz: float = 50.0,
        warmup: float = 0.5,
        enable_change: bool = True,
        enable_absolute: bool = True,
        freeze_on_trigger: bool = True,
        on_collision: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._bot = bot
        with open(calib_path) as f:
            calib = json.load(f)
        self._use_torque = calib["signal"] == "torque"
        self._v_eps = float(calib.get("v_eps", 0.03))
        self._m = float(calib["payload_mass_kg"])
        self._m_extra = 0.0
        self._k = np.asarray(calib["k"], dtype=float)
        self._c = np.asarray(calib["c"], dtype=float)
        self._d = np.asarray(calib.get("d", np.zeros(7)), dtype=float)  # viscous (older calibrations: 0)
        self._b = np.asarray(calib["b"], dtype=float)
        rms = np.asarray(calib.get("moving_rms", np.ones(7)), dtype=float)
        # Weights for payload re-estimation: trust low-noise identifiable joints.
        ident = np.asarray(calib.get("identifiable", [True] * 7), dtype=bool)
        self._retare_w = np.where(ident, 1.0 / np.maximum(rms, 1e-3) ** 2, 0.0)

        def _resolve(value, key, fallback):
            if value is None:
                if key in calib:
                    value = np.asarray(calib[key], dtype=float)
                    logger.info(f"CollisionMonitor: {key} from calibration: {np.round(value, 3)}")
                else:
                    value = fallback
                    logger.warning(f"CollisionMonitor: '{key}' missing in calibration — "
                                   f"using fallback {np.round(fallback, 3)}")
            return np.full(7, value, dtype=float) if np.isscalar(value) \
                else np.asarray(value, dtype=float)

        self._thresholds = _resolve(thresholds, "banded_suggested_thresholds", 6.0 * rms)
        self._change_thresholds = _resolve(change_thresholds, "change_suggested_thresholds",
                                           1.5 * self._thresholds)

        if change_window is None:
            change_window = float(calib.get("change_window_s", 0.1))
        self._poll_dt = 1.0 / poll_hz
        self._window_ticks = max(1, int(round(change_window / self._poll_dt)))

        # Gravity model: bare URDF + (+1 kg payload basis) — same construction
        # as the calibration script, so predictions match it exactly.
        from calibrate_gravity_model import build_payload_model
        ik = LeftArmIK()
        self._model = ik.model
        self._data_u = self._model.createData()
        self._model_p, self._data_p = build_payload_model(ik)
        self._left_qidx = [self._model.idx_qs[i] for i in ik.left_idx]
        self._torso_qidx = [self._model.idx_qs[i] for i in ik.torso_idx]
        self._vidx = [self._model.idx_vs[i] for i in ik.left_idx]
        self._q_tmpl = np.clip(
            pin.neutral(self._model),
            self._model.lowerPositionLimit,
            self._model.upperPositionLimit,
        )

        self._n_required = n_joints_required
        self._n_consecutive = n_consecutive
        self._warmup_ticks = int(round(warmup / self._poll_dt))
        self._tick = 0
        self._enable_change = enable_change
        self._enable_absolute = enable_absolute
        self._freeze_on_trigger = freeze_on_trigger
        self._on_collision = on_collision

        self._running = False
        self._triggered = False
        self._retaring = False
        self._streak_abs = 0
        self._streak_chg = 0
        # History of (res, v) for the change layer.
        self._hist: deque = deque(maxlen=self._window_ticks + 1)
        self._last_excess = np.zeros(7)
        self._last_change_excess = np.zeros(7)
        self.trigger_info: dict | None = None
        self._thread: threading.Thread | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitor thread."""
        if self._running:
            logger.warning("CollisionMonitor is already running.")
            return
        self.reset()
        self._tick = 0  # re-run the warmup on every start
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CollisionMonitor")
        self._thread.start()
        logger.info(
            f"CollisionMonitor started  (abs={np.round(self._thresholds, 2)}, "
            f"change={np.round(self._change_thresholds, 2)} @ "
            f"{self._window_ticks * self._poll_dt * 1000:.0f} ms, "
            f"n_joints={self._n_required}, n_consecutive={self._n_consecutive}, "
            f"poll={1 / self._poll_dt:.0f} Hz)"
        )

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("CollisionMonitor stopped.")

    def reset(self) -> None:
        """Clear the triggered flag and history (re-arms the monitor)."""
        self._triggered = False
        self._streak_abs = 0
        self._streak_chg = 0
        self._hist.clear()
        self.trigger_info = None

    @property
    def triggered(self) -> bool:
        """True if a collision has been detected since the last reset."""
        return self._triggered

    @property
    def last_excess(self) -> np.ndarray:
        """Layer B (absolute) excess at the most recent poll."""
        return self._last_excess.copy()

    @property
    def last_change_excess(self) -> np.ndarray:
        """Layer A (change) excess at the most recent poll."""
        return self._last_change_excess.copy()

    def set_extra_payload(self, mass_kg: float) -> None:
        """Set the grasped-object mass [kg] directly (0 after release)."""
        self._m_extra = float(mass_kg)
        # The payload step changes the residual level; clear the change-layer
        # history so it is not read as an impact.
        self._hist.clear()
        self._streak_abs = 0
        self._streak_chg = 0
        logger.info(f"CollisionMonitor: extra payload set to {self._m_extra:.2f} kg")

    def retare_payload(self, duration: float = 3.0, min_moving_samples: int = 25) -> float:
        """Estimate the grasped-object mass from the residual and re-baseline.

        IMPORTANT: the arm must be MOVING while this samples. At standstill
        stiction (±c) carries the extra weight, so the current shows no load
        and the estimate comes out near zero. Only samples where a joint is
        in the kinetic regime (|v| > v_eps) are used — call this while the
        arm executes a slow stroke (the demo runs one automatically on 'r').

        Collision detection is SUSPENDED while sampling (an unmodeled payload
        makes the stiction-to-motor load transfer at motion onset look like an
        impact, which would abort the very retare meant to fix it) — keep the
        stroke path clear. If too few moving samples were seen, the previous
        payload estimate is kept unchanged. The change-layer history is
        cleared afterwards so the prediction step from the new payload value
        is not itself mistaken for an impact.

        Returns:
            Estimated extra mass [kg] (also applied to the prediction).
        """
        logger.info("CollisionMonitor: retaring — collision detection suspended, keep the path clear.")
        self._retaring = True
        try:
            n = max(1, int(duration / self._poll_dt))
            num = 0.0
            den = 0.0
            n_moving = 0
            for _ in range(n):
                left_q, torso_q, v, y = self._read()
                moving = np.abs(v) > self._v_eps
                if moving.any():
                    g_u, phi = self._gravity(left_q, torso_q)
                    sgn = np.where(moving, np.sign(v), 0.0)
                    pred_base = (self._k * (g_u + self._m * phi)
                                 + self._c * sgn + self._d * v + self._b)
                    res = y - pred_base
                    kphi = self._k * phi
                    w = self._retare_w * moving  # stationary joints carry no load info
                    num += float(np.sum(w * res * kphi))
                    den += float(np.sum(w * kphi * kphi))
                    n_moving += 1
                time.sleep(self._poll_dt)
            if n_moving < min_moving_samples or den < 1e-9:
                logger.warning(
                    f"CollisionMonitor: retare saw only {n_moving} moving samples — "
                    f"estimate unreliable, keeping extra payload {self._m_extra:.2f} kg. "
                    f"Retare again while the arm is moving."
                )
                return self._m_extra
            self._m_extra = num / den
            logger.info(f"CollisionMonitor: retared — extra payload {self._m_extra:.2f} kg "
                        f"({n_moving} moving samples)")
            return self._m_extra
        finally:
            # Drop stale history: the payload step changes the residual level,
            # which the change layer would otherwise read as an impact.
            self._hist.clear()
            self._streak_abs = 0
            self._streak_chg = 0
            self._retaring = False

    def residuals(self) -> dict[str, np.ndarray]:
        """One-shot residual computation (for threshold tuning / dry runs)."""
        left_q, torso_q, v, y = self._read()
        pred = self._predict(left_q, torso_q, v)
        return {
            "signal": y,
            "pred": pred,
            "vel": v,
            "excess": banded_excess(y, pred, v, self._c, self._v_eps),
            "change_excess": self._last_change_excess.copy(),
        }

    # ── model ─────────────────────────────────────────────────────────────────

    def _gravity(self, left_q: np.ndarray, torso_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """URDF gravity torque and +1 kg payload basis at the given state."""
        q = self._q_tmpl.copy()
        for j, qi in enumerate(self._torso_qidx):
            q[qi] = torso_q[j]
        for j, qi in enumerate(self._left_qidx):
            q[qi] = left_q[j]
        q = np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)
        g_u = pin.computeGeneralizedGravity(self._model, self._data_u, q)[self._vidx]
        g_p = pin.computeGeneralizedGravity(self._model_p, self._data_p, q)[self._vidx]
        return g_u, g_p - g_u

    def _predict(self, left_q: np.ndarray, torso_q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Model-predicted signal for the given joint state."""
        g_u, phi = self._gravity(left_q, torso_q)
        tau = g_u + (self._m + self._m_extra) * phi
        sgn = np.where(np.abs(v) > self._v_eps, np.sign(v), 0.0)
        return self._k * tau + self._c * sgn + self._d * v + self._b

    # ── internal ──────────────────────────────────────────────────────────────

    def _read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arm = self._bot.left_arm
        left_q = arm.get_joint_pos().astype(float)
        torso_q = self._bot.torso.get_joint_pos().astype(float)
        v = arm.get_joint_vel().astype(float)
        y = (arm.get_joint_torque() if self._use_torque
             else arm.get_joint_current()).astype(float)
        return left_q, torso_q, v, y

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
        if self._retaring:
            return  # detection suspended during payload re-estimation

        left_q, torso_q, v, y = self._read()
        pred = self._predict(left_q, torso_q, v)
        res = y - pred

        # Layer B — absolute banded residual (suspended during retare)
        excess_abs = banded_excess(y, pred, v, self._c, self._v_eps)
        self._last_excess = excess_abs
        if self._enable_absolute and not self._retaring:
            if np.sum(excess_abs > self._thresholds) >= self._n_required:
                self._streak_abs += 1
            else:
                self._streak_abs = 0
        else:
            self._streak_abs = 0

        # Layer A — change of residual over the window
        excess_chg = np.zeros(7)
        if self._enable_change and len(self._hist) == self._hist.maxlen:
            res_old, v_old = self._hist[0]
            excess_chg = change_excess(res, res_old, v, v_old, self._c, self._v_eps)
        self._last_change_excess = excess_chg
        if self._enable_change:
            if np.sum(excess_chg > self._change_thresholds) >= self._n_required:
                self._streak_chg += 1
            else:
                self._streak_chg = 0
        self._hist.append((res, v))

        self._tick += 1
        if self._tick <= self._warmup_ticks:
            self._streak_abs = 0
            self._streak_chg = 0
            return

        layer = None
        if self._streak_abs >= self._n_consecutive:
            layer, excess, thr = "absolute", excess_abs, self._thresholds
        elif self._streak_chg >= self._n_consecutive:
            layer, excess, thr = "change", excess_chg, self._change_thresholds
        if layer is None:
            return

        self._triggered = True
        self.trigger_info = {"layer": layer, "excess": excess.copy()}
        logger.warning(
            f"CollisionMonitor: COLLISION on left arm ({layer} layer)! "
            f"joints={np.where(excess > thr)[0] + 1}  "
            f"excess={np.round(excess, 3)}  thresholds={np.round(thr, 3)}"
        )
        if self._on_collision is not None:
            try:
                self._on_collision(excess)
            except Exception as cb_exc:
                logger.warning(f"CollisionMonitor: on_collision callback raised {cb_exc}")
        if self._freeze_on_trigger:
            self._freeze_arms()

    def _freeze_arms(self) -> None:
        """Hold both arms at their current joint positions."""
        try:
            left_pos = self._bot.left_arm.get_joint_pos().astype(float)
            right_pos = self._bot.right_arm.get_joint_pos().astype(float)
            self._bot.set_joint_pos({
                "left_arm": left_pos,
                "right_arm": right_pos,
            })
            logger.warning("CollisionMonitor: arms frozen at current position.")
        except Exception as exc:
            logger.error(f"CollisionMonitor: freeze failed — {exc}")


# ── standalone entry point ─────────────────────────────────────────────────────

def main(dry_run: bool = False, print_hz: float = 5.0) -> None:
    """Run the monitor standalone until Ctrl+C.

    Args:
        dry_run: If True, never freezes the arms — just prints the live
            residuals of both layers and their running peaks (useful for
            threshold tuning: push the arm and watch what a contact produces).
        print_hz: Print rate in dry-run mode.
    """
    from dexcontrol.robot import Robot

    check_environment()
    logger.info("Connecting to robot...")
    bot = Robot()
    monitor = CollisionMonitor(bot, freeze_on_trigger=not dry_run)

    try:
        if dry_run:
            logger.info("Dry run: printing residuals (no freeze). Push the arm to test.")
            monitor._enable_absolute = False  # streaks unused; residuals() only
            peaks_abs = np.zeros(7)
            peaks_chg = np.zeros(7)
            while True:
                left_q, torso_q, v, y = monitor._read()
                pred = monitor._predict(left_q, torso_q, v)
                res = y - pred
                excess_abs = banded_excess(y, pred, v, monitor._c, monitor._v_eps)
                excess_chg = np.zeros(7)
                if len(monitor._hist) == monitor._hist.maxlen:
                    res_old, v_old = monitor._hist[0]
                    excess_chg = change_excess(res, res_old, v, v_old, monitor._c, monitor._v_eps)
                monitor._hist.append((res, v))
                peaks_abs = np.maximum(peaks_abs, excess_abs)
                peaks_chg = np.maximum(peaks_chg, excess_chg)
                logger.info(f"abs={np.round(excess_abs, 2)} peak={np.round(peaks_abs, 2)}  "
                            f"chg={np.round(excess_chg, 2)} peak={np.round(peaks_chg, 2)}")
                time.sleep(1.0 / print_hz)
        else:
            monitor.start()
            logger.info("Monitoring... Ctrl+C to stop.")
            while not monitor.triggered:
                time.sleep(0.2)
            logger.warning("Collision detected — arms held. Ctrl+C to exit.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        if not dry_run:
            monitor.stop()
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
