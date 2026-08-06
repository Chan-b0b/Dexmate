"""Suction pick-and-place for ik_demo (built on arm.ArmMover).

Stage 1 — detect-and-freeze descent: descend vertically by streaming per-tick
IK via arm.set_joint_pos_vel (finite-diff velocity feedforward, ~100 Hz), and
stop on the tared vertical wrench force. Two-signal pick: force = contact,
DI0 vacuum = seal. Per-descent tare; separate hard-force limits for pick
(empty cup) and place (battery in cup). Two-speed profile: fast in free air,
slow creep in the contact zone.

Planned refinement (PLAN.md): replace the creep + seal-press with admittance
control (bounded contact force) once this is verified on the robot.

Force sensing uses the arm's native wrench_sensor (6-vector), tared and
projected onto base-vertical via the EE rotation — no external read_force.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    from . import config as cfg
    from .arm import ArmMover
    from .drivers import suction_io
    from .drivers.bcr import BackgroundScanner
except ImportError:  # allow `python suction.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover
    from drivers import suction_io
    from drivers.bcr import BackgroundScanner


def _axis_steps(step: float, max_v: float) -> list[float]:
    """0, +step, -step, +2*step, -2*step, ... out to max_v."""
    if step <= 0.0 or max_v <= 0.0:
        return [0.0]
    out = [0.0]
    v = step
    while v <= max_v + 1e-9:
        out.append(v)
        out.append(-v)
        v += step
    return out


@dataclass
class PickResult:
    success: bool
    reason: str                      # contact / sealed / force_limit / vacuum_timeout / max_descent / unreachable / misseat
    contact_ee_z: float | None = None
    barcode: str | None = None
    contact_info: dict | None = None  # tared base-frame wrench + cmd-vs-measured EE yaw at contact
    recover_attempts: int = 0         # misseat-recovery retries taken before this result
    final_yaw_rad: float | None = None  # wrist yaw after recovery re-orients (caller must release at this yaw)
    recover_history: list | None = None  # per-attempt recovery record (step taken + that contact's wrench)


class SuctionMover(ArmMover):
    """Suction pick/place on the suction arm (cfg.ARM_SIDE)."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side=cfg.ARM_SIDE, ee_frame=cfg.EE_FRAME)
        self._wrench = getattr(self._arm, "wrench_sensor", None)
        if self._wrench is None:
            logger.warning("[suction] {} arm has no wrench sensor — contact detection OFF", self._side)
        self._force_baseline = np.zeros(3)
        self._torque_baseline = np.zeros(3)

    def __enter__(self) -> "SuctionMover":
        return self

    def __exit__(self, *_) -> None:
        try:
            suction_io.suction_off()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Force sensing (native wrench, tared, projected to base-vertical)
    # ------------------------------------------------------------------
    def tare(self, n: int | None = None) -> None:
        """Capture the wrench baseline. Call with NO contact (cup free)."""
        if self._wrench is None:
            return
        n = int(n or cfg.TARE_SAMPLES)
        samples = []
        for _ in range(n):
            s = np.asarray(self._wrench.get_wrench_state(), dtype=float).ravel()
            if s.size < 6:                      # force-only transport: pad torques
                s = np.concatenate([s[:3], np.zeros(3)])
            samples.append(s[:6])
            time.sleep(0.005)
        base = np.mean(samples, axis=0)
        self._force_baseline, self._torque_baseline = base[:3], base[3:6]
        logger.info("[suction] tared (baseline |f|={:.2f}N |m|={:.3f}Nm)",
                    float(np.linalg.norm(self._force_baseline)),
                    float(np.linalg.norm(self._torque_baseline)))

    def vertical_force(self) -> float | None:
        """|base-vertical component| of the tared contact force (N), or None."""
        if self._wrench is None:
            return None
        raw = self._wrench.get_wrench_state()[:3].astype(float) - self._force_baseline
        return float(abs((self.current_ee_rotation() @ raw)[2]))

    def contact_wrench(self) -> "tuple[np.ndarray, np.ndarray] | None":
        """Tared 6-axis wrench rotated to the BASE frame: (force N, torque Nm),
        or None (no sensor / force-only transport). NOTE the sensor sits at the
        wrist, SUCTION_LENGTH above the cup tip — lateral tip forces lever into
        mx/my; verify signs from logged data before acting on them."""
        if self._wrench is None:
            return None
        s = np.asarray(self._wrench.get_wrench_state(), dtype=float).ravel()
        if s.size < 6:
            return None
        R = self.current_ee_rotation()
        return R @ (s[:3] - self._force_baseline), R @ (s[3:6] - self._torque_baseline)

    def _contact_snapshot(self, q_cmd: np.ndarray) -> "dict | None":
        """Diagnostics captured at the moment of a place contact: tared base
        wrench + commanded-vs-measured EE yaw (tracking error under load —
        checks the 'yaw drifts near the case' hypothesis with live data)."""
        info: dict = {}
        fm = self.contact_wrench()
        if fm is not None:
            f, mo = fm
            info.update(fx=float(f[0]), fy=float(f[1]), fz=float(f[2]),
                        mx=float(mo[0]), my=float(mo[1]), mz=float(mo[2]))
        try:
            _, eul_cmd = self.fk(np.asarray(q_cmd, dtype=float))
            _, eul_meas = self.fk(self._live_arm_q())
            d = (float(np.rad2deg(eul_meas[2] - eul_cmd[2])) + 180.0) % 360.0 - 180.0
            info.update(yaw_cmd_deg=float(np.rad2deg(eul_cmd[2])),
                        yaw_meas_deg=float(np.rad2deg(eul_meas[2])),
                        yaw_track_err_deg=d)
        except Exception:  # noqa: BLE001 — diagnostics must never kill a place
            pass
        return info or None

    # ------------------------------------------------------------------
    # Vertical descent (detect-and-freeze)
    # ------------------------------------------------------------------
    def _descent_speed(self, z: float, creep_z: float, elapsed: float) -> float:
        """Descent cup-tip speed with two smoothstep shapes and no velocity step:
        ramp IN from rest over DESCENT_RAMP_S (rest->descend handoff), and blend
        fast->creep over DESCENT_CREEP_BLEND_M above creep_z (so there's no jerk
        at the creep line). At/below creep_z the speed is the creep speed."""
        fast, creep = cfg.DESCENT_APPROACH_SPEED_M_S, cfg.DESCENT_CREEP_SPEED_M_S
        band = max(float(cfg.DESCENT_CREEP_BLEND_M), 1e-6)
        if z <= creep_z:
            base = creep
        elif z >= creep_z + band:
            base = fast
        else:
            f = (z - creep_z) / band            # 0 at creep_z -> 1 at band top
            f = f * f * (3.0 - 2.0 * f)          # smoothstep
            base = creep + (fast - creep) * f
        r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
        return base * (r * r * (3.0 - 2.0 * r))  # smoothstep ramp-in

    def _descend_to_contact(self, target_ee_z: float, rpy, force_limit: float,
                            start_q: np.ndarray, tick_cb=None) -> PickResult:
        """Descend straight down (x,y,rpy held) until contact / hard-limit / floor.

        Streams per-tick IK with a finite-diff velocity feedforward. Two-speed:
        fast until ``DESCENT_CREEP_GAP_M`` above the expected contact z, then a
        slow creep so one reaction tick can't over-press. Suction state is the
        caller's responsibility (off for pick approach).

        ``start_q`` is the approach move's commanded target joints: the descent
        continues the stream from exactly there (its FK pose, not a fresh live
        solve) so there's no command discontinuity at the handoff.

        ``tick_cb`` (optional supervisor hook, ik_VLM): called once per tick as
        ``tick_cb(ee_z, vertical_force_or_None)``; a truthy return halts the
        descent -> PickResult(False, "monitor_abort"). Checked AFTER the force
        branch, so a real contact / hard-limit tick still classifies as itself.
        """
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        creep_z = target_ee_z + cfg.DESCENT_CREEP_GAP_M
        descended = 0.0
        elapsed = 0.0

        def _halt(q):
            self._arm.set_joint_pos_vel(np.asarray(q), np.zeros(len(q)))

        while descended < cfg.DESCENT_MAX_M:
            speed = self._descent_speed(z, creep_z, elapsed)
            z_next = z - speed * dt
            sol = self.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q)
                return PickResult(False, "unreachable", z)
            self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)

            f = self.vertical_force()
            if f is not None:
                if f > force_limit:
                    _halt(sol.q)
                    logger.warning("[suction] hard push {:.1f}N at ee_z={:.4f} — abort", f, z_next)
                    return PickResult(False, "force_limit", z_next,
                                      contact_info=self._contact_snapshot(sol.q))
                if f > cfg.FORCE_CONTACT_THRESHOLD_N:
                    _halt(sol.q)
                    logger.info("[suction] contact {:.1f}N at ee_z={:.4f}", f, z_next)
                    return PickResult(True, "contact", z_next,
                                      contact_info=self._contact_snapshot(sol.q))
            if tick_cb is not None and tick_cb(z_next, f):
                _halt(sol.q)
                logger.warning("[suction] descent halted by the supervisor at ee_z={:.4f}", z_next)
                return PickResult(False, "monitor_abort", z_next,
                                  contact_info=self._contact_snapshot(sol.q))

            descended += (z - z_next)
            z, prev_q = z_next, sol.q
            elapsed += dt
            time.sleep(dt)

        _halt(prev_q)
        logger.warning("[suction] max descent ({:.2f}m) without contact", cfg.DESCENT_MAX_M)
        return PickResult(False, "max_descent", z)

    # ------------------------------------------------------------------
    # Pick / place
    # ------------------------------------------------------------------
    def _lift_to_transport(self, rpy, to_clear_only: bool = False) -> None:
        """Lift to SAFE_TRANSPORT_Z: straight up (xy held per tick) to
        LIFT_CLEAR_EE_Z — clear of the case walls from any layer's pick —
        then the remaining free-air ascent as a faster joint-space move_ee
        (endpoint xy held; the arc in between is harmless up there). Either
        leg falling short (the per-tick stream can dead-end on a diverging IK
        branch right at the reach boundary — observed 4 mm under LIFT_CLEAR at
        the place column) hands off to _best_effort_ascent.

        ``to_clear_only``: return right after the wall-clear vertical and skip
        the joint-space ascent — the caller overlaps the remaining rise with a
        chassis leg. Used for BOTH chassis legs: the empty-cup place return,
        and (user-verified 0806: a held part's bottom clears the box walls at
        LIFT_CLEAR height) the loaded pick->target leg."""
        pos, _ = self.current_ee_pose()
        z_clear = min(max(float(pos[2]), cfg.LIFT_CLEAR_EE_Z), cfg.SAFE_TRANSPORT_Z)
        q = self.move_ee_vertical(z_clear, rpy)
        if q is None:
            self._best_effort_ascent(rpy)
            return
        if to_clear_only or z_clear >= cfg.SAFE_TRANSPORT_Z:
            return
        x, y = self.fk(q)[0][:2]
        if self.move_ee([float(x), float(y), cfg.SAFE_TRANSPORT_Z], rpy, quiet=True) is None:
            self._best_effort_ascent(rpy)

    def _best_effort_ascent(self, rpy) -> None:
        """Recover as much transport height as possible after a lift leg fell
        short: scan z from SAFE_TRANSPORT_Z DOWN (DESCENT_CHECK_STEP_M steps)
        at the CURRENT xy with fresh min-motion solves from the halted config
        — the streamed per-tick chain dead-ends on one branch while a static
        solve can converge (descent_reachable proves these columns statically
        reachable) — and joint-move to the FIRST valid solution that gains at
        least LIFT_RECOVER_MIN_GAIN_M. Only runs with the EE already near or
        above LIFT_CLEAR_EE_Z (LIFT_RECOVER_MIN_CLEAR_M band): the recovery
        move is joint-space, so its EE arc must not happen down between the
        case walls."""
        pos, _ = self.current_ee_pose()
        x, y, z_now = float(pos[0]), float(pos[1]), float(pos[2])
        if z_now < cfg.LIFT_CLEAR_EE_Z - cfg.LIFT_RECOVER_MIN_CLEAR_M:
            logger.warning("[suction] lift fell short at z={:.4f} — below the wall-clear "
                           "band, staying put (no joint-space recovery)", z_now)
            return
        seed = self._live_arm_q()
        z = float(cfg.SAFE_TRANSPORT_Z)
        while z > z_now + cfg.LIFT_RECOVER_MIN_GAIN_M:
            sol = self.solve_pose((x, y, z), rpy, seed=seed, min_motion=True)
            if sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision:
                logger.info("[suction] best-effort ascent: z {:.4f} -> {:.4f} "
                            "(transport target {:.2f})", z_now, z, cfg.SAFE_TRANSPORT_Z)
                self.move_joints(sol.q)
                return
            z -= float(cfg.DESCENT_CHECK_STEP_M)
        logger.warning("[suction] best-effort ascent: no reachable z above {:.4f} at "
                       "xy=({:.3f},{:+.3f}) — staying", z_now, x, y)

    def _approach_and_hover(self, ee_pos, rpy, ez):
        """Travel to the column at transport height (sideways clearance), then
        drop straight to a REACHABLE hover over the true xy — as ONE blended
        stream (move_joints_through): the transport waypoint is crossed at
        blend speed instead of the old full stop. The hover arrival stays a
        full stop — the force tare and the descent stream both start from
        rest.

        The transport approach can fall a few mm short horizontally when the arm
        nears its reach limit up high; the lower hover (ez + HOVER_HEIGHT_M) is
        well inside reach and solved FROM the approach solution (the same
        branch the old move-then-solve-live chain landed on), so the arm
        recovers the true xy before the vertical descent (which holds xy) —
        preventing an offset, misaligned seat.
        Returns the aligned hover config, or None if either leg is unreachable."""
        approach = np.array([ee_pos[0], ee_pos[1], cfg.SAFE_TRANSPORT_Z])
        sol_app = self.solve_pose(approach, rpy, seed=self._live_arm_q(), min_motion=True)
        if sol_app.pos_err_m > cfg.REACH_TOL_M:  # transport leg: shortfall recovered by the hover
            logger.error("[arm] target unreachable ({:.1f}mm short) — not moving",
                         sol_app.pos_err_m * 1000)
            return None
        hover_z = min(cfg.SAFE_TRANSPORT_Z, ez + cfg.HOVER_HEIGHT_M)
        sol_hov = self.solve_pose([ee_pos[0], ee_pos[1], hover_z], rpy,
                                  seed=sol_app.q, min_motion=True)
        if sol_hov.pos_err_m > cfg.REACH_TOL_M:
            logger.error("[arm] target unreachable ({:.1f}mm short) — not moving",
                         sol_hov.pos_err_m * 1000)
            return None
        self.move_joints_through([sol_app.q, sol_hov.q])
        return sol_hov.q

    def pick(self, pose, expected_z=None, lift_to_clear: bool = False) -> PickResult:
        """Approach from transport, fast-descend (suction OFF) to creep_z, then
        suction ON and creep to contact + seal — suction is on THROUGH the creep,
        sealing on contact (same seal strategy as the gated pick, minus the scan).
        Ends at transport holding the part. ``expected_z`` overrides the taught
        contact z (layer stacking: the orchestrator passes the measured z)."""
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] pick: approach@transport -> hover -> descend -> suction on -> creep-seal")
        q_hover = self._approach_and_hover(ee_pos, rpy, ez)
        if q_hover is None:
            return PickResult(False, "unreachable")
        suction_io.suction_off()
        self.tare()  # empty cup, no contact
        creep_z = ez + cfg.DESCENT_CREEP_GAP_M
        _last_q, z, reason = self._descend_open(creep_z, rpy, q_hover, cfg.FORCE_HARD_LIMIT_N)
        if reason != "at_floor":
            return PickResult(False, reason, z)
        suction_io.suction_on()
        return self._seal_and_lift(rpy, lift_to_clear)

    def _seal_and_lift(self, rpy, lift_to_clear: bool = False) -> PickResult:
        """Suction already ON at creep height: creep-seal, then lift to
        transport on success (suction off on failure). Shared pick tail.
        ``lift_to_clear`` stops the lift at the wall-clear height (see
        _lift_to_transport)."""
        res = self._creep_seal(rpy, self._live_arm_q())
        if res.success:
            # Relieve the creep-contact press before lifting (mirrors place()'s
            # RELEASE_PRELIFT_M) — otherwise the lift's first motion has to break
            # the residual seat press while already carrying the part.
            if cfg.SEAL_PRELIFT_M > 0.0:
                pos, _ = self.current_ee_pose()
                self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
            # Lift straight up to clear, then to transport — ready to travel.
            self._lift_to_transport(rpy, to_clear_only=lift_to_clear)
        else:
            suction_io.suction_off()
        return res

    def pick_retreat(self, pose, expected_z=None, touch_n=10.0,
                     retreat_m=0.03) -> PickResult:
        """pick() variant (VLA collection default): suction ON from creep_z as
        usual, but instead of stopping at the seal, press on to ``touch_n``
        (tared vertical N), RETREAT ``retreat_m`` straight up, and HOVER there
        (no re-descent) until the vacuum grabs the case and seals — then lift
        immediately. The hover height is referenced to the LIVE touch z (the
        actual surface), not the pressed-in commanded z. A seal that latches
        during the press skips the retreat and lifts right away; no seal
        within VACUUM_SEAL_TIMEOUT_S fails the take ('vacuum_timeout')."""
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] pick_retreat: touch {:.1f}N -> +{:.0f}mm hover -> wait seal",
                    touch_n, retreat_m * 1e3)
        q_hover = self._approach_and_hover(ee_pos, rpy, ez)
        if q_hover is None:
            return PickResult(False, "unreachable")
        suction_io.suction_off()
        self.tare()  # empty cup, no contact
        creep_z = ez + cfg.DESCENT_CREEP_GAP_M
        _last_q, z, reason = self._descend_open(creep_z, rpy, q_hover, cfg.FORCE_HARD_LIMIT_N)
        if reason != "at_floor":
            return PickResult(False, reason, z)
        suction_io.suction_on()
        _q, z, reason = self._creep_to_force(rpy, self._live_arm_q(), touch_n)
        if reason not in ("touched", "sealed"):
            suction_io.suction_off()
            return PickResult(False, reason, z)
        if reason == "touched":
            z = float(self.current_ee_pose()[0][2]) + retreat_m  # live surface + gap
            q = self.move_ee_vertical(z, rpy)
            if q is None:
                suction_io.suction_off()
                return PickResult(False, "unreachable", z)
            # HOVER: hold here with suction on; the vacuum pulls the case film
            # up into the cup. No re-descent (a creep-seal here reads as an
            # up-down bounce and re-presses the case).
            vac = suction_io.VacuumMonitor(); vac.start()
            try:
                sealed = False
                deadline = time.time() + cfg.VACUUM_SEAL_TIMEOUT_S
                while time.time() < deadline:
                    self._arm.set_joint_pos_vel(q, np.zeros(len(q)))
                    if vac.is_sealed():
                        sealed = True
                        logger.info("[suction] sealed from hover at ee_z={:.4f}", z)
                        break
                    time.sleep(0.05)
            finally:
                threading.Thread(target=vac.stop, daemon=True).start()
            if not sealed:
                suction_io.suction_off()
                return PickResult(False, "vacuum_timeout", z)
        # Sealed (mid-press or from hover) -> lift immediately.
        if cfg.SEAL_PRELIFT_M > 0.0:
            pos, _ = self.current_ee_pose()
            self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
        self._lift_to_transport(rpy)
        return PickResult(True, "sealed", z)

    def place(self, pose, expected_z=None, misseat_tol_m=None, tick_cb=None,
              lift_to_clear: bool = False) -> PickResult:
        """Hover above the seat, descend to contact within buffer, release.
        On a failed descent the part is HELD (suction on) until the operator
        confirms the release — unreachable/max_descent can end mid-air, where
        an automatic blow-off drops the battery from height.
        ``expected_z`` overrides the taught seat z (layer stacking).
        ``misseat_tol_m``: contact more than this ABOVE ``expected_z`` means the
        part landed on the rim/jig instead of dropping into the seat (a proper
        seat sits 5-15mm lower) — held for the operator like a failed descent,
        instead of blindly releasing a misaligned part. Pass it only with a
        measured-anchored ``expected_z`` (the model plane drifts too much)."""
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] place: approach@transport -> hover -> descend -> release")
        q_hover = self._approach_and_hover(ee_pos, rpy, ez)
        if q_hover is None:
            return PickResult(False, "unreachable")
        self.tare()  # battery in cup, hovering (no contact yet)
        res = self._descend_to_contact(ez, rpy, cfg.FORCE_HARD_LIMIT_PLACE_N, q_hover,
                                       tick_cb=tick_cb)
        if res.reason == "monitor_abort":
            # the supervising layer (ik_VLM) owns the recovery — return with the
            # part still HELD (suction on): no release, no lift, no operator gate
            return res
        if (res.reason == "contact" and misseat_tol_m is not None
                and res.contact_ee_z is not None):
            above = res.contact_ee_z - ez
            if above > float(misseat_tol_m):
                logger.warning("[suction] contact {:+.1f}mm ABOVE the expected seat "
                               "(tol {:.0f}mm) — rim-landing, part NOT seated",
                               above * 1000.0, float(misseat_tol_m) * 1000.0)
                res.reason = "misseat"  # success recomputed from reason below
        if res.reason == "misseat" and int(cfg.PLACE_RECOVER_ATTEMPTS) > 0:
            res = self._misseat_recover(ez, rpy, float(misseat_tol_m), res)
        if res.final_yaw_rad is not None:
            # recovery re-oriented the wrist — release/prelift/lift at THAT yaw,
            # not the original one (rotating back while pressed would drag the part)
            rpy = (rpy[0], rpy[1], res.final_yaw_rad)
        if res.reason != "contact":
            logger.warning("[suction] place descent failed ({}) — holding the battery "
                           "(suction ON), waiting for the operator", res.reason)
            input(f"place-failed[{res.reason}]> hand-guide the part if needed; "
                  f"Enter to blow-off release + retreat (the run continues): ")
        pos, _ = self.current_ee_pose()
        if cfg.RELEASE_PRELIFT_M > 0.0:
            self.move_ee_vertical(pos[2] + cfg.RELEASE_PRELIFT_M, rpy)
        suction_io.release()
        # lift_to_clear: stop at the wall-clear height (cup EMPTY here) — the
        # caller starts the return chassis leg immediately and folds the
        # remaining rise into the parallel view park (both target z=1.10)
        self._lift_to_transport(rpy, to_clear_only=lift_to_clear)
        res.success = res.reason in ("contact",)
        return res

    def _misseat_recover(self, ez: float, rpy, tol: float, res: PickResult) -> PickResult:
        """Rim-landing recovery: the part is HELD (suction on) just above the
        seat with an error the system can't observe upstream. Per attempt,
        lift slightly and either
          - FORCE-GUIDED TRANSLATION (Phase 2): the misseat contact's tared
            lateral force points where the obstruction pushes the part — the
            free side (0806 L4 bat1: fx=-5.8N ≡ operator's "move -x"). If
            |f_lat| >= PLACE_RECOVER_FORCE_MIN_N and the XY excursion cap
            allows, step PLACE_RECOVER_XY_STEP_M along it (yaw kept), or
          - the next blind wrist-yaw pattern step (absorbs the
            staging-dependent in-hand twist constant trims can't track),
        then creep back down; success = a contact inside the seat band."""
        pos, _ = self.current_ee_pose()
        x, y = float(pos[0]), float(pos[1])
        x0, y0 = x, y
        yaw_steps = list(cfg.PLACE_RECOVER_YAW_PATTERN_RAD)   # no-mz blind fallback
        blind_xy = list(cfg.PLACE_RECOVER_BLIND_XY_M)         # no-force flat-landing fallback
        cur_off = 0.0 if res.final_yaw_rad is None else float(res.final_yaw_rad) - rpy[2]
        cur_yaw = rpy[2] + cur_off
        step = float(cfg.PLACE_RECOVER_YAW_STEP_RAD)          # adaptive state
        ydir, prev_mz = 0.0, None
        history: list[dict] = []
        for i in range(1, int(cfg.PLACE_RECOVER_ATTEMPTS) + 1):
            info = res.contact_info or {}
            yaw_now = rpy[2] if res.final_yaw_rad is None else float(res.final_yaw_rad)
            f_lat = (float(info.get("fx", 0.0)), float(info.get("fy", 0.0)))
            f_mag = float(np.hypot(*f_lat))
            room = np.hypot(x - x0, y - y0) + cfg.PLACE_RECOVER_XY_STEP_M \
                <= float(cfg.PLACE_RECOVER_XY_MAX_M) + 1e-9
            if "fx" in info and f_mag >= float(cfg.PLACE_RECOVER_FORCE_MIN_N) and room:
                mode = "force"
                dx = f_lat[0] / f_mag * float(cfg.PLACE_RECOVER_XY_STEP_M)
                dy = f_lat[1] / f_mag * float(cfg.PLACE_RECOVER_XY_STEP_M)
                x, y = x + dx, y + dy
                logger.warning("[suction] misseat recover {}/{}: contact {:+.1f}mm high, "
                               "f_lat=({:+.1f},{:+.1f})N — force-guided step "
                               "({:+.1f},{:+.1f})mm", i, int(cfg.PLACE_RECOVER_ATTEMPTS),
                               (res.contact_ee_z - ez) * 1000.0, f_lat[0], f_lat[1],
                               dx * 1000.0, dy * 1000.0)
            elif blind_xy:
                # no force signal — FLAT landing (nothing pushes sideways on a
                # flat top): walk the blind offset pattern before any yaw
                mode = "xy_blind"
                dx0, dy0 = blind_xy.pop(0)
                x, y = x0 + dx0, y0 + dy0   # ABSOLUTE offsets from the commanded pose
                logger.warning("[suction] misseat recover {}/{}: contact {:+.1f}mm high, "
                               "f_lat={:.1f}N (<{:.1f}N, flat landing) — blind step to "
                               "({:+.1f},{:+.1f})mm", i, int(cfg.PLACE_RECOVER_ATTEMPTS),
                               (res.contact_ee_z - ez) * 1000.0, f_mag,
                               float(cfg.PLACE_RECOVER_FORCE_MIN_N),
                               dx0 * 1000.0, dy0 * 1000.0)
            else:
                if "fx" in info and f_mag >= float(cfg.PLACE_RECOVER_FORCE_MIN_N) and not room:
                    logger.warning("[suction] misseat recover: XY cap {:.0f}mm reached — "
                                   "force still says ({:+.1f},{:+.1f})N but falling to yaw",
                                   float(cfg.PLACE_RECOVER_XY_MAX_M) * 1000.0,
                                   f_lat[0], f_lat[1])
                mz = info.get("mz")
                if mz is not None and abs(float(mz)) >= float(cfg.PLACE_RECOVER_MZ_MIN_NM):
                    # mz-feedback: compare this contact's torque with the
                    # previous one to decide the next rotation (see config)
                    mz = float(mz)
                    mode = "yaw_mz"
                    if ydir == 0.0:
                        ydir = 1.0 if mz > 0 else -1.0  # first move: toward the torque
                    elif prev_mz is not None:
                        if (mz > 0) != (prev_mz > 0):
                            ydir, step = -ydir, step * 0.5   # overshot: reverse + refine
                        elif abs(mz) > abs(prev_mz):
                            ydir = -ydir                     # wrong way: reverse
                        # else: |mz| shrinking — keep going
                    prev_mz = mz
                    new_off = float(np.clip(cur_off + ydir * step,
                                            -float(cfg.PLACE_RECOVER_YAW_MAX_RAD),
                                            float(cfg.PLACE_RECOVER_YAW_MAX_RAD)))
                    if abs(new_off - cur_off) < 1e-6:
                        if not yaw_steps:
                            break                # capped and no fallback left
                        mode = "yaw"
                        new_off = float(yaw_steps.pop(0))   # capped: blind step
                    cur_off = new_off
                elif yaw_steps:
                    mode = "yaw"
                    cur_off = float(yaw_steps.pop(0))
                else:
                    break  # no force/torque signal and the pattern is spent
                cur_yaw = rpy[2] + cur_off
                logger.warning("[suction] misseat recover {}/{}: contact {:+.1f}mm high, "
                               "f_lat={:.1f}N, mz={} ({}) — yaw to dyaw={:+.1f}deg "
                               "(step {:.1f}deg)", i, int(cfg.PLACE_RECOVER_ATTEMPTS),
                               (res.contact_ee_z - ez) * 1000.0, f_mag,
                               "n/a" if not isinstance(mz, float)
                               else "{:+.3f}Nm".format(mz),
                               mode, float(np.rad2deg(cur_off)),
                               float(np.rad2deg(step)))
            lift_z = float(res.contact_ee_z) + float(cfg.PLACE_RECOVER_LIFT_M)
            self.move_ee_vertical(lift_z, (rpy[0], rpy[1], yaw_now))  # lift at the AS-CONTACTED yaw
            rpy_i = (rpy[0], rpy[1], cur_yaw)
            q_up = self.move_ee([x, y, lift_z], rpy_i)  # re-pose at the lifted height
            if q_up is None:
                continue  # unreachable here — try the next step
            res2 = self._descend_to_contact(ez, rpy_i, cfg.FORCE_HARD_LIMIT_PLACE_N, q_up)
            res2.recover_attempts = i
            res2.final_yaw_rad = rpy_i[2]
            info2 = res2.contact_info or {}
            history.append({
                "attempt": i, "mode": mode,
                "dyaw_deg": float(np.rad2deg(rpy_i[2] - rpy[2])),
                "dx_mm": (x - x0) * 1000.0, "dy_mm": (y - y0) * 1000.0,
                "reason": res2.reason,
                "z_mm": (None if res2.contact_ee_z is None
                         else (res2.contact_ee_z - ez) * 1000.0),
                "fx": info2.get("fx"), "fy": info2.get("fy"), "mz": info2.get("mz"),
            })
            res2.recover_history = history
            if (res2.reason == "contact" and res2.contact_ee_z is not None
                    and (res2.contact_ee_z - ez) <= tol):
                logger.info("[suction] misseat recovered (attempt {}: dyaw={:+.1f}deg, "
                            "dxy=({:+.1f},{:+.1f})mm, contact {:+.1f}mm vs seat)", i,
                            float(np.rad2deg(rpy_i[2] - rpy[2])),
                            (x - x0) * 1000.0, (y - y0) * 1000.0,
                            (res2.contact_ee_z - ez) * 1000.0)
                return res2
            if res2.reason != "contact":
                # force_limit / unreachable mid-recovery — stop probing, operator gate
                return res2
            res2.reason = "misseat"
            res = res2
        logger.warning("[suction] misseat recovery exhausted — handing to the operator")
        return res

    # ------------------------------------------------------------------
    # Barcode-gated battery pick
    # ------------------------------------------------------------------
    def pick_gated(self, pose, case_center=None, expected_z=None,
                   lift_to_clear: bool = False) -> PickResult:
        """Battery pick with a barcode gate. Suction OFF, fast-descend to creep_z
        while scanning; then suction ON and creep to contact + seal. If nothing
        read by creep_z, sweep x/y at creep_z (battery's side of the case center,
        no lift/tilt) until it reads first. Exhausted -> grab anyway (no divert).
        ``expected_z`` overrides the taught contact z (layer stacking).
        Returns PickResult with .barcode."""
        case_center = cfg.SOURCE_CASE_CENTER if case_center is None else case_center
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] pick(gated): approach -> hover -> scan-descend -> gate -> seal")
        q_hover = self._approach_and_hover(ee_pos, rpy, ez)
        if q_hover is None:
            return PickResult(False, "unreachable")
        suction_io.suction_off()
        self.tare()

        creep_z = ez + cfg.DESCENT_CREEP_GAP_M
        scanner = BackgroundScanner().start()
        code = None
        try:
            last_q, z, reason = self._descend_open(creep_z, rpy, q_hover, cfg.FORCE_HARD_LIMIT_N)
            if reason != "at_floor":
                return PickResult(False, reason, z)
            code = scanner.result()
            if code is None:
                logger.info("[suction] no read by creep_z — sweeping")
                code = self._sweep_scan(ee_pos, rpy, scanner, case_center)
        finally:
            scanner.stop()
        logger.info("[suction] barcode: {!r}", code)

        suction_io.suction_on()
        res = self._creep_seal(rpy, self._live_arm_q())
        res.barcode = code
        if res.success:
            # Relieve the creep-contact press before lifting — see pick().
            if cfg.SEAL_PRELIFT_M > 0.0:
                pos, _ = self.current_ee_pose()
                self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
            self._lift_to_transport(rpy, to_clear_only=lift_to_clear)
        else:
            suction_io.suction_off()
        return res

    def _descend_open(self, z_floor, rpy, start_q, force_limit):
        """Descend (suction unchanged) straight down to z_floor — no soft-contact
        stop (the goal is the floor). Ramp speed in and decelerate to creep speed
        into z_floor (so the halt / creep-seal handoff isn't a velocity step);
        abort on force>force_limit. Returns (last_q, z, reason) with reason in
        {at_floor, force_limit, unreachable}."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        elapsed = 0.0

        def _halt(q):
            self._arm.set_joint_pos_vel(np.asarray(q), np.zeros(len(q)))

        while z > z_floor + 1e-4:
            speed = self._descent_speed(z, z_floor, elapsed)
            z_next = max(z_floor, z - speed * dt)
            sol = self.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q); return prev_q, z, "unreachable"
            self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
            f = self.vertical_force()
            if f is not None and f > force_limit:
                _halt(sol.q)
                logger.warning("[suction] hard push {:.1f}N at ee_z={:.4f} during scan-descent", f, z_next)
                return sol.q, z_next, "force_limit"
            z, prev_q = z_next, sol.q
            elapsed += dt
            time.sleep(dt)
        _halt(prev_q)
        return prev_q, z, "at_floor"

    def _creep_seal(self, rpy, start_q) -> PickResult:
        """Suction already ON. Creep straight down until the vacuum seals (DI0,
        primary) or force contact (then hold + wait for the seal). Abort on hard force."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        descended = 0.0
        vac = suction_io.VacuumMonitor(); vac.start()

        def _halt(q):
            self._arm.set_joint_pos_vel(np.asarray(q), np.zeros(len(q)))

        try:
            while descended < cfg.DESCENT_MAX_M:
                if vac.is_sealed():
                    _halt(prev_q)
                    logger.info("[suction] vacuum sealed at ee_z={:.4f}", z)
                    return PickResult(True, "sealed", z)
                z_next = z - cfg.DESCENT_CREEP_SPEED_M_S * dt
                sol = self.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
                if sol.pos_err_m > cfg.REACH_TOL_M:
                    _halt(prev_q); return PickResult(False, "unreachable", z)
                self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
                f = self.vertical_force()
                if f is not None:
                    if f > cfg.FORCE_HARD_LIMIT_N:
                        _halt(sol.q); return PickResult(False, "force_limit", z_next)
                    if f > cfg.FORCE_CONTACT_THRESHOLD_N:
                        _halt(sol.q)  # touched — hold and wait for the seal to form
                        deadline = time.time() + cfg.VACUUM_SEAL_TIMEOUT_S
                        while time.time() < deadline:
                            self._arm.set_joint_pos_vel(sol.q, np.zeros(len(sol.q)))
                            if vac.is_sealed():
                                logger.info("[suction] sealed after contact at ee_z={:.4f}", z_next)
                                return PickResult(True, "sealed", z_next)
                            time.sleep(0.05)
                        return PickResult(False, "vacuum_timeout", z_next)
                descended += (z - z_next); z, prev_q = z_next, sol.q
                time.sleep(dt)
            _halt(prev_q)
            return PickResult(False, "max_descent", z)
        finally:
            # Off the critical path: stop() blocks ~1.3s on the socketio
            # disconnect, which held the arm frozen between seal and lift
            # (the post-seal dwell visible in every collected take).
            threading.Thread(target=vac.stop, daemon=True).start()

    def _creep_to_force(self, rpy, start_q, touch_n: float):
        """Suction already ON. Creep straight down until the tared vertical
        force exceeds ``touch_n`` ('touched'), the vacuum seals ('sealed'), or
        abort ('force_limit'/'unreachable'/'max_descent'). Returns
        (last_q, z, reason). pick_retreat's press leg — unlike _creep_seal it
        does NOT hold and wait for the seal at contact."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        descended = 0.0
        vac = suction_io.VacuumMonitor(); vac.start()

        def _halt(q):
            self._arm.set_joint_pos_vel(np.asarray(q), np.zeros(len(q)))

        try:
            while descended < cfg.DESCENT_MAX_M:
                if vac.is_sealed():
                    _halt(prev_q)
                    logger.info("[suction] sealed mid-press at ee_z={:.4f}", z)
                    return prev_q, z, "sealed"
                z_next = z - cfg.DESCENT_CREEP_SPEED_M_S * dt
                sol = self.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
                if sol.pos_err_m > cfg.REACH_TOL_M:
                    _halt(prev_q); return prev_q, z, "unreachable"
                self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
                f = self.vertical_force()
                if f is not None:
                    if f > cfg.FORCE_HARD_LIMIT_N:
                        _halt(sol.q); return sol.q, z_next, "force_limit"
                    if f > touch_n:
                        _halt(sol.q)
                        logger.info("[suction] bounce touch {:.1f}N at ee_z={:.4f} — retreat", f, z_next)
                        return sol.q, z_next, "touched"
                descended += (z - z_next); z, prev_q = z_next, sol.q
                time.sleep(dt)
            _halt(prev_q)
            return prev_q, z, "max_descent"
        finally:
            threading.Thread(target=vac.stop, daemon=True).start()  # see _creep_seal

    def _sweep_line(self, start_q, target_xyz, rpy, scanner):
        """Straight-line lateral stream to target_xyz (per-tick IK, smoothstep
        ramp-in to BCR_SWEEP_SPEED_M_S), polling the scanner EVERY tick — a
        read halts the sweep mid-line. Returns the code, or None at the end
        (also on a tick's IK failing: partial pass, logged). ``start_q`` is
        the previous move's commanded joints — the stream continues from
        exactly there, as the descent legs do."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        p = np.asarray(pos, dtype=float).copy()
        tgt = np.asarray(target_xyz, dtype=float)
        dist = float(np.linalg.norm(tgt - p))
        if dist < 1e-6:
            return scanner.result()
        u = (tgt - p) / dist

        def _halt(q):
            self._arm.set_joint_pos_vel(np.asarray(q), np.zeros(len(q)))

        traveled = elapsed = 0.0
        while traveled < dist:
            r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
            speed = float(cfg.BCR_SWEEP_SPEED_M_S) * (r * r * (3.0 - 2.0 * r))
            step = min(speed * dt, dist - traveled)
            p = p + u * step
            sol = self.solve_pose(p, rpy, seed=prev_q, min_motion=True)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q)
                logger.warning("[suction] sweep line IK failed at ({:.3f},{:+.3f}) — "
                               "pass cut short", p[0], p[1])
                return scanner.result()
            self._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
            prev_q = sol.q
            traveled += step
            elapsed += dt
            code = scanner.result()
            if code is not None:
                _halt(sol.q)
                logger.info("[suction] barcode read mid-sweep at ({:.3f},{:+.3f})",
                            p[0], p[1])
                return code
            time.sleep(dt)
        _halt(prev_q)
        return scanner.result()

    def _sweep_scan(self, ee_pos, rpy, scanner, case_center):
        """Raised a bit above the creep z — no tilt — Y-FIRST sweep for a read:
        the barcode sits CENTERED in case-local x, so one CONTINUOUS case-local
        y pass (scanner polled per tick) runs at dx=0 first; only if it reads
        nothing do x offsets (nearest first) each rerun the y pass, direction
        alternating so there's no return leg. The y pass is clamped at the case
        center line — crossing it would read the OTHER slot's barcode and
        misidentify this battery. Returns the code or None if exhausted (arm
        back over the pick point at the pre-lift z; the following creep-seal
        descends from there)."""
        cx, cy, _cz, cyaw = case_center
        c, s = float(np.cos(cyaw)), float(np.sin(cyaw))
        z = float(self.current_ee_pose()[0][2]) + cfg.BCR_SWEEP_LIFT_M   # lift to sweep
        bx, by = float(ee_pos[0]) - cx, float(ee_pos[1]) - cy
        loc_by = -s * bx + c * by                          # battery's case-local y (side)
        y_lo, y_hi = -float(cfg.BCR_SEARCH_MAX_Y_M), float(cfg.BCR_SEARCH_MAX_Y_M)
        if loc_by > 0.0:
            y_lo = max(y_lo, -loc_by)   # other-slot guard (case-local y = 0 line)
        elif loc_by < 0.0:
            y_hi = min(y_hi, -loc_by)
        code = None
        forward = True
        for dx in _axis_steps(cfg.BCR_SEARCH_X_STEP_M, cfg.BCR_SEARCH_MAX_X_M):
            a, b = (y_lo, y_hi) if forward else (y_hi, y_lo)
            wx0 = float(ee_pos[0]) + c * dx - s * a
            wy0 = float(ee_pos[1]) + s * dx + c * a
            q0 = self.move_ee([wx0, wy0, z], rpy)   # to this pass's start
            if q0 is None:
                continue
            code = scanner.result()                 # the approach may have read it
            if code is None:
                wx1 = float(ee_pos[0]) + c * dx - s * b
                wy1 = float(ee_pos[1]) + s * dx + c * b
                logger.info("[suction] barcode sweep: dx={:+.0f}mm, y {:+.0f}->{:+.0f}mm",
                            dx * 1000, a * 1000, b * 1000)
                code = self._sweep_line(q0, (wx1, wy1, z), rpy, scanner)
            if code is not None:
                break
            forward = not forward
        # back over the real pick point before the seal descent
        self.move_ee([float(ee_pos[0]), float(ee_pos[1]), z - cfg.BCR_SWEEP_LIFT_M], rpy)
        return code if code is not None else scanner.result()


# ---------------------------------------------------------------------------
# On-robot test:  python suction.py [POSE_NAME]   (default CASE_PICK)
#   go home -> pick(pose) -> lift to transport.  Place a real part under the cup.
# ---------------------------------------------------------------------------
def _test_on_robot() -> None:
    import sys
    from dexcontrol.robot import Robot

    name = next((a for a in sys.argv[1:] if not a.startswith("-")), "CASE_PICK")
    if name not in cfg.TAUGHT_POSES:
        logger.error("unknown pose {!r}; choose from {}", name, list(cfg.TAUGHT_POSES))
        return

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + ACTIVATES SUCTION: home -> pick({}) -> lift.", name)
    logger.warning("Place a part under the cup. Clear workspace, e-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    with Robot() as bot:
        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            logger.info("-> home")
            m.move_joints(m._home_seed)
            if "--gated" in sys.argv:   # barcode-gated pick (scan + sweep + gated seal)
                res = m.pick_gated(cfg.TAUGHT_POSES[name])
            else:
                res = m.pick(cfg.TAUGHT_POSES[name])  # plain pick; ends at transport
            logger.info("PICK result: success={} reason={} contact_ee_z={} barcode={}",
                        res.success, res.reason, res.contact_ee_z, res.barcode)


if __name__ == "__main__":
    _test_on_robot()
