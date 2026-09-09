"""Suction pick-and-place for ik_demo (built on arm.ArmMover).

Stage 1 — detect-and-freeze descent: descend vertically by streaming per-tick
IK via arm.set_joint_pos_vel (finite-diff velocity feedforward, ~100 Hz), and
stop on the vertical wrench force. Two-signal pick: force = contact,
DI0 vacuum = seal. Rolling wrench reference (no tare); separate limits for pick
(empty cup) and place (battery in cup). Two-speed profile: fast in free air,
slow creep in the contact zone.

Planned refinement (PLAN.md): replace the creep + seal-press with admittance
control (bounded contact force) once this is verified on the robot.

Force sensing uses the arm's native wrench_sensor (6-vector), referenced and
projected onto base-vertical via the EE rotation — no external read_force.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    from . import config as cfg
    from .arm import ArmMover, TickPacer
    from .drivers import suction_io
    from .drivers.bcr import BackgroundScanner
except ImportError:  # allow `python suction.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover, TickPacer
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
    auto_release: bool = False        # failed place that should release WITHOUT the operator gate


class SuctionMover(ArmMover):
    """Suction pick/place on the suction arm (cfg.ARM_SIDE)."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side=cfg.ARM_SIDE, ee_frame=cfg.EE_FRAME)
        self._wrench = getattr(self._arm, "wrench_sensor", None)
        if self._wrench is None:
            logger.warning("[suction] {} arm has no wrench sensor — contact detection OFF", self._side)
        # rolling wrench reference (see cfg.WRENCH_REF_WINDOW_S) — replaces the
        # per-descent tare: every force reading is the change from the median of
        # the PRECEDING window, so slow drift cancels instead of eating the
        # threshold. Frozen at contact (freeze_reference) so a sustained press
        # cannot be absorbed into its own zero.
        self._wref: "deque[np.ndarray]" = deque(
            maxlen=max(2, int(cfg.WRENCH_REF_WINDOW_S * cfg.CONTROL_HZ)))
        self._ref_dt = 1.0 / float(cfg.CONTROL_HZ)
        self._ref_t: "float | None" = None
        self._ref_frozen = False

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
    def _ref_read(self) -> "tuple[np.ndarray, np.ndarray] | None":
        """(raw 6-vector, rolling reference 6-vector), or None while the window
        is still warming up / no sensor.

        The window is advanced on a TIME basis, not per call: a tick may read
        the wrench two or three times (vertical_force, contact_wrench, the debug
        trace) and the reference has to span cfg.WRENCH_REF_WINDOW_S either way.

        The reference is the MEDIAN of the window as it stood BEFORE this
        sample, so a contact never contributes to the zero it is measured
        against. A 20-50ms contact ramp reaches only the last few of the
        window's samples, which a median ignores — by the time it could move
        the median the caller has long since crossed its threshold and frozen.
        """
        if self._wrench is None:
            return None
        s = np.asarray(self._wrench.get_wrench_state(), dtype=float).ravel()
        if s.size < 6:                      # force-only transport: pad torques
            s = np.concatenate([s[:3], np.zeros(3)])
        raw = s[:6]
        ref = (np.median(np.asarray(self._wref), axis=0)
               if len(self._wref) else None)
        now = time.perf_counter()
        if not self._ref_frozen and (self._ref_t is None
                                     or now - self._ref_t >= self._ref_dt):
            self._ref_t = now
            self._wref.append(raw)
        warm = max(2, int(cfg.WRENCH_REF_WARMUP_S * cfg.CONTROL_HZ))
        if ref is None or len(self._wref) < warm:
            return None
        return raw, ref

    def freeze_reference(self, on: bool = True) -> None:
        """Stop (or resume) updating the rolling reference. Freeze the moment a
        contact is declared: the press servo then holds a force measured against
        the PRE-contact state, and a sustained press cannot creep into its own
        zero. Resuming clears the window, so the next descent re-warms."""
        self._ref_frozen = bool(on)
        if not on:
            self._wref.clear()
            self._ref_t = None

    def tare(self, n: int | None = None) -> None:
        """Prime the rolling reference in place (blocking). No longer a
        baseline capture — kept because ik_VLM.supervisor calls it before a
        stationary tap test, where nothing is moving to fill the window."""
        if self._wrench is None:
            return
        self.freeze_reference(False)
        n = int(n or self._wref.maxlen)
        for _ in range(n):
            self._ref_read()
            time.sleep(self._ref_dt)
        logger.info("[suction] wrench reference primed ({} samples, |f|={:.2f}N)",
                    len(self._wref),
                    float(np.linalg.norm(np.median(np.asarray(self._wref), axis=0)[:3])))

    @property
    def _force_baseline(self) -> np.ndarray:
        """Current rolling reference, force part (ik_VLM.signals reads this)."""
        if not len(self._wref):
            return np.zeros(3)
        return np.median(np.asarray(self._wref), axis=0)[:3]

    @property
    def _torque_baseline(self) -> np.ndarray:
        if not len(self._wref):
            return np.zeros(3)
        return np.median(np.asarray(self._wref), axis=0)[3:6]

    def vertical_force(self) -> float | None:
        """|base-vertical CHANGE| of the contact force (N) against the rolling
        reference, or None while the reference is warming up.

        The driver reports in the BASE frame already — see cfg.WRENCH_AXIS_SIGN
        — so the sign array is the whole correction and no EE rotation is
        applied. |.| discards the z sign, which is why vertical contact
        detection worked even while the lateral latch did not.
        """
        r = self._ref_read()
        if r is None:
            return None
        raw, ref = r
        return float(abs(((raw[:3] - ref[:3]) * np.asarray(cfg.WRENCH_AXIS_SIGN))[2]))

    def vertical_force_signed(self) -> float | None:
        """SIGNED base-vertical change (N): + = the seat pushing UP on the part
        (a press), - = the part PULLING down on the cup (wedged, and the arm is
        lifting against it).

        The press servo must use this, never ``vertical_force``'s |.|: with the
        magnitude only, a sign reversal reads as "still pressing too hard", so
        the servo lifts further, which increases the pull, which keeps the
        magnitude high — it elevators a stuck part upward instead of releasing
        it (0903 battery place: fz_signed -8.2N while z rose +13mm; 0901 the
        same runaway reached +36mm and was misdiagnosed as a wall bind).
        Detection thresholds keep using the magnitude — a hard PULL is just as
        much a contact, and just as much a reason to abort.
        """
        r = self._ref_read()
        if r is None:
            return None
        raw, ref = r
        return float(((raw[:3] - ref[:3]) * np.asarray(cfg.WRENCH_AXIS_SIGN))[2])

    def contact_wrench(self) -> "tuple[np.ndarray, np.ndarray] | None":
        """Tared 6-axis wrench in the BASE frame: (force N, torque Nm), or None
        (no sensor / force-only transport).

        The driver reports base-frame already, so cfg.WRENCH_AXIS_SIGN is the
        whole correction and NO EE rotation is applied — rotating again injects
        a yaw-dependent lateral mirror that makes the corner wall-latch read the
        wrong axis. The LATERAL channels give the force the TOOL applies (not
        the reaction on it), so a wall reads drive-aligned — see
        cfg.WRENCH_AXIS_SIGN. NOTE the sensor sits at the wrist, SUCTION_LENGTH
        above the cup tip — lateral tip forces lever into mx/my (~0.2m arm),
        with the torque channels' sign inverted relative to r x f.
        """
        r = self._ref_read()
        if r is None:
            return None
        raw, ref = r
        sign = np.asarray(cfg.WRENCH_AXIS_SIGN)
        return ((raw[:3] - ref[:3]) * sign, (raw[3:6] - ref[3:6]) * sign)

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
    def _descent_speed(self, z: float, creep_z: float, elapsed: float,
                       fast: "float | None" = None) -> float:
        """Descent cup-tip speed with two smoothstep shapes and no velocity step:
        ramp IN from rest over DESCENT_RAMP_S (rest->descend handoff), and blend
        fast->creep over DESCENT_DECEL_BAND_M above creep_z (so there's no jerk
        at the creep line). At/below creep_z the speed is the creep speed.

        ``fast``: cruise override (default DESCENT_APPROACH_SPEED_M_S). The
        corner seat passes the slower CORNER_DESCENT_SPEED_M_S — it is the only
        place descent that still streams per-tick from the hover, and the
        tracking deviation that cruise builds is what threw the case into the
        bin wall (see the config comment)."""
        creep = cfg.DESCENT_CREEP_SPEED_M_S
        fast = cfg.DESCENT_APPROACH_SPEED_M_S if fast is None else float(fast)
        band = max(float(cfg.DESCENT_DECEL_BAND_M), 1e-6)
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

    def _creep_speed(self, elapsed: float) -> float:
        """Creep speed with the SAME smoothstep ramp-in the two-speed descent
        profile uses.

        The creep loops used to command DESCENT_CREEP_SPEED_M_S on their very
        first tick — a 0 -> 0.04 m/s step, ~8 m/s^2 of commanded accel with
        unbounded jerk, and it lands in the SLOWEST part of the whole motion
        where it is most visible. The descent profile's own peak is 1.6 m/s^2,
        so the handoff was 5x the motion it hands off to. It got worse when the
        pick's approach became a planned move that ends AT REST on the creep
        line: the step then IS the entire handoff. (The place side never had
        this — _descend_corner_seat goes through _descent_speed, which ramps.)
        """
        r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
        return float(cfg.DESCENT_CREEP_SPEED_M_S) * (r * r * (3.0 - 2.0 * r))

    def _descend_to_contact(self, target_ee_z: float, rpy, force_limit: float,
                            start_q: np.ndarray, tick_cb=None,
                            contact_n: "float | None" = None) -> PickResult:
        """Descend straight down (x,y,rpy held) until contact / hard-limit / floor.

        Force checks (and tick_cb's f) stay off for the first
        cfg.WRENCH_REF_WARMUP_S while the rolling wrench reference fills; no
        tare and no stationary pause (see ``_ref_read``).

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
        self.freeze_reference(False)      # fresh rolling reference per descent
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()

        def _halt(q):
            self._send(np.asarray(q), np.zeros(len(q)))

        pace = TickPacer(dt)
        while descended < cfg.DESCENT_MAX_M:
            speed = self._descent_speed(z, creep_z, elapsed)
            z_next = z - speed * dt
            sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q)
                return PickResult(False, "unreachable", z)
            z_next = float(p[2])
            self._send(sol.q, (sol.q - prev_q) / dt)

            f = self.vertical_force()
            if f is not None:
                if f > force_limit:
                    _halt(sol.q); self.freeze_reference()
                    logger.warning("[suction] hard push {:.1f}N at ee_z={:.4f} — abort", f, z_next)
                    return PickResult(False, "force_limit", z_next,
                                      contact_info=self._contact_snapshot(sol.q))
                if f > float(cfg.FORCE_CONTACT_THRESHOLD_N if contact_n is None
                             else contact_n):
                    _halt(sol.q); self.freeze_reference()
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
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                self._track_trace("place descend", [x, y, z], prev_q,
                                  tick_acc / tick_n * 1000.0,
                                  extra=" f={}".format("-" if f is None
                                                       else format(f, "+.1f")))
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0

        _halt(prev_q)
        logger.warning("[suction] max descent ({:.2f}m) without contact", cfg.DESCENT_MAX_M)
        return PickResult(False, "max_descent", z)

    # ------------------------------------------------------------------
    # Pick / place
    # ------------------------------------------------------------------
    def _lift_to_transport(self, rpy, to_clear_only: bool = False,
                           creep_out_m: float = 0.0) -> None:
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
        # ``creep_out_m``: leave the slot the way the descent entered it, creeping
        # out of the first stretch (where the part is still between the walls)
        # before cruising. OFF by default — it is only worth its ~1.2s where the
        # part is actually boxed in on the way out. Not needed after a PLACE (the
        # cup is empty) nor after a battery pick; the case pick, which lifts out
        # of the source bin, opts in.
        q = self.move_ee_vertical(z_clear, rpy, creep_out_m=creep_out_m)
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
        seed = self._start_q()
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

    def _approach_and_hover(self, ee_pos, rpy, ez, to_creep_z: bool = False,
                            tick_cb=None, approach_z: "float | None" = None,
                            creep_gap: "float | None" = None):
        """Travel to the column at transport height (sideways clearance), then
        drop over the true xy — as ONE blended stream (move_joints_through), the
        intermediate waypoints crossed at blend speed rather than stopped at.

        ``to_creep_z``: end the stream at the CREEP line (ez +
        DESCENT_CREEP_GAP_M) instead of the hover, so the hover becomes a
        blended junction and the per-tick vertical descent begins only where it
        matters. Above the creep line the cup does not need to hold a vertical
        line at all, and handing that stretch to the planner is strictly better
        there: ruckig bounds accel/jerk (the per-tick stream commanded up to 84
        rad/s^2 against a 2.5 limit, which is what threw the cup 34-40mm off
        line, 0903) and the arm gets the whole planned deceleration to settle
        into the creep waypoint. NOT for the case place: the case drives
        sideways from the hover on down (air_travel=max, tall bin walls) and it
        earns its 37-68mm of lateral travel during that stretch — a planned
        move would take that away.

        ``tick_cb(z, f)``: optional per-tick guard for the streamed move, same
        shape as the descent loops'. With ``to_creep_z`` the planner replaces a
        stretch that used to be force-monitored, so the caller can keep a hard
        force limit over it — the expected contact z can be tens of mm off
        (0903 warp planes read -33 and -41mm vs the model), and a surface inside
        the planned stretch would otherwise be met unguarded.

        The transport approach can fall a few mm short horizontally when the arm
        nears its reach limit up high; the lower waypoints are well inside reach
        and solved FROM the approach solution (the same branch the old
        move-then-solve-live chain landed on), so the arm recovers the true xy
        before the vertical descent (which holds xy) — preventing an offset,
        misaligned seat.
        ``approach_z``: fly the first (sideways-clearance) leg at this height
        instead of SAFE_TRANSPORT_Z, and cap the hover with it. For a column
        that is only reachable LOW — the lid drop-off does not solve anywhere
        near transport height — where the default would fail the very first
        leg. The hover is dropped when it coincides with the approach.

        Returns the config the stream ends at, or None if a leg is unreachable."""
        z_app = float(cfg.SAFE_TRANSPORT_Z if approach_z is None else approach_z)
        approach = np.array([ee_pos[0], ee_pos[1], z_app])
        sol_app = self.solve_pose(approach, rpy, seed=self._start_q(), min_motion=True)
        if sol_app.pos_err_m > cfg.REACH_TOL_M:  # transport leg: shortfall recovered by the hover
            logger.error("[arm] target unreachable ({:.1f}mm short) — not moving",
                         sol_app.pos_err_m * 1000)
            return None
        qs, seed = [sol_app.q], sol_app.q
        # The hover waypoint STAYS, also on the to_creep_z path. Dropping it
        # (tried 0903, to kill a ~2 deg pitch ring at its blended junction) put
        # transport->creep in ONE long joint-space segment: the Cartesian arc
        # grew, the arm was still behind it when the stream ended, _start_q()
        # saw more than CMD_CARRY_TOL_RAD of disagreement and fell back to the
        # LIVE config — so the creep descended from where the arm actually was.
        # Measured 50.4 and 61.7mm too far in x on two runs, i.e. the pick
        # grabbed the case 5-6cm off. The hover waypoint is what recovers the
        # true xy before the vertical leg starts holding it, exactly as its
        # original comment said; the pitch ring is the cheaper problem.
        zs = [min(z_app, ez + cfg.HOVER_HEIGHT_M)]
        if abs(zs[0] - z_app) < 1e-3:
            zs = []          # a low approach IS the hover — no duplicate waypoint
        if to_creep_z:
            # DESCENT_CREEP_GAP_SETTLED_M, not DESCENT_CREEP_GAP_M: this stream
            # ENDS here at rest and _settle_at confirms the arrival, so there is
            # no cruise-built deviation to creep off — the gap only has to cover
            # the error in ez. A caller whose ez is a guess passes the long one.
            zs.append(ez + float(cfg.DESCENT_CREEP_GAP_SETTLED_M
                                 if creep_gap is None else creep_gap))
        for z in zs:
            sol = self.solve_pose([ee_pos[0], ee_pos[1], z], rpy,
                                  seed=seed, min_motion=True)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                logger.error("[arm] target unreachable ({:.1f}mm short) — not moving",
                             sol.pos_err_m * 1000)
                return None
            qs.append(sol.q)
            seed = sol.q
        if to_creep_z:
            logger.info("[suction] approach: transport -> hover -> creep line "
                        "z={:.4f} as one blended stream (no hover stop)", zs[-1])
        self.move_joints_through(qs, tick_cb=tick_cb)
        self._settle_at(qs[-1])
        return qs[-1]

    def _settle_at(self, q_target) -> float:
        """Hold the final command until the arm actually gets there. Returns the
        residual xy distance in metres (0.0 when there is no robot to read).

        The vertical leg that follows holds whatever xy it STARTS from, and it
        starts either from this config or — once the arm disagrees with it by
        CMD_CARRY_TOL_RAD — from the LIVE one. So a stream that ends with the
        arm still behind it does not fail, it just descends somewhere else:
        0903 measured 12.9-57.4mm of x on five consecutive places, all of it
        landing straight in the place position. The planned stream already ends
        at zero velocity; it just returned on its last command tick without
        waiting, and the hover FULL STOP used to cover for that.
        """
        dt = 1.0 / float(cfg.CONTROL_HZ)
        q_target = np.asarray(q_target, dtype=float)
        try:
            p_cmd, _ = self.fk(q_target)
            deadline = time.time() + float(cfg.APPROACH_SETTLE_MAX_S)
            d = float("inf")
            while True:
                self._send(q_target, np.zeros(len(q_target)))
                p_live, _ = self.fk(self._live_arm_q())
                d = float(np.hypot(p_live[0] - p_cmd[0], p_live[1] - p_cmd[1]))
                if d <= float(cfg.APPROACH_SETTLE_TOL_M) or time.time() > deadline:
                    break
                time.sleep(dt)
            if d > float(cfg.APPROACH_SETTLE_TOL_M):
                logger.warning("[suction] approach did NOT settle: {:.1f}mm off in "
                               "xy after {:.1f}s (cmd {:.4f},{:.4f} vs live "
                               "{:.4f},{:.4f}) — the descent holds the xy it "
                               "STARTS from, so this becomes a place offset",
                               d * 1000.0, float(cfg.APPROACH_SETTLE_MAX_S),
                               p_cmd[0], p_cmd[1], p_live[0], p_live[1])
            else:
                logger.info("[suction] approach settled to {:.2f}mm in xy", d * 1000.0)
            return d
        except Exception:  # noqa: BLE001 — never kill a move over a settle read
            return 0.0

    def _approach_force_guard(self, limit: float):
        """tick_cb for a planned approach stretch: halt on a hard force.

        With ``to_creep_z`` the planner replaces a stretch that used to be
        force-monitored by the descent loop, and the expected contact z can be
        tens of mm off (0903 warp planes read -33 and -41mm vs the model), so a
        real surface can sit inside the planned stretch. The rolling reference
        warms itself up over the first WRENCH_REF_WARMUP_S of reads
        (vertical_force returns None until then), so this is live for all but
        the first ~0.1s."""
        def _cb(z, _q) -> bool:
            f = self.vertical_force()
            if f is not None and f > float(limit):
                logger.warning("[suction] approach halted: {:.1f}N at ee_z={:.4f} "
                               "(limit {:.0f}N) — a surface inside the planned "
                               "stretch, above the creep line", f, z, limit)
                return True
            return False
        return _cb

    def pick(self, pose, expected_z=None, lift_to_clear: bool = False,
             creep_gap: "float | None" = None,
             contact_n: "float | None" = None,
             force_limit: "float | None" = None,
             retry_offset_m: "float | None" = None,
             lift_creep_out_m: float = 0.0,
             retry_dir: "tuple[float, float] | None" = None) -> PickResult:
        """Approach from transport to the hover as one blended stream (suction
        OFF) and settle there, then descend per-tick — cruise, the long
        DESCENT_DECEL_BAND_M slowdown, creep — through the creep line WITHOUT
        stopping; suction ON (async) at the creep line and creep to contact +
        seal, sealing on contact (same seal strategy as the gated pick, minus
        the scan). Same descent path as pick_retreat. The 0903-0909 variant
        ended the planned stream AT the creep line instead: that leg ran up to
        0.48 m/s and swung the cup 65mm / 12.7 deg in flight (0907 logs), and
        then waited ~0.75s for _settle_at — the per-tick profile with the wide
        decel band was verified smooth on the collect_case_pick runs (0909).
        Ends at transport holding the part. ``expected_z`` overrides the taught
        contact z (layer stacking: the orchestrator passes the measured z).
        ``retry_dir``: base-frame (dx, dy) direction for the seal-retry nudge —
        see _seal_with_retry."""
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] pick: approach@transport -> hover -> per-tick "
                    "descent -> suction on at creep line -> creep-seal")
        if suction_io.is_suction_commanded_on():
            # BEFORE the stream, not after: suction_off is blocking (two HTTP
            # round-trips + a 0.5s settle) and at the creep line it would put
            # back the very pause the single stream removes. Re-asserted only
            # when needed — on a normal pick the cup is already OFF from the
            # previous release.
            suction_io.suction_off()
        hard = float(cfg.FORCE_HARD_LIMIT_N if force_limit is None else force_limit)
        q_hover = self._approach_and_hover(ee_pos, rpy, ez,
                                           tick_cb=self._approach_force_guard(hard))
        if q_hover is None:
            return PickResult(False, "unreachable")
        # Creep gap: the LONG one by default. The 20mm settled gap was sized for
        # a stream that stops and settles on the creep line; this descent runs
        # through it, so the gap must also absorb the fast-stretch lag.
        creep_z = ez + float(cfg.DESCENT_CREEP_GAP_M if creep_gap is None else creep_gap)
        _last_q, z, reason = self._descend_open(creep_z, rpy, q_hover, hard,
                                                halt_at_floor=False)
        if reason != "at_floor":
            return PickResult(False, reason, z)
        suction_io.suction_on_async()   # off the critical path — creep continues
        # start_elapsed=DESCENT_RAMP_S: _descend_open did NOT halt, see
        # _creep_to_force's start_elapsed docstring.
        return self._seal_and_lift(rpy, lift_to_clear,
                                   lift_creep_out_m=lift_creep_out_m,
                                   retry_dir=retry_dir, contact_n=contact_n,
                                   force_limit=force_limit,
                                   retry_offset_m=retry_offset_m,
                                   start_elapsed=cfg.DESCENT_RAMP_S)

    def _seal_with_retry(self, rpy, retry_dir=None,
                         contact_n: "float | None" = None,
                         force_limit: "float | None" = None,
                         retry_offset_m: "float | None" = None,
                         start_elapsed: float = 0.0) -> PickResult:
        """Creep-seal (suction already ON), retrying a failed seal: on
        vacuum_timeout — touched but the vacuum never latched — lift back to
        creep height (suction off, empty cup) and creep-seal again, up to
        PICK_SEAL_RETRIES times. Only vacuum_timeout retries: force_limit is
        a safety stop and unreachable is geometry — re-pressing won't help.
        The tare baseline stays valid (vertical lift, rotation unchanged).

        ``retry_dir``: base-frame (dx, dy) unit direction along which each retry
        is offset by PICK_SEAL_RETRY_OFFSET_M, the SIGN ALTERNATING off the
        first attempt's aim (+, -, +, ...). Absolute, not cumulative: retry 2
        goes to aim - offset, not back through aim + offset. Without a direction
        the retry re-presses the very point that just failed, which only helps
        when the miss was an error in the expected contact z. The lateral hop is
        a move_ee at the lifted height with an empty cup — its joint-space path
        arcs, which at 5mm and DESCENT_CREEP_GAP_M of clearance is nothing."""
        # start_elapsed applies to the FIRST creep only: a retry lifts and
        # stops first, so it ramps from rest like before.
        res = self._creep_seal(rpy, self._start_q(), contact_n=contact_n,
                               force_limit=force_limit, start_elapsed=start_elapsed)
        aim_xy = None
        for i in range(1, int(cfg.PICK_SEAL_RETRIES) + 1):
            if res.success or res.reason != "vacuum_timeout":
                break
            suction_io.suction_off()
            pos, _ = self.current_ee_pose()
            if aim_xy is None:
                aim_xy = (float(pos[0]), float(pos[1]))
            logger.warning("[suction] seal failed (vacuum_timeout) — lift {:.0f}mm "
                           "and retry {}/{}", cfg.DESCENT_CREEP_GAP_M * 1000.0,
                           i, int(cfg.PICK_SEAL_RETRIES))
            z_lift = float(pos[2]) + cfg.DESCENT_CREEP_GAP_M
            if self.move_ee_vertical(z_lift, rpy) is None:
                break  # can't lift from here — hand the failure back as-is
            step = float(cfg.PICK_SEAL_RETRY_OFFSET_M if retry_offset_m is None
                         else retry_offset_m)
            if retry_dir is not None and step > 0.0:
                d = step * (1.0 if i % 2 else -1.0)
                tgt = (aim_xy[0] + d * float(retry_dir[0]),
                       aim_xy[1] + d * float(retry_dir[1]), z_lift)
                logger.info("[suction] retry {} nudged {:+.0f}mm along "
                            "({:+.2f},{:+.2f}) -> cup ({:.3f},{:+.3f})",
                            i, d * 1000.0, float(retry_dir[0]),
                            float(retry_dir[1]), tgt[0], tgt[1])
                if self.move_ee(tgt, rpy) is None:
                    # the nudged column is out of reach — pressing the un-nudged
                    # point again is what we just proved does not work
                    break
            suction_io.suction_on()
            res = self._creep_seal(rpy, self._start_q(), contact_n=contact_n,
                               force_limit=force_limit)
        return res

    def _seal_and_lift(self, rpy, lift_to_clear: bool = False,
                       lift_creep_out_m: float = 0.0,
                       retry_dir: "tuple[float, float] | None" = None,
                       contact_n: "float | None" = None,
                       force_limit: "float | None" = None,
                       retry_offset_m: "float | None" = None,
                       start_elapsed: float = 0.0) -> PickResult:
        """Suction already ON at creep height: creep-seal (with seal retries),
        then lift to transport on success (suction off on failure). Shared pick
        tail. ``lift_to_clear`` stops the lift at the wall-clear height (see
        _lift_to_transport). ``contact_n`` / ``force_limit`` pass a gentler
        force pair down for one pick (cardboard — see _creep_seal)."""
        res = self._seal_with_retry(rpy, retry_dir, contact_n=contact_n,
                                    force_limit=force_limit,
                                    retry_offset_m=retry_offset_m,
                                    start_elapsed=start_elapsed)
        if res.success:
            # Relieve the creep-contact press before lifting (mirrors place()'s
            # RELEASE_PRELIFT_M) — otherwise the lift's first motion has to break
            # the residual seat press while already carrying the part.
            if cfg.SEAL_PRELIFT_M > 0.0:
                pos, _ = self.current_ee_pose()
                self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
            # Lift straight up to clear, then to transport — ready to travel.
            self._lift_to_transport(rpy, to_clear_only=lift_to_clear,
                                    creep_out_m=lift_creep_out_m)
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
        if suction_io.is_suction_commanded_on():
            # re-assert only when needed: every suction command costs two HTTP
            # round-trips + a fixed 0.5s controller settle (suction_io._run) —
            # on a normal pick the cup is already OFF (the previous release)
            suction_io.suction_off()
        # empty-cup tare happens IN-STREAM during the descent (_descend_open)
        creep_z = ez + cfg.DESCENT_CREEP_GAP_M
        _last_q, z, reason = self._descend_open(creep_z, rpy, q_hover,
                                                cfg.FORCE_HARD_LIMIT_N,
                                                halt_at_floor=False)
        if reason != "at_floor":
            return PickResult(False, reason, z)
        suction_io.suction_on_async()   # off the critical path — creep continues
        # start_elapsed=DESCENT_RAMP_S: _descend_open above did NOT halt (it
        # ended already running at creep speed) — see _creep_to_force's
        # start_elapsed docstring.
        _q, z, reason = self._creep_to_force(rpy, self._start_q(), touch_n,
                                             start_elapsed=cfg.DESCENT_RAMP_S)
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
                    self._send(q, np.zeros(len(q)))
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
              backoff_m: "float | None" = None,
              lift_to_clear: bool = False,
              corner_seat: "str | None" = None,
              corner_touch_first: bool = False,
              approach_z: "float | None" = None,
              lift_z: "float | None" = None,
              creep_gap: "float | None" = None) -> PickResult:
        """Hover above the seat, descend to contact within buffer, release.
        On a failed descent the part is HELD (suction on) until the operator
        confirms the release — unreachable/max_descent can end mid-air, where
        an automatic blow-off drops the battery from height.
        ``expected_z`` overrides the taught seat z (layer stacking).
        ``misseat_tol_m``: contact more than this ABOVE ``expected_z`` means the
        part landed on the rim/jig instead of dropping into the seat (a proper
        seat sits 5-15mm lower) — held for the operator like a failed descent,
        instead of blindly releasing a misaligned part. Pass it only with a
        measured-anchored ``expected_z`` (the model plane drifts too much).
        ``backoff_m``: override the pre-release wall back-off (default:
        CASE_/BATTERY_CORNER_BACKOFF_M for the part type). Pass 0.0 to release
        hard against the datum corner — the RUN'S FIRST case does that, because
        it becomes the datum every later place is measured from, so backing it
        off would move the reference itself. The cost is the cup retreating
        while the part is still preloaded against the walls, which is what the
        back-off exists to prevent.

        ``approach_z`` / ``lift_z``: fly in at, and lift back to, this height
        instead of SAFE_TRANSPORT_Z. Both for a seat that is only reachable low
        (the lid drop-off at LID_PLACE_TORSO_DEG): the default approach leg and
        the default lift both aim at transport height, which does not solve at
        that xy, so the place would fail before descending and then climb until
        the lift stalled.

        ``corner_seat`` ("case" / "battery" / None): descend AND register in
        one guarded stream — the aim is biased *_CORNER_AIM_BIAS_M away from
        the datum corner, and the descent itself drives the held part toward
        the corner, each axis stopping on its own wall contact, so the walls
        fix the final position (see ``_descend_corner_seat``); replaces
        ``_misseat_recover`` for that place. The two part types differ in
        WHEN the drive runs: the case drives from the creep blend band down
        (the TALL bin walls catch it anywhere in there), the battery descends straight and
        drives only AFTER the first vertical contact (its slot walls are LOW
        — an airborne drift past the slot could never be pulled back).
        Unverified on the robot.

        ``corner_touch_first``: give a "case" corner seat the BATTERY's timing
        instead — descend straight to the first vertical contact, then drive.
        The case's airborne drive is what registers it against the tall bin
        walls, but the airborne LATCH is only as trustworthy as the descent
        that carries it: 0905's two single-case runs froze x after 4-6mm of
        travel, 67-104mm above the seat, on a 3.3-5.0N lateral spike, and then
        "registered" at (-0,+19)mm with x contributing nothing. air_travel=0
        disarms the airborne latch entirely (see ``latch_armed``), so nothing
        can latch before the part is down and the drive is real. The drive is
        not a DRAG, though: the touchdown having fixed the real seat z, the
        case rises CASE_CORNER_DRIVE_LIFT_M and crosses the whole
        CASE_CORNER_AIM_BIAS_M hanging clear of the surface, so the lateral
        channel carries the wall reaction with no mu * (weight + press) mixed
        into it, and it is set back down on the press servo only once both
        walls have latched. The battery never gets the lift (low slot walls). The run's FIRST case uses it: its
        target bin is empty (it lands on the floor, not on a case below) and it
        is the datum every later place is measured from, so a phantom latch
        there offsets the whole run."""
        ee_pos, rpy = self.taught_target(pose)
        # NOTE corner_seat: the *_CORNER_AIM_BIAS_M shift away from the datum
        # corner is applied by the CALLER (chassis_sequence.run_item) BEFORE
        # its reach pre-check, so the checked pose is the flown pose — no
        # bias is added here.
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] place: approach@transport -> hover -> descend -> release")
        # One blended stream down to the creep line for everything EXCEPT the
        # case corner seat: the case drives sideways WHILE descending
        # (air_travel=max) and earns its 37-68mm of lateral travel over the
        # creep-band stretch, which a planned leg ending at ez +
        # DESCENT_CREEP_GAP_SETTLED_M would fly straight past — so it keeps the
        # hover + per-tick descent.
        to_creep = corner_seat != "case"
        q_hover = self._approach_and_hover(
            ee_pos, rpy, ez, to_creep_z=to_creep,
            tick_cb=(self._approach_force_guard(cfg.FORCE_HARD_LIMIT_PLACE_N)
                     if to_creep else None),
            approach_z=approach_z, creep_gap=creep_gap)
        if q_hover is None:
            return PickResult(False, "unreachable")
        if corner_seat:
            # no stationary hover tare: the baseline is sampled IN-STREAM on
            # the descent's free-air stretch (see _descend_corner_seat)
            max_travel = float(cfg.CASE_CORNER_MAX_TRAVEL_M if corner_seat == "case"
                               else cfg.BATTERY_CORNER_MAX_TRAVEL_M)
            res = self._descend_corner_seat(ez, rpy, cfg.FORCE_HARD_LIMIT_PLACE_N,
                                            q_hover, max_travel=max_travel,
                                            air_travel=(max_travel
                                                        if (corner_seat == "case"
                                                            and not corner_touch_first)
                                                        else 0.0),
                                            lat_speed=float(
                                                cfg.CASE_CORNER_SPEED_M_S
                                                if corner_seat == "case"
                                                else cfg.BATTERY_CORNER_SPEED_M_S),
                                            drive_lift=float(
                                                cfg.CASE_CORNER_DRIVE_LIFT_M
                                                if (corner_seat == "case"
                                                    and corner_touch_first) else 0.0),
                                            backoff=float(
                                                (cfg.CASE_CORNER_BACKOFF_M
                                                 if corner_seat == "case"
                                                 else cfg.BATTERY_CORNER_BACKOFF_M)
                                                if backoff_m is None else backoff_m),
                                            misseat_tol_m=misseat_tol_m,
                                            tick_cb=tick_cb)
        else:
            # loaded-cup tare happens IN-STREAM during the descent
            res = self._descend_to_contact(ez, rpy, cfg.FORCE_HARD_LIMIT_PLACE_N,
                                           q_hover, tick_cb=tick_cb)
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
        if (not corner_seat) and res.reason == "misseat" \
                and int(cfg.PLACE_RECOVER_ATTEMPTS) > 0:
            res = self._misseat_recover(ez, rpy, float(misseat_tol_m), res)
        if res.final_yaw_rad is not None:
            # recovery re-oriented the wrist — release/prelift/lift at THAT yaw,
            # not the original one (rotating back while pressed would drag the part)
            rpy = (rpy[0], rpy[1], res.final_yaw_rad)
        if res.reason != "contact":
            if res.auto_release:
                logger.warning("[suction] place descent failed ({}) — auto "
                               "blow-off release, NO operator gate (run continues)",
                               res.reason)
            else:
                logger.warning("[suction] place descent failed ({}) — holding the "
                               "battery (suction ON), waiting for the operator",
                               res.reason)
                input(f"place-failed[{res.reason}]> hand-guide the part if needed; "
                      f"Enter to blow-off release + retreat (the run continues): ")
        if res.auto_release:
            # the press servo may have LIFTED the part while unwinding the
            # contact overshoot (observed +10mm) — a prelift on top of that
            # blow-drops it from ~20mm (the 0806 release-drop failure mode).
            # Set it DOWN to a light touch first and release with NO prelift.
            # (_creep_to_force can't do this: a held part reads as sealed.)
            self._set_down(rpy)
        else:
            pos, _ = self.current_ee_pose()
            if cfg.RELEASE_PRELIFT_M > 0.0:
                self.move_ee_vertical(pos[2] + cfg.RELEASE_PRELIFT_M, rpy)
        suction_io.release()
        if lift_z is not None:
            # low column: straight back up to where the approach came in
            self.move_ee_vertical(float(lift_z), rpy)
        else:
            # lift_to_clear: stop at the wall-clear height (cup EMPTY here) — the
            # caller starts the return chassis leg immediately and folds the
            # remaining rise into the parallel view park (both target z=1.10)
            self._lift_to_transport(rpy, to_clear_only=lift_to_clear)
        res.success = res.reason in ("contact",)
        return res

    def _column_reachable(self, x: float, y: float, rpy, z_top: float, z_bottom: float) -> bool:
        """IK pre-check for a straight-down column at (x,y) from z_top to
        z_bottom, warm-chained so successive solves stay on one branch — same
        check as chassis_sequence.descent_reachable, scoped to an arbitrary
        column instead of the live EE height. Used by misseat recovery to
        pre-check a candidate BEFORE physically moving there: a candidate that
        instead fails partway through the actual re-descent aborts recovery
        entirely (force_limit-like, see _misseat_recover) — catching it here
        first lets the caller skip to the next candidate."""
        zs = np.arange(z_top, z_bottom - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
        if zs[-1] > z_bottom + 1e-9:
            zs = np.append(zs, z_bottom)
        seed = None
        for z in zs:
            sol = self.solve_pose((x, y, float(z)), rpy, seed=seed, min_motion=seed is not None)
            ok = (sol.pos_err_m <= cfg.REACH_TOL_M) and sol.in_limits and not sol.in_collision
            if not ok:
                logger.warning("[suction] misseat recover column pre-check FAILED at "
                               "z={:.3f} (err={:.1f}mm, in_lim={}, col={}) for "
                               "xy=({:.3f},{:+.3f})", z, sol.pos_err_m * 1000,
                               sol.in_limits, sol.in_collision, x, y)
                return False
            seed = sol.q
        return True

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
            rpy_i = (rpy[0], rpy[1], cur_yaw)
            if not self._column_reachable(x, y, rpy_i, lift_z, ez):
                logger.warning("[suction] misseat recover {}/{}: candidate "
                               "({:.3f},{:+.3f}) yaw={:+.1f}deg unreachable over the "
                               "re-descent column ({:.3f}->{:.3f}) — skipping to the "
                               "next step", i, int(cfg.PLACE_RECOVER_ATTEMPTS), x, y,
                               float(np.rad2deg(cur_yaw)), lift_z, ez)
                continue
            self.move_ee_vertical(lift_z, (rpy[0], rpy[1], yaw_now))  # lift at the AS-CONTACTED yaw
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

    def _descend_corner_seat(self, target_ee_z: float, rpy, force_limit: float,
                             start_q: np.ndarray, max_travel: float,
                             air_travel: float, lat_speed: float,
                             backoff: float, drive_lift: float = 0.0,
                             misseat_tol_m=None,
                             tick_cb=None) -> PickResult:
        """Case-place descent with corner registration folded in: ONE guarded
        stream from the hover — descend AND drive toward the jig's datum
        corner, each axis stopping on its own contact, with no halt between
        "descend" and "register". The corner walls fix the final position
        (accuracy = the 1-2mm jig fit) regardless of the +4..12mm landing
        scatter — replaces ``_misseat_recover`` and the old post-contact /
        sweep-spiral recoveries for the case.

        z: the plain two-speed descent profile until the first vertical
        contact, cruising at the corner seat's slower CORNER_DESCENT_SPEED_M_S,
        then hands off IN-STREAM (``drive_lift`` = 0) to a light-press servo
        (CASE_CORNER_PRESS_N) instead of halting. The wrench baseline is
        taken against the ROLLING reference (cfg.WRENCH_REF_WINDOW_S), frozen
        at the contact handoff — force decisions wait only for the ~0.1s
        warm-up, and slow drift never eats the threshold.

        x,y: drive toward CASE_CORNER_DIR at ``lat_speed`` (CASE_/BATTERY_
        CORNER_SPEED_M_S) from the creep blend band (creep_z +
        DESCENT_CREEP_BLEND_M) on down, where the descent is decelerating and
        tracks its command — never at cruise, where the arm trails it tilted
        and off-axis (battery, and a case placed with corner_touch_first:
        only after contact, via air_travel=0) — each axis latches
        where it
        bumps its datum wall (its wall-reaction channel over
        CASE_CORNER_STOP_N, above the sliding-friction baseline — the lateral
        wrench frame is MIRRORED vs base (x/y swapped), so that is NOT the
        channel named after the axis: see cfg.CASE_CORNER_LAT_CHANNEL) and
        rides the corner
        straight down to contact. Per-axis travel is
        capped at ``air_travel`` before the first vertical contact and
        ``max_travel`` after (the airborne hold is silent and resumes once
        pressing; the cap only flags "no wall" when pressing): the case's
        tall bin walls catch it anywhere in that band so air = max, while the battery
        passes air = 0 — it descends straight and drives only after contact
        (its slot walls are too low to stop an airborne drift).

        ``drive_lift`` > 0 (a case placed with place(corner_touch_first=True))
        inserts a third phase between those two: at touchdown, which is what
        fixed the real seat z, the stream RISES off the seat and holds there
        while the lateral drive crosses to the walls, so the drive registers on
        wall reaction alone instead of fighting mu * (weight + press) of
        sliding friction over the case's 50mm of aim bias. The rise ends when
        the FORCE says the seat has let go (CASE_CORNER_LIFT_FREE_N, against
        the reference frozen at touchdown), not at a fixed distance —
        ``drive_lift`` is only the floor and CASE_CORNER_LIFT_MAX_M the cap,
        because the cup's compression and the arm's lag both eat an EE-measured
        rise (0905: 8mm of EE lift moved the case off the seat by nothing). The press servo is
        idle for that phase (there is no contact to hold) and takes over again
        the moment both axes latch, setting the case down onto its seat before
        the release — so the reported contact z is the pressed seat depth in
        every mode. The seat/misseat judgement uses the TOUCHDOWN z while the
        case hangs. Never for the battery: 8mm can clear its low slot walls.

        "In the slot" = the press reached the expected seat (within
        ``misseat_tol_m``, when given) OR the dual-signal sink fired (z sink
        + fz drop in one window: a rim landing dropping in mid-drive); the
        drive pauses while a drop settles so a half-dropped case can't be
        wedged in diagonally. Success = in-slot AND both axes WALL-LATCHED
        (the release precondition is real wall contact — a travel-cap stop
        without wall force is HELD for the operator, never blown off
        unregistered): back the axes off the walls by ``backoff``
        (CASE_/BATTERY_CORNER_BACKOFF_M, applied AWAY from the datum corner)
        (preload relief, so the cup retreat can't drag the registered case)
        and report the pressed z as the contact (no extra settle press —
        the ~5N press z is the seat depth within ~1mm). Both
        axes stopped without reaching the slot (after CASE_CORNER_DROP_GRACE_S
        of pressing), or press timeout, returns "misseat" — the caller's
        operator gate applies. A hard push once pressing LIFTS to relieve
        (CASE_CORNER_RELIEF_SPEED_M_S) instead of aborting — the spike is
        descent-servo lag, not a crash — and only an exhausted relief
        headroom (CASE_CORNER_RELIEF_MAX_M above the first contact = true
        jam) returns "force_limit". unreachable / max_descent /
        monitor_abort mirror ``_descend_to_contact``."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x0, y0, z = float(pos[0]), float(pos[1]), float(pos[2])
        cmd = [x0, y0]
        p_prev = (x0, y0, z)   # last pose actually commanded (solve_step)
        dirs = [float(np.sign(cfg.CASE_CORNER_DIR[0])),
                float(np.sign(cfg.CASE_CORNER_DIR[1]))]
        lat_step = float(lat_speed) * dt
        creep_z = target_ee_z + cfg.DESCENT_CREEP_GAP_M
        win_n = max(1, int(cfg.CASE_CORNER_SINK_WINDOW_S * cfg.CONTROL_HZ))
        z_hist: list[float] = []
        fz_hist: list[float] = []
        pressing = False           # False: two-speed descent; True: press servo
        lifting = False            # rising off the seat before the drive
        lift_done = False          # hanging clear, driving on wall reaction only
        lift_min = lift_cap = 0.0  # rise bounds, set at touchdown
        seating = False            # lifted drive done, pressing back down
        sunk = False
        settling = False
        settle_t = 0.0
        stopped = [False, False]   # force-latched OR travel-capped
        latched = [False, False]   # force-latched only (gets the back-off)
        f_last = [0.0, 0.0]
        relieved = [0.0, 0.0]        # lateral relief travel used, per axis
        relief_logged = [False, False]
        descended = 0.0
        elapsed = 0.0
        press_t = 0.0
        grace = 0.0
        dbg_t = 0.0
        press_capped = False
        trace_s = float(cfg.DESCENT_TRACE_S)
        tick_acc, tick_n, t_tick = 0.0, 0, time.perf_counter()

        def _halt(q):
            self._send(np.asarray(q), np.zeros(len(q)))

        def _press_step(z_now: float, fz_signed: "float | None") -> float:
            """z is a height (descend subtracts) — too much force (above target)
            must RAISE z (back off), too little must LOWER it.

            Takes the SIGNED force (see vertical_force_signed): a negative
            reading is the part PULLING, i.e. already lifted too far, and the
            error then drives z back DOWN. Fed the magnitude instead, this
            servo runs away upward against a wedged part."""
            if fz_signed is None:
                return z_now
            err = fz_signed - float(cfg.CASE_CORNER_PRESS_N)   # measured - target
            # DEADBAND: inside the band the drive already calls the press good
            # enough (at_press), so hold z instead of hunting. Without it this
            # servo oscillates — the contact is ~3 N/mm and the arm tracks with
            # a lag, so proportional-only chasing of an exact 5N overshoots and
            # reverses forever (0903 18:00: z 2.8mm p-p, fz swinging +1 <-> +9N,
            # which also jammed the settle pause open and hung the place).
            if abs(err) <= float(cfg.CASE_CORNER_DRIVE_PRESS_TOL_N):
                return z_now
            dz = float(np.clip(cfg.CASE_CORNER_PRESS_KP * err,
                               -cfg.CASE_CORNER_PRESS_MAX_SPEED_M_S,
                               cfg.CASE_CORNER_PRESS_MAX_SPEED_M_S)) * dt
            return z_now + dz

        def _sink_window(z_now: float, fz_now: "float | None") -> tuple[bool, float]:
            """(sank, window z-span): sank = z sink + fz drop together within
            the window; the span alone gates the post-drop settle."""
            z_hist.append(z_now)
            fz_hist.append(fz_now if fz_now is not None else 0.0)
            if len(z_hist) > win_n:
                z_hist.pop(0); fz_hist.pop(0)
            span = max(z_hist) - min(z_hist)
            if len(z_hist) < win_n:
                return False, span
            dz = z_hist[0] - z_hist[-1]      # positive = sank
            dfz = fz_hist[0] - fz_hist[-1]   # positive = fz dropped
            return (dz > cfg.CASE_CORNER_SINK_DZ_M
                    and dfz > cfg.CASE_CORNER_SINK_FZ_DROP_N), span

        registered = False
        contact_z = None    # z at the pressing handoff (relief anchor)
        relieving = False
        self.freeze_reference(False)      # fresh rolling reference per descent
        pace = TickPacer(dt)
        while True:
            # None only while the reference warms up (WRENCH_REF_WARMUP_S);
            # no tare, no stationary hover pause
            fz = self.vertical_force()               # |.| for detection/limits
            fz_s = self.vertical_force_signed()      # signed, for the servo
            over = fz is not None and fz > force_limit
            if not pressing:
                if over or (fz is not None and fz > cfg.FORCE_CONTACT_THRESHOLD_N):
                    # in-stream handoff: descent -> press servo. The creep's
                    # tracking lag can blow through the contact AND hard-push
                    # thresholds in one tick — a hard first touch IS the
                    # contact, handled by the relief below (never an abort)
                    pressing = True
                    contact_z = z
                    if float(drive_lift) > 0.0:
                        # LIFT-then-drive: the touchdown fixed the real seat z;
                        # rise off it so the drive reads wall reaction with no
                        # sliding friction mixed in. Ends on FORCE (the seat no
                        # longer carrying the case), between these two bounds.
                        lifting = True
                        lift_min = z + float(drive_lift)
                        lift_cap = z + float(cfg.CASE_CORNER_LIFT_MAX_M)
                    # freeze the reference AT contact: from here every force is
                    # measured against the pre-contact state, so the press
                    # servo's target is the force it actually adds and a
                    # sustained press cannot drift into its own zero
                    self.freeze_reference()
                    # Log the LATERAL state with it: a case riding down a wall it
                    # already latched transmits vertical drag, which can reach
                    # FORCE_CONTACT_THRESHOLD_N on its own — a "contact" declared
                    # with an axis already latched may be wall drag, not the seat.
                    logger.info("[suction] corner descent: first contact {:.1f}N at "
                                "ee_z={:.4f} — press servo on, drive uncapped "
                                "(latched=({},{}), lateral f=({:+.1f},{:+.1f})N)",
                                fz, z, latched[0], latched[1], f_last[0], f_last[1])
                else:
                    speed = self._descent_speed(z, creep_z, elapsed,
                                                fast=cfg.CORNER_DESCENT_SPEED_M_S)
                    z_next = z - speed * dt
                    descended += (z - z_next)
                    z = z_next
                    if descended >= cfg.DESCENT_MAX_M:
                        _halt(prev_q)
                        logger.warning("[suction] max descent ({:.2f}m) without contact",
                                       cfg.DESCENT_MAX_M)
                        return PickResult(False, "max_descent", z)
            if pressing:
                if over:
                    # over the hard limit: LIFT to relieve instead of aborting
                    # — the spike is descent-servo lag converging after the
                    # handoff (0824: 14N -> 20.2N in ~40ms WHILE the press
                    # servo raised at its 0.02 m/s cap), not a crash. Abort
                    # only when the relief headroom is exhausted (true jam).
                    if contact_z is None:
                        contact_z = z
                    if z - contact_z > float(cfg.CASE_CORNER_RELIEF_MAX_M):
                        _halt(prev_q)
                        logger.warning("[suction] corner descent: {:.1f}N despite "
                                       "{:.0f}mm of relief — jammed, abort", fz,
                                       (z - contact_z) * 1000.0)
                        return PickResult(False, "force_limit", z,
                                          contact_info=self._contact_snapshot(prev_q))
                    if not relieving:
                        relieving = True
                        logger.warning("[suction] corner descent: hard push {:.1f}N "
                                       "at ee_z={:.4f} — lifting to relieve", fz, z)
                    z += float(cfg.CASE_CORNER_RELIEF_SPEED_M_S) * dt
                elif lifting:
                    relieving = False
                    # Rise until the SEAT is no longer carrying the case (the
                    # frozen reference makes fz ~ 0 mean "hanging on the cup
                    # again"), never less than drive_lift and never past the
                    # cap. Distance alone cannot say this: the cup compresses
                    # and the arm lags, so 8mm of EE rise lifted the case by
                    # nothing on 0905 and the drive dragged it.
                    free = (fz_s is not None
                            and fz_s <= float(cfg.CASE_CORNER_LIFT_FREE_N))
                    if z >= lift_cap - 1e-9 or (z >= lift_min and free):
                        lifting, lift_done = False, True
                        if free:
                            logger.info("[suction] corner descent: lifted {:.1f}mm "
                                        "off the seat (fz {:+.1f}N) — the drive now "
                                        "sees wall reaction only",
                                        (z - contact_z) * 1000.0, fz_s)
                        else:
                            logger.warning("[suction] corner descent: lifted to the "
                                           "{:.0f}mm cap with the seat still carrying "
                                           "{:+.1f}N — the case may be pinned; the "
                                           "drive runs, but a latch inside a few mm "
                                           "of travel is friction, not a wall",
                                           float(cfg.CASE_CORNER_LIFT_MAX_M) * 1000.0,
                                           -1.0 if fz_s is None else fz_s)
                    else:
                        z = min(lift_cap,
                                z + float(cfg.DESCENT_CREEP_SPEED_M_S) * dt)
                elif lift_done:
                    relieving = False        # hold z: the walls do the work now
                else:
                    relieving = False
                    z = _press_step(z, fz_s)
                    if seating and contact_z is not None:
                        # the set-down after a lifted drive starts OUT of
                        # contact, so the servo is lowering against fz ~ 0 —
                        # bound it, or a case hung up on the walls never
                        # satisfies it and rides 0.02 m/s down for the whole
                        # timeout (cfg.CASE_CORNER_SETDOWN_MAX_M)
                        if z < contact_z - float(cfg.CASE_CORNER_SETDOWN_MAX_M):
                            _halt(prev_q)
                            logger.warning("[suction] corner descent: set-down went "
                                           "{:.0f}mm below the touchdown without "
                                           "reaching {:.1f}N — the case is hung up on "
                                           "the walls, not seated",
                                           (contact_z - z) * 1000.0,
                                           float(cfg.CASE_CORNER_PRESS_N))
                            return PickResult(False, "misseat", z,
                                              contact_info=self._contact_snapshot(prev_q))
                    # The press servo only ever RAISES while fz stays over the
                    # light target, and when the load is a WALL BIND (a latched
                    # axis holding position) rather than the seat, lifting never
                    # relieves it: the servo rails at its speed cap and
                    # elevators the part out of the corner (0901: +36mm at
                    # ~0.014 m/s, fz stuck 8-10N, y then rode 98mm above its
                    # wall and never latched). Cap the excursion at the same
                    # headroom the hard-push relief uses, so the part stays in
                    # the corner where the lateral walls can still register.
                    if contact_z is not None:
                        z_cap = contact_z + float(cfg.CASE_CORNER_RELIEF_MAX_M)
                        if z > z_cap:
                            z = z_cap
                            if not press_capped:
                                press_capped = True
                                lat_now = float(np.hypot(f_last[0], f_last[1]))
                                logger.warning(
                                    "[suction] corner descent: press servo railed "
                                    "({:.1f}N vs {:.1f}N target) — lift capped at "
                                    "{:.0f}mm above contact. Lateral is {:.1f}N, so "
                                    "this is {}", fz,
                                    float(cfg.CASE_CORNER_PRESS_N),
                                    float(cfg.CASE_CORNER_RELIEF_MAX_M) * 1000.0,
                                    lat_now,
                                    "a wall bind (lifting cannot relieve it)"
                                    if lat_now > 2.0 * float(cfg.CASE_CORNER_STOP_N)
                                    else "NOT a wall bind — a force that ignores z "
                                         "means the tare baseline is off, not contact")
                sank_now, z_span = _sink_window(z, fz)
                if sank_now and not sunk:
                    sunk, settling = True, True
                    logger.info("[suction] corner descent: slot drop at "
                                "({:.3f},{:+.3f}), z={:.4f} — pausing the drive "
                                "to settle", cmd[0], cmd[1], z)
                if settling:
                    settle_t += dt
                    if z_span < float(cfg.CASE_CORNER_SETTLE_DZ_M):
                        settling = False
                    elif settle_t > float(cfg.CASE_CORNER_SETTLE_MAX_S):
                        settling = False
                        logger.warning("[suction] corner descent: settle pause hit "
                                       "its {:.1f}s cap with z still moving "
                                       "{:.1f}mm p-p — releasing the drive anyway "
                                       "(it would otherwise stay locked until the "
                                       "timeout)",
                                       float(cfg.CASE_CORNER_SETTLE_MAX_S),
                                       z_span * 1000.0)
                else:
                    settle_t = 0.0
                press_t += dt
                if press_t > float(cfg.CASE_CORNER_TIMEOUT_S):
                    _halt(prev_q)
                    logger.warning("[suction] corner descent TIMEOUT ({:.0f}s pressing, "
                                   "in_slot={} stopped=({},{})) — misseat",
                                   press_t, sunk, stopped[0], stopped[1])
                    return PickResult(False, "misseat", z,
                                      contact_info=self._contact_snapshot(prev_q))
            if tick_cb is not None and tick_cb(z, fz):
                _halt(prev_q)
                logger.warning("[suction] descent halted by the supervisor at ee_z={:.4f}", z)
                return PickResult(False, "monitor_abort", z,
                                  contact_info=self._contact_snapshot(prev_q))
            # corner drive: on for the WHOLE descent, paused while a slot drop
            # settles and, once PRESSING, until the press servo has actually
            # reached its target (cfg.CASE_CORNER_DRIVE_PRESS_TOL_N). Friction is
            # mu * press and pushes exactly the way a wall reaction does, so
            # driving at an arbitrary press force makes the latch unreadable —
            # holding the press at its 5N design point keeps friction at a known
            # 1.4N, well under CASE_CORNER_STOP_N. Uses the SIGNED force: |fz|
            # would let a 5N PULL through as if it were the 5N press.
            # Airborne (not yet pressing) drives freely, on its own part-type
            # travel budget — see docstring.
            at_press = (fz_s is not None
                        and abs(fz_s - float(cfg.CASE_CORNER_PRESS_N))
                        <= float(cfg.CASE_CORNER_DRIVE_PRESS_TOL_N))
            # LIFT-then-drive waits on HEIGHT, not force: hanging clear of the
            # seat there is no press to reach, and the friction the press gate
            # exists to keep constant is zero.
            ready = False if lifting else (True if lift_done else at_press)
            # Lateral force monitoring runs EVERY tick, outside the drive gate:
            # the load keeps building while the drive is paused (the descent is
            # what builds it), so gating the abort/relief on the drive gate
            # meant the guard slept exactly when it was needed.
            fm = self.contact_wrench()
            if fm is not None:
                f_last = [float(fm[0][0]), float(fm[0][1])]
                lat = float(np.hypot(f_last[0], f_last[1]))
                if lat > float(cfg.CASE_CORNER_LAT_LIMIT_N):
                    _halt(prev_q)
                    logger.warning("[suction] corner descent: lateral {:.1f}N "
                                   "(fx={:+.1f}, fy={:+.1f}) over the {:.0f}N "
                                   "limit at trav=({:+.1f},{:+.1f})mm — a missed "
                                   "wall latch grinding into the wall, abort",
                                   lat, f_last[0], f_last[1],
                                   float(cfg.CASE_CORNER_LAT_LIMIT_N),
                                   (cmd[0] - x0) * 1000.0, (cmd[1] - y0) * 1000.0)
                    return PickResult(False, "force_limit", z,
                                      contact_info=self._contact_snapshot(prev_q))
                # RELIEF: an axis whose wall reaction is building gives way,
                # LATCHED OR NOT — a frozen axis cannot otherwise shed a load
                # the descent keeps adding, and its only other outcome is the
                # abort above. Capped so it can never unwind the registration.
                for i in (0, 1):
                    ch, sgn = cfg.CASE_CORNER_LAT_CHANNEL[i]
                    if f_last[ch] * sgn < float(cfg.CASE_CORNER_RELIEF_N):
                        continue
                    if relieved[i] >= float(cfg.CASE_CORNER_RELIEF_TRAVEL_M):
                        continue
                    cmd[i] -= dirs[i] * lat_step
                    relieved[i] += lat_step
                    if not relief_logged[i]:
                        relief_logged[i] = True
                        logger.warning("[suction] corner descent: {} relieving — "
                                       "{}={:+.1f}N over {:.0f}N at trav={:+.1f}mm "
                                       "(latched={}), backing off up to {:.0f}mm",
                                       ("x", "y")[i], ("fx", "fy")[ch], f_last[ch],
                                       float(cfg.CASE_CORNER_RELIEF_N),
                                       (cmd[i] - (x0, y0)[i]) * 1000.0, latched[i],
                                       float(cfg.CASE_CORNER_RELIEF_TRAVEL_M) * 1000.0)
            # Airborne, the drive also waits for the descent to SLOW DOWN: at
            # cruise the arm trails its command off-axis (0905 case place: 55mm
            # of x, 10 deg of pitch, 83mm at the cup tip), so a case driven into
            # its wall up there arrives tilted and corner-first and the reaction
            # goes straight past CASE_CORNER_STOP_N into the lateral abort
            # (15:20:35: 1.4 -> 26.3N in 90ms). Below creep_z + the blend band
            # the profile is decelerating into the creep and the deviation is
            # already shedding, so the wall is met slowly, flat and readable.
            # Travel budget for the case: ~35mm of blend (~0.055 m/s mean) plus
            # the 50mm creep gap is ~2.3s before touchdown = ~91mm of lateral at
            # CASE_CORNER_SPEED_M_S, against the 37-68mm the walls have needed.
            # Post-contact (pressing) the gate is off — that drive is uncapped
            # and is the battery's only mode (air_travel=0 never drives here).
            slowed = pressing or z <= creep_z + float(cfg.DESCENT_CREEP_BLEND_M)
            if slowed and not settling and (fz is None or not pressing or ready):
                cap = float(max_travel if pressing else air_travel)
                # A force latch means "this axis was DRIVEN into its datum
                # wall". For the battery (air_travel=0) there is no lateral
                # drive before the vertical contact, so an airborne latch cannot
                # mean that — whatever it touched on the way down is not its
                # wall, and latching freezes the axis before it ever drives
                # (0903: x latched at 0.0mm of travel, 89mm ABOVE the seat, and
                # the place then "registered" with x contributing nothing). The
                # CASE is the opposite BY DESIGN: it drives (from the blend
                # band down) and its tall bin walls legitimately catch it
                # anywhere in there (0903: latched at 37-68mm of travel, well
                # above contact), so it keeps the airborne latch — unless the
                # caller asked for corner_touch_first, which drops the case to
                # air_travel=0 precisely to give that latch up.
                latch_armed = pressing or air_travel > 0.0
                for i, name in ((0, "x"), (1, "y")):
                    if stopped[i]:
                        continue
                    # the reaction lands on the OTHER channel (the lateral
                    # wrench frame is MIRRORED vs base, x/y swapped) — log the
                    # channel that was actually tested, not the axis' namesake,
                    # or the line lies about which wall it saw (0903)
                    ch, sgn = cfg.CASE_CORNER_LAT_CHANNEL[i]
                    if latch_armed and f_last[ch] * sgn >= float(cfg.CASE_CORNER_STOP_N):
                        stopped[i] = latched[i] = True
                        logger.info("[suction] corner descent: {} wall at {:+.1f}mm "
                                    "({}={:+.1f}N, ee_z={:.4f})", name,
                                    (cmd[i] - (x0, y0)[i]) * 1000.0,
                                    ("fx", "fy")[ch], f_last[ch], z)
                    elif abs(cmd[i] - (x0, y0)[i]) >= cap:
                        if pressing:   # true cap — airborne it just holds
                            stopped[i] = True
                            logger.warning("[suction] corner descent: {} travel cap "
                                           "{:.0f}mm with NO wall contact", name,
                                           cap * 1000.0)
                    else:
                        cmd[i] += dirs[i] * lat_step
            # TEMP debug: live force/travel trace while the corner drive runs.
            # The old ee=/ee_z fields came from FK(prev_q) — the COMMANDED
            # joints — so they only ever showed the IK residual (0.1mm) and
            # were blind to what the robot actually did; _track_trace below
            # reports that split properly against the live joints.
            dbg_t += dt
            if dbg_t >= 0.25:
                dbg_t = 0.0
                fm_dbg = self.contact_wrench()
                # must mirror the real drive condition above, or the trace
                # lies about why the drive is or is not moving
                gate = ((fz is None or not pressing or ready)
                        and not settling and slowed)
                logger.info("[suction] corner dbg: z={:.4f} fz={} fz_signed={} "
                            "at_press={} "
                            "f=({},{})N m=({},{})Nm trav=({:+.1f},{:+.1f})mm "
                            "gate={} pressing={} "
                            "stopped=({},{}) latched=({},{})",
                            z,
                            "-" if fz is None else format(fz, "+.1f"),
                            "-" if fm_dbg is None else format(float(fm_dbg[0][2]), "+.1f"),
                            at_press,
                            "-" if fm_dbg is None else format(float(fm_dbg[0][0]), "+.1f"),
                            "-" if fm_dbg is None else format(float(fm_dbg[0][1]), "+.1f"),
                            "-" if fm_dbg is None else format(float(fm_dbg[1][0]), "+.2f"),
                            "-" if fm_dbg is None else format(float(fm_dbg[1][1]), "+.2f"),
                            (cmd[0] - x0) * 1000.0, (cmd[1] - y0) * 1000.0,
                            gate, pressing, stopped[0], stopped[1],
                            latched[0], latched[1])
                if trace_s and tick_n:
                    self._track_trace("corner", [cmd[0], cmd[1], z], prev_q,
                                      tick_acc / tick_n * 1000.0,
                                      extra=" trav=({:+.1f},{:+.1f})mm".format(
                                          (cmd[0] - x0) * 1000.0,
                                          (cmd[1] - y0) * 1000.0))
                    tick_acc, tick_n = 0.0, 0
            if pressing and stopped[0] and stopped[1] and not settling:
                # while the case hangs for the drive, z is the lifted hold —
                # the seat judgement belongs to the TOUCHDOWN z
                z_seat = float(contact_z) if (lifting or lift_done) else z
                in_slot = sunk or (misseat_tol_m is None
                                   or z_seat - target_ee_z <= float(misseat_tol_m))
                if in_slot and not (latched[0] and latched[1]):
                    # in the slot but an axis ran out of travel with NO wall
                    # force: seated but NOT registered — never blow-off an
                    # unregistered case (raise CASE_CORNER_MAX_TRAVEL_M if the
                    # walls are genuinely farther than the cap)
                    _halt(prev_q)
                    logger.warning("[suction] corner descent: in the slot but "
                                   "wall-latched only (x={}, y={}) — HELD for "
                                   "the operator", latched[0], latched[1])
                    return PickResult(False, "misseat", z,
                                      contact_info=self._contact_snapshot(prev_q))
                if in_slot:
                    if lift_done:
                        # both datum walls found with nothing but wall reaction
                        # in the lateral channel — now set the case back down on
                        # the press servo (which the cleared lift_done
                        # hands control back to) and finish once it is seated
                        lift_done, seating = False, True
                        logger.info("[suction] corner descent: both walls "
                                    "registered {:.1f}mm CLEAR of the seat at "
                                    "({:+.1f},{:+.1f})mm — setting the case down",
                                    (z - float(contact_z)) * 1000.0,
                                    (cmd[0] - x0) * 1000.0, (cmd[1] - y0) * 1000.0)
                    elif at_press or not seating:
                        registered = True
                        break
                else:
                    # registered in xy but no drop yet (walls that protrude above
                    # the rim stop the drive first) — keep pressing for the sink
                    grace += dt
                    if grace > float(cfg.CASE_CORNER_DROP_GRACE_S):
                        break
            sol, p = self.solve_step(prev_q, p_prev, (cmd[0], cmd[1], z), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                if not pressing:   # mirror _descend_to_contact: hard abort in air
                    _halt(prev_q)
                    return PickResult(False, "unreachable", z)
            else:
                # a shortened tick (joint-speed cap) re-bases the schedule to the
                # pose actually commanded — drive travel and press z alike
                cmd[0], cmd[1], z = float(p[0]), float(p[1]), float(p[2])
                p_prev = (cmd[0], cmd[1], z)
                self._send(sol.q, (sol.q - prev_q) / dt)
                prev_q = sol.q
            elapsed += dt
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1
            t_tick = now

        if not registered:
            # xy registered on the walls but the slot-entry signal never came
            # (grace expired) — release WITHOUT the operator gate (user 0824:
            # this case is fine to drop and move on; the taught-z in_slot
            # judgement is the usual culprit, not the case)
            _halt(prev_q)
            logger.warning("[suction] corner descent: in_slot=False with both "
                           "axes stopped after {:.1f}s grace — auto release, "
                           "run continues", grace)
            return PickResult(False, "misseat", z, auto_release=True,
                              contact_info=self._contact_snapshot(prev_q))

        # relieve the wall preload (both axes latched here) before the release
        bx = cmd[0] - dirs[0] * float(backoff)
        by = cmd[1] - dirs[1] * float(backoff)
        n = max(1, int(round(float(np.hypot(bx - cmd[0], by - cmd[1])) / lat_step)))
        pace = TickPacer(dt)
        for i in range(1, n + 1):
            sol = self.solve_pose([cmd[0] + (bx - cmd[0]) * i / n,
                                   cmd[1] + (by - cmd[1]) * i / n, z],
                                  rpy, seed=prev_q, min_motion=True)
            if sol.pos_err_m <= cfg.REACH_TOL_M:
                self._send(sol.q, (sol.q - prev_q) / dt)
                prev_q = sol.q
            pace.wait()
        _halt(prev_q)
        logger.info("[suction] corner descent: registered on both walls at "
                    "({:+.1f},{:+.1f})mm from the aim, seat z={:.4f}",
                    (cmd[0] - x0) * 1000.0, (cmd[1] - y0) * 1000.0, z)
        # no final settle press (it read as a needless extra ~10N push): the
        # press servo already holds the seat at ~CASE_CORNER_PRESS_N, so the
        # commanded z IS the contact depth within ~1mm of elastic compression
        res = PickResult(True, "contact", z,
                         contact_info=self._contact_snapshot(prev_q))
        res.recover_history = [{
            "attempt": 1, "mode": "corner", "dyaw_deg": 0.0,
            "dx_mm": (cmd[0] - x0) * 1000.0, "dy_mm": (cmd[1] - y0) * 1000.0,
            "reason": res.reason,
            "z_mm": (z - target_ee_z) * 1000.0,
            "fx": f_last[0], "fy": f_last[1], "mz": None,
        }]
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
        if suction_io.is_suction_commanded_on():
            # re-assert only when needed (two HTTP calls + 0.5s settle) — the
            # cup is already OFF on a normal pick
            suction_io.suction_off()
        # empty-cup tare happens IN-STREAM during the scan-descent (_descend_open)

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
        res = self._seal_with_retry(rpy)
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

    def _descend_open(self, z_floor, rpy, start_q, force_limit,
                      halt_at_floor: bool = True):
        """Descend (suction unchanged) straight down to z_floor — no soft-contact
        stop (the goal is the floor). Ramp speed in and decelerate to creep speed
        into z_floor (so the halt / creep-seal handoff isn't a velocity step);
        abort on force>force_limit. The rolling wrench reference warms up in
        stream (cfg.WRENCH_REF_WARMUP_S, ~0.1s) — no tare, no stationary hover
        pause; the force check is off only for that warm-up.
        Returns (last_q, z, reason) with reason in
        {at_floor, force_limit, unreachable}."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        elapsed = 0.0
        self.freeze_reference(False)      # fresh rolling reference per descent
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()

        def _halt(q):
            self._send(np.asarray(q), np.zeros(len(q)))

        pace = TickPacer(dt)
        while z > z_floor + 1e-4:
            speed = self._descent_speed(z, z_floor, elapsed)
            z_next = max(z_floor, z - speed * dt)
            sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q); return prev_q, z, "unreachable"
            z_next = float(p[2])
            self._send(sol.q, (sol.q - prev_q) / dt)
            f = self.vertical_force()
            if f is not None and f > force_limit:
                _halt(sol.q); self.freeze_reference()
                logger.warning("[suction] hard push {:.1f}N at ee_z={:.4f} during scan-descent", f, z_next)
                return sol.q, z_next, "force_limit"
            z, prev_q = z_next, sol.q
            elapsed += dt
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                self._track_trace("pick descend", [x, y, z], prev_q,
                                  tick_acc / tick_n * 1000.0)
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0
        # z_floor is the CREEP line, and the profile has already decelerated to
        # creep speed by here — a caller that continues straight into a creep
        # (pick / pick_retreat) passes halt_at_floor=False so the velocity runs
        # on unbroken instead of being notched to zero for the handoff.
        # pick_gated must still stop: it queries the barcode scanner and may
        # sweep sideways next.
        if halt_at_floor:
            _halt(prev_q)
        return prev_q, z, "at_floor"

    def _creep_seal(self, rpy, start_q, contact_n: "float | None" = None,
                    force_limit: "float | None" = None,
                    start_elapsed: float = 0.0) -> PickResult:
        """Suction already ON. Creep straight down until the vacuum seals (DI0,
        primary) or force contact (then hold + wait for the seal). Abort on hard
        force.

        ``start_elapsed``: seed the ramp-in clock (see _creep_to_force) when the
        caller's descent already runs at creep speed and did NOT halt.

        ``contact_n`` / ``force_limit`` override FORCE_CONTACT_THRESHOLD_N and
        FORCE_HARD_LIMIT_N for one pick. A CARDBOARD lid dents under the global
        pair, which was sized for the case/battery (cfg.BOX_LID_CONTACT_N)."""
        touch = float(cfg.FORCE_CONTACT_THRESHOLD_N if contact_n is None
                      else contact_n)
        hard = float(cfg.FORCE_HARD_LIMIT_N if force_limit is None else force_limit)
        if contact_n is not None or force_limit is not None:
            logger.info("[suction] creep-seal thresholds: contact {:.1f}N, abort "
                        "{:.1f}N (default {:.1f}/{:.1f})", touch, hard,
                        cfg.FORCE_CONTACT_THRESHOLD_N, cfg.FORCE_HARD_LIMIT_N)
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        descended = 0.0
        elapsed = float(start_elapsed)
        self.freeze_reference(False)      # fresh rolling reference
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        vac = suction_io.VacuumMonitor(); vac.start()

        def _halt(q):
            self._send(np.asarray(q), np.zeros(len(q)))

        try:
            pace = TickPacer(dt)
            while descended < cfg.DESCENT_MAX_M:
                if vac.is_sealed():
                    _halt(prev_q)
                    logger.info("[suction] vacuum sealed at ee_z={:.4f}", z)
                    return PickResult(True, "sealed", z)
                z_next = z - self._creep_speed(elapsed) * dt
                sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
                if sol.pos_err_m > cfg.REACH_TOL_M:
                    _halt(prev_q); return PickResult(False, "unreachable", z)
                z_next = float(p[2])
                self._send(sol.q, (sol.q - prev_q) / dt)
                f = self.vertical_force()
                if f is not None:
                    if f > hard:
                        _halt(sol.q); return PickResult(False, "force_limit", z_next)
                    if f > touch:
                        # touched — freeze the reference, hold, await the seal
                        _halt(sol.q); self.freeze_reference()
                        deadline = time.time() + cfg.VACUUM_SEAL_TIMEOUT_S
                        while time.time() < deadline:
                            self._send(sol.q, np.zeros(len(sol.q)))
                            if vac.is_sealed():
                                logger.info("[suction] sealed after contact at ee_z={:.4f}", z_next)
                                return PickResult(True, "sealed", z_next)
                            time.sleep(0.05)
                        return PickResult(False, "vacuum_timeout", z_next)
                descended += (z - z_next); z, prev_q = z_next, sol.q
                elapsed += dt
                pace.wait()
                now = time.perf_counter()
                tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
                t_tick = now
                if trace_s and trace_t >= trace_s:
                    self._track_trace("creep-seal", [x, y, z], prev_q,
                                      tick_acc / tick_n * 1000.0,
                                      extra=" f={}".format("-" if f is None
                                                           else format(f, "+.1f")))
                    trace_t, tick_acc, tick_n = 0.0, 0.0, 0
            _halt(prev_q)
            return PickResult(False, "max_descent", z)
        finally:
            # Off the critical path: stop() blocks ~1.3s on the socketio
            # disconnect, which held the arm frozen between seal and lift
            # (the post-seal dwell visible in every collected take).
            threading.Thread(target=vac.stop, daemon=True).start()

    def _set_down(self, rpy, touch_n: float = 3.0, max_drop_m: float = 0.05) -> None:
        """Lower a HELD part straight down to a light touch before a release
        (auto-release path: the press servo may have lifted the part while
        unwinding the contact overshoot — a blow-off from height re-creates
        the 0806 release-drop slide). Ignores the vacuum state (a held part
        reads sealed); stops on ``touch_n``, ``max_drop_m``, or reach failure.
        Best-effort: any stop just releases from wherever it got to."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(self._start_q(), dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        dropped = 0.0
        elapsed = 0.0
        self.freeze_reference(False)      # fresh rolling reference
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        pace = TickPacer(dt)
        while dropped < float(max_drop_m):
            z_next = z - self._creep_speed(elapsed) * dt
            sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                break
            z_next = float(p[2])
            self._send(sol.q, (sol.q - prev_q) / dt)
            f = self.vertical_force()
            if f is not None and f > float(touch_n):
                self.freeze_reference()
                logger.info("[suction] set-down touch {:.1f}N at ee_z={:.4f}", f, z_next)
                break
            dropped += (z - z_next)
            z, prev_q = z_next, sol.q
            elapsed += dt
            pace.wait()
            now = time.perf_counter()
            tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
            t_tick = now
            if trace_s and trace_t >= trace_s:
                self._track_trace("set-down", [x, y, z], prev_q,
                                  tick_acc / tick_n * 1000.0,
                                  extra=" f={}".format("-" if f is None
                                                       else format(f, "+.1f")))
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0
        self._send(prev_q, np.zeros(len(prev_q)))

    def _creep_to_force(self, rpy, start_q, touch_n: float,
                        start_elapsed: float = 0.0):
        """Suction already ON. Creep straight down until the tared vertical
        force exceeds ``touch_n`` ('touched'), the vacuum seals ('sealed'), or
        abort ('force_limit'/'unreachable'/'max_descent'). Returns
        (last_q, z, reason). pick_retreat's press leg — unlike _creep_seal it
        does NOT hold and wait for the seal at contact.

        ``start_elapsed``: seed the ramp-in clock past DESCENT_RAMP_S when the
        caller's own descent already reached creep speed and did NOT halt
        (_descend_open(..., halt_at_floor=False)) — otherwise this loop's
        elapsed=0 first tick commands 0 m/s right after the previous stream's
        last tick commanded creep speed, a one-tick velocity step exactly like
        the 0 -> creep step _creep_speed's ramp-in exists to avoid."""
        dt = 1.0 / float(cfg.CONTROL_HZ)
        prev_q = np.asarray(start_q, dtype=float)
        pos, _ = self.fk(prev_q)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        descended = 0.0
        elapsed = float(start_elapsed)
        self.freeze_reference(False)      # fresh rolling reference
        trace_s = float(cfg.DESCENT_TRACE_S)
        trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
        vac = suction_io.VacuumMonitor(); vac.start()

        def _halt(q):
            self._send(np.asarray(q), np.zeros(len(q)))

        try:
            pace = TickPacer(dt)
            while descended < cfg.DESCENT_MAX_M:
                if vac.is_sealed():
                    _halt(prev_q)
                    logger.info("[suction] sealed mid-press at ee_z={:.4f}", z)
                    return prev_q, z, "sealed"
                z_next = z - self._creep_speed(elapsed) * dt
                sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_next), rpy, dt)
                if sol.pos_err_m > cfg.REACH_TOL_M:
                    _halt(prev_q); return prev_q, z, "unreachable"
                z_next = float(p[2])
                self._send(sol.q, (sol.q - prev_q) / dt)
                f = self.vertical_force()
                if f is not None:
                    if f > cfg.FORCE_HARD_LIMIT_N:
                        _halt(sol.q); self.freeze_reference()
                        return sol.q, z_next, "force_limit"
                    if f > touch_n:
                        _halt(sol.q); self.freeze_reference()
                        logger.info("[suction] bounce touch {:.1f}N at ee_z={:.4f} — retreat", f, z_next)
                        return sol.q, z_next, "touched"
                descended += (z - z_next); z, prev_q = z_next, sol.q
                elapsed += dt
                pace.wait()
                now = time.perf_counter()
                tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
                t_tick = now
                if trace_s and trace_t >= trace_s:
                    self._track_trace("creep-press", [x, y, z], prev_q,
                                      tick_acc / tick_n * 1000.0,
                                      extra=" f={}".format("-" if f is None
                                                           else format(f, "+.1f")))
                    trace_t, tick_acc, tick_n = 0.0, 0.0, 0
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
            self._send(np.asarray(q), np.zeros(len(q)))

        traveled = elapsed = 0.0
        pace = TickPacer(dt)
        while traveled < dist:
            r = min(1.0, elapsed / max(float(cfg.DESCENT_RAMP_S), 1e-6))
            speed = float(cfg.BCR_SWEEP_SPEED_M_S) * (r * r * (3.0 - 2.0 * r))
            step = min(speed * dt, dist - traveled)
            p_next = p + u * step
            sol, p_next = self.solve_step(prev_q, p, p_next, rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q)
                logger.warning("[suction] sweep line IK failed at ({:.3f},{:+.3f}) — "
                               "pass cut short", p_next[0], p_next[1])
                return scanner.result()
            self._send(sol.q, (sol.q - prev_q) / dt)
            prev_q = sol.q
            traveled += float(np.dot(p_next - p, u))
            p = p_next
            elapsed += dt
            code = scanner.result()
            if code is not None:
                _halt(sol.q)
                logger.info("[suction] barcode read mid-sweep at ({:.3f},{:+.3f})",
                            p[0], p[1])
                return code
            pace.wait()
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
