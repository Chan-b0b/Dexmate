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


def _raster_offsets(x_step: float, max_x: float, y_step: float, max_y: float) -> list[tuple[float, float]]:
    """Raster (dx, dy) offsets in the case-local frame: for each y row (0,
    +y_step, -y_step, ... out to max_y — a small perturbation), sweep x
    monotonically from -max_x to +max_x (a deliberate wide excursion, not
    jittering) before moving to the next y row. Used by the barcode sweep
    when no read happens by creep_z."""
    x_vals = []
    v = -max_x
    while v <= max_x + 1e-9:
        x_vals.append(v)
        v += x_step
    out: list[tuple[float, float]] = []
    for dy in _axis_steps(y_step, max_y):
        for dx in x_vals:
            out.append((dx, dy))
    return out


@dataclass
class PickResult:
    success: bool
    reason: str                      # contact / sealed / force_limit / vacuum_timeout / max_descent / unreachable
    contact_ee_z: float | None = None
    barcode: str | None = None


class SuctionMover(ArmMover):
    """Suction pick/place on the suction arm (cfg.ARM_SIDE)."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side=cfg.ARM_SIDE, ee_frame=cfg.EE_FRAME)
        self._wrench = getattr(self._arm, "wrench_sensor", None)
        if self._wrench is None:
            logger.warning("[suction] {} arm has no wrench sensor — contact detection OFF", self._side)
        self._force_baseline = np.zeros(3)

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
            samples.append(self._wrench.get_wrench_state()[:3].astype(float))
            time.sleep(0.005)
        self._force_baseline = np.mean(samples, axis=0)
        logger.info("[suction] tared (baseline |f|={:.2f}N)", float(np.linalg.norm(self._force_baseline)))

    def vertical_force(self) -> float | None:
        """|base-vertical component| of the tared contact force (N), or None."""
        if self._wrench is None:
            return None
        raw = self._wrench.get_wrench_state()[:3].astype(float) - self._force_baseline
        return float(abs((self.current_ee_rotation() @ raw)[2]))

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
                            start_q: np.ndarray) -> PickResult:
        """Descend straight down (x,y,rpy held) until contact / hard-limit / floor.

        Streams per-tick IK with a finite-diff velocity feedforward. Two-speed:
        fast until ``DESCENT_CREEP_GAP_M`` above the expected contact z, then a
        slow creep so one reaction tick can't over-press. Suction state is the
        caller's responsibility (off for pick approach).

        ``start_q`` is the approach move's commanded target joints: the descent
        continues the stream from exactly there (its FK pose, not a fresh live
        solve) so there's no command discontinuity at the handoff.
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
                    return PickResult(False, "force_limit", z_next)
                if f > cfg.FORCE_CONTACT_THRESHOLD_N:
                    _halt(sol.q)
                    logger.info("[suction] contact {:.1f}N at ee_z={:.4f}", f, z_next)
                    return PickResult(True, "contact", z_next)

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
    def _lift_to_transport(self, rpy) -> None:
        """Lift to SAFE_TRANSPORT_Z: straight up (xy held per tick) to
        LIFT_CLEAR_EE_Z — clear of the case walls from any layer's pick —
        then the remaining free-air ascent as a faster joint-space move_ee
        (endpoint xy held; the arc in between is harmless up there)."""
        pos, _ = self.current_ee_pose()
        z_clear = min(max(float(pos[2]), cfg.LIFT_CLEAR_EE_Z), cfg.SAFE_TRANSPORT_Z)
        q = self.move_ee_vertical(z_clear, rpy)
        if q is None or z_clear >= cfg.SAFE_TRANSPORT_Z:
            return
        x, y = self.fk(q)[0][:2]
        self.move_ee([float(x), float(y), cfg.SAFE_TRANSPORT_Z], rpy, quiet=True)

    def _approach_and_hover(self, ee_pos, rpy, ez):
        """Travel to the column at transport height (sideways clearance), then
        drop straight to a REACHABLE hover over the true xy.

        The transport approach can fall a few mm short horizontally when the arm
        nears its reach limit up high; the lower hover (ez + HOVER_HEIGHT_M) is
        well inside reach, so the arm recovers the true xy before the vertical
        descent (which holds xy) — preventing an offset, misaligned seat.
        Returns the aligned hover config, or None if either leg is unreachable."""
        approach = np.array([ee_pos[0], ee_pos[1], cfg.SAFE_TRANSPORT_Z])
        if self.move_ee(approach, rpy, quiet=True) is None:  # transport leg: shortfall recovered by the hover
            return None
        hover_z = min(cfg.SAFE_TRANSPORT_Z, ez + cfg.HOVER_HEIGHT_M)
        return self.move_ee([ee_pos[0], ee_pos[1], hover_z], rpy)

    def pick(self, pose, expected_z=None) -> PickResult:
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
        res = self._creep_seal(rpy, self._live_arm_q())
        if res.success:
            # Relieve the creep-contact press before lifting (mirrors place()'s
            # RELEASE_PRELIFT_M) — otherwise the lift's first motion has to break
            # the residual seat press while already carrying the part.
            if cfg.SEAL_PRELIFT_M > 0.0:
                pos, _ = self.current_ee_pose()
                self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
            # Lift straight up to clear, then to transport — ready to travel.
            self._lift_to_transport(rpy)
        else:
            suction_io.suction_off()
        return res

    def place(self, pose, expected_z=None) -> PickResult:
        """Hover above the seat, descend to contact within buffer, release.
        ``expected_z`` overrides the taught seat z (layer stacking)."""
        ee_pos, rpy = self.taught_target(pose)
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] place: approach@transport -> hover -> descend -> release")
        q_hover = self._approach_and_hover(ee_pos, rpy, ez)
        if q_hover is None:
            return PickResult(False, "unreachable")
        self.tare()  # battery in cup, hovering (no contact yet)
        res = self._descend_to_contact(ez, rpy, cfg.FORCE_HARD_LIMIT_PLACE_N, q_hover)
        # Release even on force_limit — we're at the seat; blow the part off.
        pos, _ = self.current_ee_pose()
        if cfg.RELEASE_PRELIFT_M > 0.0:
            self.move_ee_vertical(pos[2] + cfg.RELEASE_PRELIFT_M, rpy)
        suction_io.release()
        self._lift_to_transport(rpy)
        res.success = res.reason in ("contact",)
        return res

    # ------------------------------------------------------------------
    # Barcode-gated battery pick
    # ------------------------------------------------------------------
    def pick_gated(self, pose, case_center=None, expected_z=None) -> PickResult:
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
            self._lift_to_transport(rpy)
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
            vac.stop()

    def _sweep_scan(self, ee_pos, rpy, scanner, case_center):
        """Raised a bit above the creep z — no tilt — raster x/y around the
        pick point looking for a read: x sweeps its full far-to-close range at
        each fixed y, staying on the battery's side of the case center.
        Returns the agreed code or None if exhausted (arm left over the pick point
        at the raised z; the following creep-seal descends from there)."""
        cx, cy, _cz, cyaw = case_center
        c, s = float(np.cos(cyaw)), float(np.sin(cyaw))
        z = float(self.current_ee_pose()[0][2]) + cfg.BCR_SWEEP_LIFT_M   # lift to sweep
        bx, by = float(ee_pos[0]) - cx, float(ee_pos[1]) - cy
        loc_by = -s * bx + c * by                          # battery's case-local y (side)
        for dx, dy in _raster_offsets(cfg.BCR_SEARCH_X_STEP_M, cfg.BCR_SEARCH_MAX_X_M,
                                      cfg.BCR_SEARCH_Y_STEP_M, cfg.BCR_SEARCH_MAX_Y_M):
            if loc_by != 0.0 and (loc_by + dy) * loc_by < 0.0:
                continue                                    # would cross toward the other slot
            wx = float(ee_pos[0]) + c * dx - s * dy
            wy = float(ee_pos[1]) + s * dx + c * dy
            if self.move_ee([wx, wy, z], rpy) is None:
                continue
            time.sleep(cfg.BCR_SCAN_DWELL_S)
            code = scanner.result()
            if code is not None:
                # back over the real pick point before the seal descent
                self.move_ee([float(ee_pos[0]), float(ee_pos[1]), z-cfg.BCR_SWEEP_LIFT_M], rpy)
                return code
        self.move_ee([float(ee_pos[0]), float(ee_pos[1]), z-cfg.BCR_SWEEP_LIFT_M], rpy)  # return over the pick point
        return scanner.result()


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
