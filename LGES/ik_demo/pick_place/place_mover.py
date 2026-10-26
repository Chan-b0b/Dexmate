"""PlaceMover: SuctionMover with the PLACE path copied out for redesign.

The methods below are verbatim copies from suction.SuctionMover (as of
2026-10-08), so editing them here changes only the pick_place package — chassis_sequence
and everything else keep using suction.py. Copied: place (flow + release gate),
_descend_corner_seat (wall push-in / contact / seat decision), _misseat_recover,
_column_reachable, and the hover / lift / set-down steps place calls.
Everything else (pick, descend_to_contact, force reads, tare) is inherited.
Not a copy (1012): _descend_straight, the place descent with no corner drive.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from .place_config import (cfg, PICK_WALL_RELIEF_M, LIFT_VERTICAL_TO_Z,
                           PLACE_RELEASE_SETTLE_S)
from ..arm import TickPacer
from ..drivers import suction_io
from ..suction import PickResult, SuctionMover


class PlaceMover(SuctionMover):
    def pin_torso(self, target=None, vel_scale=None, timeout=None) -> None:
        """arm.py reads config.TORSO_JOINTS directly, so route the default
        target through place_config's OVERRIDES."""
        if target is None:
            target = cfg.TORSO_JOINTS
        super().pin_torso(target, vel_scale, timeout)

    def pick(self, pose, expected_z=None, lift_to_clear: bool = False,
             creep_gap: "float | None" = None,
             contact_n: "float | None" = None,
             force_limit: "float | None" = None,
             retry_offset_m: "float | None" = None,
             lift_creep_out_m: float = 0.0,
             retry_dir: "tuple[float, float] | None" = None) -> PickResult:
        """SuctionMover.pick with the fast descent PLANNED instead of streamed:
        transport -> hover -> creep line is one ruckig stream that ends at rest
        (_approach_and_hover to_creep_z) and _settle_at confirms the arrival
        before the creep starts. The per-tick fast descent it replaces led the
        commanded path by 30-45mm / 7 deg at the creep handoff, and _start_q
        then restarted the creep from the LIVE (leading) config, so the cup
        sealed ~28mm forward of the taught pose in every run (0.707 -> 0.7356).
        Long creep gap: expected_z is a model guess, the planned leg is not
        force-stopped by contact the way the per-tick descent is."""
        ee_pos, rpy = self.taught_target(pose)
        self._pick_aim_xy = (float(ee_pos[0]), float(ee_pos[1]))
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] pick: approach@transport -> hover -> creep line planned "
                    "as one stream -> settle -> suction on -> creep-seal")
        if suction_io.is_suction_commanded_on():
            suction_io.suction_off()
        hard = float(cfg.FORCE_HARD_LIMIT_N if force_limit is None else force_limit)
        q = self._approach_and_hover(
            ee_pos, rpy, ez, to_creep_z=True, tick_cb=self._approach_force_guard(hard),
            creep_gap=float(cfg.DESCENT_CREEP_GAP_M if creep_gap is None else creep_gap))
        if q is None:
            return PickResult(False, "unreachable")
        suction_io.suction_on_async()
        return self._seal_and_lift(rpy, lift_to_clear,
                                   lift_creep_out_m=lift_creep_out_m,
                                   retry_dir=retry_dir, contact_n=contact_n,
                                   force_limit=force_limit,
                                   retry_offset_m=retry_offset_m)

    def _seal_and_lift(self, rpy, lift_to_clear: bool = False,
                       lift_creep_out_m: float = 0.0, retry_dir=None,
                       contact_n=None, force_limit=None, retry_offset_m=None,
                       start_elapsed: float = 0.0) -> PickResult:
        """SuctionMover._seal_and_lift + a wall-relief slide before the lift: a
        case seated in the bin corner (corner_seat place) touches the x/y walls,
        and lifting it straight up drags the bin toward the robot (0.825 -> 0.711
        in x over 3 runs). After the seal and pre-lift, slide the cup
        PICK_WALL_RELIEF_M diagonally away from the corner, then lift as usual."""
        res = self._seal_with_retry(rpy, retry_dir, contact_n=contact_n,
                                    force_limit=force_limit,
                                    retry_offset_m=retry_offset_m,
                                    start_elapsed=start_elapsed)
        if not res.success:
            suction_io.suction_off()
            return res
        if getattr(self, "_pick_aim_xy", None) is not None:
            self._tip_snapshot("pick seal", self._pick_aim_xy, rpy)
        if cfg.SEAL_PRELIFT_M > 0.0:
            pos, _ = self.current_ee_pose()
            self.move_ee_vertical(pos[2] + cfg.SEAL_PRELIFT_M, rpy)
        if PICK_WALL_RELIEF_M > 0.0:
            pos, _ = self.current_ee_pose()
            dx = -np.sign(cfg.CASE_CORNER_DIR[0]) * PICK_WALL_RELIEF_M
            dy = -np.sign(cfg.CASE_CORNER_DIR[1]) * PICK_WALL_RELIEF_M
            logger.info("[pick] wall relief: slide ({:+.1f},{:+.1f})mm off the corner walls "
                        "before the lift", dx * 1000, dy * 1000)
            if self.move_ee_line([pos[0] + dx, pos[1] + dy, pos[2]], rpy) is None:
                logger.warning("[pick] wall relief move failed — lifting from here")
        self._lift_to_transport(rpy, to_clear_only=lift_to_clear,
                                creep_out_m=lift_creep_out_m)
        return res

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
        z_vert = min(z_clear, max(float(pos[2]), float(LIFT_VERTICAL_TO_Z)))
        q = self.move_ee_vertical(z_vert, rpy, creep_out_m=creep_out_m)
        if q is None:
            self._best_effort_ascent(rpy)
            return
        if z_vert < z_clear - 1e-3:
            # above the walls: one planned move for the rest of the wall-clear rise
            x, y = self.fk(q)[0][:2]
            if self.move_ee([float(x), float(y), z_clear], rpy, quiet=True) is None:
                self._best_effort_ascent(rpy)
                return
        if to_clear_only or z_clear >= cfg.SAFE_TRANSPORT_Z:
            return
        x, y = self.fk(q)[0][:2]
        if self.move_ee([float(x), float(y), cfg.SAFE_TRANSPORT_Z], rpy, quiet=True) is None:
            self._best_effort_ascent(rpy)

    def _approach_and_hover(self, ee_pos, rpy, ez, to_creep_z: bool = False,
                            tick_cb=None, approach_z: "float | None" = None,
                            creep_gap: "float | None" = None,
                            hover_m: "float | None" = None):
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
        into the creep waypoint. NOT for either corner seat (see place()'s
        ``to_creep``): the case drives sideways from the hover on down
        (air_travel=max, tall bin walls) and earns its 37-68mm of lateral
        travel during that stretch, which a planned move would take away; and
        for the battery this leg WAS the whole descent into the slot, which the
        arm tracked 56-94mm behind at up to 18 deg of pitch (0909).

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
        zs = [min(z_app, ez + float(cfg.HOVER_HEIGHT_M if hover_m is None else hover_m))]
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

    def place(self, pose, expected_z=None, misseat_tol_m=None, tick_cb=None,
              backoff_m: "float | None" = None,
              lift_to_clear: bool = False,
              corner_seat: "str | None" = None,
              corner_touch_first: bool = False,
              approach_z: "float | None" = None,
              lift_z: "float | None" = None,
              hover_m: "float | None" = None,
              release_above_m: "float | None" = None) -> PickResult:
        """Hover above the seat, descend to contact within buffer, release.
        ``corner_seat=None`` (pick_place, 1012): straight down from the hover
        onto the aim, no lateral drive (_descend_straight); ``hover_m``
        overrides HOVER_HEIGHT_M for it. ``release_above_m`` (with corner_seat=None):
        stop that far above the expected z and release there — no force stop, no
        prelift; a contact before that height is held for the operator.
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
        walls have latched. The battery never gets the lift (low slot walls). Used by the standalone
        one-case-to-bin task (case_to_bin), whose bin is empty — the case lands
        on the floor, not on a case below. The SEQUENCE's first case does NOT
        use it (0910): it keeps the case's own creep-band drive like every case
        above it."""
        ee_pos, rpy = self.taught_target(pose)
        # NOTE corner_seat: the *_CORNER_AIM_BIAS_M shift away from the datum
        # corner is applied by the CALLER (chassis_sequence.run_item) BEFORE
        # its reach pre-check, so the checked pose is the flown pose — no
        # bias is added here.
        ez = float(ee_pos[2]) if expected_z is None else float(expected_z)
        logger.info("[suction] place: approach@transport -> hover -> descend -> release")
        # Every place ends the planned stream at the HOVER and descends per-tick
        # from there. The plain place used to plan on down to the creep line;
        # 1012 it takes the corner seats' per-tick descent (minus the drive) for
        # the battery reason below. The corner seats' two reasons:
        #   case    — it drives sideways WHILE descending (air_travel=max) and
        #             earns its 37-68mm of lateral travel over the creep-band
        #             stretch, which a planned leg ending at ez +
        #             DESCENT_CREEP_GAP_SETTLED_M would fly straight past.
        #   battery — 0910: the planned hover->creep leg was the whole descent
        #             into the slot, and the arm could not track it. Five
        #             battery places on 0909 measured the arm 56-94mm behind in
        #             x and 81-172mm behind in z at 8.5-18.4 deg of pitch (tip
        #             80-141mm off line) at the moment the battery enters the
        #             case mouth; the command was already AT the creep line, so
        #             the last ~150mm was the arm's own servo closing that error
        #             at whatever speed it liked, with no speed profile and no
        #             force guard over it. _settle_at only reads xy, so it
        #             logged "settled 1.97mm" straight through the plunge.
        #             The per-tick path caps the cruise (CORNER_DESCENT_SPEED_M_S),
        #             blends fast->creep over DESCENT_DECEL_BAND_M and guards
        #             force every tick — the same profile the case has used
        #             since 0905 and the case PICK since 0909.
        q_hover = self._approach_and_hover(
            ee_pos, rpy, ez, approach_z=approach_z,
            hover_m=(cfg.BATTERY_PLACE_HOVER_M if corner_seat == "battery" else hover_m))
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
            res = self._descend_straight(
                ez, rpy, q_hover, tick_cb=tick_cb,
                stop_z=None if release_above_m is None else ez + float(release_above_m))
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
        if res.reason not in ("contact", "at_height"):
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
        elif res.reason != "at_height":    # at_height: release right where it stopped
            pos, _ = self.current_ee_pose()
            if cfg.RELEASE_PRELIFT_M > 0.0:
                self.move_ee_vertical(pos[2] + cfg.RELEASE_PRELIFT_M, rpy)
        self._tip_snapshot("place release", ee_pos, rpy)
        suction_io.release()
        if lift_z is not None:
            # low column: straight back up to where the approach came in
            self.move_ee_vertical(float(lift_z), rpy)
        else:
            # lift_to_clear: stop at the wall-clear height (cup EMPTY here) — the
            # caller starts the return chassis leg immediately and folds the
            # remaining rise into the parallel view park (both target z=1.10)
            self._lift_to_transport(rpy, to_clear_only=lift_to_clear)
        res.success = res.reason in ("contact", "at_height")
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
        CORNER_SPEED_M_S) from wherever the commanded descent speed has fallen
        to CASE_CORNER_DRIVE_GATE_M_S on down, where the descent is decelerating and
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
        speed = float(cfg.CORNER_DESCENT_SPEED_M_S)   # last commanded descent
        # speed; the lateral drive's airborne gate reads it (only the descent
        # branch below reassigns it, and the gate short-circuits once pressing)
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
                                                fast=cfg.CORNER_DESCENT_SPEED_M_S,
                                                touch_z=target_ee_z)
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
            # (15:20:35: 1.4 -> 26.3N in 90ms). Once the commanded speed is down
            # to CASE_CORNER_DRIVE_GATE_M_S the profile is well into its
            # deceleration and the deviation is already shedding, so the wall is
            # met slowly, flat and readable. The gate is on the SPEED and not,
            # as before, on a height that happened to mean that speed under the
            # old flat-creep profile (see the config comment for what the
            # tapered one costs the case's airborne travel budget: 35mm at
            # CASE_CORNER_SPEED_M_S, against the 37-68mm the walls have needed,
            # with the remainder falling after first contact).
            # Post-contact (pressing) the gate is off — that drive is uncapped
            # and is the battery's only mode (air_travel=0 never drives here).
            slowed = pressing or speed <= float(cfg.CASE_CORNER_DRIVE_GATE_M_S)
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

    def _tip_snapshot(self, tag: str, aim_xy, aim_rpy) -> None:
        """1012 diagnosis — the place lands -x even when the case goes back on the
        spot it was picked from, so the camera's view is the same at both ends
        and the arm is the suspect: where the CUP TIP is vs the aim xy at the
        moment that fixes the part (pick seal / place release), from the
        commanded and the live joints. The tip, not the EE: a live tilt swings
        it by SUCTION_LENGTH_M * sin(tilt). Also the EE tilt itself (cmd / live
        rpy minus the aim rpy): only what the JOINT ENCODERS see — link flex or
        gear backlash under the load is invisible to FK. Never raises."""
        try:
            L = float(cfg.SUCTION_LENGTH_M)

            def tip(q):
                p, rpy = self.fk(np.asarray(q, dtype=float))
                return p + Rotation.from_euler("xyz", rpy).as_matrix()[:, 2] * L

            t_cmd, t_live = tip(self._q_cmd), tip(self._live_arm_q())
            ax, ay = float(aim_xy[0]), float(aim_xy[1])

            def tilt(q):   # deg, wrapped to [-180, 180)
                r = np.asarray(self.fk(np.asarray(q, dtype=float))[1])
                return np.rad2deg((r - np.asarray(aim_rpy, dtype=float) + np.pi)
                                  % (2.0 * np.pi) - np.pi)

            r_cmd, r_live = tilt(self._q_cmd), tilt(self._live_arm_q())
            logger.info("[{}] cup tip - aim ({:.4f},{:+.4f}): cmd ({:+.1f},{:+.1f})mm, "
                        "LIVE ({:+.1f},{:+.1f})mm, tip z live {:.4f} | EE rpy - aim: "
                        "cmd ({:+.2f},{:+.2f},{:+.2f})deg, LIVE ({:+.2f},{:+.2f},{:+.2f})deg",
                        tag, ax, ay,
                        (t_cmd[0] - ax) * 1000.0, (t_cmd[1] - ay) * 1000.0,
                        (t_live[0] - ax) * 1000.0, (t_live[1] - ay) * 1000.0,
                        float(t_live[2]), *r_cmd, *r_live)
        except Exception as e:  # noqa: BLE001 — a diagnostic must not stop a place
            logger.warning("[{}] cup tip snapshot failed ({})", tag, e)

    def _descend_straight(self, target_ee_z: float, rpy, start_q: np.ndarray,
                          tick_cb=None, stop_z: "float | None" = None) -> PickResult:
        """pick_place's place descent (1012): the corner seat's VERTICAL profile —
        per-tick from the hover, CORNER_DESCENT_SPEED_M_S cruise, decelerating
        into ``target_ee_z`` (touch taper) — with no lateral drive: x,y,rpy held
        every tick, stop at the first FORCE_CONTACT_THRESHOLD_N. As in
        _descend_corner_seat, a first touch that the creep's tracking lag carries
        past the hard limit is still the contact, not an abort. unreachable /
        max_descent / monitor_abort mirror _descend_to_contact.

        ``stop_z``: descend to this height instead (decelerating into it), hold
        PLACE_RELEASE_SETTLE_S for the arm to catch up and return "at_height";
        a force contact on the way is "early_contact" (failure)."""
        if stop_z is not None:
            target_ee_z = float(stop_z)
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
            speed = self._descent_speed(z, creep_z, elapsed,
                                        fast=cfg.CORNER_DESCENT_SPEED_M_S,
                                        touch_z=target_ee_z)
            z_cmd = z - speed * dt
            if stop_z is not None:
                z_cmd = max(z_cmd, target_ee_z)
            sol, p = self.solve_step(prev_q, (x, y, z), (x, y, z_cmd), rpy, dt)
            if sol.pos_err_m > cfg.REACH_TOL_M:
                _halt(prev_q)
                return PickResult(False, "unreachable", z)
            z_next = float(p[2])
            self._send(sol.q, (sol.q - prev_q) / dt)

            f = self.vertical_force()
            if f is not None and f > cfg.FORCE_CONTACT_THRESHOLD_N:
                _halt(sol.q); self.freeze_reference()
                if stop_z is not None:
                    logger.warning("[suction] straight place: contact {:.1f}N at ee_z={:.4f}, "
                                   "{:+.1f}mm above the release height — part NOT released",
                                   f, z_next, (z_next - target_ee_z) * 1000.0)
                    return PickResult(False, "early_contact", z_next,
                                      contact_info=self._contact_snapshot(sol.q))
                logger.info("[suction] straight place: contact {:.1f}N at ee_z={:.4f} "
                            "({:+.1f}mm vs expected)", f, z_next,
                            (z_next - target_ee_z) * 1000.0)
                return PickResult(True, "contact", z_next,
                                  contact_info=self._contact_snapshot(sol.q))
            if stop_z is not None and z_cmd <= target_ee_z:   # commanded, not the IK's p
                _halt(sol.q)
                time.sleep(float(PLACE_RELEASE_SETTLE_S))
                live = float(self.current_ee_pose()[0][2])
                logger.info("[suction] straight place: at the release height ee_z={:.4f} "
                            "(live {:.4f}, {:+.1f}mm)", z_next, live,
                            (live - z_next) * 1000.0)
                return PickResult(True, "at_height", z_next)
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
                self._track_trace("place straight", [x, y, z], prev_q,
                                  tick_acc / tick_n * 1000.0,
                                  extra=" f={}".format("-" if f is None
                                                       else format(f, "+.1f")))
                trace_t, tick_acc, tick_n = 0.0, 0.0, 0

        _halt(prev_q)
        logger.warning("[suction] max descent ({:.2f}m) without contact", cfg.DESCENT_MAX_M)
        return PickResult(False, "max_descent", z)

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

