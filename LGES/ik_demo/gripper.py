"""Right-arm Robotiq gripper for ik_demo (built on arm.ArmMover).

GripperMover = ArmMover on the right arm + the Robotiq USB gripper driver.
Motion primitives are ArmMover's (move_joints / move_ee / move_ee_vertical /
solve_step); this module only adds the gripper hardware. The old two-arm divert
handoff choreography (side grip at the suction EE, taught lower-right place
sequence) was removed 2026-09-03 — the right arm's next job is a vertical pick
of a paper box, mirroring the left arm's descent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
    from .arm import ArmMover
    from .drivers.robotiq import RobotiqGripper
    from .drivers.robotiq_usb import RobotiqGripperUSB
except ImportError:  # allow `python gripper.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover
    from drivers.robotiq import RobotiqGripper
    from drivers.robotiq_usb import RobotiqGripperUSB


@dataclass(frozen=True)
class BoxPose:
    """A detected box in base_link: top-face center (x, y, top_z) in metres and
    the yaw (rad, CCW+) of its LONG axis. Stand-in for the box detector's
    output until that model lands — box_pick.py feeds a hand-typed one."""
    x: float
    y: float
    top_z: float
    yaw: float


@dataclass
class BoxPickResult:
    success: bool
    reason: str                    # grasped | no_object | contact | not_detected
                                   # | unreachable | descent_failed | lift_failed | dry_run
                                   # | gripper_fault (the close was refused: the
                                   # gripper dropped off the RS485 bus)
    yaw_used: "float | None" = None  # EE yaw flown (rad) — the finger-closing direction
    box: "BoxPose | None" = None     # the box the attempt was planned for (run_box_pick fills it)


class GripperMover(ArmMover):
    """Right-arm mover that drives the Robotiq gripper."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
        # Set by initialize(): the Robotiq driver over whichever transport
        # answered (cfg.ROBOTIQ_TRANSPORT). Both drivers share one interface.
        self.gripper = None
        # Right wrist wrench (contact guard on the box descent); None = no guard.
        self._wrench = getattr(self._arm, "wrench_sensor", None) if robot is not None else None
        self._force_baseline: np.ndarray | None = None

    def tare_wrench(self) -> bool:
        """Average BOX_TARE_SAMPLES wrench readings with the arm at rest and keep
        the force part as the zero. False (guard off) without a sensor / data."""
        if self._wrench is None:
            return False
        raw = []
        for _ in range(int(cfg.BOX_TARE_SAMPLES)):
            w = self._wrench.get_wrench_state()
            if w is not None and np.all(np.isfinite(w)):
                raw.append(np.asarray(w, dtype=float).ravel()[:3])
            time.sleep(0.005)
        if len(raw) < max(3, int(cfg.BOX_TARE_SAMPLES) // 2):
            logger.warning("[gripper] wrench tare got {} readings — contact guard OFF", len(raw))
            self._force_baseline = None
            return False
        self._force_baseline = np.mean(raw, axis=0)
        logger.info("[gripper] wrench tared ({} samples, |f| {:.1f}N)", len(raw),
                    float(np.linalg.norm(self._force_baseline)))
        return True

    def vertical_force(self) -> "float | None":
        """|tared vertical (base z) force| in N, or None without a tare."""
        if self._wrench is None or self._force_baseline is None:
            return None
        w = self._wrench.get_wrench_state()
        if w is None:
            return None
        return float(abs(np.asarray(w, dtype=float).ravel()[2] - self._force_baseline[2]))

    def axis_force(self, axis) -> "float | None":
        """Tared force along a base-frame unit ``axis``, in N (signed: positive
        means pushed ALONG +axis), or None without a tare / sensor.

        vertical_force() answers the box descent's question (base z). A SIDE
        approach needs the force along the direction it is travelling — the
        wrench is measured in the wrist frame, so it is rotated to base and then
        projected."""
        if self._wrench is None or self._force_baseline is None:
            return None
        w = self._wrench.get_wrench_state()
        if w is None:
            return None
        f = np.asarray(w, dtype=float).ravel()[:3] - self._force_baseline
        a = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            return None
        return float(np.dot(self.current_ee_rotation() @ f, a / n))

    def initialize(self) -> bool:
        """Reset + activate + open the gripper over the first transport that
        answers: the right arm's EE pass-through ("ee") and/or the USB-RS485
        adapter ("usb") per cfg.ROBOTIQ_TRANSPORT. Returns True on success."""
        mode = str(cfg.ROBOTIQ_TRANSPORT).lower()
        order = {"auto": ("ee", "usb"), "ee": ("ee",), "usb": ("usb",)}.get(mode)
        if order is None:
            raise ValueError(f"ROBOTIQ_TRANSPORT must be auto/ee/usb, got {mode!r}")
        other = "left" if self._side == "right" else "right"
        ee_sides = ((cfg.ROBOTIQ_EE_SIDE,) if cfg.ROBOTIQ_EE_SIDE else (self._side, other))
        candidates = []
        for name in order:
            if name == "ee":
                candidates += [(f"the {s} arm's EE connector", RobotiqGripper(self._robot, side=s))
                               for s in ee_sides]
            else:
                candidates.append(("the USB-RS485 adapter", RobotiqGripperUSB()))
        for label, g in candidates:
            if g.initialize():
                self.gripper = g
                self.gripper.open()
                logger.info("[gripper] initialized over {} (reset + activated + opened)", label)
                if (isinstance(g, RobotiqGripper) and g._side != self._side
                        and not cfg.ROBOTIQ_EE_SIDE):
                    # auto-discovered on the OTHER arm's bus: flag it (a pinned
                    # ROBOTIQ_EE_SIDE means the wiring is known and intended)
                    logger.warning("[gripper] the gripper bus is on the {} arm's connector while "
                                   "THIS mover drives the {} arm — make sure the gripper is "
                                   "physically mounted on the {} arm", g._side, self._side,
                                   self._side)
                return True
            logger.warning("[gripper] {}: no gripper answered", label)
        logger.warning("[gripper] initialization failed ({}) — gripper disabled",
                       ", ".join(l for l, _ in candidates))
        return False

    def pick_box(self, box: BoxPose, dry: bool = False, mode: str = "carry") -> BoxPickResult:
        """Straight-down box pick — the right-arm mirror of the suction pick.

        hover (BOX_HOVER_HEIGHT_M above the grasp) -> vertical descent to the
        grasp height (fingertips BOX_GRASP_DEPTH_M below the box top), fast
        down to BOX_DESCENT_CREEP_FROM_M above it and at CREEP from there ->
        close -> vertical lift back to the hover. The EE yaw is the finger-closing
        direction (BOX_GRASP_YAW_OFFSET_RAD), normalised into [0, pi) — ONE
        wrist angle, no 180 deg alternative (see ``plan_box``). It is flown
        only if its hover AND whole descent column solve (reach, joint band,
        self-collision); otherwise this returns "unreachable" and the caller
        moves the CHASSIS (box_reach_offset). The vertical legs are
        move_ee_vertical streams, so the joint-speed guard and pacing apply as
        on the left arm.

        ``dry``: plan and log only, no motion (works headless). ``mode`` says
        what happens after the fingers close on the wall:
          "carry"     keep holding and lift to the hover (the caller carries)
          "lift_test" raise the box BOX_LIFT_TEST_M, hold BOX_GRIP_HOLD_S, set
                      it back down at the grasp height (wrench re-tared with the
                      box's weight, so an early touchdown stops the lowering),
                      OPEN, then retreat empty to the hover
          "release"   hold BOX_GRIP_HOLD_S, OPEN, retreat empty (landing check)
        An empty grasp always re-opens and retreats. Ends at the hover; the
        caller homes from there. The descent has a wrench contact guard
        (BOX_CONTACT_FORCE_N) tared at the hover, live on both legs.
        """
        if mode not in ("carry", "lift_test", "release"):
            raise ValueError(f"pick_box mode must be carry/lift_test/release, got {mode!r}")
        if not dry and self.gripper is None:
            logger.error("[gripper] pick_box: gripper not initialized — call initialize() first")
            return BoxPickResult(False, "no_gripper")
        rpy, ee_grasp, ee_hover = self._grasp_geometry(box)
        x, y, z_grasp, z_hover = float(ee_grasp[0]), float(ee_grasp[1]), \
            float(ee_grasp[2]), float(ee_hover[2])
        plan = self.plan_box(box)
        if plan is None:
            logger.error("[gripper] box at ({:.3f},{:+.3f}) top_z={:.3f}: the tilted wall grasp "
                         "does not solve there — not moving", box.x, box.y, box.top_z)
            return BoxPickResult(False, "unreachable")
        rpy, q_hover = plan
        logger.info("[gripper] pick_box plan: fingertip ({:.3f},{:+.3f},{:.3f}) on the wall "
                    "({:.0f}mm under the rim {:.3f}) -> EE ({:.3f},{:+.3f},{:.3f}), hover z={:.3f}, "
                    "tilt {:.0f}deg, yaw {:+.2f} rad", box.x, box.y,
                    box.top_z - cfg.BOX_GRASP_DEPTH_M, cfg.BOX_GRASP_DEPTH_M * 1000.0,
                    box.top_z, x, y, z_grasp, z_hover, cfg.BOX_GRASP_TILT_DEG, rpy[2])
        if dry:
            return BoxPickResult(True, "dry_run", rpy[2])

        self.gripper.open()
        self._approach_hover(box, rpy, q_hover)
        time.sleep(0.3)                     # let the arm settle before zeroing the wrench
        # Last point where the fingers are provably CLEAR (BOX_HOVER_HEIGHT_M -
        # BOX_GRASP_DEPTH_M = 12cm of fingertip clearance over the box top) and
        # the arm is at rest — so the last point where a gripper that fell out
        # of its activated state can be re-activated, since activation sweeps
        # the fingers. 0911: the open above was accepted and 12s later, after
        # the approach stream, the close found gFLT=5 / gSTA=0 — the gripper
        # drops out DURING arm motion, not during idle, and the recovery that
        # used to sit inside goto() fired at the grasp pose where the fingers
        # straddle the box wall and could not complete its calibration cycle.
        if not self.gripper.ensure_activated():
            logger.error("[gripper] gripper not ready at the hover — NOT descending")
            return BoxPickResult(False, "gripper_fault", rpy[2])
        guard = self.tare_wrench()          # at rest, gripper open: zero for the descent
        contact: list[float] = []            # [z, force] when the guard fires

        def _stop() -> bool:
            f = self.vertical_force()
            if f is not None and f > float(cfg.BOX_CONTACT_FORCE_N):
                contact[:] = [float(self.fk(self._q_cmd)[0][2]), f]
                return True
            return False

        # Two legs (0914, the user's call): fast while nothing can be touched,
        # then CREEP the last cfg.BOX_DESCENT_CREEP_FROM_M with the guard live,
        # so a contact is met gently and located to ~1mm instead of ~4mm. The
        # creep leg passes creep_out_m = its own length, which is how
        # move_ee_vertical is told to hold creep for the WHOLE leg (its normal
        # shape only creeps the last DESCENT_CREEP_BLEND_M into the target).
        z_creep = min(z_hover, z_grasp + float(cfg.BOX_DESCENT_CREEP_FROM_M))
        legs = []
        if z_hover - z_creep > 1e-4:
            legs.append((z_creep, 0.0))                  # fast, ordinary profile
        legs.append((z_grasp, z_creep - z_grasp))        # creep the whole way
        logger.info("[gripper] descend {:.0f}mm straight down: {:.0f}mm fast, then {:.0f}mm at "
                    "creep{}", (z_hover - z_grasp) * 1000.0, (z_hover - z_creep) * 1000.0,
                    (z_creep - z_grasp) * 1000.0,
                    f" (contact guard {cfg.BOX_CONTACT_FORCE_N:.0f}N)" if guard else " (NO force guard)")
        for z_to, creep_all in legs:
            q = self.move_ee_vertical(z_to, rpy, stop_fn=_stop if guard else None,
                                      creep_out_m=creep_all)
            if q is None:
                logger.error("[gripper] descent stalled — halted above the box, NOT gripping")
                return BoxPickResult(False, "descent_failed", rpy[2])
            if contact:
                logger.warning("[gripper] CONTACT {:.1f}N at EE z={:.4f} ({:+.0f}mm above the "
                               "grasp height) — not closing, lifting back to the hover",
                               contact[1], contact[0], (contact[0] - z_grasp) * 1000.0)
                self.move_ee_vertical(z_hover, rpy)
                return BoxPickResult(False, "contact", rpy[2])
        closed = self.gripper.close()           # blocks until the Robotiq reports done
        grasped = self.gripper.is_object_grasped()
        logger.info("[gripper] close -> {}", "GRASPED" if grasped else "no object")
        if not closed and not grasped:
            # The gripper refused the command and the driver's one reset +
            # activate retry did not bring it back (0910 box discard, see
            # RobotiqGripper.goto): the fingers never moved, so this is NOT a
            # missed grasp and neither a re-detect nor a chassis move can fix
            # it. Skip the open below (it would be refused the same way, and
            # the fingers are still where they were) and lift away empty.
            logger.error("[gripper] the CLOSE was refused — the gripper is off the bus, "
                         "this is not a missed grasp; lifting empty")
            self.move_ee_vertical(z_hover, rpy)
            return BoxPickResult(False, "gripper_fault", rpy[2])
        if grasped and mode == "lift_test":
            z_up = z_grasp + float(cfg.BOX_LIFT_TEST_M)
            logger.info("[gripper] lift test: +{:.0f}mm with the box, hold {:.1f}s, set back down",
                        float(cfg.BOX_LIFT_TEST_M) * 1000.0, float(cfg.BOX_GRIP_HOLD_S))
            if self.move_ee_vertical(z_up, rpy) is None:
                logger.error("[gripper] lift test stalled — setting the box down from here")
            time.sleep(float(cfg.BOX_GRIP_HOLD_S))
            # zero the wrench WITH the box hanging: the set-down shows up as a
            # force step, so a box that slipped low in the grip stops early
            touchdown_guard = self.tare_wrench()
            contact[:] = []
            if self.move_ee_vertical(z_grasp, rpy, stop_fn=_stop if touchdown_guard else None) is None:
                logger.error("[gripper] set-down stalled — releasing here")
            if contact:
                logger.info("[gripper] touchdown {:.1f}N at EE z={:.4f} ({:+.0f}mm vs the grasp "
                            "height)", contact[1], contact[0], (contact[0] - z_grasp) * 1000.0)
            self.gripper.open()
        elif grasped and mode == "release":
            time.sleep(float(cfg.BOX_GRIP_HOLD_S))
            logger.info("[gripper] release before the lift (landing check) — lifting empty")
            self.gripper.open()
        elif not grasped:
            self.gripper.open()
        logger.info("[gripper] lift {:.0f}mm straight up{}", (z_hover - z_grasp) * 1000.0,
                    " with the box" if grasped and mode == "carry" else "")
        if self.move_ee_vertical(z_hover, rpy) is None:
            logger.error("[gripper] lift stalled below the hover")
            return BoxPickResult(grasped, "lift_failed", rpy[2])
        return BoxPickResult(grasped, "grasped" if grasped else "no_object", rpy[2])

    def _wall_outward(self, box_yaw: float) -> np.ndarray:
        """Unit base-frame xy pointing OUT of the box across the grabbed wall.

        The grasp point is the midpoint of the long wall on the robot's right,
        so the box's short axis THROUGH that wall is the outward normal — same
        construction (and same right-pointing flip) box_pick's
        box_pose_from_detection used to place the grasp point."""
        u = np.array([-np.sin(float(box_yaw)), np.cos(float(box_yaw))])
        return -u if u[1] > 0.0 else u

    def _approach_hover(self, box: BoxPose, rpy, q_hover: np.ndarray) -> None:
        """Fly home -> hover through up to two waypoints so the fingertips stay
        out of the box (see cfg.BOX_APPROACH_* for the measurements): the
        mid-path pose lifted BOX_APPROACH_LIFT_M, then the grasp point pushed
        BOX_APPROACH_SIDE_M outside the grabbed wall at hover height.

        Every waypoint is optional: one that does not solve (or is out of band /
        in self-collision) is dropped with a warning and the remaining ones
        still fly, so the worst case is the old single move straight to the
        hover. Legs run as separate move_joints — full stops at the waypoints,
        which is exactly the straight joint interpolation the clearances were
        measured on (blending would round the corners back into the box)."""
        q_start = np.asarray(self._start_q(), dtype=float)
        q_hover = np.asarray(q_hover, dtype=float)
        wps: list[tuple[str, np.ndarray]] = []

        lift = float(cfg.BOX_APPROACH_LIFT_M)
        if lift > 0.0:
            frac = float(cfg.BOX_APPROACH_MID_FRAC)
            q_mid = q_start + frac * (q_hover - q_start)
            p_mid, rpy_mid = self.fk(q_mid)
            sol = self.solve_pose([float(p_mid[0]), float(p_mid[1]), float(p_mid[2]) + lift],
                                  tuple(np.asarray(rpy_mid, dtype=float)),
                                  seed=q_mid, min_motion=True)
            if sol.converged and sol.in_limits and not sol.in_collision:
                wps.append((f"lifted mid (+{lift * 100:.0f}cm)", sol.q))
            else:
                logger.warning("[gripper] approach: lifted mid waypoint does not solve "
                               "({:.1f}mm, converged={}, in_limits={}, collision={}) — skipping it",
                               sol.pos_err_m * 1000.0, sol.converged, sol.in_limits, sol.in_collision)

        side = float(cfg.BOX_APPROACH_SIDE_M)
        if side > 0.0:
            out = self._wall_outward(box.yaw)
            p_h, _ = self.fk(q_hover)
            seed = wps[-1][1] if wps else q_start
            sol = self.solve_pose([float(p_h[0]) + float(out[0]) * side,
                                   float(p_h[1]) + float(out[1]) * side,
                                   float(p_h[2])], rpy, seed=seed, min_motion=True)
            if sol.converged and sol.in_limits and not sol.in_collision:
                wps.append((f"outside the wall (+{side * 100:.0f}cm)", sol.q))
            else:
                logger.warning("[gripper] approach: outside-the-wall waypoint does not solve "
                               "({:.1f}mm, converged={}, in_limits={}, collision={}) — skipping it",
                               sol.pos_err_m * 1000.0, sol.converged, sol.in_limits, sol.in_collision)

        if not wps:
            logger.info("[gripper] -> hover (joint-space, no approach waypoint)")
            self.move_joints(q_hover)
            return
        # ONE blended stream through the waypoints — they are crossed at speed
        # instead of stopped at, so the approach is a single continuous motion
        # rather than the go-stop-go-stop of chained move_joints.
        #
        # The rounding the blend introduces was measured (0911, replaying the
        # exact junction velocities move_joints_through builds): the blended
        # path clears the box by 93mm against the full-stop path's 87mm, i.e.
        # it cuts the corners AWAY from the box, and it is 0.4s quicker. If a
        # blended segment turns out infeasible or self-colliding,
        # move_joints_through drops that junction to zero velocity on its own,
        # which is exactly the old full-stop behaviour.
        logger.info("[gripper] -> hover through {} waypoint(s), one blended stream: {}",
                    len(wps), ", ".join(label for label, _ in wps))
        self.move_joints_through([q for _, q in wps] + [np.asarray(q_hover, dtype=float)])

    def _grasp_geometry(self, box: BoxPose) -> "tuple[tuple, np.ndarray, np.ndarray]":
        """(rpy, ee_grasp, ee_hover) for the TILTED wall pinch of ``box``.

        The fingertip — not the EE — is what has to land on the wall, and with
        the tool tipped cfg.BOX_GRASP_TILT_DEG off vertical it hangs
        BOX_FINGER_LENGTH_M along that tilted axis, i.e. over 10cm to the SIDE
        of the EE. So the target is built fingertip-first and the EE placed
        back along the tool axis from it.

        Orientation: R0(yaw) @ Rx(tilt) — the straight-down grasp turned about
        its OWN x, which is the finger-closing axis running across the wall, so
        the fingers keep straddling the wall squarely while the hand leans
        along it. See cfg.BOX_GRASP_TILT_DEG for where the 55 deg came from.

        ONE wrist angle, selected by cfg.BOX_GRASP_YAW_OFFSET_RAD. The
        finger-closing direction is a LINE, so box.yaw and box.yaw + pi are the
        same grasp with the wrist turned over, and the detector's own [0,180)
        yaw wrap means the SAME physical box can be reported on either side of
        it (read at 179.4 deg on one run and 1.2 deg on the next). Adding the
        offset alone would therefore hand back wrists 180 deg apart for an
        unchanged box, so take the representative within +-90 deg OF THE
        OFFSET: the offset is then the wrist selector, and flipping it by pi
        turns the wrist over for good.

        The hover sits straight ABOVE the grasp EE, because the descent that
        follows is a vertical move_ee_vertical stream holding this orientation
        (the fingertip therefore also travels straight down)."""
        off = float(cfg.BOX_GRASP_YAW_OFFSET_RAD)
        yaw = off + float((float(box.yaw) + np.pi / 2.0) % np.pi) - np.pi / 2.0
        R = (Rotation.from_euler("xyz", [np.pi, 0.0, yaw]).as_matrix()
             @ Rotation.from_euler("x", np.deg2rad(float(cfg.BOX_GRASP_TILT_DEG))).as_matrix())
        rpy = tuple(float(v) for v in Rotation.from_matrix(R).as_euler("xyz"))
        tip = np.array([float(box.x), float(box.y),
                        float(box.top_z) - float(cfg.BOX_GRASP_DEPTH_M)])
        ee_grasp = tip - R @ np.array([0.0, 0.0, float(cfg.BOX_FINGER_LENGTH_M)])
        ee_hover = ee_grasp + np.array([0.0, 0.0, float(cfg.BOX_HOVER_HEIGHT_M)])
        return rpy, ee_grasp, ee_hover

    def plan_box(self, box: BoxPose, seed: "np.ndarray | None" = None,
                 quiet: bool = False) -> "tuple[tuple, np.ndarray] | None":
        """(rpy, q_hover) for the straight-down grasp of ``box`` at its ONE
        finger-closing yaw (see the comment below), if that hover CONVERGES
        (1 mm) and its whole descent column solves (reach, joint band,
        self-collision) — else None, which the caller answers with a chassis
        move, not a wrist flip. Split out of pick_box so box_reach_offset can
        ask the same question at dozens of candidate chassis offsets;
        ``quiet`` drops the log line for that."""
        rpy, ee_grasp, ee_hover = self._grasp_geometry(box)
        x, y, z_grasp, z_hover = float(ee_grasp[0]), float(ee_grasp[1]), \
            float(ee_grasp[2]), float(ee_hover[2])
        if seed is None:
            # the taught branch, NOT the live config: see BOX_GRASP_SEED_JOINTS
            seed = np.asarray(cfg.BOX_GRASP_SEED_JOINTS, dtype=float)
        # An out-of-reach grasp is the CHASSIS's problem — no wrist flip
        # fallback: the caller's ladder asks box_reach_offset for the smallest
        # move that solves.
        sol = self.solve_pose([x, y, z_hover], rpy, seed=seed, min_motion=True)
        # CONVERGED (1 mm), not merely within REACH_TOL_M: a hover that
        # solves a few mm short near the reach edge starts the vertical
        # stream with a residual the first ticks must burn off (0903
        # harness: 7.7 mm hover -> 4x joint-speed cap on tick one)
        hover_ok = sol.converged and sol.in_limits and not sol.in_collision
        col_ok = hover_ok and self.column_reachable(x, y, rpy, z_hover, z_grasp, seed=sol.q,
                                                    quiet=quiet)
        if not quiet:
            logger.info("[gripper] pick_box yaw {:+.2f} tilt {:.0f}deg: hover {} (err {:.1f}mm, "
                        "converged={}, in_limits={}, collision={}), column {}",
                        rpy[2], cfg.BOX_GRASP_TILT_DEG, "OK" if hover_ok else "NO",
                        sol.pos_err_m * 1000.0, sol.converged, sol.in_limits, sol.in_collision,
                        "OK" if col_ok else "NO")
        if hover_ok and col_ok:
            return rpy, sol.q
        return None

    def box_reach_offset(self, box: BoxPose) -> "tuple[float, float] | None":
        """Smallest chassis move (dx forward, dy left, m) after which plan_box
        solves for ``box``, or None if nothing within
        cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M does. The chassis moving (dx, dy)
        puts the box at (x - dx, y - dy) in base_link. Offsets on the
        CHASSIS_ADJUST_STEP_M grid, L1 nearest-first; a spot that still plans
        CHASSIS_ADJUST_REACH_MARGIN_M off in +-x / +-y is preferred (so the
        open-loop move does not land on the edge of reach), else the nearest
        bare pass. The right-arm twin of chassis_sequence._auto_adjust."""
        t0 = time.monotonic()
        seed = self._start_q() if self._robot is not None else self._home_seed

        def plans(dx: float, dy: float) -> bool:
            b = BoxPose(float(box.x) - dx, float(box.y) - dy, float(box.top_z), float(box.yaw))
            return self.plan_box(b, seed=seed, quiet=True) is not None

        step = float(cfg.CHASSIS_ADJUST_STEP_M)
        lim = float(cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M)
        m = float(cfg.CHASSIS_ADJUST_REACH_MARGIN_M)
        n = int(round(lim / step))
        cands = sorted(((i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
                        if 0 < abs(i) + abs(j) <= n),
                       key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), abs(o[1])))
        bare = None
        for dx, dy in cands:
            if not plans(dx, dy):
                continue
            if bare is None:
                bare = (dx, dy)
            if all(plans(dx + ox, dy + oy) for ox, oy in ((m, 0.0), (-m, 0.0), (0.0, m), (0.0, -m))):
                logger.info("[gripper] box grasp solves after a chassis move of dx {:+.3f} m, "
                            "dy {:+.3f} m (with {:.0f}mm slack; {:.1f}s searched)",
                            dx, dy, m * 1000.0, time.monotonic() - t0)
                return dx, dy
        if bare is not None:
            logger.warning("[gripper] box grasp: no chassis offset within {:.2f} m solves with "
                           "{:.0f}mm of slack — nearest bare pass dx {:+.3f} m, dy {:+.3f} m "
                           "({:.1f}s searched)", lim, m * 1000.0, bare[0], bare[1],
                           time.monotonic() - t0)
            return bare
        logger.warning("[gripper] box grasp: NO chassis offset within {:.2f} m solves "
                       "({:.1f}s searched)", lim, time.monotonic() - t0)
        return None

    def set_down_box(self, drop_m: float) -> bool:
        """Set a HELD box down where the arm is: straight down ``drop_m`` from
        the current commanded pose (the hover pick_box("carry") ended at),
        OPEN, lift back up. The wrench is tared with the box hanging, so an
        early touchdown (floor higher than where it was picked) stops the
        lowering — the lift_test set-down, at a new spot. No detection, no
        gentle place: the box lands wherever the arm is. Ends at the start
        height; the caller homes from there."""
        if self.gripper is None:
            logger.error("[gripper] set_down_box: gripper not initialized")
            return False
        pos, rpy = self.fk(self._q_cmd)
        z_top = float(pos[2])
        z_down = z_top - float(drop_m)
        guard = self.tare_wrench()          # zero WITH the box's weight
        contact: list[float] = []

        def _stop() -> bool:
            f = self.vertical_force()
            if f is not None and f > float(cfg.BOX_CONTACT_FORCE_N):
                contact[:] = [float(self.fk(self._q_cmd)[0][2]), f]
                return True
            return False

        logger.info("[gripper] set down: {:.0f}mm straight down from EE z={:.3f}{}",
                    float(drop_m) * 1000.0, z_top,
                    f" (touchdown guard {cfg.BOX_CONTACT_FORCE_N:.0f}N)" if guard else " (NO force guard)")
        if self.move_ee_vertical(z_down, rpy, stop_fn=_stop if guard else None) is None:
            logger.error("[gripper] set-down stalled — releasing here")
        if contact:
            logger.info("[gripper] touchdown {:.1f}N at EE z={:.4f}", contact[1], contact[0])
        self.gripper.open()
        if self.move_ee_vertical(z_top, rpy) is None:
            logger.error("[gripper] lift after the release stalled")
            return False
        return True

    def probe_seat(self, box: BoxPose) -> bool:
        """DIAGNOSIS ONLY (0914): descend PAST the planned grasp height at creep
        and log the wrist force against height, so the PALM-ON-RIM seat can be
        read off a curve instead of guessed. Nothing closes, nothing is picked
        up, no config is changed by this.

        Why: the grasp height today is the DETECTED rim minus
        BOX_GRASP_DEPTH_M, and the rim is the measured floor plus the
        BOX_WALL_HEIGHT_M constant. 0914 that constant was 35mm too big (depth
        read the wall at ~105mm), the descent stopped 34mm high and the fingers
        took 31mm of wall instead of 65mm. But the tilted pinch has the palm
        sitting right over the grabbed wall, so the rim rides up into the open
        jaws and hits it — a hard stop at a height the GRIPPER sets, not the
        rim estimate. See cfg.BOX_PALM_ALONG_TOOL_M for the prediction this is
        here to confirm.

        Two legs, because the contact must happen SLOWLY: fast down to
        BOX_SEAT_PROBE_START_M above the planned grasp height (the ordinary
        BOX_CONTACT_FORCE_N guard still on — anything met up there is not the
        seat), then creep_out_m = the whole remaining leg, so the probe is at
        DESCENT_CREEP_SPEED_M_S when it meets the rim (one tick of overshoot
        ~0.8mm instead of ~3mm at cruise). Samples every tick into
        ZTRACK_LOG_DIR/seat_probe_*.csv and stops only at
        BOX_SEAT_PROBE_BACKSTOP_N. Ends back at the hover."""
        if self.gripper is None:
            logger.error("[gripper] probe_seat: gripper not initialized")
            return False
        plan = self.plan_box(box)
        if plan is None:
            logger.error("[gripper] probe_seat: the grasp does not solve at this box pose")
            return False
        rpy, q_hover = plan
        _, ee_grasp, ee_hover = self._grasp_geometry(box)
        z_grasp, z_hover = float(ee_grasp[2]), float(ee_hover[2])
        tilt = np.deg2rad(float(cfg.BOX_GRASP_TILT_DEG))
        drop = float(cfg.BOX_FINGER_LENGTH_M) * np.cos(tilt)   # EE z -> fingertip z
        seat_depth = (float(cfg.BOX_FINGER_LENGTH_M)
                      - float(cfg.BOX_PALM_ALONG_TOOL_M)) * np.cos(tilt)
        z_start = z_grasp + float(cfg.BOX_SEAT_PROBE_START_M)
        z_end = z_grasp - float(cfg.BOX_SEAT_PROBE_EXTRA_M)
        logger.warning("[gripper] SEAT PROBE (no grip): creep from EE z={:.4f} to {:.4f}, i.e. "
                       "fingertip {:.0f}mm to {:.0f}mm under the detected rim {:.3f}. Palm "
                       "predicted to seat at {:.1f}mm (BOX_GRASP_DEPTH_M is {:.0f}mm). "
                       "Backstop {:.0f}N.", z_start, z_end,
                       (float(box.top_z) - (z_start - drop)) * 1000.0,
                       (float(box.top_z) - (z_end - drop)) * 1000.0, float(box.top_z),
                       seat_depth * 1000.0, float(cfg.BOX_GRASP_DEPTH_M) * 1000.0,
                       float(cfg.BOX_SEAT_PROBE_BACKSTOP_N))

        self.gripper.open()
        self._approach_hover(box, rpy, q_hover)
        time.sleep(0.3)                     # settle before zeroing the wrench
        if not self.gripper.ensure_activated():
            logger.error("[gripper] gripper not ready at the hover — NOT descending")
            return False
        if not self.tare_wrench():
            logger.error("[gripper] no wrench tare — the probe has nothing to measure")
            return False

        early: list[float] = []

        def _stop_early() -> bool:
            f = self.vertical_force()
            if f is not None and f > float(cfg.BOX_CONTACT_FORCE_N):
                early[:] = [float(self.fk(self._q_cmd)[0][2]), f]
                return True
            return False

        if self.move_ee_vertical(z_start, rpy, stop_fn=_stop_early) is None:
            logger.error("[gripper] approach to the probe start stalled — lifting")
            self.move_ee_vertical(z_hover, rpy)
            return False
        if early:
            logger.error("[gripper] {:.1f}N at EE z={:.4f}, {:+.0f}mm ABOVE the probe start — "
                         "something is in the way up there (rim far higher than detected, or "
                         "the rack): probe aborted, NOT pressing on it", early[1], early[0],
                         (early[0] - z_start) * 1000.0)
            self.move_ee_vertical(z_hover, rpy)
            return False

        rows: list[tuple[float, float, float, float]] = []
        t0 = time.perf_counter()

        def _sample() -> bool:
            w = self._wrench.get_wrench_state()
            if w is None:
                return False
            f = np.asarray(w, dtype=float).ravel()[:3] - self._force_baseline
            z_ee = float(self.fk(self._q_cmd)[0][2])
            rows.append((time.perf_counter() - t0, z_ee, float(f[2]),
                         float(np.linalg.norm(f))))
            return abs(float(f[2])) > float(cfg.BOX_SEAT_PROBE_BACKSTOP_N)

        leg = abs(z_start - z_end)
        if self.move_ee_vertical(z_end, rpy, stop_fn=_sample, creep_out_m=leg) is None:
            logger.error("[gripper] probe descent stalled — lifting from here")
        self._write_seat_probe(rows, box, z_grasp, drop, seat_depth)
        logger.info("[gripper] probe done — lifting {:.0f}mm back to the hover, gripper open",
                    (z_hover - float(self.fk(self._q_cmd)[0][2])) * 1000.0)
        if self.move_ee_vertical(z_hover, rpy, creep_out_m=cfg.DESCENT_CREEP_GAP_M) is None:
            logger.error("[gripper] lift back to the hover stalled")
            return False
        return True

    def _write_seat_probe(self, rows, box: BoxPose, z_grasp: float, drop: float,
                          seat_depth: float) -> None:
        """probe_seat's samples -> one CSV next to the ztrack logs, plus the
        first crossing of each force level in the run log (which is what the
        threshold has to be picked from)."""
        if not rows:
            logger.error("[gripper] seat probe collected NO samples — is the wrench publishing?")
            return
        depth = [(float(box.top_z) - (z - drop)) * 1000.0 for _, z, _, _ in rows]
        fz = [abs(r[2]) for r in rows]
        logger.info("[gripper] seat probe: {} samples, fingertip {:.0f}..{:.0f}mm under the rim, "
                    "|fz| {:.2f}..{:.2f}N", len(rows), depth[0], depth[-1], min(fz), max(fz))
        for level in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
            hit = next((i for i, v in enumerate(fz) if v > level), None)
            if hit is None:
                logger.info("[gripper] seat probe: never reached {:.0f}N", level)
                continue
            logger.info("[gripper] seat probe: {:.0f}N first at fingertip depth {:.1f}mm "
                        "(EE z={:.4f}, {:+.1f}mm vs the planned grasp, {:+.1f}mm vs the "
                        "predicted palm seat {:.1f}mm)", level, depth[hit], rows[hit][1],
                        (rows[hit][1] - z_grasp) * 1000.0,
                        depth[hit] - seat_depth * 1000.0, seat_depth * 1000.0)
        if cfg.ZTRACK_LOG_DIR is None:
            return
        from pathlib import Path  # noqa: PLC0415
        out = Path(cfg.ZTRACK_LOG_DIR) / f"seat_probe_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as fh:
                fh.write("t_s,ee_z,tip_depth_mm,fz_n,fmag_n\n")
                for (t, z, f_z, f_mag), d in zip(rows, depth):
                    fh.write(f"{t:.4f},{z:.5f},{d:.2f},{f_z:.4f},{f_mag:.4f}\n")
            logger.info("[gripper] seat probe CSV -> {}", out)
        except OSError as e:
            logger.warning("[gripper] seat probe CSV not written ({})", e)



# ---------------------------------------------------------------------------
# On-robot smoke test: python gripper.py   (init + open/close, no arm motion)
# ---------------------------------------------------------------------------
def _test_on_robot() -> None:
    from dexcontrol.robot import Robot

    logger.warning("Gripper smoke test: reset/activate/open, then close, then open. No arm motion.")
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return
    with Robot() as bot:
        g = GripperMover(bot)
        if not g.initialize():
            return
        import time
        time.sleep(0.5)
        g.gripper.close(); time.sleep(1.0)
        logger.info("object grasped? {}", g.gripper.is_object_grasped())
        g.gripper.open()


if __name__ == "__main__":
    _test_on_robot()
