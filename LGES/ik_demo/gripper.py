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
        grasp height (fingertips BOX_GRASP_DEPTH_M below the box top) -> close
        -> vertical lift back to the hover. The EE yaw is the finger-closing
        direction (BOX_GRASP_YAW_OFFSET_RAD); yaw and yaw + pi are both tried
        and the first whose hover AND whole descent column solve (reach, joint
        band, self-collision) is flown. The vertical legs are move_ee_vertical
        streams, so the joint-speed guard and pacing apply as on the left arm.

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
        (BOX_CONTACT_FORCE_N) tared at the hover.
        """
        if mode not in ("carry", "lift_test", "release"):
            raise ValueError(f"pick_box mode must be carry/lift_test/release, got {mode!r}")
        if not dry and self.gripper is None:
            logger.error("[gripper] pick_box: gripper not initialized — call initialize() first")
            return BoxPickResult(False, "no_gripper")
        z_grasp = float(box.top_z) - float(cfg.BOX_GRASP_DEPTH_M) + float(cfg.BOX_FINGER_LENGTH_M)
        z_hover = z_grasp + float(cfg.BOX_HOVER_HEIGHT_M)
        x, y = float(box.x), float(box.y)
        plan = self.plan_box(box)
        if plan is None:
            logger.error("[gripper] box at ({:.3f},{:+.3f}) top_z={:.3f}: no straight-down grasp "
                         "solves for either yaw — not moving", x, y, box.top_z)
            return BoxPickResult(False, "unreachable")
        rpy, q_hover = plan
        logger.info("[gripper] pick_box plan: xy=({:.3f},{:+.3f}) hover z={:.3f} -> grasp z={:.3f} "
                    "(fingertip z={:.3f}), yaw {:+.2f} rad", x, y, z_hover, z_grasp,
                    z_grasp - float(cfg.BOX_FINGER_LENGTH_M), rpy[2])
        if dry:
            return BoxPickResult(True, "dry_run", rpy[2])

        self.gripper.open()
        logger.info("[gripper] -> hover (joint-space)")
        self.move_joints(q_hover)
        time.sleep(0.3)                     # let the arm settle before zeroing the wrench
        guard = self.tare_wrench()          # at rest, gripper open: zero for the descent
        contact: list[float] = []            # [z, force] when the guard fires

        def _stop() -> bool:
            f = self.vertical_force()
            if f is not None and f > float(cfg.BOX_CONTACT_FORCE_N):
                contact[:] = [float(self.fk(self._q_cmd)[0][2]), f]
                return True
            return False

        logger.info("[gripper] descend {:.0f}mm straight down{}", (z_hover - z_grasp) * 1000.0,
                    f" (contact guard {cfg.BOX_CONTACT_FORCE_N:.0f}N)" if guard else " (NO force guard)")
        q = self.move_ee_vertical(z_grasp, rpy, stop_fn=_stop if guard else None)
        if q is None:
            logger.error("[gripper] descent stalled — halted above the box, NOT gripping")
            return BoxPickResult(False, "descent_failed", rpy[2])
        if contact:
            logger.warning("[gripper] CONTACT {:.1f}N at EE z={:.4f} ({:+.0f}mm above the grasp "
                           "height) — not closing, lifting back to the hover",
                           contact[1], contact[0], (contact[0] - z_grasp) * 1000.0)
            self.move_ee_vertical(z_hover, rpy)
            return BoxPickResult(False, "contact", rpy[2])
        self.gripper.close()                    # blocks until the Robotiq reports done
        grasped = self.gripper.is_object_grasped()
        logger.info("[gripper] close -> {}", "GRASPED" if grasped else "no object")
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

    def plan_box(self, box: BoxPose, seed: "np.ndarray | None" = None,
                 quiet: bool = False) -> "tuple[tuple, np.ndarray] | None":
        """(rpy, q_hover) for the straight-down grasp of ``box`` — the first of
        the two finger-closing yaws (yaw, yaw + pi) whose hover CONVERGES (1 mm)
        and whose whole descent column solves (reach, joint band,
        self-collision) — or None. Split out of pick_box so box_reach_offset
        can ask the same question at dozens of candidate chassis offsets;
        ``quiet`` drops the per-yaw log line for that."""
        z_grasp = float(box.top_z) - float(cfg.BOX_GRASP_DEPTH_M) + float(cfg.BOX_FINGER_LENGTH_M)
        z_hover = z_grasp + float(cfg.BOX_HOVER_HEIGHT_M)
        x, y = float(box.x), float(box.y)
        if seed is None:
            seed = self._start_q() if self._robot is not None else self._home_seed
        for half_turn in (0, 1):
            yaw = float(box.yaw) + float(cfg.BOX_GRASP_YAW_OFFSET_RAD) + half_turn * np.pi
            yaw = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)      # wrap to [-pi, pi)
            rpy = (np.pi, 0.0, yaw)
            sol = self.solve_pose([x, y, z_hover], rpy, seed=seed, min_motion=True)
            # CONVERGED (1 mm), not merely within REACH_TOL_M: a hover that
            # solves a few mm short near the reach edge starts the vertical
            # stream with a residual the first ticks must burn off (0903
            # harness: 7.7 mm hover -> 4x joint-speed cap on tick one)
            hover_ok = sol.converged and sol.in_limits and not sol.in_collision
            col_ok = hover_ok and self.column_reachable(x, y, rpy, z_hover, z_grasp, seed=sol.q,
                                                        quiet=quiet)
            if not quiet:
                logger.info("[gripper] pick_box yaw {:+.2f}: hover {} (err {:.1f}mm, converged={}, "
                            "in_limits={}, collision={}), column {}", yaw, "OK" if hover_ok else "NO",
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
