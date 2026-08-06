"""Right-arm Robotiq gripper for ik_demo (built on arm.ArmMover).

Used for the barcode divert: a matched battery held by the suction cup at
transport is gripped by the right-arm gripper (side approach), suction releases,
and the gripper carries it to a taught lower-right joint pose and opens.

Primitives only (grip_at, place_ee_seq); the two-arm handoff choreography lives
in sequence.py, which coordinates the suction arm + this gripper.

Gripper poses (HANDOFF_GRIP_OFFSET, PLACE_LOWER_RIGHT_EE_SEQ) are from the old
demo — RE-TEACH on the robot before trusting the divert.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
    from .arm import ArmMover
    from .drivers.robotiq_usb import RobotiqGripperUSB
except ImportError:  # allow `python gripper.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover
    from drivers.robotiq_usb import RobotiqGripperUSB


class GripperMover(ArmMover):
    """Right-arm mover that drives the Robotiq gripper."""

    def __init__(self, robot) -> None:
        super().__init__(robot=robot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
        self.gripper = RobotiqGripperUSB()

    def initialize(self) -> bool:
        """Reset + activate + open the gripper. Returns True on success."""
        if not self.gripper.initialize():
            logger.warning("[gripper] initialization failed — gripper disabled")
            return False
        self.gripper.open()
        logger.info("[gripper] initialized (reset + activated + opened)")
        return True

    def grip_at(self, pos, rpy=None) -> bool:
        """Side-approach grasp at base-frame *pos*; True if an object was gripped.

        Opens, backs off along the approach axis (gripper tool +z) by
        GRIPPER_PREGRASP_STANDOFF_M, then runs standoff -> grasp as ONE
        blended stream (move_joints_through: the standoff is crossed at blend
        speed instead of a full stop), and closes. The grasp solve chains from
        the standoff solution — the same branch the old standoff-then-live
        two-move chain landed on.
        """
        grasp_pos = np.asarray(pos, dtype=float)
        rpy = np.asarray(cfg.GRIPPER_GRASP_RPY if rpy is None else rpy, dtype=float)
        self.gripper.open()
        approach = Rotation.from_euler("xyz", rpy).as_matrix()[:, 2]  # tool +z in base_link
        pre = grasp_pos - cfg.GRIPPER_PREGRASP_STANDOFF_M * approach
        sol_pre = self.solve_pose(pre, rpy, seed=self._live_arm_q(), min_motion=True)
        if sol_pre.pos_err_m > cfg.REACH_TOL_M:
            logger.error("[gripper] grasp pose unreachable")
            return False
        sol_grasp = self.solve_pose(grasp_pos, rpy, seed=sol_pre.q, min_motion=True)
        if sol_grasp.pos_err_m > cfg.REACH_TOL_M:
            logger.error("[gripper] grasp pose unreachable")
            return False
        logger.info("[gripper] pre-grasp standoff -> grasp")
        self.move_joints_through([sol_pre.q, sol_grasp.q])
        self.gripper.close()
        gripped = self.gripper.is_object_grasped()
        logger.info("[gripper] grip_at -> {}", "GRIPPED" if gripped else "no object")
        return gripped

    def place_ee_seq(self, seq=None, on_step: dict | None = None) -> bool:
        """Walk the taught EE place sequence and release the battery lower-right.

        Each step is (label, is_relative, pos, rpy). REL steps add to the last
        COMMANDED pose — positions add, but rotations compose in SO(3) (Euler
        components don't add, and near gimbal lock the readout flips). The
        gripper partial-opens to release at the step whose label contains
        "lower", then fully opens at the end. Returns False if a step is
        unreachable (arm left where it stalled).

        These are fixed taught waypoints, not a live sensing leg. REL steps
        (small, genuinely relative to wherever the grip happened to land) are
        solved chained from the previous step's own solved joints, same as a
        live correction. ABS steps reset the seed to self._home_seed instead
        of chaining: some of these need a big joint-space reconfiguration to
        reach (see check_place_seq.py's joint_jump flag), and that
        reconfiguration is a genuine local-minimum trap for the differential
        solver from SOME branches — reproduced live (98mm short, reliably, on
        an already-verified-reachable point) and confirmed offline: the exact
        branch grip_at()'s pre-grasp standoff leaves "To Right 1" on doesn't
        converge for "To Right 2" even with an unconstrained (non-min-motion)
        posture target, while home_seed converges cleanly for every ABS step
        regardless. move_joints doesn't care what seed found the target, only
        that it's valid — so anchoring ABS solves on a seed proven to converge
        beats chaining from whatever branch the live sequence happened onto.

        ``on_step`` optionally maps a step label to a zero-arg callback invoked
        right before that step's move is issued — e.g. to kick off a
        concurrent move on the other arm at a specific point in this sequence.

        Execution: every step is SOLVED up front (an invalid step aborts
        BEFORE any motion — the old per-step loop walked the arm to the
        failing step first), then the whole sequence streams as one blended
        trajectory (move_joints_through). The release step and any step with
        an on_step callback stay full stops; the other waypoints are crossed
        at blend speed.
        """
        seq = cfg.PLACE_LOWER_RIGHT_EE_SEQ if seq is None else seq
        on_step = on_step or {}
        if not seq:
            raise ValueError("PLACE_LOWER_RIGHT_EE_SEQ not set — teach it first")
        cmd_pos, cmd_rpy = self.current_ee_pose()
        seed = self._live_arm_q()
        qs: list[np.ndarray] = []
        stops: set[int] = set()
        at_wp: dict[int, object] = {}
        pre_cbs: list = []   # on_step callbacks for step 0 (fire before the stream)

        def _add_cb(idx: int, fn) -> None:
            prev = at_wp.get(idx)
            at_wp[idx] = fn if prev is None else (lambda: (prev(), fn()))

        for i, (label, is_relative, pos, rpy) in enumerate(seq):
            if label in on_step:   # "before step i's move" = at waypoint i-1
                if i == 0:
                    pre_cbs.append(on_step[label])
                else:
                    _add_cb(i - 1, on_step[label])
            if is_relative:
                cmd_pos = cmd_pos + np.asarray(pos, dtype=float)
                cmd_rpy = (Rotation.from_euler("xyz", rpy)
                           * Rotation.from_euler("xyz", cmd_rpy)).as_euler("xyz")
            else:
                cmd_pos = np.asarray(pos, dtype=float)
                cmd_rpy = np.asarray(rpy, dtype=float)
                seed = self._home_seed  # ABS waypoint: anchor on a seed proven to converge
            sol = self.solve_pose(cmd_pos, cmd_rpy, seed=seed, min_motion=True)
            if not sol.valid:
                logger.error("[gripper] place step {} invalid (err={:.1f}mm converged={} "
                             "collision={} in_limits={}) — aborting before moving", label,
                             sol.pos_err_m * 1000, sol.converged, sol.in_collision, sol.in_limits)
                return False
            qs.append(sol.q)
            seed = sol.q
            if "lower" in label.lower():
                stops.add(i)

                def _release(lb=label):
                    logger.info("[gripper] at {} — partial open (release)", lb)
                    self.gripper.partial_open()
                _add_cb(i, _release)
        for cb in pre_cbs:
            cb()
        logger.info("[gripper] place sequence: {} steps as one blended stream", len(qs))
        self.move_joints_through(qs, stops=stops, at_waypoint=at_wp)
        self.gripper.open()
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
