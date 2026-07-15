"""Headless validator for cfg.PLACE_LOWER_RIGHT_EE_SEQ — no arm motion.

Solves the taught right-arm handoff/place sequence to joints and checks each
step converges, is collision-free, and in-limits — matching exactly how
gripper.place_ee_seq() solves it live: REL steps chain-seeded from the
previous step, ABS steps re-anchored on self._home_seed (some ABS waypoints
need a big joint-space reconfiguration to reach, and chaining from whatever
branch the live sequence happened onto is a real local-minimum trap for the
differential solver — home_seed converges reliably where a chained seed can
silently fail; see gripper.place_ee_seq's docstring). The joint_jump figure
is logged for visibility only, NOT a pass/fail criterion — move_joints
handles an arbitrarily large jump fine, jerk-limited, regardless of size.
This is the "validate before streaming" check PLAN.md calls for, applied to
this one hand-taught sequence rather than the cached fixed poses (see
arm.cache_taught_poses).

The starting grip is reconstructed by replicating grip_at()'s OWN chain
(home -> pre-grasp standoff -> grasp, each live-seeded) rather than solving
the grasp pose directly from home_seed — the standoff leg lands the solver
on a different (still valid) branch than a direct solve would, and that's
the exact branch the live sequence has to converge from. Skipping it here
previously produced a false pass (this checker said "OK", the robot then
failed at "To Right 2" by 98mm, twice, from the real grip chain).

Run headless (uses cfg.TORSO_JOINTS):    python -m ik_demo.check_place_seq
Run at the live torso (no motion):       python -m ik_demo.check_place_seq --robot
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

try:
    from . import config as cfg
    from .arm import ArmMover
except ImportError:  # allow `python check_place_seq.py` from inside ik_demo/
    import config as cfg
    from arm import ArmMover

# Flag a step whose solved joints jump more than this (rad, L2 over the 7
# joints) from the previous step's solution — a diagnostic threshold, not a
# hard limit.
MAX_STEP_JOINT_JUMP_RAD = 1.0


def _grip_start(mover: ArmMover, slot_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the gripper's commanded pose AND joints right after
    grip_at() — same geometry _divert() computes live: hover at the FIXED
    cfg.HANDOFF_HOVER_XY (same regardless of which battery slot triggered the
    divert) at transport height, + HANDOFF_GRIP_OFFSET, at the gripper's
    grasp orientation. Replicates grip_at()'s OWN chain (home -> pre-grasp
    standoff -> grasp, each live-seeded) to get the real starting joints —
    solving grasp_pos directly from home_seed (skipping the standoff leg)
    lands on a different branch and gave a false pass here before.
    ``slot_name`` no longer affects the result — kept as a param so the two
    call sites in _check() still read as "check both divert paths"."""
    del slot_name
    hover = np.array([cfg.HANDOFF_HOVER_XY[0], cfg.HANDOFF_HOVER_XY[1], cfg.SAFE_TRANSPORT_Z])
    grasp_pos = hover + np.asarray(cfg.HANDOFF_GRIP_OFFSET, dtype=float)
    grasp_rpy = np.asarray(cfg.GRIPPER_GRASP_RPY, dtype=float)
    approach = Rotation.from_euler("xyz", grasp_rpy).as_matrix()[:, 2]
    pre = grasp_pos - cfg.GRIPPER_PREGRASP_STANDOFF_M * approach
    q = mover.solve_pose(pre, grasp_rpy, seed=mover._home_seed, min_motion=True).q
    q = mover.solve_pose(grasp_pos, grasp_rpy, seed=q, min_motion=True).q
    return grasp_pos, grasp_rpy, q


def check_sequence(mover: ArmMover, start_pos: np.ndarray, start_rpy: np.ndarray,
                    start_q: np.ndarray, seq=None) -> bool:
    """Walk ``seq`` (default cfg.PLACE_LOWER_RIGHT_EE_SEQ) from (start_pos,
    start_rpy, start_q), mirroring gripper.place_ee_seq's REL/ABS bookkeeping
    AND its seed choice (REL chains from the previous step; ABS re-anchors on
    mover._home_seed) — solve+report instead of moving. Returns True iff
    every step is valid (converged, collision-free, in-limits); joint_jump is
    logged for visibility only."""
    seq = cfg.PLACE_LOWER_RIGHT_EE_SEQ if seq is None else seq
    if not seq:
        logger.error("PLACE_LOWER_RIGHT_EE_SEQ not set — nothing to check")
        return False
    cmd_pos, cmd_rpy = start_pos.copy(), start_rpy.copy()
    seed = start_q.copy()
    prev_q = start_q.copy()
    all_ok = True
    for label, is_relative, pos, rpy in seq:
        if is_relative:
            cmd_pos = cmd_pos + np.asarray(pos, dtype=float)
            cmd_rpy = (Rotation.from_euler("xyz", rpy)
                       * Rotation.from_euler("xyz", cmd_rpy)).as_euler("xyz")
        else:
            cmd_pos = np.asarray(pos, dtype=float)
            cmd_rpy = np.asarray(rpy, dtype=float)
            seed = mover._home_seed  # mirror place_ee_seq's ABS-step seed reset
        sol = mover.solve_pose(cmd_pos, cmd_rpy, seed=seed, min_motion=True)
        jump = float(np.linalg.norm(sol.q - prev_q))
        ok = sol.valid
        all_ok = all_ok and ok
        logger.info(
            "{} {:10s} err={:6.2f}mm converged={:<5} collision={:<5} in_limits={:<5} joint_jump={:.3f}rad{}",
            "OK  " if ok else "BAD ", label, sol.pos_err_m * 1000,
            str(sol.converged), str(sol.in_collision), str(sol.in_limits), jump,
            "  <-- big reconfig (informational only)" if jump > MAX_STEP_JOINT_JUMP_RAD else "",
        )
        seed, prev_q = sol.q, sol.q
    return all_ok


def _check(mover: ArmMover) -> bool:
    overall = True
    for slot in ("BAT_SLOT_1", "BAT_SLOT_2"):
        logger.info("=== grip start: {} ===", slot)
        start_pos, start_rpy, start_q = _grip_start(mover, slot)
        overall = check_sequence(mover, start_pos, start_rpy, start_q) and overall
    logger.info("PLACE_LOWER_RIGHT_EE_SEQ: {}", "ALL OK" if overall else "SOME STEPS BAD")
    return overall


def _selftest() -> None:
    """Headless (torso = cfg.TORSO_JOINTS): python -m ik_demo.check_place_seq"""
    logger.info("=== check_place_seq headless self-test (no robot, torso={}) ===",
                np.round(cfg.TORSO_JOINTS, 3))
    mover = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
    _check(mover)


def _verify_on_robot() -> None:
    """At the live torso, NO motion: python -m ik_demo.check_place_seq --robot"""
    from dexcontrol.robot import Robot

    with Robot() as bot:
        mover = ArmMover(robot=bot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
        live_torso = np.asarray(bot.torso.get_joint_pos(), dtype=float)
        logger.info("live torso (rad): {}  | cfg.TORSO_JOINTS: {}",
                    np.round(live_torso, 3), np.round(cfg.TORSO_JOINTS, 3))
        if np.max(np.abs(live_torso - np.asarray(cfg.TORSO_JOINTS))) > 0.05:
            logger.warning("live torso differs from cfg.TORSO_JOINTS — results here "
                           "won't match what actually happens on the robot.")
        _check(mover)


if __name__ == "__main__":
    import sys
    if "--robot" in sys.argv:
        _verify_on_robot()
    else:
        _selftest()
