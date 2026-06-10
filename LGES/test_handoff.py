"""Partial handoff test: suction pick -> lift -> horizontal move -> right gripper grip -> hold.

Runs only up to the grip and then pauses so you can inspect whether the grip
is solid before committing to a place move. Press Enter to open the gripper and
exit cleanly; Ctrl-C exits immediately.

    python -m test_handoff                      # default speed (0.4×)
    python -m test_handoff --speed 0.2          # extra slow (0.2× all arm moves + Robotiq)
    python -m test_handoff --src BAT_SRC_2      # second battery

--speed scales every arm motion duration (pick descent step, lift, horizontal travel,
gripper approach) AND the Robotiq close speed proportionally.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot

_LGES_DIR = os.path.dirname(os.path.abspath(__file__))
if _LGES_DIR not in sys.path:
    sys.path.insert(0, _LGES_DIR)
from utils import set_head_pitch  # noqa: E402

from case_battery_demo import config as cfg
from case_battery_demo import suction_io
from case_battery_demo.grasp import GripperMover, SuctionMover
from case_battery_demo.home_pose import go_to_default_pose
from case_battery_demo.sequence import _EE_PLACE_SEQ_PATH, _parse_ee_sequence, _pose


def _apply_speed(speed: float) -> None:
    """Scale only the gripper-approach and Robotiq-close speed by *speed*.

    Pick / lift / horizontal travel run at their normal config speeds — only the
    right-arm gripper approach and the Robotiq close byte are slowed so the
    handoff is easy to inspect without penalising the pick sequence.
    """
    s = max(0.05, min(1.0, speed))
    cfg.HANDOFF_GRIP_DURATION_S = cfg.HANDOFF_GRIP_DURATION_S / s
    cfg.ROBOTIQ_SPEED = max(1, int(cfg.ROBOTIQ_SPEED * s))
    logger.info("[test_handoff] gripper approach / Robotiq speed scaled to {:.0f}%", s * 100)


def main(
    src: str = "BAT_SRC_1",
    dst: str = "BAT_SLOT_1",
    speed: float = 0.4,
    next_src: str = "",
) -> None:
    """
    Args:
        src: Taught source pose name (BAT_SRC_1 or BAT_SRC_2).
        dst: Taught destination pose — used only for the horizontal move direction.
        speed: Motion speed multiplier 0.0..1.0 (default 0.4 = 40% of normal speed).
        next_src: Pose the suction arm hovers to after handoff (default: other battery source).
    """
    logger.warning("=" * 60)
    logger.warning("Partial handoff test: pick -> lift -> move -> GRIP (hold).")
    logger.warning("Ensure workspace is clear and e-stop is within reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    _apply_speed(speed)

    src_pose = _pose(src)
    dst_pose = _pose(dst)

    # Auto-infer next hover: toggle between the two battery sources.
    _SRC_CYCLE = {"BAT_SRC_1": "BAT_SRC_2", "BAT_SRC_2": "BAT_SRC_1"}
    next_src = next_src or _SRC_CYCLE.get(src, src)
    next_hover_pose = _pose(next_src)

    with Robot() as bot:
        suction_io.suction_off()

        left = SuctionMover(bot)
        right = GripperMover(bot)

        # --- readiness ---
        if left.software_estop_active():
            if input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            left._robot.estop.deactivate()
            time.sleep(0.5)
        left.ensure_ready()
        right.ensure_ready()

        if not right.gripper.available:
            logger.error("Right EE pass-through unavailable — cannot drive Robotiq. Aborting.")
            return
        if not right.gripper.activate():
            logger.error("Robotiq activation failed. Aborting.")
            return

        set_head_pitch(bot, pitch_deg=30.0)
        go_to_default_pose(bot)

        # --- 1. Pick ---
        logger.info("=== PICK: {} ===", src)
        result = left.pick(src_pose)
        if not result.success:
            logger.error("Pick failed (trigger={}). Aborting.", result.trigger)
            return
        logger.info("Pick OK at z={:.4f}m", float(result.contact_position_base[2]))

        # --- 2. Lift to SAFE_TRANSPORT_Z ---
        logger.info("=== LIFT ===")
        left.lift()

        # --- 3. Horizontal move toward dst at transport Z ---
        logger.info("=== MOVE HORIZONTAL -> {} ===", dst)
        left.move_to(dst_pose)

        # Read suction EE pose AFTER settling at transport
        suction_pos, suction_rpy = left.current_ee_pose()
        logger.info("Suction EE at transport: x={:.4f} y={:.4f} z={:.4f}", *suction_pos)

        # --- 4. Compute gripper target from live suction pose ---
        offset = np.asarray(cfg.HANDOFF_GRIP_OFFSET, dtype=float)
        grasp_pos = suction_pos + offset
        grasp_rpy = np.asarray(cfg.GRIPPER_GRASP_RPY, dtype=float)
        logger.info(
            "Gripper target: x={:.4f} y={:.4f} z={:.4f}  rpy={} deg",
            *grasp_pos, np.round(np.degrees(grasp_rpy), 1).tolist(),
        )

        # --- 5. Open gripper, move to pre-grasp standoff ---
        logger.info("=== RIGHT ARM: open + approach standoff ===")
        right.gripper.open()

        from case_battery_demo.grasp import _rpy_to_matrix
        approach_axis = _rpy_to_matrix(*grasp_rpy)[:, 2]   # local +z = approach direction
        pre_pos = grasp_pos - cfg.GRIPPER_PREGRASP_STANDOFF_M * approach_axis
        right._move_ee_to(pre_pos, grasp_rpy, cfg.HANDOFF_GRIP_DURATION_S)
        right._wait_until_arrived(pre_pos, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
        logger.info("At standoff. Moving in to grip position...")

        # --- 6. Move in to grasp pose ---
        right._move_ee_to(grasp_pos, grasp_rpy, cfg.HANDOFF_GRIP_DURATION_S)
        right._wait_until_arrived(grasp_pos, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

        # --- 7. Close (speed already scaled by _apply_speed) ---
        logger.info("=== CLOSE (Robotiq speed=0x{:02x} force=0x{:02x}) ===", cfg.ROBOTIQ_SPEED, cfg.ROBOTIQ_FORCE)
        final_pos = right.gripper.goto(cfg.ROBOTIQ_CLOSE_POS, speed=cfg.ROBOTIQ_SPEED, force=cfg.ROBOTIQ_FORCE)
        gripped = right.gripper.is_object_grasped()
        logger.info("Gripper closed -> gPO={} | object_grasped={}", final_pos, gripped)

        if not gripped:
            logger.error(
                "Gripper did NOT detect an object (gOBJ != 2, gPO={}) — "
                "keeping suction ON. Press Enter to open gripper and exit.", final_pos,
            )
            try:
                input()
            except KeyboardInterrupt:
                pass
            finally:
                right.gripper.open()
            return

        # Grip confirmed — release suction now.
        logger.info("Grip confirmed. Turning suction OFF.")
        suction_io.suction_off()
        time.sleep(0.3)

        # --- 8. Execute taught EE pose sequence via IK (slowly) ---
        steps = _parse_ee_sequence(_EE_PLACE_SEQ_PATH)
        if not steps:
            logger.error("No poses found in {} — cannot execute sequence.", _EE_POSE_SEQ_PATH)
            right.gripper.open()
            return

        step_duration = 4.0 / max(0.05, speed)
        logger.info("Executing {} EE pose steps at {:.1f}s each.", len(steps), step_duration)

        suction_retract_thread: threading.Thread | None = None

        for i, (label, vec, rpy_val, is_relative) in enumerate(steps):
            if is_relative:
                cur_pos, cur_rpy = right.current_ee_pose()
                pos = cur_pos + vec
                rpy = cur_rpy + rpy_val   # direct addition works well for single-axis deltas
                logger.info("=== STEP: {} (REL) delta_pos={} delta_rpy_deg={} ===",
                            label, np.round(vec, 4).tolist(),
                            np.round(np.degrees(rpy_val), 1).tolist())
            else:
                pos, rpy = vec, rpy_val
                logger.info("=== STEP: {} ===", label)
            right._move_ee_to(pos, rpy, step_duration)
            right._wait_until_arrived(pos, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

            # After the first gripper step, launch suction arm lift + move to
            # the next hover position in the background so both arms travel
            # concurrently while the gripper continues its sequence.
            if i == 0:
                def _retract_suction():
                    logger.info("[suction] lift + move to next hover: {}", next_src)
                    left.lift()
                    left.move_to(next_hover_pose)
                    logger.info("[suction] arrived at next hover.")
                suction_retract_thread = threading.Thread(
                    target=_retract_suction, name="suction-retract", daemon=True
                )
                suction_retract_thread.start()

            if "lower" in label.lower():
                logger.info("At Lower pose — partial open (pos={}).", cfg.ROBOTIQ_PARTIAL_OPEN_POS)
                right.gripper.partial_open()

        # Wait for the suction arm to finish before exiting.
        if suction_retract_thread is not None:
            suction_retract_thread.join()

        logger.info("Sequence complete.")


if __name__ == "__main__":
    tyro.cli(main)
