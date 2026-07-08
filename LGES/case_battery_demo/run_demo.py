"""Entry point for the case + battery suction demo.

Run from the LGES directory so the package import resolves:

    python -m case_battery_demo.run_demo                 # forward only
    python -m case_battery_demo.run_demo --undo          # forward then undo
    python -m case_battery_demo.run_demo --undo-only     # undo from taught poses
    python -m case_battery_demo.run_demo --loop          # forward+undo, repeat
    python -m case_battery_demo.run_demo --dashboard     

``--undo-only`` does not require a prior forward run in this process; it
builds the undo sequence directly from the taught poses (so place z falls
back to the taught src z rather than a recorded pick z).
"""

from __future__ import annotations

import os
import random
import sys

import tyro
from loguru import logger

from dexcontrol.robot import Robot

# LGES/ is the parent of this package; expose its utils.py for import.
_LGES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LGES_DIR not in sys.path:
    sys.path.insert(0, _LGES_DIR)
from utils import set_head_pitch  # noqa: E402

from .grasp import GripperMover, SuctionMover
from .home_pose import go_to_default_pose
from .sequence import TaskOrchestrator, set_episode_shift
from . import suction_io
from . import config as cfg


def main(undo: bool = False, undo_only: bool = False, loop: bool = False, skip_confirmation: bool = False, dashboard: bool = True, record: bool = True, record_dir: str = "recordings", instruction: str = "") -> bool:
    """Run the forward choreography, optionally followed by the undo.

    Args:
        undo: If True, replay the sequence in reverse after the forward run.
        undo_only: If True, skip the forward run and only execute the undo
            sequence derived from the taught poses.
        loop: If True, repeat forward + undo indefinitely until Ctrl-C.
            Implies --undo.
        skip_confirmation: Skip the interactive safety prompt.
        dashboard: If True, spool live camera/joints/EE/wrench for the web
            viewer (run ``python -m case_battery_demo.dashboard.server``
            in a separate terminal to watch it).
        record: If True, enable the episode recorder (implies the dashboard
            spool). Episodes are cut automatically at each sub-task boundary
            (case/battery pick & place, hand_off, gripper handling), each with
            its own instruction and success flag. SPACE arms/disarms auto
            cutting; 'd' discards the last saved take. The dashboard Record
            button still drives ad-hoc manual takes.
        record_dir: Where kept takes are written (one dir per episode).
        instruction: Fallback instruction for manual takes / unknown phases
            (auto episodes use cfg.PHASE_INSTRUCTIONS).
    """
    if loop:
        undo = True
    if undo_only and loop:
        logger.error("--undo-only is incompatible with --loop.")
        return False
    # Per-run episode shift for VLA spatial diversity: sampled and announced
    # BEFORE the confirmation prompt so the operator can place the physical
    # case/battery stacks shifted by the same amount, then confirm.
    shift_xy = (0.0, 0.0)
    shift_max = float(getattr(cfg, "EPISODE_XY_SHIFT_MAX_M", 0.0))
    if shift_max > 0.0:
        shift_xy = (random.uniform(-shift_max, shift_max),
                    random.uniform(-shift_max, shift_max))
        set_episode_shift(*shift_xy)

    logger.warning("=" * 60)
    logger.warning("About to move the REAL robot arm with suction.")
    logger.warning("Ensure the workspace is clear and the e-stop is reachable.")
    if shift_max > 0.0:
        logger.warning("EPISODE SHIFT (base_link): dx={:+.1f} mm, dy={:+.1f} mm", shift_xy[0] * 1000.0, shift_xy[1] * 1000.0)
        logger.warning("-> Place the case + battery stacks shifted by this from their taught spots.")
    if loop:
        logger.warning("LOOP MODE: forward + undo will repeat until Ctrl-C.")
    logger.warning("=" * 60)
    if not skip_confirmation:
        if input("Continue? [y/N]: ").strip().lower() != "y":
            logger.info("Cancelled.")
            return False

    # The head camera is disabled in the default robot config, so the dashboard
    # gets no frames unless we explicitly enable it (joints/EE/wrench come from
    # motor components, which is why those show up but the image stays blank).
    robot_configs = None
    if dashboard or record:
        from dexcontrol.core.config import get_robot_config
        robot_configs = get_robot_config()
        robot_configs.enable_sensor("head_camera")
        robot_configs.sensors["head_camera"].transport = "zenoh"

    with Robot(configs=robot_configs) as bot:
        # Start with suction off so we never grab during the approach.
        suction_io.suction_off()
        with SuctionMover(bot) as mover:
            if mover.software_estop_active():
                logger.warning("Software E-Stop is ACTIVE — the arm cannot move until released.")
                if input("Release software E-Stop and enable the arm? [y/N]: ").strip().lower() != "y":
                    logger.info("Leaving E-Stop engaged; aborting.")
                    return False
                release = True
            else:
                release = False
            if not mover.ensure_ready(release_estop=release):
                logger.error("Arm not ready (E-Stop still active?). Aborting.")
                return False

            # Tilt the head down to 30° so cameras see the workspace.
            set_head_pitch(bot, pitch_deg=30.0)

            # Right-arm Robotiq gripper for the barcode-matched battery handoff.
            # If its EE pass-through isn't available, the demo falls back to the
            # original suction-only behaviour (no scanning, no diversion).
            gripper = GripperMover(bot)
            gripper.ensure_ready(release_estop=release)
            if not gripper.initialize():
                logger.warning("Right gripper unavailable — barcode handoff disabled.")
                gripper = None

            publisher = None
            recorder = None
            keys = None
            if dashboard or record:
                from .dashboard.publisher import DEFAULT_SPOOL_DIR, DashboardPublisher
                sink = None
                if record:
                    from .dashboard.recorder import KeyListener, RecordController
                    recorder = RecordController(
                        out_dir=record_dir, spool_dir=DEFAULT_SPOOL_DIR, instruction=instruction
                    )
                    recorder.set_meta_extra({
                        "episode_xy_shift_m": [round(shift_xy[0], 4), round(shift_xy[1], 4)],
                    })
                    recorder.start()
                    sink = recorder.feed
                    keys = KeyListener(recorder).start()
                publisher = DashboardPublisher(bot, on_sample=sink).start()

            orch = TaskOrchestrator(mover, gripper, recorder=recorder)
            iteration = 0
            try:
                while True:
                    iteration += 1
                    if loop:
                        logger.info("=== Loop iteration {} ===", iteration)

                    # Send both arms to the joint-space "Home" pose before
                    # each forward choreography.
                    go_to_default_pose(bot)

                    if not undo_only:
                        repeats = max(1, int(getattr(cfg, "FORWARD_REPEATS", 1)))
                        forward_ok = True
                        for k in range(repeats):
                            if repeats > 1:
                                logger.info("--- Forward pass {}/{} (repeat={}) ---", k + 1, repeats, k)
                            if not orch.run_forward(repeat=k):
                                logger.error("Forward sequence failed at pass {} — leaving robot where it stopped.", k + 1)
                                forward_ok = False
                                break
                        if not forward_ok:
                            return False
                    if undo or undo_only:
                        if not orch.run_undo():
                            logger.error("Undo sequence failed — leaving robot where it stopped.")
                            return False
                    if not loop:
                        break
            except KeyboardInterrupt:
                logger.warning("Loop interrupted by user after {} iteration(s).", iteration)
                return True
            finally:
                if keys is not None:
                    keys.stop()
                if recorder is not None:
                    recorder.stop()
                if publisher is not None:
                    publisher.stop()
    return True


if __name__ == "__main__":
    success = tyro.cli(main)
    raise SystemExit(0 if success else 1)
