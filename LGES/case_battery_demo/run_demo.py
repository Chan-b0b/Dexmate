"""Entry point for the case + battery suction demo.

Run from the LGES directory so the package import resolves:

    python -m case_battery_demo.run_demo                 # forward only
    python -m case_battery_demo.run_demo --undo          # forward then undo
    python -m case_battery_demo.run_demo --undo-only     # undo a prior run*
    python -m case_battery_demo.run_demo --loop          # forward+undo, repeat

(*--undo-only requires the forward moves to have been recorded; in a single
process that means running forward first. It is provided mainly for clarity.)
"""

from __future__ import annotations

import os
import sys

import tyro
from loguru import logger

from dexcontrol.robot import Robot

# LGES/ is the parent of this package; expose its utils.py for import.
_LGES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LGES_DIR not in sys.path:
    sys.path.insert(0, _LGES_DIR)
from utils import set_head_pitch  # noqa: E402

from .grasp import SuctionMover
from .home_pose import go_to_default_pose
from .sequence import TaskOrchestrator
from . import suction_io


def main(undo: bool = False, loop: bool = False, skip_confirmation: bool = False) -> bool:
    """Run the forward choreography, optionally followed by the undo.

    Args:
        undo: If True, replay the sequence in reverse after the forward run.
        loop: If True, repeat forward + undo indefinitely until Ctrl-C.
            Implies --undo.
        skip_confirmation: Skip the interactive safety prompt.
    """
    if loop:
        undo = True
    logger.warning("=" * 60)
    logger.warning("About to move the REAL robot arm with suction.")
    logger.warning("Ensure the workspace is clear and the e-stop is reachable.")
    if loop:
        logger.warning("LOOP MODE: forward + undo will repeat until Ctrl-C.")
    logger.warning("=" * 60)
    if not skip_confirmation:
        if input("Continue? [y/N]: ").strip().lower() != "y":
            logger.info("Cancelled.")
            return False

    with Robot() as bot:
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

            orch = TaskOrchestrator(mover)
            iteration = 0
            try:
                while True:
                    iteration += 1
                    if loop:
                        logger.info("=== Loop iteration {} ===", iteration)

                    # Send both arms to the joint-space "Home" pose before
                    # each forward choreography.
                    go_to_default_pose(bot)

                    if not orch.run_forward():
                        logger.error("Forward sequence failed — leaving robot where it stopped.")
                        return False
                    if undo:
                        if not orch.run_undo():
                            logger.error("Undo sequence failed — leaving robot where it stopped.")
                            return False
                    if not loop:
                        break
            except KeyboardInterrupt:
                logger.warning("Loop interrupted by user after {} iteration(s).", iteration)
                return True
    return True


if __name__ == "__main__":
    success = tyro.cli(main)
    raise SystemExit(0 if success else 1)
