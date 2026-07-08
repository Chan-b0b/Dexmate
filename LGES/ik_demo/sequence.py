"""Forward task orchestration for ik_demo.

Forward only (no undo): move the case left -> right, then seat two batteries in
it. Each move is a suction pick(src) -> place(dst); pick and place both start
and end at transport height, so the sideways travel lives inside place's
approach. Retry-on-failure is config-gated.

Barcode divert + gripper handoff (for the battery moves) are layered on in
barcode.py / gripper.py (steps 6-7); the hook is `_run_move`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

try:
    from . import config as cfg
    from .suction import SuctionMover
    from .gripper import GripperMover
    from .barcode import is_target
    from .drivers import suction_io
except ImportError:  # allow `python sequence.py` from inside ik_demo/
    import config as cfg
    from suction import SuctionMover
    from gripper import GripperMover
    from barcode import is_target
    from drivers import suction_io


@dataclass(frozen=True)
class Move:
    label: str      # phase name
    src: str        # taught-pose name to pick from
    dst: str        # taught-pose name to place at


# The forward choreography.
FORWARD_MOVES: tuple[Move, ...] = (
    Move("case", "CASE_PICK", "CASE_PLACE_R"),
    Move("battery_1", "BAT_SRC_1", "BAT_SLOT_1"),
    Move("battery_2", "BAT_SRC_2", "BAT_SLOT_2"),
)


class TaskOrchestrator:
    """Runs the forward sequence on a SuctionMover."""

    def __init__(self, mover: SuctionMover, gripper: "GripperMover | None" = None) -> None:
        self._mover = mover
        self._gripper = gripper
        # Each column is anchored on its clean layer-1 contact z; later layers are
        # PREDICTED as anchor +/- (layer-1)*LAYER_PITCH_M (sources shrink, targets
        # grow). The anchor never updates from later layers, so one misaligned seat
        # is flagged (not absorbed) and can't corrupt subsequent predictions.
        self._z_anchor: dict[str, float] = {}

    def run_forward(self) -> bool:
        """Build NUM_LAYERS layers; each runs the full choreography. Returns True
        iff every move of every layer succeeded."""
        for layer in range(1, int(cfg.NUM_LAYERS) + 1):
            logger.info("===== LAYER {}/{} =====", layer, cfg.NUM_LAYERS)
            for mv in FORWARD_MOVES:
                logger.info("=== phase: {} ({} -> {}) [layer {}] ===", mv.label, mv.src, mv.dst, layer)
                if not self._run_move(mv, layer):
                    logger.error("phase {} failed on layer {} — stopping (robot left where it is).",
                                 mv.label, layer)
                    return False
        logger.info("=== all {} layers complete ===", cfg.NUM_LAYERS)
        return True

    def _predicted_z(self, name: str, layer: int, is_source: bool) -> "float | None":
        """Predicted contact z: anchor +/- (layer-1)*pitch. None until anchored
        (layer 1, or a column whose layer-1 contact was never measured) -> the
        descent falls back to the taught z."""
        anchor = self._z_anchor.get(name)
        if anchor is None:
            return None
        sign = -1.0 if is_source else 1.0
        return anchor + sign * (layer - 1) * cfg.LAYER_PITCH_M

    def _record_z(self, name: str, z: "float | None", predicted: "float | None") -> None:
        """Anchor a column on its first measured contact; on later layers, flag a
        seat that deviates from the prediction by more than the misalign band."""
        if z is None:
            return
        if name not in self._z_anchor:
            self._z_anchor[name] = z          # layer-1 anchor (clean seat)
        if predicted is not None:
            resid = z - predicted
            if abs(resid) > cfg.LAYER_MISALIGN_FRAC * cfg.LAYER_PITCH_M:
                logger.warning("[{}] seat z={:.4f} vs predicted {:.4f} ({:+.1f}mm) — possible misalignment",
                               name, z, predicted, resid * 1000.0)
            else:
                logger.info("[{}] seat z={:.4f} ({:+.1f}mm vs predicted)", name, z, resid * 1000.0)

    def _run_move(self, mv: Move, layer: int) -> bool:
        """pick(src) -> place(dst), with config-gated retry on a failed pick.

        Battery moves scan the barcode DURING the pick descent; a target match
        is logged but still placed in its case slot like any other battery
        (gripper divert removed — sequence may change to not need a handoff).
        Descent z is predicted from each column's layer-1 anchor + constant pitch.
        """
        # Battery moves scan for the barcode during the pick (barcode-gated pick).
        gated = mv.label.startswith("battery")
        pred_src = self._predicted_z(mv.src, layer, is_source=True)
        pred_dst = self._predicted_z(mv.dst, layer, is_source=False)
        attempts = max(1, int(cfg.MAX_PHASE_ATTEMPTS)) if cfg.RETRY_FAILED_PHASE else 1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                logger.warning("retry {}/{} for phase {} (pausing {:.1f}s)",
                               attempt, attempts, mv.label, cfg.PHASE_RETRY_DELAY_S)
                time.sleep(cfg.PHASE_RETRY_DELAY_S)

            if gated:
                res = self._mover.pick_gated(cfg.TAUGHT_POSES[mv.src], expected_z=pred_src)
            else:
                res = self._mover.pick(cfg.TAUGHT_POSES[mv.src], expected_z=pred_src)
            if not res.success:
                logger.warning("pick failed in {}: {}", mv.label, res.reason)
                continue  # retry the pick (arm is safe: suction off, or held-fail)
            self._record_z(mv.src, res.contact_ee_z, pred_src)
            if res.barcode is not None:
                logger.info("[{}] barcode {!r} (target={})", mv.label, res.barcode, is_target(res.barcode))
            if is_target(res.barcode):
                logger.info("[{}] TARGET battery {!r} — placing in case (divert disabled)",
                            mv.label, res.barcode)
            pres = self._mover.place(cfg.TAUGHT_POSES[mv.dst], expected_z=pred_dst)
            self._record_z(mv.dst, pres.contact_ee_z, pred_dst)
            return True
        return False


# ---------------------------------------------------------------------------
# On-robot run:  python sequence.py
#   home -> forward sequence (case + 2 batteries). Behind a safety prompt.
# ---------------------------------------------------------------------------
def _run_on_robot() -> None:
    import sys
    from dexcontrol.robot import Robot

    use_gripper = "--gripper" in sys.argv   # enable the barcode divert

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION through the full forward sequence:")
    for mv in FORWARD_MOVES:
        logger.warning("   {}: {} -> {}", mv.label, mv.src, mv.dst)
    if use_gripper:
        logger.warning("Barcode divert ENABLED (target codes -> right gripper).")
    logger.warning("Place the case + batteries at the taught spots. E-stop in reach.")
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

            gripper = None
            if use_gripper:
                gripper = GripperMover(bot)
                gripper.ensure_ready(release_estop=release)
                if not gripper.initialize():
                    logger.warning("gripper unavailable — divert disabled")
                    gripper = None

            logger.info("-> home")
            m.move_joints(m._home_seed)
            ok = TaskOrchestrator(m, gripper).run_forward()
            logger.info("-> home")
            m.move_joints(m._home_seed)
            logger.info("sequence {}", "OK" if ok else "FAILED")


if __name__ == "__main__":
    _run_on_robot()
