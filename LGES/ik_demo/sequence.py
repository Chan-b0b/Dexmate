"""Forward task orchestration for ik_demo.

Forward only (no undo): move the case left -> right, then seat two batteries in
it. Each move is a suction pick(src) -> place(dst); pick and place both start
and end at transport height, so the sideways travel lives inside place's
approach. Retry-on-failure is config-gated.

Battery moves scan the barcode during the pick descent (barcode.py) and the
result is logged here. The divert itself lives in chassis_sequence
(_divert_case_place); the old right-arm gripper handoff was removed.
"""
#python -m ik_demo.sequence


#export ROBOT_IP=192.168.50.20


from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

try:
    from . import config as cfg
    from .suction import SuctionMover
    from .barcode import is_target
    from .drivers import suction_io
    from .go_home import safe_home
except ImportError:  # allow `python sequence.py` from inside ik_demo/
    import config as cfg
    from suction import SuctionMover
    from barcode import is_target
    from drivers import suction_io
    from go_home import safe_home


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

    def __init__(self, mover: SuctionMover) -> None:
        self._mover = mover
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

        Battery moves scan the barcode DURING the pick descent; the result is
        logged (TARGET_BARCODES match or not) and the battery is seated in its
        case slot either way — this fixed-station flow has no divert.
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
    from pathlib import Path

    from dexcontrol.robot import Robot

    # perception/utils.py (sibling package) for set_head_pitch — same
    # sys.path setup chassis_sequence.py uses to reach it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
    from utils import set_head_pitch  # noqa: E402

    use_dashboard = "--dashboard" in sys.argv  # spool camera/joints/EE/wrench for the web viewer

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION through the full forward sequence:")
    for mv in FORWARD_MOVES:
        logger.warning("   {}: {} -> {}", mv.label, mv.src, mv.dst)
    if use_dashboard:
        logger.warning("Dashboard spool ENABLED — view with run_dashboard_demo.sh.")
    logger.warning("Place the case + batteries at the taught spots. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    # The head camera is disabled in the default robot config, so the
    # dashboard gets no frames unless we explicitly enable it here.
    robot_configs = None
    if use_dashboard:
        from dexcontrol.core.config import get_robot_config
        robot_configs = get_robot_config()
        robot_configs.enable_sensor("head_camera")
        robot_configs.sensors["head_camera"].transport = "zenoh"

    suction_io.suction_off()
    with Robot(configs=robot_configs) as bot:
        set_head_pitch(bot, angle=30.0)
        publisher = None
        if use_dashboard:
            from .dashboard_publish import DashboardPublisher
            publisher = DashboardPublisher(bot).start()
        try:
            with SuctionMover(bot) as m:
                release = m.software_estop_active()
                if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                    return
                if not m.ensure_ready(release_estop=release):
                    logger.error("arm not ready — aborting")
                    return

                logger.info("-> home")
                m.move_joints(m._home_seed)
                ok = TaskOrchestrator(m).run_forward()
                if ok:
                    logger.info("-> home")
                    m.move_joints(m._home_seed)
                else:
                    # Failed mid-move — the arm may be low over a box; lift clear
                    # before the joint-space home move instead of homing right away.
                    safe_home(m)
                logger.info("sequence {}", "OK" if ok else "FAILED")
        finally:
            if publisher is not None:
                publisher.stop()


if __name__ == "__main__":
    _run_on_robot()
