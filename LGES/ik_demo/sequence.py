"""Forward task orchestration for ik_demo.

Forward only (no undo): move the case left -> right, then seat two batteries in
it. Each move is a suction pick(src) -> place(dst); pick and place both start
and end at transport height, so the sideways travel lives inside place's
approach. Retry-on-failure is config-gated.

Barcode divert + gripper handoff (for the battery moves) are layered on in
barcode.py / gripper.py (steps 6-7); the hook is `_run_move`.
"""
#python -m ik_demo.sequence --gripper


#export ROBOT_IP=192.168.50.20


from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from loguru import logger

try:
    from . import config as cfg
    from .suction import SuctionMover
    from .gripper import GripperMover
    from .barcode import is_target
    from .drivers import suction_io
    from .go_home import safe_home
except ImportError:  # allow `python sequence.py` from inside ik_demo/
    import config as cfg
    from suction import SuctionMover
    from gripper import GripperMover
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

    def __init__(self, mover: SuctionMover, gripper: "GripperMover | None" = None) -> None:
        self._mover = mover
        self._gripper = gripper
        # Each column is anchored on its clean layer-1 contact z; later layers are
        # PREDICTED as anchor +/- (layer-1)*LAYER_PITCH_M (sources shrink, targets
        # grow). The anchor never updates from later layers, so one misaligned seat
        # is flagged (not absorbed) and can't corrupt subsequent predictions.
        self._z_anchor: dict[str, float] = {}
        # Right-arm place+home running in the background after a divert (see
        # _divert) — joined before the next grip, and at the end of run_forward.
        self._gripper_thread: "threading.Thread | None" = None

    def run_forward(self) -> bool:
        """Build NUM_LAYERS layers; each runs the full choreography. Returns True
        iff every move of every layer succeeded."""
        try:
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
        finally:
            # Never leave the caller (which homes both arms next) racing a
            # still-running right-arm background place/home.
            self._join_gripper()

    def _join_gripper(self) -> None:
        if self._gripper_thread is not None:
            self._gripper_thread.join()
            self._gripper_thread = None

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

        Battery moves scan the barcode DURING the pick descent; a target match is
        diverted to the right-arm gripper (two-arm handoff) when a gripper is
        available, else seated in its case slot like any other battery. A divert
        that can't grip falls back to the case place.
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
            if self._gripper is not None and is_target(res.barcode):
                logger.info("[{}] TARGET battery {!r} — diverting to right-arm gripper",
                            mv.label, res.barcode)
                if self._divert(mv):
                    return True
                logger.warning("[{}] divert failed — seating in case instead", mv.label)
            pres = self._mover.place(cfg.TAUGHT_POSES[mv.dst], expected_z=pred_dst)
            self._record_z(mv.dst, pres.contact_ee_z, pred_dst)
            return True
        return False

    def _divert(self, mv: Move) -> bool:
        """Two-arm handoff of the suction-held battery to the right-arm gripper.

        The suction arm first carries the battery sideways (at transport height,
        no descent) to a fixed hover xy (cfg.HANDOFF_HOVER_XY — moving right),
        the SAME spot regardless of which battery (1 or 2) triggered the divert;
        the gripper then side-grips at that (new) suction EE pose +
        HANDOFF_GRIP_OFFSET. Once it confirms a grasp, suction releases.

        The gripper's place sequence (carry -> lower/release -> retract) and its
        return home then run on a background thread; this call blocks only until
        that sequence reaches "To Right 2" (carry-away done, clear of the
        handoff spot) and returns True from there — the left arm is then free
        to get on with the REST of the choreography (the next pick) while the
        right arm finishes placing and homing in parallel. The next divert (or
        run_forward, at the end) joins that thread before the gripper is used
        again. Returns False (leaving the part still on the cup, so the caller
        can fall back to a case place) if the place sequence isn't taught or
        the gripper reports no object.
        """
        if not cfg.PLACE_LOWER_RIGHT_EE_SEQ:
            logger.warning("[{}] PLACE_LOWER_RIGHT_EE_SEQ not taught — skipping divert", mv.label)
            return False
        self._join_gripper()  # can't grip again until the previous place+home finished
        pos, rpy = self._mover.current_ee_pose()
        tx, ty = cfg.HANDOFF_HOVER_XY
        logger.info("[{}] divert: suction arm -> fixed hover (moving right)", mv.label)
        if self._mover.move_ee([tx, ty, float(pos[2])], rpy) is None:
            logger.warning("[{}] target hover unreachable — gripping at current pose instead", mv.label)
        pos, _ = self._mover.current_ee_pose()
        off = cfg.HANDOFF_GRIP_OFFSET
        grasp_pos = [float(pos[i]) + off[i] for i in range(3)]
        logger.info("[{}] divert: right-arm side-grip at suction EE + offset", mv.label)
        if not self._gripper.grip_at(grasp_pos):
            logger.warning("[{}] gripper reported no object — aborting divert", mv.label)
            return False
        suction_io.release()            # suction lets go; the gripper now holds the battery

        cleared = threading.Event()

        def _run_place_and_home() -> None:
            ok = self._gripper.place_ee_seq(on_step={"To Right 2": cleared.set})
            cleared.set()  # don't strand the left arm if "To Right 2" was never reached
            if not ok:
                # place_ee_seq only releases at "Lower" (or fully at the end) — a
                # failure before that leaves the battery still clamped. Homing
                # anyway would drag it through a trajectory never meant to carry
                # a payload, so check first and stop in place if it's still held.
                if self._gripper.gripper.is_object_grasped():
                    logger.error("[{}] right-arm place sequence failed (background) — "
                                 "still holding the battery, NOT homing (left where it stalled)",
                                 mv.label)
                    return
                logger.error("[{}] right-arm place sequence failed (background) — "
                             "gripper empty, homing", mv.label)
            logger.info("[{}] right arm -> home (background)", mv.label)
            self._gripper.move_joints(self._gripper._home_seed)

        self._gripper_thread = threading.Thread(target=_run_place_and_home, daemon=True)
        self._gripper_thread.start()
        cleared.wait()
        logger.info("[{}] right arm clear of handoff — left arm continuing", mv.label)
        return True


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

    use_gripper = "--gripper" in sys.argv     # enable the barcode divert
    use_dashboard = "--dashboard" in sys.argv  # spool camera/joints/EE/wrench for the web viewer

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION through the full forward sequence:")
    for mv in FORWARD_MOVES:
        logger.warning("   {}: {} -> {}", mv.label, mv.src, mv.dst)
    if use_gripper:
        logger.warning("Barcode divert ENABLED (target codes -> right gripper).")
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

                gripper = None
                if use_gripper:
                    gripper = GripperMover(bot)
                    gripper.ensure_ready(release_estop=release)
                    if not gripper.initialize():
                        logger.warning("gripper unavailable — divert disabled")
                        gripper = None

                logger.info("-> home")
                m.move_joints(m._home_seed)
                if gripper is not None:
                    logger.info("-> right arm home")
                    gripper.move_joints(gripper._home_seed)
                ok = TaskOrchestrator(m, gripper).run_forward()
                if ok:
                    logger.info("-> home")
                    m.move_joints(m._home_seed)
                else:
                    # Failed mid-move — the arm may be low over a box; lift clear
                    # before the joint-space home move instead of homing right away.
                    safe_home(m)
                if gripper is not None:
                    logger.info("-> right arm home")
                    gripper.move_joints(gripper._home_seed)
                logger.info("sequence {}", "OK" if ok else "FAILED")
        finally:
            if publisher is not None:
                publisher.stop()


if __name__ == "__main__":
    _run_on_robot()
