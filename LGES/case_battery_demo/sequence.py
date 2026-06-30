"""Task orchestrator: case + battery choreography with undo.

Forward sequence:
    1. case      : CASE_PICK     -> CASE_PLACE_R
    2. battery_1 : BAT_SRC_1      -> BAT_SLOT_1   (into the moved case)
    3. battery_2 : BAT_SRC_2      -> BAT_SLOT_2

Undo replays the recorded moves in reverse order with from/to swapped, which
naturally gives "batteries right -> left first, then the case":
    battery_2 : BAT_SLOT_2 -> BAT_SRC_2
    battery_1 : BAT_SLOT_1 -> BAT_SRC_1
    case      : CASE_PLACE_R -> CASE_PICK

Every move is pick -> lift -> move_to -> place, so the case AND each battery
are raised to SAFE_TRANSPORT_Z before any sideways travel.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from . import bcr
from . import config as cfg
from . import suction_io
from .dashboard.barcode import DEFAULT_SPOOL_DIR
from .grasp import GripperMover, Pose, SuctionMover
from .home_pose import DEFAULT_POSE_PATH, _parse_joint_lines


def _publish_target_height(repeat: int) -> None:
    """Spool the current case-place target height for the dashboard panel.

    CASE_PLACE_R's Z is raised by ``repeat * dst_dz`` each forward pass; this is
    the value the bin-detection panel shows against the depth-measured height at
    the bin centre. Best-effort: a failed write must never disturb the demo.
    """
    base_z = cfg.TAUGHT_POSES["CASE_PLACE_R"][2]
    dst_dz = cfg.Z_STEP_PER_REPEAT.get("case", (0.0, 0.0))[1]
    payload = {
        "stamp": time.time(),
        "repeat": repeat,
        "dst_dz": dst_dz,
        "target_z_m": base_z + repeat * dst_dz,
    }
    path = os.path.join(DEFAULT_SPOOL_DIR, "target.json")
    try:
        os.makedirs(DEFAULT_SPOOL_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _publish_barcode_read(code: str | None, matched: bool, label: str) -> None:
    """Spool the latest decoded barcode so the dashboard panel can show it.

    The barcode image feed (dashboard/barcode.py) is image-only and never
    triggers a read, so the decoded string only exists here. Best-effort: a
    failed write must never disturb the demo.
    """
    payload = {"stamp": time.time(), "label": label, "code": code, "matched": matched}
    path = os.path.join(DEFAULT_SPOOL_DIR, "barcode_read.json")
    try:
        os.makedirs(DEFAULT_SPOOL_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# EE place-sequence helpers (shared with test_handoff.py)
# ---------------------------------------------------------------------------

_EE_PLACE_SEQ_PATH = os.path.join(os.path.dirname(__file__), "taught_ee_poses_right.txt")
_EE_PLACE_LABELS = ["To Right 1", "To Right 2", "To Right 3", "Lower", "Back"]


def _parse_ee_sequence(path: str) -> list[tuple[str, np.ndarray, np.ndarray, bool]]:
    """Parse an EE pose file into (label, vec[3], rpy[3], is_relative) steps.

    Two line formats (space-separated):
      x y z roll pitch yaw          — absolute base-frame pose
      REL dx dy dz droll dpitch dyaw — delta added to the current live EE pose
    Labels are assigned from _EE_PLACE_LABELS in order.
    """
    steps = []
    idx = 0
    with open(path) as f:
        for i, raw in enumerate(f):
            tokens = raw.strip().split()
            if not tokens:
                continue
            relative = tokens[0].upper() == "REL"
            nums = tokens[1:] if relative else tokens
            if len(nums) != 6:
                raise ValueError(
                    f"Line {i + 1}: expected 6 numbers, got {len(nums)}: {raw.strip()!r}"
                )
            arr = np.array([float(v) for v in nums])
            label = _EE_PLACE_LABELS[idx] if idx < len(_EE_PLACE_LABELS) else f"step_{idx}"
            steps.append((label, arr[:3], arr[3:], relative))
            idx += 1
    return steps


@dataclass
class Move:
    label: str
    src: Pose
    dst: Pose
    # Sealed z observed during the forward pick of *this* move (cup-tip in
    # base_link). Captured by the orchestrator after a successful pick and
    # used as the place z when this move is reversed for undo, so the
    # battery returns to exactly the height it was picked from instead of
    # whatever (possibly stale) z was taught.
    actual_pick_z: float | None = field(default=None, repr=False)

    def reversed(self) -> "Move":
        # When undoing, place at the recorded pick z if we have one;
        # otherwise fall back to the taught src z.
        if self.actual_pick_z is not None:
            new_pos = self.src.pos.copy()
            new_pos[2] = float(self.actual_pick_z)
            new_dst = Pose(pos=new_pos, rpy=self.src.rpy)
        else:
            new_dst = self.src
        return Move(label=f"undo:{self.label}", src=self.dst, dst=new_dst)


def _pose(name: str) -> Pose:
    """Build a Pose from a taught config entry, erroring if untaught.

    Accepts a 3-tuple (xyz, uses the vertical GRASP_ORIENTATION_RPY) or a
    6-tuple (xyz + recorded roll, pitch, yaw)."""
    vals = cfg.TAUGHT_POSES.get(name)
    if vals is None:
        raise ValueError(f"Taught pose {name!r} is not set — capture it with teach_pose.py")
    if len(vals) == 6:
        import numpy as np
        return Pose(pos=np.array(vals[:3], dtype=float), rpy=np.array(vals[3:], dtype=float))
    if len(vals) == 3:
        return Pose.from_xyz(vals)
    raise ValueError(f"Taught pose {name!r} must have 3 or 6 values, got {len(vals)}")


# Equal x,y shift applied to every taught pose this run (VLA spatial
# diversity). Set once at startup via set_episode_shift(); the operator places
# the physical objects shifted by the same amount.
_EPISODE_SHIFT_XY = np.zeros(2)


def set_episode_shift(dx: float, dy: float) -> None:
    _EPISODE_SHIFT_XY[:] = (dx, dy)


def build_forward_moves(repeat: int = 0) -> list[Move]:
    """Build the three forward moves, offset in Z for stacking.

    ``repeat`` is the 0-based pass index. Each move's src and dst Z are
    shifted by ``repeat * cfg.Z_STEP_PER_REPEAT[label]`` so the source stack
    shrinks and the target stack grows across repeats. Pose objects are
    freshly constructed here, so mutating them is safe. The run's episode
    x,y shift is applied to every src and dst.
    """
    moves = [
        Move("case", _pose("CASE_PICK"), _pose("CASE_PLACE_R")),
        Move("battery_1", _pose("BAT_SRC_1"), _pose("BAT_SLOT_1")),
        Move("battery_2", _pose("BAT_SRC_2"), _pose("BAT_SLOT_2")),
    ]
    for m in moves:
        m.src.pos[:2] += _EPISODE_SHIFT_XY
        m.dst.pos[:2] += _EPISODE_SHIFT_XY
    if repeat:
        for m in moves:
            src_dz, dst_dz = cfg.Z_STEP_PER_REPEAT.get(m.label, (0.0, 0.0))
            m.src.pos[2] += repeat * src_dz
            m.dst.pos[2] += repeat * dst_dz
    return moves


class TaskOrchestrator:
    def __init__(self, mover: SuctionMover, gripper: GripperMover | None = None,
                 recorder=None) -> None:
        self._mover = mover
        self._gripper = gripper
        self._recorder = recorder  # RecordController or None; auto-cuts VLA episodes
        self._done: list[Move] = []  # successfully executed moves, for undo

    def _rec_begin(self, phase: str) -> None:
        if self._recorder is not None:
            self._recorder.episode_begin(phase)

    def _rec_end(self, success: bool) -> None:
        if self._recorder is not None:
            self._recorder.episode_end(success)

    def _rec_barcode(self, confirmed: bool) -> None:
        if self._recorder is not None:
            self._recorder.set_barcode_confirmed(confirmed)

    def _execute(self, move: Move) -> bool:
        # Reload config so edits made between moves (during --loop, or while
        # the operator is tuning thresholds in another editor) take effect on
        # the next move. Both this module and grasp.py hold the same module
        # object via `from . import config as cfg`, so importlib.reload
        # updates the constants in place for both.
        importlib.reload(cfg)
        logger.info("=== Move: {} ===", move.label)

        # Each move starts with the barcode unconfirmed; set True once a battery
        # scan reads a code (below). Reset here rather than after the place so an
        # aborted pick or a case move can't leak a stale True into the next take.
        self._rec_barcode(False)

        # Forward battery picks resolve the barcode BEFORE grabbing: pick() runs
        # the scan gate (descend to a floor above contact, scan, x/y spiral
        # search on a no-read), then seals. The case move and undo moves never
        # scan. The scanner runs in a daemon thread so it can't stall the loop.
        scan_this = move.label.startswith("battery") and self._gripper is not None
        scanner = bcr.BackgroundScanner() if scan_this else None

        self._rec_begin(f"{move.label}_pick")
        result = self._mover.pick(move.src, scanner=scanner, scan_gate=scan_this)

        if not result.success:
            self._rec_end(False)
            logger.error("[{}] pick failed (trigger={}) — aborting", move.label, result.trigger)
            return False
        # Remember the sealed z so a future undo of this same move places
        # the battery back at exactly the height it was picked from.
        if result.contact_position_base is not None:
            move.actual_pick_z = float(result.contact_position_base[2])
            logger.info("[{}] recorded pick z={:.4f}m for undo", move.label, move.actual_pick_z)

        # The pick episode ends once the object is lifted clear of the stack.
        self._mover.lift()
        self._rec_end(True)

        # The divert decision only needs the pick result, so it is made BEFORE
        # the transport: that way the transport frames are recorded under the
        # episode they actually belong to (place vs hand_off).
        code = result.barcode
        if scan_this:
            _publish_barcode_read(code, bool(code) and code in cfg.TARGET_BARCODES, move.label)
            if code is not None:
                self._rec_barcode(True)  # destination info now known for the place
        # Divert to the right gripper when the barcode matches a target, OR when
        # the scan gate's spiral search ran and exhausted with no read at all (an
        # unreadable battery is treated like a target and quarantined). A no-read
        # without the gate (e.g. gate disabled) places normally, as before.
        divert = scan_this and (code in cfg.TARGET_BARCODES or (result.scan_gated and code is None))
        self._rec_begin("hand_off" if divert else f"{move.label}_place")

        # Horizontal travel to the transport pose — shared by both branches.
        self._mover.move_to(move.dst)

        if divert:
            if code is None:
                logger.warning("[{}] no barcode read after search — diverting to right gripper", move.label)
            else:
                logger.info("[{}] barcode {!r} matches target — diverting to right gripper", move.label, code)
            if self._handoff_to_gripper():
                # Diverted batteries leave the case workflow, so they are not
                # recorded for undo (undo skips them).
                logger.info("[{}] done (placed lower-right via gripper)", move.label)
                return True
            logger.warning("[{}] gripper handoff failed — falling back to normal place", move.label)
            self._rec_end(False)  # no-op if the handoff already closed its episode
            self._rec_begin(f"{move.label}_place")
        elif code is not None:
            logger.info("[{}] barcode {!r} is not a target — normal place", move.label, code)

        self._mover.place(move.dst)
        self._rec_end(True)
        self._done.append(move)
        logger.info("[{}] done", move.label)
        return True

    def _handoff_to_gripper(self) -> bool:
        """Grip the suction-held battery, release suction, execute the place sequence.

        1. Right gripper grips the battery at the transport pose.
        2. Suction off (grip confirmed).
        3. Execute the EE pose sequence from taught_ee_poses_right.txt (REL support).
           - After the first step, suction arm lifts clear in a background thread.
           - At the "Lower" step, gripper partial-opens to release the battery.
        Returns False if the grip check fails (caller falls back to suction place).
        """
        suction_pos, _ = self._mover.current_ee_pose()
        grasp_pos = np.asarray(suction_pos, dtype=float) + np.asarray(cfg.HANDOFF_GRIP_OFFSET, dtype=float)
        if not self._gripper.grip_at(grasp_pos):
            return False

        suction_io.suction_off()
        time.sleep(0.3)
        # Battery is now in the gripper — the hand_off episode is complete.
        self._rec_end(True)

        steps = _parse_ee_sequence(_EE_PLACE_SEQ_PATH)
        if not steps:
            logger.error("[handoff] no poses in {} — cannot place", _EE_PLACE_SEQ_PATH)
            self._gripper.gripper.open()
            return False

        dur = cfg.EE_PLACE_STEP_DURATION_S
        suction_retract_done = threading.Event()

        try:
            default_poses = _parse_joint_lines(DEFAULT_POSE_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[handoff] could not read default pose: {}", exc)
            default_poses = None

        def _retract_suction():
            # The suction arm is already at SAFE_TRANSPORT_Z (lifted before the
            # handoff), so the lift is just a clearance no-op; the actual
            # "move back" is returning the left/suction arm to its default pose
            # so it clears the gripper instead of hovering at the handoff point.
            try:
                self._mover.lift(cfg.SAFE_TRANSPORT_Z)
                if default_poses is not None:
                    self._mover.move_arm_joints(default_poses[0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("[handoff] suction retract failed: {}", exc)
            finally:
                suction_retract_done.set()

        self._rec_begin("gripper_battery_handling")
        # REL steps build on the *commanded* pose, not the live FK readout:
        # near roll=±180°/pitch=±90° the Euler readout can jump representation
        # (wraparound / gimbal lock), and adding a delta to the flipped triple
        # commands a wrist ~180° off. Deltas compose in rotation space for the
        # same reason — Euler components don't add.
        cmd_pos, cmd_rpy = self._gripper.current_ee_pose()
        for i, (label, vec, rpy_val, is_relative) in enumerate(steps):
            if is_relative:
                pos = cmd_pos + vec
                rpy = (Rotation.from_euler("xyz", rpy_val)
                       * Rotation.from_euler("xyz", cmd_rpy)).as_euler("xyz")
                logger.info("[handoff] {} (REL) dpos={} drpy_deg={}", label,
                            np.round(vec, 4).tolist(), np.round(np.degrees(rpy_val), 1).tolist())
            else:
                pos, rpy = vec, rpy_val
                logger.info("[handoff] {}", label)
            cmd_pos, cmd_rpy = pos, rpy

            self._gripper._move_ee_to(pos, rpy, dur)
            self._gripper._wait_until_arrived(pos, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)

            # After the first step, lift the suction arm clear concurrently.
            if i == 0:
                threading.Thread(target=_retract_suction, name="suction-retract", daemon=True).start()

            if "lower" in label.lower():
                logger.info("[handoff] at Lower — partial open")
                self._gripper.gripper.partial_open()

        suction_retract_done.wait(timeout=10.0)

        logger.info("[handoff] returning right arm to default pose")
        try:
            if default_poses is not None:
                self._gripper.move_arm_joints(default_poses[1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[handoff] could not return right arm to default: {}", exc)

        self._rec_end(True)
        return True

    def _execute_with_retry(self, move: Move) -> bool:
        """Run a move, retrying on failure until it succeeds.

        A move "fails" only when its pick fails — one of force_limit (cup hit a
        hard force), max_descent (cup never touched the part) or vacuum_timeout
        (touched but the suction never sealed). place() never reports failure.

        Retrying is gated by ``cfg.RETRY_FAILED_PHASE`` (read live, since
        ``_execute`` reloads config each attempt). The sequence stops only on a
        software E-Stop (returns False) or Ctrl-C (KeyboardInterrupt propagates
        out of the sleep / motion). Each retry re-runs the full pick, which lifts
        to transport z first, so it re-approaches from above rather than pressing
        in place. Failed attempts are still recorded as failed episodes.
        """
        attempt = 0
        while True:
            if self._mover.software_estop_active():
                logger.error("[{}] software E-Stop active — stopping (no retry).", move.label)
                return False
            attempt += 1
            if self._execute(move):
                if attempt > 1:
                    logger.success("[{}] recovered on attempt {}.", move.label, attempt)
                return True
            if not getattr(cfg, "RETRY_FAILED_PHASE", True):
                return False  # retry disabled — propagate the failure (old behaviour)
            delay = float(getattr(cfg, "PHASE_RETRY_DELAY_S", 2.0))
            logger.warning(
                "[{}] failed (attempt {}) — retrying in {:.0f}s. "
                "Ctrl-C or E-Stop to stop.", move.label, attempt, delay,
            )
            time.sleep(delay)

    def run_forward(self, repeat: int = 0) -> bool:
        _publish_target_height(repeat)
        for move in build_forward_moves(repeat):
            if not self._execute_with_retry(move):
                return False
        logger.info("Forward sequence complete (repeat={}).", repeat)
        return True

    def run_undo(self) -> bool:
        """Undo every successfully executed move, most-recent first.

        If no moves have been recorded in this process (e.g. when invoked via
        ``--undo-only`` without a prior forward run), fall back to the taught
        poses and walk the full forward stack in reverse: for each repeat
        index from ``FORWARD_REPEATS - 1`` down to ``0`` we take the reversed
        forward moves for that pass. That way each undo place lands at
        ``src + k * src_dz`` (and pick comes from ``dst + k * dst_dz``)
        instead of releasing every item at the bare taught src z, which
        would otherwise be too high after several stacking repeats.
        """
        if self._done:
            moves_to_undo = [m.reversed() for m in reversed(self._done)]
        else:
            repeats = max(1, int(getattr(cfg, "FORWARD_REPEATS", 1)))
            logger.info(
                "No recorded moves — undoing from taught poses across {} repeat(s).",
                repeats,
            )
            moves_to_undo = []
            for k in range(repeats - 1, -1, -1):
                for m in reversed(build_forward_moves(repeat=k)):
                    moves_to_undo.append(m.reversed())
        for move in moves_to_undo:
            if not self._execute_with_retry(move):
                return False
        # Clear so a follow-up forward run starts with a fresh history (used by
        # --loop to alternate forward/undo repeatedly).
        self._done.clear()
        logger.info("Undo complete.")
        return True
