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


def build_forward_moves(repeat: int = 0) -> list[Move]:
    """Build the three forward moves, offset in Z for stacking.

    ``repeat`` is the 0-based pass index. Each move's src and dst Z are
    shifted by ``repeat * cfg.Z_STEP_PER_REPEAT[label]`` so the source stack
    shrinks and the target stack grows across repeats. Pose objects are
    freshly constructed here, so mutating them is safe.
    """
    moves = [
        Move("case", _pose("CASE_PICK"), _pose("CASE_PLACE_R")),
        Move("battery_1", _pose("BAT_SRC_1"), _pose("BAT_SLOT_1")),
        Move("battery_2", _pose("BAT_SRC_2"), _pose("BAT_SLOT_2")),
    ]
    if repeat:
        for m in moves:
            src_dz, dst_dz = cfg.Z_STEP_PER_REPEAT.get(m.label, (0.0, 0.0))
            m.src.pos[2] += repeat * src_dz
            m.dst.pos[2] += repeat * dst_dz
    return moves


class TaskOrchestrator:
    def __init__(self, mover: SuctionMover, gripper: GripperMover | None = None) -> None:
        self._mover = mover
        self._gripper = gripper
        self._done: list[Move] = []  # successfully executed moves, for undo

    def _execute(self, move: Move) -> bool:
        # Reload config so edits made between moves (during --loop, or while
        # the operator is tuning thresholds in another editor) take effect on
        # the next move. Both this module and grasp.py hold the same module
        # object via `from . import config as cfg`, so importlib.reload
        # updates the constants in place for both.
        importlib.reload(cfg)
        logger.info("=== Move: {} ===", move.label)

        # Forward battery picks get barcode-scanned during the descent; the
        # case move and undo moves never scan. The scanner runs in a daemon
        # thread so it can't stall the 50 ms descent loop.
        scan_this = move.label.startswith("battery") and self._gripper is not None
        scanner = bcr.BackgroundScanner() if scan_this else None

        # Scanner starts only when the suction EE enters BCR_SCAN_Z_THRESHOLD_M
        # above the target — not at hover — so reads are taken in the final
        # centimetres where the barcode is closest to the camera.
        result = self._mover.pick(
            move.src,
            near_target_callback=scanner.start if scanner is not None else None,
        )

        if scanner is not None:
            scanner.stop()

        if not result.success:
            logger.error("[{}] pick failed (trigger={}) — aborting", move.label, result.trigger)
            return False
        # Remember the sealed z so a future undo of this same move places
        # the battery back at exactly the height it was picked from.
        if result.contact_position_base is not None:
            move.actual_pick_z = float(result.contact_position_base[2])
            logger.info("[{}] recorded pick z={:.4f}m for undo", move.label, move.actual_pick_z)

        # Normal lift + horizontal travel to the transport pose — shared by
        # both the suction-place and gripper-handoff branches.
        self._mover.lift()
        self._mover.move_to(move.dst)

        code = scanner.result() if scanner is not None else None
        if scanner is not None:
            _publish_barcode_read(code, bool(code) and code in cfg.TARGET_BARCODES, move.label)
        if code is not None and code in cfg.TARGET_BARCODES:
            logger.info("[{}] barcode {!r} matches target — diverting to right gripper", move.label, code)
            if self._handoff_to_gripper():
                # Diverted batteries leave the case workflow, so they are not
                # recorded for undo (undo skips them).
                logger.info("[{}] done (placed lower-right via gripper)", move.label)
                return True
            logger.warning("[{}] gripper handoff failed — falling back to normal place", move.label)
        elif code is not None:
            logger.info("[{}] barcode {!r} is not a target — normal place", move.label, code)

        self._mover.place(move.dst)
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

        steps = _parse_ee_sequence(_EE_PLACE_SEQ_PATH)
        if not steps:
            logger.error("[handoff] no poses in {} — cannot place", _EE_PLACE_SEQ_PATH)
            self._gripper.gripper.open()
            return False

        dur = cfg.EE_PLACE_STEP_DURATION_S
        suction_retract_done = threading.Event()

        def _retract_suction():
            self._mover.lift(cfg.SAFE_TRANSPORT_Z)
            suction_retract_done.set()

        for i, (label, vec, rpy_val, is_relative) in enumerate(steps):
            if is_relative:
                cur_pos, cur_rpy = self._gripper.current_ee_pose()
                pos = cur_pos + vec
                rpy = cur_rpy + rpy_val
                logger.info("[handoff] {} (REL) dpos={} drpy_deg={}", label,
                            np.round(vec, 4).tolist(), np.round(np.degrees(rpy_val), 1).tolist())
            else:
                pos, rpy = vec, rpy_val
                logger.info("[handoff] {}", label)

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
            poses = _parse_joint_lines(DEFAULT_POSE_PATH)
            self._gripper.move_arm_joints(poses[1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[handoff] could not return right arm to default: {}", exc)

        return True

    def run_forward(self, repeat: int = 0) -> bool:
        _publish_target_height(repeat)
        for move in build_forward_moves(repeat):
            if not self._execute(move):
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
            if not self._execute(move):
                return False
        # Clear so a follow-up forward run starts with a fresh history (used by
        # --loop to alternate forward/undo repeatedly).
        self._done.clear()
        logger.info("Undo complete.")
        return True
