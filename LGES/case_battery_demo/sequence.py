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
from dataclasses import dataclass, field

from loguru import logger

from . import config as cfg
from .grasp import Pose, SuctionMover


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
    def __init__(self, mover: SuctionMover) -> None:
        self._mover = mover
        self._done: list[Move] = []  # successfully executed moves, for undo

    def _execute(self, move: Move) -> bool:
        # Reload config so edits made between moves (during --loop, or while
        # the operator is tuning thresholds in another editor) take effect on
        # the next move. Both this module and grasp.py hold the same module
        # object via `from . import config as cfg`, so importlib.reload
        # updates the constants in place for both.
        importlib.reload(cfg)
        logger.info("=== Move: {} ===", move.label)
        result = self._mover.pick(move.src)
        if not result.success:
            logger.error("[{}] pick failed (trigger={}) — aborting", move.label, result.trigger)
            return False
        # Remember the sealed z so a future undo of this same move places
        # the battery back at exactly the height it was picked from.
        if result.contact_position_base is not None:
            move.actual_pick_z = float(result.contact_position_base[2])
            logger.info("[{}] recorded pick z={:.4f}m for undo", move.label, move.actual_pick_z)
        self._mover.lift()
        self._mover.move_to(move.dst)
        self._mover.place(move.dst)
        self._done.append(move)
        logger.info("[{}] done", move.label)
        return True

    def run_forward(self, repeat: int = 0) -> bool:
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
