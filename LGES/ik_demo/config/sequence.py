"""Layer loop, stack geometry and per-phase failure recovery (sequence.py).

Also holds FLOOR_Z_BASE_M / LAYER_PITCH_M: the stack heights every warp
plane and expected contact z is derived from, which is what the layer
loop iterates over.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Task sequence / failure recovery (sequence.py)
# ---------------------------------------------------------------------------
# Layers to build. Each layer runs the full choreography (case + 2 batteries).
# XY repeats every layer; the descent finds the real z per layer, and the
# orchestrator feeds each pose's measured contact z forward so the creep zone
# tracks the growing target / shrinking source stacks (see TaskOrchestrator).
NUM_LAYERS: int = 5
# Per-layer stack pitch (m), induced from measured contacts: sources drop / targets
# rise by this each layer. Each column is anchored on its clean layer-1 contact and
# stepped by this constant, so a single misaligned seat can't corrupt later layers'
# predictions (it's flagged instead — see TaskOrchestrator._record_z).
LAYER_PITCH_M: float = 0.0138
# Empty target-box floor height (base_link z, no layers). Mirrors case_detection's
# measured FLOOR_Z_BASE_M (case_detection/config.py) — that copy is the source of
# truth (measured by measure_floor_z.py); keep this one in sync with it by hand.
FLOOR_Z_BASE_M: float = 0.60

# A measured seat deviating from the predicted z by more than this fraction of the
# pitch is logged as a probable misalignment.
LAYER_MISALIGN_FRAC: float = 0.5
# Chained seat expectations (ZTracker.expected_ee_z). Instead of extrapolating
# each column's own first contact by the nominal LAYER_PITCH_M — which drifts,
# since 0903 measured the real target pitch at ~17mm against a 13.8mm config —
# predict each item from the thing it PHYSICALLY RESTS ON, measured in this run:
#   target battery = THIS layer's case seat        + BATTERY_SEAT_ABOVE_CASE_M
#   target case    = PREVIOUS layer's top battery  + CASE_SEAT_ABOVE_BATTERY_M
#   source battery = THIS layer's source case anchor (no offset — 0903 measured
#                    case/battery source contacts within 1.6mm at layer 4:
#                    0.8018 / 0.8009 / 0.8002)
# Measured spreads: battery-above-case +7.7..+16.4mm over 8 pairs (median ~11),
# case-above-previous-battery +2.0..+8.0mm over 3 pairs (~5). Both values here
# are set ABOVE those medians on purpose: an expectation that lands HIGH only
# lengthens the creep, while one that lands LOW puts the planned approach into
# the part before the creep even starts (0903 measured -9.8, -11.1, -13.3mm of
# "creep" on battery picks, i.e. contact ABOVE the creep line).
BATTERY_SEAT_ABOVE_CASE_M: float = 0.015
CASE_SEAT_ABOVE_BATTERY_M: float = 0.008
# Per-run CSV of the ZTracker events (contact z vs predicted, warp-plane
# measured vs model, misseat/force_limit failures) — the layer-by-layer error
# data separated from the run log. None disables.
ZTRACK_LOG_DIR: str | None = "/home/dexmate/LGES/Dexmate/LGES/ik_demo/ztrack_logs"
# Per-run TEXT log (the full logger stream, paired with the CSV above by its
# timestamp). loguru's default sink is stderr ONLY, so before this every
# warning and every DESCENT_TRACE_S trace line lived and died in the terminal —
# a finished run left nothing to read back. None disables.
RUN_LOG_DIR: str | None = "/home/dexmate/LGES/Dexmate/LGES/ik_demo/run_logs"
RETRY_FAILED_PHASE: bool = True     # retry a failed pick/place instead of aborting
MAX_PHASE_ATTEMPTS: int = 3         # attempts per move before giving up
PHASE_RETRY_DELAY_S: float = 2.0    # pause before a retry (also lets config edits land)
