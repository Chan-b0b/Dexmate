"""Place-redesign constants: OVERRIDES win over ik_demo.config, everything else
falls through to it. Used only by place_mover.PlaceMover and __main__.py, so
chassis_sequence / suction.py keep reading config untouched.

Add a value to try it, e.g.  OVERRIDES = {"CASE_CORNER_AIM_BIAS_M": 0.03}
A name that does not exist in config is rejected (typo guard); to introduce a
brand-new constant, add it to config first or read it from this module directly.

Not covered: the INHERITED suction methods (pick, _descend_to_contact, force
reads) read config directly — copy them into place_mover.py to override those.
"""
from __future__ import annotations

from loguru import logger

from .. import config as _base

OVERRIDES: dict = {
    "TORSO_JOINTS": (1.2, 2.70, 0.35),  # deg [55, 140, 25]
    # Table raised: measured floor 0.60 -> 0.67 (FLOOR_Z_BASE_M in config). The
    # values below are the old ones +0.07, except the rim = floor + wall height.
    "SOURCE_CASE_CENTER": (0.72, 0.0, 0.87, 0.0),   # z_ee was 0.80; x was 0.88: unreachable with this torso (pre-check err 16-29mm, auto-adjust vs centering loop)
    "DIVERT_BIN_PLANE_Z_M": 0.6838,                 # was 0.6138
    "BOX_RIM_Z_M": 0.805,                           # was 0.65 (0.67 + BOX_WALL_HEIGHT_M 0.135)
    "SAFE_TRANSPORT_Z": 1.17,                       # was 1.1
    "LIFT_CLEAR_EE_Z": 1.12,                        # was 1.05
}


# pick_place-only (not a config constant): after the cup seals on a corner-seated
# case, slide it this far AWAY from the corner walls (CASE_CORNER_DIR) before the
# vertical lift, so the case does not drag the bin up/toward the robot. 0 = off.
# 1012: off — no corner-seated cases any more (places go straight down onto the
# detection), so there is no wall to slide off. Was 0.008.
PICK_WALL_RELIEF_M: float = 0.0


# pick_place-only: the straight per-tick vertical lift is kept only up to this
# EE z (a held case's bottom must clear the bin rim, 0.805, with margin); above
# it the rest of the rise to LIFT_CLEAR_EE_Z is one planned (jerk-limited) move.
# The per-tick stream at cruise speed lags the command (pitch error -10 deg, cup
# tip 80mm off at z=1.10, 0.85-1.10 lift traces), the planner does not.
# >= LIFT_CLEAR_EE_Z keeps the old all-vertical lift.
LIFT_VERTICAL_TO_Z: float = 1.0


# pick_place-only: --obb-depth (1012). The plane height is read from the depth
# pixels inside the detected OBB: their base z values are histogrammed in
# OBB_DEPTH_BIN_M bins, the fullest bin is the plane, and the height is the
# median of the pixels within OBB_DEPTH_TOL_M of it (a cluster, not the median of
# the whole box — the case is transparent, so the box holds a lot of depth from
# below / through it). Fewer than OBB_DEPTH_MIN_PX pixels in the cluster, or a
# cluster under OBB_DEPTH_MIN_FRAC of the valid pixels, = no plane.
OBB_DEPTH_BIN_M: float = 0.002
OBB_DEPTH_TOL_M: float = 0.005
OBB_DEPTH_MIN_PX: int = 300
OBB_DEPTH_MIN_FRAC: float = 0.15
# 1012: the plane comes from the BATTERY TOPS, not the case. The case is
# transparent and its face read anywhere in a ~15mm range frame to frame (the
# case-border band too: the far wall's inner face is in it); the batteries are
# opaque and read 0.711-0.713 in all 7 saved frames. Only the inner box (the OBB
# scaled by OBB_DEPTH_INNER_FRAC about its centre = the battery slots) is read,
# and the HIGHEST cluster holding >= OBB_DEPTH_MIN_FRAC of it is the battery top.
# The batteries seen are always the LOWER case's (an empty case is picked off a
# battery case; a case is placed onto one), so:
#   face of the case holding them = battery top - BATTERY_TOP_ABOVE_FACE_M
#   face of an empty case nested on it = that + one LAYER_PITCH_M
# BATTERY_TOP_ABOVE_FACE_M: 4 target frames 15.6-17.1mm; the nested source
# then predicts 2.2mm below the battery top vs 2.4-3.8 measured (cup contact).
OBB_DEPTH_INNER_FRAC: float = 0.7
BATTERY_TOP_ABOVE_FACE_M: float = 0.016
# The re-detect on the measured plane may only MOVE the same case: a plane
# change shifts xy along the camera ray by ~dz * (ground distance / camera
# height) — tens of mm at most. A jump past this is the detector picking a
# different box (its argmax conf can flip with the warp; seen offline on a
# 0903 frame: 0.45m) -> treated as a failed read.
OBB_DEPTH_MAX_SHIFT_M: float = 0.08


# pick_place-only (1012 experiment): the place does NOT descend to a force
# contact — it stops this far ABOVE the expected contact ee z (depth plane +
# SUCTION_LENGTH_M) and releases there, no prelift. A force contact before that
# height is held for the operator ("early_contact"). None = the force-stop place.
PLACE_RELEASE_ABOVE_M: "float | None" = None
# hold at the release height this long (the arm trails the stream) before releasing
PLACE_RELEASE_SETTLE_S: float = 0.3


# pick_place-only: the detected case centre (base x, y) must land inside this
# window before the pick / place; outside it the chassis drives the case to the
# window's CENTRE (forward/back for x, strafe for y), re-detects, and repeats up
# to WINDOW_MAX_MOVES times before handing the keyboard to the operator.
# (centre, half-width) in m. None = no window.
WINDOW_X_M: "tuple[float, float] | None" = (0.68, 0.01)
WINDOW_Y_M: "tuple[float, float] | None" = (0.0, 0.03)
WINDOW_MAX_MOVES: int = 4


class _Cfg:
    def __getattr__(self, name):
        if name in OVERRIDES:
            return OVERRIDES[name]
        return getattr(_base, name)


cfg = _Cfg()

for _k, _v in OVERRIDES.items():
    if not hasattr(_base, _k):
        raise AttributeError(f"place_config.OVERRIDES: {_k!r} is not a config constant")
    logger.warning("place_config override: {} = {!r} (config: {!r})", _k, _v, getattr(_base, _k))
