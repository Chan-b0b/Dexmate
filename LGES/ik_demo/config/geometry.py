"""Base-frame geometry: transport heights, the taught case/battery targets,
and the stances everything is reachable from.

resolve_poses() lives here — it is what turns ONE case center into every
pick/place pose — along with HOME_JOINTS_*, TORSO_JOINTS and the torso /
robot-connection settings that go with holding that stance.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

import numpy as np

from .robot import GRASP_ORIENTATION_RPY

# ---------------------------------------------------------------------------
# Transport geometry
# ---------------------------------------------------------------------------
# Base-frame Z the cup tip is raised to for collision-free sideways transport;
# must clear the source stack and both box walls. (measured 2026-06-09)
SAFE_TRANSPORT_Z: float = 1.1
# Absolute EE z the straight-up (xy-held, move_ee_vertical) lift leg ends at —
# clear of the case walls from any layer's pick, so the held part is never
# dragged sideways while still between them. Above it the ascent is a faster
# joint-space move_ee (its sideways arc is harmless once clear), and on the
# lift_to_clear paths the remaining rise to SAFE_TRANSPORT_Z overlaps the
# chassis leg (see chassis_sequence._park_during_legs / _view_park).
# Reference formula at today's constants: FLOOR_Z_BASE 0.6 + 2*LAYER_PITCH
# 0.0138 + SUCTION_LENGTH_M 0.155 + 0.15 clearance ~= 0.93.
# Had been raised to 1.1 = SAFE_TRANSPORT_Z, which silently disabled the split:
# _lift_to_transport's `z_clear >= SAFE_TRANSPORT_Z` then returned early every
# time, so the WHOLE ~300mm lift ran as the per-tick vertical stream at
# DESCENT_APPROACH_SPEED_M_S. 0903 measured the cost — commanded vertical
# (ik 0.0mm) but the arm tracking 74mm off line and 22.6 deg in pitch, i.e. a
# ~130mm cup-tip excursion while carrying the part.
LIFT_CLEAR_EE_Z: float = 1.05
# Best-effort ascent after a lift leg fell short (the per-tick vertical stream
# can dead-end on a diverging IK branch right at the reach boundary while the
# column is statically reachable): the joint-space recovery move only runs when
# the EE is already within this of LIFT_CLEAR_EE_Z (or above) — lower means the
# part is still between the case walls, where a joint-space arc could sweep
# them — and only moves for a height gain of at least the MIN_GAIN.
LIFT_RECOVER_MIN_CLEAR_M: float = 0.03
LIFT_RECOVER_MIN_GAIN_M: float = 0.01
# Height the cup tip hovers above a target before the descent leg (m).
HOVER_HEIGHT_M: float = 0.25
# Physical suction tube length, L_gripper_base origin -> cup tip (m).
# INFORMATIONAL ONLY — NOT added to taught targets: the taught poses were
# recorded as the L_gripper_base (EE) pose, the same frame the IK solves for,
# so they are used directly (adding this would double-count; see arm.taught_target).
# The EE->cup-tip offset, measured from the real grab: median of 230 SOURCE
# case pick contacts (ztrack_logs 08/10-08/27, contact_ee_z minus the model
# plane FLOOR_Z_BASE_M + layers*LAYER_PITCH_M; median 0.1554, stdev 2.8mm).
# NOTE this is really (cup length + floor-model error) — exactly the constant
# every consumer needs vs the model plane. Previous 0.176 (2026-07-02, taught
# EE z 0.81 minus 5-layer top face 0.6341) predates the cup change and the
# FLOOR_Z_BASE_M raise. Target PLACE contacts run ~21mm lower (0.1343 median,
# case nesting/press compression) — anchored predictions cover those.
# Used by the detection pipeline to turn a detected top-face z into an EE target;
# confirm/refine against sequence.py's contact_ee_z.
SUCTION_LENGTH_M: float = 0.155

# ---------------------------------------------------------------------------
# Target geometry (base_link, m / rad) — single source center + displacement.
#
# Everything hangs off ONE source case center (the top layer; detection can
# override it). In the case-local frame (x-fwd, y-left):
#   - the cup grabs the case slightly left of center      -> CASE_GRAB_OFFSET
#   - the two battery slots are fixed, symmetric           -> SLOT_OFFSETS
# The case and BOTH batteries then move to the target by the SAME displacement:
#       target = source + DISPLACEMENT
# so we only define the source; the targets are "added up".
#
# z is APPROXIMATE here — descend-to-contact measures the real grab/seat height,
# and because we always approach from SAFE_TRANSPORT_Z and descend until contact,
# a STACK of layers is handled automatically (the descent finds whatever's on
# top). The measured contact z per layer is what tells us the layer pitch.
# ---------------------------------------------------------------------------
# DEFAULT only — every chassis_sequence pick/place re-resolves from the detected
# center. This is the OPERATING geometry (0903): the chassis centering puts the
# case on the chassis center line (CHASSIS_CENTER_CASE_Y_M = 0; detected centers
# that day ranged -0.03..+0.16) at yaw 0; x is the auto-adjust reference
# (_auto_adjust reads [0]; detections settle at 0.86-0.87); z is the measured
# top-of-stack EE contact at the run-start stack (SRC_LAYERS_REMAINING = 4):
# median contact_ee_z 0.8015 over 10 source case picks (model FLOOR_Z_BASE_M +
# 4 * LAYER_PITCH_M + SUCTION_LENGTH_M = 0.810), +LAYER_PITCH_M per extra layer.
# The previous (0.87, 0.455, 0.964) was the pre-chassis fixed-station staging
# (case 45 cm to the left), out of reach at the current torso stance.
SOURCE_CASE_CENTER: tuple[float, float, float, float] = (0.9, 0.0, 0.80, 0.0)  # x, y, z_ee, yaw
GRASP_YAW: float = 3.1415                       # cup approach yaw, relative to the case frame
HALF_SLOT_SPACING_M: float = 0.08           # slots at (0, ±this) around the slot-pair center
# Case-local (dx, dy, dz) offsets from the case center.
CASE_GRAB_OFFSET: tuple[float, float, float] = (-0.03, 0.05, 0.0)   # cup grabs slightly left (+y)
# Slot-pair CENTER vs the detected OBB center (case-local). Both battery targets
# shift by this together — fixes the common-bias symptom (bat1 undershoots,
# bat2 overshoots by the same amount = center biased +y). Spacing stays symmetric.
SLOT_CENTER_OFFSET: tuple[float, float, float] = (-0.04, -0.10, 0.0)
SLOT_OFFSETS: dict[int, tuple[float, float, float]] = {
    1: (SLOT_CENTER_OFFSET[0], -0.03 - HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # right slot (robot -y)
    2: (SLOT_CENTER_OFFSET[0], -0.03 + HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # left slot  (robot +y)
}
# Displacement source -> target, same for the case and both batteries (base_link).
# In the chassis flow the target case is centered exactly like the source, so
# xy is 0 and only the stack height differs: (TGT_LAYERS_REMAINING -
# SRC_LAYERS_REMAINING) * LAYER_PITCH_M = -3 * 0.0138 at the run-start stacks.
# z is approximate (descent measures the real seat height). The old
# (0, -0.4135, -0.068) was the fixed-station target box 41 cm to the right,
# which only the legacy sequence.py flow used.
DISPLACEMENT: tuple[float, float, float] = (0.0, 0.0, -0.041)  # dx, dy, dz


def resolve_poses(source_center: tuple | None = None) -> dict[str, tuple]:
    """All targets in base_link, from a single source case center. Sources are
    case-local offsets rotated by the center yaw + added; each target is its
    source + DISPLACEMENT. Pass a detected source center to override the default
    (everything follows it). z is approximate — descend-to-contact finds the real
    height, so a layer stack is handled by the descent, not by these numbers."""
    sx, sy, sz, syaw = source_center if source_center is not None else SOURCE_CASE_CENTER
    c, s = float(np.cos(syaw)), float(np.sin(syaw))
    roll, pitch = GRASP_ORIENTATION_RPY[0], GRASP_ORIENTATION_RPY[1]
    yaw = syaw + GRASP_YAW
    dxg, dyg, dzg = DISPLACEMENT

    def src(off: tuple[float, float, float]) -> tuple:
        dx, dy, dz = off
        return (sx + c * dx - s * dy, sy + s * dx + c * dy, sz + dz, roll, pitch, yaw)

    def to_target(p: tuple) -> tuple:
        return (p[0] + dxg, p[1] + dyg, p[2] + dzg, roll, pitch, yaw)

    case_pick = src(CASE_GRAB_OFFSET)
    b1, b2 = src(SLOT_OFFSETS[1]), src(SLOT_OFFSETS[2])
    return {
        "CASE_PICK": case_pick,   "CASE_PLACE_R": to_target(case_pick),
        "BAT_SRC_1": b1,          "BAT_SLOT_1":   to_target(b1),
        "BAT_SRC_2": b2,          "BAT_SLOT_2":   to_target(b2),
    }


# Default resolved targets (case at CASE_CENTER). Consumers use this unchanged;
# detection would call resolve_poses(detected_center) instead.
TAUGHT_POSES: dict[str, tuple[float, ...]] = resolve_poses()

# Home / default arm joint configs (from default_pose.txt). Used as the IK seed
# (differential IK converges reliably from a near-workspace seed; a zero seed
# stalls short) and as the stance the demo returns to. The demo runs with the
# torso at TORSO_JOINTS.
# j7 was -1.3780 = the URDF hard stop exactly, OUTSIDE the IK band (-1.278 at
# JOINT_RANGE_FRAC 0.92): arm._configuration clipped it, so FK of the home /
# view-park stance read ~20 mm off and every leg seeded from it started from a
# wrong pose estimate. Moved inside the band (1.6 deg margin; EE shift ~2 mm,
# cup tip ~2 cm) — RE-CHECK the view-park camera clearance on the robot.
HOME_JOINTS_LEFT: tuple[float, ...] = (-2.2555, 1.3993, 2.5, -2.1348, -0.2685, 0.9856, -1.25)
HOME_JOINTS_RIGHT: tuple[float, ...] = (-1.0066, -0.6759, 0.2385, -2.2329, 0.8087, 1.0881, 0.0507)

# Torso joint angles (rad) the demo/teaching runs at. Taught base_link poses are
# only reachable at this torso pose (torso moves the arm base). arm.py reads the
# live torso when a robot is attached; this is the headless / validation value.
TORSO_JOINTS: tuple[float, float, float] = (0.52359878, 1.91986218, 0.26179939)  # deg [30, 110, 15]

# Torso motion speed, as a fraction of the hardware ceiling, for every torso
# command the demo sends (arm.pin_torso). The torso goes through the robot's
# internal motion plugin (trajectory smoothing + gravity comp), NOT the raw
# set_joint_pos position channel it used until 0904 — that channel publishes a
# step target with no trajectory generation, so a stance CHANGE (as opposed to
# the few mm of drift correction pin_torso was written for) lurched. 0.2 is
# gentler than the dexcontrol example default of 0.3.
TORSO_VEL_SCALE: float = 0.2

# Robot() construction retries. Its last step (_set_default_state) reads the
# TORSO state to compensate the head home pose, and that read intermittently
# lands before the torso's state subscriber has a parsed sample — the
# constructor raises from inside dexcontrol. It is a startup race (it comes and
# goes; a fresh connect works), so the whole construction is retried rather
# than the run dying at the door. See arm.connect_robot.
ROBOT_CONNECT_ATTEMPTS: int = 3
ROBOT_CONNECT_DELAY_S: float = 3.0

# Head pitch (deg, the set_head_pitch convention) that Robot() construction
# should leave the head at. The same _set_default_state noted above sends the
# head to its config "home" pose, which for vega is [0,0,0] — the HORIZON once
# compensate_torso_pitch adds torso_pitch - pi/2. Every demo then immediately
# calls set_head_pitch(24) to look at the box, 66 deg further down, so the head
# visibly rises at startup and then comes back down. connect_robot overrides
# that "home" entry (in a config object built per connect, so nothing global is
# mutated) to land here in ONE move instead.
#
# The conversion lives in connect_robot: set_head_pitch drives head_j1 to
# torso_pitch - deg2rad(this), while init drives it to home[0] + torso_pitch
# - pi/2, so home[0] = deg2rad(90 - this) makes the two identical. Both read
# the LIVE torso pitch at their own call time, so a torso move in between is
# handled by whichever runs later.
#
# 24 = the BEV detection angle chassis_sequence and case_to_bin set right after
# connecting, so for them the later set_head_pitch becomes a no-op. A tool that
# wants another angle (reach_in_place's 15) still calls set_head_pitch itself
# and works unchanged. None disables the override entirely.
HEAD_INIT_PITCH_DEG: "float | None" = 24.0
