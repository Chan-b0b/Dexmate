"""Configuration for the ik_demo case + battery suction demo.

Single-arm suction pick-and-place: move an empty case left -> right, seat two
batteries into it; barcode-matched batteries are diverted to the right-hand
gripper. Forward only (no undo). See PLAN.md for the full design.

All poses are in the robot **base_link** frame, metres / radians. Orientation is
the straight-down suction approach (cup facing -Z).

This file is grown incrementally alongside the build order: it currently holds
what the drivers + arm.py (motion core) need. Feature-specific keys are added
when their module is built — descent/force in suction.py, barcode-search
geometry in barcode.py, handoff geometry in gripper.py.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Robot model / IK
# ---------------------------------------------------------------------------
URDF_PATH: str = (
    "/opt/venv/lib/python3.12/site-packages/dexmate_urdf/"
    "robots/humanoid/vega_1p/vega_1p_gripper.urdf"
)

# Differential-IK (pink) solve. Warm-started from the live/previous config, so a
# per-move solve converges in a few iterations; the 500 cap only bites on a
# cold solve (offline pose caching). 1e-3 (~1 mm / ~0.06 deg) is the practical
# converged target — a soft posture task adds a small steady-state bias, so a
# tighter tol never converges. Lower POSTURE_COST to tighten.
IK_DT: float = 0.01
IK_MAX_ITERS: int = 500
IK_CONVERGENCE_THRESHOLD: float = 1e-3
PREFERRED_QP_SOLVER: str = "daqp"
# Nullspace posture cost. Two phases use different targets (see PLAN.md):
# offline pose-solve pulls toward joint mid-ranges (curate away from limits);
# live move_ee pins to the start config (minimise motion, stay on one branch).
POSTURE_COST: float = 1e-3
# Levenberg-Marquardt damping on the EE task — stabilises solves near
# singularities (trades a little tracking error for a lot of stability).
IK_LM_DAMPING: float = 1e-6
# Reachability tolerance for MOTION (approach / descent): a move proceeds if the
# solved EE is within this of the target, even if the tight IK_CONVERGENCE_
# THRESHOLD (used for cache validation) isn't met. Poses near the reach ceiling
# leave a few mm residual that's harmless for approach/descent.
REACH_TOL_M: float = 0.008

# ---------------------------------------------------------------------------
# Arm with the suction end-effector
# ---------------------------------------------------------------------------
ARM_SIDE: str = "left"              # "left" or "right"
EE_FRAME: str = "L_gripper_base"    # URDF frame; R_gripper_base for the right arm

# Straight-down suction approach orientation (roll, pitch, yaw), radians.
# Default for 3-tuple taught poses; full 6-tuple poses carry their own rpy.
GRASP_ORIENTATION_RPY: tuple[float, float, float] = (np.pi, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Kinematic budget — the ONE place that sets speed.
#
# All motion is time-parameterised by Ruckig under these limits; a move's
# duration derives from its path length. "Make it faster" = raise SPEED_SCALE
# (or the individual caps). Ruckig keeps everything jerk-limited and feasible;
# arm.py additionally clamps joint commands to the arm's own reported limits.
# Streamed as (pos, vel) via arm.set_joint_pos_vel at ~100 Hz.
#
# Values are conservative starting points — TUNE on the robot.
# ---------------------------------------------------------------------------
SPEED_SCALE: float = 0.7            # global multiplier on every cap below

CONTROL_HZ: float = 100.0           # motion streaming rate (set_joint_pos_vel)

# Joint-space (move_joints: cached-pose -> cached-pose travel), per joint.
MAX_JOINT_VEL: float = 1.5          # rad/s
MAX_JOINT_ACCEL: float = 5.0        # rad/s^2
MAX_JOINT_JERK: float = 30.0        # rad/s^3

# Cartesian (move_ee: sensing legs), linear.
MAX_EE_LINEAR_VEL: float = 0.25     # m/s
MAX_EE_LINEAR_ACCEL: float = 1.0    # m/s^2
MAX_EE_LINEAR_JERK: float = 5.0     # m/s^3
# Cartesian, angular.
MAX_EE_ANGULAR_VEL: float = 1.0     # rad/s
MAX_EE_ANGULAR_ACCEL: float = 4.0   # rad/s^2
MAX_EE_ANGULAR_JERK: float = 20.0   # rad/s^3

# ---------------------------------------------------------------------------
# Transport geometry
# ---------------------------------------------------------------------------
# Base-frame Z the cup tip is raised to for collision-free sideways transport;
# must clear the source stack and both box walls. (measured 2026-06-09)
SAFE_TRANSPORT_Z: float = 1.12
# Absolute EE z the straight-up (xy-held, move_ee_vertical) lift leg ends at:
# 0.15 m above the layer-2 contact = FLOOR_Z_BASE 0.566 + 2*LAYER_PITCH 0.0138
# + SUCTION_LENGTH 0.176 + 0.15 ~= 0.92 — clear of the case walls from any
# layer's pick without dragging the held part sideways. The remaining ascent
# to SAFE_TRANSPORT_Z is a faster joint-space move_ee (its sideways arc is
# harmless once clear).
LIFT_CLEAR_EE_Z: float = 0.95
# Height the cup tip hovers above a target before the descent leg (m).
HOVER_HEIGHT_M: float = 0.25
# Physical suction tube length, L_gripper_base origin -> cup tip (m).
# INFORMATIONAL ONLY — NOT added to taught targets: the taught poses were
# recorded as the L_gripper_base (EE) pose, the same frame the IK solves for,
# so they are used directly (adding this would double-count; see arm.taught_target).
# The EE->cup-tip offset, measured from the real grab: taught top-of-stack EE
# z (0.81) minus the 5-layer top face z (0.6341, measure_floor_z, 2026-07-02).
# Used by the detection pipeline to turn a detected top-face z into an EE target;
# confirm/refine against sequence.py's contact_ee_z.
SUCTION_LENGTH_M: float = 0.176

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
SOURCE_CASE_CENTER: tuple[float, float, float, float] = (0.87, 0.454781, 0.81, 0.0)  # x,y,z,yaw (top layer)
GRASP_YAW: float = 1.92                       # cup approach yaw, relative to the case frame
HALF_SLOT_SPACING_M: float = 0.08           # slots at (0, ±this) around the slot-pair center
# Case-local (dx, dy, dz) offsets from the case center.
CASE_GRAB_OFFSET: tuple[float, float, float] = (0.0, 0.0564, 0.0)   # cup grabs slightly left (+y)
# Slot-pair CENTER vs the detected OBB center (case-local). Both battery targets
# shift by this together — fixes the common-bias symptom (bat1 undershoots,
# bat2 overshoots by the same amount = center biased +y). Spacing stays symmetric.
SLOT_CENTER_OFFSET: tuple[float, float, float] = (0.0, -0.04, 0.0)
SLOT_OFFSETS: dict[int, tuple[float, float, float]] = {
    1: (SLOT_CENTER_OFFSET[0], SLOT_CENTER_OFFSET[1] - HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # right slot (robot -y)
    2: (SLOT_CENTER_OFFSET[0], SLOT_CENTER_OFFSET[1] + HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # left slot  (robot +y)
}
# Displacement source -> target, same for the case and both batteries (base_link).
# z is approximate (descent measures the real seat height).
DISPLACEMENT: tuple[float, float, float] = (0.0, -0.413506, -0.0679)  # dx, dy, dz


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
HOME_JOINTS_LEFT: tuple[float, ...] = (-2.2555, 1.3993, 2.6261, -2.1348, -0.2685, 0.9856, -1.3780)
HOME_JOINTS_RIGHT: tuple[float, ...] = (-0.155076, -0.443368, 0.191845, -1.6711, -0.1628, 1.3960, -0.1480)
# Torso joint angles (rad) the demo/teaching runs at. Taught base_link poses are
# only reachable at this torso pose (torso moves the arm base). arm.py reads the
# live torso when a robot is attached; this is the headless / validation value.
TORSO_JOINTS: tuple[float, float, float] = (1.04719755, 2.7925268, 0.52359878)  # deg [60, 160, 30]

# ---------------------------------------------------------------------------
# Suction hardware (weblogic HTTP API + DI0 vacuum monitor)
# ---------------------------------------------------------------------------
SUCTION_HOST: str = "192.168.5.1"
SUCTION_BASE_URL: str = f"http://{SUCTION_HOST}/api/dc/weblogic"
SUCTION_ON_ID: int = 3587
SUCTION_OFF_ID: int = 763
BLOW_ON_ID: int = 963
BLOW_OFF_ID: int = 5089

# ---------------------------------------------------------------------------
# Suction descent / contact (suction.py) — detect-and-freeze descent.
# Two-signal pick: wrench vertical force = contact, DI0 vacuum = seal.
# ---------------------------------------------------------------------------
TARE_SAMPLES: int = 20                      # wrench baseline samples (no contact)
DESCENT_APPROACH_SPEED_M_S: float = 0.15    # fast free-air descent (cup-tip)
DESCENT_CREEP_SPEED_M_S: float = 0.03       # slow creep in the contact zone
DESCENT_RAMP_S: float = 0.3                 # ease descent speed in from 0 (no jerk
                                            # from the rest->descend handoff)
DESCENT_CREEP_BLEND_M: float = 0.03         # decelerate fast->creep smoothly over this
                                            # band ABOVE creep_z (no velocity step at
                                            # the creep line -> no jerk at creep height)
DESCENT_CREEP_GAP_M: float = 0.05           # creep starts this far above expected contact
DESCENT_MAX_M: float = 0.40                 # safety: max descent distance
FORCE_CONTACT_THRESHOLD_N: float = 10.0      # |vertical force| -> contact
FORCE_HARD_LIMIT_N: float = 20.0            # pick abort (empty cup)
FORCE_HARD_LIMIT_PLACE_N: float = 15.0      # place abort (battery in cup). Must sit
                                            # well above FORCE_CONTACT_THRESHOLD so a
                                            # normal seating contact registers as
                                            # contact (seat+release), not a hard abort.
VACUUM_SEAL_TIMEOUT_S: float = 5.0          # DI0 takes ~3-4s to latch
SEAL_PRELIFT_M: float = 0.001               # relieve contact press before suction on
RELEASE_PRELIFT_M: float = 0.014            # lift before the blow-off release
PLACE_Z_BUFFER_M: float = 0.10              # accept a seat within this of the taught z

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
# A measured seat deviating from the predicted z by more than this fraction of the
# pitch is logged as a probable misalignment.
LAYER_MISALIGN_FRAC: float = 0.5
RETRY_FAILED_PHASE: bool = True     # retry a failed pick/place instead of aborting
MAX_PHASE_ATTEMPTS: int = 3         # attempts per move before giving up
PHASE_RETRY_DELAY_S: float = 2.0    # pause before a retry (also lets config edits land)

# ---------------------------------------------------------------------------
# Cognex barcode reader (DataMan, DMCC over telnet)
# ---------------------------------------------------------------------------
BCR_HOST: str = "192.168.50.101"
BCR_PORT: int = 23
# A scan is accepted only if >= BCR_MIN_READS successful reads agree; stop
# triggering once BCR_MAX_READS have landed.
BCR_MIN_READS: int = 2
BCR_MAX_READS: int = 4
BCR_SCAN_TIMEOUT_S: float = 1.0     # per-trigger telnet timeout (s)

# ---------------------------------------------------------------------------
# Right-hand Robotiq gripper (Modbus RTU over the right arm's EE pass-through)
# ---------------------------------------------------------------------------
ROBOTIQ_SLAVE_ID: int = 0x09
ROBOTIQ_OPEN_POS: int = 0                  # 0 = open, 255 = closed
ROBOTIQ_PARTIAL_OPEN_POS: int = 40         # partial open to avoid ground contact
ROBOTIQ_CLOSE_POS: int = 255
ROBOTIQ_SPEED: int = 0x80                  # 0..255
ROBOTIQ_FORCE: int = 0x80                  # 0..255
# If CLOSE_POS - gPO >= this, treat the gripper as holding an object even when
# gOBJ != 2 (slim objects can reach the requested position while still gripped).
ROBOTIQ_GRIP_MIN_GAP: int = 5

# ---------------------------------------------------------------------------
# Barcode divert (barcode.py) — scan DURING the battery pick descent.
# A battery whose agreed barcode is in TARGET_BARCODES is diverted to the
# right-hand gripper instead of seated in the case. Empty list = never divert.
# ---------------------------------------------------------------------------
TARGET_BARCODES: list[str] = ["UDCG7B0289", "UDCG7B0291"]

# Barcode-gated battery pick: scan during the fast (suction-off) descent down to
# creep_z. If read there -> suction ON, creep to contact, seal. If NOT read ->
# sweep x/y AT creep_z (no lift, no tilt), bounded to the battery's side of the
# case center, re-scanning each waypoint; read -> grab; exhausted -> grab anyway
# (no divert). Sweep offsets are ellipse rings in the case-local frame.
BCR_SCAN_DWELL_S: float = 0.4            # dwell at a scan waypoint to gather reads
BCR_SWEEP_LIFT_M: float = 0.05           # raise the sweep plane this far above creep_z
                                         # (clear the battery tops / better read focus)
BCR_SEARCH_RING_STEP_M: float = 0.005    # ring spacing outward from the pick point
BCR_SEARCH_MAX_RADIUS_X_M: float = 0.08  # outermost ring, case-local x extent
BCR_SEARCH_MAX_RADIUS_Y_M: float = 0.02  # outermost ring, case-local y extent
BCR_SEARCH_ANGLES: int = 6               # waypoints per ring

# ---------------------------------------------------------------------------
# Right-arm gripper handoff (gripper.py). The suction arm holds the battery at
# transport; the right-arm gripper grips it (side approach) and places it.
# GEOMETRY BELOW IS FROM THE OLD DEMO — RE-TEACH on the robot before trusting.
# ---------------------------------------------------------------------------
GRIPPER_EE_FRAME: str = "R_gripper_base"
# Gripper grasp = suction EE pose + this offset, approached from the side (rpy).
HANDOFF_GRIP_OFFSET: tuple[float, float, float] = (0.0624, -0.1073, -0.1508)
GRIPPER_GRASP_RPY: tuple[float, float, float] = (-1.387, np.pi / 2, 0.0)
GRIPPER_PREGRASP_STANDOFF_M: float = 0.06   # back off along the approach axis, then move in
# Right-arm joint pose where the gripper releases the diverted battery (lower-right).
PLACE_LOWER_RIGHT_JOINTS: list[float] | None = None   # TEACH before enabling divert

# ---------------------------------------------------------------------------
# Chassis-based detection pick&place (chassis_sequence.py).
# The chassis strafes the source (left) / target (right) to roughly robot-center;
# a BEV case detection recenters the case in base_link at EACH visit, so the
# OPEN-LOOP strafe (move_sideways = speed*time, no odometry) need not be precise.
# TUNE the strafe speed/time and the park/default poses on the robot.
# ---------------------------------------------------------------------------
# True (testing, until Navigation owns the chassis): every strafe leg becomes an
# interactive prompt — drive with `l/r [dist_m] [speed]` commands (same grammar
# as move_chassis.py), `d` when in position. False: fixed speed*time legs below.
CHASSIS_MANUAL: bool = True

# Safe homing (go_home.py): if an arm's EE (L/R_gripper_base) sits below
# HOME_LIFT_MIN_EE_Z (i.e. down in/near a box), first lift it STRAIGHT UP to
# HOME_LIFT_EE_Z (same xy/orientation), then run the joint move home. EE-frame
# base_link z values, m.
HOME_LIFT_MIN_EE_Z: float = 0.95
HOME_LIFT_EE_Z: float = 1.05
CHASSIS_STRAFE_SPEED_MS: float = 0.1   # m/s magnitude (move_sideways: + left, - right)
CHASSIS_TURN_SPEED_RADS: float = 0.2   # rad/s magnitude for in-place yaw (turn: + ccw, - cw)
CHASSIS_STRAFE_TIME_S: float = 7.2      # seconds per leg (~distance = speed*time)
CHASSIS_SETTLE_S: float = 1.0           # settle pause after a strafe, before detecting
# layers_remaining fed to the BEV detector (sets the warp plane top_face_z =
# FLOOR_Z_BASE_M + layer*LAYER_PITCH_M). Source is a full stack picked top-down
# (first pick = full height); target is built up from the floor. Warping at the
# correct plane keeps the case metric-constant, so an L1-trained detector still
# works at other layers.
# These are the STARTING stack heights only: chassis_sequence.run loops over
# layers and steps its runtime copies (source -1, target +1) after each
# completed layer, until the source is exhausted. Set these to the physical
# stack heights at run start (an aborted run logs the values to resume with).
SRC_LAYERS_REMAINING: int = 5           # source stack height at run start
TGT_LAYERS_REMAINING: int = 1           # target stack height at run start
# Arm joint pose that clears the head-camera view of the target while an item is
# carried during transport (TUNE; defaults to the left-arm home).
ARM_VIEW_PARK_JOINTS: tuple[float, ...] = HOME_JOINTS_LEFT
# Preferred Cartesian view-park: EE base_link (x, y, z) the held item is carried
# at — push y LEFT (+) until the head view of the target is clear; orientation is
# kept as-picked (no load rotation). Falls back to ARM_VIEW_PARK_JOINTS if
# unreachable or None. (TUNE y; z should clear both box walls like SAFE_TRANSPORT_Z.)
# Offline reach scan (reach_sweep, z=1.10): y=0.30 -> x[0.69,1.0]; y=0.40 ->
# x[0.82,1.0]; y=0.50 -> x[0.86,1.0]. Keep x inside the window for the chosen y.
ARM_VIEW_PARK_EE_POS: tuple[float, float, float] | None = (0.90, 0.50, 1.10)

# Pre-flight descent reachability check (chassis_sequence): before a pick/place
# moves at all, the FULL descent column at the target xy — from the current EE
# height all the way down to the BOTTOM (box floor + suction length), regardless
# of the expected layer — must solve reachable (err<=REACH_TOL_M, in limits, no
# self-collision). A near-boundary detection (e.g. x~1.03) otherwise sends the
# streamed descent through a near-singular thrash (observed: 126.8N crash).
DESCENT_CHECK_BOTTOM_EE_Z: float = 0.74     # ~ box floor 0.566 + SUCTION_LENGTH 0.176
DESCENT_CHECK_STEP_M: float = 0.02          # z step of the pre-flight sweep
# Default target case center (base_link x, y, z_EE, yaw) used for the FIRST case
# placement, when no case is detected at the target yet (TUNE x/y).
# NOTE z is an EE z (same convention as the detected path: top_face + SUCTION
# LENGTH), NOT a top-face z. 0.7545 = measured seat contact of the first case on
# the empty target floor (2026-07-03), matching the model floor 0.566 + pitch
# 0.0138 + suction 0.176 = 0.756 within ~1mm. Descend-to-contact still refines.
TARGET_DEFAULT_CASE_CENTER: tuple[float, float, float, float] = (0.94, 0.0, 0.7545, 0.0)
