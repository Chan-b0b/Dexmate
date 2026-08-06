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
    "/home/dexmate/miniconda3/lib/python3.13/site-packages/dexmate_urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"
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
REACH_TOL_M: float = 0.01
# IK-side joint-range margin: solve/validate inside this fraction of each
# joint's URDF position range (centered), keeping solutions off the hard
# stops on every joint. URDF and dexcontrol's hardware clamps are untouched.
JOINT_RANGE_FRAC: float = 0.95

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
# duration derives from its path length. "Make it faster" = raise SPEED_SCALE_
# LEFT/RIGHT (or the individual caps). Ruckig keeps everything jerk-limited and feasible;
# arm.py additionally clamps joint commands to the arm's own reported limits.
# Streamed as (pos, vel) via arm.set_joint_pos_vel at ~100 Hz.
#
# Values are conservative starting points — TUNE on the robot.
# ---------------------------------------------------------------------------
SPEED_SCALE_LEFT: float = 1.0       # multiplier on every cap below, left (suction) arm
SPEED_SCALE_RIGHT: float = 0.5      # multiplier on every cap below, right (gripper) arm
# (0.7 = normal; lowered for first slow handoff test)

CONTROL_HZ: float = 200.0          # motion streaming rate (set_joint_pos_vel)

# Joint-space (move_joints: cached-pose -> cached-pose travel), per joint.
MAX_JOINT_VEL: float = 1.5          # rad/s
MAX_JOINT_ACCEL: float = 5.0        # rad/s^2
MAX_JOINT_JERK: float = 30.0        # rad/s^3

# Waypoint blending (move_joints_through: multi-waypoint sequences, e.g. the
# right-arm divert place). Junctions are crossed at this fraction of vmax on
# the joints that keep direction across the junction (direction-reversing
# joints cross at 0), so the arm doesn't fully stop at every taught waypoint.
# The full fraction applies only when both adjacent hops move the joint at
# least BLEND_FULL_DIST — shorter hops scale down to avoid overshoot.
JOINT_BLEND_FRAC: float = 0.3           # 0 = stop at every waypoint (old behavior)
JOINT_BLEND_FULL_DIST_RAD: float = 0.2  # per-joint hop size for the full fraction

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
SAFE_TRANSPORT_Z: float = 1.10
# Absolute EE z the straight-up (xy-held, move_ee_vertical) lift leg ends at:
# 0.15 m above the layer-2 contact = FLOOR_Z_BASE 0.566 + 2*LAYER_PITCH 0.0138
# + SUCTION_LENGTH 0.176 + 0.15 ~= 0.92 — clear of the case walls from any
# layer's pick without dragging the held part sideways. The remaining ascent
# to SAFE_TRANSPORT_Z is a faster joint-space move_ee (its sideways arc is
# harmless once clear).
LIFT_CLEAR_EE_Z: float = 0.95
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
CASE_GRAB_OFFSET: tuple[float, float, float] = (-0.04, 0.07, 0.0)   # cup grabs slightly left (+y)
# Slot-pair CENTER vs the detected OBB center (case-local). Both battery targets
# shift by this together — fixes the common-bias symptom (bat1 undershoots,
# bat2 overshoots by the same amount = center biased +y). Spacing stays symmetric.
SLOT_CENTER_OFFSET: tuple[float, float, float] = (-0.04, -0.04, 0.0)
SLOT_OFFSETS: dict[int, tuple[float, float, float]] = {
    1: (SLOT_CENTER_OFFSET[0], 0 - HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # right slot (robot -y)
    2: (SLOT_CENTER_OFFSET[0], 0 + HALF_SLOT_SPACING_M, SLOT_CENTER_OFFSET[2]),  # left slot  (robot +y)
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
DESCENT_APPROACH_SPEED_M_S: float = 0.20    # fast free-air descent (cup-tip)
DESCENT_CREEP_SPEED_M_S: float = 0.03       # slow creep in the contact zone
DESCENT_RAMP_S: float = 0.2                 # ease descent speed in from 0 (no jerk
                                            # from the rest->descend handoff)
DESCENT_CREEP_BLEND_M: float = 0.03         # decelerate fast->creep smoothly over this
                                            # band ABOVE creep_z (no velocity step at
                                            # the creep line -> no jerk at creep height)
DESCENT_CREEP_GAP_M: float = 0.05           # creep starts this far above expected contact
DESCENT_MAX_M: float = 0.40                 # safety: max descent distance
FORCE_CONTACT_THRESHOLD_N: float = 10.0      # |vertical force| -> contact
FORCE_HARD_LIMIT_N: float = 20.0            # pick abort (empty cup)
FORCE_HARD_LIMIT_PLACE_N: float = 20.0      # place abort (battery in cup). Must sit
                                            # well above FORCE_CONTACT_THRESHOLD so a
                                            # normal seating contact registers as
                                            # contact (seat+release), not a hard abort.
VACUUM_SEAL_TIMEOUT_S: float = 8.0          # DI0 takes ~3-4s to latch
SEAL_PRELIFT_M: float = 0.00               # relieve contact press before suction on
RELEASE_PRELIFT_M: float = 0.014            # lift before the blow-off release
PLACE_Z_BUFFER_M: float = 0.10              # accept a seat within this of the taught z
PLACE_MISSEAT_TOL_M: float = 0.005          # place contact this far ABOVE the expected
                                            # seat z = rim-landing (misseat): hold, don't
                                            # release. Only applied when the expectation
                                            # is a measured-contact anchor (ZTracker) —
                                            # the model plane drifts too much (0804 L5).
                                            # A proper seat drops 5-15mm past the rim
                                            # (measured 2026-08-05), so 5mm splits rim
                                            # vs seat while riding out anchor noise.
# A battery column with NO anchor of its own borrows a first-place expectation
# (0805 L1 battery_2 seated on a divider +18.9mm high, unchecked — a column's
# first contact used to be trusted blindly). Preference order + tolerance:
PLACE_MISSEAT_TOL_SIBLING_M: float = 0.008  # from the OTHER battery's anchor — seats
                                            # are symmetric (2.9mm spread seen 0805)
BATTERY_OVER_CASE_MAX_M: float = 0.02       # from the CASE's place anchor — the case
                                            # grab face IS the battery compartment, so a
                                            # battery seat is at most one battery
                                            # thickness above the case seat (< 2cm;
                                            # +12.3mm measured 0805)
CASE_PICK_RELEASE_WAIT_S: float = 2.0       # wait after release before returning home

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
# Per-run CSV of the ZTracker events (contact z vs predicted, warp-plane
# measured vs model, misseat/force_limit failures) — the layer-by-layer error
# data separated from the run log. None disables.
ZTRACK_LOG_DIR: str | None = "/home/dexmate/LGES/Dexmate/LGES/ik_demo/ztrack_logs"
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
BCR_MIN_READS: int = 1
BCR_MAX_READS: int = 4
BCR_SCAN_TIMEOUT_S: float = 1.0     # per-trigger telnet timeout (s)

# ---------------------------------------------------------------------------
# Right-hand Robotiq gripper (Modbus RTU over a USB-RS485 serial adapter,
# drivers/robotiq_usb.py; the old EE pass-through driver is drivers/robotiq.py)
# ---------------------------------------------------------------------------
# Serial port of the USB-RS485 adapter (e.g. "/dev/ttyUSB0"). None = auto-detect
# (exactly one /dev/ttyUSB* or /dev/ttyACM* must be present).
ROBOTIQ_USB_PORT: str | None = None
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
# sweep AT the raised plane (no tilt), bounded to the battery's side of the
# case center; read -> grab; exhausted -> grab anyway (no divert). The barcode
# sits CENTERED in case-local x, so the sweep is Y-FIRST: one CONTINUOUS
# case-local y pass at dx=0 (scanner polled every control tick, reads happen
# in motion — no waypoint dwells), then, only if empty, x offsets nearest
# first (+/-X_STEP, +/-2*X_STEP ... to MAX_X), each rerunning the y pass
# (direction alternating, so there's no wasted return leg).
BCR_SWEEP_LIFT_M: float = 0.1           # raise the sweep plane this far above creep_z
                                         # (clear the battery tops / better read focus)
BCR_SWEEP_SPEED_M_S: float = 0.05        # continuous y-pass EE speed (reads in motion)
BCR_SEARCH_MAX_Y_M: float = 0.05         # +/- extent of the continuous y pass
BCR_SEARCH_X_STEP_M: float = 0.02        # x-offset step of the fallback columns
BCR_SEARCH_MAX_X_M: float = 0.05         # +/- extent of the x fallback offsets

# ---------------------------------------------------------------------------
# Right-arm gripper handoff (gripper.py). The suction arm holds the battery at
# transport; the right-arm gripper grips it (side approach) and places it.
# GEOMETRY BELOW IS FROM THE OLD DEMO — RE-TEACH on the robot before trusting.
# ---------------------------------------------------------------------------
GRIPPER_EE_FRAME: str = "R_gripper_base"
# Suction arm hovers here (moving right, at whatever transport height it's
# already at) before every divert grip — ONE fixed xy regardless of which
# battery (1 or 2) triggered the divert, instead of each slot's own (different)
# xy, so the grip point (and everything downstream of it) is consistent run
# to run. Reuses BAT_SLOT_1's xy.
HANDOFF_HOVER_XY: tuple[float, float] = (TAUGHT_POSES["BAT_SLOT_2"][0], TAUGHT_POSES["BAT_SLOT_2"][1])
# Gripper grasp = suction EE pose + this offset, approached from the side (rpy).
HANDOFF_GRIP_OFFSET: tuple[float, float, float] = (0.07, -0.10, -0.185)
GRIPPER_GRASP_RPY: tuple[float, float, float] = (-1.5, np.pi / 2, 0.0)
GRIPPER_PREGRASP_STANDOFF_M: float = 0.08   # back off along the approach axis, then move in
# After the handoff release (suction off, gripper holds the battery), the left
# suction arm retreats to this EE position (base_link, m) — out to the LEFT at
# hover height — so it clears the right arm's place motion. Orientation is kept
# as-is. Shifted from y=0.50 toward BAT_SLOT_1's hover xy (moved right) per
# reach_sweep: reachable at (x=0.90, y=-0.078725, z=1.10), err=0.0mm. (TUNE)
HANDOFF_LEFT_CLEAR_EE_POS: tuple[float, float, float] = (0.90, -0.078725, 1.10)
# Right-arm EE place sequence, run AFTER the gripper grips the suction-held
# battery and suction releases. Each step is (label, is_relative, (x,y,z) m,
# (roll,pitch,yaw) rad in base_link). REL steps add to the last COMMANDED pose
# (rotations compose in SO(3), not by Euler addition — the readout flips near
# gimbal lock). The gripper partial-opens to release at the step whose label
# contains "lower". Ported from the old demo's taught_ee_poses_right.txt, then
# reworked: step 1 (REL) carries the battery clear of the grip point; step 2
# ("To Right 2") is now a fixed ABSOLUTE waypoint instead of REL — composing
# it relative to the grip point meant its landing spot (and reachability)
# depended on exactly where the suction arm's hover-and-grip happened to land,
# which is why it kept failing. Fixed at a point verified reachable from both
# battery slots' grip points; the 3 remaining absolute poses fine-position,
# lower/release, and retract.
# POSITIONS ARE FROM THE OLD DEMO — RE-TEACH on the robot before trusting.
#
# Orientation: "To Right 2" faces forward (tool +z ~= base_link +x) — composing
# through GRIPPER_GRASP_RPY's exact pitch=pi/2 couples a yaw turn into roll,
# hence the "+ pi/2" below rather than 0. The 3 remaining absolute poses share
# that same forward orientation tilted down a little (only position differs
# between them) instead of 3 independently-taught (and gimbal-lock-garbled)
# triples.
HANDOFF_DOWN_TILT_RAD: float = np.radians(30.0)   # "a little" downward pitch
HANDOFF_RIGHT_FORWARD_RPY: tuple[float, float, float] = (
    GRIPPER_GRASP_RPY[0] + np.pi / 2, np.pi / 2, 0.0,
)
HANDOFF_RIGHT_FORWARD_DOWN_RPY: tuple[float, float, float] = (
    HANDOFF_RIGHT_FORWARD_RPY[0], HANDOFF_RIGHT_FORWARD_RPY[1] + HANDOFF_DOWN_TILT_RAD, HANDOFF_RIGHT_FORWARD_RPY[2],
)
# Verified reachable (err<0.2mm, no collision, in-limits) from BOTH BAT_SLOT_1
# and BAT_SLOT_2 grip points — see check_place_seq.py.
PLACE_LOWER_RIGHT_EE_SEQ: list | None = [
    ("To Right 1", True,  (0.0, 0.0, -0.06),               (0.0, 0.0, 0.0)),
    ("To Right 2", False, (1.04, -0.4587, 0.9692),         HANDOFF_RIGHT_FORWARD_RPY),
    ("To Right 3", False, (0.876799, -0.507694, 0.925),    HANDOFF_RIGHT_FORWARD_DOWN_RPY),
    ("Lower",      False, (0.751889, -0.548590, 0.70),   HANDOFF_RIGHT_FORWARD_DOWN_RPY),
    ("Back",       False, (0.635044, -0.548479, 0.72),     HANDOFF_RIGHT_FORWARD_DOWN_RPY),
]

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
# Speed for the LONG station<->station legs only (auto-move); small centering /
# adjust corrections stay at CHASSIS_STRAFE_SPEED_MS — at higher speeds a
# 3-10 cm move is ramp-dominated and lands poorly, while long-leg arrival error
# is absorbed by centering + the per-direction leg learning. dexcontrol clips
# to the robot's max_lin_vel (~0.5). If the carried case slips on the cup at
# higher accel (watch the placement), lower this back.
CHASSIS_LEG_SPEED_MS: float = 0.2
CHASSIS_TURN_SPEED_RADS: float = 0.2   # rad/s magnitude for in-place yaw (turn: + ccw, - cw)
CHASSIS_STRAFE_TIME_S: float = 7.2      # seconds per leg (~distance = speed*time)
CHASSIS_SETTLE_S: float = 1.0           # settle pause after a strafe, before detecting
# --auto-move (chassis_sequence CLI flag): chassis legs run automatically — each
# source<->target leg is a fixed open-loop DISTANCE at CHASSIS_STRAFE_SPEED_MS
# (overrides CHASSIS_MANUAL). Station spacing is known ~0.6-0.7 m; detection
# recenters at each visit, so the leg only needs to land the case in view/reach.
CHASSIS_AUTO_STRAFE_DIST_M: float = 0.7
# Auto-adjust on a failed reach pre-check (auto-move only): turn the chassis so
# the detected case yaw matches the taught reference (yaw 0), then translate so
# the case center lands on the reference (x from SOURCE_CASE_CENTER at the
# source / TARGET_DEFAULT_CASE_CENTER at the target, y from
# CHASSIS_CENTER_CASE_Y_M), re-detect, retry. After MAX_ATTEMPTS the operator
# gets the interactive keyboard prompt, as in manual mode. Per-attempt motion
# is clamped; deadbands skip micro-moves.
CHASSIS_ADJUST_MAX_ATTEMPTS: int = 3
CHASSIS_ADJUST_MAX_TRANSLATE_M: float = 0.30   # per-attempt clamp, each axis
CHASSIS_ADJUST_MAX_TURN_DEG: float = 30.0      # per-attempt clamp, in-place turn
CHASSIS_ADJUST_MIN_TRANSLATE_M: float = 0.01   # deadband: skip smaller translations
CHASSIS_ADJUST_MIN_TURN_DEG: float = 3.0       # deadband: skip smaller turns
# Learned leg distances (auto-move): the arrival residual after each leg
# (centering/adjust strafe, minus deliberate per-item re-alignments) feeds the
# distance of the LEG THAT JUST RAN — left (target->source) and right
# (source->target) learn independently, so a direction-dependent open-loop
# travel gain calibrates out. In-memory only — final values are logged at run
# end for a manual config update. Each clamped to CHASSIS_AUTO_STRAFE_DIST_M
# +/- this; a persistent "clamped" log means the true station separation is
# outside the clamp window — fix CHASSIS_AUTO_STRAFE_DIST_M instead.
CHASSIS_LEG_LEARN_CLAMP_M: float = 0.20
# Case centering (auto-move): at EVERY station visit the chassis first turns
# in place so the detected case yaw reads 0 deg (deadband / clamp reuse
# CHASSIS_ADJUST_MIN/MAX_TURN_DEG), then strafes so the case center sits at
# CHASSIS_CENTER_CASE_Y_M in base_link (0.0 = the robot center line) — BEFORE
# the pick/place pose is computed. Detection is most accurate, the reach
# window widest (yaw 0 = the taught wrist branch), and the source/target
# biases most symmetric (so they cancel through the carry), with the case
# square and dead ahead. Up to MAX_MOVES correction rounds per visit, each
# followed by a re-detect; applied strafes feed the learned leg distance
# (turns are not tracked — small headings converge to the stack orientation).
# The reach pre-check still guards every descent — if center-line picks keep
# failing it, raise CENTER_CASE_Y toward the taught y (~0.45).
CHASSIS_CENTER_CASE_Y_M: float = 0.0
CHASSIS_CENTER_TOL_M: float = 0.05      # |case y - target| accepted without a move
CHASSIS_CENTER_MAX_MOVES: int = 5       # per-visit correction rounds
# Detection plausibility gate (auto-move): the stations are only one leg
# (~CHASSIS_AUTO_STRAFE_DIST_M) apart, so the OTHER station's stack is in the
# head-camera view and the detector returns the highest-confidence OBB
# anywhere in frame. A detection whose case-center y is farther than this
# from the expected ref is rejected as "not found" (observed: the first
# target visit locked onto the SOURCE stack and dragged the robot back left).
# Must be well below the leg distance and above the arrival error (~0.15).
CHASSIS_DETECT_Y_GATE_M: float = 0.50
# Final-detection refinement: the detection a pick/place pose is computed from
# is the MEDIAN of this many fresh-frame samples (x/y/yaw; z comes from the
# warp plane, identical across samples). Robust to single-frame OBB jitter and
# one bad fit; does nothing for systematic bias. Centering rounds stay
# single-shot. 1 = off. Cost: ~0.2-0.5 s per extra sample (fresh-frame wait +
# YOLO inference).
DETECT_MEDIAN_SAMPLES: int = 5
# Bin-aligned divert positioning (chassis_sequence): before a gripper divert,
# the head camera finds the divert bin (case_detection detect_bin) and the
# chassis strafes so the bin center sits at DIVERT_BIN_TARGET_Y_M in base_link
# (+left; 0.10 = 10 cm left of the robot center line). After the move the bin
# is re-detected and a residual over DIVERT_BIN_TOL_M gets ONE more
# correction. The net move is strafed back after the divert (or before the
# fallback case place) so the normal target geometry is restored.
DIVERT_BIN_TARGET_Y_M: float = 0.0
# Plane z the bin bbox center is projected at (base_link, ~bin rim height).
# Only weakly affects Y once the bin nears the center line — the projection-
# ray bias scales with the lateral offset, and the re-detect pass converges
# the residual — so a rough value is fine. (TUNE if the first-move overshoot
# looks large in the logs.)
DIVERT_BIN_PLANE_Z_M: float = 0.55
DIVERT_BIN_TOL_M: float = 0.03          # accepted Y residual after the first move
DIVERT_BIN_MAX_STRAFE_M: float = 0.6    # per-move safety clamp on the align strafe
# Fallback when NO bin is detected: fixed extra rightward strafe (the original
# open-loop behavior), strafed back like the aligned move. 0.0 = stay put.
DIVERT_EXTRA_RIGHT_M: float = 0.1
# Chassis command timing compensation (move_chassis, DISTANCE-based legs only;
# the legacy speed*time legs keep their empirically tuned values untouched).
# dexcontrol streams a timed velocity command for max(wait_time - 1.0, 0) s
# (chassis.py _execute_timed_command), so a distance leg adds the clipped
# second back. Keep this at 1.0 (the library's fixed clamp): the ~7 cm
# over-travel that motivated lowering it was the firmware COASTING past a
# dropped single-shot stop command — fixed by streaming the stop instead
# (move_chassis._stop, CHASSIS_STOP_STREAM_S below). Lowering DEAD_TIME makes
# small centering moves silently under-drive (a <=7 cm move streams 0 s).
CHASSIS_CMD_DEAD_TIME_S: float = 1.0
# The post-leg stop is STREAMED (zero velocity at ~50 Hz, steering kept) for
# this long — a single stop command was observed to get lost, letting the
# chassis coast ~0.7 s past the leg.
CHASSIS_STOP_STREAM_S: float = 0.3
# Pre-steer before a distance leg: command the leg's steering angles first and
# poll chassis.steering_angle until within TOL (or TIMEOUT, then drive anyway),
# so none of the timed drive is spent pivoting the wheels — dexcontrol's own
# sequential-steer hold resolves to 0 s through the same -1.0 s clamp.
CHASSIS_PRESTEER_TOL_RAD: float = 0.05
CHASSIS_PRESTEER_TIMEOUT_S: float = 5.0
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
TGT_LAYERS_REMAINING: int = 1          # target stack height at run start
# Arm joint pose that clears the head-camera view of the target while an item is
# carried during transport (TUNE; defaults to the left-arm home).
ARM_VIEW_PARK_JOINTS: tuple[float, ...] = HOME_JOINTS_LEFT
# Preferred Cartesian view-park: EE base_link (x, y, z) the held item is carried
# at — push y LEFT (+) until the head view of the target is clear; orientation is
# kept as-picked (no load rotation). Falls back to ARM_VIEW_PARK_JOINTS if
# unreachable or None. (TUNE y; z should clear both box walls like SAFE_TRANSPORT_Z.)
# Offline reach scan (reach_sweep, z=1.10): y=0.30 -> x[0.69,1.0]; y=0.40 ->
# x[0.82,1.0]; y=0.50 -> x[0.86,1.0]. Keep x inside the window for the chosen y.
ARM_VIEW_PARK_EE_POS: tuple[float, float, float] | None = (0.90, 0.40, 1.10)

# Online FiLM authority probe: safe, non-contact height sweep above the detected
# case top. Values are cup-tip clearances; the probe converts them to EE z with
# SUCTION_LENGTH_M. Policy predictions are logged only and never commanded.
VLA_FILM_PROBE_CLEARANCES_M: tuple[float, ...] = (0.25, 0.10, 0.03)
VLA_FILM_PROBE_SETTLE_S: float = 0.5
# Counterfactual fz sweep in raw-force units. Converted to FiLM input units with
# the checkpoint runtime _fz_tau, so this stays interpretable across calibrations.
VLA_FILM_PROBE_FZ_DELTAS_N: tuple[float, ...] = (-3.0, 3.0)

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
TARGET_DEFAULT_CASE_CENTER: tuple[float, float, float, float] = (0.92, 0.0, 0.7545, 0.0)
# Closed-loop seed place: the place center comes from a bin detection AFTER the
# align strafe, plus this bias — the bbox-center projection reads the bin ~47mm
# FORWARD of its true center (front wall + plane mismatch; measured on-robot
# 2026-08-06, hand-centered cup vs detection). Detect fail -> the default pose.
SEED_BIN_CENTER_OFFSET: tuple[float, float] = (-0.0469, +0.0009)
# Misseat gate for the seed place: contact more than this ABOVE the taught seat
# z means the case landed on the bin wall/rim, not the floor (walls sit several
# cm higher; taught z is hand-measured, drift-free at this fixed station).
SEED_MISSEAT_TOL_M: float = 0.025
# Per-layer forward trim on TARGET places (chassis_sequence): every place of a
# layer lands (tgt_layers - 1) * this further +x (base_link forward) — layer 1
# gets no trim. Compensates the forward placement bias that grows with the
# stack height (the BEV warp-plane geometry biases the detected center along
# the camera ray as the top face rises). Applies to the case AND the layer's
# battery seats (same detection, same bias). 0.0 disables.
PLACE_X_LAYER_TRIM_M: float = 0.0   # was 0.008 for the MODEL-plane regime; with the
                                    # measured plane + PLACE_X_PLANE_TRIM the per-layer
                                    # bias is gone (0806: target dual_x flat +7..+11mm
                                    # across L1-L4) and the trim showed a clean forward
                                    # dose-response on battery seats (L2 ok / L3 one
                                    # recovery / L4 +24mm unrecoverable, operator: "x를
                                    # 더 가깝게"). Restore 0.008 only with model planes.
# Constant x trim on TARGET places whenever the MEASURED warp plane was used
# for the detection (tgt_plane active). The taught offsets/trims were tuned
# against the MODEL plane's constant xy bias; the measured (true) plane removes
# that bias, which un-cancels the tuning: 0805 dual-plane probe measured the
# net shift as +12.8/+13.7mm forward (source pick -5..-7mm grab offset carried
# into the place + target detection +6.6..+7.8mm) — both L1 batteries landed
# ~+13mm too far +x, battery_2 on the divider. This pulls it back. Verify with
# the dual_x CSV rows + seat quality on the next run; 0.0 disables.
PLACE_X_PLANE_TRIM_M: float = -0.013
# Constant yaw trim (rad, base CCW+) on TARGET place wrists: the part arrives
# systematically twisted on the cup (unobservable — the system never sees the
# battery; 0805: both batteries landed ~2deg twisted, the empty case conformed
# to battery_1's angle [target det yaw +0.2deg -> -1.6deg after its seat], the
# loaded case couldn't for battery_2 -> jam at +26mm). This pre-rotates the
# wrist to land the part square. +0.031 (~+1.8deg) cancels the 0805 estimate;
# if a test run doubles the twist instead, flip the sign. 0.0 disables.
PLACE_YAW_TRIM_RAD: float = 0.0 #낮출수록 CW
# Post-place verification (CASE places): after the release, park the arm to
# clear the head view and re-detect the just-placed case BEFORE the chassis
# moves — landed-vs-intended (dx, dy, dyaw) in one base frame (CSV
# place_chk_x/y/yaw). Case places align to the layer BELOW, so a constant
# per-place bias accumulates layer over layer (0805: L2 +1.4mm, L3 barely in,
# L4 misseat +11.5 — "a bit left" each layer); this measures that bias
# directly instead of guessing trims. Costs one sync view-park + detection
# per case place (the park loses its overlap with the return strafe).
PLACE_VERIFY_DETECT: bool = True
# Misseat contact recovery (suction.place): on a rim-landing, keep the part
# held, lift PLACE_RECOVER_LIFT_M above the failed contact, re-orient the
# wrist yaw by the next pattern step (offsets from the commanded place yaw),
# and creep back down; success = a contact inside the seat band. Absorbs the
# staging-dependent in-hand twist that constant yaw trims can't track (0805
# tuned trim broke on the 0806 restage). Every place contact also snapshots
# the tared 6-axis wrench + commanded-vs-measured EE yaw (CSV contact_wrench /
# contact_yaw) — Phase 2 uses the mz sign to pick the search direction first.
PLACE_RECOVER_ATTEMPTS: int = 10
PLACE_RECOVER_YAW_PATTERN_RAD: tuple[float, ...] = (0.026, -0.026, 0.052, -0.052)
PLACE_RECOVER_LIFT_M: float = 0.010         # re-orient height above the failed contact
# Phase 2 — force-guided translation: on a misseat contact with a tared lateral
# force of at least FORCE_MIN, the obstruction is pushing the part toward the
# free side (0806 L4 bat1: fx=-5.8N and the operator's verdict was "move -x"),
# so step XY along that force direction (yaw kept) instead of the next yaw
# step. Total XY excursion from the commanded pose is capped by XY_MAX.
PLACE_RECOVER_XY_STEP_M: float = 0.003
PLACE_RECOVER_FORCE_MIN_N: float = 1.5  # was 3.0 — a FLAT landing (part on top of a
                                        # divider) converts almost no press into
                                        # lateral push, so weak-but-real signals
                                        # were filtered and recovery went yaw-only
                                        # (0806 bat2). Noise floor ~0.5N.
PLACE_RECOVER_XY_MAX_M: float = 0.015   # was 0.009 — TOTAL displacement cap; with a
                                        # mixed-direction force the per-axis reach was
                                        # ~4mm and the cap silently dropped recovery to
                                        # yaw-only (0806 bat1 "needed more -y")
# mz-FEEDBACK adaptive yaw: the slot edges twist a misaligned part TOWARD
# alignment, so each contact's mz is both a direction and a progress signal.
# Per yaw attempt, compare mz with the previous contact's:
#   |mz| shrank            -> helping: keep direction, same step
#   sign flipped           -> overshot: reverse, halve the step
#   |mz| grew (same sign)  -> wrong way: reverse (self-corrects an unknown
#                             sign convention without a config flip)
# Engages when |mz| >= MZ_MIN (below it there is no torque information —
# fall back to the blind PATTERN above). Cumulative offset capped at YAW_MAX.
# Caveat: mz carries an (r x f_lat) term when the contact is off the cup
# axis; the feedback comparisons tolerate a constant bias but validate with
# the recover_step CSV rows. Observed mz range 0.03-0.16 Nm (0806).
PLACE_RECOVER_MZ_MIN_NM: float = 0.05
PLACE_RECOVER_YAW_STEP_RAD: float = 0.026   # initial adaptive step (~1.5 deg)
PLACE_RECOVER_YAW_MAX_RAD: float = 0.070    # |cumulative yaw offset| cap (~4 deg)
# Blind XY fallback for FLAT landings: a part resting on a flat jig/divider
# top converts no press into lateral push (f_lat < FORCE_MIN) and yaw wiggles
# don't move it (0806 L2/L3 case: 10+9 yaw-only attempts, contact z pinned at
# +7mm) — so with no force signal walk this ABSOLUTE offset pattern from the
# commanded pose, before any yaw. -x first: every measured run landed cases
# +4..+12mm FORWARD of intended (place_chk_x), so backward is the prior. On a
# flat landing mz is (r x f) junk — ignore it until the pattern is spent.
PLACE_RECOVER_BLIND_XY_M: tuple[tuple[float, float], ...] = (
    (0.01, 0.0), (0.02, 0.0), (0.023, 0.0),
    (0.02, -0.003), (0.02, +0.003), (0.015, 0.0))
