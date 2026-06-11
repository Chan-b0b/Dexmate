"""Configuration for the case + battery suction demo.

Single-arm suction pick-and-place choreography:

    1. Move the empty plastic case from the left (source) box to the
       right (target) box.
    2. Pick two batteries from the source and seat them in the moved case.
    3. Undo everything (batteries right -> left, then the case).

All poses are expressed in the robot **base_link** frame, in metres.
Orientation is the straight-down suction approach (cup facing -Z).

Nothing here moves the torso — the chosen arm must reach both boxes on its
own. If a taught pose is unreachable, reposition the boxes rather than
turning the torso.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Robot model / IK  (constants copied from grasp_box/config.py so this module
# is self-contained and does not depend on a fragile ``import config``)
# ---------------------------------------------------------------------------
URDF_PATH: str = (
    "/opt/venv/lib/python3.12/site-packages/dexmate_urdf/"
    "robots/humanoid/vega_1p/vega_1p_gripper.urdf"
)

# Per-tick motion trace. When enabled, every set_joint_pos call appends a row
# (timestamp, leg label, commanded 7 joints, actual 7 joints) to TRACE_PATH so
# a jerk can be localized to a specific joint + leg, and commanded-vs-actual
# tells us whether we command the discontinuity or it's servo tracking.
TRACE_ENABLED: bool = True
TRACE_PATH: str = "/tmp/cns_trace.csv"

IK_DT: float = 0.01
IK_MAX_ITERS: int = 500
# Combined position+orientation twist-norm tolerance. A soft posture task adds
# a small steady-state bias, so 1e-4 is unrealistically tight; 1e-3 (~1 mm /
# ~0.06 deg) is the practical converged target. Lower POSTURE_COST to tighten.
IK_CONVERGENCE_THRESHOLD: float = 1e-3
CONTROL_DT: float = 0.02            # 50 Hz control loop
PREFERRED_QP_SOLVER: str = "daqp"

# Nullspace regularization. With the torso locked the 7-DOF arm has one
# redundant DOF for a full 6-DOF target; the posture task pulls that DOF toward
# the joint mid-ranges (auto-computed from URDF limits) to keep joints away
# from their motor stops. Set to 0.0 to disable centering entirely.
POSTURE_COST: float = 1e-3
# Levenberg-Marquardt damping on the EE task — stabilizes solves near
# singularities (trades a little tracking error for a lot of stability).
IK_LM_DAMPING: float = 1e-6

# ---------------------------------------------------------------------------
# Arm with the suction end-effector
# ---------------------------------------------------------------------------
ARM_SIDE: str = "left"              # "left" or "right"
EE_FRAME: str = "L_gripper_base"    # URDF frame; switch to R_gripper_base for right

# Straight-down suction approach orientation (roll, pitch, yaw) in radians.
# Roll/pitch make the cup vertical (-Z). The yaw here is a nominal default; in
# teach mode the yaw is initialized to (current yaw + YAW_OFFSET_DEG) so the
# battery aligns with the plastic-case slot, then recorded per pose.
# TODO: tune roll/pitch so the cup tip points straight down in base_link.
GRASP_ORIENTATION_RPY: tuple[float, float, float] = (np.pi, 0.0, 0.0)

# Yaw applied relative to the arm's current yaw when reorienting to vertical in
# teach mode, so the battery fits the case. +90 = counter-clockwise (from
# above). Flip the sign if it turns the wrong way.
YAW_OFFSET_DEG: float = 110

# ---------------------------------------------------------------------------
# Right-hand Robotiq gripper (driven over the RIGHT arm's EE pass-through,
# Modbus RTU). A target battery is handed from the suction cup to this gripper
# at the transport stage and then placed lower-right.
# ---------------------------------------------------------------------------
GRIPPER_EE_FRAME: str = "R_gripper_base"   # right-arm IK frame for the gripper
ROBOTIQ_SLAVE_ID: int = 0x09
ROBOTIQ_OPEN_POS: int = 0                  # 0 = open, 255 = closed
ROBOTIQ_PARTIAL_OPEN_POS: int = 40        # partial open to avoid ground contact; tune
ROBOTIQ_CLOSE_POS: int = 255
ROBOTIQ_SPEED: int = 0x80                  # 0..255
ROBOTIQ_FORCE: int = 0x80                  # 0..255
# Minimum position gap from CLOSE_POS to consider the gripper holding an object.
# gOBJ==2 (stopped on object) is the primary signal, but slim objects may
# reach gOBJ==3 (at requested position) while still physically gripped.
# If CLOSE_POS - gPO >= this value, treat it as gripped regardless of gOBJ.
# Tune: with nothing in the gripper gPO should be ~CLOSE_POS; with the
# battery it will stop a few counts short. Start at 5 and raise if needed.
ROBOTIQ_GRIP_MIN_GAP: int = 5

# Suction -> gripper handoff at the transport pose. The gripper grips the
# battery the suction cup is holding at SAFE_TRANSPORT_Z; the grip target is
# computed live from the suction EE pose plus this offset (base_link metres),
# approached horizontally from the side. TUNE on the robot.
# Measured 2026-06-09 (gripper EE - suction EE at the handoff pose):
#   dx ~aligned, dy = -0.127 (gripper side reach), dz = -0.179 (battery hangs below cup).
HANDOFF_GRIP_OFFSET: tuple[float, float, float] = (0.0624, -0.1073, -0.1508)
# Exact-horizontal side approach: pitch = 90deg (approach axis horizontal toward
# the battery), gripper body level; roll holds the posed approach direction.
GRIPPER_GRASP_RPY: tuple[float, float, float] = (-1.387, np.pi / 2, 0.0)
GRIPPER_PREGRASP_STANDOFF_M: float = 0.06  # back off along the approach axis, then move in
HANDOFF_GRIP_DURATION_S: float = 1.5
# Duration (s) per EE pose step in the post-grip place sequence.
EE_PLACE_STEP_DURATION_S: float = 2.0
# Right-arm joint pose where the gripper releases the battery (lower-right).
# Teach with: python -m case_battery_demo.teach_joint_pose --side right
PLACE_LOWER_RIGHT_JOINTS: list[float] | None = None

# ---------------------------------------------------------------------------
# Suction hardware
# ---------------------------------------------------------------------------
SUCTION_HOST: str = "192.168.5.1"
SUCTION_BASE_URL: str = f"http://{SUCTION_HOST}/api/dc/weblogic"

# ---------------------------------------------------------------------------
# Cognex barcode reader (DataMan, DMCC over telnet). The dashboard pulls the
# last captured frame with ``||>IMAGE.SEND`` (image-only — it never triggers a
# read, so it can't fight the demo's own scanning).
# ---------------------------------------------------------------------------
BCR_HOST: str = "192.168.50.101"
BCR_PORT: int = 23

# Batteries we're looking for: a battery whose decoded barcode is in this list
# is diverted to the right-hand gripper (placed lower-right); everything else
# follows the normal suction-into-case workflow. Empty = nothing matches, so
# the demo behaves exactly like the original suction-only choreography.
TARGET_BARCODES: list[str] = ['UDCG7B0307', 'UDCG7B0289', 'UDCG7B0291']
# The barcode is read during the suction pick descent (bcr.BackgroundScanner).
# A scan is accepted only if at least BCR_MIN_READS successful reads were
# collected and they all agree; any disagreement is treated as "no target".
BCR_MIN_READS: int = 2
BCR_SCAN_TIMEOUT_S: float = 1.0     # per-trigger telnet timeout (s)
# Start scanning when the suction EE is within this distance above the target z.
# Keeps reads to the final centimetres of descent where the barcode is closest.
BCR_SCAN_Z_THRESHOLD_M: float = 0.25

# Scan gate: resolve the barcode BEFORE the suction grab. The cup descends to a
# floor just above contact (suction off) and scans; if nothing reads, it lifts a
# little and walks an expanding-ring spiral in x/y, re-scanning at each waypoint,
# to bring a mis-aligned label into the top-down reader's view; then it returns
# to the suction point and grabs. Set BCR_SCAN_GATE_ENABLED=False to fall back to
# the old behavior (scan during the seal descent, decide after the grab).
BCR_SCAN_GATE_ENABLED: bool = True
BCR_SCAN_FLOOR_OFFSET_M: float = 0.01   # stop the scan descent this far above contact
BCR_SCAN_APPROACH_SPEED_M_S: float = 0.04  # slow approach speed for the scan descent toward the battery
BCR_SCAN_DWELL_S: float = 0.4           # dwell at a scan pose to gather >= BCR_MIN_READS
BCR_SEARCH_LIFT_M: float = 0.05         # "lift a bit" before starting the spiral
BCR_SEARCH_RING_STEP_M: float = 0.005   # spiral ring spacing
BCR_SEARCH_MAX_RADIUS_M: float = 0.02   # outermost spiral ring
BCR_SEARCH_ANGLES: int = 6              # waypoints per ring
BCR_SEARCH_MOVE_S: float = 0.8          # travel time between spiral waypoints
BCR_SEARCH_ROLL_DEG: float = 5.0        # roll tilt applied at each waypoint (alternating +/-); 0 disables
BCR_SEARCH_PITCH_DEG: float = 5.0       # pitch tilt at each waypoint (flips every 2 waypoints); 0 disables

# weblogic program IDs (from suction/test_suction.py)
SUCTION_ON_ID: int = 3587
SUCTION_OFF_ID: int = 763
BLOW_ON_ID: int = 963
BLOW_OFF_ID: int = 5089

# Suction seal detection.
# Primary signal is DI0 (dInput[0]) over the controller's socketio stream:
# it goes T the moment a vacuum seal is achieved while suction is
# commanded ON. See VacuumMonitor in suction_io.py.
#
# toolA (the tool/pump current) is intentionally NOT used as a seal
# signal — the OFF idle baseline (~0.012 A) is higher than the
# running-pump current (~0.006 A), making it ambiguous without extra
# state. It is still surfaced for diagnostics/logging only.

# Physical suction tube length from EE origin to cup tip (m).
SUCTION_LENGTH_M: float = 0.25
# Height the cup tip hovers above a target before activating suction (m).
HOVER_HEIGHT_M: float = 0.10

# ---------------------------------------------------------------------------
# Descent (pick) — descend until vacuum seal / contact, like suction_grasp.py
# ---------------------------------------------------------------------------
DESCENT_MAX_STEP_M: float = 0.005   # max step size, far from target (0.004/DT=80 mm/s).
                                     # Only caps the SAFE upper region: the hover-leg
                                     # speed is derived from this (pick() ties them so the
                                     # handoff stays matched) and the descent loop's
                                     # far-from-target traverse. Contact gentleness is set
                                     # by DESCENT_MIN_STEP_M + the KP ramp, not this.
DESCENT_MIN_STEP_M: float = 0.0002   # min step size (at target, ~20 mm/s)
DESCENT_KP: float = 0.1            # step = clip(dist_to_target * KP, min, max)
DESCENT_DT_S: float = 0.05          # 50 ms between steps

# Place descent runs slower than pick: the cup is carrying a battery, the seat
# tolerance is tight, and we want the impact (if any) to be gentle. Halving
# the step caps gives ~10 cm/s peak / ~4 mm/s near goal at the same 50 ms tick.
PLACE_DESCENT_MAX_STEP_M: float = 0.002
PLACE_DESCENT_MIN_STEP_M: float = 0.0001
PLACE_DESCENT_KP: float = 0.05
PLACE_DESCENT_DT_S: float = 0.05

MAX_DESCENT_M: float = 0.40         # safety: stop after 40 cm of descent
PLACE_Z_BUFFER_M: float = 0.1      # stop placing when within 10 mm of target z
LIFT_STEP_M: float = 0.005          # 5 mm per step (lift: ~50 mm/s)
# Small upward jog performed *before* the blow/suction-off pulse, so the cup
# breaks contact with the placed object cleanly (no sticking, no blowback
# pushing the part). Set to 0.0 to disable.
RELEASE_PRELIFT_M: float = 0.014    # 10 mm pre-release lift
# Jitter DISABLED: at the far cross-body place reach the IK is ill-conditioned,
# so this small Cartesian wobble (esp. the angular terms) was amplified into
# erratic 10 rad/s joint commands — the place-descent jerk (confirmed via the
# motion trace: pick descent, identical but jitterless, is dead smooth). Re-
# enable a SMALL amount only if a battery actually fails to seat without it,
# and prefer X/Y over the angular terms (orientation wobble is the worst).
JITTER_AMPLITUDE_M: float = 0.001     # ±X/Y sinusoidal jitter during place descent
JITTER_FREQ_HZ: float = 3.0         # jitter cycles per second
JITTER_YAW_AMPLITUDE_RAD: float = np.deg2rad(0.0)    # yaw wobble in sync with X/Y jitter
JITTER_ROLL_AMPLITUDE_RAD: float = np.deg2rad(0.0)   # roll wobble (cos-phase)
JITTER_PITCH_AMPLITUDE_RAD: float = np.deg2rad(0.0)  # pitch wobble (sin-phase, offset)
# Linearly ramp jitter amplitude from 0 -> 1 (× the configured amplitudes
# above) over the first JITTER_RAMP_S seconds of place descent. Without this
# the cos-phase / sin(+π/2) terms snap to full amplitude on step 0 and the
# cup visibly flicks at the start of the descent.
JITTER_RAMP_S: float = 0.5

# ---------------------------------------------------------------------------
# Contact detection (wrist wrench, via grasp_box/read_force.py)
# ---------------------------------------------------------------------------
FORCE_CONTACT_THRESHOLD_N: float = 5.0
FORCE_HARD_LIMIT_N: float = 20.0          # pick(): empty cup, low baseline noise
FORCE_HARD_LIMIT_PLACE_N: float = 10.0    # place(): cup carrying battery — kept
                                          # tight to protect the battery from
                                          # being crushed if the seat is wrong.
                                          # If this trips on noise, the tared
                                          # baseline is bad — investigate via
                                          # the diagnostic logs in place().
VACUUM_SEAL_TIMEOUT_S: float = 5.0  # wait for seal after contact before failing (DI0 takes ~3-4s to trigger)

# ---------------------------------------------------------------------------
# Gross motion
# ---------------------------------------------------------------------------
MOVE_DURATION_S: float = 2.0     # travel time to a hover pose

# ---- Trajectory smoothing (sideways travel + vertical lift) --------------
# When True, _move_ee_to() interpolates the EE pose linearly in Cartesian
# space with a smoothstep time profile and warm-started per-substep IK,
# instead of doing a single IK solve to the target and smoothstepping in
# joint space. Gives a straight EE path with ease-in / ease-out velocity
# and avoids elbow swings between distant configurations. Set False to
# A/B against the original joint-space interp.
USE_CARTESIAN_INTERP: bool = True
# Substep period for the Cartesian-interpolated travel. 50 Hz matches the
# control loop the arm already runs at.
EE_TRAVEL_DT_S: float = 0.02
# Smoothstep "easing strength" for sideways travel and lift:
#   "cubic"   -> 3t² - 2t³  (C¹ continuous, zero velocity at endpoints)
#   "quintic" -> 6t⁵ - 15t⁴ + 10t³ (C² continuous, also zero accel at endpoints)
# Quintic is gentler on the motors (no jerk spike at start/stop); cubic is
# what the original _move_to_joints used.
SMOOTH_PROFILE: str = "quintic"
# Lift / vertical Cartesian Z is duration-driven rather than constant-step,
# so the speed cap doubles as the slope of the smoothstep ramp. Peak speed
# of a smoothstep at midpoint is ~1.5× (cubic) / ~1.875× (quintic) the
# average; size this so the peak stays within the arm's comfortable range.
LIFT_AVG_SPEED_M_S: float = 0.15   # 15 cm/s average -> ~28 cm/s peak (quintic)
LIFT_MIN_DURATION_S: float = 0.4   # never shorter than this, even for tiny lifts
# Sideways move_to() arrival check: after the joint trajectory finishes,
# poll the live EE pose until it's within MOVE_ARRIVAL_TOL_M of the
# (x, y, SAFE_TRANSPORT_Z) target before returning, capped at
# MOVE_ARRIVAL_TIMEOUT_S. Prevents the next pick/place from starting
# while the arm is still settling under position-mode lag.
MOVE_ARRIVAL_TOL_M: float = 0.005      # 5 mm
MOVE_ARRIVAL_TIMEOUT_S: float = 1.0
# Time for the vertical descent leg from SAFE_TRANSPORT_Z down to the
# per-target hover_z (i.e. pose.z + HOVER_HEIGHT_M). The approach is split
# into two legs so the cup never sweeps diagonally through the workspace:
#   leg 1: (current) -> (x, y, SAFE_TRANSPORT_Z)  via MOVE_DURATION_S
#   leg 2: (x, y, SAFE_TRANSPORT_Z) -> (x, y, hover_z)  via APPROACH_DESCENT_S
APPROACH_DESCENT_S: float = 1.0
# Base-frame Z the cup tip is raised to for collision-free sideways transport.
# Must clear the source stack and both box walls.
# Set from the live suction EE Z at the handoff pose (measured 2026-06-09).
SAFE_TRANSPORT_Z: float = 1.0978

# ---------------------------------------------------------------------------
# Taught poses in base_link frame. Each entry is either:
#   (x, y, z)                          -> uses GRASP_ORIENTATION_RPY (vertical)
#   (x, y, z, roll, pitch, yaw)        -> full pose as recorded
# Capture with teach_pose.py (jog with cup held vertical, SPACE to record,
# paste the printed 6-tuple here). Leave as None until taught; run validates.
#
#   CASE_PICK     empty plastic case, top of the left/source stack
#   CASE_PLACE_R  where the case is set down in the right/target box
#   BAT_SRC_1/2   the two batteries' pick positions in the source
#                 (TODO confirm: same case beneath the removed one, or a
#                  separate designated source case?)
#   BAT_SLOT_1/2  the two battery seats inside the moved case (right box)
# ---------------------------------------------------------------------------
TAUGHT_POSES: dict[str, tuple[float, ...]] = {
    "CASE_PICK":    (0.701310,  0.529836,  0.8215, -3.141523, -0.000023,  1.920076),
    "CASE_PLACE_R": (0.776267,  0.104816,  0.7309, -3.141531,  0.000216,  1.919917),
    "BAT_SRC_1":    (0.688931,  0.389726,  0.8160, -3.141483,  0.000227,  1.919921),
    "BAT_SLOT_1":   (0.763274, -0.034724,  0.7421,  3.141523,  0.000298,  1.919760),
    "BAT_SRC_2":    (0.688875,  0.529836,  0.8160,  3.141340, -0.000154,  1.919955),
    "BAT_SLOT_2":   (0.763807,  0.106673,  0.7421,  3.141363,  0.000238,  1.920136),
}

# ---------------------------------------------------------------------------
# Forward-only stacking
# ---------------------------------------------------------------------------
# Number of forward passes to chain into a single stacking sequence. Each pass
# is offset in Z by repeat_index * Z_STEP_PER_REPEAT[label] — the source
# stack shrinks (negative src_dz) and the target stack grows (positive dst_dz)
# as repeats progress. Undo is intentionally not stack-aware; use forward-only
# when this is > 1.
FORWARD_REPEATS: int = 5

# Per-move (src_dz, dst_dz) in metres. Applied as:
#     pose.z += repeat_index * dz       (repeat_index starts at 0)
# Tune to the actual case/battery height. Defaults assume ~20 mm case stack
# pitch and ~15 mm battery stack pitch — measure and adjust.
Z_STEP_PER_REPEAT: dict[str, tuple[float, float]] = {
    # label:        (src_dz,  dst_dz)
    # Measured from pass1→pass2 deltas: src drops ~14 mm/repeat, dst rises
    # ~16–18 mm/repeat. Bias dst slightly UP (safe — cup releases a hair
    # high) and src slightly DOWN in magnitude (descent finds contact in
    # budget). Re-measure after a few repeats and tune.
    "case":         (-0.015,  0.015),
    "battery_1":    (-0.015,  0.015),
    "battery_2":    (-0.015,  0.015),
}
