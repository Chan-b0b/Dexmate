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
RELEASE_PRELIFT_M: float = 0.018    # 18 mm pre-release lift
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
FORCE_CONTACT_THRESHOLD_N: float = 2.0
FORCE_HARD_LIMIT_N: float = 15.0          # pick(): empty cup, low baseline noise
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
MOVE_DURATION_S: float = 2.0      # travel time to a hover pose

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
# TODO: set from the teach step.
SAFE_TRANSPORT_Z: float = 1.05  # 10 mm above max recorded hover Z (1.190 m, line 15 of taught_ee_poses.txt)

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
    "CASE_PICK":    (0.701310,  0.539836,  0.8215, -3.141523, -0.000023,  1.920076),
    "CASE_PLACE_R": (0.776267,  0.114816,  0.7309, -3.141531,  0.000216,  1.919917),
    "BAT_SRC_1":    (0.688931,  0.389726,  0.8160, -3.141483,  0.000227,  1.919921),
    "BAT_SLOT_1":   (0.763274, -0.034724,  0.7421,  3.141523,  0.000298,  1.919760),
    "BAT_SRC_2":    (0.688875,  0.564910,  0.8160,  3.141340, -0.000154,  1.919955),
    "BAT_SLOT_2":   (0.763807,  0.136673,  0.7421,  3.141363,  0.000238,  1.920136),
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
    "case":         (-0.015,  0.018),
    "battery_1":    (-0.015,  0.018),
    "battery_2":    (-0.015,  0.018),
}
