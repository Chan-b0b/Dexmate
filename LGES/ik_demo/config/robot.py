"""Robot model, IK solver budget and the arm's kinematic speed caps.

URDF + frames, IK tolerances and joint-range margins, and the ONE place
that sets motion speed (SPEED_SCALE_*, MAX_JOINT_*, MAX_EE_*).

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Robot model / IK
# ---------------------------------------------------------------------------
URDF_PATH: str = (
    "/home/dexmate/.local/lib/python3.12/site-packages/dexmate_urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"
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
JOINT_RANGE_FRAC: float = 0.92
# Joint-limit avoidance inside the live (min-motion) IK streams. min_motion pins
# the nullspace posture target to the seed, so a joint that reaches its band
# stays pinned for the rest of the leg and the QP re-routes the motion through
# the other joints in one tick (the visible elbow kink). Instead, a joint within
# MARGIN of its band edge gets its posture target nudged back inward by up to
# STEP per solve — a gentle nullspace pull (<= STEP per tick = 0.4 rad/s at
# 200 Hz) that pre-empts the pin smoothly; joints away from their limits keep
# the pure min-motion target. Either 0 disables. (arm._min_motion_target)
LIMIT_AVOID_MARGIN_RAD: float = 0.10
LIMIT_AVOID_STEP_RAD: float = 0.002

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
SPEED_SCALE_LEFT: float = 1.0        # multiplier on every cap below, left (suction) arm
SPEED_SCALE_RIGHT: float = 0.2      # multiplier on every cap below, right (gripper) arm
# (0.7 = normal; lowered for first slow handoff test)

CONTROL_HZ: float = 100.0          # motion streaming rate (set_joint_pos_vel)

# Joint-space (move_joints: cached-pose -> cached-pose travel), per joint.
MAX_JOINT_VEL: float = 2.0          # rad/s
MAX_JOINT_ACCEL: float = 5.0        # rad/s^2
MAX_JOINT_JERK: float = 30.0        # rad/s^3
# Per-tick joint-step guard for the Cartesian streams (arm.solve_step): a
# streamed leg's joint speed is capped at the SAME per-joint budget as
# move_joints (MAX_JOINT_VEL * SPEED_SCALE). A tick whose IK solution would
# exceed it has its Cartesian step shortened by the excess ratio and re-solved,
# so the leg slows where the joints can't keep up (a wrist/shoulder riding its
# band) instead of kinking. A joint step STILL over this multiple of the cap
# after shortening means the joints must move with (almost) no EE motion —
# usually a stream starting from a hover that solved a few mm / deg short of
# its target near the reach edge. That tick is rate-limited (joint step clamped
# to the cap, schedule held) so the arm converges over a few ticks; a hold
# lasting more than STREAM_CLAMP_MAX_S is a genuine singular reshuffle and the
# tick is reported unreachable (the callers' existing failure path).
STREAM_JOINT_STEP_MAX_X: float = 1.5
STREAM_CLAMP_MAX_S: float = 1.0
# Primitive boundaries start from the last COMMANDED joints instead of the
# measured ones when the two agree within this (per joint, rad): the measured
# q lags the command by the tracking error, and planning from it steps the arm
# BACK by that lag at every stop. A larger gap means the command is stale
# (E-stop, hand-guiding at an operator gate, another controller) -> live q.
CMD_CARRY_TOL_RAD: float = 0.05
# Settle wait after a planned approach, before the vertical leg starts.
# The planned stream ends at ZERO velocity but returns on its last command
# tick, so the arm is still catching up — and whatever xy it happens to be at
# becomes the descent's xy, because the vertical leg holds what it STARTS from
# (and _start_q() falls back to the live config once it disagrees with the
# command by CMD_CARRY_TOL_RAD). 0903 measured 12.9, 16.7, 26.2, 50.8 and
# 57.4mm of x on five consecutive places, all of it landing straight in the
# place position. The old hover FULL STOP used to absorb this; a blended
# junction does not, which is why it appeared when the approach became one
# continuous stream. Waiting is the fix, not putting the stop back: the motion
# already ends here, it just was not waited for.
APPROACH_SETTLE_TOL_M: float = 0.002
APPROACH_SETTLE_MAX_S: float = 1.0

# Waypoint blending (move_joints_through: multi-waypoint sequences, e.g. the
# right-arm divert place). Junctions are crossed at this fraction of vmax on
# the joints that keep direction across the junction (direction-reversing
# joints cross at 0), so the arm doesn't fully stop at every taught waypoint.
# The full fraction applies only when both adjacent hops move the joint at
# least BLEND_FULL_DIST — shorter hops scale down to avoid overshoot.
JOINT_BLEND_FRAC: float = 0.5           # 0 = stop at every waypoint (old behavior)
JOINT_BLEND_FULL_DIST_RAD: float = 0.2  # per-joint hop size for the full fraction

# Cartesian (move_ee: sensing legs), linear.
MAX_EE_LINEAR_VEL: float = 0.5     # m/s
MAX_EE_LINEAR_ACCEL: float = 1.0    # m/s^2
MAX_EE_LINEAR_JERK: float = 5.0     # m/s^3
# Cartesian, angular.
MAX_EE_ANGULAR_VEL: float = 1.0     # rad/s
MAX_EE_ANGULAR_ACCEL: float = 4.0   # rad/s^2
MAX_EE_ANGULAR_JERK: float = 20.0   # rad/s^3
