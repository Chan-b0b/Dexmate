"""Standing-object show (stand_place.py): pick standing cylinders from the desk
with the right Robotiq gripper, approaching horizontally from behind, and lay
them in the black bin.

One domain of the ik_demo configuration, split out of gripper.py 2026-09-06 on
the user's request. Consumers import the FACADE (config/__init__.py) —
``from . import config as cfg`` — which re-exports every name here.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Standing-object pick + lay-down place (stand_place.py), 2026-09-06.
# Two objects stand on a desk in front of the robot: a small cylinder and a
# taller rectangular block. The right gripper approaches each HORIZONTALLY from
# behind (fingers pointing +x, closing along base y), pinches it at mid-height,
# lifts, turns the wrist 90 deg about base y so the object lies along x with the
# tool pointing down, then lowers it into the black bin like the box descent.
# ---------------------------------------------------------------------------
# Right-arm home for this task (user-taught 0906): gripper already faces forward
# at (0.88, -0.41, 1.01), so the approach legs start from the right orientation.
STAND_HOME_JOINTS_RIGHT: tuple[float, ...] = (-1.1110, -0.5930, 0.5064, -2.2317, 0.8302, 0.0725, 0.0830)
# Desk top, base_link z. measure_floor_z.py 0906: 0.7152 / 0.7158 (81/81 pts, 2 mm).
STAND_DESK_Z_M: float = 0.716
# Per-class geometry (m). grasp_h: EE (R_gripper_base) height above the desk at
# the pinch — the heights the user flew by hand with pose_tune on 0906 (EE 85 mm
# and 58 mm above the desk; the sideways gripper body cleared the desk at 58).
# lying_half: half the object thickness once it lies down (sets the place
# height). grip_pos: Robotiq position (0..255) the fingers stopped at on the
# object in that session — reference for "did we close on the right thing".
STAND_OBJECTS: dict[str, dict[str, float]] = {
    "rectangle": {"footprint": 0.05, "height": 0.15, "grasp_h": 0.085, "lying_half": 0.025, "grip_pos": 128},
    "cylinder":  {"footprint": 0.03, "height": 0.07, "grasp_h": 0.058, "lying_half": 0.015, "grip_pos": 155},
}
# Tool z (fingers) = +x, fingers close along base y  ->  rpy (-90, 0, -90) deg.
# Yawed approaches rotate this about base z. The user's home pose FKs to
# (-1.36, 0.05, -1.46), i.e. this orientation within a few degrees.
STAND_APPROACH_RPY: tuple[float, float, float] = (-np.pi / 2, 0.0, -np.pi / 2)
# Tool down, fingers closing along y: (180, 0, +90) deg. The half-turn twin
# (180, 0, -90) would make the turn from the approach a pure +90 deg about base
# y, but the right wrist cannot reach it with the tool down anywhere over the
# desk (0906 offline scan: 50-300 mm short), while +90 solves for x 0.7-0.8,
# y -0.15..+0.15, z 0.86-1.03. So the in-air turn is the shortest rotation
# between the two (180 deg about the axis (1,0,-1)/sqrt2): the object tilts
# sideways on the way and ends lying along x with its top toward the robot.
STAND_PLACE_RPY: tuple[float, float, float] = (np.pi, 0.0, np.pi / 2)
# R_gripper_base origin -> fingertip along tool z, MEASURED on the robot 0906
# (stand_place --measure-fingers: closed fingertips on the desk at EE z 0.8008,
# desk 0.716). The box pick's BOX_FINGER_LENGTH_M guess of 0.14 is 5.5 cm long.
STAND_FINGER_LENGTH_M: float = 0.085
# Object centre this far inside the fingertips at the pinch. 0.012 puts the
# 3 cm cylinder's underside 3 mm BELOW the fingertips, so on the place descent
# the cylinder meets the bin floor first and is released resting on it (at
# 0.02 the tips landed first and the cylinder dropped the remaining gap, by an
# amount that varied with how deep it had been gripped -> ~1 cm x scatter).
STAND_PAD_DEPTH_M: float = 0.012
STAND_STANDOFF_M: float = 0.08         # straight-line entry starts this far behind the pinch pose
STAND_PRE_LIFT_M: float = 0.15         # joint-space leg lands this far above the standoff; short column down
STAND_APPROACH_SPEED_M_S: float = 0.04 # entry speed (descent creep)
STAND_CONTACT_N: float = 5.0           # entry / touchdown guard on the tared wrist wrench
STAND_ROTATE_STEPS: int = 6            # IK waypoints along the in-air turn
# Joint-speed scale for the in-air turn ONLY (None = the run's --speed). 0906:
# the first-placed cylinder always landed ~1 cm toward the robot, less at
# --speed 0.3 than 0.5, with pick depth / slot / order / drop all ruled out ->
# the cylinder slides along its axis in the pads during the turn. Slow the turn.
STAND_TURN_SPEED_SCALE: float | None = None   # 0.15 tried 0906: no effect on the x offset
# Black bin. Inner floor from the 0906 touchdown (cylinder met the floor
# 12 mm above a desk+5 mm guess -> desk+17 mm). The wall height is NOT known
# ("high", the user says); the transport height is therefore a value proven on
# the robot rather than rim + finger + clearance: 1.026 m carried the upright
# cylinder and the laid one over the wall three times without touching. Lower
# it (more reach to the left) only once the rim is measured.
# 0906 17:05 (user's call): three re-gripped placements touched at EE z
# 0.8332-0.8335 with the cylinder 1.2 cm inside the tips -> floor = 0.8335 -
# 0.015 - 0.073 = 0.7455 = desk + 3.0 cm. The touchdown guard still decides.
STAND_BIN_FLOOR_Z_M: float = STAND_DESK_Z_M + 0.030
STAND_TRANSPORT_Z_M: float = 1.026
# Where the bin should sit in base_link for the most reach margin (0906 offline
# scan of the whole place chain — carry, in-air turn, descent — on a 2.5 cm
# grid, cylinder picked at (0.98,-0.28)): centre (0.925, -0.125) is 14.6 cm from
# the nearest failing slot in every direction, and slots from y -0.35 to +0.15
# solve at that x. The chassis is to be driven so the DETECTED bin lands near
# this; the place slots are then laid out around it.
STAND_BIN_TARGET_BASE_XY: tuple[float, float] = (0.925, -0.125)
# Slot offset from the bin centre along y. 0906: the job became ONE cylinder for
# now (up to four later), so it goes to the centre; 0.05 was the two-object
# left / right layout.
STAND_PLACE_Y_OFFSET_M: float = 0.0
# Aim this far BELOW the computed resting height; the 5 N touchdown guard stops
# the descent where the object actually meets the floor. 0906 16:29: at 5 mm
# the guard never fired on either placement, so the cylinder was released in
# the air and the deeper-held one fell further and skidded in x. The resting
# height is uncertain by the pinch depth (+/- 3 cm, until the reach constant is
# calibrated) plus the guessed bin floor, so aim well below and let the guard
# find the floor.
STAND_PLACE_OVERSHOOT_M: float = 0.03
# The place descent is two legs: normal speed down to resting + SLOW_FROM, then a
# short leg (SLOW_FROM + OVERSHOOT long) whose creep-in band covers the whole
# range where the floor can actually be.
STAND_PLACE_SLOW_FROM_M: float = 0.04
STAND_PLACE_CREEP_M_S: float = 0.03       # constant speed of that last leg (tracking lag ~1 mm)
STAND_LIFT_TEST_M: float = 0.10        # --pick-only: lift this far, set back down, release
# Soft grip (stand_place.soft_close, ported from grasp_box/pose_tune.py): the
# Robotiq force controller squeezes ~20 N even at force 0, so instead stream a
# slow close, watch the motor current (gCU, ~10 mA/count) and freeze the
# position target the moment both fingers load up. Values measured on this
# gripper (grasp_box/config.py); re-tune CU_STOP per object.
STAND_SOFT_GRIP_SPEED: int = 0x20
STAND_SOFT_GRIP_FORCE: int = 0
# Contact by CURRENT: gCU (~10 mA/count) at or above CU_STOP continuously for
# CU_HOLD_S. Contact by STALL: the finger position has not advanced for
# STALL_S. Either counts only once the fingers have travelled MIN_TRAVEL
# counts from where they started (start-up inrush / stale status excluded).
# 0906 trace (run 155127): free-running current reads 0-4 counts, contact 5,
# and the gripper's own force-0 controller stops the fingers ~0.2 s after
# contact anyway (gOBJ 2, hold at the object's ~20 N minimum). So CU_STOP sits
# above the free-run noise — 3 fired early on a 4-count spike — and in
# practice the gripper's own stop is the contact signal.
# 0906 16:24: even at 6 the current fired on a GRAZE (finger side brushing the
# cylinder at gPO 155-166, then closing freely to 191) -> disabled (255 is
# never reached). Contact = the gripper's own stop, or the finger stall.
STAND_SOFT_GRIP_CU_STOP: int = 255
STAND_SOFT_GRIP_CU_HOLD_S: float = 0.15
STAND_SOFT_GRIP_STALL_S: float = 0.25
STAND_SOFT_GRIP_MIN_TRAVEL: int = 10
# Counts commanded PAST the contact position once the fingers have touched.
# 1 = pose_tune's "elastic squeeze only" (the cylinder could still turn in the
# pads). 0906 the user asked for a firmer hold: 25 takes the 3 cm cylinder from
# ~155 at contact to ~180; force stays 0, so the gripper's own controller caps
# the squeeze at its ~20 N minimum.
STAND_SOFT_GRIP_SQUEEZE: int = 25
STAND_SOFT_GRIP_TIMEOUT_S: float = 6.0
# Grasp check (stand_place.grasp_with_check): the finger position at CONTACT must
# be within +/- TOL of the object's grip_pos (cylinder: 155 -> 140..170), else
# the grip is re-tried once after shifting SHIFT along the entry line (back when
# the object stopped the fingers late = deep toward the palm; forward when the
# fingers closed on nothing).
STAND_GRIP_POS_TOL: int = 15
STAND_REGRIP_SHIFT_M: float = 0.028
# Release in the bin: open only this many counts past the grip position (about
# 0.33 mm/count on the 2F-85) — enough for the object to drop free, not a full
# open next to the bin wall. Full open happens before the next pick. 0906: 30
# (~1 cm) still looked like a wide open on the robot -> 12 (~4 mm).
STAND_RELEASE_OPEN_COUNTS: int = 12

# ---------------------------------------------------------------------------
# Detection (show_detect.py) + chassis alignment (stand_place --detect --chassis)
# ---------------------------------------------------------------------------
# Both OBB models (runs/obb/cylinder, runs/obb/show = the black bin) were
# trained on BEV frames captured 0906 at head angle 30 (joint -5.3 deg), torso
# TORSO_JOINTS, warped at capture_bev's default plane top_face_z(1) = 0.6138.
# Run-time detection warps at the SAME plane; centres are then moved to the
# height of the feature (bev.reproject_plane).
STAND_HEAD_ANGLE_DEG: float = 30.0
STAND_DET_PLANE_Z_M: float = 0.6138
STAND_CYL_WEIGHTS: str = "runs/obb/cylinder/weights/best.pt"   # relative to case_detection/
STAND_BIN_WEIGHTS: str = "runs/obb/show/weights/best.pt"
STAND_DET_CONF: float = 0.40             # YOLO predict threshold (both models)
STAND_CYL_CONF: float = 0.70             # keep only confident cylinders (standing ones score 0.89+ in training frames)
# Bin rim height (base z): the bin OBB is its rim, so the detected centre is
# moved to THIS plane. TODO measure — "high", the user says; guess 15 cm wall.
STAND_BIN_RIM_Z_M: float = STAND_DESK_Z_M + 0.15
# Calibration offsets added to the reprojected centres (m). Start at 0; set
# from the first --detect --dry runs against the hand-verified cylinder spot
# (pinch EE 0.860 -> cylinder ~(0.933,-0.28)).
# 0906 17:41 (first --detect --chassis run): detected cylinder x 0.942, the
# fingers closed deep (207) at the nominal pinch and clean (154) 2.8 cm back
# -> the detector reads the standing cylinder ~3 cm too far. One sample; watch
# whether the first close lands in band on the next runs.
STAND_CYL_DET_OFFSET_XY: tuple[float, float] = (-0.03, 0.0)
STAND_BIN_DET_OFFSET_XY: tuple[float, float] = (0.0, 0.0)
# Where the bin must sit in base_link before placing: the spot every 0906 place
# ran at. --chassis drives the base so the DETECTED bin lands here (forward by
# bin_x - this_x, left by bin_y - this_y); the slots are laid out around it.
STAND_BIN_PLACE_XY: tuple[float, float] = STAND_BIN_TARGET_BASE_XY   # (0.925, -0.125), user's call 0906
# Cylinder pick side. Offline scan 0906 (whole pick + place chain, 2.5 cm
# grid): every cylinder spot with x 0.85..1.125 and y -0.55..+0.075 solves.
# The window below is the user's choice (0906): x with ~2 cm margin, y from
# 15 cm right to 5 cm left of the body centre. A detected cylinder OUTSIDE it
# is first brought to the window's MIDPOINT by the chassis.
STAND_CYL_PICK_WINDOW: tuple[tuple[float, float], tuple[float, float]] = ((0.87, 1.10), (-0.15, 0.05))
STAND_CHASSIS_SPEED_MS: float = 0.05     # slow, as asked (strafe/forward legs, m/s)
STAND_CHASSIS_MIN_M: float = 0.01        # deadband per axis
STAND_CHASSIS_MAX_M: float = 0.60        # per-cycle safety clamp per axis
# After the chassis has brought the bin to the place spot, the bin is detected
# once more (arm above it, holding the cylinder) and the place point is shifted
# by the residual unless it is absurd (larger than this; the place spot has
# ~14 cm of reach margin all round, so 10 cm shifts still solve).
STAND_PLACE_REFINE_MAX_M: float = 0.10
