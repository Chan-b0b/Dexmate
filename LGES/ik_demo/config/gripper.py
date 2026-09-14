"""Right arm: Robotiq gripper, Cognex barcode reader, barcode divert, box pick.

Everything that is NOT the left suction arm's pick-and-place: the reader
wiring and scan gates, the divert decision, the gripper's Modbus link and
the vertical box pick's geometry.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

import numpy as np

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
# How the Modbus frames reach the gripper. "ee": through the right arm's own
# RS485 end-effector connector (dexcontrol EE pass-through, needs the EE type
# UNKNOWN so right_arm.enable_ee_pass_through is True — drivers/robotiq.py).
# "usb": a USB-RS485 adapter on this computer (drivers/robotiq_usb.py, port
# below). "auto": try "ee" first, then "usb" — whichever answers the
# reset/activate handshake wins. 0903: the gripper was re-cabled from the USB
# adapter to the arm connector (no /dev/ttyUSB* on the Thor any more).
ROBOTIQ_TRANSPORT: str = "auto"
# Which arm's EE connector the gripper's RS485 cable is plugged into ("left" /
# "right"); None = try the moving arm's own connector first, then the other.
# 0903 (confirmed on the robot): the gripper is MOUNTED on the right arm but
# its cable runs to the LEFT arm's connector (right bus silent, both buses at
# 115200). Pinned so initialize() skips the silent bus (5 s timeout). Set back
# to None / "right" if the cable is ever moved.
ROBOTIQ_EE_SIDE: str | None = "left"
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
# A battery whose agreed barcode is in TARGET_BARCODES is diverted into the
# divert case (chassis_sequence._divert_case_place) instead of seated in the
# target case. Empty list = never divert.
# ---------------------------------------------------------------------------
# TARGET_BARCODES: list[str] = ["UDCG7B0289", "UDCG7B0291"]

TARGET_BARCODES: list[str] = ['UDCG7B0289', 'UDCG7B0294']
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
# Right arm (gripper.py). The divert handoff choreography (HANDOFF_* geometry,
# PLACE_LOWER_RIGHT_EE_SEQ) was removed 2026-09-03; only the EE frame remains.
# The right arm's next job is a vertical paper-box pick, mirroring the left arm.
# ---------------------------------------------------------------------------
GRIPPER_EE_FRAME: str = "R_gripper_base"
# Right-arm straight-down box pick (gripper.pick_box / box_pick.py). The EE tool
# +z points down (rpy = (pi, 0, yaw)), as the suction cup does. The Robotiq
# fingers close along the EE x axis (URDF: R_gripper_j1/j2 sit at +/-x and
# hinge about y), so the EE yaw IS the closing direction in base_link. The
# detector reports the box yaw as its LONG axis -> close across the short side
# = box yaw + pi/2.
#
# This is also the WRIST SELECTOR. The gripper is symmetric under a half turn,
# so the closing direction is a line and there are always two wrists that
# realise it; plan_box takes the one within +-90 deg of THIS value (it no
# longer tries both and picks by reach — an out-of-reach grasp moves the
# CHASSIS instead, gripper.plan_box / box_reach_offset). Flip it by pi to turn
# the wrist over. Reach is not symmetric between the two: over the run logs
# +pi/2 solved 9 of 10 logged box poses and -pi/2 solved 1 of 10, so the
# -pi/2 wrist will ask the chassis to move more often.
# 0910: -pi/2 (was +pi/2, at the user's request — verify on the robot).
BOX_GRASP_YAW_OFFSET_RAD: float = -np.pi / 2
# TILTED wall pinch (gripper._grasp_geometry). The gripper does NOT come down
# vertically: the tool is tipped this far off straight-down, ROLLED ABOUT ITS
# OWN X (the finger-closing axis, which runs across the grabbed wall), so the
# hand leans ALONG the wall while the fingers still straddle it squarely.
#
# 0911, taught by hand: the operator posed the arm until it actually gripped
# the box, moving ONLY j6 and j7 from the right arm's home. Fitting
# R0(yaw) @ Rx(tilt) to that pose gives tilt = 55 deg (6.9 deg residual) and a
# yaw within 2 deg of the one plan_box already computes — so the tilt is the
# whole difference. What it buys, all measured against the vertical grasp:
#   * reach: the vertical grasp could not reach the TRUE rim at all (the hover
#     above it missed by 7.6mm even from the best seed). Tilted, all four test
#     grasp points solve with the full 15cm hover.
#   * the approach stops swinging: home -> grasp is wrist-dominant, so the
#     fingertip wanders 14mm sideways instead of the vertical grasp's 300mm
#     out-and-back (the arm no longer has to flip the hand over: j1-j5 stay
#     within 4-8 deg of home).
#   * half the time: 5.0s against 10.1s.
#   * the depth comes out right: the hand-tuned fingertip sat 35mm under the
#     depth-measured rim, against BOX_GRASP_DEPTH_M's intended 30mm.
BOX_GRASP_TILT_DEG: float = 55.0
# IK branch reference for that grasp. The tilted pose has several solutions;
# seeded from home the solver lands 40+ deg away in j1-j5 (a different posture
# that happens to reach the same EE pose). Seeded from the taught pose it lands
# within 4-8 deg of home in j1-j5 with only j6/j7 turning — the posture the
# operator actually validated on the box. This IS that pose, as printed by
# grasp_box/joint_tune.py.
BOX_GRASP_SEED_JOINTS: tuple[float, ...] = (-1.0066, -0.6760, 0.2384,
                                            -2.2329, 0.8088, -0.7019, -0.3993)
BOX_FINGER_LENGTH_M: float = 0.14     # R_gripper_base origin -> fingertip along tool z. MEASURE on the
                                      # gripper (URDF has the finger hinges 29 mm out, no tips); TUNE
# Fingertips this far below the box RIM at the grasp — how much cardboard wall
# ends up between the jaws. 0911: 0.03 put the fingertip at z=0.713 against a
# measured rim of 0.7432, which is 14.4mm HIGHER than where the operator's
# hand-tuned grip sat (0.6986, the pose BOX_GRASP_SEED_JOINTS came from). 0.045
# lands on that hand-tuned height. The box floor is ~95mm below it, so there is
# room to go deeper if the grip still feels shallow.
BOX_GRASP_DEPTH_M: float = 0.065
BOX_HOVER_HEIGHT_M: float = 0.15      # fingertips above the box top before / after the vertical leg
# WHERE on the detected box to grip (box_pick.box_pose_from_detection): the
# box (~0.62-0.68 m long, ~0.4-0.45 m wide) is far wider than the Robotiq
# opening, so the pick pinches a WALL from above — the midpoint of the long
# wall on the robot's RIGHT (base -y): one finger inside the box, one outside.
# INSET moves the pinch point from the wall line toward the box center (m);
# 0 = fingers centered on the wall.
BOX_GRASP_EDGE_INSET_M: float = 0.0
# Extra grasp-point shift in BASE y (+ = the robot's left), applied after the
# wall midpoint is computed. Straight base-frame trim for where the fingers
# actually land on the wall, independent of the box's own frame (the INSET
# above moves along the box's short axis instead). 0910: +5 mm, on the robot.
BOX_GRASP_Y_OFFSET_M: float = 0.00
# Approach to the box hover (gripper._approach_hover). Home -> hover in ONE
# joint move sweeps the fingertips ACROSS the box interior: at home the hand
# points UP (tool z ~ +0.92 base z) and at the grasp it points DOWN, and that
# flip happens mid-flight — measured 0910 on the logged box pose, the lowest
# fingertip over the box footprint was 95 mm above the rim, at (0.982, -0.206),
# i.e. 10 cm inside the grabbed wall. Two waypoints raise that to 167 mm:
#   1. the mid-path pose (MID_FRAC of the way home->hover) LIFTED by LIFT_M,
#      keeping the orientation the path already has there;
#   2. the grasp point pushed SIDE_M straight out of the box along the grabbed
#      wall's outward normal, at hover height — so the last leg comes in from
#      OUTSIDE the wall and never crosses the interior.
# Raising the hover instead is not an option: the arm is at its reach edge over
# the box (pointing down solves only to ~20 mm above the hover), which is also
# why pre-rotating the shoulder roll does not help — it turns the hand inside
# the upper hemisphere and never gets it pointing down.
# A waypoint that does not solve is skipped (straight to the hover, the old
# behaviour) — the numbers above are for one measured box pose.
# 0911: BOTH OFF. These existed for the VERTICAL grasp, whose approach swept
# the fingertips across the box interior (95mm over the rim) because the arm
# had to flip the hand over on the way. The tilted wall pinch
# (BOX_GRASP_TILT_DEG) is wrist-dominant and its fingertip never enters the box
# footprint at all, so the detour is pure cost: with the waypoints 8.79s, with
# them off 5.69s, and the clearance is "never over the box" either way. Kept
# (rather than deleted) because they are the fallback if the tilt is ever
# turned back down toward vertical.
BOX_APPROACH_MID_FRAC: float = 0.7    # where along home->hover to take the lifted pose
BOX_APPROACH_LIFT_M: float = 0.0      # how far to lift it (0 = skip this waypoint)
BOX_APPROACH_SIDE_M: float = 0.0      # stand this far outside the wall before coming in
# Pre-position the chassis so the DETECTED box centre lands here (base_link x,
# y) before the pick runs — chassis_sequence._center_box. The tilted grasp now
# solves over a wide area, so this is not about reach: it is about the metal
# RACK the box sits on. 0911 18:19 failed with "CONTACT 8.2N 13mm above the
# grasp height" at a grasp point of y=-0.448, i.e. the fingers met the rack
# upright, not the wall; the same run then GRASPED at y=-0.285 after the
# chassis had moved. Successful box centres that day: -0.119, -0.176, -0.210;
# the failure sat at -0.279. This is the -0.119 one, the furthest from the rack.
BOX_CENTER_REF_XY: tuple[float, float] = (0.911, -0.119)
BOX_CENTER_TOL_M: float = 0.04          # |box centre - ref| accepted without moving
BOX_CENTER_MAX_MOVES: int = 3           # correction rounds, each with a re-detect
BOX_CENTER_MAX_MOVE_M: float = 0.30     # per-round clamp, each axis
BOX_GRIP_HOLD_S: float = 1.0          # hold at the top of the lift test / before a release
BOX_LIFT_TEST_M: float = 0.10         # lift-test mode: raise the gripped box this far, set it back down, release
# Box rim height (base_link z, m): the BEV detection plane and the pick's
# top_z. Now only the FIRST GUESS and the fallback — detect_box measures the
# box FLOOR from head-camera depth and rebuilds the rim as
# floor + BOX_WALL_HEIGHT_M (see _rim_from_floor_depth). 0.65 is the value the
# 0903 air tests ran at (0.60 raised by 5 cm on the user's call), and 0911
# measured it ~81 mm LOW: depth read the floor at 0.5911 (cfg.FLOOR_Z_BASE_M
# 0.600, independently established, agrees to 9 mm) and the walls tape-measure
# 14 cm, so the rim is ~0.731. At 0.650 the fingers went 93 mm down the wall
# instead of BOX_GRASP_DEPTH_M, ending 29 mm off the box floor.
BOX_RIM_Z_M: float = 0.65
# Tape-measured inside wall height, box floor -> rim (m). The rim is rebuilt
# from the MEASURED floor plus this rather than read directly, because the rim
# itself is a thin cardboard edge and its depth window straddles the outside
# wall face all the way down to whatever the box stands on: the head looks down
# 24 deg, so that near-vertical face packs ~122 mm of height into a few pixels
# (0911 rim window: 172 mm of p95-p5 spread, against 21 mm on the flat floor).
# The floor is wide, flat and reads clean, so this turns one noisy per-run
# measurement into one solid per-run measurement plus a fixed box property.
BOX_WALL_HEIGHT_M: float = 0.135
# Gates on that floor measurement — a failed gate keeps BOX_RIM_Z_M above and
# logs why. SPREAD: p95-p5 inside the depth window; the floor reads ~21 mm, so
# anything near the wall height means the window is not on the floor. DEV: how
# far the measured floor may sit from cfg.FLOOR_Z_BASE_M, which the case
# stacking model already pins for this same box — a bigger disagreement means
# the window found the contents or the rim, not the floor.
BOX_FLOOR_DEPTH_MAX_SPREAD_M: float = 0.05
BOX_FLOOR_DEPTH_MAX_DEV_M: float = 0.05
# Contact guard on the box descent: the right wrist wrench is tared at the hover
# (BOX_TARE_SAMPLES readings at rest, ~200 Hz) and the descent halts when the
# vertical force moves more than this from the baseline — fingers landing on
# the rim (rim higher than BOX_RIM_Z_M) or on the contents. Rest noise is
# ~0.2 N (0903 probe), so 7 N is well clear. The pick then lifts back to the
# hover and reports "contact" instead of closing on nothing.
BOX_CONTACT_FORCE_N: float = 7.0
BOX_TARE_SAMPLES: int = 50
# Where the descent stops hurrying: it runs fast from the hover down to this
# far ABOVE the planned grasp height, then CREEPs (DESCENT_CREEP_SPEED_M_S)
# the rest with the guard live. 0914, the user's call: 0.14. Above this point
# nothing can be touched, so speed there is free; below it the force reading
# is what the pick depends on, and a contact met at cruise is both violent and
# badly located (0.35 m/s = 3.5mm of overshoot per tick against 0.8mm at
# creep). Note BOX_HOVER_HEIGHT_M is 0.15, so at 0.14 the fast leg is the
# first 10mm and effectively the WHOLE descent creeps — 1.9s instead of 0.6s.
# Lower it (0.05 covers the 34mm rim error that made 0914's grip shallow) to
# buy that time back.
BOX_DESCENT_CREEP_FROM_M: float = 0.14
# --- seat probe (gripper.probe_seat, DIAGNOSIS ONLY) ----------------------
# The tilted wall pinch puts the PALM — the body between the two fingers —
# straight over the grabbed wall, so as the hand comes down the rim rides up
# into the open jaws and lands on the palm. That seat is the only hard
# vertical stop on the descent, and it is a fixed GRIPPER property: it does
# not care what BOX_WALL_HEIGHT_M says the rim height is, which is what went
# wrong on 0914 (wall read 105mm against the 140mm constant -> the fingers
# took only 31mm of wall instead of BOX_GRASP_DEPTH_M's 65mm and the grip was
# shallow). probe_seat descends PAST the planned grasp height at creep and
# logs vertical force against height, so the seat shows up as a measured
# curve before anything is built on it.
# Palm face (finger hinge line) along the tool z axis from the EE origin, m.
# 0.029 is the URDF hinge offset; MEASURE it on the gripper. With
# BOX_FINGER_LENGTH_M and BOX_GRASP_TILT_DEG this predicts the seat at a
# fingertip depth of (0.14-0.029)*cos(55deg) = 63.7mm under the rim, i.e.
# within 1.3mm of BOX_GRASP_DEPTH_M — the probe is there to confirm that.
BOX_PALM_ALONG_TOOL_M: float = 0.029
BOX_SEAT_PROBE_START_M: float = 0.03   # creep leg starts this far ABOVE the planned grasp height
# ... and ends this far BELOW it, so the predicted seat is in between. Capped
# by the BOX FLOOR, not by the seat: the fingertip ends (GRASP_DEPTH + this)
# under the rim, and 0914's depth put the floor 105mm under it — 0.04 would
# have parked the fingertip exactly ON the floor. 0.025 ends 90mm down, 15mm
# clear of that floor and still 26mm past the predicted 63.7mm seat.
BOX_SEAT_PROBE_EXTRA_M: float = 0.025
# Hard abort for the probe: the palm is never pressed onto the rim harder than
# this. 5N at the user's call (0914), BELOW BOX_CONTACT_FORCE_N's 7N — the
# cardboard rim is not worth leaning on to find out where it folds. So the
# probe measures the force RISE and where it starts, not the collapse: it
# either stops the moment it reads 5N (the seat depth, read off that point) or
# reaches the bottom of the window without getting there (the seat takes less
# than 5N, read off the curve).
BOX_SEAT_PROBE_BACKSTOP_N: float = 5.0
