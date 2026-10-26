"""Chassis-driven detection pick&place (chassis_sequence.py).

The strafe legs and their timing compensation, bin alignment, the seed /
reference centers, per-layer place trims, the detection-driven place
recovery, the pre-flight descent reach check and the view park.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

from .sequence import FLOOR_Z_BASE_M, LAYER_PITCH_M, RUN_LOG_DIR
from .geometry import HOME_JOINTS_LEFT, SUCTION_LENGTH_M

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
HOME_LIFT_MIN_EE_Z: float = 1.1
HOME_LIFT_EE_Z: float = 1.1
CHASSIS_STRAFE_SPEED_MS: float = 0.1   # m/s magnitude (move_sideways: + left, - right)
# Speed for the LONG station<->station legs only (auto-move); small centering /
# adjust corrections stay at CHASSIS_STRAFE_SPEED_MS — at higher speeds a
# 3-10 cm move is ramp-dominated and lands poorly, while long-leg arrival error
# is absorbed by centering + the per-direction leg learning. dexcontrol clips
# to the robot's max_lin_vel (~0.5). If the carried case slips on the cup at
# higher accel (watch the placement), lower this back.
CHASSIS_LEG_SPEED_MS: float = 0.35
# Default speed for the MANUAL prompt's `l/r/f/b` moves (_manual_strafe) when no
# speed is typed. Manual only — the auto window / centering moves stay at
# CHASSIS_STRAFE_SPEED_MS. A typed speed (`r 2.0 0.15`) still overrides it, and
# a bare `r` (no distance: fixed CHASSIS_STRAFE_TIME_S) keeps the strafe speed.
CHASSIS_MANUAL_SPEED_MS: float = 0.3
CHASSIS_TURN_SPEED_RADS: float = 0.2   # rad/s magnitude for in-place yaw (turn: + ccw, - cw)
CHASSIS_STRAFE_TIME_S: float = 7.2      # seconds per leg (~distance = speed*time)
CHASSIS_SETTLE_S: float = 0.0           # settle pause after a strafe, before detecting
# 1014: 0.5 -> 0 — wheels already still ~0.15 s after the S-ramp ends, the
# 0.3 s streamed stop covers it; UNVERIFIED: watch _refine_det spread for sway
# --auto-move (chassis_sequence CLI flag): chassis legs run automatically — each
# source<->target leg is a fixed open-loop DISTANCE at CHASSIS_STRAFE_SPEED_MS
# (overrides CHASSIS_MANUAL). Station spacing is known ~0.6-0.7 m; detection
# recenters at each visit, so the leg only needs to land the case in view/reach.
# source <-> the BUFFER station to the RIGHT (the code's "target" station: the
# case stack being built), right there, left back. Was CHASSIS_AUTO_STRAFE_DIST_M
# (renamed 1012). Learned + saved across runs as "main" in CHASSIS_LEG_STATE_FILE.
BUFFER_STRAFE_DIST_M: float = 0.6
# Auto-adjust on a failed reach pre-check (auto-move): turn the chassis so the
# detected case yaw reads 0 (as before), then drive the SMALLEST forward/back +
# strafe that puts the failing pose's descent column inside reach — offsets on
# a STEP_M grid out to MAX_TRANSLATE_M (L1), nearest-first, preferring one that
# still solves REACH_MARGIN_M off in +-x/+-y so the open-loop move does not
# land on the edge of reach. Re-detect, retry, up to MAX_ATTEMPTS; then the
# keyboard prompt (immediately, if no offset within the clamp solves at all).
# 0907: replaced the drive to a taught reference point, which dragged both axes
# to it (a pose 4 cm short in x got strafed 20-30 cm in y) — see _auto_adjust.
CHASSIS_ADJUST_MAX_ATTEMPTS: int = 5
CHASSIS_ADJUST_STEP_M: float = 0.03            # offset search grid
CHASSIS_ADJUST_REACH_MARGIN_M: float = 0.02    # slack demanded around the chosen spot
CHASSIS_ADJUST_MAX_TRANSLATE_M: float = 0.30   # per-attempt clamp, each axis
CHASSIS_ADJUST_MAX_TURN_DEG: float = 30.0      # per-attempt clamp, in-place turn
CHASSIS_ADJUST_MIN_TRANSLATE_M: float = 0.01   # deadband: skip smaller translations
CHASSIS_ADJUST_MIN_TURN_DEG: float = 8.0       # deadband: skip smaller turns
# STRICT first alignment. The run's very first source centering follows the
# OPERATOR's manual park, so it is the one alignment with no leg arrival behind
# it and the one whose error every later leg inherits: run() seeds the leg
# distances from it, and the taught columns are all referenced to it. Every
# later centering only has to correct one leg's arrival, so it can live with the
# normal deadbands; this one is worth extra rounds.
# Reachable-ness: the BEV y detection spreads only 0-7mm across frames (0903
# run logs), and detected yaw is steady to ~1 deg, so the limit is the
# open-loop strafe gain (~20% by the 0903 divert residuals) — a 20mm target
# needs 2-3 rounds from a 100mm start, which is why the round budget goes up
# too. Failing to converge is not an abort: _center_case returns its last
# detection and the run proceeds.
CHASSIS_START_CENTER_TOL_M: float = 0.02      # vs CHASSIS_CENTER_TOL_M 0.05
CHASSIS_START_MIN_TURN_DEG: float = 3.0       # vs CHASSIS_ADJUST_MIN_TURN_DEG 8.0
CHASSIS_START_CENTER_MAX_MOVES: int = 8       # vs CHASSIS_CENTER_MAX_MOVES 5
# Learned leg distances (auto-move): the arrival residual after each leg
# (centering/adjust strafe, minus deliberate per-item re-alignments) feeds the
# distance of the LEG THAT JUST RAN — left (target->source) and right
# (source->target) learn independently, so a direction-dependent open-loop
# travel gain calibrates out. Saved across runs (CHASSIS_LEG_STATE_FILE below)
# and logged at run end. Each clamped to BUFFER_STRAFE_DIST_M
# +/- this; a persistent "clamped" log means the true station separation is
# outside the clamp window — fix BUFFER_STRAFE_DIST_M instead.
CHASSIS_LEG_LEARN_CLAMP_M: float = 0.20
# The same for the leg's FORWARD/BACK part (1012): the window's dx at each
# arrival is learned too (same gain) and flown inside the next strafe as one
# diagonal — +dx going LEFT, -dx going RIGHT, one number per leg. 1012 runs: the
# target / divert cases sit ~7-9cm further out than the source, so every
# arrival paid a window x move (target +9, source -7, divert +7cm). Clamped to
# +/- this around 0.
CHASSIS_LEG_X_LEARN_CLAMP_M: float = 0.15
# Gain on each arrival residual fed into a learned leg distance. Below 1.0 the
# leg AVERAGES its residuals instead of tracking the last one — at 1.0 learn()
# is a pure integrator with unity gain, so one noisy centering measurement moves
# the distance by its whole amount. That is fine where the residuals are small
# (0903 main leg, source side: +-2..5mm, distance stayed 0.498-0.505) and bad
# where they are not: the DIVERT leg measured +96, -140, -106mm and swung
# 0.600 -> 0.696 -> 0.556 -> 0.450, a 246mm spread, and it only gets 1-3
# samples per run (divert fires only on a barcode match) so there is nothing to
# average it out. At 0.3 the same residuals give a 74mm spread.
# Raise it if a leg converges too slowly; lower it if a leg chases noise.
CHASSIS_LEG_LEARN_GAIN: float = 0.3
# 1012: the learned leg distances PERSIST across runs: run() saves them here at
# its end (aborted runs too) and the next run starts from them instead of the
# config value — unless the config value they were learned against has changed
# since (the station layout moved: start fresh). The CHASSIS_LEG_LEARN_CLAMP_M
# window stays centred on the CONFIG value, so runs cannot walk it away.
# None = in-run only (the old behaviour).
CHASSIS_LEG_STATE_FILE: "str | None" = (
    None if RUN_LOG_DIR is None else f"{RUN_LOG_DIR}/chassis_legs.json")
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
# (~BUFFER_STRAFE_DIST_M) apart, so the OTHER station's stack is in the
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
DETECT_MEDIAN_SAMPLES: int = 3
# Bin-aligned divert positioning (chassis_sequence): before a divert,
# the head camera finds the divert bin (case_detection detect_bin) and the
# chassis strafes so the bin center sits at DIVERT_BIN_TARGET_Y_M in base_link
# (+left; 0.10 = 10 cm left of the robot center line). After the move the bin
# is re-detected and a residual over DIVERT_BIN_TOL_M gets ONE more
# correction. The net move is strafed back after the divert (or before the
# fallback case place) so the normal target geometry is restored.
DIVERT_BIN_TARGET_Y_M: float = -0.0293  # 1001 intrinsics fix (old-K bias kept physical, see geometry.py); was 0.0
# BEV warp plane for the bin detection (base_link z). This must be the height
# of the FACE THE LABELS TRACE, not the bin rim: case_detection/labeled_bin_bev
# traces the bin's inner BOTTOM (class 0) on a canvas warped at top_face_z(1),
# and warping one image at two planes is a homothety about the camera centre
# (case_detection/bev.py), so a plane off by dz scales the reported center
# about the camera nadir — dz here is almost pure x error, ~0.7*dz.
#
# 0904: was 0.70 (the rim), i.e. 86 mm above the labeled face, which read the
# bin 39.5 mm TOO CLOSE in x. Recovered with case_detection/calib_bin_plane.py
# from the labels' own size (the homothety scale IS the size ratio): the class-0
# face measures 664x379 mm at z=0.6138. Cross-check on the same data: class 1
# (the lid) recovers z=0.8936 against a hand-measured LID_PLANE_Z_M=0.89.
#
# Y is nearly immune (<3 mm at DIVERT_BIN_TARGET_Y_M=0): the camera sits on the
# bin's y centre line, so the parallax is radial. That is why _align_to_bin
# worked through the wrong plane — it only reads Y, and its measured gain
# divided the residual scale out. Expect that gain to sit nearer 1.0 now.
#
# HEADS UP: the seed place reads this detection's x, so with
# SEED_BIN_CENTER_OFFSET left at its pre-fix value the place point moves FORWARD
# — by 22 to 70 mm over the three recorded runs. It is a homothety about the
# camera nadir, NOT a constant: the shift grows with the bin's distance from
# that nadir, which is why hand-tuning a fixed x offset could never cancel it
# at more than one bin position. Re-tune that offset on-robot from this run;
# it is a placement bias only now.
# 1012 raised table (floor 0.60 -> 0.67): from pick_place's place_config, where it ran on the robot; was 0.6138
DIVERT_BIN_PLANE_Z_M: float = 0.6838
# The bin OBB set is 2-class (case_detection/dataset_bin/data.yaml:
# names: [bin, bin_top]); class 0 is the bin. Unfiltered, find_bin_bev returns
# whichever box scored highest — which can be the LID (see LID_CLS_ID).
BIN_CLS_ID: int = 0
DIVERT_BIN_TOL_M: float = 0.05          # accepted Y residual after the first move
DIVERT_BIN_MAX_STRAFE_M: float = 0.6    # per-move safety clamp on the align strafe
# Fallback when NO bin is detected: fixed extra rightward strafe (the original
# open-loop behavior), strafed back like the aligned move. 0.0 = stay put.
DIVERT_EXTRA_RIGHT_M: float = 0.1
# Divert-case place (chassis_sequence): a TARGET_BARCODES battery is carried
# LEFT from the source by this fixed strafe, placed into the divert case there
# (BEV-detected, suction place — slot order: first target -> left slot
# BAT_SRC_2, second -> right slot BAT_SRC_1), then the chassis strafes back
# right to the source. Both legs are open-loop (no ChassisNav learning).
#확인
# source <-> the TARGET station to the LEFT (the divert case / the first case's
# seed bin), left there, right back. Was DIVERT_CASE_STRAFE_LEFT_M (renamed
# 1012). Learned + saved across runs as "divert" in CHASSIS_LEG_STATE_FILE.
TARGET_STRAFE_DIST_M: float = 0.5   # tune on site
DIVERT_CASE_LAYERS: int = 1              # BEV warp plane: divert case stack height
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
# 1014: 0.3 -> 0.1 — the S-ramp already ends at ~0 m/s, so a lost stop has
# nothing left to coast on; UNVERIFIED: watch the chassis-watch distances.
CHASSIS_STOP_STREAM_S: float = 0.1
# Distance legs (move_chassis._drive) ramp the wheel speed up from 0 and back
# down to 0 instead of stepping it: S-shaped (cosine) ramps at this AVERAGE
# acceleration (m/s^2 of wheel speed; the peak is pi/2 of it), the drive
# lengthened by one ramp time so the commanded distance is unchanged. 0.35 m/s
# legs ramp ~1.2 s each way, 0.1 m/s moves ~0.3 s. 0 = the old stepped drive.
# 1012, UNVERIFIED on the robot: open-loop distances assume the wheels track
# the ramp — watch the arrival residuals of the first runs.
CHASSIS_ACCEL_MS2: float = 0.3
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
# completed layer, until the source is exhausted (an aborted run logs the
# values to resume with). 1012: SRC counts the CASE+BATTERY LAYERS of the source
# (the bottom case, whose batteries are the last layer, is moved on its own
# afterwards) — the count used on the robot (pick_place, chassis_sequence), and
# the one the BEV model planes and the battery-top gate are consistent with
# (a layer-k battery top reads top_face_z(k + 1)). It used to be the physical
# case count (= layers + 1), which ran one layer too few for that input.
SRC_LAYERS_REMAINING: int = 2           # source case+battery layers at run start (was 3 cases)
TGT_LAYERS_REMAINING: int = 1          # target stack height at run start
# FINAL case (run() epilogue): the layer loop moves the
# SRC_LAYERS_REMAINING layers, so ONE case (the bottom one, no batteries)
# is still in the source box. It goes through the SAME per-item cycle as every
# case above it — normal learned leg RIGHT, target case detection, aligned
# corner-seat place on the target stack, normal leg back LEFT — just with no
# batteries following it.
FINAL_CASE_TO_BIN: bool = True
# Arm joint pose that clears the head-camera view of the target while an item is
# carried during transport (TUNE; defaults to the left-arm home).
ARM_VIEW_PARK_JOINTS: tuple[float, ...] = HOME_JOINTS_LEFT
# Preferred Cartesian view-park: EE base_link (x, y, z) the held item is carried
# at — push y LEFT (+) until the head view of the target is clear; orientation is
# kept as-picked (no load rotation). Falls back to ARM_VIEW_PARK_JOINTS if
# unreachable or None. (TUNE y; z should clear both box walls like SAFE_TRANSPORT_Z.)
# Offline reach scan (reach_sweep, z=1.10): y=0.30 -> x[0.69,1.0]; y=0.40 ->
# x[0.82,1.0]; y=0.50 -> x[0.86,1.0]. Keep x inside the window for the chosen y.
# 0824 offline reach scan: at z=1.10 the y>=0.45 band is reachable ONLY at
# x 0.95-1.00 (the x=0.95 column is OK across y 0.25-0.55). Every failed
# point tried on 0824 sat in a scan-"short" cell — check park_scan before
# moving this again, and re-run it after any torso change.
# 0901 re-scan (new TORSO_JOINTS, at the as-picked yaw GRASP_YAW=1.92 +/-0.15,
# achieved-yaw-checked): the old (0.95, 0.50, 1.20) is 46mm short — every run
# fell back to the joint park (HOME, EE yaw -160deg), which is why the held
# part arrived fully rotated. y>=0.45 is no longer reachable at any z; the
# widest band-robust y is 0.40, z tracking SAFE_TRANSPORT_Z (same wall
# clearance as transport — matches the z=1.10 reach-sweep window x[0.82,1.0]
# at y=0.40). Scan with the PICK yaw, not yaw 0 — reachability differs by
# >100mm between them here. RE-VERIFY with reach_sweep after any height change.
# 0902 re-scan (GRASP_YAW changed 1.92 -> 3.1415): the 0901 window is stale —
# (0.90, 0.40, 1.10) went 32.5mm short live, falling back to the joint park
# again. Fresh reach_sweep at z=1.10: y=0.40 -> x[0.97,1.01] (narrow, kept for
# camera clearance); y=0.30 -> x[0.81,1.07]; y=0.20 -> x[0.70,1.08] (widest,
# but untested for view clearance). x=0.99 is the y=0.40 window center
# (~20mm margin each side). RE-VERIFY with reach_sweep after any further
# GRASP_YAW/torso/height change.
# 0903: x had been pulled IN to 0.80, which is the WRONG DIRECTION here — the
# binding constraint is the joint-range box, not distance, so 0.80 solves 89.6mm
# short and the Cartesian park fell back to the joint park 26/26 times across
# four runs. Restored to 0.99. The fallback cost is not just the park pose: it is
# HOME_JOINTS_LEFT, cup at pitch +18.8 deg, so the following park->transport leg
# swung the COMMANDED EE pitch 22.7 deg (vs 1.4 and 2.5 deg on the legs after it)
# while carrying the part — that was the "cup pitches near the hover" report.
# Offline re-scan agrees with the 0902 on-robot window (x[0.965,1.015] vs
# x[0.97,1.01]). 0.99 is still only 7.1mm inside the 10mm REACH_TOL_M gate at the
# worst yaw; z is what actually binds, and at z=1.05 the whole x[0.955,1.050]
# range solves to 0.0-0.4mm — worth dropping the park 50mm if it stays clear of
# the box walls there.
# 0903b: dropped z to 1.05 (== LIFT_CLEAR_EE_Z, already the case-wall-clear
# height) instead of chasing GRASP_YAW — offline scan at y=0.40 confirms the
# reachable x window widens 50mm -> 165mm ([0.910,1.075], 0.0mm err at x=0.99)
# at the SAME GRASP_YAW, and stays wide across the achieved-yaw +/-0.24 rad
# range. Pick reachability (CASE_PICK) is untouched — it stayed <=0.4mm err
# across the same yaw sweep, confirming pick was never the binding constraint.
# 1012: the raised-table TORSO_JOINTS (1.2, 2.70, 0.35) put (0.99, 0.40, 1.05)
# 62mm short — pick_place fell back to the joint park 6/6 times. At this torso
# the y=0.40 window is x[0.70,0.90] at z 1.05-1.12, x[0.75,0.85] at 1.17;
# x 0.80 sits in the middle. z = SAFE_TRANSPORT_Z so _view_park's rise is the
# only vertical move (at 1.12, ran on the robot in pick_place, it rose to 1.17
# and came back down). Offline: <=0.3mm from the as-lifted pose for CASE_PICK /
# BAT_SRC_1 / BAT_SRC_2 at EE yaw 165-195. z 1.17 UNVERIFIED on the robot.
ARM_VIEW_PARK_EE_POS: tuple[float, float, float] | None = (0.80, 0.40, 1.17)
# Pre-flight descent reachability check (chassis_sequence): before a pick/place
# moves at all, the FULL descent column at the target xy — from the current EE
# height all the way down to the BOTTOM (box floor + suction length), regardless
# of the expected layer — must solve reachable (err<=REACH_TOL_M, in limits, no
# self-collision). A near-boundary detection (e.g. x~1.03) otherwise sends the
# streamed descent through a near-singular thrash (observed: 126.8N crash).
DESCENT_CHECK_BOTTOM_EE_Z: float = FLOOR_Z_BASE_M + SUCTION_LENGTH_M  # box floor + suction length
DESCENT_CHECK_STEP_M: float = 0.02          # z step of the pre-flight sweep
# PLACES only (1013, the user's call): the column stops this far below the
# place's expected contact instead of at the box floor. Every chassis_sequence
# place releases PLACE_RELEASE_ABOVE_M above it and never goes lower; 1013
# 14:58 a layer-2 slot at (0.720,-0.149) solved 0.7mm at the place height but
# failed the floor check (10.4mm at 0.820) and blocked the place.
PLACE_CHECK_BELOW_EXPECTED_M: float = 0.03
# FIRST-case (seed) REFERENCE center (base_link x, y, z_EE, yaw). Not a blind
# place target anymore — the seed always places from a bin detection: y is
# the bin-align target, x the non-seed auto-adjust reference, z_EE the seed
# pose height.
# NOTE z is an EE z (same convention as the detected path: top_face + SUCTION
# LENGTH), NOT a top-face z — model top_face_z at the STARTING target stack
# height (TGT_LAYERS_REMAINING, before the run's own +1-per-layer stepping) +
# SUCTION_LENGTH_M. Was a hand-tuned literal (0.87, last measured seat contact
# on-robot) that had drifted from this formula (0.566 floor + 0.176 suction era)
# — RE-VERIFY the seat contact on-robot after any FLOOR_Z_BASE_M change; the
# formula tracks the model, descend-to-contact still refines the real one.
TARGET_DEFAULT_CASE_CENTER: tuple[float, float, float, float] = (
    # x was 0.9 — shifted with the 1001 intrinsics fix (see SOURCE_CASE_CENTER)
    0.88, 0.0, FLOOR_Z_BASE_M + TGT_LAYERS_REMAINING * LAYER_PITCH_M + SUCTION_LENGTH_M, 0.0,
)
# Closed-loop seed place: the place center comes from a bin detection AFTER the
# align strafe, plus this bias — the bbox-center projection reads the bin ~47mm
# FORWARD of its true center (front wall + plane mismatch; measured on-robot
# 2026-08-06, hand-centered cup vs detection). Detect fail -> the default pose.
# This is THE x/y knob for every case-into-bin place: the LAYER LOOP's seed
# place (menu 1), _place_case_in_bin (menu 4, --case-bin) and the standalone
# case_to_bin script all read it, so tuning here keeps them identical (and
# live_detect_bev draws the aim from it too). Both paths then apply the same
# CASE_CORNER_AIM_BIAS_M off the datum corner and place(corner_seat="case"),
# and neither applies PLACE_X_PLANE_TRIM_M — verified 0914, they are aim-wise
# identical. (The menu numbers used to be 3 and 2 here; --lid and --box were
# inserted ahead of them and the numbering shifted.)
# 0914: x -0.12 -> -0.09, i.e. the case lands 30mm FURTHER FORWARD (operator
# call, on-robot). UNVERIFIED at this value.
SEED_BIN_CENTER_OFFSET: tuple[float, float] = (-0.0115, 0.01)  # 1001 intrinsics fix (old-K bias kept physical, see geometry.py); was (-0.03, 0.0)
# UNUSED since 0914: the bin place (menu 4, one case -> bin) corner-seats like
# every other case place, so it aims at SEED_BIN_CENTER_OFFSET + the corner bias
# and the walls finish the job. Kept for the measurements below, which is what a
# plain no-corner-drive aim would need again.
# What it was for — a PLAIN bin place: the case
# lands where it AIMS, there is no corner drive after it. The seed place aims
# at center + SEED_BIN_CENTER_OFFSET shifted CASE_CORNER_AIM_BIAS_M (30mm)
# AWAY from the datum corner (+x, -y), and its corner drive then registers only
# 6-13mm back toward the corner (0907 13:59-14:02 seeds), so aiming the plain
# place at center + SEED_BIN_CENTER_OFFSET put the case ~25mm nearer and ~25mm
# further left than the seed case sits (0907: "close in x, against the left
# wall in y"). Start where the seed case actually ends up: SEED at the time
# (-0.05,+0.05) + (+0.03,-0.03) aim bias - (~0.01,~0.01) drive. Tune on-site:
# compare the "[...] bin center (x,y) -> place (x,y)" log line with where the
# case sat; SEED_BIN_CENTER_OFFSET is NOT touched by that tuning.
BIN_PLACE_CENTER_OFFSET: tuple[float, float] = (0.0, 0.0)
# Lower bound on the DETECTED bin y for a case place into the bin
# (_place_case_in_bin, task 2). The bin place has no y alignment of its own —
# the chassis is positioned externally — so the only thing that used to object
# to a badly-parked bin was the IK reach pre-check, and that passes far past
# the useful window: 0914 14:11 placed from a bin detected at y=-0.159, i.e.
# the case's seat point 109mm right of the centre line and the whole descent
# flown across the body. Below this the chassis strafes RIGHT to bring the bin
# back up to it and re-detects (up to CHASSIS_ADJUST_MAX_ATTEMPTS rounds).
# -0.05 is the case's own centering ref: a case CENTER at this y puts its
# grab/seat point (the +0.05 case-frame offset) on the robot centre line, so
# the floor says "the seat point may be left of centre, never right of it".
BIN_PLACE_MIN_Y_M: float = -0.05
# Seed bin search: a failed detection walks these chassis strafes
# (+left/-right, m) between re-detects — alternating outward from the start —
# before handing the operator the keyboard. There is NO blind default-pose
# place: the seed always places from a detection, trusted AS-IS (no deviation
# gates — the reach pre-check and the corner-seat wall-latch release
# precondition catch a bad aim).
SEED_BIN_SEARCH_STRAFES_M: tuple[float, ...] = (+0.05, -0.10, +0.15, -0.20, +0.25)
# Seed bin re-detection frames: up to this many fresh frames per _detect_bin_xy
# call, combined by per-axis MEDIAN over the successful ones (mirrors
# DETECT_MEDIAN_SAMPLES for cases) — a single missed/jittery frame no longer
# drops the seed place to the default-pose fallback. Every frame missing ->
# None (fallback as before). 1 = old single-shot behavior.
SEED_BIN_DETECT_N: int = 3
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
PLACE_X_LAYER_TRIM_M: float = 0.000   # was 0.008 for the MODEL-plane regime; with the
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
PLACE_X_PLANE_TRIM_M: float = -0.01
# Constant yaw trim (rad, base CCW+) on TARGET place wrists: the part arrives
# systematically twisted on the cup (unobservable — the system cnever sees the
# battery; 0805: both batteries landed ~2deg twisted, the empty case conformed
# to battery_1's angle [target det yaw +0.2deg -> -1.6deg after its seat], the
# loaded case couldn't for battery_2 -> jam at +26mm). This pre-rotates the
# wrist to land the part square. +0.031 (~+1.8deg) cancels the 0805 estimate;
# if a test run doubles the twist instead, flip the sign. 0.0 disables.
PLACE_YAW_TRIM_RAD: float = 0.01 #낮출수록 CW
# Post-place verification (CASE places): after the release, park the arm to
# clear the head view and re-detect the just-placed case BEFORE the chassis
# moves — landed-vs-intended (dx, dy, dyaw) in one base frame (CSV
# place_chk_x/y/yaw). Case places align to the layer BELOW, so a constant
# per-place bias accumulates layer over layer (0805: L2 +1.4mm, L3 barely in,
# L4 misseat +11.5 — "a bit left" each layer); this measures that bias
# directly instead of guessing trims. Costs one sync view-park + detection
# per case place (the park loses its overlap with the return strafe).
# 1013: OFF (user) — log-only since the 1012 vision-only place dropped the
# trims; the return leg gets its parallel park back (~2.2 s per case place).
PLACE_VERIFY_DETECT: bool = False
# Misseat contact recovery (suction.place): on a rim-landing, keep the part
# held, lift PLACE_RECOVER_LIFT_M above the failed contact, re-orient the
# wrist yaw by the next pattern step (offsets from the commanded place yaw),
# and creep back down; success = a contact inside the seat band. Absorbs the
# staging-dependent in-hand twist that constant yaw trims can't track (0805
# tuned trim broke on the 0806 restage). Every place contact also snapshots
# the tared 6-axis wrench + commanded-vs-measured EE yaw (CSV contact_wrench /
# contact_yaw) — Phase 2 uses the mz sign to pick the search direction first.
PLACE_RECOVER_ATTEMPTS: int = 6
PLACE_RECOVER_YAW_PATTERN_RAD: tuple[float, ...] = (0.026, -0.026, 0.052, -0.052)
PLACE_RECOVER_LIFT_M: float = 0.010         # re-orient height above the failed contact
# Phase 2 — force-guided translation: on a misseat contact with a tared lateral
# force of at least FORCE_MIN, the obstruction is pushing the part toward the
# free side (0806 L4 bat1: fx=-5.8N and the operator's verdict was "move -x"),
# so step XY along that force direction (yaw kept) instead of the next yaw
# step. Total XY excursion from the commanded pose is capped by XY_MAX.
PLACE_RECOVER_XY_STEP_M: float = 0.005
PLACE_RECOVER_FORCE_MIN_N: float = 1.0  # was 3.0 — a FLAT landing (part on top of a
                                        # divider) converts almost no press into
                                        # lateral push, so weak-but-real signals
                                        # were filtered and recovery went yaw-only
                                        # (0806 bat2). Noise floor ~0.5N.
PLACE_RECOVER_XY_MAX_M: float = 0.02   # was 0.009 — TOTAL displacement cap; with a
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
    (0.01, 0.0), (0.02, 0.0), (-0.01, 0.0), (-0.02, 0.0),
    (0.01, -0.01), (0.01, +0.01))


# ---------------------------------------------------------------------------
# 1012: case height from the BATTERY TOPS (ported from pick_place --obb-depth,
# robot-checked there on same-spot 3->2 and 2->1 layer case pick/place)
# ---------------------------------------------------------------------------
# The case is transparent: the depth of its face read anywhere in a ~15mm range
# frame to frame (a battery's top ~14mm proud of the face, the far wall's inner
# face, nothing at all), and a plane that high pulls the BEV detection toward
# the robot (~0.5mm of x per mm of plane). The batteries are opaque and read
# 0.711-0.713 in every saved frame. So: inside the detected case box, only the
# inner OBB_DEPTH_INNER_FRAC (the battery slots) is read (>= OBB_DEPTH_MIN_PX
# valid depth pixels), its pixels go to base z, and the battery top is their
# OBB_DEPTH_PCT percentile minus BATTERY_TOP_PCT_BIAS_M. (1012: was the highest
# cluster holding >= 15% of the pixels — a lone battery, half the slots empty,
# reads a sloped top that straddled that bar: 0.6918-0.7053 over 3 frames, one
# miss. p90 over the 37 saved frames: 1 battery 3.4mm spread vs 13.5, 2
# batteries 6.8 vs 13; it reads 2.9 +-1.2mm above the cup contact, hence the
# bias, which keeps the constants below calibrated as they were.) The batteries seen are always the LOWER case's (an empty case
# is picked off a battery case; a case is placed onto one), so
#   BEV plane (xy) = battery top - BATTERY_TOP_ABOVE_FACE_M  (+1 LAYER_PITCH_M at
#                    the source: the empty case nested on them)
#   expected z     = battery top + SUCTION_LENGTH_M  (the empty case rests ON the
#                    batteries: place contact == pick contact within 2mm)
# A battery top outside +-BATTERY_TOP_GATE_M of top_face_z(level + 1) (where
# every frame read it; level = the battery case's layer: the source count, or
# the target count - 1 because the first layer's case goes to the seed bin while
# the counter steps) means no battery was seen (bottom case, empty slots) ->
# no plane, the operator is handed the keyboard. The re-detect on the measured
# plane may only MOVE the same case, by tens of mm: a jump past
# OBB_DEPTH_MAX_SHIFT_M is the detector picking a different box.
OBB_DEPTH_MIN_PX: int = 300
OBB_DEPTH_PCT: float = 90.0
BATTERY_TOP_PCT_BIAS_M: float = 0.003       # p90 - cup contact: 2.9 +-1.2mm (11 picks)
OBB_DEPTH_INNER_FRAC: float = 0.7
OBB_DEPTH_MAX_SHIFT_M: float = 0.08
BATTERY_TOP_ABOVE_FACE_M: float = 0.016     # 4 target frames: 15.6-17.1mm
BATTERY_TOP_GATE_M: float = 0.020           # 7 frames + 6 runs: within +-5mm; 1013: 10->20 (a +11.0mm read tripped it)
# Every battery-top read is saved (frame, joints, box, result) under
# RUN_LOG_DIR/obb_frames/<run stamp>/ for offline replay. False = off.
OBB_DEPTH_SAVE_FRAMES: bool = True
# 1015, LOG ONLY (changes no motion): on every arrival at an orange bin (target,
# seed / divert bin) save the right wrist photo + head RGB + depth + joints under
# RUN_LOG_DIR/wrist_watch/<run stamp>/, so bin_pose/analyze_wrist_watch.py can
# score the wrist-photo bin tracking (no-view-park study) against the head depth
# on real runs. Needs the right arm at home (= its wrist-camera view pose).
WRIST_WATCH_SAVE_FRAMES: bool = True

# Case / battery places go STRAIGHT down onto the detection (no corner aim bias,
# no lateral drive, no trims, no misseat recovery): per-tick from the hover at
# the corner seat's vertical profile, stop at the first contact force. With this
# set, the descent instead stops this far ABOVE the expected z and releases
# there with no prelift (a contact before it = "early_contact", held for the
# operator). None = the force-stop place. NOTE place contacts ran 8-11mm BELOW
# the battery-top expected z in pick_place, so 0.005 releases ~13-16mm up.
PLACE_RELEASE_ABOVE_M: "float | None" = 0.02
PLACE_RELEASE_SETTLE_S: float = 0.0         # hold at the release height first

# Manual-mode (not --auto) chassis window: the detected case centre (base x, y)
# must land inside it before a pick / place; outside, the chassis drives the
# case to the window CENTRE (both axes, one diagonal move), re-detects, and
# repeats up to WINDOW_MAX_MOVES times before handing the operator the keyboard.
# (centre, half-width) in m; None = no window on that axis. --auto keeps its own
# centering (_center_case).
WINDOW_X_M: "tuple[float, float] | None" = (0.72, 0.05)
WINDOW_Y_M: "tuple[float, float] | None" = (0.0, 0.05)
WINDOW_MAX_MOVES: int = 4
