"""--lid: take a box lid off with the cup, stack it on the lid on the floor.

Both chassis legs are hand-driven, so nothing here describes a path: only
the two detections (lid on the box, lid on the floor), the grab offset,
and the place stance + heights.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

from .chassis import DIVERT_BIN_PLANE_Z_M
from .geometry import SAFE_TRANSPORT_Z, SUCTION_LENGTH_M
from .gripper import BOX_RIM_Z_M

# ---------------------------------------------------------------------------
# --lid: take the LID off a box, then stack it on the lid lying on the floor
# ---------------------------------------------------------------------------
# The bin OBB set is 2-class (case_detection/dataset_bin/data.yaml:
# names: [bin, bin_top]), so the lid is class 1. Unfiltered, find_bin_bev
# returns whichever box scored highest — which can be the bin itself.
LID_CLS_ID: int = 1
# BEV warp plane for the lid detection = the LID's OWN top face, since that is
# the surface whose OBB center maps linearly to base xy (and the cup lands on
# it, so it doubles as the expected contact face: expected EE z = this +
# SUCTION_LENGTH_M). Get it wrong and the detected xy carries a projection bias
# AND the creep line starts at the wrong height.
#
# 0904: was seeded at DIVERT_BIN_PLANE_Z_M (0.70) = 198mm too LOW, so the
# planned approach drove the cup into the lid 15cm above the creep line
# (operator E-stopped it). Recovered from that run's last two planned-leg
# traces: the live EE was 1.1161 then 1.0527 (commanded z + track_z), i.e.
# descending 0.254 m/s, and the cup tip sits SUCTION_LENGTH_M*cos(9deg) below
# the EE — so the lid top is between 0.836 (a full trace interval later) and
# 0.900 (right at the last sample). The E-stop lands AFTER the touch, so the
# touch is nearer the top of that range. 0.89 was the working value, with a
# RE-MEASURE note attached; above 0.895 the creep line passes SAFE_TRANSPORT_Z
# and the approach has nowhere to come from, so that was the ceiling.
#
# 0906 RE-MEASURED, from the depth stream instead of jog_ee: 17 detections
# across 0904/0905/0906 put the lid's top face at 0.7911-0.8045 (median
# 0.7982), a 13mm spread over three days. 0.89 was 92mm HIGH.
#
# Why depth is believed over the two estimates that produced 0.89 (the E-stop
# reconstruction above, and a 0.8936 cross-check from the labels' own size):
# the picks that SEAL do so with depth reading ~0.798 and the ~92mm
# reprojection that reading drives already applied. If depth were the wrong
# one, that reprojection would be aiming the cup ~100mm off the lid and
# nothing would ever stick. Two outlier reads (0.559, 0.575 on 0906 12:56;
# 0.820 with a 59mm relief, i.e. the window straddling an edge) are excluded.
#
# Almost nothing reads this as a HEIGHT any more — cfg.LID_DEPTH_REFINE
# measures the surface per frame and _lid_pick_aim takes that. It is the BEV
# WARP plane, plus the fallback height for when depth returns nothing. Both
# improve here: the reprojection shrinks from ~92mm to ~0, and the warp moves
# CLOSER to the plane the detector was trained at (all 320 dataset_bin/train
# images were captured at L1 = 0.6138), so the lid's apparent scale is nearer
# what the model saw.
LID_PLANE_Z_M: float = 0.798
# Cup grab point as an offset from the detected LID center, in the LID's own
# frame (rotated by its yaw, exactly like CASE_GRAB_OFFSET).
LID_GRAB_OFFSET_M: tuple[float, float] = (-0.05, 0.020)


def canonical_lid_yaw(yaw_deg: float) -> float:
    """Fold an OBB long-axis yaw into [-90, +90) — the branch nearest 0.

    The OBB reports the long axis, which is 180-deg ambiguous, and the two
    branches are NOT interchangeable downstream: LID_GRAB_OFFSET_M is rotated by
    this yaw, so a flip MIRRORS the cup's grab point about the lid centre. At
    the taught (-50,+20)mm that is a 100mm swing in x and a 44mm one in y.

    It showed up as run-to-run drift — the same physical lid read 175.9 deg on
    0905 and 1.5 deg on 0906, so the cup aimed 48mm IN FRONT of the lid centre
    on one run and 50mm BEHIND it on the other. Both consequences were observed:
    0905's far branch put the pick at x 1.022, where the IK went singular and
    the pick died 'unreachable' (the folded value is 0.926); and a pick and a
    place that land on different branches aim at MIRRORED points of their lids,
    which drops the held lid ~100mm off the one on the floor.

    Folding at the source fixes both, because every consumer then reads one
    branch — the pick wrist and aim, the place aim, _center_lid's turn, the
    depth sampling window, lid_probe. The near-0 branch is also the
    reach-friendly one here: it puts the cup BEHIND the lid centre, and it is
    large x that the pick column runs out of.

    Ambiguous only near +-90 deg, where the two branches are equidistant. The
    lid is nominally axis-aligned and every detection logged so far sits within
    5 deg of 0 or 180, far from that boundary.

    Lives in the config rather than next to its caller because there are two
    of them and they must not drift: chassis_sequence._detect_lid_xy folds
    every detection here, and case_detection/live_detect_bev.py's ee_target
    folds the same way so the red X it draws is the point the run will
    actually fly. That viewer already imports this package as ikcfg."""
    return float((float(yaw_deg) + 90.0) % 180.0 - 90.0)


# Chassis alignment target for the lid PICK (cup point, base_link x/y): after
# the operator parks, the chassis turns the lid square and drives until the cup
# point sits here, the same idea as _center_case's refs. Deadbands/rounds reuse
# the strict start-alignment constants (CHASSIS_START_*).
#
# Chosen from an offline scan of the pick column (1.095 -> 1.045, the lid being
# nearly at the reach ceiling), per wrist yaw: at y=+0.05 the reachable x runs
# [0.70,1.06] for every yaw the carry park allows except 260 deg, while at
# y<=0.00 the window collapses (wrist 240 deg: nothing at all, wrist 220 deg:
# x[0.86,0.88]). x=0.85 sits mid-window for all of them. The operator cannot
# park to that accuracy by eye, and being 10cm off in the wrong direction here
# is the difference between a comfortable pick and no solution at all.
LID_CENTER_XY_M: tuple[float, float] = (0.85, 0.05)
# Hard floor on how close the alignment may bring the cup point in x at the
# PICK — the counterpart of LID_PLACE_MIN_X_M, 10mm under the reference above so
# an open-loop leg that overshoots forward cannot close on the box. The pick
# column is wide open here (x[0.70,1.06] at y=+0.05 for every wrist yaw), so
# the floor costs no reach; it exists because the box is what the robot would
# hit, not because IK runs out.
LID_CENTER_MIN_X_M: float = 0.84
# Where the lid waits at the UNLOAD station: sent as JOINT ANGLES, not a
# Cartesian pose, and moved to WHILE the torso leans (see run_lid). Joints
# because the pose is only meaningful at LID_PLACE_TORSO_DEG and only on the
# elbow branch it was taught on — an IK solve could land on another one, and
# during a simultaneous torso move there is no fixed base to aim a Cartesian
# target at anyway. Operator-taught.
#
# At LID_PLACE_TORSO_DEG this is EE (0.8496, 0.6112, 0.5799), rpy (-3.0223,
# 0.2286, -2.3722) — verified against the robot's own reading to 0.0mm / 0.00
# deg. Cup NOT vertical there (13 deg of pitch); the lid's underside rides at
# ~0.43, well clear of the floor lid at LID_FLOOR_PLANE_Z_M. Kept exactly as
# measured: j7 sits 0.4 deg outside the IK joint band (the robot's own limit is
# wider than JOINT_RANGE_FRAC allows), and run_lid clips it in with
# ArmMover.clip_to_band so the model and the command agree — 0.4 deg on the
# wrist's last axis moves the cup by nothing.
LID_UNLOAD_STOW_JOINTS: tuple[float, ...] = (
    -2.2551, 1.3992, 1.9901, -1.3383, -0.1979, 0.6648, -1.2849)
# Right arm during the same stow: parked out of the way while the left arm
# holds the lid and the torso leans to LID_PLACE_TORSO_DEG. Live-read off the
# robot at that stance (0904), same taught-joints reasoning as the left side —
# meaningful only at LID_PLACE_TORSO_DEG and on the elbow branch it was read on.
LID_UNLOAD_RIGHT_JOINTS: tuple[float, ...] = (
    -1.0062, -0.6403, 0.7091, -1.8360, 0.8074, 1.0878, 0.0514)
# --- unload side ------------------------------------------------------------
# The operator hand-drives the chassis from the picked-lid box to the unload
# spot (no taught retreat leg), so nothing here describes that path.
# At the unload spot there is ANOTHER lid already lying on the floor, and the
# held one is stacked onto it: the place xy/yaw come from detecting THAT lid on
# the plane below, exactly like the pick.
#
# This xy does DOUBLE duty. It is where the chassis alignment drives the cup
# point to (the place column only solves over a limited patch at the place
# stance, and hand-driving lands within 10-20cm of it), and it is the blind
# fallback if the detection fails — as a CUP position, so the grab offset is
# not applied to it in that case.
LID_PLACE_XY_M: tuple[float, float] = (0.86, 0.30)
# Hard floor on how close the chassis alignment may bring the cup point in x.
# The reference above sits exactly on it, so the guard is normally redundant —
# deliberately: closing on the drop-off is the one correction with something in
# front of the robot, and an open-loop leg that overshoots (or a reference
# lowered later) must not be able to walk into it. Costs no reach: the place
# column still solves out to x 0.95 at LID_PLACE_TORSO_DEG.
LID_PLACE_MIN_X_M: float = 0.860
# How many detect -> check -> pick-a-spot -> align rounds the place gets before
# it gives up. >1 because ONE round validates a spot at the xy and surface
# height it measured BEFORE the chassis moved, and then flies whatever it
# actually arrived at: 0906 10:25 the 63mm alignment landed 6mm off in y (well
# inside _center_lid's 20mm deadband) and the re-measured surface read 5.5mm
# lower, which together left the bottom of the column 11.9mm short against
# REACH_TOL_M's 10mm — a few more cm of chassis would have solved it, but there
# was no second round to ask for them.
LID_PLACE_ALIGN_MAX_TRIES: int = 8
# Slack _nearest_place_spot must have IN HAND before it accepts a cup xy: the
# column has to solve with the surface this much higher AND lower, and with the
# cup this far off in Y. The nearest-first grid search otherwise returns the
# first candidate that merely passes — a spot on the EDGE of the reachable
# region, with zero room for the drifts above.
#
# NOT applied in x, and 10mm rather than the 20mm arrival deadband, because the
# x band cannot pay for it: the 0904 sweep found the column solving at x
# 0.84-0.90 at contact, and LID_PLACE_MIN_X_M clips that to 0.860-0.90 — about
# 40mm, so demanding +/-20mm of x slack leaves a single point and +/-10mm
# leaves 20mm. y is the affordable axis (solves across y 0-0.30) and was also
# the bigger error on 0906 (6mm, vs 1mm in x); x drift is what the retry rounds
# are for. 10mm covers that run's 6mm y / 5.5mm surface drift.
#
# If no spot has this much slack the search is re-run without it, rather than
# falling back to the taught point which ignores the detection entirely.
LID_PLACE_SPOT_MARGIN_M: float = 0.01
# Expected contact EE z: where the creep line is set from (creep starts
# DESCENT_CREEP_GAP_M above it) and the descent's floor. Force sensing, not this
# number, actually ends the descent, so erring HIGH only lengthens the creep.
#
# 0904 MEASURED, and it is now only the FALLBACK — cfg.LID_DEPTH_REFINE takes
# the real contact height from the depth map per frame. The value comes from
# lid_probe's touch test, which descended the empty cup onto the lid and
# stopped on force at ee_z 0.2846 (surface 0.1296). Depth had said 0.147 at a
# nearby spot and the profile 0.132 — the two independent sensors agree inside
# ~2cm, while the old guesses (0.35, then 0.40, then 0.50) were 20cm out. The
# earlier numbers came from converting a "0.40 above the ground" reading with a
# ground height taken off the URDF wheel axles; that conversion was wrong.
#
# 0906: depth has now measured this contact 12 times, 0.2821-0.3109 (median
# 0.3046) — 0.285 sat 20mm BELOW the median and below 10 of the 12 reads. The
# spread is real, not noise: the held lid stacks onto whatever is already on
# the floor, so the contact rises with the pile. Moved to the median. Erring
# HIGH is the safe direction (a high guess only starts the creep early and
# lengthens it; a low one lets the fast planned approach run deeper before the
# creep takes over), so raise this rather than lower it if in doubt.
LID_PLACE_EE_Z_M: float = 0.305
# The descent STARTS here instead of SAFE_TRANSPORT_Z: the drop-off column is
# only reachable low down at this stance (z=1.10 at this xy is 200mm+ short, so
# the normal transport approach cannot fly it), and starting 100mm up leaves
# 50mm of planned approach above the creep line — the same shape as every other
# place, just lower. Tracks LID_PLACE_EE_Z_M by a fixed 100mm: with the lid
# place passing the long creep gap (DESCENT_CREEP_GAP_M = 50mm), a start of
# contact+50mm would leave NO planned stretch at all. It is that GAP that
# matters, not the absolute value — run_lid re-applies it to whatever contact
# height depth measures. At the measured 0.285 contact the column solves at
# x 0.84-0.90 for y 0-0.30 (x 0.95 needs y>=0.15); x 1.00 does not, which is
# why the lid where it actually sits (x~1.04) needs the chassis to close in.
LID_PLACE_START_EE_Z_M: float = 0.405
# Torso stance for the place. The lid column at LID_PLACE_XY_M does NOT solve
# at the demo stance (TORSO_JOINTS: 30mm short at ee_z 0.35) — this leans the
# arm base down and over so it does. arm.pin_torso re-models at it, and the
# stance is restored to TORSO_JOINTS after the release.
LID_PLACE_TORSO_DEG: tuple[float, float, float] = (14.8, 59.0, -60.3)
# BEV warp plane for detecting the lid ALREADY ON THE FLOOR at the unload spot
# = its TOP FACE. Measured: the cup touched down on it at EE z 0.35, one
# cup-length above, so 0.35 - SUCTION_LENGTH_M.
#
# DECOUPLED from LID_PLACE_EE_Z_M on 0904, when that was raised to 0.40. The
# two answer different questions — where the surface IS, versus where the creep
# should start — and only the descent tolerates being wrong. Erring high on the
# creep line just lengthens the creep (force sensing stops the descent), while
# a plane 50mm too high drags the detected xy along the camera ray: offline, a
# lid truly at (0.86, +0.300) would read as (0.829, +0.277), a 31 x 23mm bias
# straight into the placed position. Re-measure and set THIS if the floor lid's
# height actually changes (stacked higher, on a pallet); leave it alone when
# only the release height is being tuned.
LID_FLOOR_PLANE_Z_M: float = 0.295 - SUCTION_LENGTH_M
# Head pitch for that detection, in set_head_pitch's convention (head j1 =
# torso_pitch_deg - this). RE-DERIVED for the 0904 stance, whose torso pitch is
# -14.5 deg (j1+j3+90-j2) rather than the old 0: 24 puts the floor spot at
# pixel row 317 of 631, dead centre. 20 through 50 all keep it in frame (rows
# 290-500), and the BEV mapper is built from the LIVE head joints either way,
# so this only has to keep the lid comfortably framed.
LID_PLACE_HEAD_PITCH_DEG: float = 24.0
# Where --lid writes its DIAGNOSTIC IMAGES: the frame + BEV a committed place
# was decided from, and the depth-sampling markers. Its own folder, not
# RUN_LOG_DIR — that one is text a run appends to and greps through, and
# hundreds of PNGs alongside it make both harder to use. The run stamp stays in
# the filename, so an image still pairs with its run_<stamp>.log. None disables
# image saving.
LID_IMAGE_DIR: str | None = "/home/dexmate/LGES/Dexmate/LGES/ik_demo/lid_images"

# Refine every lid detection with the ZED DEPTH: measure the surface height
# under the detected center and move the center onto that height (an exact
# homothety about the camera centre, bev.reproject_plane — no re-warp, no
# re-detect), then take the expected contact z from the same measurement.
#
# This splits two things the warp plane was doing at once. LID_PLANE_Z_M /
# LID_FLOOR_PLANE_Z_M stay the DETECTION plane, chosen so the lid appears at a
# scale the model recognises; the SURFACE HEIGHT is measured per frame. 0904
# measured the cost of conflating them: the floor lid read (0.870,+0.009) at
# the configured 0.345 plane while depth put its face at 0.147 — 198mm of
# plane error, 168mm of pure x bias in the detected center, and a column that
# tested reachable at the biased xy and was NOT at the real one.
#
# LID_DEPTH_CONTACT_PCT: which depth percentile in the window answers. 50 is
# the face's height (what the reprojection wants); a LOW percentile is the
# NEAREST point, i.e. the tallest bump, which is what a descending cup lands on
# first — measured lids carry ~20mm of embossing.
# ---------------------------------------------------------------------------
# --box-lid: the PAPER box lid. Same drop-off spot as --lid, but the suction
# only lifts it clear and the RIGHT GRIPPER takes it from the side (the old
# battery-handoff pattern), then just opens above the drop spot — no gentle
# place needed.
# ---------------------------------------------------------------------------
# The box OBB set is 2-class (dataset_box/data.yaml: names: [box, box_top]), so
# the paper LID is class 1. detect_box_bev also size-gates on BOX_BEV_SIZE_M.
BOX_LID_CLS_ID: int = 1
# Cup grab point on the PAPER lid, in its own frame — the CENTRE, unlike the bin
# lid's LID_GRAB_OFFSET_M. A 0.62 x 0.43 sheet of cardboard is floppy: gripped
# off-centre it hinges about the cup and the far end drags, and the centre is
# also the only spot both the side handoff and the drop pose were taught around.
# Being (0,0) it is immune to the OBB's 180-deg branch (canonical_lid_yaw folds
# it anyway, but there is nothing left to mirror).
#
# A CONSTANT rather than the literal it used to be inside run_box_lid: the BEV
# viewer (case_detection/live_detect_bev.py) draws the EE target it reads from
# this package, and with no constant to read it drew box_top with the BIN lid's
# -50,+20mm offset — a marker 54mm from where the arm actually goes.
BOX_LID_GRAB_OFFSET_M: tuple[float, float] = (0.0, -0.02)
# Warp plane for that detection. SEED ONLY: cfg.LID_DEPTH_REFINE measures the
# real surface per frame and reprojects the centre onto it, so this just has to
# put the lid at a scale the model knows (its frames were warped near
# top_face_z(1) = 0.6138) and land the first depth sample on the lid — good to
# about +-10cm, see the walk table in .claude/notes/lid-mode-and-bev-planes.md.
# 0905 MEASURED with lid_probe's touch test, at the GENTLE gate: contact at
# ee_z 0.8795, so the surface is 0.8795 - SUCTION_LENGTH_M = 0.7245.
#
# The same lid read 0.8539 through the 10N global gate — 25.6mm lower for 6N
# more, i.e. the cardboard gives about 4.3 mm/N and the 10N measurement was
# already denting it. Take the gentle number: it is the undeformed surface, the
# one a non-contact depth measurement also sees.
#
# Worth setting even though depth refines it per frame, because this is ALSO
# the fallback the expected contact z comes from: at the original 0.65 seed the
# fallback contact would be 0.805 and the creep line 0.855, which is BELOW this
# surface — the planned approach would have arrived already pressing.
BOX_LID_PLANE_Z_M: float = 0.7245
# Contact / abort forces for the PAPER lid pick, in place of the global
# FORCE_CONTACT_THRESHOLD_N (10 N) and FORCE_HARD_LIMIT_N (20 N). Cardboard
# dents long before either: 0905's touch test tripped the 10 N gate and read
# 11.3 N at ee_z 0.8539 (so ~1.3 N of overshoot in one reaction tick at creep
# speed). 4 N is still 20x the wrench's ~0.2 N rest noise, so it will not
# false-trigger, and it leaves the abort well under a crush.
BOX_LID_CONTACT_N: float = 4.0
BOX_LID_FORCE_LIMIT_N: float = 8.0
# --- handoff: suction lifts, the RIGHT GRIPPER takes it from the side --------
# The handoff: lift straight up to here over the pick xy, then carry the lid
# this far to +y before the gripper comes in.
#
# The carry is there because the gripper cannot otherwise reach the lid's -y
# edge. With the wrist pinned pointing +y its reach runs out around EE y -0.28,
# and the edge sits half a lid-width to the -y side of wherever the cup is:
# offline at the pick spot the entry is only 60mm long at lid yaw 0 and does not
# solve AT ALL at +8 deg, while 150mm to the left every case from -8 to +8 gets
# the full BOX_LID_APPROACH_M. Aiming at a FIXED handoff pose was tried first
# (0906, (0.85,0.30) — a 25cm drag) and pulled back out; this is the smallest
# move that buys the room, and it is flown as a straight line at the DESCENT
# CREEP speed, the slowest leg in the file, because a big sheet held by its
# centre on one cup is exactly what peels off when hurried sideways.
BOX_LID_HANDOFF_Z_M: float = SAFE_TRANSPORT_Z
BOX_LID_HANDOFF_SHIFT_Y_M: float = 0.18
# Side-grasp orientation for the gripper: tool z (the fingers) points +y so it
# comes in from the robot's RIGHT, and the fingers close along base z — one
# above the lid, one below. That is rpy (-90, -90, 0) deg.
BOX_LID_SIDE_RPY: tuple[float, float, float] = (-1.5707963, -1.5707963, 0.0)
# How far PAST the lid's -y edge the fingertips end up (bite depth), how far
# short of that the straight-line entry starts, and how far BELOW the sheet the
# fingers ride while doing it.
#
# All three were flown by hand on the robot (0906 11:55, box_lid_jog, run
# 20260906_115137; lid at (0.908,+0.039) yaw -6.6 deg, short side 0.385, sheet
# at z 0.945):
#   * the fingers went 10mm BELOW the sheet plane, entered from 15mm past the
#     edge (EE y -0.2770) and force-stopped 112mm past it (EE y -0.1795)
#   * closing there took the lid off the cup cleanly
# Nothing resists a finger crossing a floppy sheet, so the entry only stops
# once the paper goes taut against the cup that anchors it — 112mm in, ~80mm
# short of the cup axis. The aim is therefore set AT that observed stop and the
# FORCE STOP still ends the entry, exactly like the suction descent; if the
# guard stays silent the leg overruns by 3mm instead of driving at the cup.
# The earlier 0.07 / 0.08 pair was never touched to a lid.
BOX_LID_GRASP_INSET_M: float = 0.115
# Fingers BELOW the sheet plane (negative = below). The sheet drapes off a
# centre cup, so the pads want to be under its hanging edge, not level with the
# plane the cup defines.
BOX_LID_GRASP_DZ_M: float = -0.0
# Extra shift applied to the grasp point AFTER the force-guarded side entry
# stops, before the gripper closes — base_link y, added to wherever the force
# stop actually landed (negative = further past the edge, deeper toward the
# cup). UNVERIFIED on the robot.
BOX_LID_GRASP_Y_SHIFT_M: float = -0.02
# Force along the approach direction that ends the side entry. The right wrist
# carries the gripper, so its rest reading is heavier than the suction wrist's —
# 3N is well clear of noise and gentle on cardboard (the lid gives ~4.3 mm/N).
BOX_LID_SIDE_CONTACT_N: float = 3.0
# Entry length. 0.10 puts the start exactly where the hand-flown one was
# (15mm past the edge); it is SHORTENED automatically when the gripper
# cannot reach that far out, which is common — see run_box_lid.
BOX_LID_APPROACH_M: float = 0.10
BOX_LID_SIDE_SPEED_M_S: float = 0.05
# Fallback for the lid's SHORT side, which sets where its right edge is
# (edge_y = centre_y - short/2). Normally the detection's own measured size is
# used — detect_box_bev size-gates against case_detection's BOX_BEV_SIZE_M, so
# a detection that passed has a trustworthy size — and this only covers a
# detection that somehow arrived without one. Lives here because ik_demo must
# not read the detection package's config.
BOX_LID_SHORT_FALLBACK_M: float = 0.43
# If the CENTRE will not seal (vacuum_timeout), the pick lifts and retries this
# far along the lid's SHORT axis (+y in the lid frame), sign alternating — see
# _seal_with_retry. Same 5mm and same axis as the bin lid, which is the global
# PICK_SEAL_RETRY_OFFSET_M default.
#
# Was 50mm along the LONG axis, on the reasoning that a centre which will not
# seal has a seam / print / dome there and must reach genuinely different
# material, the long axis being where a 0.62 x 0.43 sheet has room. Dropped to
# 5mm on 0906 by operator call. It stays a constant of its own rather than
# falling through to the global default so the paper lid's choice is still
# stated here, and so tuning the case pick's nudge cannot silently move it.
BOX_LID_SEAL_RETRY_OFFSET_M: float = 0.005
# Where the gripper OPENS to drop the lid, at LID_PLACE_TORSO_DEG (the same
# leaned stance the bin lid is placed from — the right arm cannot get below
# z 0.65 at the demo stance, which would mean a 52cm drop). Offline: reachable
# at the lean, ~32cm above the floor lid's measured 0.13 surface. No gentle
# place is wanted here, just an open.
BOX_LID_DROP_EE_POS: tuple[float, float, float] = (0.80, 0.30, 0.45)
LID_DEPTH_REFINE: bool = True
LID_DEPTH_CONTACT_PCT: float = 10.0
