"""Suction hardware and everything the cup does on the way down.

The vacuum IO endpoints, the rolling wrench reference, the two-speed
descent + contact/seal thresholds, and the corner-seat registration
(drive the held part into the jig's two datum walls) at the end.

One domain of the ik_demo configuration, split out 2026-09-04. Consumers import
the FACADE (config/__init__.py) — ``from . import config as cfg`` — which
re-exports every name here; nothing imports this module directly.
"""

from __future__ import annotations

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
# Rolling wrench reference (replaces the per-descent tare). Force readings are
# the CHANGE from the median of the preceding WRENCH_REF_WINDOW_S of raw
# samples, frozen the moment a contact is declared.
#
# Why the change and not an absolute baseline: the drift happens AFTER the
# baseline is captured, so no absolute zero can fix it. 0903, free air with
# nothing touching, ALREADY tared, within a single descent — fx wandered 0.9 to
# 3.2N and fy 2.3 to 2.5N. That is the whole CASE_CORNER_STOP_N budget (2.5N),
# and it showed: all ten wall latches of the 16:56 run fired at 2.5-3.6N with
# their travel scattered -41 to 0mm, i.e. indistinguishable from drift, while
# the two verified single-wall contacts read 5-10N SUSTAINED. Taping the wrist
# also shifted the absolute baseline (fy +1.5N) and tripled its run-to-run
# scatter (fx 0.21 -> 0.6N) — a rolling reference absorbs both.
#
# Separation of scales is what makes it work: fast noise ~0.6N, drift 1-3N over
# ~1s, a real contact 5-10N in 20-50ms. A median over the PRECEDING window
# tracks the drift and ignores the contact (which contaminates only the last
# few samples of the window, and a median shrugs those off).
WRENCH_REF_WINDOW_S: float = 0.25      # rolling reference window
WRENCH_REF_WARMUP_S: float = 0.10      # force decisions stay OFF until the window
                                       # holds this much (was 0.7s under the tare:
                                       # DESCENT_RAMP_S + 100 samples, ~120mm of
                                       # descent with no force check at all)
# Axis sign correction for the tared wrench. The driver already reports the
# wrench in the BASE frame, so this is the WHOLE correction — do NOT rotate by
# the EE rotation on top of it (see suction.contact_wrench).
#
# Rotating by R = Rz(psi) @ Rx(pi) injects Rx(pi)'s lateral block diag(1,-1) —
# a MIRROR whose axis moves with the wrist yaw, which no diagonal sign array
# can undo.
#
# z is negated: the driver's z runs along the tool axis (down when the cup
# faces down), so a seat pushing UP on the part reads NEGATIVE. Every current
# consumer takes |fz|, so this only matters to new SIGNED z logic.
#
# LATERAL: the frame is MIRRORED about the 45 deg line (x and y SWAPPED),
# det = -1 — not rotated, and not fixable by any sign array. Pinned by the two
# operator-confirmed single-wall samples of 0903 (each one wall only, so the
# reactions do not superpose):
#   Y wall, +y drive (15:24): fx = -5.4..-9.6N sustained, fy never past +1.3N,
#                             my/fx = +0.19m
#   X wall, -x drive (15:36): fy = +4.8..+10.0N sustained, fx within +-1.2N,
#                             mx/fy = -0.18m
# Both levers land at the real contact depth below the sensor, so both loads
# are genuine. A wall reaction is +x for the X wall and -y for the Y wall, so
# true x -> measured y and true y -> measured x: a SWAP. A rotation R(90) would
# send true y to measured -x instead, which contradicts the X-wall sample.
#
# This is why no earlier attempt worked: a diagonal WRENCH_AXIS_SIGN cannot
# express a swap, so every guess fixed one axis and broke the other (and
# wrench_probe's det check flags exactly this). Two earlier readings of
# TWO-wall samples ("drive-aligned", then "~90 deg rotated") were both wrong —
# with both axes in contact the reactions superpose and no per-axis sign is
# recoverable from the sum.
#
# The array stays UNFLIPPED and the correction lives at its single consumer,
# CASE_CORNER_LAT_CHANNEL: this array multiplies the TORQUE too, so bending
# x,y here would drag mx,my along for no gain (lateral torque is diagnostics
# only) and still could not express the swap.
WRENCH_AXIS_SIGN: tuple[float, float, float] = (1.0, 1.0, -1.0)
DESCENT_APPROACH_SPEED_M_S: float = 0.3    # fast free-air descent (cup-tip)
# Cruise for the CORNER-SEAT descent only (_descend_corner_seat) — the one place
# descent still streamed per-tick from the hover. The arm's tracking error is
# proportional to the commanded speed and it is NOT along the commanded column:
# 0905 case place at 0.3 m/s measured 40-60mm of z lag, 30-55mm of x drift and
# 5-10 deg of pitch (= up to 83mm of cup-tip deviation off the vertical line),
# all of it collapsing to 1-2mm / 0.1 deg the moment the profile reached the
# 0.04 creep. A case that arrives at its datum wall in that state hits it
# tilted and corner-first: 15:20:35 went 1.4N -> 26.3N of lateral in 90ms and
# aborted on CASE_CORNER_LAT_LIMIT_N with the case held over the bin. 0.08
# keeps the same profile shape at ~1/4 the deviation for +1.4s per case place.
# Only the CASE reaches this: the battery's approach ends BELOW creep_z
# (DESCENT_CREEP_GAP_SETTLED_M), so its corner descent creeps from tick one.
CORNER_DESCENT_SPEED_M_S: float = 0.15
DESCENT_CREEP_SPEED_M_S: float = 0.06      # slow creep in the contact zone
DESCENT_RAMP_S: float = 0.2                 # ease descent speed in from 0 (no jerk
                                            # from the rest->descend handoff)
DESCENT_CREEP_BLEND_M: float = 0.05         # decelerate fast->creep smoothly over this
# band ABOVE creep_z. Sized to spread the deceleration, not just to avoid a
# velocity step: the smoothstep already zeroes ACCEL at both ends of the band
# (0903 measured the creep-entry jerk at +1.8 m/s^3, negligible), so the jerk
# that matters is the INTERIOR peak of the decel itself — 121 m/s^3 at the old
# 30mm. Widening to 50mm (= DESCENT_CREEP_GAP_M, so the profile decelerates
# continuously to contact with no constant-speed plateau) drops that to 49 for
# +0.18s. 80mm would give 20 but eats into the cruise. A quintic smootherstep
# was measured and rejected: it zeroes the boundary jerk (+1.8 -> 0) but RAISES
# the interior peak (121 -> 126), since the same velocity change is packed into
# a shorter effective stretch.
# Decel band for the VERTICAL descent profile only (_descent_speed): the
# fast->creep slowdown is spread over this much travel above creep_z. 0909
# recordings (collect_case_pick, 0.3 cruise): the arm trails the command by
# ~0.2s, so a 50mm decel (0.25s) ended with the arm still at 0.23 m/s — it
# overshot ~10mm, bounced ~10mm up and swung pitch +6.7 -> -2 deg at the creep
# line. 120mm gives the slowdown ~0.6s, longer than the lag, for ~+0.15s per
# descent. DESCENT_CREEP_BLEND_M above stays as the corner seat's lateral-drive
# gate so the place's travel budget is unchanged.
DESCENT_DECEL_BAND_M: float = 0.12
DESCENT_CREEP_GAP_M: float = 0.05           # creep starts this far above expected contact.
# Sized from the CUP DEVIATION the fast stretch leaves behind, not from the
# contact-detection margin. The arm lags its command while moving fast and the
# lag is off-path: 0903 battery place, at 0.2 m/s the cup sat 40mm to the side
# of the commanded vertical line, and it only decays once the descent slows —
# roughly halving per 11mm of creep travel (40.7mm at the creep line -> 21.2mm
# 11mm later -> ~1mm once stopped). The old 0.027 was far too short: the part
# still arrived ~20mm off, and with 80mm slot spacing that puts the battery on
# the case wall instead of in the slot (the 0903 15:44/15:58 misseats). ~50-60mm
# of creep is what 40mm needs to decay; 70mm leaves margin. The case place only
# built 7mm of deviation and recovered inside 27mm, which is why it succeeded on
# the same runs. Cost: +43mm of creep at DESCENT_CREEP_SPEED_M_S, ~+1.1s per
# descent, on the PICK legs too (they share this value; their deviation is
# already small, so for them it is pure time).
# Creep gap for a descent whose approach ends AT REST on the creep line and has
# been confirmed there by _settle_at (the pick and every non-corner place, i.e.
# to_creep_z=True). The 50mm above buys TIME for a cruise-built cup deviation to
# decay — and those legs no longer have one: 0904's log has the pick's creep
# ("seal") entering at |track_xy| = 0.5, 0.3, 0.5mm, while the case place
# ("corner"), which still streams per-tick from the hover, enters at 3.5 -> 7.9
# -> 21.8mm and does need the distance. With nothing to shed, the only jobs
# left for the gap are the contact-detection overshoot (sub-mm at creep speed)
# and the error in the EXPECTED contact z, which is what 20mm budgets: the
# chained ZTracker expectations land inside ~10mm. It is the error budget, so
# a column whose expected z is a GUESS must pass the long gap explicitly
# (pick/place take creep_gap=...) — --lid does, its plane being a ±3cm estimate.
# Saves (50-20)mm / DESCENT_CREEP_SPEED_M_S = 0.75s per pick and per battery
# place; a contact ABOVE this gap happens during the planned leg, which is
# force-guarded (FORCE_HARD_LIMIT_N) but hits at cruise speed.
DESCENT_CREEP_GAP_SETTLED_M: float = 0.02
DESCENT_MAX_M: float = 0.40                 # safety: max descent distance
# TEMP diagnostic: every per-tick vertical stream (the suction descents and
# move_ee_vertical) logs commanded-vs-solved-vs-MEASURED EE this often, in
# seconds — see ArmMover._track_trace for what the fields mean. The descents
# command a fixed xy and only vary z, so the commanded path is vertical BY
# CONSTRUCTION and a visibly slanted descent has to be either IK residual
# (silent up to REACH_TOL_M) or tracking. 0.0 disables the whole trace.
DESCENT_TRACE_S: float = 0.25
# Show the per-tick traces in the TERMINAL too, or only in the run log file.
# They fire 4x/s per stream and drown the lines an operator actually watches
# (contacts, wall latches, failures), while being the thing every 0903
# diagnosis was built on — so off here, kept in the file. DESCENT_TRACE_S = 0.0
# switches them off everywhere.
DESCENT_TRACE_TO_TERMINAL: bool = False
FORCE_CONTACT_THRESHOLD_N: float = 10.0      # |vertical force| -> contact
FORCE_HARD_LIMIT_N: float = 20.0            # pick abort (empty cup)
FORCE_HARD_LIMIT_PLACE_N: float = 20.0      # place abort (battery in cup). Must sit
                                            # well above FORCE_CONTACT_THRESHOLD so a
                                            # normal seating contact registers as
                                            # contact (seat+release), not a hard abort.
VACUUM_SEAL_TIMEOUT_S: float = 5.0          # DI0 takes ~3-4s to latch, so this
                                            # is ~1-2s of margin over a seal that
                                            # is going to happen. Cut from 8.0 on
                                            # 2026-09-05: the wait runs on EVERY
                                            # attempt, so with PICK_SEAL_RETRIES
                                            # it was 24s of held-still time before
                                            # a pick could be called failed.
PICK_SEAL_RETRIES: int = 2                  # on vacuum_timeout: lift DESCENT_CREEP_GAP_M
                                            # and creep-seal again, up to this many times
# Lateral nudge on a seal retry, along a direction the CALLER supplies (the lid
# pick passes the lid's own +y). Signs ALTERNATE off the original aim: retry 1
# goes +this, retry 2 goes -this, so each retry samples a fresh patch of surface
# instead of re-pressing the point that just refused to seal. Without it a retry
# only helps when the miss was an error in the EXPECTED contact z — a crease, a
# label edge or an embossing step under the cup fails the same way all three
# times, which is exactly the lid's failure mode (LID_PLANE_Z_M is a +-3cm
# estimate AND the lid has ~20mm of embossing). Only picks that pass a direction
# get the nudge; every other pick lifts and re-presses in place as before.
# 5mm keeps the cup on the same feature (the grab point is offset by
# LID_GRAB_OFFSET_M, tens of mm from any edge) while clearing a local defect.
# UNVERIFIED on the robot.
PICK_SEAL_RETRY_OFFSET_M: float = 0.005
SEAL_PRELIFT_M: float = 0.00               # relieve contact press before suction on
RELEASE_PRELIFT_M: float = 0.015            # lift before the blow-off release
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
# Case corner-seating (suction.place(corner_seat=True), every case place): the
# case jig gives two hard datum WALLS at one corner, so instead of trying to
# hit the 1-2mm jig fit against the +4..12mm landing scatter (or searching for
# it — the removed sweep/spiral), descend and register in ONE guarded stream:
# aim the descent a few mm AWAY from that corner and, while descending, DRIVE
# the held case toward it — z hands off to a light press servo at the first
# vertical contact (no halt), and each lateral axis stops on its own
# wall-contact force at any height (tall walls guide the case down the
# corner). The corner becomes the datum: final position accuracy = the jig
# clearance, independent of detection error, and the two walls square residual
# yaw through the cup compliance. A rim landing drops into the slot mid-drive
# — "dropped" = a sudden z sink (the press servo drops z further than its own
# max step in one window) TOGETHER WITH an fz drop off the maintained press,
# so one noisy sample can't trigger early. All gains unverified on the robot.
CASE_CORNER_DIR: tuple[float, float] = (-1.0, +1.0)  # base-frame signs toward
# the datum corner ("lower-left" as seen from the robot: backward + left).
# -x is data-backed (every measured landing was +4..12mm forward of intended);
# the +y sign is a guess — VERIFY against the jig on-site, flip if the walls
# are on the -y side.

#확인
CASE_CORNER_AIM_BIAS_M: float = 0.03       # shift the descent aim AWAY from
# the corner so the drive always approaches from the free side.
CASE_CORNER_SPEED_M_S: float = 0.04        # lateral drive speed. The drive is
# on for the WHOLE descent (hover -> contact): the tall bin walls latch each
# axis at whatever height the case bumps them, then it rides the corner down.
CASE_CORNER_STOP_N: float = 4.5            # per-axis stop threshold, applied to
# that axis' wall-reaction channel (NOT the channel named after the axis — see
# CASE_CORNER_LAT_CHANNEL and WRENCH_AXIS_SIGN). Must clear the
# sliding-friction baseline (~mu*press = 1-2N, right at the 1.5N
# PLACE_RECOVER_FORCE_MIN_N noise floor).
# KNOWN WEAKNESS (not yet addressed): the check runs every tick (200Hz, ~400
# samples per descent) and latches on a SINGLE crossing, permanently — a
# stopped axis is never re-examined. At 2.5N with ~1N noise that is 2.5 sigma,
# and a real wall read only 3.7N on the weaker axis, so a false latch per
# descent is likely. A hold/debounce (force must stay over the threshold
# continuously for ~50ms) fixes this cheaply.
# Which wrench channel carries each drive axis' wall reaction, as
# (channel index, sign): axis i latches when f[channel] * sign >= STOP_N.
# Entry 0 is the X drive axis, entry 1 the Y drive axis; channel 0 is fx, 1 fy.
# The lateral wrench frame is MIRRORED (x/y swapped) vs base — see
# WRENCH_AXIS_SIGN — so each axis reads on the OTHER channel, and the two
# signs differ because the two drives oppose each other (-x vs +y):
#   X wall -> fy >= +STOP_N   MEASURED 0903 15:36 (single-wall, confirmed).
#   Y wall -> fx <= -STOP_N   MEASURED 0903 15:24, and again 15:35/15:36 on
#     three separate places (fx = -2.5, -2.6, -3.0N).
# Getting this wrong is not benign, in either direction: with the axes crossed
# the Y wall's reaction latched X, freezing x 6.8mm short of its own aim while
# y drove its full 100mm cap into the wall it was already pressing (15:24,
# place_chk_y +46.6mm); with X's sign inverted, X latched at 0.0mm of travel on
# a -2.5N noise sample (15:36 battery_2, tare sd_f 0.89N) while the real X wall
# went unseen for the whole 100mm cap.
CASE_CORNER_LAT_CHANNEL: tuple[tuple[int, float], ...] = ((1, +1.0), (0, -1.0))
# Lateral force abort for the corner drive: |(fx,fy)| over this halts and
# returns force_limit. FORCE_HARD_LIMIT_PLACE_N only guards the VERTICAL force,
# so before this there was NO cap on the sideways push at all — 0901 14:46 the
# y drive missed its latch and ground into the wall to 41.5N (still climbing at
# ~0.5 N/mm) with fz a harmless 8.5N, so nothing stopped it. Set above a
# genuine wall reaction (3-20N observed) but below a grind.
CASE_CORNER_LAT_LIMIT_N: float = 25.0
# Lateral RELIEF: above this wall reaction an axis backs OFF (opposite its
# drive), latched or not, up to CASE_CORNER_RELIEF_TRAVEL_M. The z press servo
# has had a relief since day one; the lateral axes had none, so they could only
# freeze or grind to the abort above — and freezing does not stop the load,
# because the descent keeps going: 0903 18:55 both axes latched with the
# lateral command FROZEN at (-34.8,+16.4)mm, and fy still climbed 2.9 -> 13.5
# -> 15.6 -> 30.0 -> 32.7N as z fell 0.8063 -> 0.7965, wedging the part into
# the corner until the 25N abort. `if stopped[i]: continue` meant the axis
# holding that load had no way to give.
# Sits between STOP_N (2.5, a wall touch) and LAT_LIMIT_N (25, a grind), so a
# normal registration preload is untouched and only a build-up gives way. The
# gap between the two is the deadband that keeps it from fighting the drive.
CASE_CORNER_RELIEF_N: float = 8.0
CASE_CORNER_RELIEF_TRAVEL_M: float = 0.008   # cap per axis, so relief can never
                                             # unwind the whole registration
CASE_CORNER_MAX_TRAVEL_M: float = 0.20      # per-axis travel cap, airborne AND
# pressing (aim bias + landing scatter + wall clearance + margin) — must reach
# the corner walls from the aim, or the drive stops short of registration.
CASE_CORNER_BACKOFF_M: float = 0.01         # back the force-latched axes off the
# walls before the release, AWAY from the datum corner: dirs=(-1,+1) so this is
# x +BACKOFF, y -BACKOFF. Relieves the wall preload so the cup retreat cannot
# drag the registered case, and at 10mm it also clears the case of the walls
# before the blow-off. Battery has its own (below) — its slot is tighter.
BATTERY_CORNER_BACKOFF_M: float = 0.002     # battery keeps the minimal preload
                                            # relief: 10mm inside a battery slot
                                            # would pull it back off its datum
CASE_CORNER_TIMEOUT_S: float = 20.0
CASE_CORNER_DROP_GRACE_S: float = 1.0       # both axes stopped but no drop yet
# (walls that protrude above the rim can register xy BEFORE the drop): keep
# pressing this long for the sink before classifying it a misseat.
CASE_CORNER_PRESS_N: float = 5.0            # light press held during the drive
# LIFT-then-drive (only a case placed with place(corner_touch_first=True), i.e.
# the run's first one): once the descent has TOUCHED the seat, rise this far
# before the lateral drive runs, so the case hangs clear of the surface it
# would otherwise be dragged across and the lateral channel carries the wall
# reaction ALONE. A pressed drag adds mu * (part weight + press) pointing
# exactly the way a wall reaction does — 0903 measured the ratio at ~0.27 of
# the vertical for a battery on a case face, which is the 1.4N the 4N
# CASE_CORNER_STOP_N is sized against, but the touch-first case has to cross
# the whole CASE_CORNER_AIM_BIAS_M (50mm, vs the battery's 10mm) on a
# different surface pair, so that margin is no longer something to lean on.
# 8mm: the contact is ~3 N/mm and FORCE_CONTACT_THRESHOLD_N declares touchdown
# at 10N, so ~3mm of that lift only unwinds the contact compression and ~5mm
# is real air. The bin walls run at least 65mm above the seat (0904-0905
# airborne latches at ee_z = seat + 50..80mm), so 8mm stays deep between them
# — which is why the BATTERY must never get this: its slot walls are low
# enough that 8mm could clear them and the drive would push it over the top.
# After both axes latch the case is set back DOWN on the press servo
# (CASE_CORNER_PRESS_N) before the release, so the reported contact z is still
# the real seat depth.
# This is the MINIMUM rise, not the whole story: it is measured at the EE, and
# between the EE and the case sit the cup's own compression (touchdown is
# declared at FORCE_CONTACT_THRESHOLD_N = 10N, so the bellows is already loaded)
# and the arm's tracking lag (0905 16:13: -3.3mm while lifting). 8mm of EE rise
# left the case still ON the seat — the drive then dragged it and latched x on
# 3.2mm of travel at 8.1N of sustained friction, 47mm short of the wall. So the
# lift ENDS ON FORCE, not on distance (CASE_CORNER_LIFT_FREE_N below); this
# value only keeps it from ending on the first noisy sample.
CASE_CORNER_DRIVE_LIFT_M: float = 0.008
# The lift is done when the vertical force says the seat is no longer carrying
# the case. The wrench reference is FROZEN at touchdown, where the cup was
# holding the case in free air, so a signed reading back at ~0 means exactly
# "the case hangs on the cup again, as it did on the way down" — the one
# measurement that is not confounded by cup compliance or tracking lag. 1N of
# margin over sensor noise.
CASE_CORNER_LIFT_FREE_N: float = 1.0
# Cap on the rise, so a case that is PINNED (jammed under a rim, or a reference
# that drifted) cannot elevator up out of the bin looking for a force drop that
# will never come. 25mm is well inside the walls, which run 65mm+ above the seat
# (0904-0905 airborne latches at ee_z = seat + 50..80mm). Reaching the cap with
# force still on the seat is logged as a WARNING and the drive runs anyway — the
# operator gate downstream is the right place to stop, not a held case mid-air.
CASE_CORNER_LIFT_MAX_M: float = 0.025
# How far BELOW the touchdown z the set-down after a lifted drive may servo
# before giving up (-> misseat, operator gate). The press servo lowers whenever
# the force is under CASE_CORNER_PRESS_N, and the set-down STARTS out of contact
# (fz ~ 0 while the case hangs), so without a floor a case hung up on the walls
# would never satisfy the servo and it would keep descending at its 0.02 m/s cap
# for the whole CASE_CORNER_TIMEOUT_S — 400mm into the bin. 20mm is the bound
# because a case that legitimately drops past the rim into its slot goes 5-15mm
# (measured 2026-08-05, see PLACE_MISSEAT_TOL_M); anything deeper is not a seat.
CASE_CORNER_SETDOWN_MAX_M: float = 0.020
# While PRESSING, the corner drive runs only once the press servo has actually
# REACHED its target, within this tolerance (so the band is
# CASE_CORNER_PRESS_N +- this). Sliding friction is mu * press and pushes the
# same way a wall reaction does, so driving at an arbitrary press force makes
# the wall latch unreadable: 0903 measured mu ~ 0.28 (fz 8-12N while the
# lateral channel read 2.6-2.8N), and the old gate borrowed
# FORCE_CONTACT_THRESHOLD_N (10N, a CONTACT-detection number) — which allows a
# press whose friction ALONE is 2.8N, over the 2.5N latch threshold. Every
# post-contact latch that run fired at 2.6-3.0N, i.e. on friction.
# At 5 +- 2.5N the worst-case friction is 2.1N, under the threshold.
# The band has to be generous because the force swings while the part settles
# into its seat (0903: 14.1 -> 2.1N, and 8.9 -> -1.2N on another place). If it
# is never satisfied the drive simply never runs and CASE_CORNER_TIMEOUT_S ends
# the place as a misseat — a safe failure, not a hang.
CASE_CORNER_DRIVE_PRESS_TOL_N: float = 2.5
CASE_CORNER_PRESS_KP: float = 0.004         # z-servo gain, (m/s) per N of error
CASE_CORNER_PRESS_MAX_SPEED_M_S: float = 0.02
CASE_CORNER_RELIEF_SPEED_M_S: float = 0.08  # upward z speed while OVER the
# hard-push limit in press mode: lift to relieve instead of aborting. Must
# outrun the descent-servo-lag force buildup (0824: 14N -> 20.2N in ~40ms
# while the press servo raised at its 0.02 cap).
CASE_CORNER_RELIEF_MAX_M: float = 0.02      # relief headroom above the first
# contact; still over the limit past this = a true jam -> force_limit abort.
CASE_CORNER_SINK_WINDOW_S: float = 0.3      # rolling window for the sink check.
# MUST exceed SINK_DZ_M / PRESS_MAX_SPEED (the sink is measured on the
# COMMANDED z, which the servo caps at PRESS_MAX_SPEED — the removed sweep
# search used 0.15s x 20mm/s = 3mm/window against a 4mm threshold, so its
# sink could never fire).
CASE_CORNER_SINK_DZ_M: float = 0.004        # z sink within the window -> signal 1
CASE_CORNER_SINK_FZ_DROP_N: float = 3.0     # fz drop off the press within the
                                            # window -> signal 2 (both required)
# Battery corner-seating (suction.place(corner_seat="battery"), every target
# battery place): same corner registration as the case — shared direction /
# stop-force / press / sink gains above — but the battery descends STRAIGHT
# (no drive in the air: its slot walls are low, and an airborne drift past
# the slot could never be pulled back) and drives toward the slot's
# lower-left walls only AFTER the first vertical contact. Replaces
# _misseat_recover for target battery places.
BATTERY_CORNER_AIM_BIAS_M: tuple[float, float] = (0.01, 0.01)  # (x, y) aim shift
                                             # away from the corner (free-side
                                             # approach) — x widened past y for
                                             # a firmer forward-wall contact
BATTERY_CORNER_SPEED_M_S: float = 0.02      # lateral drive speed (slot-scale,
                                             # separate from the case's)
BATTERY_CORNER_MAX_TRAVEL_M: float = 0.04   # per-axis post-contact travel cap
# (bias + landing scatter + slot clearance — the old recovery's XY excursion
# cap); must reach the slot walls or the place gates to the operator.

# Hard cap on the post-drop settle pause. The pause waits for the z wobble to
# fall under CASE_CORNER_SETTLE_DZ_M, which assumes the wobble DOES decay — an
# oscillating press servo keeps it open forever and the lateral drive stays
# locked, so an axis that has not found its wall can never move to find it
# (0903 18:00: z hunting 2.8mm p-p, settling stuck True, gate False, one axis
# frozen and the place spinning to CASE_CORNER_TIMEOUT_S). Past this the pause
# is force-released with a warning: a wedged-in-diagonally risk is better than
# a guaranteed timeout.
CASE_CORNER_SETTLE_MAX_S: float = 1.5
CASE_CORNER_SETTLE_DZ_M: float = 0.001      # window z-span below this after a
                                            # sink = drop finished, resume the
                                            # drive (paused so a half-dropped
                                            # case can't be wedged diagonally)
