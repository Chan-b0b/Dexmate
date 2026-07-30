"""Config for right-gripper force-pick episode collection (collect.py).

Distances in metres, angles in radians unless noted.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------
# Default grip pose in base_link: gripper horizontal, pointing forward — where
# every jog phase starts. Teach it at the ik_demo torso stance (collect.py
# pins the torso to ik_demo cfg.TORSO_JOINTS, same as every other ik_demo
# motion script — teaching at a different torso pose makes this unreachable
# or shifted). Paste as (x, y, z, roll, pitch, yaw).
# None = skip the move and jog from wherever the arm currently is.
DEFAULT_GRIP_POSE: tuple[float, ...] | None = (0.75, -0.20, 0.68, -1.57, -0.00, -1.570233)

# IK seed + fallback for the project home — NOT the home itself while
# HOME_MATCH_GRIP_RPY is on (then home = HOME_POS @ DEFAULT_GRIP_POSE rpy,
# solved at startup seeded from THIS, which anchors the arm posture/branch
# the solve lands on — keep it a comfortable configuration). It IS the
# literal home when HOME_MATCH_GRIP_RPY = False. None disables the project
# home entirely (ik_demo arms-up home, no rpy snapping).
HOME_JOINTS: tuple[float, ...] | None = (
    -1.3051, -1.55, +0.2919, -1.8710, -0.1628, +1.1959, -0.1480)

# Snap the home ORIENTATION to DEFAULT_GRIP_POSE's rpy: home keeps
# HOME_JOINTS' position but the wrist starts grasp-aligned (solved by IK once
# at startup; aborts if the combination is unreachable). False = use
# HOME_JOINTS exactly as given.
HOME_MATCH_GRIP_RPY: bool = True

# Home EE position in base_link, overriding HOME_JOINTS' FK position (the
# orientation still comes from DEFAULT_GRIP_POSE via HOME_MATCH_GRIP_RPY,
# and HOME_JOINTS seeds the IK). None = keep HOME_POS.
HOME_POS: tuple[float, float, float] | None = (0.673, -0.30, 0.961)

# After the home solve, walk the elbow-swivel null space to push j2 as
# negative as the pose allows (right arm: j2 in [-1.553, +0.453], negative =
# elbow out / armpit open). Approach/grasp solves chain from home, so the
# open posture carries through the episode.
HOME_OPEN_ELBOW: bool = False

# Project-local upper bound for R_arm_j2 (rad): every IK solve in this
# session keeps j2 at or below this, so the elbow never folds inward past
# neutral (URDF upper limit is +0.453). Applied to the QP limit constraint,
# seed clipping, and validity checks alike. None = keep the URDF limit.
J2_MAX: float | None = 0.0

STANDOFF_M: float = 0.20   # back-off along the tool axis before/after the grasp
STANDOFF_MIN_M: float = 0.10   # deep retreats can leave the workspace (too close to
                               # the torso for the forward orientation) — collect.py
                               # shrinks the depth 5 cm at a time down to this floor
LIFT_DZ_M: float = 0.15    # lift height above the grasp z
HOME_LIFT_DZ_M: float = 0.20   # relative lift before the joint move home (objects are
                               # small — beats ik_demo safe_home's absolute 1.05 m)
HOLD_S: float = 1.0        # hold at the top (records hold current / slip) before the take ends

# Random put-down: after a successful (still-held) lift, release at a
# uniform-random xy inside this base_link box, and use that spot as the next
# episode's grasp target (jog-free chaining; 'j' at the prompt re-jogs).
# SET THIS TO YOUR TABLE'S CLEAR AREA. z / orientation reuse the grasp pose.
# Samples are IK-checked and resampled, so the box may exceed reach a little.
PLACE_X_RANGE: tuple[float, float] = (0.70, 0.80)
PLACE_Y_RANGE: tuple[float, float] = (-0.25, -0.15)
PLACE_MAX_TRIES: int = 20

# ---------------------------------------------------------------------------
# Keyboard jog
# ---------------------------------------------------------------------------
JOG_STEP_M: float = 0.01        # position step per keypress (+/- to change)
JOG_STEP_MIN_M: float = 0.0025
JOG_STEP_MAX_M: float = 0.05
JOG_OSTEP_DEG: float = 5.0      # orientation step per keypress

# ---------------------------------------------------------------------------
# Gripper (Robotiq rFR/rSP raw units, 0..255)
# ---------------------------------------------------------------------------
GRIP_FORCE: int = 0      # rFR 0..255 — note 0 still gives the gripper's minimum force
                         # (~20 N on a 2F-85), not zero. 'f N' at the prompt overrides;
                         # logged in meta per episode.
GRIP_SPEED: int = 0x20   # slow close = gentler contact + finer gCU force profile.
                         # Floor ~0x18 with SOFT_GRIP=False: the driver only waits
                         # 3 s for close confirmation (robotiq_usb wait_until_done)
                         # and a full close at 0x20 (~36 mm/s) already takes ~2.4 s.
                         # SOFT_GRIP has its own timeout — any speed works there.

# Contact-stop "soft grip": the Robotiq force controller cannot go below
# ~20 N even at force 0, so instead of letting it stall, stream a slow close,
# watch the motor current, and FREEZE the position target at first contact
# (+SQUEEZE counts). Holding a position applies only the elastic squeeze —
# effective force well below the controller floor, bounded by detection
# latency (~1 poll ≈ 1-2 mm of extra travel at GRIP_SPEED 0x20).
SOFT_GRIP: bool = True
SOFT_GRIP_CU_STOP: int = 3     # gCU counts (~10 mA each) treated as contact. The 2F
                                # has ONE motor for both fingers (no per-finger
                                # sensing): a single finger nudging the object aside
                                # reads ~2-4 sustained, a BOTH-finger squeeze ramps
                                # through 5+ — this sits between the two.
SOFT_GRIP_CU_CONSECUTIVE: int = 5  # polls in a row >= CU_STOP before freezing:
                                   # filters single-poll noise spikes and brief
                                   # one-finger grazes; the object self-centers while
                                   # the close continues until BOTH fingers load up
SOFT_GRIP_SQUEEZE: int = 1      # counts commanded past the contact position (hold margin)
SOFT_GRIP_TIMEOUT_S: float = 6.0

# Release-to-contact: lower the held object onto the table on the wrist F/T
# signal instead of dropping it from a fixed height (the ~2 cm drop let the
# object bounce/tip — exactly the offset the next auto-grasp then inherits).
# The wrench is tared at RELEASE_START_DZ_M above the put-down z with the
# object still held, then a slow creep descends until the tared base-vertical
# force says table contact. Sensor missing -> falls back to the fixed drop.
RELEASE_ON_CONTACT: bool = True
RELEASE_CONTACT_N: float = 2.0    # tared vertical force = contact (fz noise sigma ~0.6 N)
RELEASE_START_DZ_M: float = 0.03  # re-align + tare height above the put-down z
RELEASE_FLOOR_DZ_M: float = 0.02  # stop and open anyway below grasp_z - this
RELEASE_SPEED_M_S: float = 0.02   # creep speed of the monitored descent

# ---------------------------------------------------------------------------
# Collision monitor (Collision/collision_monitor.py — model-based, two-layer)
# ---------------------------------------------------------------------------
# Gravity+friction-model residual monitor (the one validated on the left arm).
# Needs a RIGHT-arm calibration first:
#     python Collision/calibrate_gravity_model.py --side right
# (~2 min of arm sweeps; start from the ik_demo right home). Thresholds
# default to the calibration file's suggestions. Intentional-contact legs
# (grip close, place-to-contact) suppress the monitor; OBJECT_MASS_KG feeds
# its Layer B payload compensation while the object is held (0 = skip).
OBJECT_MASS_KG: float = 0.1

# Threshold scales on the calibration's suggested values. Layer B (absolute
# residual) is payload-sensitive: carrying the object during the put-down
# transfer measured j2 excess 1.21 A vs the suggested 1.137 A threshold, so
# it gets headroom by default (raise OBJECT_MASS_KG instead for a tighter
# guard). Layer A (impact) is payload-robust — keep at 1.0.
COLLISION_ABS_SCALE: float = 1.5
COLLISION_CHG_SCALE: float = 1.2

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
RECORD_HZ: float = 15.0
PHASE: str = "gripper_pick"
OBJECT_DEFAULT: str = "juice"
INSTRUCTION_TMPL: str = "pick up the {obj} with the gripper"  # keep phrasing stable across a dataset
