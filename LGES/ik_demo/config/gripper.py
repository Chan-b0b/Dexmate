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

TARGET_BARCODES: list[str] = ["UDCG7B0292", "UDCG7B0289"]
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
# = box yaw + pi/2. The gripper is symmetric under a half turn, so pick_box
# tries yaw and yaw + pi and flies the first whose whole column solves.
BOX_GRASP_YAW_OFFSET_RAD: float = np.pi / 2
BOX_FINGER_LENGTH_M: float = 0.14     # R_gripper_base origin -> fingertip along tool z. MEASURE on the
                                      # gripper (URDF has the finger hinges 29 mm out, no tips); TUNE
BOX_GRASP_DEPTH_M: float = 0.03       # fingertips this far below the box top at the grasp
BOX_HOVER_HEIGHT_M: float = 0.15      # fingertips above the box top before / after the vertical leg
# WHERE on the detected box to grip (box_pick.box_pose_from_detection): the
# box (~0.62-0.68 m long, ~0.4-0.45 m wide) is far wider than the Robotiq
# opening, so the pick pinches a WALL from above — the midpoint of the long
# wall on the robot's RIGHT (base -y): one finger inside the box, one outside.
# INSET moves the pinch point from the wall line toward the box center (m);
# 0 = fingers centered on the wall.
BOX_GRASP_EDGE_INSET_M: float = 0.0
BOX_GRIP_HOLD_S: float = 1.0          # hold at the top of the lift test / before a release
BOX_LIFT_TEST_M: float = 0.10         # lift-test mode: raise the gripped box this far, set it back down, release
# Box rim height (base_link z, m), assumed CONSTANT: the BEV detection plane
# and the pick's top_z unless box_pick gets --top-z / --box-long-m. 0.65 is
# the value the 0903 air tests ran at (0.60 raised by 5 cm on the user's call).
BOX_RIM_Z_M: float = 0.65
# Contact guard on the box descent: the right wrist wrench is tared at the hover
# (BOX_TARE_SAMPLES readings at rest, ~200 Hz) and the descent halts when the
# vertical force moves more than this from the baseline — fingers landing on
# the rim (rim higher than BOX_RIM_Z_M) or on the contents. Rest noise is
# ~0.2 N (0903 probe), so 7 N is well clear. The pick then lifts back to the
# hover and reports "contact" instead of closing on nothing.
BOX_CONTACT_FORCE_N: float = 7.0
BOX_TARE_SAMPLES: int = 50
