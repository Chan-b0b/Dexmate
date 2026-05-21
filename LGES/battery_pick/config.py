"""Configuration for battery pick modules (perception + suction grasp).

Battery-pick-specific constants are defined here.
Shared IK/URDF parameters are imported from the parent LGES config.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    CONTROL_DT,
    IK_CONVERGENCE_THRESHOLD,
    IK_DT,
    IK_MAX_ITERS,
    PREFERRED_QP_SOLVER,
    URDF_PATH,
)

__all__ = [
    "URDF_PATH",
    "IK_DT",
    "IK_MAX_ITERS",
    "IK_CONVERGENCE_THRESHOLD",
    "CONTROL_DT",
    "PREFERRED_QP_SOLVER",
]

# ---------------------------------------------------------------------------
# Camera intrinsics — ZED X Mini (matches box_detection_utils.py)
# ---------------------------------------------------------------------------
CAMERA_FX: float = 366.21429443359375
CAMERA_FY: float = 366.21429443359375
CAMERA_CX: float = 497.73809814453125
CAMERA_CY: float = 315.53277587890625

# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

# Pixel-space ROI for the source box [x1, y1, x2, y2].
# Tune these so only the source box is included.
SOURCE_BOX_ROI: tuple[int, int, int, int] = (150, 100, 850, 500)

# Valid depth range for battery surfaces (meters).
DEPTH_MIN_M: float = 0.2
DEPTH_MAX_M: float = 2.0

# Pixels within this depth of the ROI minimum are treated as battery tops.
BATTERY_TOP_DEPTH_TOL_M: float = 0.03  # 3 cm tolerance

# Minimum contour area (px²) to be considered a battery.
MIN_BATTERY_AREA_PX: int = 800

# ---------------------------------------------------------------------------
# Grasp arm
# ---------------------------------------------------------------------------
ARM_SIDE: str = "left"  # "left" or "right"
EE_FRAME: str = "L_gripper_base"  # URDF end-effector frame name

# End-effector approach orientation (roll, pitch, yaw) in radians.
# Tune so the suction cup faces the battery top surface.
GRASP_ORIENTATION_RPY: tuple[float, float, float] = (0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Suction hardware
# ---------------------------------------------------------------------------
SUCTION_BASE_URL: str = "http://192.168.1.1/api/dc/weblogic"
SUCTION_HOST: str = "192.168.1.1"

# Physical suction tube length from EE origin to cup tip (m).
SUCTION_LENGTH_M: float = 0.25

# Height the cup tip hovers above the battery before activating suction (m).
HOVER_HEIGHT_M: float = 0.10

# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------
DESCENT_STEP_M: float = 0.005   # 5 mm per step
DESCENT_DT_S: float = 0.05      # 50 ms between steps → 20 Hz descent
MAX_DESCENT_M: float = 0.30     # safety: stop after 30 cm of descent

# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

# Wrench sensor force magnitude (N) above which contact is assumed.
FORCE_CONTACT_THRESHOLD_N: float = 3.0

# Hard-stop force limit (N) — emergency retract if exceeded.
FORCE_HARD_LIMIT_N: float = 25.0

# Time (s) to wait for vacuum seal after force contact before declaring failure.
VACUUM_SEAL_TIMEOUT_S: float = 2.0

# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

# Duration (s) for the arm to travel from its current pose to the hover pose.
MOVE_DURATION_S: float = 3.0
