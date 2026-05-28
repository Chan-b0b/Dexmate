"""Configuration file for Pink IK solver.

This file contains paths and configuration parameters for the robot and trajectories.
"""

import os

# URDF file path
URDF_PATH = "/home/dexmate/xhand/xr_teleoperate_dm/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"

# Trajectory file paths (relative to custom_examples directory)
DEFAULT_TRAJECTORY_FILE = "trajectories/VegaTask1-021-Grab.txt"

# IK solver parameters
IK_DT = 0.01  # Time step for IK iterations
IK_MAX_ITERS = 500  # Maximum IK iterations
IK_CONVERGENCE_THRESHOLD = 1e-4  # Convergence threshold

# Task costs
LEFT_EE_POSITION_COST = 2.0  # [cost] / [m]
LEFT_EE_ORIENTATION_COST = 1.0  # [cost] / [rad]
RIGHT_EE_POSITION_COST = 2.0  # [cost] / [m]
RIGHT_EE_ORIENTATION_COST = 1.0  # [cost] / [rad]
POSTURE_COST = 1e-3  # [cost] / [rad]

# Control parameters
CONTROL_DT = 0.02  # Control loop time step (50 Hz)

# Box detection parameters
BOX_DETECTION_MAX_ATTEMPTS = 10  # Maximum attempts to detect box
CAMERA_TIMEOUT = 5.0  # Camera activation timeout (seconds)

# QP solver preference (will use first available if preferred not found)
PREFERRED_QP_SOLVER = "daqp"

# Force-based grasp parameters (used by GRAB trajectory step type)
GRAB_FORCE_THRESHOLD_N = 10.0   # Stop squeezing when either hand exceeds this force (Newtons)
GRAB_MIN_HAND_SEPARATION_M = 0.25  # Back off one step when Y-separation between hands drops below this (metres)

# Force threshold during IK motion steps — pauses and waits for SPACE when exceeded
IK_FORCE_PAUSE_THRESHOLD_N = 300
