"""ZED-left-camera geometry: deproject a depth pixel into the camera frame and
transform it into base_link.

Intrinsics and the hand-derived torso+head forward kinematics are lifted from
perception/perception.py (the validated pick-perception pipeline) so the
dashboard's bin-centre height matches what that code computes. Pure numpy — no
robot handle or heavy deps — so it runs in the decoupled detector process,
which reads the live joints from the spooled state.json.
"""

from __future__ import annotations

import numpy as np

# ZED left-camera intrinsics, rectified SVGA 960x600 stream. load_intrinsics()
# replaces these with the live camera info when it can; these are the FALLBACK
# for when the info query fails (it sometimes times out). Values = this robot's
# camera info (ZED X Mini, read 1001). The old perception.py copy (cx 497.74,
# cy 315.53) was 15/8 px off: a tape on the robot centre line read y +28..34 mm
# with it and +-2 mm with these.
ZED_FX = 366.04998779296875
ZED_FY = 366.04998779296875
ZED_CX = 482.7677001953125
ZED_CY = 307.1591796875
INTRINSICS_SOURCE = "fallback"


def load_intrinsics(robot) -> bool:
    """Take fx/fy/cx/cy from the head camera's info service (the calibration of
    the camera actually fitted, at the stream's resolution). Returns False and
    keeps the fallback constants if the info is unavailable or malformed."""
    global ZED_FX, ZED_FY, ZED_CX, ZED_CY, INTRINSICS_SOURCE
    try:
        left = robot.sensors.head_camera.get_camera_info()["intrinsics"]["left"]
        fx, fy, cx, cy = (float(left[k]) for k in ("fx", "fy", "cx", "cy"))
    except Exception:  # noqa: BLE001 — no info / timeout / layout change: keep the fallback
        return False
    if not all(np.isfinite(v) and v > 0 for v in (fx, fy, cx, cy)):
        return False
    ZED_FX, ZED_FY, ZED_CX, ZED_CY = fx, fy, cx, cy
    INTRINSICS_SOURCE = "camera_info"
    return True


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _T_base_arm_center(q_torso: np.ndarray) -> np.ndarray:
    """base_link -> arm_center 4x4 (torso joints only), vega_1p URDF."""
    q = np.asarray(q_torso, dtype=np.float64).ravel()[:3]
    q1, q2, q3 = q[0], q[1], q[2]
    T_0_1 = _trans(-0.235, 0.0, 0.248) @ _rot_y(-q1)
    T_1_2 = _trans(0.396, 0.0, 0.082) @ _rot_y(q2)
    T_2_3 = _trans(-0.40718, 0.0, 0.09764) @ _rot_y(-q3)
    T_0_3 = T_0_1 @ T_1_2 @ T_2_3
    return T_0_3 @ _trans(-0.05908, 0.0, 0.44528)


def _head_l3_pose_from_joints(q_torso, q_head) -> np.ndarray:
    """base_link -> head_l3 4x4."""
    q_h = np.asarray(q_head, dtype=np.float64).ravel()[:3]
    h1, h2, h3 = q_h[0], q_h[1], q_h[2]
    T_base_ac = _T_base_arm_center(q_torso)
    T_ac_l1 = _trans(-0.0735, -0.0725, 0.014) @ _rot_y(h1)
    T_l1_l2 = _trans(0.0, 0.0725, -0.0035) @ _rot_z(h2)
    T_l2_l3 = _trans(0.0, 0.002, 0.0495) @ _rot_y(-h3)
    return T_base_ac @ T_ac_l1 @ T_l1_l2 @ T_l2_l3


def zed_left_camera_pose_from_joints(q_torso, q_head) -> np.ndarray:
    """base_link -> zed_left_camera 4x4."""
    T_base_l3 = _head_l3_pose_from_joints(q_torso, q_head)
    T_l3_cam = _trans(0.0365, 0.023, 0.0489) @ _rot_rpy(-1.57079, 0, -1.57079)
    return T_base_l3 @ T_l3_cam


def deproject_pixel(u: float, v: float, depth: float) -> np.ndarray:
    """Pixel (u, v) + depth (m) -> 3D point in the ZED left-camera frame."""
    x = ((u - ZED_CX) / ZED_FX) * depth
    y = ((v - ZED_CY) / ZED_FY) * depth
    return np.array([x, y, depth], dtype=np.float64)


def transform_zed_point_to_base(point_in_zed, q_torso, q_head) -> np.ndarray:
    """Transform a ZED-camera-frame point into base_link."""
    T_base_cam = zed_left_camera_pose_from_joints(q_torso, q_head)
    p = np.array([point_in_zed[0], point_in_zed[1], point_in_zed[2], 1.0], dtype=np.float64)
    return (T_base_cam @ p)[:3]
