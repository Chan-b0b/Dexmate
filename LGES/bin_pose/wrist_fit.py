"""Locate the fixed orange bin in the RIGHT WRIST image (no depth): match the
orange mask to the silhouette of a tapered bin box (rim 722x417 @ z 0.891,
foot ~600x390 @ table z) projected through the wrist camera.

The right arm now sits at one pose all run (HOME_JOINTS_RIGHT = the view pose),
so the wrist camera is effectively fixed to the robot body: its base-frame pose
is calibrated once (calibrate) against a bin pose measured with the head depth
camera at an unoccluded moment, then reused (locate)."""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))

W, H = 960, 600
K = np.array([[366.61773681640625, 0, 479.5], [0, 366.3883056640625, 299.5], [0, 0, 1]])
CAM_T = np.array([0.0002, -0.07459, 0.0017504])          # URDF R_camera_joint (dexs gripper)
CAM_R = R.from_euler("x", -0.087266).as_matrix()
RIM = (0.722, 0.417, 0.891)
FOOT = (0.600, 0.390, 0.665)
SCALE = 0.5                                               # work at half resolution


def wrist_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 28) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 90)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.resize(m, (int(W * SCALE), int(H * SCALE)), interpolation=cv2.INTER_NEAREST).astype(bool)


def urdf_cam_pose(q_right, q_torso):
    """base <- wrist camera from the arm FK + the URDF mount."""
    from ik_demo import config as cfg
    from ik_demo.arm import ArmMover
    cfg.TORSO_JOINTS = tuple(float(v) for v in q_torso)
    arm = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
    pos, rpy = arm.fk(np.asarray(q_right, dtype=float))
    Re = R.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4); T[:3, :3] = Re @ CAM_R; T[:3, 3] = pos + Re @ CAM_T
    return T


def bin_corners(cx, cy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for L, Wd, z in (RIM, FOOT):
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            u, v = a * L / 2, b * Wd / 2
            out.append((cx + c * u - s * v, cy + s * u + c * v, z))
    return np.array(out)


def silhouette(T_base_cam, pose):
    P = bin_corners(*pose)
    q = (np.linalg.inv(T_base_cam) @ np.c_[P, np.ones(len(P))].T)[:3]
    if (q[2] <= 0.05).any():
        return None
    uv = (K @ q)[:2] / q[2] * SCALE
    m = np.zeros((int(H * SCALE), int(W * SCALE)), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(uv.T.astype(np.float32)).astype(np.int32), 1)
    return m.astype(bool)


def iou_loss(T, pose, mask):
    s = silhouette(T, pose)
    if s is None:
        return 1.0
    return 1.0 - (s & mask).sum() / max((s | mask).sum(), 1)


def perturb(T, d):
    """d = (rx, ry, rz [rad], tx, ty, tz [m]) applied in the camera frame."""
    D = np.eye(4); D[:3, :3] = R.from_rotvec(d[:3]).as_matrix(); D[:3, 3] = d[3:]
    return T @ D


def calibrate(T0, pose_truth, masks):
    f = lambda d: np.mean([iou_loss(perturb(T0, d), pose_truth, m) for m in masks])
    best = None
    for start in ([0] * 6, [0.03, 0, 0, 0, 0, 0], [0, 0.03, 0, 0, 0, 0], [0, 0, 0.03, 0, 0, 0]):
        r = minimize(f, start, method="Powell", options=dict(xtol=1e-4, ftol=1e-5, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return perturb(T0, best.x), best


def locate(T, mask, init):
    f = lambda p: iou_loss(T, p, mask)
    best = None
    for dx in (0.0, 0.03, -0.03):
        for dy in (0.0, 0.03, -0.03):
            r = minimize(f, [init[0] + dx, init[1] + dy, init[2]], method="Powell",
                         options=dict(xtol=1e-5, ftol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    return best.x, 1.0 - best.fun
