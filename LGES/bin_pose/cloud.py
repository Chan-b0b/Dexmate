"""Head-camera depth -> base_link point cloud (+ orange-bin colour mask). Offline."""
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
sys.path.insert(0, str(_HERE.parents[0] / "case_battery_demo" / "dashboard"))
import bev  # noqa: E402
import camera_geometry as cg  # noqa: E402


def base_cloud(npz, step=2):
    """(N,3) base_link points, (N,3) rgb, (N,2) pixel uv for every `step`-th valid pixel."""
    rgb, D = npz["rgb"], npz["depth"]
    assert D.shape == rgb.shape[:2], "depth must be at the RGB resolution"
    K = bev.intrinsic_matrix()
    v, u = np.mgrid[0:D.shape[0]:step, 0:D.shape[1]:step]
    d = D[v, u]
    ok = np.isfinite(d) & (d > 0.2) & (d < 3.0)
    u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok].astype(np.float64)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)])
    T = cg.zed_left_camera_pose_from_joints(npz["q_torso"], npz["q_head"])
    P = (T @ pc)[:3].T
    return P, rgb[v.astype(int), u.astype(int)], np.stack([u, v], 1)


def orange_mask(colors):
    hsv = cv2.cvtColor(colors.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    return (hsv[:, 0] >= 8) & (hsv[:, 0] <= 28) & (hsv[:, 1] >= 120) & (hsv[:, 2] >= 90)
