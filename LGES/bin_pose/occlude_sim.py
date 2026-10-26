"""Simulate the held case + arm over the target bin: drop every depth pixel the
occluder would cover, refit the bin rim, compare with the unoccluded fit."""
import sys

import cv2
import numpy as np

sys.path.insert(0, "../case_detection")
import depth_plane as dp  # noqa: E402
from fit_bin import fit, rim_points  # noqa: E402

LW = (0.720, 0.413)
CASE = (0.50, 0.35)            # held case footprint (CASE_BEV_SIZE_M)
CUP_LEN = 0.15                 # EE -> case top (SUCTION_LENGTH_M)
CASE_H = 0.08                  # held case thickness (guess)
SHOULDER = (0.05, 0.25, 1.35)  # left shoulder, base frame (rough)
ARM_R = 0.06                   # forearm/upper-arm half thickness (rough)


def occluder_mask(npz, case_c, ee_z, yaw):
    q_t, q_h = npz["q_torso"], npz["q_head"]
    H, W = npz["depth"].shape
    m = np.zeros((H, W), np.uint8)
    c, s = np.cos(yaw), np.sin(yaw)
    top, bot = ee_z - CUP_LEN, ee_z - CUP_LEN - CASE_H
    pts = []
    for z in (top, bot):
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            u, v = a * CASE[0] / 2, b * CASE[1] / 2
            pts.append((case_c[0] + c * u - s * v, case_c[1] + s * u + c * v, z))
    px = np.array([dp.base_to_pixel(p, q_t, q_h) for p in pts], np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(px).astype(np.int32), 1)
    # arm: tube from the EE (cup top) up to the shoulder, sampled as thick segments
    ee = np.array([case_c[0], case_c[1] + 0.077, ee_z]); sh = np.array(SHOULDER)
    for t in np.linspace(0, 1, 12):
        p = ee + t * (sh - ee)
        try:
            u, v = dp.base_to_pixel(p, q_t, q_h)
            u2, _ = dp.base_to_pixel(p + [0, ARM_R, 0], q_t, q_h)
        except ValueError:          # the part near the shoulder is behind the camera
            continue
        if not (-W < u < 2 * W and -H < v < 2 * H):
            continue
        cv2.circle(m, (int(u), int(v)), max(2, int(abs(u2 - u))), 1, -1)
    return m.astype(bool)
