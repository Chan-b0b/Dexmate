"""Fit the orange bin's top rim as a rectangle (cx, cy, yaw[, L, W]) in base_link top view.

Rim points = orange points within RIM_BAND of the rim height. Each point is scored by
its distance to the nearest rectangle edge (robust Huber), so a partly hidden rim
(arm / held part over it) still constrains the pose as long as some of each side shows.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from cloud import base_cloud, orange_mask

RIM_BAND = 0.03          # m below the rim top kept as rim points
ROI = dict(x=(0.3, 1.4), y=(-0.4, 0.4), zmax=0.93)


def rim_points(npz, roi=ROI, keep=None, rim_z=None, band=0.015, local_top=False):
    """Orange points at the rim. `rim_z` (known: the bin is fixed) keeps only
    |z - rim_z| < band — without it the rim height is taken as the 97th
    percentile, which a TRANSPARENT held part over the bin pulls up (orange seen
    through it, depth at the part's surface). Returns (points, rim height used)."""
    P, C, uv = base_cloud(npz)
    m = orange_mask(C) & (P[:, 0] > roi["x"][0]) & (P[:, 0] < roi["x"][1]) \
        & (P[:, 1] > roi["y"][0]) & (P[:, 1] < roi["y"][1]) & (P[:, 2] < roi["zmax"])
    if keep is not None:
        m &= keep(P, uv)
    Q = P[m]
    if local_top:
        # rim = the locally highest orange points (3 cm cells, within 1.2 cm of the
        # cell top, above 0.84): the near rim reads ~2.5 cm low in the head depth
        # (1015 chassis_offset), so no single height band holds all four sides
        Q = Q[Q[:, 2] > 0.84]
        cell = np.floor(Q[:, :2] / 0.03).astype(np.int64)
        key = cell[:, 0] * 100003 + cell[:, 1]
        order = np.argsort(key)
        k_sorted = key[order]
        starts = np.r_[0, np.nonzero(np.diff(k_sorted))[0] + 1]
        top = np.maximum.reduceat(Q[order, 2], starts)
        cell_top = np.empty(len(Q)); cell_top[order] = np.repeat(top, np.diff(np.r_[starts, len(Q)]))
        return Q[Q[:, 2] > cell_top - 0.012], rim_z
    if rim_z is not None:
        # The head camera sees the table tilted 2-3 deg (extrinsic error), so over the
        # bin's 0.72 m the near rim reads ~3 cm low and fell out of a flat band. Fit
        # the rim PLANE to the band points, then re-select around that plane.
        sel = np.abs(Q[:, 2] - rim_z) < band
        for _ in range(2):
            A = np.c_[np.ones(sel.sum()), Q[sel, 0], Q[sel, 1]]
            coef = np.linalg.lstsq(A, Q[sel, 2], rcond=None)[0]
            sel = np.abs(Q[:, 2] - (coef[0] + coef[1] * Q[:, 0] + coef[2] * Q[:, 1])) < band
        return Q[sel], rim_z
    ztop = np.percentile(Q[:, 2], 97)
    return Q[Q[:, 2] > ztop - RIM_BAND], ztop


def _edge_dist(p, xy):
    cx, cy, th, L, W = p
    c, s = np.cos(th), np.sin(th)
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    u, v = c * dx + s * dy, -s * dx + c * dy           # bin frame
    du, dv = np.abs(u) - L / 2, np.abs(v) - W / 2
    outside = np.hypot(np.maximum(du, 0), np.maximum(dv, 0))
    inside = np.maximum(du, dv)                          # <= 0 inside: distance to the nearest edge
    return np.where((du > 0) | (dv > 0), outside, -inside)


def fit(xy, LW=None, init=None):
    if init is None:
        (cx, cy), (a, b), ang = cv2.minAreaRect(xy.astype(np.float32))
        th = np.radians(ang)
        if a < b:
            a, b, th = b, a, th + np.pi / 2
        init = (cx, cy, th, a, b)
    th0 = init[2]
    th0 = (th0 + np.pi / 2) % np.pi - np.pi / 2          # long axis within +-90 deg
    if LW is None:
        r = least_squares(lambda p: _edge_dist(p, xy), [init[0], init[1], th0, init[3], init[4]],
                          loss="huber", f_scale=0.005)
        return r.x
    r = least_squares(lambda p: _edge_dist([*p, *LW], xy), [init[0], init[1], th0],
                      loss="huber", f_scale=0.005)
    return np.array([*r.x, *LW])
