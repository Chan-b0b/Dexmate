"""Robot motion relative to the fixed bin from the RIGHT WRIST camera alone
(right arm at home = camera fixed on the robot body), against a REFERENCE wrist
image taken when the bin pose was known (head depth, arm out of the way).

No bin shape model: the reference image's EDGE pixels near the bin rim are lifted
onto the rim plane (z RIM_Z) through the wrist camera pose T, moved by a candidate
bin displacement (dx, dy, dyaw in base), re-projected, and scored by their distance
to the query image's edges (chamfer). T only has to be roughly right: an error in
it scales / turns the measured displacement a little, it does not offset it."""
import cv2
import numpy as np
from scipy.optimize import minimize

from wrist_fit import K, H, W

RIM_Z = 0.891
BAND_M = 0.03            # reference edge pixels within this of the rim rectangle (rim plane)
INSIDE_M = 0.0          # how far INSIDE the rim outline edges are kept: the bin's contents
                         # (cases / batteries stacked between visits) change, the rim does not


def edges(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    return cv2.Canny(g, 40, 110)


def pixel_to_plane(T, uv, z=RIM_Z):
    d = np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T
    d = T[:3, :3] @ d
    t = (z - T[2, 3]) / d[2]
    return (T[:3, 3][:, None] + t * d).T


def project(T, P):
    q = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T)[:3]
    return ((K @ q)[:2] / q[2]).T, q[2]


def rim_band_pixels(T, rgb, bin_pose, LW=(0.720, 0.413)):
    """Reference edge PIXELS whose rim-plane point (through T) lies within BAND_M of the
    rim outline — chosen once, so a calibration that varies T cannot change the set."""
    v, u = np.nonzero(edges(rgb))
    uv = np.c_[u, v].astype(float)
    P = pixel_to_plane(T, uv)
    cx, cy, yaw = bin_pose
    c, s = np.cos(yaw), np.sin(yaw)
    a, b = c * (P[:, 0] - cx) + s * (P[:, 1] - cy), -s * (P[:, 0] - cx) + c * (P[:, 1] - cy)
    near = ((np.abs(np.abs(a) - LW[0] / 2) < BAND_M) & (np.abs(b) < LW[1] / 2 + BAND_M)) | \
           ((np.abs(np.abs(b) - LW[1] / 2) < BAND_M) & (np.abs(a) < LW[0] / 2 + BAND_M))
    return uv[near]


def rim_band_points(T, rgb, bin_pose, LW=(0.720, 0.413)):
    """Reference edge pixels whose rim-plane point lies within BAND_M of the rim outline."""
    v, u = np.nonzero(edges(rgb))
    P = pixel_to_plane(T, np.c_[u, v].astype(float))
    cx, cy, yaw = bin_pose
    c, s = np.cos(yaw), np.sin(yaw)
    du, dv = P[:, 0] - cx, P[:, 1] - cy
    a, b = c * du + s * dv, -s * du + c * dv
    inside_deep = (np.abs(a) < LW[0] / 2 - INSIDE_M) & (np.abs(b) < LW[1] / 2 - INSIDE_M)
    a, b = c * du + s * dv, -s * du + c * dv
    dist = np.maximum(np.abs(np.abs(a) - LW[0] / 2) * (np.abs(b) <= LW[1] / 2 + BAND_M),
                      0) + 0 * b
    near = ((np.abs(np.abs(a) - LW[0] / 2) < BAND_M) & (np.abs(b) < LW[1] / 2 + BAND_M)) | \
           ((np.abs(np.abs(b) - LW[1] / 2) < BAND_M) & (np.abs(a) < LW[0] / 2 + BAND_M))
    return P[near & ~inside_deep]


def motion_cost(p, P, T, dt, ref_pose):
    """Bin moved by (dx, dy, dyaw) about the reference bin centre."""
    dx, dy, dth = p
    cx, cy, _ = ref_pose
    c, s = np.cos(dth), np.sin(dth)
    Q = P.copy()
    Q[:, 0] = cx + c * (P[:, 0] - cx) - s * (P[:, 1] - cy) + dx
    Q[:, 1] = cy + s * (P[:, 0] - cx) + c * (P[:, 1] - cy) + dy
    uv, z = project(T, Q)
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    if ok.mean() < 0.3:
        return 1e3
    d = dt[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
    return float(np.mean(np.minimum(d, 15.0)) + 15.0 * (1 - ok.mean()))   # truncated chamfer, px


def track(T, ref_rgb, ref_pose, qry_rgb, search_m=0.12, search_deg=4.0):
    P = rim_band_points(T, ref_rgb, ref_pose)
    dt = cv2.distanceTransform((edges(qry_rgb) == 0).astype(np.uint8), cv2.DIST_L2, 5)
    best = None
    for dx in np.arange(-search_m, search_m + 1e-9, 0.02):
        for dy in np.arange(-search_m, search_m + 1e-9, 0.02):
            for dth in np.radians(np.arange(-search_deg, search_deg + 1e-9, 1.0)):
                c = motion_cost((dx, dy, dth), P, T, dt, ref_pose)
                if best is None or c < best[0]:
                    best = (c, (dx, dy, dth))
    r = minimize(motion_cost, best[1], args=(P, T, dt, ref_pose), method="Powell",
                 options=dict(xtol=1e-5, ftol=1e-4))
    return r.x, r.fun, len(P)
