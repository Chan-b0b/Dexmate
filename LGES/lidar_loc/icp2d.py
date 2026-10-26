"""Planar (x, y, yaw) point-to-plane ICP between two base_link lidar clouds."""
import numpy as np
from scipy.spatial import cKDTree

def crop(P, x=(0.42, 2.5), y=(-1.5, 1.5), z=(0.03, 1.0)):
    m = (P[:,0]>x[0])&(P[:,0]<x[1])&(P[:,1]>y[0])&(P[:,1]<y[1])&(P[:,2]>z[0])&(P[:,2]<z[1])
    return P[m].astype(np.float64)

def voxel(P, v=0.01):
    k = np.floor(P / v).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[i]

def normals(P, k=12):
    t = cKDTree(P); _, idx = t.query(P, k)
    N = np.empty_like(P)
    for i, nb in enumerate(idx):
        Q = P[nb] - P[nb].mean(0); w, v = np.linalg.eigh(Q.T @ Q); N[i] = v[:, 0]
    return N

def icp(src, ref, iters=40, max_d=0.05):
    """Return (dx, dy, dyaw) s.t. R(dyaw) src + t ~= ref, plus inlier count/rms."""
    Nr = normals(ref); tree = cKDTree(ref)
    th = tx = ty = 0.0
    for it in range(iters):
        c, s = np.cos(th), np.sin(th)
        S = src.copy(); S[:,0] = c*src[:,0] - s*src[:,1] + tx; S[:,1] = s*src[:,0] + c*src[:,1] + ty
        d, j = tree.query(S, distance_upper_bound=max_d)
        ok = np.isfinite(d); S, R, N = S[ok], ref[j[ok]], Nr[j[ok]]
        hz = np.hypot(N[:,0], N[:,1]) > 0.3          # only vertical-ish surfaces constrain x/y/yaw
        S, R, N = S[hz], R[hz], N[hz]
        r = np.sum((S - R) * N, 1)
        J = np.stack([N[:,0], N[:,1], -S[:,1]*N[:,0] + S[:,0]*N[:,1]], 1)
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        tx += dx[0]; ty += dx[1]; th += dx[2]
        if np.abs(dx).max() < 1e-6: break
        max_d = max(0.01, max_d * 0.8)
    return tx, ty, th, len(r), float(np.sqrt(np.mean(r**2)))
