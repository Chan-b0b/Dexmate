"""Render what a wrist_view candidate would see: the head depth cloud (true colours)
splatted into the virtual wrist camera with a z-buffer, the held case drawn as a
grey box, the finger strip shaded."""
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, ".")
from cloud import base_cloud  # noqa: E402
from wrist_view import (CAM_R, CAM_T, CASE_HALF, FINGER_V, H, W, held_case_box,  # noqa: E402
                        project)

frame, idx = sys.argv[1], [int(i) for i in sys.argv[2:]] or [0, 1]
cands = np.load("wrist_view_candidates.npy", allow_pickle=True)
d = np.load(frame)
P, C, _ = base_cloud(d, step=1)
box = held_case_box((0.751, -0.009, 0.86, np.radians(1.5)))
tiles = []
for i in idx:
    r = cands[i]
    Rm = R.from_euler("xyz", r["rpy"]).as_matrix()
    T = np.eye(4); T[:3, :3] = Rm @ CAM_R; T[:3, 3] = r["ee"] + Rm @ CAM_T
    u, v, z = project(T, P)
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img = np.zeros((H, W, 3), np.uint8); zb = np.full((H, W), np.inf)
    order = np.argsort(-z[ok])
    uu, vv, cc, zz = u[ok][order].astype(int), v[ok][order].astype(int), C[ok][order], z[ok][order]
    for du in (0, 1, 2):
        for dv in (0, 1, 2):
            a, b = np.clip(uu + du, 0, W - 1), np.clip(vv + dv, 0, H - 1)
            img[b, a] = cc
    # held case box faces
    c, yaw, (z0, z1) = box
    cs, sn = np.cos(yaw), np.sin(yaw)
    corners = []
    for zc in (z0, z1):
        for a_, b_ in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            x, y = a_ * CASE_HALF[0], b_ * CASE_HALF[1]
            corners.append((c[0] + cs * x - sn * y, c[1] + sn * x + cs * y, zc))
    cu, cv_, cz = project(T, np.array(corners))
    if (cz > 0).all():
        hull = cv2.convexHull(np.c_[cu, cv_].astype(np.int32))
        ov = img.copy(); cv2.fillConvexPoly(ov, hull, (150, 150, 150)); img = cv2.addWeighted(ov, 0.75, img, 0.25, 0)
    img[int(FINGER_V):] = (img[int(FINGER_V):] * 0.35).astype(np.uint8)
    cv2.putText(img, f"#{i} cam ({r['C'][0]:.2f},{r['C'][1]:+.2f},{r['C'][2]:.2f}) roll {r['roll']:+.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, f"case visible {r['vis']*100:.0f}%  rim in view {r['rim']*100:.0f}%", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, "grey = held case, dark strip = Robotiq fingers", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(img)
out = np.vstack([np.hstack(tiles[k:k + 2]) if len(tiles[k:k + 2]) == 2 else np.hstack([tiles[k], np.zeros_like(tiles[k])]) for k in range(0, len(tiles), 2)])
cv2.imwrite("wrist_view_sim_1015.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("wrote wrist_view_sim_1015.png")
