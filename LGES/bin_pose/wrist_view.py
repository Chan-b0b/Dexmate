"""Offline search: a right-arm pose whose wrist camera sees the destination bin
while the LEFT arm holds the case over it (the held_view_capture carry pose).

Camera model: ZED X One GS on the right wrist, looking along the gripper's
approach axis (EE +z; at home it sees the ceiling with the Robotiq fingers at
the image bottom). Mount = URDF R_camera_joint (dexs gripper, an approximation
for the Robotiq): 74.6mm along EE -y, tilted -5 deg about EE x; optical frame =
camera_link (x right = EE x, y down = EE y, z forward). Intrinsics from the
camera's own info (fx 366.6, SVGA 960x600).

For every candidate camera position (grid right of the bin) and roll, aim the
camera at the bin, solve the right arm IK with the LEFT arm frozen at its carry
pose in the collision model, then score from a real head-camera depth frame of
the destination bin:
  - placed-case visibility: case-top points in the wrist image, not behind the
    held case or a bin wall, clear of the finger strip at the image bottom;
  - head-view cost: bin rim points whose head-camera ray passes near the right arm;
  - clearance: right-arm links vs the held case box, along the joint-space path
    from home too.

    ~/vla_venv/bin/python wrist_view.py <obb_frame.npz> [--bin cx cy yaw_deg]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_HERE))
from ik_demo import config as cfg  # noqa: E402
from ik_demo.config import resolve_poses  # noqa: E402

from cloud import base_cloud  # noqa: E402

W, H = 960, 600
FX, FY, CX, CY = 366.61773681640625, 366.3883056640625, 479.5, 299.5
CAM_T = np.array([0.0002, -0.07459, 0.0017504])
CAM_R = R.from_euler("x", -0.087266).as_matrix()
FINGER_V = 0.72 * H            # rows below this: Robotiq fingers (from the home image)
BIN_LW, RIM_Z, FLOOR_Z = (0.720, 0.413), 0.89, 0.70
CASE_HALF = (0.25, 0.175)      # held case (CASE_BEV_SIZE_M / 2)
CASE_THICK = 0.08              # held case thickness (guess)
CLEAR_M = 0.08                 # min right-arm link clearance to the held case box
FINGER_LEN = 0.18              # R_gripper_base -> Robotiq finger tips along EE +z (conservative)
TABLE_X0 = 0.33                # base x where the table + bins begin
ABOVE_BINS_Z = RIM_Z + 0.06    # every arm point over the table stays above this
RIGHT_LINKS = [f"R_arm_l{i}" for i in range(1, 8)] + ["R_gripper_base"]


def held_case_box(center):
    """Axis-aligned-in-case-frame box of the held case at the carry height."""
    top = cfg.SAFE_TRANSPORT_Z - cfg.SUCTION_LENGTH_M
    return np.array(center[:2]), center[3], (top - CASE_THICK, top)


def in_box_xy(p, c, yaw, half, pad=0.0):
    d = p[..., :2] - c
    cs, sn = np.cos(yaw), np.sin(yaw)
    u, v = cs * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + cs * d[..., 1]
    return (np.abs(u) <= half[0] + pad) & (np.abs(v) <= half[1] + pad)


def ray_hits_case(C, P, box):
    """Does the segment C->P pass through the held case box (slab test, case frame)?"""
    c, yaw, (z0, z1) = box
    cs, sn = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
    o = Rz @ np.array([C[0] - c[0], C[1] - c[1], C[2]])
    d = (Rz @ (P - [c[0], c[1], 0]).T).T - o
    lo = np.array([-CASE_HALF[0], -CASE_HALF[1], z0])
    hi = np.array([CASE_HALF[0], CASE_HALF[1], z1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - o) / d, (hi - o) / d
    tmin = np.nanmax(np.minimum(t0, t1), axis=1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=1)
    return (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)


def ray_hits_wall(C, P, bc, byaw):
    """Segment C->P crosses a bin wall below the rim (at an interior point)."""
    cs, sn = np.cos(byaw), np.sin(byaw)
    def to_bin(X):
        d = X[..., :2] - bc
        return np.stack([cs * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + cs * d[..., 1], X[..., 2]], -1)
    c, p = to_bin(C[None])[0], to_bin(P)
    hit = np.zeros(len(P), bool)
    for ax, half in ((0, BIN_LW[0] / 2), (1, BIN_LW[1] / 2)):
        for s in (-1, 1):
            w = s * half
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w - c[ax]) / (p[:, ax] - c[ax])
            ok = (t > 0) & (t < 0.98)
            X = c + t[:, None] * (p - c)
            other = 1 - ax
            hit |= ok & (np.abs(X[:, other]) <= (BIN_LW[other] / 2)) & (X[:, 2] < RIM_Z) & (X[:, 2] > FLOOR_Z - 0.05)
    return hit


def project(T_base_cam, P):
    q = (np.linalg.inv(T_base_cam) @ np.c_[P, np.ones(len(P))].T)[:3].T
    z = q[:, 2]
    u, v = FX * q[:, 0] / np.maximum(z, 1e-6) + CX, FY * q[:, 1] / np.maximum(z, 1e-6) + CY
    return u, v, z


def look_rotation(C, L, roll):
    z = (L - C) / np.linalg.norm(L - C)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rc = np.c_[x, y, z] @ R.from_euler("z", roll).as_matrix()
    return Rc


def segment_dist(P, A, B):
    """Distance from points P (N,3) to segment AB."""
    ab = B - A
    t = np.clip(((P - A) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * ab), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--bin", nargs=3, type=float, default=(0.742, 0.019, 4.08))
    ap.add_argument("--case", nargs=3, type=float, default=(0.751, -0.009, 1.5),
                    help="held case centre x y yaw_deg (picked, carried straight over)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--mount", help="EE->camera 4x4 .npy (calibrated); default = URDF mount")
    ap.add_argument("--xs", nargs=3, type=float, default=(0.30, 0.91, 0.10), help="start stop step")
    ap.add_argument("--ys", nargs=3, type=float, default=(-0.60, -0.24, 0.07))
    ap.add_argument("--zs", nargs=3, type=float, default=(1.00, 1.45, 0.07))
    ap.add_argument("--rolls", nargs="+", type=float, default=[0, 45, 90, 135, 180, -135, -90, -45])
    ap.add_argument("--rank", choices=("case", "rim"), default="case",
                    help="case: placed-case visibility; rim: the worse of short-side / long-side rim visibility")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global CAM_T, CAM_R
    if args.mount:
        M = np.load(args.mount)
        CAM_R, CAM_T = M[:3, :3], M[:3, 3]

    from ik_demo.arm import ArmMover
    left = ArmMover(robot=None, side="left", ee_frame=cfg.EE_FRAME)
    center = (args.case[0], args.case[1], 0.86, np.radians(args.case[2]))
    pk = resolve_poses(center)["CASE_PICK"]
    lq = left.solve_pose((pk[0], pk[1], cfg.SAFE_TRANSPORT_Z), tuple(pk[3:]),
                         seed=np.asarray(cfg.HOME_JOINTS_LEFT), min_motion=False).q
    cfg.HOME_JOINTS_LEFT = tuple(float(v) for v in lq)     # collision model: left arm at carry
    right = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
    fids = [right._model.getFrameId(n) for n in RIGHT_LINKS]
    box = held_case_box(center)

    def link_points(q):
        cfg_ = right._configuration(q)
        pin.framesForwardKinematics(right._model, right._data, cfg_.q)
        return np.array([right._data.oMf[f].translation for f in fids])

    def clearance(q):
        pts = link_points(q)
        Tee = right._data.oMf[right._ee_frame_id]
        tip = Tee.translation + Tee.rotation[:, 2] * FINGER_LEN
        pts = np.vstack([pts, tip])
        dense = np.concatenate([np.linspace(a, b, 6) for a, b in zip(pts[:-1], pts[1:])])
        if ((dense[:, 0] > TABLE_X0) & (dense[:, 2] < ABOVE_BINS_Z)).any():
            return False, dense
        c, yaw, (z0, z1) = box
        inside = in_box_xy(dense, c, yaw, CASE_HALF, CLEAR_M) & (dense[:, 2] > z0 - CLEAR_M) & (dense[:, 2] < z1 + CLEAR_M)
        return not inside.any(), dense

    npz = np.load(args.frame)
    P, Cl, uv = base_cloud(npz)
    bc, byaw = np.array(args.bin[:2]), np.radians(args.bin[2])
    inner = in_box_xy(P, bc, byaw, (BIN_LW[0] / 2 - 0.03, BIN_LW[1] / 2 - 0.03)) & (P[:, 2] > FLOOR_Z - 0.01) & (P[:, 2] < 0.82)
    case_pts = P[inner][::3]
    rim = P[in_box_xy(P, bc, byaw, (BIN_LW[0] / 2 + 0.01, BIN_LW[1] / 2 + 0.01))
            & ~in_box_xy(P, bc, byaw, (BIN_LW[0] / 2 - 0.04, BIN_LW[1] / 2 - 0.04))
            & (np.abs(P[:, 2] - RIM_Z) < 0.03)][::3]
    cs_, sn_ = np.cos(byaw), np.sin(byaw)
    rim_short = np.abs(cs_ * (rim[:, 0] - bc[0]) + sn_ * (rim[:, 1] - bc[1])) > BIN_LW[0] / 2 - 0.05
    import camera_geometry as cg
    head_c = cg.zed_left_camera_pose_from_joints(npz["q_torso"], npz["q_head"])[:3, 3]
    L = np.array([bc[0], bc[1], 0.76])
    home_q = np.asarray(cfg.HOME_JOINTS_RIGHT)
    print(f"{len(case_pts)} placed-case pts, {len(rim)} rim pts; left carry q {np.round(lq, 3).tolist()}")

    res = []
    for cx in np.arange(*args.xs):
        for cy in np.arange(*args.ys):
            for cz in np.arange(*args.zs):
                C = np.array([cx, cy, cz])
                for roll in np.radians(args.rolls):
                    Rc = look_rotation(C, L, roll)
                    Ree = Rc @ CAM_R.T
                    pe = C - Ree @ CAM_T
                    rpy = R.from_matrix(Ree).as_euler("xyz")
                    sol = right.solve_pose(tuple(pe), tuple(rpy), seed=home_q, min_motion=False)
                    if not (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision):
                        continue
                    pos, rpy2 = right.fk(sol.q)
                    ang = np.degrees(np.arccos(np.clip(R.from_euler("xyz", rpy2).as_matrix()[:, 2] @ Ree[:, 2], -1, 1)))
                    if ang > 5:
                        continue
                    ok, dense = clearance(sol.q)
                    if not ok:
                        continue
                    path_ok = all(clearance(home_q + t * (sol.q - home_q))[0] and not right.in_collision(home_q + t * (sol.q - home_q))
                                  for t in np.linspace(0, 1, 15))
                    T = np.eye(4); T[:3, :3] = R.from_euler("xyz", rpy2).as_matrix() @ CAM_R; T[:3, 3] = pos + R.from_euler("xyz", rpy2).as_matrix() @ CAM_T
                    u, v, z = project(T, case_pts)
                    vis = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < FINGER_V)
                    vis &= ~ray_hits_case(T[:3, 3], case_pts, box) & ~ray_hits_wall(T[:3, 3], case_pts, bc, byaw)
                    ru, rv, rz = project(T, rim)
                    rin = (rz > 0.1) & (ru >= 0) & (ru < W) & (rv >= 0) & (rv < FINGER_V)
                    rim_vis = rin.mean()
                    rclear = rin & ~ray_hits_case(T[:3, 3], rim, box)
                    rshort_vis = rclear[rim_short].mean() if rim_short.any() else 0.0
                    rlong_vis = rclear[~rim_short].mean()
                    head_block = np.zeros(len(rim), bool)
                    for a, b in zip(dense[:-1], dense[1:]):
                        mid = (a + b) / 2
                        # rim point blocked if the arm segment passes within 6 cm of its head ray
                        t = np.clip(((mid - head_c) @ (rim - head_c).T) / np.sum((rim - head_c) ** 2, 1), 0, 1)
                        head_block |= np.linalg.norm(head_c + t[:, None] * (rim - head_c) - mid, axis=1) < 0.06
                    zc = T[:3, 2]
                    res.append(dict(heading=np.degrees(np.arctan2(zc[1], zc[0])),
                                    elev=np.degrees(np.arcsin(-zc[2])), C=C, roll=np.degrees(roll), q=sol.q, ee=pos, rpy=rpy2, vis=vis.mean(),
                                    rim=rim_vis, short=rshort_vis, long=rlong_vis, head=head_block.mean(), path=path_ok, dist=np.linalg.norm(L - C)))
    print(f"{len(res)} reachable, clear candidates")
    key = {"case": lambda r: r["vis"] - r["head"],
           "rim": lambda r: min(r["short"], r["long"]) - r["head"]}[args.rank]
    res.sort(key=lambda r: (-(r["path"]), -key(r)))
    for r in res[:args.top]:
        print(f"cam ({r['C'][0]:.2f},{r['C'][1]:+.2f},{r['C'][2]:.2f}) roll {r['roll']:+4.0f} heading {r['heading']:+4.0f} down {r['elev']:3.0f} | case visible {r['vis']*100:4.0f}% "
              f"rim in view {r['rim']*100:4.0f}% (unoccluded: short sides {r['short']*100:3.0f}% long {r['long']*100:3.0f}%) | head rim blocked {r['head']*100:4.0f}% | path from home clear {r['path']} "
              f"| EE ({r['ee'][0]:.3f},{r['ee'][1]:+.3f},{r['ee'][2]:.3f}) | q {np.round(r['q'], 3).tolist()}")
    np.save(args.out or (_HERE / "wrist_view_candidates.npy"), np.array(res, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
