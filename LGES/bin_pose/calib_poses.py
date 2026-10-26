"""Offline: right-arm poses for the wrist-camera calibration (AprilTag lying flat
on the battery tops in the bin in front of the robot).

Every pose must: reach (IK within REACH_TOL_M, joint band, no self-collision —
left arm at home), keep every right-arm point + the finger tips above the bins
over the table (x > 0.33 -> z >= 0.95), and see the whole tag well inside the
image (margin, above the finger strip, tag seen at < 55 deg off its normal).
From those, pick N with the most DIFFERENT camera orientations (rotation
diversity is what pins the mount), order them for short joint moves, and check
every joint-space leg (home -> 1 -> ... -> N -> home) the same way.

Aiming uses the rough mount from the 1015 bin-silhouette fit — good enough to
keep the tag in frame; the calibration then measures the real mount.

    ~/vla_venv/bin/python calib_poses.py [--tag x y z] [--n 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))
from ik_demo import config as cfg  # noqa: E402
from ik_demo.arm import ArmMover  # noqa: E402

sys.path.insert(0, str(_HERE))
from wrist_view import ray_hits_wall  # noqa: E402

W, H = 960, 600
K = np.array([[366.61773681640625, 0, 479.5], [0, 366.3883056640625, 299.5], [0, 0, 1]])
TAG_M = 0.150
MARGIN_PX, FINGER_V = 80, 0.70 * H
MAX_JOINT_FROM_HOME = 1.0   # rad, every joint: keep the views natural (1st try swung the wrist ~3 rad)
FINGER_LEN, TABLE_X0, ABOVE_BINS_Z = 0.18, 0.33, 0.95
LINKS = [f"R_arm_l{i}" for i in range(1, 8)] + [cfg.GRIPPER_EE_FRAME]


def look_rotation(C, L, roll):
    z = (L - C) / np.linalg.norm(L - C)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z] @ R.from_euler("z", roll).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs=3, type=float, default=(0.72, -0.05, 0.75))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mount", default=str(_HERE / "wrist_mount_cal_1015.npy"))
    ap.add_argument("--bin", nargs=3, type=float, default=None,
                    help="bin cx cy yaw_deg: drop views whose tag rays cross a bin wall")
    ap.add_argument("--rolls", nargs="+", type=float, default=[-45, 0, 45])
    ap.add_argument("--out", default=str(_HERE / "calib_poses_1015.npy"))
    args = ap.parse_args()
    M = np.load(args.mount)
    tag = np.array(args.tag)
    corners = tag + np.array([[a * TAG_M / 2, b * TAG_M / 2, 0] for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    arm = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
    fids = [arm._model.getFrameId(n) for n in LINKS]
    home = np.asarray(cfg.HOME_JOINTS_RIGHT, dtype=float)

    def above_bins(q):
        conf = arm._configuration(q)
        pin.framesForwardKinematics(arm._model, arm._data, conf.q)
        pts = [arm._data.oMf[f].translation.copy() for f in fids]
        T = arm._data.oMf[arm._ee_frame_id]
        pts.append(T.translation + T.rotation[:, 2] * FINGER_LEN)
        P = np.concatenate([np.linspace(a, b, 6) for a, b in zip(pts[:-1], pts[1:])])
        return not ((P[:, 0] > TABLE_X0) & (P[:, 2] < ABOVE_BINS_Z)).any()

    def cam_pose(q):
        pos, rpy = arm.fk(q)
        Te = np.eye(4); Te[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); Te[:3, 3] = pos
        return Te @ M

    def view_ok(Tc):
        q = (np.linalg.inv(Tc) @ np.c_[corners, np.ones(4)].T)[:3]
        if (q[2] < 0.25).any() or (q[2] > 0.9).any():
            return False
        uv = (K @ q)[:2] / q[2]
        if (uv[0] < MARGIN_PX).any() or (uv[0] > W - MARGIN_PX).any() or (uv[1] < MARGIN_PX).any() or (uv[1] > FINGER_V).any():
            return False
        ray = tag - Tc[:3, 3]
        if np.degrees(np.arccos(-ray[2] / np.linalg.norm(ray))) >= 55:
            return False
        if args.bin is not None and ray_hits_wall(Tc[:3, 3], corners, np.array(args.bin[:2]),
                                                   np.radians(args.bin[2])).any():
            return False
        return True

    def leg_ok(qa, qb):
        return all(not arm.in_collision(qa + t * (qb - qa)) and above_bins(qa + t * (qb - qa))
                   for t in np.linspace(0, 1, 21))

    rng = np.random.default_rng(0)
    cands = []
    for cx in np.arange(0.30, 0.81, 0.08):
        for cy in np.arange(-0.60, -0.19, 0.08):
            for cz in np.arange(1.00, 1.46, 0.08):
                C = np.array([cx, cy, cz])
                for roll in np.radians(args.rolls):
                    L = tag + np.r_[rng.uniform(-0.05, 0.05, 2), 0]
                    Rc = look_rotation(C, L, roll)
                    Ree = Rc @ M[:3, :3].T
                    pe = C - Ree @ M[:3, 3]
                    sol = arm.solve_pose(tuple(pe), tuple(R.from_matrix(Ree).as_euler("xyz")), seed=home)
                    if not (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision):
                        continue
                    if np.abs(sol.q - home).max() > MAX_JOINT_FROM_HOME:
                        continue
                    Tc = cam_pose(sol.q)
                    if view_ok(Tc) and above_bins(sol.q):
                        cands.append((sol.q, Tc))
    print(f"{len(cands)} candidate views")
    # greedy: most different camera orientation from those already picked (start: home's view)
    picked = [(home, cam_pose(home))] if view_ok(cam_pose(home)) else []
    print("home sees the tag:", bool(picked))
    def rot_gap(Ta, Tb):
        return np.degrees(np.linalg.norm(R.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()))
    pool = list(cands)
    while len(picked) < args.n and pool:
        scores = [min(rot_gap(Tc, p[1]) for p in picked) if picked else 0 for _, Tc in pool]
        picked.append(pool.pop(int(np.argmax(scores))))
    # order: nearest joint move from home, then check every leg (fallback: via home)
    order, cur, rest = [], home, picked[:]
    while rest:
        k = int(np.argmin([np.abs(q - cur).max() for q, _ in rest]))
        q, Tc = rest.pop(k)
        if not leg_ok(cur, q):
            if leg_ok(cur, home) and leg_ok(home, q):
                order.append(("via_home", home, None))
            else:
                print("  dropped a view: no clear joint leg to it")
                continue
        order.append(("view", q, Tc))
        cur = q
    if not leg_ok(cur, home):
        print("WARNING: last view -> home leg not clear")
    views = [o for o in order if o[0] == "view"]
    for i, (_, q, Tc) in enumerate(views):
        zc = Tc[:3, 2]
        print(f"view {i}: cam ({Tc[0,3]:.2f},{Tc[1,3]:+.2f},{Tc[2,3]:.2f}) heading {np.degrees(np.arctan2(zc[1], zc[0])):+4.0f} "
              f"down {np.degrees(np.arcsin(-zc[2])):3.0f} | q {np.round(q, 4).tolist()}")
    gaps = [rot_gap(a[2], b[2]) for i, a in enumerate(views) for b in views[i + 1:]]
    print(f"orientation spread: min pairwise {min(gaps):.0f} deg, max {max(gaps):.0f} deg")
    seq = [[float(v) for v in q] for kind, q, _ in order]
    kinds = [kind for kind, _, _ in order]
    np.save(args.out, np.array(dict(seq=seq, kinds=kinds, tag=args.tag), dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
