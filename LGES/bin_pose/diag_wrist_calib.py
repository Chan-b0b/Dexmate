"""Which assumption breaks the wrist calibration? Re-solve X (mount) + Y (tag) with
extra free parameters and compare the reprojection rms:
  base | +lens distortion (k1,k2,p1,p2,k3) | +intrinsics (fx,fy,cx,cy) | +arm joint offsets (7)."""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solve_wrist_calib import K_WRIST, detect, tag_object_points, to_T, to_x  # noqa: E402
from ik_demo import config as cfg  # noqa: E402
from ik_demo.arm import ArmMover  # noqa: E402

D = sys.argv[1]
s = 0.150
zs = [np.load(f) for f in sorted(glob.glob(f"{D}/*_view*_0.npz"))]
uvs = [detect(z["right_wrist_camera_rgb"]) for z in zs]
keep = [i for i, u in enumerate(uvs) if u is not None]
zs, uvs = [zs[i] for i in keep], [uvs[i] for i in keep]
cfg.TORSO_JOINTS = tuple(float(v) for v in zs[0]["q_torso"])
arm = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
qs = [np.asarray(z["q_right_arm"], float) for z in zs]
obj = tag_object_points(s)
X0 = np.load("wrist_mount_tag.npy")


def T_ee(q):
    pos, rpy = arm.fk(q)
    T = np.eye(4); T[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); T[:3, 3] = pos
    return T


Y0 = T_ee(qs[0]) @ X0 @ to_T(np.r_[cv2.solvePnP(obj, uvs[0], K_WRIST, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)[1:3]].ravel()
                              if False else np.zeros(6))


def run(n_dist, n_K, n_joint):
    def unpack(p):
        X, Y = to_T(p[:6]), to_T(p[6:12]); i = 12
        dist = np.zeros(5); dist[:n_dist] = p[i:i + n_dist]; i += n_dist
        K = K_WRIST.copy()
        if n_K:
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = K_WRIST[0, 0] + p[i], K_WRIST[1, 1] + p[i + 1], K_WRIST[0, 2] + p[i + 2], K_WRIST[1, 2] + p[i + 3]
            i += 4
        dq = p[i:i + n_joint] if n_joint else np.zeros(7)
        return X, Y, dist, K, dq

    def resid(p):
        X, Y, dist, K, dq = unpack(p)
        out = []
        for q, uv in zip(qs, uvs):
            Tc = np.linalg.inv(T_ee(q + dq) @ X) @ Y
            rv, _ = cv2.Rodrigues(Tc[:3, :3])
            proj, _ = cv2.projectPoints(obj, rv, Tc[:3, 3], K, dist)
            out.append((proj.reshape(4, 2) - uv).ravel())
        return np.concatenate(out)
    # Y init from view 0 PnP through X0
    ok, rv, tv = cv2.solvePnP(obj, uvs[0], K_WRIST, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    Yi = T_ee(qs[0]) @ X0 @ to_T(np.r_[rv.ravel(), tv.ravel()])
    p0 = np.r_[to_x(X0), to_x(Yi), np.zeros(n_dist + (4 if n_K else 0) + n_joint)]
    r = least_squares(resid, p0, loss="huber", f_scale=1.0)
    e = r.fun.reshape(-1, 4, 2)
    X, Y, dist, K, dq = unpack(r.x)
    return np.sqrt((e ** 2).sum(-1).mean()), np.sqrt((e ** 2).sum(-1).mean(-1)), X, Y, dist, K, dq


for name, args in (("base", (0, 0, 0)), ("+distortion", (5, 0, 0)), ("+intrinsics", (0, 1, 0)),
                   ("+distortion+intrinsics", (5, 1, 0)), ("+joint offsets", (0, 0, 7)),
                   ("+all", (5, 1, 7))):
    rms, per, X, Y, dist, K, dq = run(*args)
    extra = []
    if args[0]: extra.append("dist " + str(np.round(dist, 3).tolist()))
    if args[1]: extra.append(f"fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}")
    if args[2]: extra.append("joint offsets deg " + str(np.round(np.degrees(dq), 2).tolist()))
    print(f"{name:24s} rms {rms:5.2f} px (worst view {per.max():5.2f}) | tag z {Y[2,3]:.4f} | {' | '.join(extra)}")


def home_cam(X, dq):
    return T_ee(np.asarray(cfg.HOME_JOINTS_RIGHT, float) + dq) @ X


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--home":
    probe = np.array([[0.80, 0.05, 0.89], [0.72, -0.05, 0.89], [0.80, 0.05, 0.75]])   # bin centre / run-time bin / tag height
    sols = {}
    for name, args in (("base", (0, 0, 0)), ("+joint offsets", (0, 0, 7)), ("+all", (5, 1, 7))):
        rms, per, X, Y, dist, K, dq = run(*args)
        sols[name] = home_cam(X, dq)
    ref = sols["+joint offsets"]
    for name, Tc in sols.items():
        dR = np.degrees(np.linalg.norm(R.from_matrix(ref[:3, :3].T @ Tc[:3, :3]).as_rotvec()))
        # where a point seen on the camera ray of each probe lands: compare by back-projecting the
        # same pixel through each camera onto the probe's height
        errs = []
        for P in probe:
            q = np.linalg.inv(ref) @ np.r_[P, 1]
            ray = Tc[:3, :3] @ (q[:3] / q[2])
            t = (P[2] - Tc[2, 3]) / ray[2]
            errs.append(np.linalg.norm(Tc[:3, 3] + t * ray - P) * 1000)
        print(f"home camera, {name:15s}: pos ({Tc[0,3]:.4f},{Tc[1,3]:+.4f},{Tc[2,3]:.4f}) | vs +joint offsets: "
              f"{np.linalg.norm(Tc[:3,3]-ref[:3,3])*1000:5.1f} mm {dR:.2f} deg -> same pixel lands "
              f"{', '.join(f'{e:.1f}' for e in errs)} mm apart at the bin")
