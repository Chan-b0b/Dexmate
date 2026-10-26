"""Solve the right wrist camera mount from wrist_calib_capture frames.

The tag lies still; the right arm visits several poses. For pose i the arm FK
gives T_base_ee(i), the wrist image gives the tag's 4 corners. Two unknowns:
    X = T_ee_cam   (the camera mount on the wrist)
    Y = T_base_tag (where the tag lies)
solved together by least squares on the corner reprojection error (pixels),
started from OpenCV's hand-eye solution. The head camera is NOT used for the
solve; if it also saw the tag, its own tag pose is compared with Y (= how far
the head camera's extrinsics are off).

    ~/vla_venv/bin/python solve_wrist_calib.py <capture_dir> [--tag-mm 150]
    ~/vla_venv/bin/python solve_wrist_calib.py --selftest
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))

K_WRIST = np.array([[366.61773681640625, 0, 479.5], [0, 366.3883056640625, 299.5], [0, 0, 1]])
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def tag_object_points(s):
    """Corner order of cv2.aruco / SOLVEPNP_IPPE_SQUARE, tag frame (z out of the tag)."""
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])


def detect(rgb, tag_id=0):
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or tag_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(tag_id)][0].astype(np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))   # refines in place
    return c.reshape(4, 2).astype(np.float64)


def to_T(x):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(x[:3]).as_matrix(); T[:3, 3] = x[3:]
    return T


def to_x(T):
    return np.r_[R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def project(K, T_cam_world, P):
    q = (T_cam_world @ np.c_[P, np.ones(len(P))].T)[:3]
    return ((K @ q)[:2] / q[2]).T


def pnp(K, uv, s):
    ok, rv, tv = cv2.solvePnP(tag_object_points(s), uv, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return to_T(np.r_[rv.ravel(), tv.ravel()]) if ok else None


def solve(T_base_ee, uvs, s, X0=None):
    """T_base_ee: list of 4x4; uvs: list of (4,2) wrist corners. Returns X, Y, rms px, per-view px."""
    T_cam_tag = [pnp(K_WRIST, uv, s) for uv in uvs]
    Rg = [T[:3, :3] for T in T_base_ee]; tg = [T[:3, 3] for T in T_base_ee]
    Rt = [T[:3, :3] for T in T_cam_tag]; tt = [T[:3, 3] for T in T_cam_tag]
    Rx, tx = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    U, _, Vt = np.linalg.svd(Rx)             # few / similar views: Rx can come back non-orthonormal
    Rx = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    Xh = np.eye(4); Xh[:3, :3] = Rx; Xh[:3, 3] = tx.ravel()
    inits = ([Xh] if np.all(np.isfinite(Xh)) else []) + ([X0] if X0 is not None else [])
    obj = tag_object_points(s)
    best = None
    for Xi in inits:
        Ys = [T_base_ee[i] @ Xi @ T_cam_tag[i] for i in range(len(uvs))]
        Yi = Ys[0].copy(); Yi[:3, 3] = np.mean([Y[:3, 3] for Y in Ys], 0)
        Yi[:3, :3] = R.from_matrix(np.array([Y[:3, :3] for Y in Ys])).mean().as_matrix()

        def resid(p):
            X, Y = to_T(p[:6]), to_T(p[6:])
            return np.concatenate([(project(K_WRIST, np.linalg.inv(T_base_ee[i] @ X) @ Y, obj) - uvs[i]).ravel()
                                   for i in range(len(uvs))])
        r = least_squares(resid, np.r_[to_x(Xi), to_x(Yi)], loss="huber", f_scale=1.0)
        if best is None or r.cost < best.cost:
            best = r
    e = best.fun.reshape(len(uvs), 4, 2)
    per = np.sqrt((e ** 2).sum(-1).mean(-1))
    return to_T(best.x[:6]), to_T(best.x[6:]), float(np.sqrt((e ** 2).sum(-1).mean())), per


def fk_ee(q_right, q_torso):
    from ik_demo import config as cfg
    from ik_demo.arm import ArmMover
    cfg.TORSO_JOINTS = tuple(float(v) for v in q_torso)
    arm = ArmMover(robot=None, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)
    out = []
    for q in q_right:
        pos, rpy = arm.fk(np.asarray(q, dtype=float))
        T = np.eye(4); T[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); T[:3, 3] = pos
        out.append(T)
    return out


def describe(X, home_T_ee):
    Tc = home_T_ee @ X
    zc = Tc[:3, 2]
    return (f"mount EE->cam t ({X[0,3]*1000:+.1f},{X[1,3]*1000:+.1f},{X[2,3]*1000:+.1f}) mm, cam axis in EE "
            f"{np.round(X[:3, 2], 3).tolist()}; at home: cam ({Tc[0,3]:.3f},{Tc[1,3]:+.3f},{Tc[2,3]:.3f}) "
            f"heading {np.degrees(np.arctan2(zc[1], zc[0])):+.1f} down {np.degrees(np.arcsin(-zc[2])):.1f} deg")


def selftest():
    """Known X, Y; poses from calib_poses; 0.3 px noise; start from the URDF mount (30 deg off)."""
    from ik_demo import config as cfg
    d = np.load(_HERE / "calib_poses_1015.npy", allow_pickle=True).item()
    qs = [q for q, k in zip(d["seq"], d["kinds"]) if k == "view"]
    Tee = fk_ee(qs, cfg.TORSO_JOINTS)
    X_true = np.load(_HERE / "wrist_mount_cal_1015.npy") @ to_T([0.02, -0.03, 0.01, 0.008, -0.012, 0.005])
    Y_true = np.eye(4); Y_true[:3, :3] = R.from_euler("z", 7, degrees=True).as_matrix(); Y_true[:3, 3] = d["tag"]
    rng = np.random.default_rng(1)
    s = 0.150
    uvs = [project(K_WRIST, np.linalg.inv(T @ X_true) @ Y_true, tag_object_points(s)) + rng.normal(0, 0.3, (4, 2)) for T in Tee]
    X0 = np.eye(4); X0[:3, :3] = R.from_euler("x", -0.087266).as_matrix(); X0[:3, 3] = [0.0002, -0.07459, 0.0017504]
    X, Y, rms, per = solve(Tee, uvs, s, X0=X0)
    dX = np.linalg.inv(X_true) @ X; dY = np.linalg.inv(Y_true) @ Y
    print(f"selftest: rms {rms:.2f} px | mount error {np.linalg.norm(dX[:3,3])*1000:.2f} mm "
          f"{np.degrees(np.linalg.norm(R.from_matrix(dX[:3,:3]).as_rotvec())):.3f} deg | tag error "
          f"{np.linalg.norm(dY[:3,3])*1000:.2f} mm {np.degrees(np.linalg.norm(R.from_matrix(dY[:3,:3]).as_rotvec())):.3f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", nargs="?")
    ap.add_argument("--tag-mm", type=float, default=150.0, help="MEASURED black-square side of the print")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    s = args.tag_mm / 1000.0
    files = sorted(glob.glob(f"{args.capture_dir}/*_view*.npz"))
    qs, uvs, used, heads = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        uv = detect(z["right_wrist_camera_rgb"])
        if uv is None:
            print(f"{Path(f).name}: tag NOT found in the wrist image — skipped")
            continue
        qs.append(z["q_right_arm"]); uvs.append(uv); used.append(f); heads.append(z)
    print(f"{len(uvs)}/{len(files)} wrist views with the tag")
    if len(uvs) < 4:
        raise SystemExit("need >= 4 views")
    q_torso = heads[0]["q_torso"]
    Tee = fk_ee(qs, q_torso)
    X0 = np.load(_HERE / "wrist_mount_cal_1015.npy")
    X, Y, rms, per = solve(Tee, uvs, s, X0=X0)
    for f, e in zip(used, per):
        print(f"  {Path(f).name}: {e:.2f} px")
    print(f"rms reprojection {rms:.2f} px over {len(uvs)} views")
    from ik_demo import config as cfg
    home_T = fk_ee([cfg.HOME_JOINTS_RIGHT], q_torso)[0]
    print(describe(X, home_T))
    print(f"tag in base: ({Y[0,3]:.4f},{Y[1,3]:+.4f},{Y[2,3]:.4f}), normal {np.round(Y[:3, 2], 3).tolist()}")
    np.save(Path(args.capture_dir) / "wrist_mount_tag.npy", X)
    np.save(_HERE / "wrist_mount_tag.npy", X)
    print(f"saved mount -> {_HERE / 'wrist_mount_tag.npy'}")
    # head camera vs the solved tag pose
    sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
    sys.path.insert(0, str(_HERE.parents[0] / "case_battery_demo" / "dashboard"))
    import bev
    import camera_geometry as cg
    errs = []
    for z in heads:
        uv = detect(z["head_rgb"])
        if uv is None:
            continue
        Th = cg.zed_left_camera_pose_from_joints(z["q_torso"], z["q_head"]) @ pnp(bev.intrinsic_matrix(), uv, s)
        dT = np.linalg.inv(Y) @ Th
        errs.append(np.r_[Th[:3, 3] - Y[:3, 3], np.degrees(R.from_matrix(dT[:3, :3]).as_rotvec())])
    if errs:
        e = np.mean(errs, 0)
        print(f"head camera sees the tag at ({e[0]*1000:+.1f},{e[1]*1000:+.1f},{e[2]*1000:+.1f}) mm from the wrist "
              f"solution, rotated {np.round(e[3:], 2).tolist()} deg (tag frame), over {len(errs)} frames")


if __name__ == "__main__":
    main()
