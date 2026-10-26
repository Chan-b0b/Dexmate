"""Score wrist-photo bin tracking against head-depth truth on a chassis_offset capture.

Truth per stop = head-depth rim fit (bin is fixed; the robot moved). Wrist = edge
tracking against the step-0 reference photo (wrist_track). The wrist camera pose T
starts from the rough 15:56 silhouette fit; --calibrate refines T on the EVEN stops
(T making the reference edges land on each stop's edges when moved by the truth)
and the ODD stops are scored with the refined T (held out).

    ~/vla_venv/bin/python eval_offsets.py <capture_dir> [--calibrate]
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_track as wt  # noqa: E402
from fit_bin import fit, rim_points  # noqa: E402

LW = (0.720, 0.413)


def load(d):
    steps = {}
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f, allow_pickle=True)
        steps.setdefault(int(z["step"]), []).append(z)
    return steps


def head_truth(frames):
    ps = []
    for z in frames:
        P, _ = rim_points(dict(rgb=z["head_rgb"], depth=z["head_depth"], q_torso=z["q_torso"], q_head=z["q_head"]), local_top=True)
        ps.append(fit(P[:, :2], LW=LW))
    ps = np.array(ps)[:, :3]
    return ps.mean(0), np.ptp(ps, 0)


def perturb(T, d):
    D = np.eye(4); D[:3, :3] = R.from_rotvec(d[:3]).as_matrix(); D[:3, 3] = d[3:]
    return T @ D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    steps = load(args.capture_dir)
    truth, spread = {}, {}
    for k, fr in steps.items():
        truth[k], spread[k] = head_truth(fr)
    sp = np.array(list(spread.values()))
    print(f"head truth: 2-frame spread per stop max x {sp[:,0].max()*1000:.1f} y {sp[:,1].max()*1000:.1f} mm, yaw {np.degrees(sp[:,2].max()):.2f} deg")
    ref = steps[0][0]["right_wrist_camera_rgb"]
    ref_pose = tuple(truth[0])
    qry = {k: fr[0]["right_wrist_camera_rgb"] for k, fr in steps.items()}
    dts = {k: cv2.distanceTransform((wt.edges(im) == 0).astype(np.uint8), cv2.DIST_L2, 5) for k, im in qry.items()}
    delta = {k: np.r_[truth[k][:2] - truth[0][:2], truth[k][2] - truth[0][2]] for k in truth}
    T0 = np.load(Path(__file__).resolve().parent / "wrist_T_base_cam_1015.npy")

    T = T0
    if args.calibrate:
        train = [k for k in steps if k % 2 == 0 and k != 0]

        uv_ref = wt.rim_band_pixels(T0, ref, ref_pose)     # fixed pixel set (see rim_band_pixels)

        def cost(d):
            Tp = perturb(T0, d)
            P = wt.pixel_to_plane(Tp, uv_ref)
            return np.mean([wt.motion_cost(delta[k], P, Tp, dts[k], ref_pose) for k in train])
        best = None
        for start in ([0] * 6, [0.03, 0, 0, 0, 0, 0], [0, 0.03, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.03]):
            r = minimize(cost, start, method="Powell", options=dict(xtol=1e-4, ftol=1e-4, maxiter=3000))
            if best is None or r.fun < best.fun:
                best = r
        T = perturb(T0, best.x)
        if np.linalg.norm(best.x[3:]) > 0.10 or np.degrees(np.linalg.norm(best.x[:3])) > 10:
            print("calibration moved T implausibly far — keeping the rough T")
            T = T0
        print(f"calibrated on {len(train)} even stops: edge cost {cost(np.zeros(6)):.2f} -> {best.fun:.2f} px; "
              f"T moved {np.linalg.norm(best.x[3:])*1000:.1f} mm {np.degrees(np.linalg.norm(best.x[:3])):.2f} deg")
        np.save(Path(args.capture_dir) / "wrist_T_base_cam_offsets.npy", T)

    errs = {}
    for k in sorted(steps):
        if k == 0:
            continue
        p, c, n = wt.track(T, ref, ref_pose, qry[k], search_m=0.10, search_deg=4.0)
        e = np.r_[(p[:2] - delta[k][:2]) * 1000, np.degrees(p[2] - delta[k][2])]
        errs[k] = e
        tag = "" if not args.calibrate else ("train" if k % 2 == 0 else "TEST ")
        print(f"stop {k:2d} {tag} head moved ({delta[k][0]*1000:+6.1f},{delta[k][1]*1000:+6.1f}) mm {np.degrees(delta[k][2]):+.2f} | "
              f"wrist ({p[0]*1000:+6.1f},{p[1]*1000:+6.1f}) {np.degrees(p[2]):+.2f} (cost {c:.2f}) | error ({e[0]:+5.1f},{e[1]:+5.1f}) mm {e[2]:+.2f} deg")
    for name, ks in (("all", list(errs)), ("odd (held out)", [k for k in errs if k % 2])):
        E = np.array([errs[k] for k in ks])
        print(f"{name:15s}: |xy error| median {np.median(np.linalg.norm(E[:, :2], axis=1)):.1f} mm, max {np.linalg.norm(E[:, :2], axis=1).max():.1f} mm, "
              f"within 5 mm {np.mean(np.linalg.norm(E[:, :2], axis=1) <= 5)*100:.0f}% | yaw |err| max {np.abs(E[:, 2]).max():.2f} deg")


if __name__ == "__main__":
    main()
