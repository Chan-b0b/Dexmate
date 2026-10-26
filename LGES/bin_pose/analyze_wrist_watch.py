"""Score the wrist-photo bin tracking on a real chassis_sequence run (wrist watch,
cfg.WRIST_WATCH_SAVE_FRAMES): would the right wrist camera alone have known where
the robot stood relative to the bin on each arrival?

Per station (target / left_bin), the FIRST arrival is the reference: its head-depth
bin pose and wrist photo. Every later arrival: wrist edge tracking against that
photo (wrist_track) -> raw bin motion -> linear correction
(wrist_motion_correction_1015.npy, fitted on the 1015 chassis_offset runs) ->
predicted bin pose, scored against that arrival's own head-depth bin pose. An edge
cost >= GATE_PX means "not sure" (the run would fall back to the view park).

    ~/vla_venv/bin/python analyze_wrist_watch.py ../ik_demo/run_logs/wrist_watch/<stamp>
"""
import glob
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0]))
import wrist_track as wt  # noqa: E402
from fit_bin import fit, rim_points  # noqa: E402

GATE_PX = 3.0
LW = (0.720, 0.413)


def head_tag(z, tag_m=0.150):
    """base <- tag (4x4) from the head camera (AprilTag 36h11 id 0 fixed on the table
    next to the bin, validation only), or None. Relative robot motion between two
    arrivals = how the tag moved in base — independent of what is in the bin."""
    import cv2
    sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
    import bev
    import camera_geometry as cg
    from solve_wrist_calib import detect, pnp
    uv = detect(z["head_rgb"])
    if uv is None:
        return None
    return cg.zed_left_camera_pose_from_joints(z["q_torso"], z["q_head"]) @ pnp(bev.intrinsic_matrix(), uv, tag_m)


def tag_truth(ref_pose, T_ref, T_now):
    """Ref bin pose (cx, cy, yaw) carried through the base motion the tag saw."""
    G = T_now @ np.linalg.inv(T_ref)                 # ref-base coords -> now-base coords
    c = G @ np.r_[ref_pose[0], ref_pose[1], 0.89, 1.0]
    dyaw = np.arctan2(G[1, 0], G[0, 0])
    return np.r_[c[0], c[1], ref_pose[2] + dyaw]


def head_bin(z):
    P, _ = rim_points(dict(rgb=z["head_rgb"], depth=z["head_depth"], q_torso=z["q_torso"],
                           q_head=z["q_head"]), local_top=True)
    return fit(P[:, :2], LW=LW)[:3]


def main():
    from ik_demo import config as cfg
    d = sys.argv[1]
    T = np.load(_HERE / "wrist_T_base_cam_1015.npy")
    M = np.load(_HERE / "wrist_motion_correction_1015.npy")
    home = np.asarray(cfg.HOME_JOINTS_RIGHT)
    visits = {}
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f, allow_pickle=True)
        visits.setdefault(str(z["station"]), []).append((Path(f).name, z))
    errs, rejected = [], 0
    for station, vs in visits.items():
        name0, z0 = vs[0]
        ref_pose = head_bin(z0)
        tag0 = head_tag(z0)
        print(f"== {station}: {len(vs)} arrivals; reference {name0} bin ({ref_pose[0]:.4f},{ref_pose[1]:+.4f}) yaw "
              f"{np.degrees(ref_pose[2]):+.2f}; truth = {'TAG' if tag0 is not None else 'head-depth rim (no tag seen)'}")
        for name, z in vs[1:]:
            if np.abs(z["q_right_arm"] - home).max() > 0.01:
                print(f"  {name}: right arm not at home — skipped")
                continue
            tag = head_tag(z) if tag0 is not None else None
            rim_truth = head_bin(z)
            truth = tag_truth(ref_pose, tag0, tag) if tag is not None else rim_truth
            p, cost, _ = wt.track(T, z0["right_wrist_camera_rgb"], tuple(ref_pose), z["right_wrist_camera_rgb"],
                                  search_m=0.10, search_deg=4.0)
            corr = np.r_[p[0] * 1000, p[1] * 1000, np.degrees(p[2]), 1.0] @ M      # mm, mm, deg
            pred = np.r_[ref_pose[:2] + corr[:2] / 1000, ref_pose[2] + np.radians(corr[2])]
            e = np.r_[(pred[:2] - truth[:2]) * 1000, np.degrees(pred[2] - truth[2])]
            ok = cost < GATE_PX
            rejected += not ok
            if ok:
                errs.append(e)
            moved = (truth[:2] - ref_pose[:2]) * 1000
            er = (rim_truth[:2] - truth[:2]) * 1000
            print(f"  {name:34s} robot vs reference ({moved[0]:+6.1f},{moved[1]:+6.1f}) mm [{'tag' if tag is not None else 'rim'}] | wrist error "
                  f"({e[0]:+5.1f},{e[1]:+5.1f}) mm {e[2]:+.2f} deg | edge cost {cost:.2f} {'ok' if ok else 'NOT SURE -> view park'}"
                  + (f" | head-depth rim vs tag ({er[0]:+5.1f},{er[1]:+5.1f}) mm" if tag is not None else ""))
    if errs:
        E = np.array(errs); dd = np.linalg.norm(E[:, :2], axis=1)
        print(f"\n{len(E)} scored arrivals (+{rejected} not sure): |xy error| median {np.median(dd):.1f} max {dd.max():.1f} mm, "
              f"<=5 mm {np.mean(dd <= 5) * 100:.0f}% | yaw |err| max {np.abs(E[:, 2]).max():.2f} deg")


if __name__ == "__main__":
    main()
