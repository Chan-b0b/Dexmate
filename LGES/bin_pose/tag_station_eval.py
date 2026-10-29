"""Station tags, step 2: how steady is a tag's base pose, and does the case stay
put in the tag's frame? (1026; data = held_view_capture dirs, tags 36h11 flat on
the table, black square --tag-mm.)

Per head frame and tag, the base-frame pose two ways:
  pnp    solvePnP (IPPE_SQUARE) -> T_base_cam (camera_geometry FK) @ T_cam_tag;
         its tilt (tag normal vs base z) says how well a single small oblique
         tag pins its own orientation (a flat tag should read ~0 deg)
  plane  the 4 corner rays cut with the table plane --plane-z (fixed: the PnP
         height of a small oblique tag swings by cm): x, y, yaw from those 4 points
Then the case detection (the same BEV detector as chassis_sequence) on the
frames with the arm out of the way (start / parked / return_parked) and the
case pose in the visible tag's plane frame.

    ~/vla_venv/bin/python tag_station_eval.py <capture_dir> [--tag-mm 104]
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from solve_wrist_calib import pnp
from tag_report import detect_all

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
import detect_case_bev as dcb  # noqa: E402

cg = dcb.bev.cg
CLEAR_VIEW = ("start", "parked", "return_parked")   # arm out of the head view


def set_intrinsics(capture_dir):
    f = Path(capture_dir) / "camera_info.json"
    if f.exists():
        i = json.loads(f.read_text())["head_camera"]["intrinsics"]["left"]
        cg.ZED_FX, cg.ZED_FY, cg.ZED_CX, cg.ZED_CY = i["fx"], i["fy"], i["cx"], i["cy"]
    return np.array([[cg.ZED_FX, 0, cg.ZED_CX], [0, cg.ZED_FY, cg.ZED_CY], [0, 0, 1]])


def plane_pose(uv, K, T_base_cam, z):
    """Corner rays ∩ plane z -> (x, y, yaw_deg) of the tag centre / its +x edge."""
    rays = np.linalg.inv(K) @ np.c_[uv, np.ones(4)].T            # cam frame, (3,4)
    d = T_base_cam[:3, :3] @ rays
    o = T_base_cam[:3, 3:4]
    P = o + d * ((z - o[2]) / d[2])                              # (3,4) base points
    c = P[:2].mean(axis=1)
    ex = (P[:2, 1] - P[:2, 0]) + (P[:2, 2] - P[:2, 3])           # corner 0->1 = tag +x
    return c[0], c[1], np.degrees(np.arctan2(ex[1], ex[0]))


def wrap90(a):
    return (a + 90.0) % 180.0 - 90.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag-mm", type=float, default=104.0)
    ap.add_argument("--plane-z", type=float, default=0.682,
                    help="table height the tags lie on (= ik_demo cfg.STATION_TAG_PLANE_Z_M)")
    args = ap.parse_args()
    s = args.tag_mm / 1000.0
    K = set_intrinsics(args.dir)
    rows = []
    for f in sorted(glob.glob(f"{args.dir}/*.npz")):
        z = np.load(f, allow_pickle=True)
        T_bc = cg.zed_left_camera_pose_from_joints(z["q_torso"], z["q_head"])
        for i, uv in detect_all(z["head_rgb"]).items():
            T = T_bc @ pnp(K, uv, s)
            tilt = np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))
            rows.append(dict(f=Path(f).stem, label=str(z["label"]), point=str(z["tag"]), id=i,
                             T=T, tilt=tilt, uv=uv, T_bc=T_bc, z=z))
    if not rows:
        sys.exit("no tags found")
    zt = {i: float(np.median([r["T"][2, 3] for r in rows if r["id"] == i]))
          for i in {r["id"] for r in rows}}
    for r in rows:
        r["plane"] = plane_pose(r["uv"], K, r["T_bc"], args.plane_z)

    print(f"tag {args.tag_mm:.1f} mm; table plane z {args.plane_z:.4f} (median pnp z, "
          "for reference: " + ", ".join(f"id{i} {v:.4f}" for i, v in sorted(zt.items())) + ")")
    print(f"\n{'frame':38s} {'id':>3s} | pnp x      y       z     tilt | plane x      y      yaw")
    for r in rows:
        T, (px, py, pyaw) = r["T"], r["plane"]
        print(f"{r['f']:38s} {r['id']:3d} | {T[0,3]:.4f} {T[1,3]:+.4f} {T[2,3]:.4f} {r['tilt']:5.1f} |"
              f" {px:.4f} {py:+.4f} {pyaw:+7.2f}")

    # steadiness: consecutive groups = same (label, point) = one chassis pose
    print("\nspread within one chassis pose (plane x / y mm, yaw deg; pnp z mm, tilt deg)")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["point"] if r["point"] not in ("carry", "parked", "placed")
                else "station", r["id"])].append(r)
    for (label, point, i), g in groups.items():
        P = np.array([r["plane"] for r in g])
        zz = np.array([r["T"][2, 3] for r in g]) * 1000
        tl = np.array([r["tilt"] for r in g])
        print(f"  {label:10s} {point:14s} id{i} n={len(g):2d}  x {np.ptp(P[:,0])*1000:4.1f}"
              f"  y {np.ptp(P[:,1])*1000:4.1f}  yaw {np.ptp(P[:,2]):5.2f}  | z {np.ptp(zz):4.1f}"
              f"  tilt {tl.min():4.1f}..{tl.max():4.1f}")

    # case in the tag frame, clear-view frames only
    print("\ncase detection in the visible tag's frame (clear-view frames)")
    print(f"{'frame':38s} {'id':>3s} | case base x  y      yaw  | in tag frame x  y     yaw")
    by_tag = defaultdict(list)
    for r in rows:
        if r["point"] not in CLEAR_VIEW:
            continue
        z = r["z"]
        det = dcb.detect_case_bev(z["head_rgb"], z["q_torso"], z["q_head"],
                                  layers_remaining=int(z["src_layers"]) if "start" in r["point"]
                                  or "return" in r["point"] else int(z["tgt_layers"]))
        if not det.found:
            print(f"{r['f']:38s} {r['id']:3d} | case not found")
            continue
        tx, ty, tyaw = r["plane"]
        a = np.radians(tyaw)
        dx, dy = det.base_xy[0] - tx, det.base_xy[1] - ty
        lx, ly = np.cos(a) * dx + np.sin(a) * dy, -np.sin(a) * dx + np.cos(a) * dy
        lyaw = wrap90(det.base_yaw_deg - tyaw)
        by_tag[r["id"]].append((r["f"], lx, ly, lyaw))
        print(f"{r['f']:38s} {r['id']:3d} | {det.base_xy[0]:.4f} {det.base_xy[1]:+.4f} "
              f"{wrap90(det.base_yaw_deg):+6.2f} | {lx:+.4f} {ly:+.4f} {lyaw:+6.2f}")
    for i, v in sorted(by_tag.items()):
        A = np.array([x[1:] for x in v])
        dyaw = wrap90(A[:, 2] - A[0, 2])          # the OBB long axis is 180-deg ambiguous
        print(f"  id{i}: n={len(v)}  spread in tag frame x {np.ptp(A[:,0])*1000:.1f} mm, "
              f"y {np.ptp(A[:,1])*1000:.1f} mm, yaw {np.ptp(dyaw):.2f} deg")


if __name__ == "__main__":
    main()
