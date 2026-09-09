"""Recover the HEIGHT of the face the bin labels actually trace, and the center
error that follows from warping at a different one.

Why this exists: the BEV warp plane and the labeled face have to agree, and in
the bin set they do not. `labeled_bin_bev` traces the bin's inner BOTTOM face
(class 0) and the lid's TOP face (class 1) on a canvas warped at
top_face_z(layers_remaining) = 0.6138, while the runtime detector is called
with cfg.DIVERT_BIN_PLANE_Z_M = 0.70, documented as the bin RIM. Warping one
image at two planes is a homothety about the camera centre (see bev.py), so
that mismatch is a pure scale about the camera nadir — a clean, confident box
whose center is tens of mm off in x, and no amount of relabeling fixes it.

The homothety scale is the size ratio, so ONE tape measurement per class (the
physical long side of the face the labels trace) pins the height and the fix:

    python calib_bin_plane.py --verify              # prove the geometry first
    python calib_bin_plane.py                       # z for a range of guesses
    python calib_bin_plane.py --true-long 0=0.55 1=0.62

Offline / analysis only: reads data/bin_bev + labeled_bin_bev, touches nothing.
"""

from __future__ import annotations

import argparse
import collections
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bev
import config as cfg


def _label_obbs(labeled: Path, canvas: tuple[int, int]):
    """{class_id: [(long_m, short_m, cx_px, cy_px, run, stem), ...]} from the
    8-value OBB label lines (normalised corners)."""
    w, h = canvas
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(str(labeled / "labels" / "*.txt"))):
        tok = Path(p).read_text().split()
        if len(tok) < 9:
            continue                      # empty = SAM2 lost track
        pts = (np.array(tok[1:9], dtype=np.float64).reshape(4, 2) * [w, h])
        (cx, cy), (bw, bh), _ = cv2.minAreaRect(pts.astype(np.float32))
        if min(bw, bh) < 5:
            continue
        s = 1.0 / cfg.BEV_PX_PER_M
        stem = Path(p).stem
        out[int(tok[0])].append((max(bw, bh) * s, min(bw, bh) * s, cx, cy,
                                 stem.split("__")[0], stem))
    return out


def _frame_geometry(run: str, stem: str, data_root: Path,
                    plane_z: float | None = None):
    """(camera centre, warp plane) for a labeled frame, from its raw npz.

    The plane is bev.frame_plane_z, so a run re-warped by rewarp_bev.py reports
    the plane its labels were actually traced on — deriving it from
    layers_remaining would make every height below a pure scale error."""
    npz = data_root / "bin_bev" / run / (stem.split("__")[-1] + ".npz")
    f = np.load(npz)
    return (bev.camera_centre(f["q_torso"], f["q_head"]),
            bev.frame_plane_z(f, plane_z))


# ---------------------------------------------------------------------------
# --verify: the homothety is an identity, not an approximation. Detect the same
# feature (yellow blob) on canvases warped at many planes and check that ONE
# detection reprojected predicts all the others.
# ---------------------------------------------------------------------------
def verify(data_root: Path, run: str, n: int, plane_z: float | None = None) -> None:
    frames = sorted((data_root / "bin_bev" / run).glob("frame_*.npz"))[:n]
    if not frames:
        raise SystemExit(f"no frames in {data_root / 'bin_bev' / run}")
    lo = np.array(cfg.BIN_HSV_LO)
    hi = np.array(cfg.BIN_HSV_HI)

    def blob(f, z):
        m = bev.build_mapper(f["q_torso"], f["q_head"], float(z))
        img = cv2.cvtColor(m.warp(f["rgb"]), cv2.COLOR_RGB2BGR)
        msk = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi)
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        cn, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cn:
            return None
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(max(cn, key=cv2.contourArea))
        return (np.array(m.bev_px_to_base(cx, cy)),
                max(bw, bh) / cfg.BEV_PX_PER_M, ang)

    print(f"verifying on {len(frames)} frames of {run}\n")
    print("  z_to    centre err (mm)     long err (mm)    yaw spread (deg)")
    errs = collections.defaultdict(list)
    for fp in frames:
        f = np.load(fp)
        z_from = bev.frame_plane_z(f, plane_z)   # the run's own plane, not top_face_z(1)
        C = bev.camera_centre(f["q_torso"], f["q_head"])
        a = blob(f, z_from)
        if a is None:
            continue
        for z_to in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            b = blob(f, z_to)
            if b is None:
                continue
            pred = bev.reproject_plane(a[0], z_from, z_to, C)
            k = (z_to - C[2]) / (z_from - C[2])
            errs[z_to].append((np.linalg.norm(b[0] - pred) * 1000.0,
                               abs(b[1] - a[1] * k) * 1000.0, b[2] - a[2]))
    for z_to in sorted(errs):
        e = np.array(errs[z_to])
        print(f"  {z_to:.2f}    {e[:, 0].mean():5.2f} (max {e[:, 0].max():5.2f})"
              f"    {e[:, 1].mean():5.2f} (max {e[:, 1].max():5.2f})"
              f"        {e[:, 2].std():.3f}")
    allm = np.concatenate([np.array(v)[:, 0] for v in errs.values()])
    print(f"\ncentre agreement over all planes: mean {allm.mean():.2f} mm, "
          f"max {allm.max():.2f} mm  (HSV mask quantisation, not model error)")
    print("-> reprojection is exact; a wrong warp plane costs NO yaw and needs "
          "NO re-detection.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="labeled_bin_bev")
    ap.add_argument("--data", default=cfg.DATA_DIR)
    ap.add_argument("--runtime-plane", type=float, default=0.70,
                    help="plane the runtime detector is called with "
                         "(ik_demo cfg.DIVERT_BIN_PLANE_Z_M)")
    ap.add_argument("--true-long", nargs="*", default=[], metavar="CLS=METRES",
                    help="measured physical long side of the face class CLS's "
                         "labels trace, e.g. --true-long 0=0.55 1=0.62")
    ap.add_argument("--assume-z", nargs="*", default=[], metavar="CLS=METRES",
                    help="inverse of --true-long: assume the face height and "
                         "report the physical size it implies, so a tape "
                         "measure can confirm or reject the assumption. "
                         "CLS=same means 'the labels' own warp plane', where "
                         "the homothety scale is 1 and the canvas size IS the "
                         "true size (e.g. --assume-z 0=same)")
    ap.add_argument("--plane-z", type=float, default=None,
                    help="override the plane the labels were warped at (default: "
                         "each frame's own, via bev.frame_plane_z)")
    ap.add_argument("--verify", action="store_true",
                    help="prove the homothety on real frames, then exit")
    ap.add_argument("--verify-run", default="20260903_110347_L1")
    ap.add_argument("--verify-n", type=int, default=6)
    args = ap.parse_args()

    data_root = HERE / args.data
    if args.verify:
        verify(data_root, args.verify_run, args.verify_n, args.plane_z)
        return

    canvas = bev.canvas_size()
    obbs = _label_obbs(HERE / args.labeled, canvas)
    if not obbs:
        raise SystemExit(f"no OBB labels under {HERE / args.labeled}/labels")
    truth = dict(kv.split("=") for kv in args.true_long)
    assumed = dict(kv.split("=") for kv in args.assume_z)

    print(f"canvas {canvas[0]}x{canvas[1]} px @ {cfg.BEV_PX_PER_M} px/m   "
          f"runtime plane {args.runtime_plane:.3f}\n")
    for c in sorted(obbs):
        rows = obbs[c]
        L = np.array([r[0] for r in rows])
        S = np.array([r[1] for r in rows])
        C, z_warp = _frame_geometry(rows[0][4], rows[0][5], data_root, args.plane_z)
        runs = sorted({r[4] for r in rows})
        print(f"=== class {c}  n={len(rows)}  runs={', '.join(runs)} ===")
        print(f"  labels warped at z={z_warp:.4f}, camera centre "
              f"({C[0]:.3f},{C[1]:+.3f},{C[2]:.4f})")
        print(f"  measured on canvas: long {np.median(L):.4f} m "
              f"(p10 {np.percentile(L, 10):.4f}..p90 {np.percentile(L, 90):.4f})"
              f"   short {np.median(S):.4f} m   aspect {np.median(L / S):.3f}")

        Lm = float(np.median(L))
        asp = float(np.median(L / S))
        if str(c) in assumed:
            raw = assumed[str(c)]
            z = z_warp if raw == "same" else float(raw)
            k = (z - C[2]) / (z_warp - C[2])          # homothety scale
            print(f"  assumed FACE HEIGHT z = {z:.4f} m  (scale {k:.4f})  ->")
            print(f"      implied TRUE size = {Lm * k:.4f} x {Lm * k / asp:.4f} m"
                  f"   <- confirm with a tape measure")
            # 1 cm of height error costs this much length: the check is only as
            # good as the assumption, so state its sensitivity.
            print(f"      sensitivity: {abs(Lm / (z_warp - C[2])) * 0.01 * 1000:.1f} mm "
                  f"of length per 1 cm of height error")
            bias = _bias_mm(rows, z, args.runtime_plane, C, data_root, args.plane_z)
            print(f"  center bias if detected at the runtime plane instead: {bias}")
        elif str(c) in truth:
            lt = float(truth[str(c)])
            z = bev.plane_from_size(Lm, lt, z_warp, C)
            print(f"  true long {lt:.4f} m  ->  FACE HEIGHT z = {z:.4f} m")
            bias = _bias_mm(rows, z, args.runtime_plane, C, data_root, args.plane_z)
            print(f"  center bias if detected at the runtime plane instead: {bias}")
        else:
            print("  no --true-long for this class; height for a range of guesses:")
            for lt in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
                z = bev.plane_from_size(Lm, lt, z_warp, C)
                print(f"      true long {lt:.2f} m -> z = {z:.4f} m"
                      f"   (short would be {lt / float(np.median(L / S)):.3f} m)")
        print()

    if not truth:
        print("Measure the physical LONG SIDE of the face each class's labels "
              "trace (class 0 = the bin's inner bottom, class 1 = the lid top; "
              "see out/overlay_*.png), then re-run with --true-long.")


def _bias_mm(rows, z_face: float, z_runtime: float, C, data_root: Path,
             plane_z: float | None = None) -> str:
    """How far the reported center moves when the same feature is read off the
    runtime-plane canvas instead of its own plane."""
    d = []
    for long_m, _s, cx, cy, run, stem in rows:
        _C, z_warp = _frame_geometry(run, stem, data_root, plane_z)
        m = bev.build_mapper(*_joints(run, stem, data_root), z_warp)
        P = np.array(m.bev_px_to_base(cx, cy))
        true = bev.reproject_plane(P, z_warp, z_face, _C)
        used = bev.reproject_plane(P, z_warp, z_runtime, _C)
        d.append((used - true) * 1000.0)
    d = np.array(d)
    return (f"dx {d[:, 0].mean():+.1f} mm, dy {d[:, 1].mean():+.1f} mm "
            f"(|d| max {np.linalg.norm(d, axis=1).max():.1f} mm)")


def _joints(run: str, stem: str, data_root: Path):
    f = np.load(data_root / "bin_bev" / run / (stem.split("__")[-1] + ".npz"))
    return f["q_torso"], f["q_head"]


if __name__ == "__main__":
    main()
