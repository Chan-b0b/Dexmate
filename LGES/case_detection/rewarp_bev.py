"""Re-warp a capture_bev.py run onto a different BEV plane.

capture_bev.py picks the warp plane from --layer (top_face_z(layer)), so a run
aimed at something that is NOT a stack top face — a lid on the floor, a pallet —
comes out warped at the wrong height. The frames keep the raw rgb + joints, so
this is fully recoverable offline: rebuild the mapper at the right plane and
rewrite bev/*.png. Nothing is re-collected and nothing is approximated.

The run's plane is also recorded as a `plane_z` key in each npz, because the
plane is otherwise re-derived from `layers_remaining` (bev.top_face_z) and an
arbitrary height is not expressible that way — leaving it out would let the
label/self-test paths silently disagree with the images written here.

    python rewarp_bev.py data/bin_bev/20260904_164934_L1 --plane-z 0.345
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bev


def rewarp(run: Path, plane_z: float) -> int:
    frames = sorted(run.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"no frame_*.npz in {run}")
    bev_dir = run / "bev"
    bev_dir.mkdir(exist_ok=True)

    for p in frames:
        d = dict(np.load(p))
        m = bev.build_mapper(d["q_torso"], d["q_head"], plane_z)
        cv2.imwrite(str(bev_dir / f"{p.stem}.png"),
                    cv2.cvtColor(m.warp(d["rgb"]), cv2.COLOR_RGB2BGR))
        # Atomic, so an interrupted run never leaves a truncated npz.
        d["plane_z"] = np.float64(plane_z)
        tmp = p.with_name(f"{p.stem}.tmp.npz")
        np.savez_compressed(tmp, **d)
        os.replace(tmp, p)
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="capture_bev run dir (holds frame_*.npz + bev/)")
    ap.add_argument("--plane-z", type=float, required=True,
                    help="base_link z of the face to warp onto, m")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = HERE / run
    n = rewarp(run, args.plane_z)
    print(f"re-warped {n} frames of {run.name} at plane_z={args.plane_z:.4f}")


if __name__ == "__main__":
    main()
