"""Run the detector over captured frames and report the verdict.

Two numbers decide feasibility of the depth-hole approach:
  * detection rate  — fraction of frames where the case was found
  * repeatability   — spread of pose across frames of the (static) case;
                      low spread => stable enough to grasp from.

    python run_test.py            # uses config.DATA_DIR -> config.OUT_DIR

Writes a diagnostic PNG per frame so you can eyeball where it succeeds/fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from diagnose import render
from paths import resolve_data_dir

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obb", action="store_true",
                    help="use the trained YOLO-OBB backend instead of depth-hole")
    ap.add_argument("--data", default=None,
                    help="capture run dir (default: newest data/<timestamp>/)")
    args = ap.parse_args()
    if args.obb:
        from detect_case_obb import detect_case
    else:
        from detect_case import detect_case

    data = resolve_data_dir(args.data, "case")
    out = HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(data.glob("frame_*.npz"))
    if not frames:
        print(f"No frames in {data}. Run capture.py first.")
        return 1
    print(f"reading {len(frames)} frames from {data}")

    centers, angles, scores, n_found = [], [], [], 0
    print(f"{'frame':<14}{'found':<7}{'score':<7}{'ang':<7}{'x,y,z (cam, m)'}")
    print("-" * 70)
    for fp in frames:
        f = np.load(fp)
        det = detect_case(f["depth"], f.get("rgb"))
        cv2.imwrite(str(out / (fp.stem + ".png")), render(f.get("rgb"), f["depth"], det))

        x, y, z = det.center_cam_m
        print(f"{fp.stem:<14}{str(det.found):<7}{det.size_score:<7.2f}"
              f"{det.angle_deg:<7.1f}({x:+.3f}, {y:+.3f}, {z:.3f})")
        if det.found:
            n_found += 1
            centers.append(det.center_cam_m)
            angles.append(det.angle_deg)
            scores.append(det.size_score)

    n = len(frames)
    print("-" * 70)
    print(f"detection rate : {n_found}/{n} ({100*n_found/n:.0f}%)")
    if n_found >= 2:
        c = np.array(centers)
        # angle spread, mindful of the 180-deg wrap
        a = np.deg2rad(np.array(angles) * 2)
        ang_std_deg = 0.5 * np.rad2deg(np.std(np.unwrap(a)))
        print(f"mean score     : {np.mean(scores):.2f}")
        print(f"pose repeatability (std across frames):")
        print(f"    x={c[:,0].std()*1000:.1f} mm  y={c[:,1].std()*1000:.1f} mm  "
              f"z={c[:,2].std()*1000:.1f} mm  yaw={ang_std_deg:.1f} deg")
    print(f"\nDiagnostics written to {out}/  — open the PNGs to inspect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
