"""Self-test: verify the detector recovers a KNOWN rotated footprint hole.

Runs with no robot and no real data (numpy + opencv only). It proves the
detection LOGIC is correct, separate from whether real transparent-case frames
provide a usable signal (that's what run_test.py on captured data answers).

    python test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from detect_case import detect_case, _expected_dims_px


def make_depth_with_hole(z0: float, center, angle_deg: float, speckle: float = 0.0):
    """Floor at depth z0 with a footprint-sized invalid rectangle (the case)."""
    depth = np.full((cfg.IMG_H, cfg.IMG_W), z0, dtype=np.float32)
    long_px, short_px = _expected_dims_px(z0)
    rect = (center, (long_px, short_px), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(depth, box, 0.0)  # 0 == invalid stereo depth
    if speckle > 0:  # random invalid pixels, like real stereo noise
        noise = np.random.default_rng(0).random(depth.shape) < speckle
        depth[noise] = 0.0
    return depth


def angle_err_deg(a: float, b: float) -> float:
    """Difference modulo 180 (a rectangle's long axis is defined mod 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def main() -> int:
    z0 = 0.8
    cases = [  # (center_px, angle_deg)
        ((480, 300), 0.0),
        ((480, 300), 30.0),
        ((400, 250), 75.0),
        ((550, 350), 120.0),
    ]
    ok = True
    print(f"expected footprint @ z={z0}: {tuple(round(d,1) for d in _expected_dims_px(z0))} px")
    print("-" * 64)

    for center, ang in cases:
        depth = make_depth_with_hole(z0, center, ang, speckle=0.01)
        det = detect_case(depth)

        c_err = float(np.hypot(det.center_px[0] - center[0], det.center_px[1] - center[1]))
        a_err = angle_err_deg(det.angle_deg, ang)
        passed = det.found and c_err <= 6.0 and a_err <= 6.0
        ok &= passed

        print(f"[{'PASS' if passed else 'FAIL'}] truth c={center} ang={ang:5.1f}  ->  "
              f"found={det.found} c=({det.center_px[0]:.0f},{det.center_px[1]:.0f}) "
              f"ang={det.angle_deg:5.1f} score={det.size_score:.2f} "
              f"| c_err={c_err:.1f}px a_err={a_err:.1f}deg")

    # Negative control: a flat floor with no hole must NOT be detected.
    flat = np.full((cfg.IMG_H, cfg.IMG_W), z0, dtype=np.float32)
    neg = detect_case(flat)
    neg_ok = not neg.found
    ok &= neg_ok
    print(f"[{'PASS' if neg_ok else 'FAIL'}] no-hole floor -> found={neg.found} "
          f"(score={neg.size_score:.2f})")

    print("-" * 64)
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
