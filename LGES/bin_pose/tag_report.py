"""Which AprilTags each captured frame sees (1026 station tags, step 1).

Tags 36h11 id 0..3 lie flat on the table, one right of each bin (left to right:
divert bin | id0 | source | id1 | idle target | id2 | task-2 pick bin | id3).
For every npz (held_view_capture / wrist watch) and every camera in it: the ids
found, their black-square side in px and the camera->tag distance. A tag that is
partly hidden is simply not found, so "missing" is the occlusion signal.

    ~/vla_venv/bin/python tag_report.py <capture_dir> [<capture_dir> ...] [--tag-mm 104]

Writes <capture_dir>/tag_report/<npz stem>.jpg (head | wrist, tags outlined).
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from solve_wrist_calib import DICT, K_WRIST, pnp

TAG_IDS = (0, 1, 2, 3)
# ZED X Mini left, SVGA (camera_info of the 1015 captures) — used when the dir has none
K_HEAD = np.array([[366.0962829589844, 0, 482.76666259765625],
                   [0, 366.0962829589844, 307.1612854003906], [0, 0, 1]])
CAMS = (("head", "head_rgb"), ("wrist", "right_wrist_camera_rgb"))


def load_K(capture_dir):
    """Per-camera K from camera_info.json (held_view_capture writes it), else the constants."""
    K = {"head": K_HEAD, "wrist": K_WRIST}
    f = Path(capture_dir) / "camera_info.json"
    if not f.exists():
        return K
    info = json.loads(f.read_text())
    for cam, key in (("head", "head_camera"), ("wrist", "right_wrist_camera")):
        try:
            i = info[key]["intrinsics"]
            i = i.get("left", i)
            K[cam] = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]])
        except (KeyError, TypeError):
            pass
    return K


def detect_all(rgb):
    """{id: (4,2) sub-pixel corners} for every 36h11 tag in the frame."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    out = {}
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        c = c[0].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tag-mm", type=float, default=104.0)
    args = ap.parse_args()
    s = args.tag_mm / 1000.0
    seen = defaultdict(lambda: defaultdict(int))     # (cam, point) -> id -> frames
    frames = defaultdict(int)
    for d in args.dirs:
        K = load_K(d)
        out = Path(d) / "tag_report"
        out.mkdir(exist_ok=True)
        for f in sorted(glob.glob(f"{d}/*.npz")):
            z = np.load(f, allow_pickle=True)
            point = (str(z["tag"]) if "tag" in z.files else
                     str(z["station"]) if "station" in z.files else "?")
            panels, parts = [], []
            for cam, key in CAMS:
                if key not in z.files:
                    continue
                rgb = z[key]
                tags = detect_all(rgb)
                frames[(cam, point)] += 1
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                desc = []
                for i, uv in sorted(tags.items()):
                    seen[(cam, point)][i] += 1
                    side = np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1))
                    T = pnp(K[cam], uv, s)
                    dist = np.linalg.norm(T[:3, 3]) if T is not None else float("nan")
                    desc.append(f"id{i} {side:.0f}px {dist:.2f}m")
                    cv2.polylines(vis, [uv.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(vis, f"id{i}", tuple(uv[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                parts.append(f"{cam}: " + (", ".join(desc) or "none"))
                cv2.putText(vis, f"{cam} {point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 2)
                panels.append(vis)
            if panels:
                h = min(p.shape[0] for p in panels)
                cv2.imwrite(str(out / (Path(f).stem + ".jpg")),
                            np.hstack([p[:h] for p in panels]))
            print(f"{Path(f).name:45s} {point:14s} " + " | ".join(parts))
    print("\nper camera / capture point: frames with each id (of N frames)")
    for (cam, point), n in sorted(frames.items()):
        ids = "  ".join(f"id{i} {seen[(cam, point)][i]}/{n}" for i in TAG_IDS)
        print(f"  {cam:5s} {point:14s} {ids}")


if __name__ == "__main__":
    main()
