"""Offline: does lidar scan matching predict the camera's re-detections?

For each chain of camera detections of the SAME physical object (same station, object
not moved by the robot in between), take the lidar frame at each detection (robot
still), ICP it against the chain's first frame -> base motion (R, t), and predict the
object's base-frame xy at the later visit: c_j = R^T (c_i - t). Report pred - camera.

    python compare_run.py <run_log> <lidar_rec_dir>
"""
import glob
import re
import sys
from datetime import datetime

import numpy as np

from icp2d import crop, icp, voxel

LOG, REC = sys.argv[1], sys.argv[2]

def ts(line):
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f").timestamp()

dets, moving = [], []
for ln in open(LOG):
    m = re.search(r"detection refined over \d+ samples: xy=\(([-+\d.]+),([-+\d.]+)\) yaw=([-\d.]+)deg", ln)
    if m:
        dets.append((ts(ln), float(m[1]), float(m[2]), float(m[3])))
    m = re.search(r"chassis watch\] moved ([\d.]+) m .*?: ([\d:.]+) -> ([\d:.]+)", ln)
    if m:
        day = ln[:10]
        a, b = (datetime.strptime(f"{day} {x}", "%Y-%m-%d %H:%M:%S.%f").timestamp() for x in m.group(2, 3))
        moving.append((a, b))

files = sorted(glob.glob(f"{REC}/*.npz"))
fstamp = np.array([float(np.load(f)["front_stamp"]) for f in files])

def still(t, margin=0.3):
    return not any(a - margin <= t <= b + margin for a, b in moving)

def frame_at(t):
    """Nearest still lidar frame to detection time t (within 2 s)."""
    order = np.argsort(np.abs(fstamp - t))
    for k in order:
        if abs(fstamp[k] - t) > 2.0:
            break
        if still(fstamp[k]) and not any(a < min(t, fstamp[k]) < b or a < max(t, fstamp[k]) < b
                                        or min(t, fstamp[k]) < a < max(t, fstamp[k]) for a, b in moving):
            return k
    return None

CROPS = {
    "front, under table (z<0.55)": lambda d: crop(d["front"], z=(0.03, 0.55)),
    "front, all (z<1.0)": lambda d: crop(d["front"], z=(0.03, 1.0)),
    "front+back, z<0.55": lambda d: np.vstack([crop(d["front"], z=(0.03, 0.55)),
                                              crop(d["back"], x=(-4, -0.45), y=(-4, 4), z=(0.03, 0.55))]),
}

def det_index(hhmmss):
    for i, d in enumerate(dets):
        if datetime.fromtimestamp(d[0]).strftime("%H:%M:%S.%f")[:12].startswith(hhmmss):
            return i
    raise KeyError(hhmmss)

CHAINS = [l.split() for l in sys.argv[3:]] if len(sys.argv) > 3 else None
