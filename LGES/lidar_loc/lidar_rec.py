"""Log-only lidar recorder (runs INSIDE lgcns_nav2_apriltag): every PERIOD_S save the
latest front+back clouds (base_link, cropped, 5mm voxel) to OUT/<host_time>.npz.
Commands nothing. Stops after MAX_S or Ctrl-C/kill."""
import os, sys, time, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lidar_rec"
PERIOD_S, MAX_S, VOX = 1.0, float(sys.argv[2]) if len(sys.argv) > 2 else 7200.0, 0.005
os.makedirs(OUT, exist_ok=True)

def to_xyz(m):
    r = pc2.read_points(m, field_names=("x", "y", "z"), skip_nans=True)
    P = np.stack([r["x"], r["y"], r["z"]], 1).astype(np.float32)
    P = P[(np.abs(P[:, 0]) < 4) & (np.abs(P[:, 1]) < 4) & (P[:, 2] > -0.1) & (P[:, 2] < 1.6)]
    _, i = np.unique(np.floor(P / VOX).astype(np.int32), axis=0, return_index=True)
    return P[i]

rclpy.init(); node = Node("lges_lidar_rec"); latest = {}
for k in ("front", "back"):
    node.create_subscription(PointCloud2, f"/lidar_3d_{k}/point_cloud",
                             lambda m, k=k: latest.__setitem__(k, (m, time.time())), qos_profile_sensor_data)
t0 = nxt = time.time(); n = 0
while time.time() - t0 < MAX_S:
    rclpy.spin_once(node, timeout_sec=0.05)
    if time.time() >= nxt and "front" in latest:
        nxt += PERIOD_S
        d = {}
        for k, (m, rx) in list(latest.items()):
            d[k] = to_xyz(m); d[k + "_stamp"] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9; d[k + "_rx"] = rx
        np.savez(f"{OUT}/{time.time():.3f}.npz", **d); n += 1
        if n % 60 == 1: print(f"[lidar rec] {n} frames -> {OUT}", flush=True)
