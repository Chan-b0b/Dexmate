"""Benchmark the YOLO-OBB detectors ik_demo/chassis_sequence.py uses.

No robot needed: replays saved capture frames (.npz with rgb/q_torso/q_head)
and times each stage separately —
    warp     bev.build_mapper + mapper.warp (homography + remap)
    predict  model.predict on the BEV image (preprocess + inference + NMS)
    total    warp + predict

Model load time is reported once, apart from the per-frame numbers. The first
--warmup iterations are discarded (CUDA init / cudnn autotune).

    python bench_detect.py data/show_bev/20260917_131515_L1/frame_001.npz
    python bench_detect.py data/show_bev/20260917_131515_L1/*.npz --target box --plane-z 0.9
    python bench_detect.py frame.npz --target case --iters 200

Live mode (real head-camera frames; the only motion is the head pitch, set to
--angle (default 24 deg, as chassis_sequence does) — --keep-head to skip it):

    python bench_detect.py --live --target case --layer 3 --iters 100
    python bench_detect.py --live --target bin --plane-z 0.75

Live adds a `grab` stage (camera get_obs + torso/head joint reads) to the
breakdown. Run inside the venv that has both dexcontrol and ultralytics
(~/vla_venv).
"""
from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

import bev
import config as cfg

TARGETS = {
    # target: (module, loader name, conf attr)
    "case": ("detect_case_bev", "load_model", "OBB_CONF"),
    "bin": ("detect_bin", "load_bev_model", "BIN_OBB_CONF"),
    "box": ("detect_box_bev", "load_model", "BOX_OBB_CONF"),
}


def _stats(xs_ms: list[float]) -> str:
    xs = sorted(xs_ms)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return (f"mean {statistics.fmean(xs):7.2f}  median {statistics.median(xs):7.2f}  "
            f"p95 {p95:7.2f}  min {xs[0]:7.2f}  max {xs[-1]:7.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*", help=".npz capture frame(s); cycled per iteration")
    ap.add_argument("--live", action="store_true",
                    help="grab real frames from the robot head camera instead of .npz files")
    ap.add_argument("--layer", type=int, default=1,
                    help="live: layers_remaining for the case warp plane (default 1)")
    ap.add_argument("--angle", type=float, default=24.0,
                    help="live: head pitch (deg) set before measuring — 24 = what "
                    "chassis_sequence pins for detection (default 24)")
    ap.add_argument("--keep-head", action="store_true",
                    help="live: do not move the head; measure from where it is")
    ap.add_argument("--target", choices=TARGETS, default="case")
    ap.add_argument("--weights", help="override the weights path")
    ap.add_argument("--plane-z", type=float, help="force the warp plane (m); "
                    "default = the plane recorded in / derived from each frame")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    mod_name, loader_name, conf_attr = TARGETS[args.target]
    mod = __import__(mod_name)
    conf = getattr(cfg, conf_attr)

    if args.live == bool(args.frames):
        ap.error("give either .npz frame(s) or --live")
    if args.live and args.target != "case" and args.plane_z is None:
        ap.error("--live with --target bin/box needs --plane-z")

    frames = []
    for p in args.frames:
        f = np.load(p)
        frames.append((f["rgb"], f["q_torso"], f["q_head"], bev.frame_plane_z(f, args.plane_z)))

    robot = None
    if args.live:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
        from dexcontrol.core.config import get_robot_config
        from dexcontrol.robot import Robot
        from utils import set_head_pitch
        configs = get_robot_config()
        configs.enable_sensor("head_camera")
        configs.sensors["head_camera"].transport = "zenoh"
        robot = Robot(configs=configs)
        # same as chassis_sequence: default AUTO policy idle-pauses the
        # subscribers and a resume clears their buffer (None on first read)
        for part in (robot.torso, robot.head):
            part.set_subscription_policy("always_on", recursive=True)
        robot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        if not args.keep_head:
            # same call as chassis_sequence; tol_deg returns at once if already aimed
            set_head_pitch(robot, angle=args.angle, tol_deg=2.0)

    def grab():
        obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
        rgb = obs.get("left_rgb")
        rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
        if rgb is None:
            return None
        plane = args.plane_z if args.plane_z is not None else bev.top_face_z(args.layer)
        return (rgb, np.asarray(robot.torso.get_joint_pos(), dtype=np.float64),
                np.asarray(robot.head.get_joint_pos(), dtype=np.float64), plane)

    t0 = time.perf_counter()
    model = getattr(mod, loader_name)(args.weights)
    load_s = time.perf_counter() - t0
    print(f"target={args.target} conf={conf} frames={len(frames)} "
          f"iters={args.iters} warmup={args.warmup}")
    print(f"model load: {load_s:.2f} s (once)")

    grab_ms, warp_ms, pred_ms, n_found, n_nogrid = [], [], [], 0, 0
    prev = first = None            # frame-similarity bookkeeping (outside the timed stages)
    n_same, mad_step, mad_first, centers = 0, [], [], []
    i = -1
    while len(pred_ms) < args.iters:
        i += 1
        tg = time.perf_counter()
        if args.live:
            fr = grab()
            if fr is None:
                n_nogrid += 1
                if n_nogrid > 200:
                    raise SystemExit("no camera frames")
                time.sleep(0.02)
                continue
        else:
            fr = frames[i % len(frames)]
        rgb, q_torso, q_head, plane_z = fr

        t0 = time.perf_counter()
        mapper = bev.build_mapper(q_torso, q_head, plane_z)
        bev_bgr = cv2.cvtColor(mapper.warp(rgb), cv2.COLOR_RGB2BGR)
        t1 = time.perf_counter()
        res = model.predict(bev_bgr, conf=conf, verbose=False)[0]
        t2 = time.perf_counter()

        if i < args.warmup:
            continue
        grab_ms.append((t0 - tg) * 1e3)
        warp_ms.append((t1 - t0) * 1e3)
        pred_ms.append((t2 - t1) * 1e3)
        found = res.obb is not None and len(res.obb) > 0
        n_found += found
        if found:
            k = int(np.argmax(res.obb.conf.cpu().numpy()))
            centers.append(res.obb.xywhr.cpu().numpy()[k][:2])
        if prev is not None:
            if np.array_equal(rgb, prev):
                n_same += 1
            else:
                mad_step.append(float(np.mean(cv2.absdiff(rgb, prev))))
            mad_first.append(float(np.mean(cv2.absdiff(rgb, first))))
        else:
            first = rgb
        prev = rgb

    total_ms = [w + p for w, p in zip(warp_ms, pred_ms)]
    if args.live:
        total_ms = [g + t for g, t in zip(grab_ms, total_ms)]
        robot.shutdown() if hasattr(robot, "shutdown") else None
    print(f"detections found in {n_found}/{args.iters} runs")
    if args.live:
        print(f"  grab    {_stats(grab_ms)}")
    print(f"  warp    {_stats(warp_ms)}")
    print(f"  predict {_stats(pred_ms)}")
    print(f"  total   {_stats(total_ms)}")
    n_pairs = len(pred_ms) - 1
    print(f"frame similarity over {n_pairs} consecutive pairs (RGB, 0-255 scale):")
    print(f"  byte-identical to previous frame: {n_same}/{n_pairs} "
          f"({100.0 * n_same / max(n_pairs, 1):.0f}%)  -> {n_pairs - n_same + 1} distinct frames")
    if mad_step:
        print(f"  mean |diff| between consecutive DISTINCT frames: "
              f"mean {statistics.fmean(mad_step):.3f}  max {max(mad_step):.3f}")
    if mad_first:
        print(f"  mean |diff| vs first frame (drift): last {mad_first[-1]:.3f}  max {max(mad_first):.3f}")
    if len(centers) > 1:
        c = np.asarray(centers)
        print(f"  detected OBB center jitter in BEV px: std x {c[:, 0].std():.3f}  y {c[:, 1].std():.3f}  "
              f"(range x {np.ptp(c[:, 0]):.2f}  y {np.ptp(c[:, 1]):.2f})")
    # model.predict's own breakdown of the last call (ultralytics, ms)
    sp = getattr(res, "speed", None)
    if sp:
        print("  last predict breakdown (ultralytics): "
              + "  ".join(f"{k} {v:.2f}" for k, v in sp.items()) + " ms")


if __name__ == "__main__":
    main()
