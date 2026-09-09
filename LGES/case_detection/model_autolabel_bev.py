"""Pseudo-label capture_bev.py case data with the already-trained YOLO-OBB model.

Model-based counterpart of sam2_autolabel_bev.py: instead of clicking a keyframe
and propagating a SAM2 mask, the existing detector (cfg.OBB_MODEL_PATH) predicts
an oriented box on every BEV frame. Predictions that pass the gates below become
labels immediately; the rest go to pending_case_bev/manifest.csv for
hand-correction in obb_edit_server.py. The output layout is the same one
sam2_autolabel_bev.py writes, so review.py / prepare_dataset.py / train.py all
work unchanged.

Gates — all must pass, else the frame is flagged with the failing reason:
    conf        >= --accept-conf
    2nd box      not within --margin of the best (ambiguous frame)
    metric size  within --size-tol of cfg.CASE_BEV_SIZE_M (the BEV is metric)
    label_flags  empty (tiny / huge / touching the canvas border)

Runs on the BEV pngs capture_bev.py already saved. If the BEV config or the
capture --layer was wrong, the warp itself is stale — re-label that run with
sam2_autolabel_bev.py --rewarp instead.

    python model_autolabel_bev.py                       # newest data/case_bev run
    python model_autolabel_bev.py --data data/case_bev/20260903_133943_L1 --dedupe-pose

Then:  python obb_edit_server.py    # fix the flagged frames in a browser
       python review.py --images labeled_case_bev/images --labels labeled_case_bev/labels
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from obb_label import label_flags, points_to_yolo_line
from paths import resolve_data_dir

HERE = Path(__file__).resolve().parent

# The bin has its own weights (cfg.BIN_OBB_MODEL_PATH) and footprint, so this
# tool is case-only — no --target, unlike the SAM2 labelers.
TARGET = "case"
# --dedupe-pose: a frame is a near-duplicate of the last accepted one if the
# case center moved less than this and the yaw turned less than DUP_YAW_DEG.
DUP_DIST_M = 0.02
DUP_YAW_DEG = 3.0
# Manifest read back by obb_edit_server.py. Corner points are in BEV px, in the
# model's own order, so the editor can start from exactly what was predicted;
# they are empty for a "no-det" row (nothing to start from).
MANIFEST_COLS = ["stem", "src_png", "reason", "conf",
                 "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def _long_axis_deg(w: float, h: float, r_rad: float) -> float:
    """YOLO-OBB rotation -> long-axis orientation in degrees, [0,180).
    Same convention as detect_case_bev._obb_long_axis_deg (kept local to avoid
    importing the runtime module, which pulls in the robot kinematics)."""
    deg = np.rad2deg(r_rad)
    if w < h:
        deg += 90.0
    return float(deg % 180.0)


def _dets(model, png: Path, conf: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Every oriented box on `png` as (conf, xywhr, corners_px), best first."""
    res = model.predict(str(png), conf=conf, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return []
    confs = res.obb.conf.cpu().numpy()
    xywhr = res.obb.xywhr.cpu().numpy()
    corners = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    return [(float(confs[i]), xywhr[i], corners[i]) for i in np.argsort(-confs)]


def _reject_reason(dets, w: int, h: int, args) -> str | None:
    """Why the best prediction can't be trusted as a label, or None to accept."""
    if not dets:
        return "no-det"
    conf, (_cx, _cy, bw, bh, _r), corners = dets[0]
    if conf < args.accept_conf:
        return f"low-conf {conf:.2f}"
    if len(dets) > 1 and dets[1][0] > conf - args.margin:
        return f"multi-det {len(dets)} (2nd {dets[1][0]:.2f})"
    long_m = max(bw, bh) / cfg.BEV_PX_PER_M
    short_m = min(bw, bh) / cfg.BEV_PX_PER_M
    exp_long, exp_short = max(cfg.CASE_BEV_SIZE_M), min(cfg.CASE_BEV_SIZE_M)
    if (abs(long_m - exp_long) / exp_long > args.size_tol
            or abs(short_m - exp_short) / exp_short > args.size_tol):
        return f"size {long_m:.2f}x{short_m:.2f}m"
    flags = label_flags(corners, w, h)
    return ",".join(flags) if flags else None


def _is_dup(prev: tuple[float, float, float], cx: float, cy: float, yaw: float) -> bool:
    """Same case pose as `prev` (BEV px center + long-axis yaw)?"""
    pcx, pcy, pyaw = prev
    dist_m = float(np.hypot(cx - pcx, cy - pcy)) / cfg.BEV_PX_PER_M
    dyaw = abs(yaw - pyaw) % 180.0
    return dist_m < DUP_DIST_M and min(dyaw, 180.0 - dyaw) < DUP_YAW_DEG


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["stem"]: r for r in csv.DictReader(f)}


def _write_manifest(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, MANIFEST_COLS)
        wr.writeheader()
        for stem in sorted(rows):
            wr.writerow(rows[stem])


def _manifest_row(stem: str, src: Path, reason: str, dets) -> dict:
    row = {"stem": stem, "src_png": str(src), "reason": reason, "conf": ""}
    row.update({c: "" for c in MANIFEST_COLS[4:]})
    if dets:
        conf, _xywhr, corners = dets[0]
        row["conf"] = f"{conf:.3f}"
        for i, (x, y) in enumerate(corners, start=1):
            row[f"x{i}"], row[f"y{i}"] = f"{x:.1f}", f"{y:.1f}"
    return row


def _load_model(weights: str | None):
    from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)

    path = Path(weights or cfg.OBB_MODEL_PATH)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        raise SystemExit(f"OBB weights not found at {path}. Set cfg.OBB_MODEL_PATH.")
    print(f"model {path}")
    return YOLO(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help=f"capture_bev run dir (default: newest data/{TARGET}_bev/)")
    ap.add_argument("--weights", default=None,
                    help="OBB weights to label with (default cfg.OBB_MODEL_PATH)")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="inference threshold — keep it LOW, the gates decide")
    ap.add_argument("--accept-conf", type=float, default=0.60,
                    help="min confidence to write a label without review")
    ap.add_argument("--margin", type=float, default=0.20,
                    help="flag the frame if a 2nd box is within this conf of the best")
    ap.add_argument("--size-tol", type=float, default=0.15,
                    help="max relative error per side vs cfg.CASE_BEV_SIZE_M")
    ap.add_argument("--dedupe-pose", action="store_true",
                    help=f"skip frames whose case pose moved <{DUP_DIST_M*100:.0f} cm "
                         f"and <{DUP_YAW_DEG:.0f} deg since the last accepted one")
    ap.add_argument("--force", action="store_true",
                    help="re-label frames that already have a label (overwrites "
                         "hand-corrected ones)")
    args = ap.parse_args()

    data = resolve_data_dir(args.data, f"{TARGET}_bev")
    pngs = sorted((data / "bev").glob("frame_*.png"))
    if not pngs:
        raise SystemExit(f"No BEV pngs in {data/'bev'}. Capture with: "
                         f"python capture_bev.py --target {TARGET}")
    print(f"using {len(pngs)} BEV frames from {data}")

    labeled = HERE / f"{cfg.LABELED_DIR}_{TARGET}_bev"
    pending = HERE / f"pending_{TARGET}_bev"
    out = HERE / cfg.OUT_DIR
    for d in ((labeled / "images"), (labeled / "labels"), pending, out):
        d.mkdir(parents=True, exist_ok=True)

    # Frames the editor marked as holding no case at all: never re-flag them.
    nocase_path = pending / "nocase.txt"
    nocase = set(nocase_path.read_text().split()) if nocase_path.exists() else set()
    manifest = _load_manifest(pending / "manifest.csv")

    model = _load_model(args.weights)
    n = Counter()
    prev_pose: tuple[float, float, float] | None = None

    for png in pngs:
        # Session-prefixed stem, like sam2_autolabel_bev.py, so multi-session
        # labeling accumulates in one labeled_case_bev/ (every run has frame_000).
        stem = f"{data.name}__{png.stem}"
        if stem in nocase:
            n["no-case (skipped)"] += 1
            continue
        label_path = labeled / "labels" / f"{stem}.txt"
        if label_path.exists() and not args.force:
            n["already labeled (kept)"] += 1
            continue

        img = cv2.imread(str(png))
        h, w = img.shape[:2]
        dets = _dets(model, png, args.conf)
        reason = _reject_reason(dets, w, h, args)

        if reason is not None:
            if label_path.exists():
                label_path.unlink()
                (labeled / "images" / f"{stem}.png").unlink(missing_ok=True)
                n["reverted (was labeled, now fails)"] += 1
            manifest[stem] = _manifest_row(stem, png, reason, dets)
            ov = img.copy()
            if dets:
                cv2.drawContours(ov, [dets[0][2].astype(np.int32)], 0, (0, 0, 255), 2)
            cv2.putText(ov, reason, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(out / f"flagged_{stem}.png"), ov)
            n[f"flagged: {reason.split()[0]}"] += 1
            continue

        conf, (cx, cy, bw, bh, r), corners = dets[0]
        yaw = _long_axis_deg(bw, bh, r)
        if args.dedupe_pose and prev_pose is not None and _is_dup(prev_pose, cx, cy, yaw):
            n["duplicate pose (skipped)"] += 1
            continue

        label_path.write_text(points_to_yolo_line(corners, w, h) + "\n")
        shutil.copy(png, labeled / "images" / f"{stem}.png")
        manifest.pop(stem, None)          # a re-run fixed a previously flagged frame
        prev_pose = (cx, cy, yaw)
        n["accepted"] += 1

    _write_manifest(pending / "manifest.csv", manifest)

    print()
    for k in sorted(n):
        print(f"  {n[k]:4d}  {k}")
    print(f"\nlabels -> {labeled}")
    if manifest:
        print(f"{len(manifest)} frame(s) need hand-correction "
              f"({pending/'manifest.csv'}, overlays in {out}/flagged_*.png)")
        print("Next: python obb_edit_server.py")
    else:
        print(f"Next: python review.py --images {labeled.name}/images "
              f"--labels {labeled.name}/labels")


if __name__ == "__main__":
    main()
