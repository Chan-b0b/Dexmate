"""Auto-label the case dataset with SAM2 video propagation.

Click the case on one (or a few) keyframe(s); SAM2 propagates the mask across
the whole captured sequence; each mask becomes a YOLO-OBB label. Runs offline on
a GPU box (copy data/ over). Not used at robot runtime.

Pipeline:
    data/frame_*.npz  (full RGB frames, captured with the head held still)
      -> ONE median bin ROI for the sequence  (bin is static)
      -> crop every frame identically -> staged/<idx>.jpg   (fed to SAM2)
      -> click keyframe(s) -> propagate -> per-frame mask
      -> mask -> minAreaRect -> labeled/labels/<stem>.txt  (YOLO-OBB)
         + labeled/images/<stem>.png (the crop) + out/overlay_*.png

Then:  python review.py   ->  re-run with --keyframes including any drifted frames.

Setup (GPU box):
    pip install sam2          # or: pip install git+https://github.com/facebookresearch/sam2
    # download sam2.1_hiera_small.pt + its cfg; set cfg.SAM2_CHECKPOINT / SAM2_MODEL_CFG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from bin_roi import find_bin, inset_bbox
from obb_label import mask_to_bbox_line, mask_to_obb_points, mask_to_yolo_line
from paths import resolve_data_dir

HERE = Path(__file__).resolve().parent
OBJ_ID = 1  # single case per frame


# ---------------------------------------------------------------------------
# Stage 0 — one stable ROI for the whole sequence, crop, write numbered JPEGs
# ---------------------------------------------------------------------------
def median_bin_roi(frames: list[Path]) -> tuple[int, int, int, int]:
    boxes = []
    for fp in frames:
        bb = find_bin(np.load(fp)["rgb"])
        if bb is not None:
            boxes.append(bb)
    if not boxes:
        raise SystemExit("find_bin failed on every frame — check BIN_HSV_* in config.")
    med = np.median(np.array(boxes), axis=0).round().astype(int)
    return inset_bbox(tuple(int(v) for v in med))


def resolve_roi(frames: list[Path], source: str) -> tuple[int, int, int, int]:
    """The crop ROI used for the whole sequence. cfg.BIN_ROI overrides everything;
    else 'first' = frame-0 bin, 'median' = median across frames. Already inset."""
    if cfg.BIN_ROI is not None:
        return tuple(cfg.BIN_ROI)
    if source == "first":
        bb = find_bin(np.load(frames[0])["rgb"])
        if bb is None:
            raise SystemExit("find_bin failed on the first frame — try --roi median.")
        return inset_bbox(bb)
    return median_bin_roi(frames)


def stage(frames: list[Path], roi, staged: Path) -> tuple[Path, list[str]]:
    """Crop every frame by `roi` -> staged/<idx>.jpg (roi = whole frame for bin).
    Returns (dir, stems)."""
    x, y, w, h = roi
    staged.mkdir(parents=True, exist_ok=True)
    for p in staged.glob("*.jpg"):
        p.unlink()
    stems = []
    for i, fp in enumerate(frames):
        crop = np.load(fp)["rgb"][y:y + h, x:x + w]
        cv2.imwrite(str(staged / f"{i}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        stems.append(fp.stem)
    return staged, stems


# ---------------------------------------------------------------------------
# Interactive clicker (matplotlib) with live SAM2 mask preview
# ---------------------------------------------------------------------------
def click_keyframe(image_rgb: np.ndarray, segment_cb):
    """Left-click = case (FG), right-click = background. Live mask preview.
    Close the window when satisfied. segment_cb(points, labels) -> bool mask."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    pts, labs = [], []
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(image_rgb)
    ax.set_title("L-click = case   R-click = background   |   close window when done")
    overlay = ax.imshow(np.zeros(image_rgb.shape[:2]), alpha=0.5,
                        cmap="autumn", vmin=0, vmax=1)

    def onclick(e):
        if e.inaxes != ax or e.xdata is None:
            return
        lab = 1 if e.button == 1 else 0
        pts.append([float(e.xdata), float(e.ydata)])
        labs.append(lab)
        ax.plot(e.xdata, e.ydata, "g+" if lab else "rx", markersize=12, markeredgewidth=2)
        mask = segment_cb(np.array(pts, np.float32), np.array(labs, np.int32))
        overlay.set_data(mask.astype(float))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=True)
    return np.array(pts, np.float32), np.array(labs, np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["case", "bin"], default="case",
                    help="case = OBB on bin-ROI crop; bin = axis-aligned box on full frame")
    ap.add_argument("--keyframes", default="0",
                    help="comma-separated frame indices to click (e.g. 0,40,90)")
    ap.add_argument("--roi", choices=["median", "first"], default="median",
                    help="case crop ROI source if cfg.BIN_ROI unset (ignored for --target bin)")
    ap.add_argument("--data", default=None,
                    help="capture run dir (default: newest data/<timestamp>/)")
    ap.add_argument("--no-stage", action="store_true",
                    help="reuse existing staged crops (skip re-staging)")
    args = ap.parse_args()
    target = args.target

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    data = resolve_data_dir(args.data, target)
    frames = sorted(data.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"No frames in {data}. Capture with: python capture.py --target {target}")
    print(f"using {len(frames)} frames from {data}  [target={target}]")

    # bin detector runs on the FULL frame (it finds the ROI) -> no crop;
    # case detector trains on the bin-ROI crop for resolution.
    if target == "bin":
        h0, w0 = np.load(frames[0])["rgb"].shape[:2]
        roi = (0, 0, w0, h0)
    else:
        roi = resolve_roi(frames, args.roi)
        rsrc = "cfg.BIN_ROI" if cfg.BIN_ROI is not None else args.roi
        print(f"crop ROI (x,y,w,h) = {roi}  [{rsrc}]")
        if cfg.BIN_ROI is None:
            print(f"  -> to lock for runtime, set cfg.BIN_ROI = {tuple(int(v) for v in roi)}")

    staged = HERE / f"{cfg.STAGED_DIR}_{target}"
    if args.no_stage:
        stems = [fp.stem for fp in frames]
    else:
        staged, stems = stage(frames, roi, staged)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg.SAM2_MODEL_CFG, cfg.SAM2_CHECKPOINT, device=device)

    labeled = HERE / f"{cfg.LABELED_DIR}_{target}"
    (labeled / "images").mkdir(parents=True, exist_ok=True)
    (labeled / "labels").mkdir(parents=True, exist_ok=True)
    out = HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if target == "case":
        np.save(labeled / "roi.npy", np.array(roi))  # crop used, for runtime reference

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                else torch.autocast("cpu", enabled=False))
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=str(staged))

        # Click the requested keyframes (points are registered into `state`).
        for k in [int(s) for s in args.keyframes.split(",")]:
            img = cv2.cvtColor(cv2.imread(str(staged / f"{k}.jpg")), cv2.COLOR_BGR2RGB)

            def segment_cb(points, labels, _k=k):
                _, _ids, logits = predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=_k, obj_id=OBJ_ID,
                    points=points, labels=labels)
                return (logits[0] > 0.0).cpu().numpy().squeeze()

            print(f"[keyframe {k}] click the {target} (green +), background (red x), close window")
            pts, labs = click_keyframe(img, segment_cb)
            if len(pts) == 0:
                print(f"  no clicks on frame {k} — skipped")

        # Propagate across all frames -> labels + overlays.
        n_ok = 0
        for frame_idx, _ids, logits in predictor.propagate_in_video(state):
            mask = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
            stem = stems[frame_idx]
            img_bgr = cv2.imread(str(staged / f"{frame_idx}.jpg"))
            cv2.imwrite(str(labeled / "images" / f"{stem}.png"), img_bgr)

            line = mask_to_bbox_line(mask) if target == "bin" else mask_to_yolo_line(mask)
            ov = img_bgr.copy()
            if line is not None:
                (labeled / "labels" / f"{stem}.txt").write_text(line + "\n")
                pts = mask_to_obb_points(mask).astype(np.int32)
                if target == "bin":
                    x, y, bw, bh = cv2.boundingRect(pts)
                    cv2.rectangle(ov, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                else:
                    cv2.drawContours(ov, [pts], 0, (0, 255, 0), 2)
                n_ok += 1
            else:
                (labeled / "labels" / f"{stem}.txt").write_text("")  # lost track
                cv2.putText(ov, "NO MASK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(out / f"overlay_{stem}.png"), ov)

    print(f"\nLabeled {n_ok}/{len(stems)} frames -> {labeled}")
    print(f"Next: python review.py --images {labeled.name}/images --labels {labeled.name}/labels")


if __name__ == "__main__":
    main()
