"""Auto-label capture_bev.py data with SAM2 video propagation.

BEV counterpart of sam2_autolabel.py: the training image is the FULL metric
BEV canvas (capture_bev.py already saved the warp), so there is no bin-ROI
crop stage — frames are staged straight from the run's bev/frame_*.png.
Click the target on one (or a few) keyframe(s); SAM2 propagates the mask
across the sequence; each mask becomes a label on the BEV canvas. Runs
offline on a GPU box (copy data/ over). Not used at robot runtime.

Pipeline:
    data/<target>_bev/<ts>_L<k>/bev/frame_*.png   (from capture_bev.py)
      -> staged_<target>_bev/<idx>.jpg            (fed to SAM2)
      -> click keyframe(s) -> propagate -> per-frame mask
      -> labeled_<target>_bev/labels/<stem>.txt   (OBB or axis-aligned per cfg.TRAIN_OBB)
         + labeled_<target>_bev/images/<stem>.png  + out/overlay_*.png

--rewarp rebuilds the BEV from the raw npz (per-frame plane from
layers_remaining) instead of reusing the saved pngs — use it if the BEV_*
config or the capture-time --layer was wrong. Missing/incomplete bev/ pngs
fall back to re-warping automatically.

    python sam2_autolabel_bev.py                          # newest case_bev run
    python sam2_autolabel_bev.py --target bin --keyframes 0,40

Then:  python review.py   ->  re-run with --keyframes including drifted frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from obb_label import mask_to_bbox_line, mask_to_obb_points, mask_to_yolo_line
from paths import resolve_data_dir
from sam2_autolabel import OBJ_ID, click_keyframe, stage_bev

HERE = Path(__file__).resolve().parent


def stage_pngs(pngs: list[Path], staged: Path) -> list[str]:
    """Copy the already-warped bev/frame_*.png -> staged/<idx>.jpg. Returns stems."""
    staged.mkdir(parents=True, exist_ok=True)
    for p in staged.glob("*.jpg"):
        p.unlink()
    stems = []
    for i, fp in enumerate(pngs):
        cv2.imwrite(str(staged / f"{i}.jpg"), cv2.imread(str(fp)))
        stems.append(fp.stem)
    return stems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["case", "bin"], default="case",
                    help="what to label (box type from cfg.TRAIN_OBB; on the full BEV canvas)")
    ap.add_argument("--keyframes", default="0",
                    help="comma-separated frame indices to click (e.g. 0,40,90)")
    ap.add_argument("--data", default=None,
                    help="capture_bev run dir (default: newest data/<target>_bev/)")
    ap.add_argument("--no-stage", action="store_true",
                    help="reuse existing staged images (skip re-staging)")
    ap.add_argument("--rewarp", action="store_true",
                    help="re-warp the BEV from the raw npz (plane from layers_remaining) "
                         "instead of using the saved bev/*.png — use if BEV config or "
                         "the capture --layer changed")
    args = ap.parse_args()
    target = args.target
    use_obb = cfg.TRAIN_OBB[target]

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    data = resolve_data_dir(args.data, f"{target}_bev")
    frames = sorted(data.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"No frames in {data}. Capture with: "
                         f"python capture_bev.py --target {target}")
    print(f"using {len(frames)} frames from {data}  "
          f"[target={target}, {'OBB' if use_obb else 'axis-aligned'} labels]")

    staged = HERE / f"{cfg.STAGED_DIR}_{target}_bev"
    if args.no_stage:
        stems = [fp.stem for fp in frames]
    else:
        pngs = sorted((data / "bev").glob("frame_*.png"))
        if args.rewarp or len(pngs) != len(frames):
            if not args.rewarp:
                print(f"bev/ pngs incomplete ({len(pngs)} vs {len(frames)} npz) -> re-warping from npz")
            stems = stage_bev(frames, staged)[1]
        else:
            stems = stage_pngs(pngs, staged)
    # Prefix stems with the session dir so multi-session labeling accumulates in
    # one labeled_<target>_bev/ instead of overwriting (every session has frame_000..).
    stems = [f"{data.name}__{s}" for s in stems]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg.SAM2_MODEL_CFG, cfg.SAM2_CHECKPOINT, device=device)

    labeled = HERE / f"{cfg.LABELED_DIR}_{target}_bev"
    (labeled / "images").mkdir(parents=True, exist_ok=True)
    (labeled / "labels").mkdir(parents=True, exist_ok=True)
    out = HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

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

            line = mask_to_yolo_line(mask) if use_obb else mask_to_bbox_line(mask)
            ov = img_bgr.copy()
            if line is not None:
                (labeled / "labels" / f"{stem}.txt").write_text(line + "\n")
                pts = mask_to_obb_points(mask).astype(np.int32)
                if not use_obb:
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
