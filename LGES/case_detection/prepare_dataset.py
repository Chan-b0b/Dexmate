"""Turn labeled bin-ROI crops into a YOLO-OBB dataset.

Input (after you label the crops from `capture.py --crop-bin`):
    <src>/images/*.png        the bin-ROI crops
    <src>/labels/*.txt        YOLO-OBB labels, one row per box:
                              <class> x1 y1 x2 y2 x3 y3 x4 y4   (normalized 0..1)
Most labelers (Roboflow, CVAT, X-AnyLabeling) export exactly this.

Output: a train/val split + data.yaml ready for train.py.

    python prepare_dataset.py --src labeled --val 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with images/ and labels/ (e.g. labeled_case)")
    ap.add_argument("--name", default="case",
                    help="class name(s), comma-separated in class-index order "
                         "(e.g. 'case', or 'bin_empty,bin_full' for a 2-class bin model — "
                         "labels must already carry the matching class index per row)")
    ap.add_argument("--out", default=None, help="output dataset dir (default dataset_<name>)")
    ap.add_argument("--val", type=float, default=0.2, help="val fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = [n.strip() for n in args.name.split(",")]
    if args.out is None:
        args.out = f"dataset_{names[0]}"

    src = (HERE / args.src).resolve()
    imgs = sorted((src / "images").glob("*.png"))
    if not imgs:
        raise SystemExit(f"No images in {src/'images'}")

    paired = [(im, src / "labels" / (im.stem + ".txt")) for im in imgs]
    missing = [im.name for im, lb in paired if not lb.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no label (e.g. {missing[:3]}). "
                         f"Label every crop, or remove the unlabeled ones.")

    random.Random(args.seed).shuffle(paired)
    n_val = max(1, int(len(paired) * args.val))
    splits = {"val": paired[:n_val], "train": paired[n_val:]}

    out = HERE / args.out
    for split, items in splits.items():
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        for im, lb in items:
            shutil.copy(im, out / "images" / split / im.name)
            shutil.copy(lb, out / "labels" / split / lb.name)

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: [{', '.join(names)}]\n"
    )
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  classes={names}")
    target = next((t for t in ("case", "bin", "box", "show", "cylinder") if any(t in n for n in names)), "case")
    print(f"wrote {yaml}\nNext: python train.py --target {target}")


if __name__ == "__main__":
    main()
