"""Contact-sheet review of auto-generated YOLO-OBB labels.

Tiles every labeled crop with its oriented box drawn, so you can scan the whole
dataset at a glance. Likely-bad labels (mask too small/large, box touching the
border = probable propagation drift, or no label at all) are outlined in RED and
listed, so you know exactly which frame indices to re-click in sam2_autolabel.py.

    python review.py                       # defaults to labeled/images + labeled/labels
    python review.py --images labeled/images --labels labeled/labels --cols 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from obb_label import label_flags, yolo_line_to_points

HERE = Path(__file__).resolve().parent
GREEN, RED = (0, 255, 0), (0, 0, 255)


def _draw_cell(img_path: Path, label_path: Path, cell: int):
    """Return (cell_image_BGR, flags_list). flags empty == good."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    flags: list[str] = []

    if not label_path.exists() or not label_path.read_text().strip():
        flags = ["no-label"]
    else:
        pts = yolo_line_to_points(label_path.read_text().splitlines()[0], w, h)
        flags = label_flags(pts, w, h)
        cv2.drawContours(img, [pts], 0, RED if flags else GREEN, 2)

    scale = cell / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((cell, cell, 3), np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img

    tag = f"{img_path.stem}" + (f"  [{','.join(flags)}]" if flags else "")
    cv2.rectangle(canvas, (0, 0), (cell, 18), (0, 0, 0), -1)
    cv2.putText(canvas, tag, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                RED if flags else (255, 255, 255), 1)
    if flags:
        cv2.rectangle(canvas, (0, 0), (cell - 1, cell - 1), RED, 2)
    return canvas, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=f"{cfg.LABELED_DIR}/images")
    ap.add_argument("--labels", default=f"{cfg.LABELED_DIR}/labels")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cell", type=int, default=220)
    args = ap.parse_args()

    img_dir, lab_dir = HERE / args.images, HERE / args.labels
    out = HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".png", ".jpg")])
    if not imgs:
        print(f"No images in {img_dir}")
        return 1

    per_page = args.cols * args.rows
    flagged: list[str] = []
    page = 0
    for start in range(0, len(imgs), per_page):
        cells = []
        for ip in imgs[start:start + per_page]:
            cell, flags = _draw_cell(ip, lab_dir / (ip.stem + ".txt"), args.cell)
            cells.append(cell)
            if flags:
                flagged.append(f"{ip.stem} [{','.join(flags)}]")
        while len(cells) < per_page:
            cells.append(np.zeros((args.cell, args.cell, 3), np.uint8))
        grid = np.vstack([np.hstack(cells[r * args.cols:(r + 1) * args.cols])
                          for r in range(args.rows)])
        dst = out / f"review_page_{page:02d}.png"
        cv2.imwrite(str(dst), grid)
        print(f"wrote {dst}")
        page += 1

    print(f"\n{len(imgs)} labels reviewed, {len(flagged)} flagged:")
    for f in flagged:
        print(" ", f)
    if flagged:
        print("\nRe-click these in sam2_autolabel.py and re-propagate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
