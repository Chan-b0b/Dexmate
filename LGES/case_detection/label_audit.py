"""Audit the OBB labels of a dataset: size consistency per run, outliers, edge
truncation, and contact sheets to eyeball them.

    ~/vla_venv/bin/python label_audit.py                       # dataset_case
    ~/vla_venv/bin/python label_audit.py --dataset dataset_box

Writes to out/label_audit/<dataset>/ :
    labels.csv            every label: run, split, image, long/short px, center, flags
    sample_<run>.png      12 random labels of that run, box + size drawn on the image
    outliers.png          the labels flagged as size outliers / edge-truncated

Flags (per label):
    size   long or short side more than --tol px from its RUN median
    edge   a corner within 3 px of (or outside) the image border, i.e. the case
           is cut off and the box is partly a guess
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as cfg

HERE = Path(__file__).resolve().parent


def sheet(items, cols, tile_w=420):
    """items: list of (img_bgr, caption). Grid of downscaled tiles."""
    tiles = []
    for img, cap in items:
        s = tile_w / img.shape[1]
        t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (t.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(t, cap, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols]
        r += [np.zeros((h, tile_w, 3), np.uint8)] * (cols - len(r))
        rows.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                               for t in r]))
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_case")
    ap.add_argument("--tol", type=float, default=10.0,
                    help="px from the run median before a side counts as a size outlier")
    ap.add_argument("--n-sample", type=int, default=12)
    ap.add_argument("--n-outliers", type=int, default=24)
    args = ap.parse_args()

    root = HERE / args.dataset
    out = HERE / "out" / "label_audit" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    for f in sorted(glob.glob(str(root / "labels" / "*" / "*.txt"))):
        split = Path(f).parent.name
        stem = Path(f).stem
        img_path = root / "images" / split / (stem + ".png")
        if not img_path.exists():
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        for k, ln in enumerate(open(f)):
            v = ln.split()
            if len(v) < 9:
                continue
            p = np.array(v[1:9], float).reshape(4, 2) * [w, h]
            (cx, cy), (a, b), _ = cv2.minAreaRect(p.astype(np.float32))
            edge = bool((p.min(0) <= 3).any() or (p[:, 0] >= w - 3).any() or (p[:, 1] >= h - 3).any())
            recs.append(dict(run=stem.split("__")[0], split=split, image=str(img_path), k=k,
                             long_px=max(a, b), short_px=min(a, b), cx=cx, cy=cy,
                             edge=edge, poly=p))

    by_run = defaultdict(list)
    for r in recs:
        by_run[r["run"]].append(r)
    for run, rs in by_run.items():
        ml = np.median([r["long_px"] for r in rs])
        ms = np.median([r["short_px"] for r in rs])
        for r in rs:
            r["size_out"] = abs(r["long_px"] - ml) > args.tol or abs(r["short_px"] - ms) > args.tol
            r["dl"], r["ds"] = r["long_px"] - ml, r["short_px"] - ms

    with open(out / "labels.csv", "w", newline="") as f:
        cols = ["run", "split", "image", "k", "long_px", "short_px", "cx", "cy", "dl", "ds",
                "size_out", "edge"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    pxm = cfg.BEV_PX_PER_M
    print(f"{len(recs)} labels in {len(by_run)} runs  (BEV {pxm} px/m)")
    print("run                  n  long med/std px (mm)      short med/std px (mm)   size-out edge")
    for run, rs in sorted(by_run.items()):
        L = np.array([r["long_px"] for r in rs])
        S = np.array([r["short_px"] for r in rs])
        print("%-18s %4d  %6.1f / %5.2f (%5.1f mm)   %6.1f / %5.2f (%5.1f mm)   %4d %4d" % (
            run, len(rs), np.median(L), L.std(), np.median(L) / pxm * 1000,
            np.median(S), S.std(), np.median(S) / pxm * 1000,
            sum(r["size_out"] for r in rs), sum(r["edge"] for r in rs)))

    def draw(r):
        img = cv2.imread(r["image"])
        col = (0, 0, 255) if (r["size_out"] or r["edge"]) else (0, 255, 0)
        cv2.polylines(img, [r["poly"].astype(np.int32)], True, col, 2)
        cap = "%s %s L%.0f S%.0f dL%+.0f%s%s" % (r["run"][-11:], Path(r["image"]).stem.split("__")[-1],
                                                r["long_px"], r["short_px"], r["dl"],
                                                " SIZE" if r["size_out"] else "",
                                                " EDGE" if r["edge"] else "")
        return img, cap

    rng = np.random.default_rng(0)
    for run, rs in by_run.items():
        pick = [rs[i] for i in rng.permutation(len(rs))[:args.n_sample]]
        cv2.imwrite(str(out / f"sample_{run}.png"), sheet([draw(r) for r in pick], 4))
    bad = sorted([r for r in recs if r["size_out"] or r["edge"]],
                 key=lambda r: -max(abs(r["dl"]), abs(r["ds"])))[:args.n_outliers]
    if bad:
        cv2.imwrite(str(out / "outliers.png"), sheet([draw(r) for r in bad], 4))
    print(f"\nflagged: size {sum(r['size_out'] for r in recs)}, edge {sum(r['edge'] for r in recs)} "
          f"of {len(recs)}\nwrote {out}/")


if __name__ == "__main__":
    main()
