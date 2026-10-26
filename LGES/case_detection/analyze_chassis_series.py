"""Offline accuracy analysis of a capture_chassis_series.py run.

For every pose: YOLO case centre (median over the saved frames), a
model-independent centre from template matching (a patch around the pose-0
YOLO centre, found again in each BEV), and the encoder truth. All relative to
pose 0, so a YOLO error that is already present at pose 0 is invisible here.

    ~/vla_venv/bin/python analyze_chassis_series.py data/chassis_series/<run>
    ~/vla_venv/bin/python analyze_chassis_series.py data/chassis_series/<run> data/chassis_series/<run2>

Writes out/chassis_series/<run>/pose_<k>.png + sheet.png: the BEV (2x) cropped
so the TRACKED case centre (green +) sits in the middle, with the YOLO box and
centre in red — the case stays put and any YOLO drift is visible.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

import bev
import config as cfg
import detect_case_bev as dcb

HERE = Path(__file__).resolve().parent
# robot move -> (axis of the case shift in base_link, sign of the case shift)
MOVE = {"f": (0, -1), "b": (0, 1), "l": (1, -1), "r": (1, 1)}


def _peak(r: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Sub-pixel template peak (1-D parabola per axis)."""
    def par(a, b, c):
        d = a - 2 * b + c
        return 0.5 * (a - c) / d if d != 0 else 0.0
    dx = par(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    dy = par(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + dx, y + dy


def analyze(run: Path) -> None:
    paths = sorted(glob.glob(str(run / "pose_*.npz")), key=lambda p: int(Path(p).stem.split("_")[1]))
    model = dcb.load_model()
    out = HERE / "out" / "chassis_series" / run.name
    out.mkdir(parents=True, exist_ok=True)
    tpl = None
    rows, tiles = [], []
    for k, p in enumerate(paths):
        z = np.load(p)
        move = str(z["move"])
        plane = bev.top_face_z(int(z["layers_remaining"]))
        per = []
        for i in range(len(z["rgb"])):
            m = bev.build_mapper(z["q_torso"][i], z["q_head"][i], plane)
            img = cv2.cvtColor(m.warp(z["rgb"][i]), cv2.COLOR_RGB2BGR)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            res = model.predict(img, conf=cfg.OBB_CONF, verbose=False)[0]
            if res.obb is None or len(res.obb) == 0:
                continue
            j = int(np.argmax(res.obb.conf.cpu().numpy()))
            yc = res.obb.xywhr.cpu().numpy()[j][:2]
            if tpl is None:
                hw, hh = 110, 70          # central tray + battery patch, BEV px
                tpl = (g[int(yc[1]) - hh:int(yc[1]) + hh, int(yc[0]) - hw:int(yc[0]) + hw].copy(), hw, hh)
            t, hw, hh = tpl
            r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            u, v = _peak(r, *loc)
            tc = np.array([u + hw, v + hh])
            per.append((*m.bev_px_to_base(*tc), *m.bev_px_to_base(*yc), score))
            if i == 0:
                big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                poly = res.obb.xyxyxyxy.cpu().numpy()[j]
                cv2.polylines(big, [(poly * 2).astype(np.int32)], True, (0, 0, 255), 2)
                cv2.circle(big, tuple((yc * 2).astype(int)), 6, (0, 0, 255), -1)
                c2 = (tc * 2).astype(int)
                cv2.drawMarker(big, tuple(c2), (0, 255, 0), cv2.MARKER_CROSS, 26, 3)
                crop = big[max(0, c2[1] - 340):c2[1] + 340, max(0, c2[0] - 460):c2[0] + 460]
                crop = cv2.copyMakeBorder(crop, 0, 680 - crop.shape[0], 0, 920 - crop.shape[1],
                                          cv2.BORDER_CONSTANT)
                off = np.hypot(*(yc - tc)) / cfg.BEV_PX_PER_M * 1000
                tx = m.bev_px_to_base(*tc)
                for n, (txt, col) in enumerate((
                        (f"pose {k}  wheels {z['wheel_mm'].mean():.0f} mm  case x~{tx[0]*1000:.0f} "
                         f"y~{tx[1]*1000:+.0f} mm", (255, 255, 255)),
                        (f"YOLO centre - tracked centre: {off:.1f} mm", (0, 0, 255)),
                        ("red = YOLO box/centre   green + = tracked case centre", (0, 255, 0)))):
                    for th, cc in ((5, (0, 0, 0)), (2, col)):
                        cv2.putText(crop, txt, (10, 34 + 34 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cc, th)
                cv2.imwrite(str(out / f"pose_{k}.png"), crop)
                tiles.append(crop)
        a = np.median(np.asarray(per), 0)
        rows.append((float(z["wheel_mm"].mean()), *(a[:4] * 1000), a[4]))
    o = np.asarray(rows)
    ax, sign = MOVE[move]
    other = 1 - ax
    tm = (o[:, 1 + ax] - o[0, 1 + ax]) * sign
    yo = (o[:, 3 + ax] - o[0, 3 + ax]) * sign
    name = "xy"[ax]
    print(f"\n{run.name}  (robot '{move}', case shift along {name}, truth = encoder)")
    print(f" pose  enc_mm  tmpl_mm  yolo_mm  tmpl-enc  yolo-enc  yolo-tmpl  score   {'xy'[other]}: tmpl / yolo  "
          f"case {name} (yolo)")
    for k in range(len(o)):
        print(f"  {k}  {o[k,0]:6.1f}  {tm[k]:7.1f}  {yo[k]:7.1f}   {tm[k]-o[k,0]:+6.1f}    {yo[k]-o[k,0]:+6.1f}"
              f"    {yo[k]-tm[k]:+6.1f}   {o[k,5]:.2f}   {o[k,1+other]-o[0,1+other]:+5.1f} / "
              f"{o[k,3+other]-o[0,3+other]:+5.1f}      {o[k,3+ax]:7.1f}")
    print(f"  gain (fit vs encoder): template {np.polyfit(o[:,0], tm, 1)[0]:.3f}   "
          f"yolo {np.polyfit(o[:,0], yo, 1)[0]:.3f}")
    blank = np.zeros_like(tiles[0])
    rws = [tiles[i:i + 3] + [blank] * (3 - len(tiles[i:i + 3])) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out / "sheet.png"), np.vstack([np.hstack(r) for r in rws]))
    print(f"  images: {out}/")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyze(Path(arg).resolve())
