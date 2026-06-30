#!/usr/bin/env python3
"""Audit the SUCTION + FORCE signals in the training recordings (the source of truth).

Reads raw states.jsonl per take (fast, no model / no lerobot). Per take it summarizes:
  - suction_cmd  : toggle frame(s) — a pick should have one false->true
  - vacuum_sealed: toggle frame(s) + LAG behind the suction command
  - |F|          : baseline (pre-grasp), the press minimum, value at suction-on / seal-on,
                   and the lift maximum
  - c^           : the FiLM contact scalar clip((|F|-F0)/tau,0,1) — does it EVER fire
                   during the descent (before grasp)?
  - data rate    : consecutive identical wrench frames (sensor slower than the 15Hz log)

The key question for descend-until-contact: at the PRESS, does |F| go UP (so c^ fires) or
DOWN (unloading -> c^ stays 0 -> the contact feature can't gate the stop)?

  /home/dexmate/vla_venv/bin/python inspect_suction_force.py [--task case_pick] [--timeline TAKE]
"""
import argparse
import json
from pathlib import Path

import numpy as np

VLA_DIR = Path(__file__).resolve().parent
RECORDINGS = VLA_DIR.parent / "recordings"
TASKS = ["case_pick", "case_place", "battery_1_pick", "battery_1_place",
         "battery_2_pick", "battery_2_place"]
F0, TAU = 14.0, 3.0


def load(take_dir):
    meta = json.loads((take_dir / "meta.json").read_text())
    frames = [json.loads(l) for l in (take_dir / "states.jsonl").open()]
    fxyz = np.array([[f["wrench"]["fx"], f["wrench"]["fy"], f["wrench"]["fz"]] for f in frames])
    Fmag = np.linalg.norm(fxyz, axis=1)
    suc = np.array([1 if f["suction_cmd"] else 0 for f in frames])
    seal = np.array([1 if f.get("vacuum_sealed") else 0 for f in frames])
    eez = np.array([f["ee"]["pos"][2] for f in frames])
    return meta, Fmag, suc, seal, eez


def toggles(b):
    return [i for i in range(1, len(b)) if b[i] != b[i - 1]]


def first_on(b):
    t = [i for i in toggles(b) if b[i] == 1]
    return t[0] if t else None


def summarize(take_dir):
    meta, Fmag, suc, seal, eez = load(take_dir)
    n = len(Fmag)
    suc_on, seal_on = first_on(suc), first_on(seal)
    chat = np.clip((Fmag - F0) / TAU, 0, 1)
    # "descent" window = start .. grasp (suction-on); for a place (starts sealed) use whole ep
    grasp = suc_on if suc_on is not None else n
    pre = slice(0, max(grasp, 1))
    baseline = float(np.median(Fmag[:min(20, n)]))     # first ~20 frames = hover
    press_min = float(Fmag[pre].min())
    press_min_i = int(np.argmin(Fmag[pre]))
    n_stale = int(np.sum(Fmag[1:] == Fmag[:-1]))        # held/stale wrench samples
    return dict(
        take=take_dir.name, n=n, success=meta.get("success"),
        suc_on=suc_on, suc_tog=len(toggles(suc)),
        seal_on=seal_on, seal_tog=len(toggles(seal)),
        seal_lag=(seal_on - suc_on) if (seal_on is not None and suc_on is not None) else None,
        baseline=baseline, press_min=press_min, press_min_i=press_min_i,
        F_at_suc=float(Fmag[suc_on]) if suc_on is not None else None,
        F_at_seal=float(Fmag[seal_on]) if seal_on is not None else None,
        F_max=float(Fmag.max()), F_max_i=int(np.argmax(Fmag)),
        chat_descent_max=float(chat[pre].max()),         # does contact c^ EVER fire pre-grasp?
        chat_at_press=float(chat[pre][press_min_i]),
        n_stale=n_stale, stale_pct=100.0 * n_stale / max(n - 1, 1),
    )


def timeline(take_dir):
    meta, Fmag, suc, seal, eez = load(take_dir)
    chat = np.clip((Fmag - F0) / TAU, 0, 1)
    print(f"\ntimeline {take_dir.name}  (instruction: {meta.get('instruction')})")
    print(f"{'i':>4} {'ee_z':>7} {'|F|':>7} {'c^':>5} {'suc':>3} {'seal':>4}")
    for i in range(len(Fmag)):
        mark = ""
        if i and suc[i] != suc[i - 1]:
            mark += " <- suction toggle"
        if i and seal[i] != seal[i - 1]:
            mark += " <- seal toggle"
        if i % 5 == 0 or mark:
            print(f"{i:>4} {eez[i]:>7.3f} {Fmag[i]:>7.2f} {chat[i]:>5.2f} {suc[i]:>3} {seal[i]:>4}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=RECORDINGS)
    ap.add_argument("--task", default=None, help="single task (default: all suction tasks)")
    ap.add_argument("--timeline", default=None, help="print full timeline for TAKE name (path or name)")
    args = ap.parse_args()

    if args.timeline:
        p = Path(args.timeline)
        if not p.is_dir():  # resolve a bare take name under any task
            cands = list(args.recordings.glob(f"*/{args.timeline}"))
            p = cands[0] if cands else None
        if p is None:
            raise SystemExit(f"take not found: {args.timeline}")
        timeline(p)
        return

    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        tdir = args.recordings / task
        if not tdir.is_dir():
            continue
        takes = sorted(p for p in tdir.iterdir() if p.is_dir())
        rows = [summarize(t) for t in takes]
        rows = [r for r in rows if r["success"]]
        if not rows:
            continue
        print(f"\n══ {task}  ({len(rows)} successful takes) ═══════════════════════════════")
        print(f"{'take':<34} {'base':>5} {'pressMin':>8} {'F@suc':>6} {'F@seal':>6} {'Fmax':>5} "
              f"{'sucOn':>5} {'sealOn':>6} {'lag':>4} {'c^desc':>6} {'stale%':>6}")
        for r in rows:
            def s(x, f="{:.1f}"):
                return f.format(x) if x is not None else "  -"
            print(f"{r['take']:<34} {s(r['baseline']):>5} {s(r['press_min']):>8} {s(r['F_at_suc']):>6} "
                  f"{s(r['F_at_seal']):>6} {s(r['F_max']):>5} {s(r['suc_on'],'{:d}'):>5} "
                  f"{s(r['seal_on'],'{:d}'):>6} {s(r['seal_lag'],'{:d}'):>4} "
                  f"{r['chat_descent_max']:>6.2f} {r['stale_pct']:>5.0f}%")
        # aggregates
        base = np.array([r["baseline"] for r in rows])
        pmin = np.array([r["press_min"] for r in rows])
        lags = np.array([r["seal_lag"] for r in rows if r["seal_lag"] is not None])
        cdesc = np.array([r["chat_descent_max"] for r in rows])
        n_drop = int(np.sum(pmin < base - TAU))      # press unloads well below baseline
        n_cfire = int(np.sum(cdesc > 0.5))           # contact c^ fires during descent
        print(f"  baseline |F| = {base.mean():.1f}±{base.std():.1f}N   "
              f"press-min |F| = {pmin.mean():.1f}±{pmin.std():.1f}N   "
              f"(press drops >τ below baseline in {n_drop}/{len(rows)} takes)")
        if lags.size:
            print(f"  seal lag behind suction-cmd: {lags.mean():.0f}±{lags.std():.0f} frames "
                  f"(~{lags.mean()/15:.1f}s @15Hz)")
        print(f"  contact c^ fires (>0.5) DURING descent in {n_cfire}/{len(rows)} takes "
              f"-> if ~0, the FiLM contact input is dead before grasp")


if __name__ == "__main__":
    main()
