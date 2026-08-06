"""Passive hover-force baseline check for the 0729 recal FiLM offsets.

Why: the recal offsets (FILM_F0=5.5, FILM_FMAG_OFF=5.5, FILM_FZ_OFF=3.0) are
anchored to the 0729-collection force distributions. The F/T bias drifts
(day/temperature/remount), and every FiLM channel is an offset from that
baseline — a drifted baseline shifts ALL channels at once (e.g. contact>0 while
still hovering), the same failure family as the 0721_0727 fz_off-misconfig runs.

Train-time hover anchors, derived from run_case_pick_0729_recal.sh's measured
c-hat anchors (hover = [0, -1.43, -0.9, 0] in (contact, fz, fmag, seal) order):
    |F|_hover = FMAG_OFF + (-0.9)*FMAG_TAU  = 5.5 - 0.9   = 4.6 N
    fz_hover  = FZ_OFF   + (-1.43)*FZ_TAU   = 3.0 - 1.0   = 2.0 N
Correction: shift = measured hover median - train anchor, applied to F0/FMAG_OFF
(and FZ_OFF with --include-fz) so live hover reproduces the train-time c-hat.

NO MOTION — this script only reads the sensor. Protocol:
  1. Put the LEFT (suction) arm at the pick-hover pose (same orientation as the
     descent; the gravity component in the sensor frame is pose-dependent).
  2. Tool EMPTY, suction OFF, arm still.
  3. python measure_force_baseline.py          # sample -> print + write env file
     python measure_force_baseline.py --dry    # print only
The wrench is read raw via get_state()["wrench"] — the same path data collection
used (collect_case_pick.py, no taring), so the comparison is apples-to-apples.

robot_eval_0729_recal.sh sources the written env file automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

VLA_DIR = Path(__file__).resolve().parent

# 0729 recal training-time values (single source: run_case_pick_0729_recal.sh)
TRAIN = {"F0": 5.5, "FMAG_OFF": 5.5, "FMAG_TAU": 1.0,
         "FZ_OFF": 3.0, "FZ_TAU": 0.7,
         "FMAG_HOVER": 4.6, "FZ_HOVER": 2.0}
DRIFT_WARN_N = 1.5   # |shift| beyond this smells like payload/suction/contact, not drift
STILL_WARN_N = 0.6   # |F| std beyond this = arm probably not still


def sample_wrench(seconds: float, hz: float, side: str) -> np.ndarray:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    rows = []
    with Robot(configs=get_robot_config()) as bot:
        arm = getattr(bot, f"{side}_arm", None)
        ws = getattr(arm, "wrench_sensor", None)
        if ws is None:
            raise SystemExit(f"{side} arm has no wrench_sensor")
        t_end = time.time() + seconds
        while time.time() < t_end:
            rows.append(np.asarray(ws.get_state()["wrench"], dtype=float)[:6])
            time.sleep(1.0 / hz)
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} samples — sensor not streaming?")
    return np.asarray(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--side", default="left", help="suction arm side (cfg.ARM_SIDE)")
    ap.add_argument("--out", type=Path, default=VLA_DIR / "film_baseline_0729.env")
    ap.add_argument("--include-fz", action="store_true",
                    help="also write the corrected FILM_FZ_OFF (default: print only)")
    ap.add_argument("--dry", action="store_true", help="measure + print, write nothing")
    args = ap.parse_args()

    print(f"[baseline] sampling {args.seconds:.0f}s @ {args.hz:.0f}Hz "
          f"({args.side} arm, raw wrench) — arm at HOVER, tool empty, suction OFF")
    w = sample_wrench(args.seconds, args.hz, args.side)
    fmag = np.linalg.norm(w[:, :3], axis=1)
    fz = w[:, 2]

    def stats(x):
        return (float(np.median(x)), float(np.percentile(x, 10)),
                float(np.percentile(x, 90)), float(np.std(x)))

    fm_med, fm_p10, fm_p90, fm_std = stats(fmag)
    fz_med, fz_p10, fz_p90, fz_std = stats(fz)
    d_fm = fm_med - TRAIN["FMAG_HOVER"]
    d_fz = fz_med - TRAIN["FZ_HOVER"]
    f0_new = round(TRAIN["F0"] + d_fm, 2)
    fmag_off_new = round(TRAIN["FMAG_OFF"] + d_fm, 2)
    fz_off_new = round(TRAIN["FZ_OFF"] + d_fz, 2)

    print(f"\n  n={len(w)}   med [p10,p90] (std)")
    print(f"  |F| : {fm_med:5.2f} [{fm_p10:.2f},{fm_p90:.2f}] ({fm_std:.2f})   "
          f"train hover {TRAIN['FMAG_HOVER']:.1f}  ->  drift {d_fm:+.2f} N")
    print(f"  fz  : {fz_med:5.2f} [{fz_p10:.2f},{fz_p90:.2f}] ({fz_std:.2f})   "
          f"train hover {TRAIN['FZ_HOVER']:.1f}  ->  drift {d_fz:+.2f} N")

    if fm_std > STILL_WARN_N:
        print(f"  WARNING: |F| std {fm_std:.2f} > {STILL_WARN_N} N — arm moving/vibrating? remeasure.")
    if abs(d_fm) > DRIFT_WARN_N or abs(d_fz) > DRIFT_WARN_N:
        print(f"  WARNING: drift > {DRIFT_WARN_N} N — check payload off, suction OFF, no contact "
              "before trusting these offsets.")

    # hover c-hat sanity: default env vs corrected (contact should be 0, fmag ~ -0.9)
    def chat(f0, fmag_off, fz_off):
        c = float(np.clip((fm_med - f0) / TRAIN["FMAG_TAU"], 0, 1))
        fm = (fm_med - fmag_off) / TRAIN["FMAG_TAU"]
        fzc = (fz_med - fz_off) / TRAIN["FZ_TAU"]
        return f"contact={c:.2f} fmag={fm:+.2f} fz={fzc:+.2f}"

    print(f"\n  hover c-hat @ default offsets  : {chat(TRAIN['F0'], TRAIN['FMAG_OFF'], TRAIN['FZ_OFF'])}")
    print(f"  hover c-hat @ corrected        : {chat(f0_new, fmag_off_new, fz_off_new)}"
          f"   (train anchor: contact=0.00 fmag=-0.90 fz=-1.43)")
    print(f"\n  corrected env: FILM_F0={f0_new} FILM_FMAG_OFF={fmag_off_new}"
          + (f" FILM_FZ_OFF={fz_off_new}" if args.include_fz
             else f"   (FILM_FZ_OFF={fz_off_new} suggested, not written — use --include-fz)"))

    if args.dry:
        print("[baseline] --dry: nothing written")
        return
    lines = [f"# measured {datetime.now().isoformat(timespec='seconds')} — n={len(w)}, "
             f"|F| med {fm_med:.2f} (drift {d_fm:+.2f}), fz med {fz_med:.2f} (drift {d_fz:+.2f})",
             f"FILM_F0={f0_new}", f"FILM_FMAG_OFF={fmag_off_new}"]
    if args.include_fz:
        lines.append(f"FILM_FZ_OFF={fz_off_new}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[baseline] wrote {args.out}")


if __name__ == "__main__":
    main()
