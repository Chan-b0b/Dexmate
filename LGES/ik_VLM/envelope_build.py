"""Build the nominal force envelope (envelope.json) offline. No robot needed.

Two sources, combinable:

  --from-signals GLOB      ik_VLM per-run signal logs (signals_*.jsonl) —
                           preferred: exact live phases + live rate.
  --from-recordings GLOB   recorder episodes (states.jsonl, e.g.
                           LGES/recordings/2026*/case_pick/*) — bootstrap
                           before any supervised run exists. Wrench is RAW
                           there, so each episode is tared on its first
                           TARE_N samples; phases are segmented from the EE
                           z velocity (descend / transport).

ONLY feed it nominal (successful) data — the envelope IS the definition of
"normal". Episodes with meta.json success=false are skipped automatically.

    python -m LGES.ik_VLM.envelope_build \
        --from-recordings 'LGES/recordings/2026*/case_pick/*' [--out PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg
from .monitor import FEATURES, EnvelopeModel
from .signals import wrench_features

TARE_N = 20
DESCEND_VZ = -0.01      # m/s, EE z velocity below this = "descend"


def _episode_ok(states_path: Path) -> bool:
    meta = states_path.parent / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            if m.get("success") is False:
                return False
        except (OSError, json.JSONDecodeError):
            pass
    return True


def _samples_from_recording(states_path: Path):
    """Yield (phase, {feature: value}) per sample of one recorder episode."""
    rows = []
    with states_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                w = d["wrench"]
                rows.append((float(d["t"]),
                             np.array([w["fx"], w["fy"], w["fz"],
                                       w["tx"], w["ty"], w["tz"]], dtype=float),
                             float(d["ee"]["pos"][2])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if len(rows) < TARE_N + 5:
        return
    tare = np.median(np.stack([w for _, w, _ in rows[:TARE_N]]), axis=0)
    ts = np.array([t for t, _, _ in rows])
    zs = np.array([z for _, _, z in rows])
    # EE z velocity over a ~0.3 s window (recorder runs 10-15 Hz)
    vz = np.gradient(zs, ts)
    prev = None
    for i in range(1, len(rows)):
        t, w, _ = rows[i]
        dt = max(t - rows[i - 1][0], 1e-3)
        tared = w - tare
        f_ax, f_lat, t_mag, df = wrench_features(tared, prev, dt)
        prev = tared
        phase = "descend" if vz[i] < DESCEND_VZ else "transport"
        yield phase, {"f_ax": f_ax, "f_lat": f_lat, "t_mag": t_mag, "df_mag": df}


def _samples_from_signals(log_path: Path):
    with log_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                # tolerate logs from before a feature existed (e.g. q_err_max)
                yield d["phase"], {k: float(d[k]) for k in FEATURES if k in d}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue


def build(signal_globs: list[str], recording_globs: list[str]) -> dict:
    acc: dict[str, dict[str, list[float]]] = {}

    def _add(phase: str, feats: dict) -> None:
        # a feature absent from the source (recordings have no q_err_max) gets
        # no stats -> no bound -> the monitor skips it in that envelope
        ph = acc.setdefault(phase, {})
        for k, v in feats.items():
            ph.setdefault(k, []).append(v)

    n_files = 0
    for g in signal_globs:
        for p in sorted(glob.glob(g)):
            n_files += 1
            for phase, feats in _samples_from_signals(Path(p)):
                _add(phase, feats)
    for g in recording_globs:
        for p in sorted(glob.glob(str(Path(g) / "states.jsonl"))
                        or glob.glob(g)):
            p = Path(p)
            if p.name != "states.jsonl":
                p = p / "states.jsonl"
            if not p.exists() or not _episode_ok(p):
                continue
            n_files += 1
            for phase, feats in _samples_from_recording(p):
                _add(phase, feats)

    stats: dict[str, dict[str, dict[str, float]]] = {}
    for phase, feats in acc.items():
        stats[phase] = {}
        for name, vals in feats.items():
            a = np.asarray(vals)
            stats[phase][name] = {"mean": float(a.mean()),
                                  "sigma": float(a.std()),
                                  "qmax": float(a.max()),
                                  "n": int(a.size)}
    logger.info("envelope from {} file(s): {}", n_files,
                {ph: stats[ph]["f_ax"]["n"] for ph in stats})
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-signals", action="append", default=[],
                    help="glob of signals_*.jsonl files")
    ap.add_argument("--from-recordings", action="append", default=[],
                    help="glob of recorder episode dirs (or states.jsonl files)")
    ap.add_argument("--out", default=cfg.ENVELOPE_PATH)
    args = ap.parse_args()
    if not args.from_signals and not args.from_recordings:
        ap.error("give at least one of --from-signals / --from-recordings")
    stats = build(args.from_signals, args.from_recordings)
    if not stats:
        logger.error("no samples found — envelope NOT written")
        return
    model = EnvelopeModel(stats)
    model.save(args.out)
    for phase, feats in stats.items():
        b = model.bounds[phase]
        logger.info("  {}: " + ", ".join(
            f"{k} mean={feats[k]['mean']:.2f} sd={feats[k]['sigma']:.2f} -> bound {b[k]:.2f}"
            for k in FEATURES if k in feats), phase)


if __name__ == "__main__":
    main()
