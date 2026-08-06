"""Offline monitor validation against recorded data. No robot needed.

Replays episodes through the Monitor and reports trips per episode. The
acceptance gate before arming the monitor on the robot:

  - SUCCESS episodes must produce 0 trips (false-positive rate)
  - known-failure episodes (if any) should trip

    python -m LGES.ik_VLM.replay_test \
        --recordings 'LGES/recordings/2026*/case_pick/*' [--k 4.0] [--envelope PATH]
    python -m LGES.ik_VLM.replay_test --signals 'LGES/ik_VLM/logs/*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from loguru import logger

from . import config as cfg
from .envelope_build import _episode_ok, _samples_from_recording, _samples_from_signals
from .monitor import EnvelopeModel, Monitor
from .signals import Tick


def _replay(samples, model: EnvelopeModel) -> list:
    mon = Monitor(model)
    trips = []
    t = 0.0
    for phase, feats in samples:
        t += 1.0
        tick = Tick(t=t, phase=phase, **feats)
        if mon.observe(tick):
            trips.append(mon.trip_info())
            mon.reset()   # keep scanning: one episode can trip more than once
    return trips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", action="append", default=[],
                    help="glob of recorder episode dirs")
    ap.add_argument("--signals", action="append", default=[],
                    help="glob of signals_*.jsonl files")
    ap.add_argument("--envelope", default=cfg.ENVELOPE_PATH)
    ap.add_argument("--k", type=float, default=cfg.ENVELOPE_K_SIGMA)
    args = ap.parse_args()

    model = EnvelopeModel.load(args.envelope, k=args.k)
    if model is None:
        logger.error("no envelope at {} — run envelope_build first", args.envelope)
        return

    episodes: list[tuple[str, str, list]] = []   # (name, kind, samples)
    for g in args.recordings:
        for p in sorted(glob.glob(g)):
            sp = Path(p) if Path(p).name == "states.jsonl" else Path(p) / "states.jsonl"
            if sp.exists():
                kind = "success" if _episode_ok(sp) else "failure"
                episodes.append((sp.parent.name, kind, list(_samples_from_recording(sp))))
    for g in args.signals:
        for p in sorted(glob.glob(g)):
            episodes.append((Path(p).name, "success", list(_samples_from_signals(Path(p)))))

    if not episodes:
        logger.error("no episodes matched")
        return

    fp = 0
    for name, kind, samples in episodes:
        trips = _replay(samples, model)
        mark = "OK " if not trips else ("FP!" if kind == "success" else "hit")
        if trips and kind == "success":
            fp += 1
        detail = "; ".join(t.describe() for t in trips[:3])
        logger.info("{} [{}] {}: {} trip(s) over {} samples {}",
                    mark, kind, name, len(trips), len(samples),
                    f"— {detail}" if detail else "")
    n_succ = sum(1 for _, k, _ in episodes if k == "success")
    logger.info("=== {} episodes, false positives on success: {}/{} (k={}) ===",
                len(episodes), fp, n_succ, args.k)
    if fp:
        logger.warning("raise --k / ENVELOPE_MIN_BAND (or clean the envelope "
                       "source) until this is 0 before arming on the robot")


if __name__ == "__main__":
    main()
