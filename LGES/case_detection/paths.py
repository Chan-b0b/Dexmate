"""Resolve which capture run to read.

capture.py writes each run to data/<target>/<timestamp>/. This picks the run a
tool should use: an explicit --data path, else the newest run for the target,
else (legacy) a top-level data/<timestamp>/ or flat data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

HERE = Path(__file__).resolve().parent


def _newest_run(d: Path) -> Path | None:
    if not d.exists():
        return None
    subs = sorted(s for s in d.glob("*/") if any(s.glob("frame_*.npz")))
    return subs[-1] if subs else None


def resolve_data_dir(arg: str | None = None, target: str = "case") -> Path:
    """Directory holding frame_*.npz for `target` ('case'|'bin')."""
    base = HERE / cfg.DATA_DIR
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else HERE / arg
    run = _newest_run(base / target)          # data/<target>/<timestamp>/
    if run:
        return run
    # legacy top-level data/<timestamp>/ predates the target split and was case
    # data, so only fall back to it for 'case' (never hand it to 'bin').
    if target == "case":
        legacy = sorted(s for s in base.glob("*/")
                        if s.name not in ("case", "bin") and any(s.glob("frame_*.npz")))
        if legacy:
            return legacy[-1]
    return base / target                      # nothing yet -> caller reports it
