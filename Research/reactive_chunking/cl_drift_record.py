#!/usr/bin/env python3
"""Closed-loop drift recorder (P0c).

Runs the REAL policy deploy (LGES/vla_training/run_policy.py) unchanged and logs
the exact per-tick model inputs + the executed action, so we can later re-run a
fresh chunk on each observation OFF the expert manifold (the policy's own
rollout). This is the honest completion of P0: expert replay hides the
open-loop gap because nothing is surprising; closed-loop drift is where
re-grounding would matter.

It does NOT modify vla_training and does NOT change deploy behaviour: it
monkey-patches run_policy.predict with an observe-only wrapper that returns the
original action unchanged and just records its inputs/output in RAM (flushed to
one .npz on exit, so the live 15 Hz loop pays ~no per-tick I/O).

Usage on the robot (your normal safe --go procedure, flags after `--`):
  /home/dexmate/vla_venv/bin/python Research/reactive_chunking/cl_drift_record.py \
      --rollout-dir Research/reactive_chunking/rollouts/run1 -- \
      --go --goto-start <take_dir> --task case_pick --force-limit 15 --max-ticks 350

Offline plumbing check (no robot): ... --rollout-dir /tmp/r -- --self-test <take_dir>
"""

import atexit
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "LGES" / "vla_training"))
import run_policy  # noqa: E402


def main():
    argv = sys.argv[1:]
    if "--rollout-dir" not in argv:
        sys.exit("usage: cl_drift_record.py --rollout-dir DIR -- <run_policy args>")
    ri = argv.index("--rollout-dir")
    roll = Path(argv[ri + 1])
    rest = argv[ri + 2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    roll.mkdir(parents=True, exist_ok=True)

    buf = []
    _orig = run_policy.predict

    def _logging_predict(policy, pre, post, state, image, instruction, depth_image=None):
        pred = _orig(policy, pre, post, state, image, instruction, depth_image)
        buf.append(dict(state=np.asarray(state), image=np.asarray(image),
                        depth=None if depth_image is None else np.asarray(depth_image),
                        pred=np.asarray(pred), instruction=instruction))
        return pred

    run_policy.predict = _logging_predict

    done = {"flag": False}

    def flush():
        if done["flag"] or not buf:
            return
        done["flag"] = True
        out = dict(
            state=np.stack([b["state"] for b in buf]),
            image=np.stack([b["image"] for b in buf]),
            pred=np.stack([b["pred"] for b in buf]),
            instruction=np.array([b["instruction"] for b in buf]),
        )
        if buf[0]["depth"] is not None:
            out["depth"] = np.stack([b["depth"] for b in buf])
        np.savez_compressed(roll / "rollout.npz", **out)
        print(f"\n[cl_drift] logged {len(buf)} ticks -> {roll / 'rollout.npz'}")

    atexit.register(flush)
    sys.argv = ["run_policy.py"] + rest
    try:
        run_policy.main()
    finally:
        flush()


if __name__ == "__main__":
    main()
