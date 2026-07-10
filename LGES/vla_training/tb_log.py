#!/usr/bin/env python3
"""Tee lerobot-train's stdout log into TensorBoard scalars.

lerobot 0.5.1 only logs to wandb; this is a decoupled parser (no fork) that
follows the training log file and writes loss / grad_norm / lr (and any other
`key:value` metrics on the step line) as scalars under a TensorBoard logdir.

  python tb_log.py <train.log> <tb_logdir>

Started in the background by train_smolvla.sh. Exits when --done-marker file
appears (the train script touches it) or on EOF after the writer is idle.
View with:  tensorboard --logdir <tb_logdir>
"""

import re
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# Matches the lerobot tracker line, e.g.:
#   step:200 smpl:6400 ep:2 epch:0.30 loss:0.123 grdn:4.5 lr:1.0e-04 updt_s:..
STEP_RE = re.compile(r"\bstep:(\d+)\b")
KV_RE = re.compile(r"([a-zA-Z_]\w*):([-+0-9.eE]+)")
# These are labels, not scalars worth plotting.
SKIP = {"step", "smpl", "ep"}


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: tb_log.py <train.log> <tb_logdir>")
    log_path = Path(sys.argv[1])
    writer = SummaryWriter(log_dir=sys.argv[2])
    # NB: "<log>.done" appended — with_suffix(".done") would REPLACE ".log",
    # so the marker the train script touches (train.log.done) is never seen
    # and this process (and the trap waiting on it) hangs forever.
    done_marker = Path(str(log_path) + ".done")

    # Wait for the log to appear, then follow it.
    while not log_path.exists():
        if done_marker.exists():
            return
        time.sleep(0.5)

    n = 0
    with log_path.open() as f:
        while True:
            line = f.readline()
            if not line:
                if done_marker.exists():
                    break
                time.sleep(0.5)
                continue
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for key, val in KV_RE.findall(line):
                if key in SKIP:
                    continue
                try:
                    writer.add_scalar(key, float(val), step)
                except ValueError:
                    pass
            n += 1
    writer.flush()
    writer.close()
    print(f"[tb_log] wrote {n} step records -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
