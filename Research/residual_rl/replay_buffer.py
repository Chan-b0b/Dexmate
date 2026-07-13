#!/usr/bin/env python
"""Episode-shard replay buffer: one JSONL file per episode, written by actor.py and
re-scanned by learner.py. Append-only and reused across many training epochs -- NOT a
single-use queue that's drained and discarded.

File format (one line per JSON object):
  {"meta": {...}}                                                    (line 0)
  {"t": 0, "state": [...15], "action_residual": [...6], "action_exec": [...7-8], "reward": r}
  ...
  {"episode_end": true, "terminal": bool, "n": N, "last_next_state": [...15] | null}

`terminal=True` only on a genuine task success (seal+lift) -- everything else that ends
an episode (human intervention via Enter, tick-cap stall, safety abort) is a TRUNCATION:
the MDP didn't reach an absorbing state, so the learner must bootstrap the return with
the critic at `last_next_state` rather than treating it as a hard done. See the
residual_rl design discussion: the human keystroke carries no reward of its own (ICM
already scores every transition automatically) -- it only decides when to stop paying
real-robot time on a rollout that's clearly not recovering.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class EpisodeWriter:
    """Actor-side: one instance per rollout process, one file per episode."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.f = None
        self.ep_id = None
        self.n = 0

    def open_episode(self, meta: dict | None = None) -> str:
        self.close(terminal=False)  # safety net if the caller forgot
        # pid + nanosecond timestamp: a plain per-instance counter can still collide
        # across two DIFFERENT EpisodeWriter instances (or two processes started in the
        # same wall-clock second), each starting its own count at 1 -- one writer would
        # silently overwrite the other's episode file. Nanosecond resolution makes that
        # collision practically impossible even for back-to-back calls.
        self.ep_id = (time.strftime("%Y%m%d-%H%M%S-") + f"{os.getpid():05x}-"
                      f"{time.time_ns() % 1_000_000_000:09d}")
        self.f = (self.root / f"{self.ep_id}.jsonl").open("w")
        self.f.write(json.dumps({"meta": meta or {}}) + "\n")
        self.n = 0
        return self.ep_id

    def step(self, state: np.ndarray, action_residual: np.ndarray, action_exec: np.ndarray,
             reward: float):
        if self.f is None:
            return
        self.f.write(json.dumps({
            "t": self.n,
            "state": [float(x) for x in state],
            "action_residual": [float(x) for x in action_residual],
            "action_exec": [float(x) for x in action_exec],
            "reward": float(reward),
        }) + "\n")
        self.f.flush()
        self.n += 1

    def close(self, terminal: bool, last_next_state: np.ndarray | None = None):
        if self.f is None:
            return
        self.f.write(json.dumps({
            "episode_end": True,
            "terminal": bool(terminal),
            "n": self.n,
            "last_next_state": [float(x) for x in last_next_state] if last_next_state is not None else None,
        }) + "\n")
        self.f.close()
        self.f = None


@dataclass
class Episode:
    states: np.ndarray           # [T, obs_dim]
    action_residual: np.ndarray  # [T, 6]
    action_exec: np.ndarray      # [T, action_dim]
    rewards: np.ndarray          # [T]
    terminal: bool
    last_next_state: np.ndarray | None  # [obs_dim], required iff not terminal
    meta: dict = field(default_factory=dict)


class ReplayBuffer:
    """Learner-side: incrementally tails closed episode files under `root`. A file is
    parsed exactly once, the moment its episode_end line first appears; already-closed
    files are never re-read. Open (still-being-written) files are skipped until closed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.episodes: dict[str, Episode] = {}
        self._seen_files: set[str] = set()
        self._flat_cache = None  # invalidated on any new episode

    def refresh(self) -> int:
        """Scan for newly-closed episode files. Returns how many were added."""
        added = 0
        for path in sorted(self.root.glob("*.jsonl")):
            if path.name in self._seen_files:
                continue
            ep = self._try_parse(path)
            if ep is None:
                continue  # still open; retry next refresh()
            self.episodes[path.stem] = ep
            self._seen_files.add(path.name)
            self._flat_cache = None
            added += 1
        return added

    @staticmethod
    def _try_parse(path: Path) -> Episode | None:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return None
        if not lines or "episode_end" not in lines[-1]:
            return None  # not closed yet
        footer = json.loads(lines[-1])
        meta = json.loads(lines[0]).get("meta", {}) if lines[0].strip() else {}
        steps = [json.loads(l) for l in lines[1:-1]]
        if not steps:
            return None  # closed with zero steps -- nothing to learn from
        return Episode(
            states=np.asarray([s["state"] for s in steps], dtype=np.float32),
            action_residual=np.asarray([s["action_residual"] for s in steps], dtype=np.float32),
            action_exec=np.asarray([s["action_exec"] for s in steps], dtype=np.float32),
            rewards=np.asarray([s["reward"] for s in steps], dtype=np.float32),
            terminal=bool(footer["terminal"]),
            last_next_state=(np.asarray(footer["last_next_state"], dtype=np.float32)
                              if footer.get("last_next_state") is not None else None),
            meta=meta,
        )

    def __len__(self) -> int:
        return sum(len(ep.rewards) for ep in self.episodes.values())

    def sample_transitions(self, batch_size: int, rng: np.random.Generator):
        """Uniform sample of (state, action_residual, ep_id, t) across all closed
        episodes -- the caller (learner.py) looks up the per-episode return at (ep_id, t)
        itself, since that depends on the current critic snapshot."""
        if not self._flat_cache:
            self._flat_cache = [(ep_id, t) for ep_id, ep in self.episodes.items()
                                 for t in range(len(ep.rewards))]
        if not self._flat_cache:
            return []
        idx = rng.integers(0, len(self._flat_cache), size=min(batch_size, len(self._flat_cache)))
        return [self._flat_cache[i] for i in idx]
