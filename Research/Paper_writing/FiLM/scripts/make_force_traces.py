#!/usr/bin/env python3
"""Fig: per-trial contact force on the robot (paper §VI, fig:forcetraces).

Left panel: contact-force traces aligned at peak (t=0). Naive traces terminate at
the external abort (marked x); conditioned traces rise, turn, and recover.
Right panel: peak contact force per trial with per-policy median.

Data: LGES/vla_training/rollouts/{smolvla_naive_0729, smolvla_film_0729_prefix_mask1}/
Contact force = |F(fx,fy,fz)| - meta.baseline_force_n  (mount offset removed).
Canonical run set per EVIDENCE.md §3 08-11 재검증 블록 / RESULTS_LINEUP.md B1.
"""
import json, glob, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/maverick/Humanoid/Dexmate/LGES/vla_training/rollouts"
OUT = "/home/maverick/Humanoid/Dexmate/Research/Paper_writing/FiLM/paper/figs"

C_NAIVE = "#D55E00"   # vermillion — validated pair (dataviz six checks, light surface)
C_COND = "#0072B2"    # blue
INK = "#333333"
GRID = "#DDDDDD"
LIMIT = 15.0

def load(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    t, f, z = [], [], []
    for line in open(os.path.join(run_dir, "states.jsonl")):
        d = json.loads(line)
        w = d["wrench"]
        t.append(d["t"])
        f.append(math.sqrt(w["fx"]**2 + w["fy"]**2 + w["fz"]**2) - meta["baseline_force_n"])
        z.append(d["ee"]["pos"][2])
    t = np.array(t) - t[0]
    f = np.array(f)
    z = np.array(z)
    # interaction window: everything before lift begins (metric decision 08-11).
    # lift start = frame after the last time EE z is within 20 mm of its deepest point;
    # beyond it |F| carries payload weight, not contact force. naive never lifts.
    lift = len(f)
    below = np.nonzero(z < z.min() + 0.020)[0]
    if len(below):
        lift = int(below[-1]) + 1
    return t, f[:lift], meta

naive = [load(d) for d in sorted(glob.glob(f"{BASE}/smolvla_naive_0729/2026*"))]
cond = [load(d) for d in sorted(glob.glob(f"{BASE}/smolvla_film_0729_prefix_mask1/L*/2026*"))]

fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)

W_PRE, W_POST = 8.0, 0.0

# example traces only (caption marks them as representative): one naive abort,
# one conditioned L5 pick near the median peak (2.5 N)
naive_dirs = sorted(glob.glob(f"{BASE}/smolvla_naive_0729/2026*"))
ex_naive = load(next(d for d in naive_dirs if "142623" in d))
ex_cond = load(f"{BASE}/smolvla_film_0729_prefix_mask1/L5/20260730-170039_r03_ep0000_case_pick")

def plot_trace(run, color, mark_abort):
    t, f, meta = run
    pk = int(np.argmax(f))
    tt = t[:len(f)] - t[pk]
    m = (tt >= -W_PRE) & (tt <= W_POST)
    ax.plot(tt[m], f[m], color=color, lw=1.3, alpha=0.95,
            solid_capstyle="round", zorder=3)
    if mark_abort:
        ax.plot(tt[m][-1], f[m][-1], marker="x", color=color, ms=5.5,
                mew=1.5, zorder=4)

plot_trace(ex_cond, C_COND, mark_abort=False)
plot_trace(ex_naive, C_NAIVE, mark_abort=True)

ax.axhline(LIMIT, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(-4.0, LIMIT + 0.5, "15 N abort limit", fontsize=6.5, color=INK)

ax.set_xlim(-W_PRE, 0.35)
ax.set_ylim(-1.2, 20)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel("Time from peak force (s)", fontsize=7)
ax.set_ylabel("Contact force (N)", fontsize=7)
ax.tick_params(labelsize=6.5, length=2.5, color=GRID)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.6, zorder=0)

# direct labels + legend (2 series)
ax.plot([], [], color=C_NAIVE, lw=1.2, label="naive (aborted ×)")
ax.plot([], [], color=C_COND, lw=1.2, label="conditioned")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", handlelength=1.4,
          borderaxespad=0.2)

fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.2)
fig.savefig(f"{OUT}/robot_force_traces.pdf")
fig.savefig(f"{OUT}/robot_force_traces.png")
print("saved", OUT)
