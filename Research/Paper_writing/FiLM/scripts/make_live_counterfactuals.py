#!/usr/bin/env python3
"""Fig: on-robot counterfactual dose sweep (paper §VI, fig:live).

Three stacked panels sharing the dose axis (transplanted normal force, N):
  (a) action response delta-dz vs real, conditioned vs naive — the crossover
  (b) conditioning input c-hat the conditioned policy received — force channels
      track the delivered dose, seal stays 0
  (c) FiLM modulation magnitude (mean |gamma|, |beta|; x*(1+gamma)+beta,
      zero-init identity) — the pathway responding, not a bypass

Data: LGES/vla_training/live_film_probes/smolvla_film_0729_prefix_mask1_recal_fromnaive/
  run3 (fair dose, 10 poses)  st_fz+3N / st_fz+6N
  run4 (high dose,  3 poses)  st_fz+6N / st_fz+9N / st_fz+12N
st_* = raw-state transplants, drift-shifted so both policies receive the same
physical dose (README there; paper quotes run4 means +1.19/+4.59/+8.56 film vs
+1.34/+3.18/+4.18 naive — reproduced here from the JSONs).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = ("/home/maverick/Humanoid/Dexmate/LGES/vla_training/live_film_probes/"
        "smolvla_film_0729_prefix_mask1_recal_fromnaive")
OUT = "/home/maverick/Humanoid/Dexmate/Research/Paper_writing/FiLM/paper/figs"

C_NAIVE = "#D55E00"   # vermillion — validated pair (dataviz six checks, light surface)
C_COND = "#0072B2"    # blue
INK = "#333333"
MUTED = "#767676"
GRID = "#DDDDDD"

RUNS = [  # (json, marker, [(dose_N, scenario_key), ...])
    (f"{BASE}/run3_0806_fair-dose_10poses_VALID/20260806-152448_live_film_authority.json",
     "o", [(3, "st_fz+3N"), (6, "st_fz+6N")]),
    (f"{BASE}/run4_0806_high-dose_fz6-9-12N_3poses_VALID/20260806-152957_live_film_authority.json",
     "s", [(6, "st_fz+6N"), (9, "st_fz+9N"), (12, "st_fz+12N")]),
]


def load(path, keys):
    d = json.load(open(path))
    doses, film, naive, chat, gam, bet, descent = [], [], [], [], [], [], []
    for dose, key in keys:
        f_dz, n_dz, ch, g, b = [], [], [], [], []
        for p in d["poses"]:
            s, base = p["scenarios"], p["baseline_scenarios"]
            f_dz.append((s[key]["dpos_m"][2] - s["real"]["dpos_m"][2]) * 1000)
            n_dz.append((base[key]["dpos_m"][2] - base["real"]["dpos_m"][2]) * 1000)
            ch.append(s[key]["film"]["c_hat"])
            g.append(s[key]["film"]["gamma"]["abs_mean"])
            b.append(s[key]["film"]["beta"]["abs_mean"])
        doses.append(dose)
        film.append(np.array(f_dz))
        naive.append(np.array(n_dz))
        chat.append(np.array(ch).mean(0))     # [contact, fz, fmag, seal]
        gam.append(np.mean(g))
        bet.append(np.mean(b))
    descent = np.mean([-p["scenarios"]["real"]["dpos_m"][2] * 1000 for p in d["poses"]])
    return doses, film, naive, np.array(chat), gam, bet, descent


runs = [(m, *load(p, k)) for p, m, k in RUNS]

fig, (ax_a, ax_b, ax_c) = plt.subplots(
    3, 1, figsize=(3.5, 4.6), dpi=300, sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1.0, 0.9], "hspace": 0.13})

rng = np.random.default_rng(0)


def style(ax):
    ax.tick_params(labelsize=6.5, length=2.5, color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.6, zorder=0)


# --- (a) response ------------------------------------------------------------
for marker, doses, film, naive, _, _, _, _ in runs:
    for vals, color in ((film, C_COND), (naive, C_NAIVE)):
        means = [v.mean() for v in vals]
        ax_a.plot(doses, means, color=color, lw=1.3, marker=marker, ms=3.4,
                  solid_capstyle="round", zorder=4,
                  markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5)
        for x, v in zip(doses, vals):
            ax_a.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=3.5,
                         color=color, alpha=0.30, linewidths=0, zorder=3)

descent_ref = runs[1][7]   # run4 mean own descent per frame
ax_a.axhline(descent_ref, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
ax_a.text(2.7, descent_ref + 0.35, "own mean descent (net retreat above)",
          fontsize=5.6, color=MUTED)
ax_a.annotate("crossover", xy=(7.9, 1.1), fontsize=6.0, color=INK,
              ha="center", style="italic")
ax_a.set_ylabel("response $\\Delta d_z$ (mm/frame)", fontsize=7)
ax_a.set_ylim(-0.8, 11.2)
ax_a.plot([], [], color=C_COND, lw=1.2, label="conditioned")
ax_a.plot([], [], color=C_NAIVE, lw=1.2, label="naive")
ax_a.legend(fontsize=6.3, frameon=False, loc="upper left", handlelength=1.4,
            borderaxespad=0.2, bbox_to_anchor=(0.09, 1.0))
r4_film, r4_naive = runs[1][2], runs[1][3]
ax_a.text(12.25, r4_film[-1].mean(), f"+{r4_film[-1].mean():.1f}", fontsize=6.0,
          color=C_COND, va="center")
ax_a.text(12.25, r4_naive[-1].mean(), f"+{r4_naive[-1].mean():.1f}", fontsize=6.0,
          color=C_NAIVE, va="center")
style(ax_a)

# --- (b) conditioning input --------------------------------------------------
for marker, doses, _, _, chat, _, _, _ in runs:
    ax_b.plot(doses, chat[:, 1], color=C_COND, lw=1.2, marker=marker, ms=3.2,
              zorder=4, markerfacecolor=C_COND, markeredgecolor="white",
              markeredgewidth=0.5)
    ax_b.plot(doses, chat[:, 2], color=C_COND, lw=1.0, ls=(0, (4, 2)),
              marker=marker, ms=3.0, zorder=3, markerfacecolor="white",
              markeredgecolor=C_COND, markeredgewidth=0.7)
    ax_b.plot(doses, chat[:, 0], color=MUTED, lw=0.9, marker=marker, ms=2.4,
              zorder=2, markerfacecolor=MUTED, markeredgecolor="none")
    ax_b.plot(doses, chat[:, 3], color=MUTED, lw=0.9, ls=(0, (4, 2)),
              marker=marker, ms=2.4, zorder=2, markerfacecolor="white",
              markeredgecolor=MUTED, markeredgewidth=0.6)
ax_b.text(12.25, 12.6, "$\\hat{c}_{f_z}$", fontsize=6.5, color=C_COND, va="center")
ax_b.text(12.25, 7.3, "$\\hat{c}_{\\|F\\|}$", fontsize=6.5, color=C_COND, va="center")
ax_b.text(12.25, 1.0, "contact", fontsize=5.8, color=MUTED, va="center")
ax_b.text(12.25, 0.0, "seal = 0", fontsize=5.8, color=MUTED, va="center")
ax_b.set_ylabel("conditioning input $\\hat{c}$", fontsize=7)
ax_b.set_ylim(-1.6, 14.2)
style(ax_b)

# --- (c) FiLM modulation ------------------------------------------------------
for marker, doses, _, _, _, gam, bet, _ in runs:
    ax_c.plot(doses, gam, color=C_COND, lw=1.2, marker=marker, ms=3.2, zorder=4,
              markerfacecolor=C_COND, markeredgecolor="white", markeredgewidth=0.5)
    ax_c.plot(doses, bet, color=C_COND, lw=1.0, ls=(0, (4, 2)), marker=marker,
              ms=3.0, zorder=3, markerfacecolor="white", markeredgecolor=C_COND,
              markeredgewidth=0.7)
ax_c.text(12.25, runs[1][6][-1], "$|\\beta|$", fontsize=6.5, color=C_COND, va="center")
ax_c.text(12.25, runs[1][5][-1], "$|\\gamma|$", fontsize=6.5, color=C_COND, va="center")
ax_c.set_ylabel("FiLM gain (mean $|\\cdot|$)", fontsize=7)
ax_c.set_xlabel("transplanted normal-force dose (N)", fontsize=7)
ax_c.set_ylim(0, 0.30)
ax_c.set_xticks([3, 6, 9, 12])
ax_c.set_xlim(2.3, 13.6)
style(ax_c)

for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
    ax.text(0.01, 1.0, tag, transform=ax.transAxes, fontsize=7, color=INK,
            va="top", fontweight="bold")

fig.subplots_adjust(left=0.13, right=0.9, top=0.985, bottom=0.075)
fig.savefig(f"{OUT}/live_counterfactuals.pdf")
fig.savefig(f"{OUT}/live_counterfactuals.png")
print("saved", OUT)
