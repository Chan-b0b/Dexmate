"""Plot how a rollout deviates from demonstrated behavior over time.

    python Research/gradual_drift/analyze.py --phase case_pick \
        --rollout Research/intervention/interventions/intervention_case_pick/<take>

With no --rollout, a held-out demo of the phase is profiled instead (leave-one-
out) so you can see what *in-distribution* deviation looks like — the control
case the rollout curve is read against.

Reads the curve to tell the two failure modes apart:
  - gradual rise in POSE deviation from early on  -> mode (i), covariate-shift
    drift of the conditioning state.
  - flat pose deviation with a sudden spike at a FORCE event / chunk boundary
    -> mode (ii), action infidelity at contact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import deviation as dv

PLOT_GROUPS = ["pose", "force"]


def summarize(name, profiles, baseline) -> dict:
    """Quantify drift: peak deviation and how much of the rollout sits above
    the demos' 95th-percentile in-distribution band."""
    out = {"rollout": name, "groups": {}}
    for g, prof in profiles.items():
        band95 = float(np.percentile(baseline[g], 95))
        above = prof > band95
        first = int(np.argmax(above)) if above.any() else -1
        out["groups"][g] = {
            "peak": float(prof.max()),
            "mean": float(prof.mean()),
            "demo_median": float(np.median(baseline[g])),
            "demo_p95": band95,
            "frac_above_p95": float(above.mean()),
            "first_frame_above_p95": first,
        }
    return out


def plot(name, phase, profiles, baseline, fmag, suction, out_png):
    n = len(PLOT_GROUPS)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.6 * (n + 1)), sharex=True)
    frames = np.arange(len(fmag))

    for ax, g in zip(axes[:n], PLOT_GROUPS):
        med = np.median(baseline[g])
        p95 = np.percentile(baseline[g], 95)
        ax.axhspan(0, p95, color="tab:green", alpha=0.08)
        ax.axhline(med, color="tab:green", ls="--", lw=1, label="demo median")
        ax.axhline(p95, color="tab:orange", ls="--", lw=1, label="demo p95 (in-dist band)")
        ax.plot(frames, profiles[g], color="tab:blue", lw=1.6, label=f"{g} deviation")
        ax.set_ylabel(f"{g}\nk-NN dist (z)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)

    # contact / suction strip
    ax = axes[-1]
    ax.plot(frames, fmag, color="tab:gray", lw=1.2, label="|force| (N)")
    ax.set_ylabel("contact")
    on = suction > 0.5
    if on.any():
        ax.fill_between(frames, 0, fmag.max() if fmag.max() > 0 else 1,
                        where=on, color="tab:purple", alpha=0.12, label="suction on")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xlabel("rollout frame")

    axes[0].set_title(f"deviation from demonstrated behavior — {name}  (phase: {phase})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos-root", default="LGES/recordings")
    ap.add_argument("--phase", default="case_pick")
    ap.add_argument("--rollout", default=None,
                    help="take dir of the closed-loop rollout; default = held-out demo")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    phase_dir = Path(args.demos_root) / args.phase
    groups = list(dict.fromkeys(PLOT_GROUPS + ["full"]))

    print(f"[baseline] leave-one-out over demos in {phase_dir} ...")
    baseline = dv.loo_baseline(phase_dir, groups, k=args.k)

    if args.rollout:
        take = dv.load_take(args.rollout)
        # an external rollout (e.g. intervention take) is compared against ALL
        # demos of the phase; a held-out demo excludes itself.
        exclude = take["name"] if Path(args.rollout).parent == phase_dir else None
    else:
        take = dv.load_take(dv.list_takes(phase_dir)[-1])
        exclude = take["name"]
        print(f"[rollout] none given -> held-out demo {take['name']}")

    ref = dv.build_reference(phase_dir, exclude_name=exclude)
    profiles = dv.deviation_profile(take["states"], ref, dv.Normalizer(ref), groups, k=args.k)

    fmag = dv.force_magnitude(take["states"])
    suction = take["states"][:, 7]
    summary = summarize(take["name"], profiles, baseline)

    out_png = Path(args.out) if args.out else Path("Research/gradual_drift/out") / f"{take['name']}.png"
    plot(take["name"], args.phase, profiles, baseline, fmag, suction, out_png)
    out_json = out_png.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"[saved] {out_png}\n[saved] {out_json}")
    for g in PLOT_GROUPS:
        s = summary["groups"][g]
        print(f"  {g:5s}  peak={s['peak']:.2f}  demo_p95={s['demo_p95']:.2f}  "
              f"frac_above_p95={s['frac_above_p95']:.0%}  first_cross@{s['first_frame_above_p95']}")


if __name__ == "__main__":
    main()
