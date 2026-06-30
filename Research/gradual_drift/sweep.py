"""Sweep the deviation diagnostic over all rollout takes, per phase.

For every phase that has both demos (LGES/recordings/<phase>) and rollout takes,
build the in-distribution band once (leave-one-out over demos), then score every
rollout take against the full demo reference. Aggregates per phase and writes a
table + a summary figure, so you can see where (which phase) and how (pose drift
vs. force-coincident) the policy leaves the demonstrated manifold.

    python Research/gradual_drift/sweep.py
    python Research/gradual_drift/sweep.py --rollouts-root Research/gradual_drift/rollouts
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
from analyze import summarize

GROUPS = ["pose", "force", "full"]


def discover(rollouts_root: Path, demos_root: Path) -> list[tuple[str, Path]]:
    """(phase, rollout_phase_dir) pairs that have matching demos.

    Handles both the intervention layout (intervention_<phase>/) and a plain
    <phase>/ layout (e.g. run_policy --log-dir output)."""
    pairs = []
    for d in sorted(p for p in rollouts_root.iterdir() if p.is_dir()):
        phase = d.name[len("intervention_"):] if d.name.startswith("intervention_") else d.name
        if (demos_root / phase).is_dir() and dv.list_takes(d):
            pairs.append((phase, d))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos-root", default="LGES/recordings")
    ap.add_argument("--rollouts-root", default="Research/intervention/interventions")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="Research/gradual_drift/out")
    args = ap.parse_args()

    demos_root, rollouts_root = Path(args.demos_root), Path(args.rollouts_root)
    pairs = discover(rollouts_root, demos_root)
    if not pairs:
        raise SystemExit(f"no phases with both demos and rollouts under {rollouts_root}")

    def shape(prof: np.ndarray, p95: float) -> dict:
        """Deviation level early vs late (rel. to the demo band) -> tells a
        rising drift from a flat offset."""
        w = max(1, len(prof) // 5)
        early, late = float(np.median(prof[:w])), float(np.median(prof[-w:]))
        return {"early_over": early / p95, "late_over": late / p95,
                "rise_over": (late - early) / p95,
                "peak": float(prof.max()), "frac": float((prof > p95).mean())}

    rows = []
    for phase, rdir in pairs:
        phase_dir = demos_root / phase
        print(f"[{phase}] baseline (LOO over demos) ...")
        baseline = dv.loo_baseline(phase_dir, GROUPS, k=args.k)
        p95 = {g: float(np.percentile(baseline[g], 95)) for g in GROUPS}
        ref = dv.build_reference(phase_dir)
        norm = dv.Normalizer(ref)
        for take in dv.list_takes(rdir):
            t = dv.load_take(take)
            prof = dv.deviation_profile(t["states"], ref, norm, GROUPS, k=args.k)
            row = {"phase": phase, "take": t["name"], "n_frames": len(t["states"])}
            for g in GROUPS:
                for kk, vv in shape(prof[g], p95[g]).items():
                    row[f"{g}_{kk}"] = vv
            rows.append(row)
        print(f"    {len(dv.list_takes(rdir))} rollout takes scored")

    # ── per-phase aggregate + shape-aware verdict ──────────────────────
    phases = [p for p, _ in pairs]
    def med(phase, key):
        v = [r[key] for r in rows if r["phase"] == phase and r[key] is not None]
        return float(np.median(v)) if v else float("nan")

    def verdict(early, late, rise, force_frac) -> str:
        starts_off = early > 1.0                  # already past demo p95 at t~0
        rises = late > 1.25 * early and rise > 0.5
        if force_frac > 0.30:
            return "contact (ii)"
        if rises and starts_off:
            return "offset+drift"
        if rises:
            return "drift (i)"
        if starts_off:
            return "offset (init-cond)"
        return "near-demo"

    agg = {}
    print("\n  phase                 n   pose early|late|rise (xband)   force:frac   verdict")
    print("  " + "-" * 84)
    for p in phases:
        n = sum(r["phase"] == p for r in rows)
        e, l, r = med(p, "pose_early_over"), med(p, "pose_late_over"), med(p, "pose_rise_over")
        ff = med(p, "force_frac")
        v = verdict(e, l, r, ff)
        agg[p] = {"n": n, "pose_early_over": e, "pose_late_over": l,
                  "pose_rise_over": r, "force_frac": ff, "verdict": v}
        print(f"  {p:20s} {n:3d}   {e:5.1f} |{l:5.1f} |{r:+5.1f}            "
              f"{ff:6.0%}   {v}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rollout_sweep.json").write_text(json.dumps({"rows": rows, "aggregate": agg}, indent=2))

    # ── figure: pose deviation EARLY vs LATE per phase (rel. to demo band) ──
    # both bars near 1 -> in-distribution; both high & equal -> offset;
    # late >> early -> gradual drift.
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(phases)); w = 0.38
    ax.bar(x - w/2, [agg[p]["pose_early_over"] for p in phases], w, color="tab:cyan",
           alpha=0.8, label="pose dev early (first 20%)")
    ax.bar(x + w/2, [agg[p]["pose_late_over"] for p in phases], w, color="tab:blue",
           alpha=0.8, label="pose dev late (last 20%)")
    for i, p in enumerate(phases):  # per-take scatter
        ev = [r["pose_early_over"] for r in rows if r["phase"] == p]
        lv = [r["pose_late_over"] for r in rows if r["phase"] == p]
        ax.scatter(np.full(len(ev), i - w/2), ev, s=10, color="teal", alpha=0.4, zorder=3)
        ax.scatter(np.full(len(lv), i + w/2), lv, s=10, color="navy", alpha=0.4, zorder=3)
    ax.axhline(1.0, color="tab:orange", ls="--", lw=1, label="demo p95 band (=1)")
    ax.set_xticks(x); ax.set_xticklabels(phases, rotation=20, ha="right")
    ax.set_ylabel("pose deviation  (× demo p95 band)")
    ax.set_title("pose deviation early vs late, by phase  "
                 f"({sum(a['n'] for a in agg.values())} rollout takes)\n"
                 "flat & high = init-cond offset · late≫early = gradual drift")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=130)
    print(f"\n[saved] {out/'sweep.png'}\n[saved] {out/'rollout_sweep.json'}")


if __name__ == "__main__":
    main()
