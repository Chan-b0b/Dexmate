"""Decoupling probe (characterization): is case_pick lift CONDITION-driven or CORRELATE-driven?

Hypothesis under test (robotics-learning paper, not a causal-identification study):
  low-data VLAs execute the contact->lift transition by following a CORRELATE
  (press duration / lift height) rather than the true CONDITION (vacuum seal /
  contact force). The expert lifts because seal is confirmed; the policy lifts
  "around the usual time".

For case_pick (suction), the true lift condition is observable: `vacuum_sealed`
(state idx 8) flips True when the seal seats. At each episode's lift onset we
record, per group (demos vs policy rollouts):

  CONDITION-referenced   seal->lift frames, |F| at lift, sealed-at-lift?
  CORRELATE-referenced   descend->lift frames (press duration), ee_z at lift

A condition-driven agent: seal->lift is tight, sealed-at-lift always True,
premature-lift rate 0. A correlate(time)-driven agent: press-duration tight,
seal->lift loose, and it lifts before seal (premature) -> the signature we expect
from the policy if the hypothesis holds.

Reuses Research/gradual_drift loaders/detectors. numpy only (matplotlib optional).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# reuse the existing recorder loaders + transition detectors
GD = Path(__file__).resolve().parent.parent / "gradual_drift"
sys.path.insert(0, str(GD))
import deviation as dv      # noqa: E402
import transitions as tr    # noqa: E402

SEAL_IDX, EEZ_IDX, SUCTION_IDX = 8, 2, 7
# Min upward travel (m) for a leg to count as a REAL pick-lift. Below this is the
# expert's force-relief retract / small adjustment (sub-cm to a few cm), NOT a lift.
# Amplitude-based so it's invariant to the stack height (real lifts head to hover).
REAL_LIFT_M = 0.08


def seal_onset(states: np.ndarray):
    """First frame where the physical vacuum seal is confirmed, or None."""
    sealed = states[:, SEAL_IDX] > 0.5
    return int(np.argmax(sealed)) if sealed.any() else None


def suction_features(s: np.ndarray) -> dict:
    """Suction-command coherence over the whole episode. A coherent pick turns
    suction on once and holds; chatter = repeated toggles, esp. releasing (1->0)
    AFTER a seal is achieved (the policy un-grasps while holding the case)."""
    suc = (s[:, SUCTION_IDX] > 0.5).astype(int)
    seal = (s[:, SEAL_IDX] > 0.5).astype(int)
    fs = seal_onset(s)
    toggles = int((np.abs(np.diff(suc)) > 0).sum()) if len(suc) > 1 else 0
    if fs is not None and fs < len(suc) - 1:
        post_off = int((np.diff(suc[fs:]) == -1).sum())     # release while holding
        seal_drops = int((np.diff(seal[fs:]) == -1).sum())  # case actually let go
    else:
        post_off = seal_drops = 0
    return {"suction_toggles": toggles,
            "post_seal_suction_off": post_off,
            "seal_drops": seal_drops}


def count_phases(s: np.ndarray, amp: float = 0.03):
    """Zigzag over smoothed z. Returns (lifts, descends): `lifts` is a list of
    (confirm_frame, rise_m) per up-leg, where rise_m = peak-minus-trough of that leg
    and confirm_frame is where it first rose >=amp off the trough (the policy has
    clearly committed to going up); `descends` is a list of confirm_frames. A leg
    registers only after z reverses by >=amp (jitter filter). rise_m lets the caller
    SEPARATE a real pick-lift (rise ~10-27cm) from the expert's small FORCE-RELIEF
    retract (rise <~few cm) via REAL_LIFT_M. Checking seal at the confirm_frame (not
    the trough, which precedes seal by a few frames) avoids false 'lift without seal'."""
    z = tr._smooth(s[:, EEZ_IDX])
    lifts: list = []
    descends: list = []
    if len(z) < 2:
        return lifts, descends
    trend = 0                       # 0 unknown, +1 up-leg, -1 down-leg
    lo = hi = trough = peak = z[0]
    lift_frame = None
    for i, v in enumerate(z):
        if trend == 0:
            lo, hi = min(lo, v), max(hi, v)
            if v - lo >= amp:
                trend, trough, peak, lift_frame = 1, lo, v, i
            elif hi - v >= amp:
                trend, peak, trough = -1, hi, v
        elif trend == 1:
            peak = max(peak, v)
            if peak - v >= amp:                       # reversed down -> close the lift leg
                lifts.append((lift_frame, float(peak - trough)))
                descends.append(i); trend, trough = -1, v
        else:  # trend == -1
            trough = min(trough, v)
            if v - trough >= amp:                      # reversed up -> open a lift leg
                trend, peak, lift_frame = 1, v, i
    if trend == 1 and lift_frame is not None:
        lifts.append((lift_frame, float(peak - trough)))
    return lifts, descends


def lift_features(take: dict) -> dict:
    """Extract condition- and correlate-referenced features at the (first) lift
    onset, plus whole-episode coherence metrics (suction chatter, lift oscillation)."""
    s = take["states"]
    lifts, descends = count_phases(s)                       # lifts: [(frame, rise_m)]
    sealmask = s[:, SEAL_IDX] > 0.5
    so = seal_onset(s)
    real = [(fr, r) for fr, r in lifts if r >= REAL_LIFT_M]  # exclude force-relief retracts
    base = {
        "name": take["name"],
        "deepest_z": float(s[:, EEZ_IDX].min()),   # lowest z reached; tracks target if condition-driven
        "sealed_ever": so is not None,             # grasp ever confirmed (False = hover / under-reach)
        "n_real_lifts": len(real),
        "n_minor_lifts": len(lifts) - len(real),    # force-relief / small adjustments
        # real lift CONFIRMED while still unsealed = "lift without suction done"
        "lifts_no_seal": int(sum(not sealmask[fr] for fr, _ in real)),
        **suction_features(s),
    }
    i = tr.detect_lift_onset(s)
    if i is None:
        return {**base, "lifted": False}
    d = tr.detect_descend_onset(s)
    return {
        **base,
        "lifted": True,
        "lift_frame": int(i),
        "force_lift": float(dv.force_magnitude(s[i:i + 1])[0]),   # condition
        "ee_z_lift": float(s[i, EEZ_IDX]),                        # correlate (pose)
        "sealed_at_lift": bool(s[i, SEAL_IDX] > 0.5),
        "press_dur": (i - d) if d is not None else None,          # correlate (time)
        "seal_to_lift": (i - so) if so is not None else None,     # condition (time)
    }


def collect(take_dirs) -> list[dict]:
    rows = []
    for td in take_dirs:
        try:
            rows.append(lift_features(dv.load_take(td)))
        except Exception as e:  # malformed take -> skip, keep going
            print(f"  [skip] {Path(td).name}: {e}")
    return rows


def _stat(vals):
    """(n, mean, std, CV) over non-None finite values."""
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    m, sd = a.mean(), a.std()
    cv = sd / abs(m) if abs(m) > 1e-9 else float("nan")
    return a.size, m, sd, cv


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    lifted = [r for r in rows if r["lifted"]]
    no_lift = n - len(lifted)
    premature = [r for r in lifted if not r["sealed_at_lift"]]
    never_sealed = [r for r in lifted if not r["sealed_ever"]]

    feats = {
        "force_lift  (CONDITION |F|)": [r["force_lift"] for r in lifted],
        "seal->lift  (CONDITION t)":   [r["seal_to_lift"] for r in lifted],
        "press_dur   (CORRELATE t)":   [r["press_dur"] for r in lifted],
        "ee_z_lift   (CORRELATE z)":   [r["ee_z_lift"] for r in lifted],
    }

    print(f"\n=== {name}  (n={n}) ===")
    print(f"  no-lift (never transitioned): {no_lift}/{n}")
    print(f"  premature lift (lifted before seal): {len(premature)}/{len(lifted)}")
    print(f"  lifted without EVER sealing:         {len(never_sealed)}/{len(lifted)}")
    print(f"  {'feature':<28} {'n':>3} {'mean':>9} {'std':>9} {'CV':>7}")
    out = {}
    for label, vals in feats.items():
        cnt, m, sd, cv = _stat(vals)
        print(f"  {label:<28} {cnt:>3} {m:>9.3f} {sd:>9.3f} {cv:>7.3f}")
        out[label] = (cnt, m, sd, cv)
    # whole-episode coherence (all rows, not just lifted)
    nrl = [r["n_real_lifts"] for r in rows]
    nml = [r["n_minor_lifts"] for r in rows]
    tog = [r["suction_toggles"] for r in rows]
    pso = [r["post_seal_suction_off"] for r in rows]
    sd = [r["seal_drops"] for r in rows]
    lns = [r["lifts_no_seal"] for r in rows]
    print("  -- whole-episode coherence --")
    print(f"  real lifts/episode (rise>={REAL_LIFT_M*100:.0f}cm): mean {np.mean(nrl):.1f} "
          f"(max {max(nrl) if nrl else 0})  | >1 (oscillation): {sum(x > 1 for x in nrl)}/{n}")
    print(f"  minor up-legs (force-relief/adjust, rise<{REAL_LIFT_M*100:.0f}cm): "
          f"mean {np.mean(nml):.2f} (max {max(nml) if nml else 0})")
    print(f"  real lift WITHOUT seal (committed to rising unsealed): mean {np.mean(lns):.2f}  "
          f"| >=1: {sum(x > 0 for x in lns)}/{n}")
    print(f"  suction toggles/ep:   mean {np.mean(tog):.1f} (max {max(tog) if tog else 0})")
    print(f"  release-while-holding (suction 1->0 after seal): mean {np.mean(pso):.2f}  "
          f"| >=1: {sum(x > 0 for x in pso)}/{n}")
    print(f"  seal actually dropped after seal:                mean {np.mean(sd):.2f}  "
          f"| >=1: {sum(x > 0 for x in sd)}/{n}")

    out["_premature_rate"] = len(premature) / max(1, len(lifted))
    out["_no_lift_rate"] = no_lift / max(1, n)
    return out


def maybe_plot(demo_rows, roll_rows, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[plot skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, axi, title in [("force_lift", ax[0], "|F| at lift (CONDITION)"),
                            ("press_dur", ax[1], "press duration (CORRELATE)")]:
        for rows, lab, c in [(demo_rows, "demo", "tab:blue"),
                             (roll_rows, "rollout", "tab:red")]:
            v = [r[key] for r in rows if r["lifted"] and r.get(key) is not None]
            v = [x for x in v if np.isfinite(x)]
            if v:
                axi.hist(v, bins=12, alpha=0.55, label=lab, color=c, density=True)
        axi.set_title(title)
        axi.legend()
    fig.suptitle("Lift onset: what is the policy locked to? (tight = locked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[figure] {out_path}")


def _parse_level(name: str):
    """Layer/height index from a level-dir name. Supports an explicit h<NUM> token
    (e.g. case_pick_h82) or a trailing _<int> layer index (the user's convention:
    case_pick_3 -> 3, where 0 = top/highest target and HIGHER index = LOWER target)."""
    m = re.search(r"h(-?\d+(?:\.\d+)?)", name)
    if m:
        return float(m.group(1))
    m = re.search(r"_(\d+)\s*$", name)
    return float(m.group(1)) if m else None


def by_height(roots, demos=None) -> None:
    """Height-decoupling: group rollouts by SET stack height (parsed from each
    take's parent dir name, e.g. .../case_pick_h82/<take> -> level 82) and regress
    lift height + force on it. condition-driven: ee_z@lift tracks the stack
    (slope ~1 in cm units), |F|@lift flat, seal always achieved. height-correlate:
    ee_z@lift flat (slope ~0), force scatters, seal fails at off-nominal heights."""
    per: dict = {}
    for root in roots:
        for td in dv.list_takes(root):
            lvl = _parse_level(Path(td).parent.name)
            if lvl is None:
                print(f"  [no level] {Path(td).parent.name}: name needs an h<NUM> or _<int> token")
                continue
            try:
                per.setdefault(lvl, []).append(lift_features(dv.load_take(td)))
            except Exception as e:   # in-progress/incomplete take (e.g. meta.json not written yet)
                print(f"  [skip] {Path(td).name}: {e}")
    if not per:
        print("no leveled rollouts found (expect --log-dir dirs named *_<int> or *_h<NUM>)")
        return

    # demo contact-depth band = ground-truth reachable target (expert always seals)
    if demos is not None:
        dd = []
        for td in dv.list_takes(demos):
            try:
                dd.append(float(dv.load_take(td)["states"][:, EEZ_IDX].min()))
            except Exception:
                pass
        if dd:
            dd = np.array(dd)
            print(f"\nDEMO deepest_z band (expert contact depth): min {dd.min():.3f}  "
                  f"med {np.median(dd):.3f}  max {dd.max():.3f}  (n={len(dd)})")

    print(f"\n{'level':>6} {'n':>3} {'sealed%':>8} {'dz(seal)':>10} {'dz(unseal)':>12} "
          f"{'under-reach':>12} {'|F|@lift':>9}")
    lvls, dz_m, se_m = [], [], []
    for lvl in sorted(per):
        rows = per[lvl]
        sealed = [r for r in rows if r["sealed_ever"]]
        unsealed = [r for r in rows if not r["sealed_ever"]]
        lifted = [r for r in rows if r["lifted"]]
        dzs = float(np.mean([r["deepest_z"] for r in sealed])) if sealed else float("nan")
        dzu = float(np.mean([r["deepest_z"] for r in unsealed])) if unsealed else float("nan")
        gap = (dzu - dzs) if (sealed and unsealed) else float("nan")
        sealed_pct = 100 * len(sealed) / len(rows)
        ff = float(np.mean([r["force_lift"] for r in lifted])) if lifted else float("nan")
        print(f"{lvl:>6.0f} {len(rows):>3} {sealed_pct:>7.0f}% {dzs:>10.3f} {dzu:>12.3f} "
              f"{gap:>+12.3f} {ff:>9.2f}")
        lvls.append(lvl); dz_m.append(float(np.mean([r["deepest_z"] for r in rows]))); se_m.append(sealed_pct)
    print("\n  under-reach = dz(unseal) - dz(seal): POSITIVE = failed episodes stop SHALLOWER "
          "(didn't descend enough = condition not used); ~0 = failures reach depth (other cause).")
    if len(lvls) >= 2:
        sdz = float(np.polyfit(lvls, dz_m, 1)[0])
        sse = float(np.polyfit(lvls, se_m, 1)[0])
        print(f"  slope deepest_z vs level = {sdz:+.4f} (0=top, higher=lower target; "
              f"condition-driven NEGATIVE; ~0 = fixed-depth under-reach)")
        print(f"  slope sealed%   vs level = {sse:+.2f} (~0 = holds across layers; NEGATIVE = collapses)")


def timeline(take_dirs, out_dir: Path) -> None:
    """Per-episode timeline: ee_z, |F|, and the suction_cmd / vacuum_sealed signals
    on a shared frame axis, with vertical markers at deepest / seal-onset / lift-onset.
    Lets you eyeball whether the seal signal tracks the descent+contact (e.g. seal
    arrives after contact, lift fires after seal) or is incoherent (under-reach = no
    contact/seal; lift without seal; suction chatter)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[timeline] matplotlib needed: {e}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for td in take_dirs:
        try:
            s = dv.load_take(td)["states"]
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {Path(td).name}: {e}")
            continue
        t = np.arange(len(s))
        z, f = s[:, EEZ_IDX], dv.force_magnitude(s)
        suc = (s[:, SUCTION_IDX] > 0.5).astype(int)
        seal = (s[:, SEAL_IDX] > 0.5).astype(int)
        so, li, dp = seal_onset(s), tr.detect_lift_onset(s), int(np.argmin(z))

        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2, 1]})
        ax[0].plot(t, z, color="tab:blue"); ax[0].set_ylabel("ee_z (m)")
        ax[1].plot(t, f, color="tab:orange"); ax[1].set_ylabel("|F| (N)")
        ax[1].axhline(15.8, ls=":", color="gray", lw=1)   # demo lift-force reference
        ax[2].step(t, seal, where="post", color="tab:green", label="vacuum_sealed")
        ax[2].step(t, suc, where="post", color="tab:red", lw=1, label="suction_cmd")
        ax[2].set_ylim(-0.15, 1.15); ax[2].set_yticks([0, 1]); ax[2].set_ylabel("signal")
        ax[2].set_xlabel("frame"); ax[2].legend(loc="center right", fontsize=8)
        for a in ax:
            for fr, c in [(dp, "gray"), (so, "tab:green"), (li, "tab:blue")]:
                if fr is not None:
                    a.axvline(fr, color=c, ls="--", lw=1, alpha=0.7)
        ax[0].set_title(f"{Path(td).name}   deepest_z={z.min():.3f}  "
                        f"seal@{so}  lift@{li}  (markers: gray=deepest grn=seal blu=lift)")
        fig.tight_layout()
        out = out_dir / f"timeline_{Path(td).name}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"[timeline] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--demos", type=Path,
                    default=repo / "LGES/recordings/case_pick")
    ap.add_argument("--rollouts", type=Path, nargs="+",
                    default=[repo / "Research/gradual_drift/rollouts/baseline"])
    ap.add_argument("--dump", action="store_true",
                    help="print per-episode rollout features (small-n eyeball)")
    ap.add_argument("--by-height", type=Path, nargs="+", default=None, metavar="ROOT",
                    help="height-decoupling: roots whose take parent dirs are named "
                         "*_h<NUM> or *_<int> (layer). Regress lift height/force on level.")
    ap.add_argument("--timeline", type=Path, nargs="+", default=None, metavar="TAKE",
                    help="plot z / |F| / suction_cmd+vacuum_sealed timelines for the given "
                         "take dir(s) -> timelines/. Glob a layer, e.g. rollouts/case_pick_4/*")
    args = ap.parse_args()

    if args.timeline:
        timeline(args.timeline, Path(__file__).resolve().parent / "timelines")
        return
    if args.by_height:
        by_height(args.by_height, args.demos)
        return

    demo_rows = collect(dv.list_takes(args.demos))
    roll_dirs = []
    for r in args.rollouts:
        roll_dirs += dv.list_takes(r)
    roll_rows = collect(roll_dirs)

    summarize(f"DEMOS  {args.demos.name}", demo_rows)
    summarize("ROLLOUTS  " + ", ".join(p.name for p in args.rollouts), roll_rows)

    if args.dump:
        print("\n--- per-rollout features ---")
        for r in roll_rows:
            tag = (f"|F|={r['force_lift']:.1f} seal->lift={r['seal_to_lift']} "
                   f"press_dur={r['press_dur']} sealed@lift={r['sealed_at_lift']}"
                   if r["lifted"] else "NO (first-)LIFT")
            print(f"  {r['name']}: {tag} | real_lifts={r['n_real_lifts']} "
                  f"minor={r['n_minor_lifts']} lifts_no_seal={r['lifts_no_seal']} "
                  f"toggles={r['suction_toggles']} rel-while-hold={r['post_seal_suction_off']}")

    maybe_plot(demo_rows, roll_rows, Path(__file__).resolve().parent / "lift_condition_probe.png")


if __name__ == "__main__":
    main()
