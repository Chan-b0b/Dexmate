#!/usr/bin/env python3
"""P0 — quantify the open-loop action-chunk gap around contact events.

Phase P0 of REACTIVE_CHUNKING_RESEARCH.md. Pure offline analysis on recorded
takes; commands no robot, trains nothing, and MODIFIES NOTHING in vla_training
(it imports it as a library).

The question: SmolVLA executes a 50-step chunk open-loop and only re-reads the
observation at chunk boundaries (modeling_smolvla.py select_action). Does the
action it is replaying mid-chunk diverge from what it would freshly predict if
it re-read the observation — and does that divergence concentrate around the
vacuum-seal contact event (where reactivity matters)?

Method (deterministic, so divergence reflects the OBSERVATION changing, not
flow-matching sampling noise): for every frame t of a take, predict a fresh
50-step chunk C[t] from that frame's recorded observation. Then:

  deploy action at t : C[s][t-s]   with chunk start s = floor(t/K)*K  (STALE)
  re-grounded at t   : C[t][0]                                        (FRESH)
  recorded (truth)   : the take's next-state delta action at t

We measure the deploy-vs-fresh divergence as a function of intra-chunk lag and
of distance-to-seal-event, the reaction lag (ticks from a seal event to the
next chunk boundary, when the policy can first react), and the descent->lift
behaviour (dz) around the seal event.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/reactive_chunking/p0_chunk_gap.py \
      [--task case_pick] [--max-takes N] [--checkpoint DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]          # .../Dexmate
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))                  # import the pipeline as a library

import cv2  # noqa: E402  (after path setup, mirrors converter)
from convert_to_lerobot import load_take, colorize_depth, IMG_W, IMG_H, FPS  # noqa: E402
from run_policy import load_policy, _to_chw  # noqa: E402

# state layout (convert_to_lerobot.STATE_NAMES): pos 0:3, quat 3:7, suction 7,
# vacuum_sealed 8, wrench 9:15.  action: dpos 0:3, drot 3:6, suction 6.
SEAL_COL = 8


def _load_rgb(p: Path) -> np.ndarray:
    """Same as convert_to_lerobot.write_episode: BGR jpg -> resize -> RGB."""
    img = cv2.imread(str(p))
    return cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)


def _load_depth(p: Path) -> np.ndarray:
    return colorize_depth(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))


def predict_chunks(policy, pre, post, states, rgb_paths, depth_paths, instruction,
                   seed: int) -> np.ndarray:
    """Per-frame fresh action chunk. Returns (N, K, 7) un-normalized actions.

    Deterministic: the same noise seed is reused for every frame, so two frames
    with identical observations yield identical chunks and any divergence is
    purely observation-driven (isolates the staleness effect from sampling)."""
    n = len(states)
    chunks = []
    for i in range(n):
        obs = {
            "observation.images.head": _to_chw(_load_rgb(rgb_paths[i])),
            "observation.images.head_depth": _to_chw(_load_depth(depth_paths[i])),
            "observation.state": torch.from_numpy(states[i]).unsqueeze(0),
            "task": instruction,
        }
        obs = pre(obs)
        policy.reset()                      # no obs/action history leaks across frames
        torch.manual_seed(seed)             # identical flow-matching noise every call
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(obs)   # (1, K, 7) normalized
        chunk = chunk[0]
        out = np.stack([post(chunk[k:k + 1]).squeeze(0).cpu().numpy()
                        for k in range(chunk.shape[0])])
        chunks.append(out)
        if (i + 1) % 25 == 0:
            print(f"      frame {i + 1}/{n}")
    return np.asarray(chunks)               # (N, K, 7)


def seal_rising_edges(states: np.ndarray) -> list[int]:
    s = states[:, SEAL_COL] > 0.5
    return [i for i in range(1, len(s)) if s[i] and not s[i - 1]]


def analyze_take(chunks: np.ndarray, actions: np.ndarray, states: np.ndarray, K: int):
    """Accumulate raw per-frame divergences for one take into dicts of lists."""
    n = len(chunks)
    events = seal_rising_edges(states)
    # distance (in ticks) from each frame to the NEXT seal event (np.inf if none)
    d2c = np.full(n, np.inf)
    for i in range(n):
        future = [e for e in events if e >= i]
        if future:
            d2c[i] = future[0] - i

    rows = []  # one dict per frame
    for t in range(n):
        s = (t // K) * K
        L = t - s
        deploy = chunks[s][L]      # stale: planned L steps ago for now
        fresh = chunks[t][0]       # re-grounded: planned now
        rec = actions[t] if t < len(actions) else None
        row = dict(
            t=t, L=L, d2c=float(d2c[t]),
            dpos_dev=float(np.linalg.norm(deploy[:3] - fresh[:3])),
            drot_dev=float(np.linalg.norm(deploy[3:6] - fresh[3:6])),
            suc_dev=float(abs(deploy[6] - fresh[6])),
            deploy_dz=float(deploy[2]), fresh_dz=float(fresh[2]),
            deploy_suc=float(deploy[6]), fresh_suc=float(fresh[6]),
        )
        if rec is not None:
            row.update(rec_dz=float(rec[2]), rec_suc=float(rec[6]),
                       deploy_vs_rec=float(np.linalg.norm(deploy[:3] - rec[:3])),
                       fresh_vs_rec=float(np.linalg.norm(fresh[:3] - rec[:3])))
        rows.append(row)

    # staleness vs intra-chunk lag L: action-for-time-t computed L steps ago vs now
    stale = []
    for L in range(K):
        for t in range(L, n):
            a = chunks[t - L][L]
            b = chunks[t][0]
            stale.append(dict(L=L,
                              dpos=float(np.linalg.norm(a[:3] - b[:3])),
                              drot=float(np.linalg.norm(a[3:6] - b[3:6])),
                              suc=float(abs(a[6] - b[6]))))

    # reaction lag: ticks from each seal event to the next chunk boundary >= it
    lags = [(int(np.ceil(e / K) * K) - e) for e in events]
    return rows, stale, lags, events


def summarize(rows, stale, lags, K, out_dir, make_plots):
    rows = [r for r in rows]
    R = {k: np.array([r[k] for r in rows if k in r]) for k in rows[0]}
    L_all = np.array([s["L"] for s in stale])

    def by_L(key):
        return np.array([np.mean([s[key] for s in stale if s["L"] == L]) for L in range(K)])

    stale_pos = by_L("dpos") * 1000.0     # mm
    stale_rot = by_L("drot") * 1000.0     # mrad
    stale_suc = by_L("suc")

    lags = np.array(lags)
    near = R["d2c"] <= 5                   # within 5 ticks before a seal event
    far = R["d2c"] > 20

    summary = {
        "frames": int(len(rows)),
        "seal_events": int(len(lags)),
        "reaction_lag_ticks": dict(mean=float(lags.mean()), max=int(lags.max()),
                                   p90=float(np.percentile(lags, 90))) if len(lags) else {},
        "reaction_lag_seconds": dict(mean=float(lags.mean() / FPS),
                                     max=float(lags.max() / FPS)) if len(lags) else {},
        "staleness_dpos_mm": dict(at_L0=float(stale_pos[0]), at_Lmax=float(stale_pos[-1]),
                                  mean_over_L=float(stale_pos.mean())),
        "staleness_drot_mrad": dict(at_L0=float(stale_rot[0]), at_Lmax=float(stale_rot[-1])),
        "deploy_vs_fresh_dpos_mm": dict(
            near_contact=float(R["dpos_dev"][near].mean() * 1000) if near.any() else None,
            far_from_contact=float(R["dpos_dev"][far].mean() * 1000) if far.any() else None),
        "suction_disagree_near_contact": dict(
            near=float((R["suc_dev"][near] > 0.5).mean()) if near.any() else None,
            far=float((R["suc_dev"][far] > 0.5).mean()) if far.any() else None),
    }
    # recorded-tracking: the last frame of each take has no next-state action,
    # so these arrays are shorter than d2c — build them with their own mask.
    has_rec = [r for r in rows if "deploy_vs_rec" in r]
    if has_rec:
        d2c_rec = np.array([r["d2c"] for r in has_rec])
        near_rec = d2c_rec <= 5
        if near_rec.any():
            dvr = np.array([r["deploy_vs_rec"] for r in has_rec])[near_rec]
            fvr = np.array([r["fresh_vs_rec"] for r in has_rec])[near_rec]
            summary["track_recorded_dpos_mm_near_contact"] = dict(
                deploy=float(dvr.mean() * 1000), fresh=float(fvr.mean() * 1000))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== P0 SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'summary.json'}")

    if make_plots:
        _plots(stale_pos, stale_rot, stale_suc, R, lags, K, out_dir)
    return summary


def _plots(stale_pos, stale_rot, stale_suc, R, lags, K, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plots skipped: {e})")
        return
    Ls = np.arange(K)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(Ls, stale_pos); ax[0].set(title="staleness: |dpos_stale - dpos_fresh|",
                                          xlabel="intra-chunk lag L (ticks)", ylabel="mm")
    ax[1].plot(Ls, stale_rot); ax[1].set(title="rotation staleness",
                                          xlabel="L (ticks)", ylabel="mrad")
    ax[2].plot(Ls, stale_suc); ax[2].set(title="suction staleness",
                                          xlabel="L (ticks)", ylabel="|Δ suction|")
    fig.tight_layout(); fig.savefig(out_dir / "staleness_vs_lag.png", dpi=110); plt.close(fig)

    # deploy-vs-fresh divergence binned by distance to seal event
    d2c = R["d2c"]; fin = np.isfinite(d2c)
    bins = np.arange(0, 31, 2)
    idx = np.digitize(d2c[fin], bins)
    centers, vals = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.any():
            centers.append(bins[b - 1]); vals.append(R["dpos_dev"][fin][m].mean() * 1000)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, vals, "o-")
    ax.invert_xaxis()
    ax.set(title="deploy-vs-fresh dpos divergence near seal event",
           xlabel="ticks until seal event (→ contact)", ylabel="mm")
    fig.tight_layout(); fig.savefig(out_dir / "divergence_vs_distance_to_contact.png", dpi=110); plt.close(fig)

    if len(lags):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(lags, bins=np.arange(0, K + 2))
        ax.set(title=f"reaction lag: ticks from seal event to next chunk boundary (K={K})",
               xlabel="ticks", ylabel="count")
        ax.axvline(lags.mean(), color="r", ls="--", label=f"mean {lags.mean():.1f} ticks "
                   f"({lags.mean()/FPS*1000:.0f} ms)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "reaction_lag_hist.png", dpi=110); plt.close(fig)
    print(f"wrote plots -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="case_pick")
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_TRAIN / "outputs/smolvla_depthseal_fixed/checkpoints/last")
    ap.add_argument("--max-takes", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "p0_results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {args.checkpoint}")
    policy, pre, post = load_policy(args.checkpoint)
    K = int(policy.config.chunk_size)
    print(f"chunk_size K = {K}, n_action_steps = {policy.config.n_action_steps}")

    take_dirs = sorted(p for p in (args.recordings / args.task).iterdir() if p.is_dir())
    if args.max_takes:
        take_dirs = take_dirs[:args.max_takes]
    print(f"task={args.task}: {len(take_dirs)} takes\n")

    all_rows, all_stale, all_lags = [], [], []
    for ti, td in enumerate(take_dirs):
        loaded = load_take(td, with_depth=True)
        if loaded is None:
            print(f"  [{ti+1}/{len(take_dirs)}] SKIP {td.name} (not success / too short)")
            continue
        instruction, rgb_paths, depth_paths, states, actions = loaded
        print(f"  [{ti+1}/{len(take_dirs)}] {td.name}: {len(states)} frames")
        chunks = predict_chunks(policy, pre, post, states, rgb_paths, depth_paths,
                                instruction, args.seed)
        rows, stale, lags, events = analyze_take(chunks, actions, states, K)
        print(f"      seal events at {events}, reaction lags {lags}")
        all_rows += rows; all_stale += stale; all_lags += lags

    if not all_rows:
        print("no usable takes."); return
    summarize(all_rows, all_stale, all_lags, K, args.out, not args.no_plots)


if __name__ == "__main__":
    main()
