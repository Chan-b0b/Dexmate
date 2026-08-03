#!/usr/bin/env python3
"""Closed-loop press simulation — how deep does the policy press before it stops?

The offline analogue of the real-robot overpress failure. Start from the REAL frame just
before each episode's first contact (image + state as recorded). Then iterate:

    dz = policy(state with wrench = first-contact direction at |F| = F0 + k * p)
    if dz < 0 (still descending): penetration p += |dz|; force rises with p (stiffness k)
    stop when dz >= 0 for 2 consecutive steps

Reported per episode: stopped?, steps, final penetration p (mm), final |F| (N).
This measures the RAMP interaction single-frame probes cannot: a policy whose brake grows
with force (FiLM c-hat) converges shallow; a policy with a fixed-size force response
(naive raw path) needs the seal (--seal-depth) or never stops.

Limitation: the image is frozen at the start frame — visual press cues are absent, which
is the conservative direction for models that lean on vision.

Env contract identical to the other probes (FILM_* must match the checkpoint); --naive
for vanilla checkpoints.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

VLA_DIR = Path(__file__).resolve().parent
EE_Z = 2
SEAL_IDX, F_LO, F_HI, W_LO, W_HI = 8, 9, 12, 9, 15
FZ_IDX = 11
DZ = 2

import film_contact  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=VLA_DIR / "datasets/lges_case_pick_0729_val")
    ap.add_argument("--repo-id", default="Chanho-Lee/lges_case_pick_0729_val")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--stiffness", type=float, default=1.0,
                    help="contact stiffness k (N per mm of penetration)")
    ap.add_argument("--f-base", type=float, default=6.8,
                    help="|F| (N) at first touch (p=0) — the measured first-contact level")
    ap.add_argument("--f-cap", type=float, default=25.0, help="force saturation (N)")
    ap.add_argument("--seal-depth", type=float, default=0.0,
                    help="penetration (mm) at which the vacuum SEALS (seal bit -> 1); "
                         "0 = seal never fires (misaligned-suction scenario)")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--start-offset", type=int, default=1,
                    help="start the press N frames BEFORE the episode's first contact: 1 = "
                         "nominal timing (policy already decelerated); 30 = contact arrives "
                         "2s EARLY, during committed descent (unexpected-height scenario)")
    ap.add_argument("--force-model", choices=("pattern", "fzdelta"), default="pattern",
                    help="'pattern' = wrench REPLACED by the demo-median first-contact "
                         "direction scaled to f_base + k*p (naive-favorable: matches the "
                         "training template). 'fzdelta' = the frame's OWN wrench plus a "
                         "normal force on fz only (delta0 + k*p) — physically what contact "
                         "adds; the static fy mount bias stays whatever it was")
    ap.add_argument("--fz-delta0", type=float, default=1.7,
                    help="[fzdelta] fz jump (N) at first touch (measured ~3.65-2.0)")
    args = ap.parse_args()

    if not args.naive:
        cond = tuple(c.strip() for c in
                     os.environ.get("FILM_COND", "contact").split(",") if c.strip())
        mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
        inject = os.environ.get("FILM_INJECT", "suffix")
        print(f"[sim] FiLM cond={cond} inject={inject} mask_force={mask_force}  "
              f"ckpt={args.checkpoint}")
        wm, ws = film_contact.load_wrench_stats(args.dataset_root)
        sm, ss = film_contact.load_seal_stats(args.dataset_root)
        dm, dsd = film_contact.load_dfmag_stats(args.dataset_root)
        film_contact.apply(
            "v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
            contact_F0=float(os.environ.get("FILM_F0", "6")),
            contact_tau=float(os.environ.get("FILM_TAU", "4")),
            fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
            fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
            mask_force=mask_force, inject=inject,
            dfmag_mean=dm, dfmag_std=dsd,
            dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")),
            fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
            fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")))
    else:
        print(f"[sim] NAIVE checkpoint  ckpt={args.checkpoint}")
    print(f"[sim] stiffness={args.stiffness}N/mm  f_base={args.f_base}N  "
          f"seal_depth={args.seal_depth or 'never'}mm  max_steps={args.max_steps}  "
          f"start_offset={args.start_offset}  force_model={args.force_model}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    model_dir = args.checkpoint / "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.eval()
    policy.config.n_action_steps = 1
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    bounds = [(e["dataset_from_index"], e["dataset_to_index"]) for e in ds.meta.episodes]

    # first-contact wrench pattern (pooled, same harvest as probe_state_authority)
    import glob
    import pandas as pd
    pool = []
    for p in sorted(glob.glob(str(args.dataset_root / "data" / "*" / "*.parquet"))):
        df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
        stt = np.stack(df["observation.state"].to_numpy())
        for e in df["episode_index"].unique():
            s = stt[(df["episode_index"] == e).to_numpy()]
            fz = s[:, FZ_IDX]
            seals = np.flatnonzero(s[:, SEAL_IDX] > 0.5)
            ts = seals[0] if len(seals) else len(s)
            cand = np.flatnonzero((fz[:-1] > 3.0) & (fz[1:] > 3.0))
            cand = cand[(cand > 10) & (cand < ts)]
            if len(cand):
                w = s[cand[0]:min(cand[0] + 3, ts)]
                pool.append(w[w[:, SEAL_IDX] <= 0.5][:, W_LO:W_HI])
    fc = np.median(np.concatenate(pool), axis=0)
    fc_unit = fc[:3] / np.linalg.norm(fc[:3])
    fc_torque = fc[3:]
    print(f"[sim] contact direction={np.round(fc_unit, 2)}  torque={np.round(fc_torque, 2)}")

    def predict(frame, raw_state):
        policy.reset()
        torch.manual_seed(0)
        obs = {"observation.state": raw_state.unsqueeze(0), "task": frame["task"]
               if "task" in frame else task}
        for k in frame:
            if k.startswith("observation.images."):
                obs[k] = frame[k].unsqueeze(0)
        with torch.inference_mode():
            a = policy.select_action(pre(obs))
        return post(a).squeeze(0).cpu().numpy()

    results = []
    for ep, (lo, hi) in enumerate(bounds):
        task = ds[lo]["task"]
        if not ("case" in task.lower() and "pick" in task.lower()
                and "place" not in task.lower()):
            continue
        states = np.stack([ds[i]["observation.state"].numpy() for i in range(lo, hi)])
        fmag = np.linalg.norm(states[:, F_LO:F_HI], axis=1)
        sealed = states[:, SEAL_IDX] > 0.5
        ts = int(np.flatnonzero(sealed)[0]) if sealed.any() else len(states)
        cand = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= 2)
        cand = cand[(cand >= max(0, ts - 60)) & (cand < ts)]
        tf = int(cand[0]) if len(cand) else (ts - 33 if ts > 33 else None)
        if tf is None or tf < args.start_offset:
            continue
        frame = ds[lo + tf - args.start_offset]  # real frame; offset 1 = just before contact
        base = frame["observation.state"].clone()

        p, stop_ct, hist = 0.0, 0, []
        outcome = "timeout"
        for step in range(args.max_steps):
            st = base.clone()
            if args.force_model == "pattern":
                f_mag = min(args.f_base + args.stiffness * p, args.f_cap)
                st[F_LO:F_HI] = torch.tensor(fc_unit * f_mag, dtype=st.dtype)
                st[F_HI:W_HI] = torch.tensor(fc_torque, dtype=st.dtype)
            else:                                # fzdelta: contact adds normal force on fz
                add = min(args.fz_delta0 + args.stiffness * p, args.f_cap)
                st[FZ_IDX] = st[FZ_IDX] + add
                f_mag = float(np.linalg.norm(st[F_LO:F_HI].numpy()))
            st[SEAL_IDX] = 1.0 if (args.seal_depth > 0 and p >= args.seal_depth) else 0.0
            dz = float(predict(frame, st)[DZ]) * 1000.0   # mm
            hist.append((f_mag, dz))
            if dz >= 0:
                stop_ct += 1
                if stop_ct >= 2:
                    outcome = "STOP"
                    break
            else:
                stop_ct = 0
                p += -dz
        sealed_in_sim = args.seal_depth > 0 and p >= args.seal_depth
        results.append((ep, states[tf, EE_Z], outcome, step + 1, p,
                        hist[-1][0], sealed_in_sim))
        traj = "  ".join(f"{f:.1f}N/{d:+.1f}" for f, d in hist[:8])
        print(f"  ep{ep} (z={states[tf, EE_Z]:.3f}): {outcome:7s} steps={step+1:>2} "
              f"depth={p:5.1f}mm  |F|end={results[-1][5]:4.1f}N  sealed={sealed_in_sim}")
        print(f"        first steps [|F|/dz(mm)]: {traj}")

    print("\n── summary ─────────────────────────────────────────────────────────")
    arr_p = np.array([r[4] for r in results])
    n_stop = sum(1 for r in results if r[2] == "STOP")
    print(f"  stopped {n_stop}/{len(results)} episodes   "
          f"penetration depth mean={arr_p.mean():.1f}mm  max={arr_p.max():.1f}mm")
    for zlab, zsel in (("z~0.763", lambda z: z < 0.79), ("z~0.817", lambda z: z >= 0.79)):
        sub = [r for r in results if zsel(r[1])]
        if sub:
            print(f"    {zlab}: stopped {sum(1 for r in sub if r[2]=='STOP')}/{len(sub)}  "
                  f"depth mean={np.mean([r[4] for r in sub]):.1f}mm")


if __name__ == "__main__":
    main()
