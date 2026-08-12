#!/usr/bin/env python3
"""State-swap counterfactual probe — does the RAW wrench/seal in observation.state
drive the action?

Complements probe_film_authority.py (which forces c-hat -> FiLM-path authority only):
here the STATE ITSELF is perturbed. Per descent frame, the wrench dims (9:15) and the
seal bit (8) are replaced with values harvested from real CONTACT frames of the same
episode (per-dim median over sealed frames -> in-distribution, not synthetic).

  naive ckpt (--naive): dz(own) vs dz(swap)          -> raw-state-path authority
  film ckpt: 2x2 factorial (state: own/swap) x (c-hat: own/swap):
      dz(own , c=own )  baseline (what deploy sees mid-descent)
      dz(swap, c=swap)  TOTAL    (deploy-realistic contact counterfactual)
      dz(swap, c=own )  RAW only (c-hat pinned to the descent frame's value)
      dz(own , c=swap)  FiLM only(= the c-hat probe, but with an in-distribution c1)
    decomposition:  dTotal = dRaw + dFiLM + interaction

Predictions per config:
  mask0: raw path is UNMASKED -> if the raw path owns the behavior, dRaw ~ dTotal and
         dFiLM ~ 0 (matches the "FiLM has no authority under mask0" c-hat-probe result).
  mask1: wrench/seal are ZEROED from the state -> the swap can only act via c-hat, so
         dRaw ~ 0 and dFiLM ~ dTotal (sanity check of the bottleneck).
  naive: dz(swap)-dz(own) is the total contact-response the raw path learned unaided.

Same env contract as probe_film_authority.py (must match the checkpoint):
  FILM_COND=contact,fz,seal FILM_MASK_FORCE=0 FILM_INJECT=prefix FILM_FZ_OFF=... \
    python probe_state_authority.py --checkpoint outputs/<run>/checkpoints/best \
      --dataset-root datasets/lges_case_pick_0729_val --repo-id Chanho-Lee/lges_case_pick_0729_val
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

VLA_DIR = Path(__file__).resolve().parent
EE_Z = 2
SEAL_IDX, F_LO, F_HI, W_LO, W_HI = 8, 9, 12, 9, 15  # state idx: seal; force xyz; full wrench
FZ_IDX = 11                                          # state idx: fz (wrench[2])
DZ = 2                                               # action idx of dz

import film_contact  # noqa: E402

# ── forced-c override (same mechanism as probe_film_authority.py): None => real ──
_FORCE = {"c": None}
_orig_cond = film_contact._condition_from_state


def _forced_cond(self, state):
    if _FORCE["c"] is None:
        return _orig_cond(self, state)
    return _FORCE["c"].to(state.device).expand(state.shape[0], -1).clone()


film_contact._condition_from_state = _forced_cond

# pi0.5 computes c-hat through its OWN module-level function — film_contact_pi05
# ._cond_from_state, called from the policy-level _set_cond closure — so patching
# film_contact alone leaves the pi05 path reading REAL c-hat and every forced cell below
# silently becomes a no-op. Install the same override there. (Closures resolve globals at
# call time, so rebinding the module attribute is enough.)
import film_contact_pi05 as fcp  # noqa: E402

_orig_cond_pi05 = fcp._cond_from_state


def _forced_cond_pi05(model, state_norm):
    if _FORCE["c"] is None:
        return _orig_cond_pi05(model, state_norm)
    return _FORCE["c"].to(state_norm.device).expand(state_norm.shape[0], -1).clone()


fcp._cond_from_state = _forced_cond_pi05

# π0 (film_contact_pi0) imported film_contact._condition_from_state into its OWN namespace
# at module load, so the film_contact rebind above doesn't reach it — the same silent-no-op
# trap as pi05. Rebind the pi0 module attribute too; π0 state is MEAN_STD-normalized like
# SmolVLA's, so film_contact's forced fn is signature-compatible as-is.
import film_contact_pi0 as fc0  # noqa: E402

fc0._condition_from_state = _forced_cond


def c_from_raw(model, raw_state: torch.Tensor) -> torch.Tensor:
    """c-hat (1, cond_dim) from a RAW (un-normalized) state vector, via the model's own
    stats buffers + the original condition fn — exactly what the FiLM path would compute.

    The two architectures normalize differently, so this has to branch: SmolVLA z-scores the
    wrench with _wrench_mean/_wrench_std, while pi0.5 quantile-normalizes the whole state to
    [-1,1] with _film_q01/_film_q99. Using the mean/std path on a pi05 model raises
    AttributeError; using it after registering those buffers would silently mis-normalize."""
    if hasattr(model, "_film_q01"):                      # pi0.5: invert the quantile map
        q01, q99 = model._film_q01, model._film_q99
        n = raw_state.shape[-1]
        span = (q99[:n] - q01[:n]).clamp_min(1e-6)
        s = torch.zeros(1, n, device=q01.device)
        raw = raw_state.to(q01.device)
        # only the dims c-hat reads need to decode back to `raw`; the others stay at the
        # quantile midpoint, which _cond_from_state never looks at
        for i in list(range(W_LO, W_HI)) + [SEAL_IDX]:
            s[0, i] = 2.0 * (raw[i] - q01[i]) / span[i] - 1.0
        with torch.no_grad():
            return _orig_cond_pi05(model, s).cpu()
    dev = model._wrench_mean.device
    s = torch.zeros(1, raw_state.shape[-1], device=dev)
    s[0, W_LO:W_HI] = (raw_state[W_LO:W_HI].to(dev) - model._wrench_mean) / model._wrench_std
    if hasattr(model, "_seal_mean"):
        s[0, SEAL_IDX] = (raw_state[SEAL_IDX].to(dev) - model._seal_mean) / model._seal_std
    with torch.no_grad():
        return _orig_cond(model, s).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path,
                    default=VLA_DIR / "datasets/lges_case_pick_0729_val")
    ap.add_argument("--stats-root", type=Path, default=None,
                    help="dataset whose stats feed c-hat (default: --dataset-root); set to the "
                         "TRAINING dataset when probing held-out episodes")
    ap.add_argument("--repo-id", default="Chanho-Lee/lges_case_pick_0729_val")
    ap.add_argument("--episode", type=int, default=None,
                    help="episode index (default: first case_pick episode)")
    ap.add_argument("--all-episodes", action="store_true",
                    help="pool descent frames over ALL case_pick episodes of the dataset "
                         "(per-episode first-contact detection for the time-to-contact split)")
    ap.add_argument("--episodes", default=None,
                    help="comma list of episode indices to pool (e.g. '0,1,2' = one layer "
                         "group); overrides --episode/--all-episodes")
    ap.add_argument("--descend-n", type=float, default=6.0,
                    help="raw |F| (N) below which a frame counts as pre-contact/descent")
    ap.add_argument("--naive", action="store_true",
                    help="vanilla checkpoint: no FiLM patch, state-swap cells only")
    ap.add_argument("--film-pi05", action="store_true",
                    help="the checkpoint is a pi0.5 FiLM run: patch via film_contact_pi05 "
                         "(suffix-only, quantile-normalized state) instead of film_contact. "
                         "Its numbers are NOT comparable to the SmolVLA recal runs (prefix, "
                         "4 channels) — read them only as sign/shape reproduction.")
    ap.add_argument("--film-pi0", action="store_true",
                    help="the checkpoint is a π0 FiLM run: patch via film_contact_pi0 "
                         "(MEAN_STD state like SmolVLA; FILM_INJECT must be state|action — "
                         "the token-level injection probe)")
    ap.add_argument("--swap", choices=("both", "wrench", "seal", "onset", "firstcontact",
                                       "fcscale", "bias"),
                    default="both",
                    help="which state dims the contact-swap replaces (default both): "
                         "isolates whether the wrench or the seal bit carries the authority. "
                         "'onset' = wrench from PRE-SEAL press frames (seal=0, pooled over ALL "
                         "episodes) — late press, ~33 frames after first contact. "
                         "'firstcontact' = wrench at the FIRST force rise (fz crosses "
                         "--fc-fz-thresh, the measured expert stop trigger; frames t..t+2, "
                         "seal=0) — the exact signal a descending robot sees at first touch")
    ap.add_argument("--fc-fz-thresh", type=float, default=3.0,
                    help="fz (N) whose first sustained crossing defines first contact")
    ap.add_argument("--fc-mag", type=float, default=8.0,
                    help="[--swap fcscale] target |F[:3]| (N): the first-contact wrench with "
                         "its force xyz rescaled to this magnitude (press-force ramp sweep)")
    ap.add_argument("--pre-contact", type=int, default=0,
                    help="evaluate ONLY the N frames immediately before each episode's first "
                         "contact (the moment the stop response matters); replaces the "
                         "|F|<descend-n frame filter. 0 = off")
    ap.add_argument("--bias", default="0,0,1",
                    help="[--swap bias] comma fx,fy,fz (N) ADDED to each frame's own force "
                         "(sensor-drift sensitivity of the BASELINE descent, not a contact "
                         "signal; seal untouched)")
    args = ap.parse_args()

    naive = args.naive
    if sum((naive, args.film_pi05, args.film_pi0)) > 1:
        ap.error("--naive, --film-pi05 and --film-pi0 are mutually exclusive")
    if not naive:
        cond = tuple(c.strip() for c in
                     os.environ.get("FILM_COND", "contact").split(",") if c.strip())
        mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
        inject = os.environ.get("FILM_INJECT", "suffix")
        stats_root = args.stats_root or args.dataset_root
        if args.film_pi05:
            # pi0.5 has no state token, so there is no prefix analogue — inject is always
            # suffix here regardless of FILM_INJECT, and apply() takes quantiles, not
            # wrench/seal/dfmag mean-std pairs.
            print(f"[probe] FiLM-pi0.5 cond={cond} inject=suffix (only) "
                  f"mask_force={mask_force}  ckpt={args.checkpoint}")
            if inject != "suffix":
                print(f"[probe] NOTE: FILM_INJECT={inject} ignored — pi0.5 is suffix-only")
            q01, q99 = fcp.load_state_quantiles(stats_root)
            fcp.apply(
                "v2", q01, q99, cond=cond,
                contact_F0=float(os.environ.get("FILM_F0", "6")),
                contact_tau=float(os.environ.get("FILM_TAU", "4")),
                fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
                fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
                fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
                fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
                dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")),
                mask_force=mask_force)
        elif args.film_pi0:
            # π0: MEAN_STD state, so film_contact's stat loaders apply verbatim; inject
            # picks the TOKEN (state|action), both inside embed_suffix.
            inject = os.environ.get("FILM_INJECT", "state")
            print(f"[probe] FiLM-pi0 cond={cond} inject={inject} mask_force={mask_force}  "
                  f"ckpt={args.checkpoint}")
            wm, ws = film_contact.load_wrench_stats(stats_root)
            sm, ss = film_contact.load_seal_stats(stats_root)
            dm, dsd = film_contact.load_dfmag_stats(stats_root)
            fc0.apply(
                "v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                contact_F0=float(os.environ.get("FILM_F0", "6")),
                contact_tau=float(os.environ.get("FILM_TAU", "4")),
                fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
                fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
                fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
                fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
                dfmag_mean=dm, dfmag_std=dsd,
                dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")),
                inject=inject, mask_force=mask_force)
        else:
            print(f"[probe] FiLM cond={cond} inject={inject} mask_force={mask_force}  "
                  f"ckpt={args.checkpoint}")
            wm, ws = film_contact.load_wrench_stats(stats_root)
            sm, ss = film_contact.load_seal_stats(stats_root)
            dm, dsd = film_contact.load_dfmag_stats(stats_root)
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
        print(f"[probe] NAIVE checkpoint (no FiLM)  ckpt={args.checkpoint}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    try:
        import train_pi05  # noqa: F401  pi05 ckpts: registers the preprocessor shim
    except ValueError:
        pass  # newer lerobot ships relative_actions_processor natively — shim collides

    model_dir = args.checkpoint / "pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(model_dir)
    policy = get_policy_class(cfg.type).from_pretrained(model_dir, config=cfg)
    policy.eval()
    policy.config.n_action_steps = 1
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})

    # ── episode + swap-state harvest ─────────────────────────────────────────
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    bounds = [(e["dataset_from_index"], e["dataset_to_index"]) for e in ds.meta.episodes]
    def is_case_pick(i):
        t = ds[bounds[i][0]]["task"].lower()
        return "case" in t and "pick" in t and "place" not in t
    case_eps = [i for i in range(len(bounds)) if is_case_pick(i)] or list(range(len(bounds)))
    if args.episodes:
        ep_list = [int(x) for x in args.episodes.split(",")]
    elif args.all_episodes:
        ep_list = case_eps
    elif args.episode is not None:
        ep_list = [args.episode]
    else:
        ep_list = [case_eps[0]]
    print(f"  episodes: {ep_list}")

    if args.swap in ("firstcontact", "fcscale"):
        # wrench at the FIRST force rise: fz > thresh for 2 consecutive frames (the measured
        # expert stop trigger — expert dz>=0 at median 0 frames from this event; seal fires
        # ~33 frames later). Frames t..t+2, strictly pre-seal, pooled over ALL episodes.
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
                cand = np.flatnonzero((fz[:-1] > args.fc_fz_thresh)
                                      & (fz[1:] > args.fc_fz_thresh))
                cand = cand[(cand > 10) & (cand < ts)]
                if not len(cand):
                    continue
                w = s[cand[0]:min(cand[0] + 3, ts)]
                w = w[w[:, SEAL_IDX] <= 0.5]
                pool.append(w[:, W_LO:W_HI])
        pool = np.concatenate(pool) if pool else np.empty((0, W_HI - W_LO))
        if not len(pool):
            print("  no first-contact frames found — cannot probe firstcontact.")
            return
        swap_wrench = np.median(pool, axis=0)
        if args.swap == "fcscale":   # same direction, force xyz rescaled to --fc-mag
            f3 = swap_wrench[:3]
            swap_wrench = swap_wrench.copy()
            swap_wrench[:3] = f3 / np.linalg.norm(f3) * args.fc_mag
        print(f"  swap wrench (median over {len(pool)} FIRST-CONTACT frames, fz>"
              f"{args.fc_fz_thresh}N trigger, all episodes): {np.round(swap_wrench, 2)}  "
              f"|F|={np.linalg.norm(swap_wrench[:3]):.1f}N  seal stays 0  [--swap {args.swap}]")
    elif args.swap == "onset":
        # pre-seal press wrench: the 3 frames immediately BEFORE the first sealed frame of
        # each episode (seal still 0) — the last signal available at the moment the policy
        # must stop. Pooled over ALL episodes. (An |F|-jump heuristic was tried first and is
        # contaminated: start-of-episode motion noise fires it 130+ frames before contact.)
        import glob
        import pandas as pd
        pool = []
        for p in sorted(glob.glob(str(args.dataset_root / "data" / "*" / "*.parquet"))):
            df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
            stt = np.stack(df["observation.state"].to_numpy())
            for e in df["episode_index"].unique():
                s = stt[(df["episode_index"] == e).to_numpy()]
                seals = np.flatnonzero(s[:, SEAL_IDX] > 0.5)
                if not len(seals) or seals[0] < 1:
                    continue
                w = s[max(0, seals[0] - 3):seals[0]]
                w = w[w[:, SEAL_IDX] <= 0.5]          # strictly pre-seal
                pool.append(w[:, W_LO:W_HI])
        pool = np.concatenate(pool) if pool else np.empty((0, W_HI - W_LO))
        if not len(pool):
            print("  no pre-seal frames found across the dataset — cannot probe onset.")
            return
        swap_wrench = np.median(pool, axis=0)
        print(f"  swap wrench (median over {len(pool)} PRE-SEAL press frames, all episodes): "
              f"{np.round(swap_wrench, 2)}  |F|={np.linalg.norm(swap_wrench[:3]):.1f}N  "
              f"seal stays 0  [--swap onset]")
    elif args.swap == "bias":
        swap_wrench = None
        bias_vec = [float(x) for x in args.bias.split(",")]
        assert len(bias_vec) == 3, "--bias needs fx,fy,fz"
        print(f"  bias mode: force += {bias_vec} N per frame (seal untouched)")
    else:
        swap_wrench = None    # both/wrench/seal: harvested per episode (sealed median) below

    def make_swap(raw):
        s = raw.clone()
        if args.swap == "bias":
            s[F_LO:F_HI] = s[F_LO:F_HI] + torch.tensor(bias_vec, dtype=s.dtype)
            return s
        if args.swap in ("both", "wrench", "onset", "firstcontact", "fcscale"):
            s[W_LO:W_HI] = torch.tensor(swap_wrench, dtype=s.dtype)
        if args.swap in ("both", "seal"):
            s[SEAL_IDX] = 1.0
        return s

    task = None                # set per episode; predict() reads it late-bound

    def predict(frame, raw_state, c):
        # fixed flow-matching noise across cells (see probe_film_authority.py)
        _FORCE["c"] = c
        policy.reset()
        torch.manual_seed(0)
        obs = {"observation.state": raw_state.unsqueeze(0), "task": task}
        for k in frame:
            if k.startswith("observation.images."):
                obs[k] = frame[k].unsqueeze(0)
        with torch.inference_mode():
            a = policy.select_action(pre(obs))
        _FORCE["c"] = None
        return post(a).squeeze(0).cpu().numpy()

    # ── per-frame cells, per episode ─────────────────────────────────────────
    rows = []                  # (i, ee_z, |F|, expert, oo, ss, so, os, ttc, ep)
    ep_contact_z = {}
    printed_chat = False
    for ep in ep_list:
        lo, hi = bounds[ep]
        task = ds[lo]["task"]
        states = np.stack([ds[i]["observation.state"].numpy() for i in range(lo, hi)])
        fmag = np.linalg.norm(states[:, F_LO:F_HI], axis=1)
        sealed = states[:, SEAL_IDX] > 0.5
        if args.swap in ("both", "wrench", "seal"):
            if not sealed.any():
                print(f"  ep{ep}: no sealed frames -> skipped")
                continue
            swap_wrench = np.median(states[sealed][:, W_LO:W_HI], axis=0)
        # first contact for the time-to-contact split: first |F| jump (>=2N/frame) within
        # 60 frames of the seal (an fz>3 trigger misfires on early-descent motion noise);
        # fallback: the train-median 33-frame gap before seal.
        ts_ep = int(np.flatnonzero(sealed)[0]) if sealed.any() else len(states)
        cand = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= 2)
        cand = cand[(cand >= max(0, ts_ep - 60)) & (cand < ts_ep)]
        tf = int(cand[0]) if len(cand) else (ts_ep - 33 if ts_ep > 33 else None)
        ep_contact_z[ep] = states[tf, EE_Z] if tf is not None else np.nan
        n_ep = 0
        for j, i in enumerate(range(lo, hi)):
            if args.pre_contact:           # only the window right before first contact
                if tf is None or not (0 < tf - j <= args.pre_contact):
                    continue
            elif fmag[j] >= args.descend_n:  # only pre-contact / descent frames
                continue
            f = ds[i]
            own = f["observation.state"]
            swp = make_swap(own)
            if not naive and not printed_chat:
                print(f"  c-hat: own(example)={c_from_raw(policy.model, own).numpy().round(3)}"
                      f"  swap={c_from_raw(policy.model, swp).numpy().round(3)}"
                      f"  (channels {cond})")
                printed_chat = True
            ttc_i = (tf - j) if tf is not None else np.nan
            d_oo = predict(f, own, None)[DZ]                   # baseline
            if naive:
                d_ss = predict(f, swp, None)[DZ]               # total (raw only exists)
                rows.append((i, states[j, EE_Z], fmag[j], f["action"].numpy()[DZ],
                             d_oo, d_ss, np.nan, np.nan, ttc_i, ep))
            else:
                c_own = c_from_raw(policy.model, own)
                c_swp = c_from_raw(policy.model, swp)
                d_ss = predict(f, swp, c_swp)[DZ]              # total
                d_so = predict(f, swp, c_own)[DZ]              # raw path only
                d_os = predict(f, own, c_swp)[DZ]              # FiLM path only
                rows.append((i, states[j, EE_Z], fmag[j], f["action"].numpy()[DZ],
                             d_oo, d_ss, d_so, d_os, ttc_i, ep))
            n_ep = n_ep + 1
        print(f"  ep{ep}: \"{task}\"  frames {lo}..{hi}  descent={n_ep}  "
              f"first-contact@+{tf}  seal@+{ts_ep}")

    if not rows:
        print("  no descent frames (raise --descend-n?).")
        return

    hdr = (f"  {'frame':>6} {'ee_z':>7} {'|F|N':>6} | {'expert':>7} {'dz(own)':>8} "
           f"{'dz(swap)':>8} {'dzRAW':>8} {'dzFiLM':>8} | {'dTot':>7} {'dRaw':>7} {'dFiLM':>7}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    step = max(1, len(rows) // 25)
    for r in rows[::step]:
        i, z, fm, e, oo, ss, so, os_ = r[:8]
        print(f"  {i:>6} {z:>7.3f} {fm:>6.1f} | {e*1000:>6.1f}m {oo*1000:>7.1f}m "
              f"{ss*1000:>7.1f}m {so*1000:>7.1f}m {os_*1000:>7.1f}m | "
              f"{(ss-oo)*1000:>6.1f}m {(so-oo)*1000:>6.1f}m {(os_-oo)*1000:>6.1f}m")

    arr = np.array([r[4:8] for r in rows])                     # oo, ss, so, os
    oo, ss, so, os_ = arr.T
    d_tot, d_raw, d_flm = ss - oo, so - oo, os_ - oo
    inter = d_tot - d_raw - d_flm

    def band(name, m):
        if not m.any():
            return
        line = (f"    {name}: n={m.sum():>3}  dz(own)={oo[m].mean()*1000:+6.2f}mm  "
                f"dTotal={d_tot[m].mean()*1000:+6.2f}mm")
        if not naive:
            line += (f"  dRaw={d_raw[m].mean()*1000:+6.2f}mm  "
                     f"dFiLM={d_flm[m].mean()*1000:+6.2f}mm  "
                     f"interact={inter[m].mean()*1000:+6.2f}mm")
        print(line)

    print("  " + "-" * (len(hdr) - 2))
    print(f"\n  decomposition over {len(rows)} descent frames "
          f"(d* = dz(cell) - dz(own,own)):")
    band("ALL frames                  ", np.ones(len(rows), bool))
    desc = oo < -0.003                                          # committed descent
    band("approach/hover dz(own)>=-3mm", ~desc)
    band("COMMITTED desc dz(own)< -3mm", desc)

    # time-to-contact breakdown: does the swap have authority far BEFORE any force rise
    # (early descent, fz still at hover baseline), or only near the surface (image-gated)?
    ttc = np.array([r[8] for r in rows], dtype=float)          # frames until contact (15fps)
    if np.isfinite(ttc).any():
        print(f"\n  time-to-contact split (pooled over {len(ep_list)} episode(s), "
              f"per-episode first-contact detection):")
        valid = np.isfinite(ttc)
        for name, m in ((">=4s before contact (early) ", valid & (ttc >= 60)),
                        ("2-4s before contact         ", valid & (ttc >= 30) & (ttc < 60)),
                        ("<2s before contact (late)   ", valid & (ttc >= 0) & (ttc < 30)),
                        ("after contact (press phase) ", valid & (ttc < 0))):
            band(name, m)
    else:
        print("\n  (no first-contact events -> no time-to-contact split)")

    # per-episode split — ABSOLUTE dz per contact height (z differs by layer)
    if len(ep_list) > 1:
        eps_arr = np.array([r[9] for r in rows])
        print("\n  per-episode split (ABSOLUTE dz, mm/step):")
        for e in ep_list:
            m = eps_arr == e
            if not m.any():
                continue
            line = (f"    ep{e} (contact z={ep_contact_z.get(e, float('nan')):.3f}): "
                    f"n={m.sum():>3}  dz(own)={oo[m].mean()*1000:+6.2f}  "
                    f"dz(swap)={ss[m].mean()*1000:+6.2f}  d={d_tot[m].mean()*1000:+6.2f}")
            if not naive:
                line += f"  (FiLM {d_flm[m].mean()*1000:+6.2f})"
            print(line)

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n── verdict ─────────────────────────────────────────────────────────")
    if not desc.any():
        print("  INCONCLUSIVE: no committed-descent frames in this episode.")
        return
    base = oo[desc].mean()
    rt = d_tot[desc].mean() / abs(base)
    if naive:
        print(f"  raw-state authority (naive): swapping wrench+seal to contact values cancels "
              f"{rt*100:.0f}% of the committed descent (dz {base*1000:+.1f} -> "
              f"{ss[desc].mean()*1000:+.1f}mm).")
        return
    rr = d_raw[desc].mean() / abs(base)
    rf = d_flm[desc].mean() / abs(base)
    print(f"  committed descent, cancellation of dz(own)={base*1000:+.1f}mm:")
    print(f"    TOTAL (state+c swap) : {rt*100:+5.0f}%")
    print(f"    RAW path only        : {rr*100:+5.0f}%   (state swap, c pinned to own)")
    print(f"    FiLM path only       : {rf*100:+5.0f}%   (c swap, state kept own)")
    if rr > max(rf, 0.0) + 0.15:
        print("  -> the RAW state path carries the contact response; FiLM is redundant here.")
    elif rf > max(rr, 0.0) + 0.15:
        print("  -> the FiLM path carries the contact response; the raw path adds little.")
    else:
        print("  -> both paths contribute comparably (or neither does — check TOTAL).")


if __name__ == "__main__":
    main()
