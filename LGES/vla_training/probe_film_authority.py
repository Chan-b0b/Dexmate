#!/usr/bin/env python3
"""Does FiLM actually drive the action? — counterfactual c-hat probe on TRAINING data.

"FiLM working" = the predicted action is a FUNCTION of c-hat. The only way to prove
it is a counterfactual: take a real (in-distribution) frame, hold image+state FIXED,
and swap ONLY c-hat. If the action (esp. dz, the descend channel) doesn't move, FiLM
has no authority over the policy — independent of whether the c-hat pipeline is correct.

The training set is the clean place to test this: real frames (no OOD), single-forward
prediction (no chunk-replay latency), and we can FORCE c-hat to any value.

Two tiers:
  Tier 0  static, no data : scale(c=0) vs scale(c=1) inside ContactFiLM. If ~0, the
                            module ignores its own input (the `scale.2 |max|` symptom).
  Tier 1  the real test   : per descent frame, predict with c FORCED to 0 vs 1.
                            Authority => dz(c=0) < 0 (descend), dz(c=1) ~ 0 (stop).

Structural FiLM config (cond/inject/mask_force) MUST match the checkpoint, via the same
env vars used at train/deploy time:
  FILM_COND=contact FILM_MASK_FORCE=0 FILM_INJECT=suffix \
    /home/dexmate/vla_venv/bin/python probe_film_authority.py \
      --checkpoint outputs/film_v2_contact_suffix/checkpoints/last
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

VLA_DIR = Path(__file__).resolve().parent
EE_Z, F_LO, F_HI = 2, 9, 12   # state idx: ee_z; force xyz = wrench[:3]
DZ = 2                         # action idx of dz

import film_contact  # noqa: E402

# ── forced-c override: monkeypatch the module-level condition fn ─────────────
# new_embed_prefix() resolves _condition_from_state by module global at call time,
# so reassigning it here is picked up by the patched model. None => real c-hat.
_FORCE = {"c": None}
_orig_cond = film_contact._condition_from_state


def _forced_cond(self, state):
    if _FORCE["c"] is None:
        return _orig_cond(self, state)
    B = state.shape[0]
    return torch.full((B, len(self._film_cond)), float(_FORCE["c"]),
                      dtype=torch.float32, device=state.device)


film_contact._condition_from_state = _forced_cond


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_DIR / "outputs/film_v3_contactfzseal_nomask/checkpoints/last")
    ap.add_argument("--dataset-root", type=Path, default=VLA_DIR / "datasets/lges_suction")
    ap.add_argument("--repo-id", default="local/lges_suction")
    ap.add_argument("--episode", type=int, default=None,
                    help="episode index (default: first case_pick episode)")
    ap.add_argument("--contact-n", type=float, default=14.5,
                    help="raw |F| (N) below which a frame counts as pre-contact/descent")
    args = ap.parse_args()

    cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact").split(",") if c.strip())
    mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
    inject = os.environ.get("FILM_INJECT", "suffix")
    print(f"[probe] FiLM cond={cond} inject={inject} mask_force={mask_force}  ckpt={args.checkpoint}")

    wm, ws = film_contact.load_wrench_stats(args.dataset_root)
    sm, ss = film_contact.load_seal_stats(args.dataset_root)
    film_contact.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                       mask_force=mask_force, inject=inject)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    model_dir = args.checkpoint / "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.eval()
    policy.config.n_action_steps = 1          # fresh single-forward prediction per frame
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})

    # ── Tier 0: does the FiLM module's output even move with c? ──────────────
    film = policy.model.contact_film
    p = next(film.parameters())
    cd = len(cond)
    c0 = torch.zeros(1, cd, device=p.device, dtype=p.dtype)
    c1 = torch.ones(1, cd, device=p.device, dtype=p.dtype)
    with torch.no_grad():
        dg = (film.scale(c1).float() - film.scale(c0).float()).abs()
        db = (film.shift(c1).float() - film.shift(c0).float()).abs()
    print("\n── Tier 0  (static, no data) ───────────────────────────────────────")
    print(f"  |Δγ| max={dg.max():.4f} mean={dg.mean():.4f}   "
          f"|Δβ| max={db.max():.4f} mean={db.mean():.4f}   (c: 0 -> 1)")
    print("  γ is a multiplicative gain on the action-token embedding; ~0 => FiLM"
          " ignores its own input.")

    # ── load one case_pick episode ──────────────────────────────────────────
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    bounds = [(e["dataset_from_index"], e["dataset_to_index"]) for e in ds.meta.episodes]
    ep = args.episode
    if ep is None:
        for i, (lo, _) in enumerate(bounds):
            t = ds[lo]["task"].lower()
            if "case" in t and "pick" in t and "place" not in t:
                ep = i
                break
        if ep is None:
            ep = 0
    lo, hi = bounds[ep]
    task = ds[lo]["task"]
    print(f"\n── Tier 1  (counterfactual on training data) ───────────────────────")
    print(f"  episode {ep}: \"{task}\"  frames {lo}..{hi}")

    def predict_with_c(frame, c):
        # FIX the flow-matching noise across the c=0/c=1/real calls: sample_actions draws
        # fresh N(0,I) noise per call, which would otherwise dominate Δdz. eval() => no
        # other stochasticity, so seeding right before select_action makes the noise
        # identical and Δdz reflects ONLY c-hat.
        _FORCE["c"] = c
        policy.reset()
        torch.manual_seed(0)
        obs = {"observation.state": frame["observation.state"].unsqueeze(0), "task": task}
        for k in frame:
            if k.startswith("observation.images."):
                obs[k] = frame[k].unsqueeze(0)
        with torch.inference_mode():
            a = policy.select_action(pre(obs))
        return post(a).squeeze(0).cpu().numpy()

    rows = []
    for i in range(lo, hi):
        f = ds[i]
        st = f["observation.state"].numpy()
        fmag = float(np.linalg.norm(st[F_LO:F_HI]))
        if fmag >= args.contact_n:          # only pre-contact / descent frames
            continue
        a0 = predict_with_c(f, 0.0)
        a1 = predict_with_c(f, 1.0)
        ar = predict_with_c(f, None)        # real c-hat (what deploy would see here)
        exp = f["action"].numpy()
        rows.append((i, st[EE_Z], fmag, exp[DZ], a0[DZ], a1[DZ], ar[DZ]))

    if not rows:
        print("  no pre-contact frames found (raise --contact-n?).")
        return

    print(f"\n  {'frame':>6} {'ee_z':>7} {'|F|N':>6} | {'expert_dz':>9} "
          f"{'dz(c=0)':>8} {'dz(c=1)':>8} {'Δdz':>8} {'dz(real)':>9}")
    print("  " + "-" * 74)
    step = max(1, len(rows) // 25)          # print up to ~25 evenly-spaced rows
    for r in rows[::step]:
        i, z, fm, e, d0, d1, dr = r
        print(f"  {i:>6} {z:>7.3f} {fm:>6.1f} | {e*1000:>8.1f}m {d0*1000:>7.1f}m "
              f"{d1*1000:>7.1f}m {(d1-d0)*1000:>7.1f}m {dr*1000:>8.1f}m")

    arr = np.array([(r[4], r[5]) for r in rows])   # dz(c=0), dz(c=1) in m
    d0, d1 = arr[:, 0], arr[:, 1]
    dd = d1 - d0
    print("  " + "-" * 74)
    print(f"  mean over {len(rows)} pre-contact frames:  dz(c=0)={d0.mean()*1000:+.2f}mm  "
          f"dz(c=1)={d1.mean()*1000:+.2f}mm  meanΔdz={dd.mean()*1000:+.2f}mm  "
          f"max|Δdz|={np.abs(dd).max()*1000:.2f}mm")

    # Phase split: the descend->stop transition can only act where the model is ALREADY
    # committed to descending. Authority there is what matters; authority only in the
    # shallow hover phase does not produce stop-on-contact.
    COMMIT = -0.003  # dz(c=0) < -3mm/step => committed descent
    desc = d0 < COMMIT
    appr = ~desc
    print("\n  phase split (the transition must fire in the COMMITTED-descent band):")
    for name, mask in (("approach/hover  dz(c=0)>=-3mm", appr),
                       ("COMMITTED desc  dz(c=0)< -3mm", desc)):
        if mask.any():
            print(f"    {name}: n={mask.sum():>3}  dz(c=0)={d0[mask].mean()*1000:+6.2f}mm  "
                  f"dz(c=1)={d1[mask].mean()*1000:+6.2f}mm  Δdz={dd[mask].mean()*1000:+6.2f}mm")

    # ── verdict ─────────────────────────────────────────────────────────────
    print("\n── verdict ─────────────────────────────────────────────────────────")
    if dg.max() < 1e-3:
        print("  Tier 0 FAIL: FiLM module output barely moves with c (|Δγ|~0) -> the"
              " conditioning layer never learned to use its input. FiLM is inert.")
    elif np.abs(dd).max() * 1000 < 1.0:
        print("  Tier 1 FAIL: module moves with c but the ACTION does not (max|Δdz|<1mm)"
              " -> expert layers/out_proj wash out the modulation. No authority.")
    elif not desc.any():
        print("  INCONCLUSIVE: no committed-descent frames in this episode.")
    else:
        d0c, d1c, ddc = d0[desc].mean(), d1[desc].mean(), dd[desc].mean()
        ratio = ddc / abs(d0c) if d0c else 0.0   # fraction of the descent c=1 cancels
        if ddc > 0 and ratio > 0.5:
            print(f"  PASS: in committed descent, c=1 cancels {ratio*100:.0f}% of the descent "
                  f"command (dz {d0c*1000:+.1f}->{d1c*1000:+.1f}mm). Strong stop-on-contact authority.")
        elif ddc > 0:
            print(f"  WEAK: c=1 only cancels {ratio*100:.0f}% of the committed descent "
                  f"(dz {d0c*1000:+.1f}->{d1c*1000:+.1f}mm). Right sign, too weak to STOP -> the depth/"
                  "image cue still dominates the descend command. Likely the under-reach cause.")
        else:
            print(f"  NO/WRONG-SIGN AUTHORITY in committed descent: c=1 dz {d0c*1000:+.1f}->"
                  f"{d1c*1000:+.1f}mm (Δ{ddc*1000:+.1f}mm). FiLM does not gate the stop where it"
                  " must; the depth shortcut owns the descend channel. (May still nudge the hover phase.)")


if __name__ == "__main__":
    main()


#FILM_COND=contact,fz,seal FILM_MASK_FORCE=0 FILM_INJECT=suffix /home/dexmate/vla_venv/bin/python probe_film_authority.py --checkpoint outputs/film_v3_contactfzseal_nomask/checkpoints/last