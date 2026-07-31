#!/usr/bin/env python3
"""Pick the best checkpoint of a run by VALIDATION loss (policy.forward flow-matching
loss on a held-out *_val dataset), create checkpoints/best -> <step>, and print a table.
Retention policy (2026-07-29): keep `last` + `best`, prune the rest (use --prune).

FiLM checkpoints: set the SAME structural env as training (FILM_COND/INJECT/MASK_FORCE);
calibration values default to the current-generation code defaults (F0=6 tau=4 fz_tau=5
fz_off=2.6) and can be overridden by env like train_film.py.

  FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
    python select_best_ckpt.py --run outputs/<run> \
      --val-root datasets/<ds>_val --repo-id Chanho-Lee/<ds>_val [--prune]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

VLA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VLA_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="outputs/<run> dir")
    ap.add_argument("--val-root", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--prune", action="store_true", help="delete checkpoints other than last+best")
    args = ap.parse_args()

    ckpts = sorted(d for d in (args.run / "checkpoints").glob("0*") if d.is_dir())
    if not ckpts:
        sys.exit(f"no checkpoints under {args.run}")

    # FiLM: patch iff the checkpoint carries contact_film weights
    from safetensors import safe_open
    with safe_open(ckpts[-1] / "pretrained_model" / "model.safetensors", framework="pt") as f:
        is_film = any(k.startswith("model.contact_film") for k in f.keys())
    if is_film:
        import film_contact
        cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
        wm, ws = film_contact.load_wrench_stats(args.val_root)
        sm, ss = film_contact.load_seal_stats(args.val_root)
        dm, dsd = film_contact.load_dfmag_stats(args.val_root)
        film_contact.apply(
            "v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
            contact_F0=float(os.environ.get("FILM_F0", "6")),
            contact_tau=float(os.environ.get("FILM_TAU", "4")),
            fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
            fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
            fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
            fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
            dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")),
            mask_force=os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False"),
            inject=os.environ.get("FILM_INJECT", "prefix"))
        print(f"[best] FiLM patched: cond={cond}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    ptype = PreTrainedConfig.from_pretrained(ckpts[-1] / "pretrained_model").type
    PolicyCls = get_policy_class(ptype)
    print(f"[best] policy type: {ptype}")

    torch.manual_seed(0)
    ds = LeRobotDataset(args.repo_id, root=args.val_root,
                        delta_timestamps=None)
    results = {}
    for ck in ckpts:
        model_dir = ck / "pretrained_model"
        policy = PolicyCls.from_pretrained(model_dir)
        # ACT's forward needs the VAE encoder, which only runs in train mode
        # (eval-mode forward crashes on None latents). Grads stay off either way.
        policy.train() if ptype == "act" else policy.eval()
        pre, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=str(model_dir),
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})
        # chunked action targets come from the dataset config the policy expects
        ds_l = LeRobotDataset(args.repo_id, root=args.val_root,
                              delta_timestamps={"action": [i / ds.fps for i in range(policy.config.chunk_size)]})
        g = torch.Generator().manual_seed(0)
        loader = torch.utils.data.DataLoader(ds_l, batch_size=args.batch_size, shuffle=True,
                                             generator=g, num_workers=4, drop_last=True)
        losses = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= args.max_batches:
                    break
                batch = pre(batch)
                loss, _ = policy.forward(batch)
                losses.append(float(loss))
        results[ck.name] = sum(losses) / len(losses)
        print(f"[best] {ck.name}: val_loss={results[ck.name]:.5f}")
        del policy
        torch.cuda.empty_cache()

    best = min(results, key=results.get)
    (args.run / "checkpoints" / "best").unlink(missing_ok=True)
    (args.run / "checkpoints" / "best").symlink_to(best)
    json.dump(results, open(args.run / "checkpoints" / "val_losses.json", "w"), indent=1)
    print(f"[best] BEST={best} (val_loss={results[best]:.5f}); 'best' symlink created")

    if args.prune:
        keep = {best, Path(os.readlink(args.run / "checkpoints" / "last")).name
                if (args.run / "checkpoints" / "last").is_symlink() else "last"}
        for ck in ckpts:
            if ck.name not in keep:
                import shutil
                shutil.rmtree(ck)
                print(f"[best] pruned {ck.name}")


if __name__ == "__main__":
    main()
