#!/usr/bin/env python3
"""Train a FiLM contact-conditioned SmolVLA (V1/V2): apply the film_contact patches,
then run lerobot's standard training loop unchanged.

  FILM_VARIANT=v2 python train_film.py --policy.path=<init ckpt> --dataset.root=... ...

Variant + dataset (for wrench-normalization stats) come from env:
  FILM_VARIANT       v2 (method) | v1 (decorrelated control)   [default v2]
  FILM_DATASET_ROOT  lerobot dataset dir with meta/stats.json  [default datasets/lges_suction]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_contact  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
_inject = os.environ.get("FILM_INJECT", "suffix")
_f0 = float(os.environ.get("FILM_F0", "6"))
_tau = float(os.environ.get("FILM_TAU", "4"))       # contact-DROP scale (graded)
_fz_tau = float(os.environ.get("FILM_FZ_TAU", "5"))  # continuous fz scale (fz_raw/30)
_fz_off = float(os.environ.get("FILM_FZ_OFF", "2.6"))  # fz centering offset (dataset fz median)
_dfmag_tau = float(os.environ.get("FILM_DFMAG_TAU", "5"))  # d|F|/dt scale (N/frame / 5)
_root = os.environ.get(
    "FILM_DATASET_ROOT", str(Path(__file__).resolve().parent / "datasets/lges_suction"))
_wm, _ws = film_contact.load_wrench_stats(_root)
_sm, _ss = film_contact.load_seal_stats(_root)
_dm, _dsd = film_contact.load_dfmag_stats(_root)   # (None, None) unless a *_dF dataset
film_contact.apply(_variant, _wm, _ws, seal_mean=_sm, seal_std=_ss, cond=_cond,
                   contact_F0=_f0, contact_tau=_tau, fz_tau=_fz_tau,
                   mask_force=_mask_force, inject=_inject,
                   dfmag_mean=_dm, dfmag_std=_dsd, dfmag_tau=_dfmag_tau, fz_off=_fz_off)
print(f"[film] patched VLAFlowMatching: variant={_variant} cond={_cond} inject={_inject} "
      f"mask_force={_mask_force} contact=clip((|F|-{_f0:.0f})/{_tau:.0f}) fz=(fz-{_fz_off:g})/{_fz_tau:.0f} "
      f"dfmag={'d|F|/%g' % _dfmag_tau if _dm is not None else 'n/a'} "
      f"wrench_mean={[round(x,2) for x in _wm.tolist()]}", file=sys.stderr)

# ── contact-transition oversampling (FILM_OVERSAMPLE_BOOST > 1 enables) ─────
# The contact dip is 1-2 frames per episode (~1% of data), so BC barely sees it
# and binds the stop to later post-seal signals instead (probe decomposition,
# 2026-07-17). Boost the sampling weight of frames within WINDOW of a sharp
# |F| drop (dfmag <= -THRESH N/frame) so the transition gets gradient exposure.
_os_boost = float(os.environ.get("FILM_OVERSAMPLE_BOOST", "0"))
if _os_boost > 1:
    import numpy as np
    import pandas as pd
    import torch

    _os_thresh = float(os.environ.get("FILM_OVERSAMPLE_THRESH", "2"))   # N/frame drop
    _os_window = int(os.environ.get("FILM_OVERSAMPLE_WINDOW", "5"))    # +- frames

    def _transition_weights(root):
        """Per-frame sampling weight over the dataset (global `index` order):
        `_os_boost` within +-`_os_window` frames of a |F| JUMP >= `_os_thresh` in either
        direction (old robot: contact = drop; new robot: contact = rise), else 1."""
        dfs = [pd.read_parquet(p, columns=["observation.state", "episode_index", "index"])
               for p in sorted((Path(root) / "data").rglob("*.parquet"))]
        df = pd.concat(dfs).sort_values("index").reset_index(drop=True)
        assert (df["index"].to_numpy() == np.arange(len(df))).all(), "non-contiguous index"
        st = np.stack(df["observation.state"].to_numpy())
        w = np.ones(len(df))
        for ep in df["episode_index"].unique():
            m = np.flatnonzero((df["episode_index"] == ep).to_numpy())
            fmag = np.linalg.norm(st[m, 9:12], axis=1)
            dips = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= _os_thresh)
            for d in dips:
                w[m[max(0, d - _os_window):d + _os_window + 1]] = _os_boost
        return w

    _orig_DataLoader = torch.utils.data.DataLoader

    class _WeightedDataLoader(_orig_DataLoader):
        # a subclass (not a wrapper fn) so accelerate's isinstance(..., DataLoader) passes
        def __init__(self, dataset, *a, **kw):
            if kw.get("shuffle") and kw.get("sampler") is None and hasattr(dataset, "meta"):
                w = _transition_weights(_root)
                assert len(w) == len(dataset), f"weights {len(w)} != dataset {len(dataset)}"
                kw["sampler"] = torch.utils.data.WeightedRandomSampler(
                    torch.as_tensor(w, dtype=torch.double), num_samples=len(dataset),
                    replacement=True)
                kw["shuffle"] = False
                n = int((w > 1).sum())
                print(f"[film-os] oversampling {n}/{len(w)} transition frames x{_os_boost:g} "
                      f"(thresh={_os_thresh:g}N/frame window=+-{_os_window}) -> "
                      f"{n * _os_boost / (n * _os_boost + len(w) - n):.1%} of samples",
                      file=sys.stderr)
            super().__init__(dataset, *a, **kw)

    torch.utils.data.DataLoader = _WeightedDataLoader

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()  # @parser.wrap() parses sys.argv — same CLI as lerobot-train


#FILM_VARIANT=v2 RUN_NAME=smoke ./train_film.sh --steps=4 --save_freq=2

# # 둘 다 (현재 기본)
# FILM_VARIANT=v2 RUN_NAME=film_v2 ./train_film.sh
# # contact만
# FILM_COND=contact FILM_VARIANT=v2 RUN_NAME=film_v2_contact ./train_film.sh
# # seal만
# FILM_COND=seal    FILM_VARIANT=v2 RUN_NAME=film_v2_seal    ./train_film.sh

# # 평가 — 반드시 학습과 같은 FILM_COND (+ FILM_MASK_FORCE)
# FILM_COND=contact,seal FILM_MASK_FORCE=0 python run_policy.py --go --task case_pick \
#   --goto-start <take> --checkpoint outputs/film_v2/checkpoints/last --film ... --loop