#!/usr/bin/env python3
"""Train a FiLM contact-conditioned GR00T N1.5 (state-token injection at the action
head's state_encoder): apply the film_contact_groot patches, then run lerobot's standard
training loop unchanged. Same env interface as train_film.py / train_film_pi0.py
(FILM_VARIANT/COND/MASK_FORCE/F0/TAU/FZ_TAU/FZ_OFF/FMAG_*/DFMAG_TAU, FILM_DATASET_ROOT).

GR00T min-max normalizes state (not MEAN_STD), so the stats come from
film_contact_groot.load_state_minmax, not film_contact's mean/std loaders.
FILM_INJECT: 'state' (state_encoder token, default) | 'layers' (per-DiT-block FF gate).

  FILM_COND=contact,fz,seal python train_film_groot.py --policy.type=groot ...
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_groot  # noqa: E402,F401  installs the sdpa fallback (no flash-attn here)
import film_contact_groot as fcg  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in
              os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
_inject = os.environ.get("FILM_INJECT", "state")
_root = os.environ.get(
    "FILM_DATASET_ROOT",
    str(Path(__file__).resolve().parent / "datasets/lges_case_pick_0729"))

_mn, _mx = fcg.load_state_minmax(_root)
fcg.apply(_variant, _mn, _mx, cond=_cond, mask_force=_mask_force, inject=_inject,
          contact_F0=float(os.environ.get("FILM_F0", "6")),
          contact_tau=float(os.environ.get("FILM_TAU", "4")),
          fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
          fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
          fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
          fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
          dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")))
print(f"[film-groot] patched GrootPolicy: variant={_variant} cond={_cond} "
      f"inject={_inject} mask_force={_mask_force} stats={_root}",
      file=sys.stderr)

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()
