#!/usr/bin/env python3
"""Train a FiLM contact-conditioned ACT (encoder/prefix-style state-token injection):
apply the film_contact_act patches, then run lerobot's standard training loop unchanged.

  FILM_VARIANT=v2 python train_film_act.py --policy.type=act --dataset.root=... ...

Env vars (same names as train_film.py / train_film_pi05.py):
  FILM_VARIANT       v2 (method) | v1 (decorrelated control)          [default v2]
  FILM_COND          comma-separated subset of contact,fz,fmag,seal,dfmag [default contact,fz,seal]
  FILM_MASK_FORCE    1/0 -- zero conditioned dims out of observation.state [default 1]
  FILM_F0/FILM_TAU   contact channel threshold/scale                  [default 6/4]
  FILM_FZ_TAU/FILM_FZ_OFF    continuous fz channel scale/offset       [default 5/2.6]
  FILM_FMAG_OFF/FILM_FMAG_TAU  continuous |F| channel offset/scale    [default 5.1/5]
  FILM_DFMAG_TAU     d|F|/dt channel scale (needs a *_dF dataset)     [default 5]
  FILM_DATASET_ROOT  lerobot dataset dir with meta/stats.json  [default datasets/lges_case_pick_0729]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_contact  # noqa: E402
import film_contact_act as fca  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
_f0 = float(os.environ.get("FILM_F0", "6"))
_tau = float(os.environ.get("FILM_TAU", "4"))
_fz_tau = float(os.environ.get("FILM_FZ_TAU", "5"))
_fz_off = float(os.environ.get("FILM_FZ_OFF", "2.6"))
_fmag_off = float(os.environ.get("FILM_FMAG_OFF", "5.1"))
_fmag_tau = float(os.environ.get("FILM_FMAG_TAU", "5"))
_dfmag_tau = float(os.environ.get("FILM_DFMAG_TAU", "5"))
_root = os.environ.get(
    "FILM_DATASET_ROOT", str(Path(__file__).resolve().parent / "datasets/lges_case_pick_0729"))
_wm, _ws = film_contact.load_wrench_stats(_root)
_sm, _ss = film_contact.load_seal_stats(_root)
_dm, _dsd = film_contact.load_dfmag_stats(_root)  # (None, None) unless a *_dF dataset

fca.apply(_variant, _wm, _ws, seal_mean=_sm, seal_std=_ss, cond=_cond,
          contact_F0=_f0, contact_tau=_tau, fz_tau=_fz_tau, fz_off=_fz_off,
          fmag_off=_fmag_off, fmag_tau=_fmag_tau, dfmag_mean=_dm, dfmag_std=_dsd,
          dfmag_tau=_dfmag_tau, mask_force=_mask_force)
print(f"[film-act] patched ACT: variant={_variant} cond={_cond} inject=encoder(prefix-style) "
      f"mask_force={_mask_force} contact=clip((|F|-{_f0:.0f})/{_tau:.0f}) "
      f"fz=(fz-{_fz_off:g})/{_fz_tau:.0f} fmag=(|F|-{_fmag_off:g})/{_fmag_tau:g}", file=sys.stderr)

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()
