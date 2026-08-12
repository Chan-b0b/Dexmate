#!/usr/bin/env python3
"""Train a FiLM contact-conditioned PI0 (π0): apply film_contact_pi0 patches, then run
lerobot's standard training loop unchanged. Same env interface as train_film.py
(FILM_VARIANT/COND/MASK_FORCE/F0/TAU/FZ_TAU/FZ_OFF/FMAG_*/DFMAG_TAU, FILM_DATASET_ROOT).

FILM_INJECT here is 'state' or 'action' — NOT prefix/suffix. In π0 the state token and the
action tokens are both in embed_suffix, so the question this run answers is "does which
TOKEN you condition matter", the token-level half of SmolVLA's prefix>suffix result.
π0 is used instead of π0.5 because π0.5 has no state token at all (state is discretized
into the text prompt), and because π0's STATE is MEAN_STD-normalized like SmolVLA's, so the
channel calibration carries over unchanged.

  FILM_INJECT=state FILM_COND=contact,fz,seal python train_film_pi0.py \
      --policy.path=lerobot/pi0_base --dataset.repo_id=... ...
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pi05  # noqa: E402,F401  registers the relative_actions_processor shim
                   # (pi0_base's preprocessor needs it too; pi0_new_line_processor is native)
import film_contact  # noqa: E402
import film_contact_pi0 as fc0  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in
              os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
_inject = os.environ.get("FILM_INJECT", "state")
if _inject not in fc0.INJECTS:
    sys.exit(f"[film-pi0] FILM_INJECT must be one of {fc0.INJECTS} (π0 has no prefix/suffix "
             f"split for FiLM: both points live in embed_suffix); got {_inject!r}")
_root = os.environ.get(
    "FILM_DATASET_ROOT",
    str(Path(__file__).resolve().parent / "datasets/lges_case_pick_0729"))

# π0 normalizes STATE with MEAN_STD, so film_contact's own stat loaders apply verbatim.
_wm, _ws = film_contact.load_wrench_stats(_root)
_sm, _ss = film_contact.load_seal_stats(_root)
_dm, _dsd = film_contact.load_dfmag_stats(_root)
fc0.apply(_variant, _wm, _ws, seal_mean=_sm, seal_std=_ss, cond=_cond,
          inject=_inject, mask_force=_mask_force,
          contact_F0=float(os.environ.get("FILM_F0", "6")),
          contact_tau=float(os.environ.get("FILM_TAU", "4")),
          fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
          fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
          fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
          fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
          dfmag_mean=_dm, dfmag_std=_dsd,
          dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")))
print(f"[film-pi0] patched PI0: variant={_variant} cond={_cond} inject={_inject} "
      f"mask_force={_mask_force} stats={_root}", file=sys.stderr)

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()
