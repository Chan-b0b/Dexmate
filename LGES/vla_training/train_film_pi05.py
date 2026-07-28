#!/usr/bin/env python3
"""Train a FiLM contact-conditioned PI05 (π0.5): apply film_contact_pi05 patches,
then run lerobot's standard training loop unchanged. Same env interface as
train_film.py (FILM_VARIANT/COND/MASK_FORCE/F0/TAU/FZ_TAU/FZ_OFF/FMAG_*/DFMAG_TAU,
FILM_DATASET_ROOT), except FILM_INJECT: π0.5 supports 'suffix' only (no state token).

  FILM_COND=contact,fz,seal python train_film_pi05.py --policy.path=lerobot/pi05_base ...
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_contact_pi05 as fcp  # noqa: E402

_variant = os.environ.get("FILM_VARIANT", "v2")
_mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
_cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
if os.environ.get("FILM_INJECT", "suffix") != "suffix":
    sys.exit("[film-pi05] π0.5 supports FILM_INJECT=suffix only (no state token for 'prefix')")
_root = os.environ.get(
    "FILM_DATASET_ROOT", str(Path(__file__).resolve().parent / "datasets/lges_case_pick_0721_0727"))
_q01, _q99 = fcp.load_state_quantiles(_root)
fcp.apply(_variant, _q01, _q99, cond=_cond, mask_force=_mask_force,
          contact_F0=float(os.environ.get("FILM_F0", "6")),
          contact_tau=float(os.environ.get("FILM_TAU", "4")),
          fz_tau=float(os.environ.get("FILM_FZ_TAU", "5")),
          fz_off=float(os.environ.get("FILM_FZ_OFF", "2.6")),
          fmag_off=float(os.environ.get("FILM_FMAG_OFF", "5.1")),
          fmag_tau=float(os.environ.get("FILM_FMAG_TAU", "5")),
          dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")))
print(f"[film-pi05] patched PI05: variant={_variant} cond={_cond} inject=suffix "
      f"mask_force={_mask_force} (state masked in tokenizer step; c-hat from batch state)",
      file=sys.stderr)

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()
